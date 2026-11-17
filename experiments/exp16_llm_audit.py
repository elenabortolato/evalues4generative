"""Experiment 16: auditing a language-model API with adaptively chosen prompts.

An API advertises a model P, here Qwen2.5-1.5B-Instruct at 8 bits, and may
answer some queries with a cheaper substitute R, the same model quantised at
4 or 3 bits. The auditor runs P locally as the digital twin. At each step it
chooses a prompt A_t from four "random output" probes, sends it to the API
(the real reply X_t), and draws M replies of P to the same prompt (the
simulations Y_t). H0: the API serves P for every prompt.

The design is semi-synthetic, as in Cortinovis, Battaglia and Caron (2026,
Section 6): the replies are real outputs of the models (exp16_llm_generate.py),
and only the routing is simulated. Under H0 the real reply and the M
simulations are drawn with replacement from the same pool of P's replies to
the chosen prompt, so they are exactly exchangeable given the past.

Scenarios
  null       the API serves P
  all-q4     a fraction 0.3 of every prompt is answered by the 4-bit model
  some-q4    a fraction 0.5 of the "color" and "number" prompts is answered
             by the 4-bit model, the other prompts by P (prompt-dependent
             routing: outside a mixture family with one routing fraction)
  all-q3     a fraction 0.3 of every prompt is answered by the 3-bit model,
             while the baselines that need R were trained on the 4-bit model
Methods
  ours-bandit   Monte Carlo e-process, logistic BLR score on hashed character
                n-grams with one block per prompt; the prompt is chosen by UCB
                on the realised log growth of each prompt, with each prompt
                kept above a tenth of the queries (adaptive actions)
  ours-uniform  the same score, prompts drawn uniformly at random
  ours-mixed    prompts chosen by UCB, but the twin replies to a random prompt
                instead of the chosen one (ignores the action: Proposition 2)
  slice         score log(1 - th + th rho_k(y)) of the mixture family
                f_th = (1 - th) p + th r with one routing fraction th for all
                prompts, th estimated from the past real replies, rho_k = r/p
                estimated offline by a classifier of R against P replies (as
                SLICE does in its LLM audit); uniform prompts
  fb-pre        score log rho_k fitted in advance on R against P (Forre and
                Brenner's pretrained score); uniform prompts
Both baselines are given replies of the substitute in advance, which ours
does not need.

Output: results/exp16.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e4g.evalues import LambdaMixture, mc_factor  # noqa: E402
from e4g.learners import BLRRatio  # noqa: E402

PROMPTS = ["digits", "number", "color", "coin"]
K, D = len(PROMPTS), 32          # prompts; hashed n-gram features per prompt
SCENARIOS = {
    "null": dict(sub=None, frac={}),
    "all-q4": dict(sub="q4_k_m", frac={k: 0.3 for k in PROMPTS}),
    "some-q4": dict(sub="q4_k_m", frac={"color": 0.5, "number": 0.5}),
    "all-q3": dict(sub="q3_k_m", frac={k: 0.3 for k in PROMPTS}),
}
BASELINE_SUB = "q4_k_m"          # the substitute the baselines are trained on
UCB_C, MIN_SHARE = 1.0, 0.1   # prompt choice: UCB on log growth, each prompt >= 10%
METHODS = ["ours-bandit", "ours-uniform", "ours-mixed", "slice", "fb-pre"]


# ------------------------------------------------------------------ features
def text_features(s: str) -> np.ndarray:
    """Character unigrams and bigrams hashed to D bins, plus length terms."""
    f = np.zeros(D)
    grams = list(s) + [s[i:i + 2] for i in range(len(s) - 1)]
    for g in grams:
        f[zlib.crc32(g.encode()) % (D - 2)] += 1.0
    if grams:
        f[: D - 2] /= np.sqrt(len(grams))
    f[D - 2], f[D - 1] = len(s) / 10.0, float(any(c.isspace() for c in s))
    return f


class Pools:
    """Replies of each model to each prompt, split into training and stream parts.

    A reply is referred to by a row (prompt k, global index i); FEAT[k][i] are
    its features. The training parts (a third of each pool) are used only by
    the baselines, to estimate rho_k = r_k / p_k in advance.
    """

    def __init__(self, path: Path):
        raw = json.loads(path.read_text())
        rng = np.random.default_rng(0)
        self.feat, self.idx = [], {}
        for k, prompt in enumerate(PROMPTS):
            texts = []
            for q in ["q8_0", "q4_k_m", "q3_k_m"]:
                reps = list(raw[f"{q}|{prompt}"])
                rng.shuffle(reps)
                n_train = len(reps) // 3
                for part, chunk in [("train", reps[:n_train]), ("stream", reps[n_train:])]:
                    self.idx[(k, q, part)] = np.arange(len(texts), len(texts) + len(chunk))
                    texts.extend(chunk)
            self.feat.append(np.array([text_features(s) for s in texts]))

    def rows(self, k, q, part, rng, n):
        ids = rng.choice(self.idx[(k, q, part)], n)
        return np.column_stack([np.full(n, k), ids]).astype(float)


POOLS: Pools | None = None


def block_features(rows: np.ndarray) -> np.ndarray:
    """One block of D features per prompt: the score can differ by prompt."""
    rows = np.atleast_2d(rows)
    out = np.zeros((len(rows), K * D))
    for r, (k, i) in enumerate(rows.astype(int)):
        out[r, k * D:(k + 1) * D] = POOLS.feat[k][i]
    return out


# ----------------------------------------------------------------- baselines
def fit_log_rho(sub: str) -> list:
    """Per prompt, a balanced logistic classifier of R (1) against P (0)."""
    from sklearn.linear_model import LogisticRegression
    models = []
    for k in range(K):
        a = POOLS.feat[k][POOLS.idx[(k, sub, "train")]]
        b = POOLS.feat[k][POOLS.idx[(k, "q8_0", "train")]]
        x = np.vstack([a, b])
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        models.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(x, y))
    return models


def log_rho(models, rows):
    rows = np.atleast_2d(rows).astype(int)
    out = np.empty(len(rows))
    for r, (k, i) in enumerate(rows):
        out[r] = models[k].decision_function(POOLS.feat[k][i][None, :])[0]
    return np.clip(out, -10, 10)


def mle_fraction(lr: np.ndarray) -> float:
    """argmax over th in [0, 1] of sum log(1 - th + th rho_i)."""
    if len(lr) == 0:
        return 0.0
    rho = np.exp(lr)
    grid = np.linspace(0, 1, 201)
    ll = np.log(np.maximum(1 - grid[:, None] + grid[:, None] * rho[None, :], 1e-300)).sum(1)
    return float(grid[np.argmax(ll)])


# ----------------------------------------------------------------------- run
def run(seed, scenario, method, horizon, m, alpha, rho_models):
    rng = np.random.default_rng(seed)
    sc = SCENARIOS[scenario]
    lam_mix = LambdaMixture()
    if method.startswith("ours"):
        twin = np.vstack([POOLS.rows(k, "q8_0", "stream", rng, 200) for k in range(K)])
        learner = BLRRatio(prior_var=1.0, mode="predictive", feature_map=block_features).fit_scaler(twin)
    n_pull, g_sum = np.zeros(K), np.zeros(K)
    past_lr, first = [], None
    for t in range(1, horizon + 1):
        # 1. action: chosen from the past only
        if method in ("ours-bandit", "ours-mixed"):
            if t <= 2 * K:
                a = (t - 1) % K
            elif n_pull.min() < MIN_SHARE * t:
                a = int(np.argmin(n_pull))      # forced exploration
            else:
                ucb = g_sum / n_pull + UCB_C * np.sqrt(np.log(t) / n_pull)
                a = int(np.argmax(ucb))
        else:
            a = int(rng.integers(K))
        # 2. real reply from the API, simulations from the twin
        routed = sc["sub"] is not None and rng.random() < sc["frac"].get(PROMPTS[a], 0.0)
        x = POOLS.rows(a, sc["sub"] if routed else "q8_0", "stream", rng, 1)
        if method == "ours-mixed":
            ys = np.vstack([POOLS.rows(int(rng.integers(K)), "q8_0", "stream", rng, 1) for _ in range(m)])
        else:
            ys = POOLS.rows(a, "q8_0", "stream", rng, m)
        # 3-4. score fixed before x is seen, and the bet
        rows = np.vstack([x, ys])
        if method.startswith("ours"):
            s = learner.score(rows)
        elif method == "slice":
            th = mle_fraction(np.asarray(past_lr))
            s = np.log(np.maximum(1 - th + th * np.exp(log_rho(rho_models, rows)), 1e-300))
        else:  # fb-pre
            s = log_rho(rho_models, rows)
        e = mc_factor(s[0], s[1:])
        w = lam_mix.update(e)
        if first is None and w >= 1 / alpha:
            first = t
            break
        # 6. learn after the bet
        n_pull[a] += 1
        g_sum[a] += np.log(0.5 + 0.5 * e)
        if method.startswith("ours"):
            learner.update(x, ys)
        elif method == "slice":
            past_lr.append(log_rho(rho_models, x)[0])
    return first, n_pull.tolist()


def init_worker(path):
    global POOLS
    POOLS = Pools(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", default=str(ROOT / "data" / "llm_audit" / "replies.json"))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "exp16.json"))
    args = ap.parse_args()
    init_worker(Path(args.replies))
    rho_models = fit_log_rho(BASELINE_SUB)
    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    res["config"] = vars(args)
    for scenario in args.scenarios.split(","):
        for method in args.methods.split(","):
            t0 = time.time()
            f = partial(run, scenario=scenario, method=method, horizon=args.horizon, m=args.m,
                        alpha=args.alpha, rho_models=rho_models)
            seeds = [10**4 * list(SCENARIOS).index(scenario) + i for i in range(args.reps)]
            with ProcessPoolExecutor(args.workers, initializer=init_worker,
                                     initargs=(Path(args.replies),)) as ex:
                out = list(ex.map(f, seeds))
            stops = [o[0] for o in out]
            hit = [s for s in stops if s is not None]
            pulls = np.mean([o[1] for o in out], 0)
            r = dict(reject=len(hit) / len(stops), median_stop=float(np.median(hit)) if hit else None,
                     pulls=(pulls / pulls.sum()).round(3).tolist(), seconds=round(time.time() - t0, 1))
            res.setdefault(scenario, {})[method] = r
            out_path.write_text(json.dumps(res, indent=1))
            print(f"{scenario:8s} {method:13s} reject {r['reject']:.3f}  median {r['median_stop']}  "
                  f"pulls {r['pulls']}  {r['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
