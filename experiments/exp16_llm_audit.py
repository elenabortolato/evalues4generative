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
  all-05b    a fraction 0.3 of every prompt is answered by a different,
             smaller model (Qwen2.5-0.5B), unexpected by the baselines,
             which were trained on the 4-bit model
  some-05b   as some-q4, with the 0.5B model as the substitute
  lowtemp    every query is answered by P at temperature 0.7 instead of one
             (real replies, exp16_llm_generate.py): a provider that changes the
             sampling settings; closer to P than to the 4-bit model
  sharp      every query is answered by a sharpened P: P's replies drawn with
             probability proportional to the square of their frequency, as
             a lower temperature would do (the most frequent colour goes
             from 23% to 44%); outside SLICE's family, and closer to P than
             to the 4-bit model, so a score fitted to the 4-bit model bets
             in the wrong direction
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
  slice-bandit  the score of slice inside our design: prompts chosen by UCB,
                the twin run on the chosen prompt
  fb-bandit     the pretrained score of fb-pre inside our design: prompts
                chosen by UCB, the twin run on the chosen prompt
  ours-simlik   our score on the simulation likelihood (Theorem 3): Gaussian
                one-step update of the posterior on the weights (ChoiceBLR,
                prior variance 1, 32 antithetic nodes, standardised features
                clipped to [-10, 10] so that they are bounded as Theorem 3
                assumes), and the bet is the
                posterior-predictive bet, the average of E(theta) over the
                nodes (Proposition 4); uniform prompts
  ours-simlik-bandit  the same inside our design (UCB prompts)
  fb-refit      Forre and Brenner's online score: theta' z(x) on the same
                features as ours, refitted every 10 steps on the whole past
                by maximising the penalised empirical growth (ridge tau,
                e4g.competitors.GrowthFit); needs no reply of the substitute;
                uniform prompts
  fb-refit-bandit  the same score inside our design (UCB prompts)
  combo-bandit  the average of two wealths on the same data, one with the
                learned score of ours-bandit and one with the pretrained score
                of fb-pre, prompts chosen by UCB on their average factor
The baselines and the methods using the pretrained score are given replies
of the substitute in advance, which the learned score does not need.

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
from e4g.competitors import GrowthFit  # noqa: E402
from e4g.learners import BLRRatio, ChoiceBLR  # noqa: E402

PROMPTS = ["digits", "number", "color", "coin"]
K, D = len(PROMPTS), 32          # prompts; hashed n-gram features per prompt
SCENARIOS = {
    "null": dict(sub=None, frac={}),
    "all-q4": dict(sub="q4_k_m", frac={k: 0.3 for k in PROMPTS}),
    "some-q4": dict(sub="q4_k_m", frac={"color": 0.5, "number": 0.5}),
    "all-q3": dict(sub="q3_k_m", frac={k: 0.3 for k in PROMPTS}),
    "all-05b": dict(sub="05b", frac={k: 0.3 for k in PROMPTS}),
    "some-05b": dict(sub="05b", frac={"color": 0.5, "number": 0.5}),
    "sharp": dict(sub="sharp", frac={k: 1.0 for k in PROMPTS}),
    "lowtemp": dict(sub="q8_0_t07", frac={k: 1.0 for k in PROMPTS}),
}
EXTRA_POOLS = ["q8_0_t07"]       # pools added later, split with their own generator
SHARPEN = 2.0                    # "sharp": P's replies drawn w.p. ~ frequency ** SHARPEN
FB_TAU = 1.0                     # ridge of fb-refit, the prior variance of our learned score
CLIP = 10.0                      # ours-simlik: standardised features clipped to [-CLIP, CLIP]
BASELINE_SUB = "q4_k_m"          # the substitute the baselines are trained on
UCB_C, MIN_SHARE = 1.0, 0.1   # prompt choice: UCB on log growth, each prompt >= 10%
METHODS = ["ours-bandit", "ours-uniform", "ours-mixed", "slice", "fb-pre", "fb-bandit",
           "combo-bandit", "slice-bandit", "fb-refit", "fb-refit-bandit",
           "ours-simlik", "ours-simlik-bandit"]


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
        self.feat, self.idx, self.sharp_w = [], {}, {}
        for k, prompt in enumerate(PROMPTS):
            texts = []
            for q in ["q8_0", "q4_k_m", "q3_k_m", "05b"]:
                if f"{q}|{prompt}" not in raw:
                    continue
                reps = list(raw[f"{q}|{prompt}"])
                rng.shuffle(reps)
                n_train = len(reps) // 3
                for part, chunk in [("train", reps[:n_train]), ("stream", reps[n_train:])]:
                    self.idx[(k, q, part)] = np.arange(len(texts), len(texts) + len(chunk))
                    texts.extend(chunk)
            for q in EXTRA_POOLS:   # own generator: the pools above stay as they were
                if f"{q}|{prompt}" not in raw:
                    continue
                reps = list(raw[f"{q}|{prompt}"])
                np.random.default_rng(1 + k).shuffle(reps)
                n_train = len(reps) // 3
                for part, chunk in [("train", reps[:n_train]), ("stream", reps[n_train:])]:
                    self.idx[(k, q, part)] = np.arange(len(texts), len(texts) + len(chunk))
                    texts.extend(chunk)
            self.feat.append(np.array([text_features(s) for s in texts]))
            for part in ("train", "stream"):    # sharpened P: weights ~ count ** (SHARPEN - 1)
                ids = self.idx[(k, "q8_0", part)]
                counts = {}
                for i in ids:
                    counts[texts[i]] = counts.get(texts[i], 0) + 1
                w = np.array([counts[texts[i]] ** (SHARPEN - 1) for i in ids], float)
                self.sharp_w[(k, part)] = w / w.sum()

    def rows(self, k, q, part, rng, n):
        if q == "sharp":
            ids = rng.choice(self.idx[(k, "q8_0", part)], n, p=self.sharp_w[(k, part)])
        else:
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


class ClippedChoiceBLR(ChoiceBLR):
    """ChoiceBLR with the standardised features clipped to [-CLIP, CLIP].

    Theorem 3 and Proposition 4 assume bounded features. A feature that is
    constant on the twin's replies (no whitespace in P's replies to a prompt,
    say) is standardised to about 1e8 when a substitute's reply has it, which
    breaks that assumption and makes the posterior precision numerically
    singular; clipping restores it.
    """

    def _z(self, x):
        return np.clip(super()._z(x), -CLIP, CLIP)


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
    lam_mix, lam_mix2 = LambdaMixture(), LambdaMixture()
    learns = method.startswith(("ours", "fb-refit")) or method == "combo-bandit"
    if learns:
        twin = np.vstack([POOLS.rows(k, "q8_0", "stream", rng, 200) for k in range(K)])
        if method.startswith("fb-refit"):
            learner = GrowthFit(tau=FB_TAU, mode="online", feature_map=block_features).fit_scaler(twin)
        elif method.startswith("ours-simlik"):
            learner = ClippedChoiceBLR(prior_var=1.0, mode="mixture",
                                       feature_map=block_features).fit_scaler(twin)
        else:
            learner = BLRRatio(prior_var=1.0, mode="predictive",
                               feature_map=block_features).fit_scaler(twin)
    n_pull, g_sum = np.zeros(K), np.zeros(K)
    past_lr, first, t_method = [], None, 0.0   # t_method: seconds spent on bets and updates
    for t in range(1, horizon + 1):
        # 1. action: chosen from the past only
        if method in ("ours-bandit", "ours-mixed", "fb-bandit", "combo-bandit", "slice-bandit",
                      "fb-refit-bandit", "ours-simlik-bandit"):
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
        tic = time.perf_counter()
        rows = np.vstack([x, ys])
        if method.startswith("ours-simlik"):
            e = learner.factor(x, ys)              # posterior-predictive bet
            w = lam_mix.update(e)
        elif method == "combo-bandit":
            # two e-processes on the same data, the learned and the pretrained
            # score; their average wealth is again a test supermartingale
            s, s2 = learner.score(rows), log_rho(rho_models, rows)
            e1, e2 = mc_factor(s[0], s[1:]), mc_factor(s2[0], s2[1:])
            w = 0.5 * (lam_mix.update(e1) + lam_mix2.update(e2))
            e = 0.5 * (e1 + e2)
        else:
            if method.startswith(("ours", "fb-refit")):
                s = learner.score(rows)
            elif method in ("slice", "slice-bandit"):
                th = mle_fraction(np.asarray(past_lr))
                s = np.log(np.maximum(1 - th + th * np.exp(log_rho(rho_models, rows)), 1e-300))
            else:  # fb-pre, fb-bandit
                s = log_rho(rho_models, rows)
            e = mc_factor(s[0], s[1:])
            w = lam_mix.update(e)
        t_method += time.perf_counter() - tic
        if first is None and w >= 1 / alpha:
            first = t
            break
        # 6. learn after the bet
        n_pull[a] += 1
        g_sum[a] += np.log(0.5 + 0.5 * e)
        tic = time.perf_counter()
        if learns:
            learner.update(x, ys)
        elif method in ("slice", "slice-bandit"):
            past_lr.append(log_rho(rho_models, x)[0])
        t_method += time.perf_counter() - tic
    return first, n_pull.tolist(), t_method, t


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
