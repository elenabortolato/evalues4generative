"""Collect replies of one language model at three quantisation levels (exp16).

The advertised model P is Qwen2.5-1.5B-Instruct quantised at 8 bits (q8_0);
the substitutes are the same model at 4 bits (q4_k_m) and 3 bits (q3_k_m),
and a different, smaller model, Qwen2.5-0.5B-Instruct at 4 bits (05b).
Each reply is a separate call with no history, at temperature 1 and top-p 1,
so the replies of one model and prompt are i.i.d. draws from its output law.
The GGUF files are read from --models (download from the Hugging Face
repositories Qwen/Qwen2.5-1.5B-Instruct-GGUF and
Qwen/Qwen2.5-0.5B-Instruct-GGUF). Needs llama-cpp-python.

Output: data/llm_audit/replies.json, {"<quant>|<prompt>": [reply, ...]},
written after each (model, prompt) pair.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROMPTS = {
    "digits": "Generate exactly 8 independent uniformly random digits from 0 to 9. "
              "Output only the compact digit string, no separators.",
    "number": "Choose one uniformly random integer between 1 and 100 inclusive. "
              "Output only the integer.",
    "color": "Name one random color. Output only the color name.",
    "coin": "Flip a fair coin 6 times. Output only the results as a string of H and T, "
            "no separators.",
}
SIZES = {"q8_0": 3000, "q4_k_m": 1500, "q3_k_m": 1500, "05b": 1500}
FILES = {q: f"qwen2.5-1.5b-instruct-{q}.gguf" for q in ["q8_0", "q4_k_m", "q3_k_m"]}
FILES["05b"] = "qwen2.5-0.5b-instruct-q4_k_m.gguf"   # a different, smaller model
OUT = Path(__file__).resolve().parents[1] / "data" / "llm_audit" / "replies.json"


def main() -> None:
    from llama_cpp import Llama
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="directory with the GGUF files")
    args = ap.parse_args()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for qi, (q, n) in enumerate(SIZES.items()):
        llm = Llama(model_path=str(Path(args.models) / FILES[q]),
                    n_ctx=256, n_threads=4, verbose=False)
        for pi, (k, prompt) in enumerate(PROMPTS.items()):
            key = f"{q}|{k}"
            if len(out.get(key, [])) >= n:
                continue
            t0, reps = time.time(), []
            for i in range(n):
                r = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}], temperature=1.0, top_p=1.0,
                    max_tokens=24, seed=10**6 * (qi + 1) + 10**5 * pi + i)
                reps.append(r["choices"][0]["message"]["content"].strip())
            out[key] = reps
            OUT.write_text(json.dumps(out))
            print(f"{key}: {n} replies, {(time.time() - t0) / n:.2f} s/reply", flush=True)


if __name__ == "__main__":
    main()
