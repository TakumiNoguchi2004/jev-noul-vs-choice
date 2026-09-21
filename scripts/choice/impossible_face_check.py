"""
Fills in the one cell the headline table couldn't show: `noul/calibration_check.py`
asks about an impossible 7th face of a six-sided die (should read ~0 if Jev
is calibrated) alongside faces 1-6, but `choice/baseline_check.py`'s criteria
only ever listed 1-6 -- there was no `choice`-side number for face 7 to
compare against.

Same state/instructions as baseline_check.py, but `criteria` adds a 7th
option ("The die shows 7") to the usual 1-6, so `choice` has to weigh a
genuinely impossible outcome against six real ones in the same distribution
-- the direct `choice` analogue of noul's face_7 question.

Run with: python scripts/choice/impossible_face_check.py --n-trials 50 --run-name <name>
"""
import argparse
import json
import os
import statistics
from os import path as ospath

import requests
from tqdm import tqdm

REPO_ROOT = ospath.dirname(ospath.dirname(ospath.dirname(ospath.abspath(__file__))))
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"

STATE = (
    "\nMy neighbour rolled a six-sided die exactly once in a probability demonstration.\n"
    "The die is unbiased: each of the six faces has probability exactly 1/6.\n"
    "Nobody has observed the outcome yet.\n\n"
    "Which face came up?\n"
)
INSTRUCTIONS = (
    "\nInfer which outcome actually occurred.\n\n"
    "The situation may be genuinely uncertain. Do not assume that the most likely\n"
    "outcome is certain. Preserve uncertainty in the probability distribution.\n"
)
CRITERIA = {str(i): f"The die shows {i}" for i in range(1, 8)}  # 1-6 real, 7 impossible
TRUE_PROB = {str(i): (1 / 6 if i <= 6 else 0.0) for i in range(1, 8)}


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README.md).")
    return key


def call_jev_dice(api_key: str) -> dict:
    payload = {
        "model": MODEL,
        "state": STATE,
        "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}},
    }
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload, timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    api_key = load_api_key()
    out_dir = ospath.join(REPO_ROOT, "results", "choice", "impossible_face", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = ospath.join(out_dir, "trials.jsonl")

    trials = []
    if ospath.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            trials = [json.loads(line) for line in f]
    n_done = len(trials)
    if n_done >= args.n_trials:
        print(f"[skip] already have {n_done}/{args.n_trials} trials")
    else:
        if n_done:
            print(f"[resume] {n_done}/{args.n_trials} already in {jsonl_path}, continuing")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for _ in tqdm(range(n_done, args.n_trials), desc="dice trials (7-way)"):
                raw = call_jev_dice(api_key)
                answer = raw["answers"]["answer"]
                rec = {
                    "choice": answer["choice"],
                    "probabilities": answer["probabilities"],
                    "confidence": answer.get("confidence"),
                    "cost": raw.get("usage", {}).get("cost"),
                    "model": raw.get("model"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                trials.append(rec)

    mean_prob_per_face = {k: statistics.mean(t["probabilities"][k] for t in trials) for k in CRITERIA}
    choice_counts = {k: 0 for k in CRITERIA}
    for t in trials:
        choice_counts[t["choice"]] += 1
    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)

    summary = {
        "n_trials": len(trials),
        "true_prob_per_face": TRUE_PROB,
        "mean_prob_per_face": mean_prob_per_face,
        "choice_distribution": choice_counts,
        "total_cost_usd": total_cost,
    }
    with open(ospath.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for k in CRITERIA:
        print(f"face {k}: mean_prob={mean_prob_per_face[k]:.3f} true={TRUE_PROB[k]:.3f}")
    print(f"choice distribution: {choice_counts}")
    print(f"cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
