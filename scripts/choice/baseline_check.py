"""
Replication check of github.com/KantaHayashiAI/jev-does-not-play-dice's "fair
six-sided die" Choice-question finding, against OUR OpenRouter access path to
Jev (that repo used the Vercel AI Gateway instead -- worth checking whether
the miscalibration shows up through a different provider too).

State/instructions/criteria below are copied VERBATIM from that repo's
examples/dice.json (trial d0020), fetched via `gh api
repos/KantaHayashiAI/jev-does-not-play-dice/contents/examples/dice.json`.
Their reference: true outcome probability is 16.7% (1/6) per face, since the
state explicitly says "unbiased... each of the six faces has probability
exactly 1/6." Their finding (400 trials, Vercel AI Gateway): mean reported
probability of the chosen face = 82.9%, observed accuracy 19.0% (~chance, as
expected -- Jev has no real information about the outcome).

This script repeats the SAME identical request N times (no per-trial wording
variation) and checks whether the reported probability of Jev's own chosen
face stays far above the calibrated 16.7%, which would replicate their
"Jev does not play dice" miscalibration finding through OpenRouter too.
There's no ground-truth die roll to score "accuracy" against here (this repo
doesn't roll a physical die) -- the point under test is calibration of the
reported probability, not prediction accuracy.

Run with: python scripts/choice/baseline_check.py --n-trials 50 --run-name <name>
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
CRITERIA = {str(i): f"The die shows {i}" for i in range(1, 7)}
TRUE_PROB = 1 / 6


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
    out_dir = ospath.join(REPO_ROOT, "results", "choice", "baseline", args.run_name)
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
            for _ in tqdm(range(n_done, args.n_trials), desc="dice trials"):
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

    top_probs = [t["probabilities"][t["choice"]] for t in trials]
    choice_counts = {str(i): 0 for i in range(1, 7)}
    for t in trials:
        choice_counts[t["choice"]] += 1
    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)

    summary = {
        "n_trials": len(trials),
        "true_prob_per_face": TRUE_PROB,
        "mean_reported_prob_of_chosen_face": statistics.mean(top_probs),
        "stdev_reported_prob_of_chosen_face": statistics.pstdev(top_probs) if len(top_probs) > 1 else 0.0,
        "choice_distribution": choice_counts,
        "total_cost_usd": total_cost,
        "reference_repo_400trial_mean": 0.829,  # KantaHayashiAI/jev-does-not-play-dice, Vercel AI Gateway
    }
    with open(ospath.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(f"n_trials={summary['n_trials']}  true_prob=1/6={TRUE_PROB:.3f}")
    print(f"mean reported prob of chosen face = {summary['mean_reported_prob_of_chosen_face']:.3f} "
          f"(repo's 400-trial mean via Vercel AI Gateway: 0.829)")
    print(f"choice distribution: {choice_counts}")
    print(f"cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
