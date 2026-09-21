"""
Follow-up to choice/bias_check.py. Those experiments found Jev's `choice`
question about a fair die is severely miscalibrated and biased toward
"The die shows 1" content specifically, regardless of key label/position.

This script asks the SAME calibration question a structurally different way:
instead of one `choice` question with 6 options, it asks SEVEN independent
`noul` (probability-of-a-proposition) questions in a single request -- "is it
true the die shows 1?" / "...shows 2?" / ... / "...shows 6?", plus a 7th
"...shows 7?" which is impossible for a six-sided die (should read ~0 if
Jev is calibrated).

A single manual trial already showed a striking contrast with the `choice`
pathology: faces 1-6 all came back ~0.15-0.17 (true=0.167), face 7 came back
0.01. This script repeats that same 7-question request n times to check
whether that was a fluke or a real difference between `noul` and `choice`
calibration for the identical scenario.

Run with: python scripts/noul/calibration_check.py --n-trials 50 --run-name <name>
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

BASE_STATE = (
    "\nMy neighbour rolled a six-sided die exactly once in a probability demonstration.\n"
    "The die is unbiased: each of the six faces has probability exactly 1/6.\n"
    "Nobody has observed the outcome yet.\n"
)

# Same 1-6 listing as the `choice` experiments' criteria, used to test whether
# merely SEEING the sibling-option list (without the `choice` mechanism's
# forced single pick) is what reintroduces the "always face 1" bias. Only
# 1-6 are listed (not 7) so face_7 stays a genuinely out-of-list/impossible
# proposition in both list_in_state and list_in_instructions, matching the
# isolated-noul baseline.
OUTCOME_LIST = (
    "\nThe possible outcomes are:\n"
    + "\n".join(f"{i}: The die shows {i}" for i in range(1, 7))
    + "\n"
)


def _noul_instructions(n: int, condition: str) -> str:
    proposition = (
        f"Determine whether the following proposition is true of the situation described: "
        f"The die shows {n}.\n\n"
        "The situation may be genuinely uncertain. Do not assume that the most likely "
        "outcome is certain. Preserve uncertainty in the probability."
    )
    if condition == "list_in_instructions":
        return OUTCOME_LIST + "\n" + proposition
    return proposition


def build_state(condition: str) -> str:
    if condition == "list_in_state":
        return BASE_STATE + OUTCOME_LIST
    return BASE_STATE


FACES_TO_ASK = list(range(1, 8))  # 1-6 real faces + 7 (impossible, should read ~0)
TRUE_PROB = {n: (1 / 6 if n <= 6 else 0.0) for n in FACES_TO_ASK}


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README.md).")
    return key


def call_jev_noul_batch(api_key: str, condition: str) -> dict:
    questions = {f"face_{n}": {"type": "noul", "instructions": _noul_instructions(n, condition)} for n in FACES_TO_ASK}
    payload = {"model": MODEL, "state": build_state(condition), "questions": questions}
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
    p.add_argument("--condition", default="isolated", choices=["isolated", "list_in_state", "list_in_instructions"],
                    help="isolated=baseline (no sibling-option list shown); "
                         "list_in_state=list added to shared `state`; "
                         "list_in_instructions=list added to each question's own `instructions`")
    args = p.parse_args()

    api_key = load_api_key()
    out_dir = ospath.join(REPO_ROOT, "results", "noul", "calibration", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = ospath.join(out_dir, "trials.jsonl")

    trials = []
    if ospath.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            trials = [json.loads(line) for line in f]
    n_done = len(trials)
    if n_done < args.n_trials:
        if n_done:
            print(f"[resume] {n_done}/{args.n_trials} already in {jsonl_path}, continuing")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for _ in tqdm(range(n_done, args.n_trials), desc=f"noul dice batch ({args.condition})"):
                raw = call_jev_noul_batch(api_key, args.condition)
                probs = {str(n): raw["answers"][f"face_{n}"]["noul"] for n in FACES_TO_ASK}
                rec = {
                    "probs": probs,
                    "sum_probs_1to6": sum(probs[str(n)] for n in range(1, 7)),
                    "cost": raw.get("usage", {}).get("cost"),
                    "model": raw.get("model"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                trials.append(rec)

    per_face = {n: [t["probs"][str(n)] for t in trials] for n in FACES_TO_ASK}
    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)
    sums = [t["sum_probs_1to6"] for t in trials]

    summary = {
        "n_trials": len(trials),
        "mean_prob_per_face": {n: statistics.mean(per_face[n]) for n in FACES_TO_ASK},
        "stdev_prob_per_face": {n: (statistics.pstdev(per_face[n]) if len(per_face[n]) > 1 else 0.0) for n in FACES_TO_ASK},
        "true_prob_per_face": TRUE_PROB,
        "mean_sum_probs_1to6": statistics.mean(sums),
        "total_cost_usd": total_cost,
    }
    with open(ospath.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for n in FACES_TO_ASK:
        print(f"face_{n}: mean={summary['mean_prob_per_face'][n]:.3f} "
              f"stdev={summary['stdev_prob_per_face'][n]:.3f} true={TRUE_PROB[n]:.3f}")
    print(f"mean sum(face_1..6) per trial = {summary['mean_sum_probs_1to6']:.3f} (ideal coherent sum = 1.0)")
    print(f"cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
