"""
Follow-up to noul/calibration_check.py / choice/bias_check.py. Those found:
- `choice` with 6 options is severely miscalibrated (locks onto "The die
  shows 1" regardless of key label, order, or value verbosity -- bias_check.py's experiments).
- `noul` (independent per-proposition probability) stays well-calibrated for
  the identical scenario, even when explicitly shown the full option list
  (B1/B2) -- so it's not about what info is visible, something about the
  `choice` mechanism itself is the problem.

This asks: does `choice` break even in the SIMPLEST possible case -- a binary
(2-option) choice, structurally as close to a single `noul` proposition as
`choice` can get? For each face n in 1-6, one `choice` question with criteria
{"yes": "The die shows n", "no": "The die does not show n"} (true P(yes) =
1/6, P(no) = 5/6). All 6 asked as parallel questions in one request per
trial, matching the first noul_check.py's batching style (not isolated into
6 separate requests -- that stricter check was explicitly skipped per user
instruction).

Caveat: "yes"/"no" as key labels introduces a DIFFERENT, well-known LLM bias
risk (acquiescence bias -- tendency to favor an affirmative "yes" token)
distinct from the "1"-content anchoring found in bias_check.py's experiments. If P(yes) comes back
inflated across the board (not just for face-content reasons), that's a
plausible confound to flag, not evidence against the bias_check.py's experiments findings.

Run with: python scripts/choice/binary_check.py --n-trials 50 --run-name <name>
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
    "Nobody has observed the outcome yet.\n"
)

INSTRUCTIONS = (
    "\nInfer which outcome actually occurred.\n\n"
    "The situation may be genuinely uncertain. Do not assume that the most likely\n"
    "outcome is certain. Preserve uncertainty in the probability distribution.\n"
)

FACES = list(range(1, 7))
TRUE_PROB_YES = 1 / 6


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README.md).")
    return key


def call_jev_binary_choice_batch(api_key: str) -> dict:
    questions = {
        f"face_{n}_choice": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {"yes": f"The die shows {n}", "no": f"The die does not show {n}"},
        }
        for n in FACES
    }
    payload = {"model": MODEL, "state": STATE, "questions": questions}
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
    out_dir = ospath.join(REPO_ROOT, "results", "choice", "binary", args.run_name)
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
            for _ in tqdm(range(n_done, args.n_trials), desc="choice binary dice batch"):
                raw = call_jev_binary_choice_batch(api_key)
                p_yes = {
                    str(n): raw["answers"][f"face_{n}_choice"]["probabilities"]["yes"]
                    for n in FACES
                }
                chosen = {
                    str(n): raw["answers"][f"face_{n}_choice"]["choice"]
                    for n in FACES
                }
                rec = {
                    "p_yes": p_yes,
                    "chosen": chosen,
                    "cost": raw.get("usage", {}).get("cost"),
                    "model": raw.get("model"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                trials.append(rec)

    per_face_pyes = {n: [t["p_yes"][str(n)] for t in trials] for n in FACES}
    per_face_chosen_yes_frac = {
        n: sum(1 for t in trials if t["chosen"][str(n)] == "yes") / len(trials) for n in FACES
    }
    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)

    summary = {
        "n_trials": len(trials),
        "mean_p_yes_per_face": {n: statistics.mean(per_face_pyes[n]) for n in FACES},
        "frac_chosen_yes_per_face": per_face_chosen_yes_frac,
        "true_p_yes": TRUE_PROB_YES,
        "total_cost_usd": total_cost,
    }
    with open(ospath.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for n in FACES:
        print(f"face_{n}: mean_p_yes={summary['mean_p_yes_per_face'][n]:.3f} "
              f"frac_chosen_yes={per_face_chosen_yes_frac[n]:.2f} true_p_yes={TRUE_PROB_YES:.3f}")
    print(f"cost: ${total_cost:.5f}")


if __name__ == "__main__":
    main()
