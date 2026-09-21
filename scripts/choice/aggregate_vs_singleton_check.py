"""
Follow-up to choice/yes_label_rotation_check.py. That found the "yes" label has no
independent pull -- content specifically describing "The die shows 1" (or,
with face 1 removed entirely, "3") wins regardless of which key it's
attached to (bias_check.py's experiments, yesno6 both-full and faces=2,3,4,5).

This tries a 3-way choice with mixed granularity: two SINGLETON options plus
one AGGREGATE/catch-all option covering the remaining 4 faces, true P =
1/6, 1/6, 4/6.

v1/v2 (fixed key names "yes"/"no_2"/"no", faces randomized underneath) found
the aggregate always wins (50/50 in both runs), massively overconfident
(~0.90 vs true 0.667), regardless of which 2 faces were drawn as singletons
or whether face 1 was one of them.

v3 (this version): the KEY NAMES "yes"/"no_2" were themselves a confound --
"no_2" always contains the literal digit "2" as text regardless of which
face it was mapped to, which could itself be a label magnet (bias_check.py's key/position variants found
canonical-looking labels attract independent of content). Keys are now
meaningless random strings (matching bias_check.py's random-key variants's methodology) for the two
singleton options and for the aggregate bucket, freshly generated each
trial. Results are aggregated by FACE NUMBER (not key name) across all
trials, building an empirical per-face win-rate and probability profile from
many random (a, b) singleton draws -- answering "does face N specifically
have elevated odds of winning as a singleton, once key-label confounds are
fully removed?"

Run with: python scripts/choice/aggregate_vs_singleton_check.py --n-trials 50 --run-name <name>
"""
import argparse
import json
import os
import random
import statistics
import string
from collections import Counter
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
ALL_FACES = list(range(1, 7))
TRUE_PROB_SINGLETON = 1 / 6
TRUE_PROB_AGGREGATE = 4 / 6


def _random_key(rng: random.Random, length: int = 4) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def build_trial(rng: random.Random) -> dict:
    a, b = rng.sample(ALL_FACES, 2)
    rest = sorted(f for f in ALL_FACES if f not in (a, b))
    key_a, key_b, key_agg = None, None, None
    used = set()
    while key_a is None or key_a in used:
        key_a = _random_key(rng)
    used.add(key_a)
    while key_b is None or key_b in used:
        key_b = _random_key(rng)
    used.add(key_b)
    while key_agg is None or key_agg in used:
        key_agg = _random_key(rng)

    criteria = {
        key_a: f"The die shows {a}",
        key_b: f"The die shows {b}",
        key_agg: f"The die shows some other number ({', '.join(str(f) for f in rest[:-1])}, or {rest[-1]})",
    }
    key_to_face = {key_a: a, key_b: b}  # key_agg deliberately excluded -- not a single face
    return {"criteria": criteria, "key_to_face": key_to_face, "key_agg": key_agg, "faces": (a, b), "rest": rest}


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README.md).")
    return key


def call_jev(api_key: str, criteria: dict) -> dict:
    payload = {
        "model": MODEL,
        "state": STATE,
        "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
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
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = load_api_key()
    out_dir = ospath.join(REPO_ROOT, "results", "choice", "aggregate_vs_singleton", args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = ospath.join(out_dir, "trials.jsonl")

    trials = []
    if ospath.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            trials = [json.loads(line) for line in f]
    n_done = len(trials)
    rng = random.Random(args.seed)
    for _ in range(n_done):  # advance rng to match already-completed trials
        build_trial(rng)

    if n_done < args.n_trials:
        if n_done:
            print(f"[resume] {n_done}/{args.n_trials} already in {jsonl_path}, continuing")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for _ in tqdm(range(n_done, args.n_trials), desc="choice aggregate dice v3"):
                trial = build_trial(rng)
                raw = call_jev(api_key, trial["criteria"])
                ans = raw["answers"]["answer"]
                chosen_key = ans["choice"]
                is_aggregate = chosen_key == trial["key_agg"]
                rec = {
                    "faces": trial["faces"],
                    "rest": trial["rest"],
                    "chosen_key": chosen_key,
                    "chosen_face": None if is_aggregate else trial["key_to_face"][chosen_key],
                    "aggregate_won": is_aggregate,
                    "p_face_a": ans["probabilities"][list(trial["key_to_face"].keys())[0]],
                    "p_face_b": ans["probabilities"][list(trial["key_to_face"].keys())[1]],
                    "p_aggregate": ans["probabilities"][trial["key_agg"]],
                    "cost": raw.get("usage", {}).get("cost"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                trials.append(rec)

    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)
    n_aggregate_won = sum(1 for t in trials if t["aggregate_won"])
    n_singleton_won = len(trials) - n_aggregate_won

    # Per-face: across every trial where face N was one of the 2 singletons,
    # how often did it specifically win (beat both its sibling singleton AND
    # the aggregate), and what mean probability did it get?
    per_face_appearances = {n: 0 for n in ALL_FACES}
    per_face_wins = {n: 0 for n in ALL_FACES}
    per_face_mean_prob_sum = {n: 0.0 for n in ALL_FACES}
    for t in trials:
        a, b = t["faces"]
        per_face_appearances[a] += 1
        per_face_appearances[b] += 1
        per_face_mean_prob_sum[a] += t["p_face_a"]
        per_face_mean_prob_sum[b] += t["p_face_b"]
        if t["chosen_face"] == a:
            per_face_wins[a] += 1
        elif t["chosen_face"] == b:
            per_face_wins[b] += 1

    print("\n=== Summary ===")
    print(f"n_trials={len(trials)}  aggregate_won={n_aggregate_won}  singleton_won={n_singleton_won}")
    print(f"mean p(aggregate)={statistics.mean(t['p_aggregate'] for t in trials):.3f} (true={TRUE_PROB_AGGREGATE:.3f})")
    print("\nPer-face (only counting trials where that face appeared as a singleton):")
    for n in ALL_FACES:
        appearances = per_face_appearances[n]
        if appearances == 0:
            print(f"  face {n}: never appeared as a singleton")
            continue
        win_rate = per_face_wins[n] / appearances
        mean_prob = per_face_mean_prob_sum[n] / appearances
        print(f"  face {n}: appearances={appearances} win_rate={win_rate:.2f} "
              f"mean_p_when_singleton={mean_prob:.3f} (true={TRUE_PROB_SINGLETON:.3f})")
    print(f"cost: ${total_cost:.5f}")

    summary = {
        "n_trials": len(trials),
        "n_aggregate_won": n_aggregate_won,
        "n_singleton_won": n_singleton_won,
        "mean_p_aggregate": statistics.mean(t["p_aggregate"] for t in trials),
        "per_face": {
            str(n): {
                "appearances": per_face_appearances[n],
                "win_rate": (per_face_wins[n] / per_face_appearances[n]) if per_face_appearances[n] else None,
                "mean_p_when_singleton": (per_face_mean_prob_sum[n] / per_face_appearances[n]) if per_face_appearances[n] else None,
            }
            for n in ALL_FACES
        },
        "total_cost_usd": total_cost,
    }
    with open(ospath.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
