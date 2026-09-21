"""
Follow-up to choice/baseline_check.py. That script replicated
github.com/KantaHayashiAI/jev-does-not-play-dice's finding that Jev's `choice`
question about a fair die always answers "1" with ~83% reported confidence
(vs the true 16.7%). This could be a Choice-format artifact rather than a
genuine calibration failure: TypeSafe's own docs
(docs.typesafe.ai/primitives/choice.md) say nothing about whether `criteria`
object key ORDER matters, and the numeric keys "1".."6" confound "written
first" with "labeled smallest number" -- a well-known LLM position-bias
confound, not necessarily evidence that Jev's probabilities are miscalibrated.

Eight conditions (n=50 each) disentangle this, all using the same die state/
instructions as baseline_check.py:

  shuffled_numeric_keys: criteria keys stay "1".."6" with their natural
    meaning (key "N" always means "the die shows N"), but the order the
    key/value pairs are WRITTEN in the JSON object is a fresh random
    permutation each trial. If Jev's choice follows whichever key was written
    first, that's position bias. If it stays locked on "1" regardless of
    where "1" is written, that's a label/token bias on "1" specifically.

  fixed_letter_keys: criteria keys are "A".."F" (A=face1, ..., F=face6),
    same request repeated every trial, no shuffling -- like the original
    baseline but with the numeric-label confound removed. Tests whether
    "always pick the first-written option" survives without numeric keys.

  shuffled_letter_keys: same A-F keys as fixed_letter_keys, order shuffled
    per trial like shuffled_numeric_keys. Answers the same position-vs-label
    question with the numeric confound fully removed.

  shuffled_values: criteria keys stay "1".."6" in FIXED order every trial,
    but which face-description each key maps to is a fresh random permutation
    each trial (key "1" might describe face 4 in one trial, face 2 the next).
    Tests whether Jev is anchoring on the KEY TOKEN "1" itself (schema-level
    bias, ignoring what the value actually says) independent of position.

  random_keys_fixed_order / random_keys_empty_values: fresh meaningless
    random-string keys each trial (no canonical/alphanumeric ordering), key
    order = generation order (not separately shuffled). The two share the
    IDENTICAL key set per trial index -- fixed_order keeps normal
    "The die shows N" values, empty_values blanks every value out entirely
    (tests whether the bias survives with zero semantic content at all).

  random_keys_shuffled_values: same random-key methodology, but the face
    each key describes is ALSO independently shuffled -- unlike
    random_keys_fixed_order, position 0 no longer always means "The die
    shows 1". Isolates PURE position bias (content present, but decoupled
    from which face-number sits at position 0).

  redundant_value_list: same skeleton as the baseline (numeric keys "1".."6",
    fixed order, identity key->face mapping) -- the ONLY change is that each
    value redundantly restates the full 6-outcome list before naming which
    face this option is, mirroring noul/calibration_check.py's
    list_in_instructions manipulation but for choice's per-option value
    field instead of a question's instructions.

Every condition also records the mean reported probability of whatever gets
chosen -- position/label bias explains WHICH option wins, but the calibration
question (is the winner's reported confidence honestly ~1/6, or still
inflated) is separate and answered regardless of which option that turns out
to be.

Run with: python scripts/choice/bias_check.py --condition shuffled_numeric_keys --n-trials 50
  (--condition all runs every condition in sequence)
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
    "Nobody has observed the outcome yet.\n\n"
    "Which face came up?\n"
)
INSTRUCTIONS = (
    "\nInfer which outcome actually occurred.\n\n"
    "The situation may be genuinely uncertain. Do not assume that the most likely\n"
    "outcome is certain. Preserve uncertainty in the probability distribution.\n"
)

NUMERIC_KEYS = [str(i) for i in range(1, 7)]
LETTER_KEYS = ["A", "B", "C", "D", "E", "F"]
FACES = list(range(1, 7))  # face N described by "The die shows N"

CONDITIONS = [
    "shuffled_numeric_keys",
    "fixed_letter_keys",
    "shuffled_letter_keys",
    "shuffled_values",
    "random_keys_fixed_order",
    "random_keys_empty_values",
    "random_keys_shuffled_values",
    "redundant_value_list",
]


def _random_keys_for_trial(seed: int, trial_idx: int, n: int = 6, length: int = 4) -> list[str]:
    """Deterministic given (seed, trial_idx) so random_keys_fixed_order and
    random_keys_empty_values share the exact same key set per trial index --
    only the value content differs between them."""
    rng = random.Random(f"{seed}-{trial_idx}")
    alphabet = string.ascii_lowercase + string.digits
    keys = []  # list, not set -- preserves generation order deterministically
    while len(keys) < n:
        k = "".join(rng.choice(alphabet) for _ in range(length))
        if k not in keys:
            keys.append(k)
    return keys


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README.md).")
    return key


def call_jev_choice(api_key: str, criteria: dict) -> dict:
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


def build_trial(condition: str, rng: random.Random, seed: int = None, trial_idx: int = None) -> dict:
    """Returns {"criteria": dict, "key_to_face": dict, "written_order": [keys...]}."""
    if condition == "shuffled_numeric_keys":
        key_to_face = {k: int(k) for k in NUMERIC_KEYS}
        order = NUMERIC_KEYS[:]
        rng.shuffle(order)
    elif condition == "fixed_letter_keys":
        key_to_face = dict(zip(LETTER_KEYS, FACES))
        order = LETTER_KEYS[:]
    elif condition == "shuffled_letter_keys":
        key_to_face = dict(zip(LETTER_KEYS, FACES))
        order = LETTER_KEYS[:]
        rng.shuffle(order)
    elif condition == "shuffled_values":
        shuffled_faces = FACES[:]
        rng.shuffle(shuffled_faces)
        key_to_face = dict(zip(NUMERIC_KEYS, shuffled_faces))
        order = NUMERIC_KEYS[:]
    elif condition in ("random_keys_fixed_order", "random_keys_empty_values"):
        random_keys = _random_keys_for_trial(seed, trial_idx)
        key_to_face = dict(zip(random_keys, FACES))
        order = random_keys[:]
    elif condition == "random_keys_shuffled_values":
        random_keys = _random_keys_for_trial(seed, trial_idx)
        face_rng = random.Random(f"{seed}-{trial_idx}-faces")
        shuffled_faces = FACES[:]
        face_rng.shuffle(shuffled_faces)
        key_to_face = dict(zip(random_keys, shuffled_faces))
        order = random_keys[:]
    elif condition == "redundant_value_list":
        key_to_face = {k: int(k) for k in NUMERIC_KEYS}
        order = NUMERIC_KEYS[:]
    else:
        raise ValueError(condition)

    if condition == "random_keys_empty_values":
        criteria = {k: "" for k in order}
    elif condition == "redundant_value_list":
        outcome_list = " ".join(f"{i}) The die shows {i}." for i in range(1, 7))
        criteria = {
            k: f"The possible outcomes are: {outcome_list} This option corresponds to: The die shows {key_to_face[k]}."
            for k in order
        }
    else:
        criteria = {k: f"The die shows {key_to_face[k]}" for k in order}
    return {"criteria": criteria, "key_to_face": key_to_face, "written_order": order}


def run_condition(condition: str, n_trials: int, api_key: str, seed: int) -> list[dict]:
    out_dir = ospath.join(REPO_ROOT, "results", "choice", "bias", condition)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = ospath.join(out_dir, "trials.jsonl")

    trials = []
    if ospath.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            trials = [json.loads(line) for line in f]
    n_done = len(trials)
    rng = random.Random(seed)
    for i in range(n_done):  # advance rng state to match already-completed trials
        build_trial(condition, rng, seed=seed, trial_idx=i)

    if n_done < n_trials:
        if n_done:
            print(f"[resume] {condition}: {n_done}/{n_trials} already in {jsonl_path}, continuing")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for i in tqdm(range(n_done, n_trials), desc=f"dice bias {condition}"):
                trial = build_trial(condition, rng, seed=seed, trial_idx=i)
                raw = call_jev_choice(api_key, trial["criteria"])
                answer = raw["answers"]["answer"]
                chosen_key = answer["choice"]
                rec = {
                    "condition": condition,
                    "written_order": trial["written_order"],
                    "key_to_face": trial["key_to_face"],
                    "first_written_key": trial["written_order"][0],
                    "choice": chosen_key,
                    "chosen_face": trial["key_to_face"][chosen_key],
                    "chosen_was_first_written": chosen_key == trial["written_order"][0],
                    "probabilities": answer["probabilities"],
                    "confidence": answer.get("confidence"),
                    "cost": raw.get("usage", {}).get("cost"),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                trials.append(rec)

    return trials


def summarize(condition: str, trials: list[dict]) -> dict:
    choice_counts = Counter(t["choice"] for t in trials)
    face_counts = Counter(t["chosen_face"] for t in trials)
    top_probs = [t["probabilities"][t["choice"]] for t in trials]
    n_first_written = sum(t["chosen_was_first_written"] for t in trials)
    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)
    summary = {
        "condition": condition,
        "n_trials": len(trials),
        "choice_key_distribution": dict(choice_counts),
        "chosen_face_distribution": dict(face_counts),
        "frac_chosen_was_first_written_key": n_first_written / len(trials) if trials else None,
        "mean_reported_prob_of_chosen": statistics.mean(top_probs) if top_probs else None,
        "total_cost_usd": total_cost,
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=CONDITIONS + ["all"])
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    api_key = load_api_key()
    conditions = CONDITIONS if args.condition == "all" else [args.condition]

    summaries = {}
    for cond in conditions:
        trials = run_condition(cond, args.n_trials, api_key, args.seed)
        summaries[cond] = summarize(cond, trials)
        s = summaries[cond]
        print(f"[done] {cond}: n={s['n_trials']} "
              f"chosen_key_dist={s['choice_key_distribution']} "
              f"frac_first_written={s['frac_chosen_was_first_written_key']:.2f} "
              f"mean_reported_prob={s['mean_reported_prob_of_chosen']:.3f}")

    out_path = ospath.join(REPO_ROOT, "results", "choice", "bias", "summary.json")
    existing = {}
    if ospath.exists(out_path):
        existing = json.load(open(out_path, encoding="utf-8"))
    existing.update(summaries)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
