"""
Follow-up to choice/binary_check.py. That test used a plain 2-option
{"yes": "shows N", "no": "does not show N"} choice per face and found no
lock-in (unlike the 6-way bias_check.py's experiments choice experiments), but a mild systematic
p(yes) decay across faces 1->6.

This test asks: is there an independent pull toward the literal LABEL "yes"
(distinct from the "face 1 content" pull found in bias_check.py's experiments)? Structure: for each
target face T in 1..6, one 6-way `choice` question where T's own option is
keyed "yes" and the other five faces are keyed "no_M" (M != T) -- e.g. for
T=2: {"yes": "The die shows 2", "no_1": "The die shows 1", "no_3": "...3",
"no_4": "...4", "no_5": "...5", "no_6": "...6"}. All 6 target-rotations (T=1
through T=6) asked as parallel questions in one request per trial.

If the pull is on the "yes" LABEL, whichever face is currently tagged "yes"
should win regardless of which face that is. If the pull is on face-1
CONTENT (per bias_check.py's experiments), "no_1" (face 1's option, wherever it's labeled) should
keep winning even when "yes" is attached to a different face -- i.e. for
T != 1, we'd see "no_1" beat "yes" more often than chance.

Run with: python scripts/choice/yes_label_rotation_check.py --n-trials 50 --run-name <name>
"""
import argparse
import json
import os
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
FACES = list(range(1, 7))


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README.md).")
    return key


def _criteria_for_target(target: int, faces: list[int]) -> dict:
    criteria = {"yes": f"The die shows {target}"}
    for m in faces:
        if m != target:
            criteria[f"no_{m}"] = f"The die shows {m}"
    return criteria


def _key_to_face(target: int, key: str) -> int:
    return target if key == "yes" else int(key.split("_")[1])


def call_jev_batch(api_key: str, faces: list[int]) -> dict:
    questions = {
        f"target_{t}": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": _criteria_for_target(t, faces)}
        for t in faces
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
    p.add_argument("--faces", default="1,2,3,4,5,6",
                    help="comma-separated face numbers to include as options, e.g. '2,3,4,5' to exclude 1 and 6 entirely")
    args = p.parse_args()
    faces = [int(x) for x in args.faces.split(",")]

    api_key = load_api_key()
    out_dir = ospath.join(REPO_ROOT, "results", "choice", "yes_label_rotation", args.run_name)
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
            for _ in tqdm(range(n_done, args.n_trials), desc="yesno6 dice batch"):
                raw = call_jev_batch(api_key, faces)
                rec = {"cost": raw.get("usage", {}).get("cost"), "model": raw.get("model"), "targets": {}}
                for t in faces:
                    ans = raw["answers"][f"target_{t}"]
                    chosen_key = ans["choice"]
                    rec["targets"][str(t)] = {
                        "chosen_key": chosen_key,
                        "chosen_face": _key_to_face(t, chosen_key),
                        "p_yes": ans["probabilities"]["yes"],
                    }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                trials.append(rec)

    total_cost = sum(t["cost"] for t in trials if t.get("cost") is not None)

    print(f"\n=== Summary (faces={faces}, per target face T, what actually won) ===")
    for t in faces:
        chosen_faces = Counter(tr["targets"][str(t)]["chosen_face"] for tr in trials)
        frac_yes_won = sum(1 for tr in trials if tr["targets"][str(t)]["chosen_key"] == "yes") / len(trials)
        mean_p_yes = sum(tr["targets"][str(t)]["p_yes"] for tr in trials) / len(trials)
        print(f"target_{t} (yes=face{t}): chosen_face_dist={dict(sorted(chosen_faces.items()))} "
              f"frac_yes_won={frac_yes_won:.2f} mean_p_yes={mean_p_yes:.3f}")
    print(f"cost: ${total_cost:.5f}")

    summary_path = ospath.join(out_dir, "summary.json")
    summary = {
        "n_trials": len(trials),
        "faces": faces,
        "per_target": {
            str(t): {
                "chosen_face_dist": dict(Counter(tr["targets"][str(t)]["chosen_face"] for tr in trials)),
                "frac_yes_won": sum(1 for tr in trials if tr["targets"][str(t)]["chosen_key"] == "yes") / len(trials),
                "mean_p_yes": sum(tr["targets"][str(t)]["p_yes"] for tr in trials) / len(trials),
            }
            for t in faces
        },
        "total_cost_usd": total_cost,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()
