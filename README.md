# Jev Noul vs Choice — Does Jev Play Dice?

Inspired by [KantaHayashiAI/jev-does-not-play-dice](https://github.com/KantaHayashiAI/jev-does-not-play-dice),
which found that [Jev](https://docs.typesafe.ai/concepts/system-one) — TypeSafe's
"System One" model, marketed on returning calibrated probabilities instead of
free-text LLM-judge reasoning — reports ~83% confidence for a face it always
picks as "1" when asked "which face came up?" on a fair six-sided die, instead
of the true 16.7%.

This project asks the natural follow-up: **is that a property of Jev's
probabilities in general, or specific to how the question is asked?**

**Short answer: it's specific to the `choice` question type.** The identical
"fair die" scenario, asked through Jev's `noul` question type (an independent
probability for a single proposition) instead of `choice` (pick one of N
labeled options), is calibrated to within about a percentage point of the true
1/6 for every face — and correctly assigns near-zero probability to an
impossible 7th face. `choice` stays badly broken no matter how the prompt is
varied.

## Headline result

| question type | mean reported P(winning face) | true P |
|---|---:|---:|
| `choice` (6-way, "which face came up?") | **0.835** | 0.167 |
| `noul` (6 independent "is it face N?" questions) | **0.157 – 0.177** per face | 0.167 |

`choice` always answers "1" (50/50 trials); `noul` spreads its answers
correctly across all six faces and assigns 0.01 to the impossible 7th face
(`stdev = 0.000` — it never wavers on that one).

## Why: root-causing the `choice` failure (not a prompt-engineering artifact)

A series of controlled variations (`scripts/choice/bias_check.py`'s 8
named conditions, plus 3 follow-up scripts), all n=50 trials, ruled out every
prompt-level explanation we could think of:

1. **Not position/order bias.** Shuffling the JSON order of the 6 options
   (`shuffled_numeric_keys`, `shuffled_letter_keys`) doesn't move the answer
   off "1" — the option written first is picked only ~16% of the time
   (chance level).
2. **Not a specific key label.** Numeric keys ("1".."6"), letter keys
   ("A".."F"), and fully random meaningless keys (`shuffled_numeric_keys`,
   `fixed_letter_keys`, `shuffled_letter_keys`, `random_keys_fixed_order`,
   `random_keys_shuffled_values`) all still lock onto whichever option's
   *content* says "The die shows 1" — 96-100% of trials, regardless of what
   the winning key is literally named.
3. **Not description verbosity.** Making every option's text maximally
   redundant (repeating the full 6-outcome list inside each value,
   `redundant_value_list`) doesn't help — confidence goes *up* to 0.897.
4. **Not "seeing the other options."** Giving `noul` the exact same 6-option
   list `choice` sees — either in the shared `state`
   (`list_in_state`) or embedded in each question's own
   `instructions` (`list_in_instructions`, matching where `choice`'s
   `criteria` physically lives) — doesn't touch its calibration at all.
5. **Not acquiescence to a "yes" label.** Rotating which face gets tagged
   `"yes"` across 6 parallel 6-way choices: the label carries *zero* pull.
   Whichever question it's in, `"no_1"` (face 1's content) keeps winning
   even when `"yes"` is attached to a completely different face
   (`scripts/choice/yes_label_rotation_check.py`).
6. **Not simply "smallest available label."** Remove face 1 from the option
   set entirely (faces 2-5 only) and a *new* magnet appears — face **3** —
   even though 2 is smaller and available.
7. **It's a winner-take-all mechanism, not a miscalibrated-but-honest one.**
   When one option is a genuine aggregate ("some other number", true P=4/6
   vs. 1/6 per singleton), `choice` *does* pick the aggregate correctly —
   but crushes it to ~0.90-0.91 confidence (true 0.667) and crushes the
   losing singletons far below their true 0.167, down to ~0.01-0.06
   (`scripts/choice/aggregate_vs_singleton_check.py`, 3 independent runs,
   fully randomized meaningless keys in the final version). Face 1 alone
   keeps a faint echo of its earlier magnetism — even losing, its mean
   probability (0.185) stays much closer to the true value than every other
   losing face (0.011–0.059).

**Working theory:** `choice`'s failure isn't something a caller can prompt
their way around — it looks like an internal decoding routine that collapses
onto a near-deterministic single winner (something like an extremely
low-temperature softmax) regardless of genuine uncertainty. When several
options are genuinely tied, the "winner" is decided by a content-level prior
baked into the underlying model — echoing the well-documented human/LLM bias
toward specific "psychologically salient" numbers (1, 3, 7, ...) when asked to
"pick something at random." `noul`'s independent per-proposition probability
appears to go through a different, much better-calibrated path.

## Reproducing

Requires an OpenRouter API key with access to `typesafe/jev-1.13`
(`api.typesafe.ai` itself was invite-only at the time of writing; OpenRouter
was a working alternative path — see `docs.typesafe.ai` for Jev's own docs).

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
python scripts/choice/baseline_check.py --n-trials 50 --run-name my_run
```

Scripts are split into `scripts/choice/` and `scripts/noul/` by which Jev
question type they exercise. Every script is independently runnable — see the
docstring at the top of each for exactly what it tests and why. Runs are
resumable (rerunning with the same `--run-name`, or the same `--condition` for
`bias_check.py`, picks up where it left off), and raw per-trial output is
always saved to `results/.../trials.jsonl` alongside a `summary.json`.

## Experiment index

| script | what it tests | key result dirs |
|---|---|---|
| `choice/baseline_check.py` | Baseline replication of the original repo's finding via OpenRouter instead of Vercel AI Gateway | `results/choice/baseline/openrouter_v1/` |
| `choice/bias_check.py` | 8 named conditions disentangling position bias, key-label bias, content bias, and description verbosity in the 6-way `choice` | `results/choice/bias/<condition>/`, `.../summary.json` |
| `noul/calibration_check.py` | `noul` in isolation (`isolated_openrouter_v1`), plus `list_in_state` (option list added to `state`) and `list_in_instructions` (added to each question's own `instructions`) | `results/noul/calibration/<condition>/` |
| `choice/binary_check.py` | Simplest possible `choice`: 2-way yes/no per face | `results/choice/binary/openrouter_v1/` |
| `choice/yes_label_rotation_check.py` | Rotates which face gets the `"yes"` label across a 6-way (and, with face 1 removed, a 4-way) choice | `results/choice/yes_label_rotation/openrouter_v1/`, `.../faces_2345_v1/` |
| `choice/aggregate_vs_singleton_check.py` | 3-way choice: 2 singleton options + 1 aggregate option covering the rest; 3 runs of increasing rigor (fixed faces → randomized faces → randomized faces + meaningless keys) | `results/choice/aggregate_vs_singleton/openrouter_v1/`, `.../randomized_v1/`, `.../randomkeys_v3/` |

`choice/bias_check.py`'s 8 conditions (`--condition <name>`, or `all` to run
every condition in sequence):

| condition | what it varies |
|---|---|
| `shuffled_numeric_keys` | keys "1".."6", JSON write-order shuffled each trial |
| `fixed_letter_keys` | keys "A".."F" (A=face1 .. F=face6), fixed order |
| `shuffled_letter_keys` | same as above, write-order shuffled each trial |
| `shuffled_values` | keys "1".."6" fixed order, but which face each key *describes* is shuffled each trial |
| `random_keys_fixed_order` | fresh meaningless random-string keys each trial, normal values |
| `random_keys_empty_values` | same random keys, but every value blanked to `""` |
| `random_keys_shuffled_values` | same random keys, plus the face-to-key mapping independently shuffled |
| `redundant_value_list` | keys "1".."6" fixed order, but each value redundantly restates the full 6-outcome list |

## Relationship to the original repo

Independent project, not affiliated with TypeSafe, OpenRouter, or the original
repo's author — this one root-causes *why* `choice` miscalibrates rather than
documenting *that* it does.

## License

MIT — see [LICENSE](LICENSE).
