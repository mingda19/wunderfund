# Experiment 6: WP-weighted Huber loss

Builds on the known-good, leaderboard-confirmed recipe exactly (48
channels, 7 layers/dilations to 64, cosine annealing from epoch 0, 15
epochs — `full_training/runs/tcn_full_best.pt`'s config, 0.6414 WP).
Isolates ONE change: `weighted_mse_loss` → `weighted_huber_loss`. Same
`|clip(target,-2,2)|` weighting, same pooled-across-the-batch structure
— deliberately **not** experiment_1's per-sequence-normalized
restructuring (that's the thing already confirmed to hurt on the real
leaderboard) — only the per-element penalty changes from squared error
to Huber.

## Why

`eda/notes.md`'s EDA found the raw targets are heavy-tailed: the train
set's `t0`/`t1` reached out to roughly ±125/±180 despite the metric
clipping to [-2, 2] for scoring. Squared error weights large residuals
quadratically, so a handful of extreme rows can dominate the gradient —
the `|clip(target,-2,2)|` weighting caps how much *weight* an extreme row
gets (max 2), but doesn't change how harshly squared error punishes a bad
prediction's *distance* from an extreme target. Huber caps that growth to
linear past `HUBER_DELTA`, a standard, well-tested robustness fix for
exactly this kind of heavy-tailed noise — never tried here.

## What was verified before trusting this

Hand-computed a small example with one residual in the quadratic region
(`|pred-target|=0.5 < delta=1.0`) and one in the linear region
(`|pred-target|=2.0 > delta=1.0`), worked through the expected
weighted-average loss by hand (0.8125), and confirmed `weighted_huber_loss`
matches exactly — both regions of the piecewise function checked, not
just the easy quadratic case.

Not run: the actual multi-hour training. Dry run first
(`N_TRAIN_SEQUENCES=20`, `N_EPOCHS=2`) before committing to the full run.

## Config reference (only what differs from the baseline)

| Knob | Default | Notes |
|---|---|---|
| `HUBER_DELTA` | 1.0 | Transition point between quadratic and linear regions. Predictions/targets are both roughly O(1) post-clip, so 1.0 is a reasonable starting point. Smaller = more robust to outliers, less sensitive near zero; try 0.5 and 2.0 as a quick sweep if the default result is ambiguous. |

Note `WEIGHTED_LOSS` (the baseline's boolean toggle) isn't present here —
this experiment's whole point is the loss function, so it's hardcoded to
`weighted_huber_loss` rather than left switchable.

## After training finishes

Same as every other experiment: `extract_numpy_weights.py` works
unchanged (loss function only affects training, not the model's
architecture or `state_dict`). Compare `best_score` against 0.6245
before deciding whether it's worth a submission.
