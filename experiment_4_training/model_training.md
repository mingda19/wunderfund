# Experiment 4: weight decay (AdamW)

Builds on the known-good, leaderboard-confirmed recipe exactly (48
channels, 7 layers/dilations to 64, `weighted_mse_loss`, cosine annealing
from epoch 0, 15 epochs — `full_training/runs/tcn_full_best.pt`'s config,
0.6414 WP). Isolates ONE change: `torch.optim.Adam` → `torch.optim.AdamW`
with a nonzero `WEIGHT_DECAY`. Nothing else differs.

## Why

No weight decay has been used in any TCN run so far (`torch.optim.Adam(...)`
with no `weight_decay` argument defaults to 0). Every full-data training
log so far shows the same pattern: training loss keeps falling while
validation WP plateaus a few epochs before the run ends — the classic
mild-overfitting signature (explicitly called out in `full_training`'s
own log analysis). Weight decay is the standard, cheap, well-understood
response to exactly this, and it's genuinely never been tried.

**AdamW, not vanilla Adam + L2 penalty**: vanilla L2 regularization (adding
`weight_decay * w` to the gradient before Adam's adaptive scaling) and
AdamW's decoupled decay (`w -= lr * weight_decay * w` applied directly,
separately from the gradient-based update) behave differently in
practice — AdamW's decoupling is now the standard/preferred form
precisely because it doesn't interact unpredictably with Adam's per-
parameter learning-rate adaptation. Using `AdamW` directly rather than
retrofitting `weight_decay=` onto `Adam`.

## What was verified before trusting this

This is the simplest of the five experiments — the only functional change
from the known-good baseline is the optimizer line. Syntax-compiled,
module-imported, and constructed the actual `AdamW` optimizer against a
real `CausalTCN(n_input=112, channels=48, dilations=(1,2,4,8,16,32,64))`
instance to confirm it builds correctly with the configured weight decay
before trusting it (not just reading the torch docs and assuming).

Not run: the actual multi-hour training. Dry run first
(`N_TRAIN_SEQUENCES=20`, `N_EPOCHS=2`) before committing to the full run.

## Config reference (only what differs from the baseline)

| Knob | Default | Notes |
|---|---|---|
| `WEIGHT_DECAY` | 0.01 | AdamW's decoupled decay coefficient. Setting this to `0.0` should reproduce the baseline's ~0.6245 local score almost exactly (a free sanity check that nothing else drifted) — worth trying first if you want to confirm the harness itself is unchanged before spending a full run on the real test. Reasonable sweep range otherwise: 0.001–0.1, log-spaced. |

Everything else matches the known-good baseline exactly.

## After training finishes

Same as every other experiment: `extract_numpy_weights.py` works
unchanged (weight decay only affects how the optimizer updates
parameters during training, not the model's architecture or the saved
`state_dict`'s shape/keys). Compare `best_score` against 0.6245 before
deciding whether it's worth a submission.
