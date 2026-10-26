# Experiment 3: deep ensemble via independently-seeded training runs

Builds on the known-good, leaderboard-confirmed recipe exactly (48
channels, 7 layers/dilations to 64, `weighted_mse_loss`, cosine annealing
from epoch 0, 15 epochs — `full_training/runs/tcn_full_best.pt`'s config,
0.6414 WP). The **only** change across runs of this script is `SEED` —
architecture, data, optimizer, schedule, everything else is identical.

## Why this is a different bet than the LightGBM ensemble that failed

`eda/notes.md` §11: TCN+LightGBM blending didn't help because LightGBM's
errors weren't *structurally* diverse from the TCN's — same underlying
information, a weaker model reading it, so there was nothing independent
to exploit. Two TCNs trained from different random seeds are a different
situation: identical capacity and architecture, but different weight
initialization and different SGD trajectories (different mini-batch
order) converge to different local optima of the *same* loss landscape.
Their mistakes are genuinely less correlated — this is the textbook
condition under which averaging reduces variance ("deep ensembles"), not
a retry of the thing that already failed for a different reason.

## How to use this

Run the script, then edit `SEED` (and `RUN_NAME`, or you'll overwrite the
previous member's checkpoints) and run again — 2-3 times total:

```
SEED=1, RUN_NAME="tcn_exp3_seed1"   (already the default)
SEED=2, RUN_NAME="tcn_exp3_seed2"
SEED=3, RUN_NAME="tcn_exp3_seed3"   (optional third member)
```

Each run is a full ~2.6-3 hour training pass (same cost as
`full_training`'s original run) — this is the most expensive of the five
experiments in wall-clock terms precisely because it's 2-3 full runs, not
one. You said training duration isn't a concern, but worth knowing going in.

`torch.manual_seed(SEED)` is called once at the top of `main()`, before
the model is constructed — this controls weight initialization. The
per-epoch data shuffle (`TRAIN_SEED`, which always equals `SEED` here)
also changes, so each member sees a genuinely different training
trajectory, not just a different starting point.

## After training finishes: averaging the ensemble at inference

Not built here — this script only produces individual checkpoints. Once
you have 2-3 trained members:

1. Run `extract_numpy_weights.py` on each member's `_best.pt`, same as
   any other checkpoint, giving 2-3 separate `.npz` weight files.
2. Build an ensemble `PredictionModel` (would live in `wunderfund/solution/`,
   analogous to `solution_pls.py`) that holds N independent
   `StreamingCausalTCN` instances — one per member, each with its own
   `.reset()` on sequence boundaries — and averages their `step()` outputs
   each row. This is a small, low-risk addition once you have real
   checkpoints to build it against (no point building and "verifying" it
   against nothing) — ask and I'll build it as soon as you have 2+
   extracted `.npz` files.
3. **Latency**: N models means roughly N× the TCN compute per row. At
   the current fused implementation's ~46µs/row for one model, 2 models
   ≈ ~92µs/row (≈57 min projected — tight), 3 models ≈ ~138µs/row (over
   budget on its own, before even adding the PLS-style per-model
   overhead). **Practically this likely means 2 members, not 3**, unless
   paired with the ring-buffer fix (still deferred, per your call) to
   reclaim headroom first. Measure for real once you have real
   checkpoints rather than trusting this extrapolation — same discipline
   as everywhere else in this project.

## What was verified before building this

This script is structurally identical to `full_training/train_TCN.py`'s
known-good version (same streaming loader, same `BlockAccumulator`-based
validation, same checkpoint/resume logic) with only `torch.manual_seed(SEED)`
added — syntax-compiled and module-imported cleanly, no new logic to
verify beyond that single addition, since every other code path is
unchanged from a script that has already been run successfully for real.

Not run: the actual multi-hour training(s). Dry run first
(`N_TRAIN_SEQUENCES=20`, `N_EPOCHS=2`) before committing to the full run,
same as every other experiment folder.

## Config reference (only what differs from the baseline)

| Knob | Default | Notes |
|---|---|---|
| `SEED` | 1 | Change this (and `RUN_NAME`) for each ensemble member. |
| `RUN_NAME` | `"tcn_exp3_seed1"` | Must change alongside `SEED` or you'll overwrite the previous member. |

Everything else matches the known-good baseline exactly.
