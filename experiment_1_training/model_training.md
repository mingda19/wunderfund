# Experiment 1: WP-aligned loss + one more layer + delayed LR decay

Builds on `full_training/train_TCN.py` (read that folder's
`model_training.md` first — this doc only covers what's different).
Three deliberate changes, bundled into one run per your request, plus two
real bugs this work uncovered and fixed along the way.

## What changed and why

### 1. New loss function: `wp_aligned_loss`

This is the direct exploitation of the `a5`/`a7` quantization finding
(`eda/notes.md` §9) you asked for — but the fix lives in *how the loss
combines rows across sequences*, not in referencing `a5`/`a7` explicitly.
Here's the reasoning:

- §9 found target scale varies a lot **between sequences** (observed step
  sizes ranged ~5x, 0.125 to 0.657, in the sampled data) and that `a5`/`a7`
  predict this scale (corr 0.65/0.74).
- The old loss (`weighted_mse_loss`, still available via `LOSS_TYPE =
  "weighted_mse"`) computes `sum(w * (pred-target)^2) / sum(w)` pooled
  over **every row in the whole batch** — sequences, time steps, and both
  targets all combined into one division. Since `w = |clip(target,-2,2)|`,
  a sequence with a bigger quantization step has systematically bigger
  weights, so it silently dominates that pooled sum. A batch with one
  high-step-size sequence and several low-step-size ones effectively
  trains mostly on the one big sequence.
- But the **actual WP metric** doesn't work that way: it computes a
  weighted correlation *per sequence*, averages `t0`/`t1` per sequence,
  then averages **equally** across sequences (`docs/data_overview.md`'s
  `S = mean over b` of `s_b`). Every sequence counts the same at the top
  level, regardless of its target scale.
- `wp_aligned_loss` mirrors that exactly: normalize by each
  *(sequence, target)* pair's own weight sum first, average the two
  targets per sequence, **then** average equally across sequences. A toy
  hand-checked example (two sequences with a 10x scale difference)
  confirmed the two losses give meaningfully different numbers — see the
  verification note at the bottom of this file.

This is a pure loss-function change — no new model output, no
architecture change, no inference-time dependency on `a5`/`a7` at all
(they're not referenced in the code). It works because the normalization
uses the *true* training-time targets, which we always have during
training; nothing needs to be inferred from `a5`/`a7` at inference time
the way a two-head scale-prediction model would.

`LOSS_TYPE` is a config toggle (`"wp_aligned"` / `"weighted_mse"` /
`"plain_mse"`) so you can A/B it against the old loss later if you want,
holding architecture fixed.

### 2. One more layer

`DILATIONS` extended to `(1,2,4,8,16,32,64,128)` — 8 layers instead of 7,
receptive field 511 steps instead of 255. Params: 70,802 (up from ~64K).
Estimated latency impact: ~1/8 more compute per step; if the 7-layer
model's measured 63.6µs/row holds its rough proportionality, expect
somewhere around 70-75µs/row, still comfortably inside the 60-minute
budget (~44-47 min projected for a 1,873-sequence test set) — **worth
actually measuring once trained**, not just trusting this estimate.

### 3. Delayed LR decay

`LR_FLAT_EPOCHS = 8`: learning rate stays at `2e-3` for epochs 1-8, then
cosine-decays to ~0 over the remaining 7 epochs (9-15). Motivated directly
by the full run's log: validation peaked at epoch 7 and never
re-exceeded that afterward, while training loss kept falling — with the
old schedule (`CosineAnnealingLR(T_max=15)` from epoch 0), the LR was
already well into its decay by epoch 7-8, so it's not clear whether the
plateau was genuine convergence or the schedule cutting the model off
early. This gives it a longer full-strength run before decay starts.

**A real bug found while verifying this**: PyTorch's `SequentialLR`
switches sub-schedulers *one epoch later* than the `milestones` value
naively suggests — `milestones=[8]` actually produces 9 flat epochs, not
8 (confirmed empirically, see the test note below). `build_scheduler()`
uses `milestones=[LR_FLAT_EPOCHS - 1]` to correct for this — verified to
produce exactly 8 flat epochs (1-8) then decay (9-15) before trusting it.
If you ever touch this function, re-verify — it's a real gotcha, not
obvious from the docs.

### 4. Validation set: all 1,873 real `valid.parquet` sequences, streamed

Per your decision: `valid.parquet` stays validation-only, never trained
on. Superseded once more, per your follow-up request: rather than the
150-sequence `valid_sample.parquet` sample, checkpoint selection now runs
against **all 1,873 sequences of the actual `valid.parquet`** — a ~12.5x
larger validation set, which should make the "best val score" signal
noticeably less noisy epoch to epoch.

This doesn't fit in memory at once (1,873 × 20,000 × 112 × 4 bytes ≈
16.8GB for the features alone — over the 16GB budget by itself, before
the model/optimizer/OS). So `evaluate_streaming()` reads `VALID_BATCH_SIZE`
(32) sequences at a time from disk, evaluates, discards, and accumulates
the six running sums needed for an exact single-pass Pearson correlation
(`n, sum_p, sum_y, sum_p·y, sum_p², sum_y²`) — the same streaming approach
already used in `eda/scripts/streaming_stats.py` for the original
full-dataset EDA. This is **not** the same as averaging per-batch
correlations (which would be a statistically different, worse estimate);
verified the math is exact by comparing it against directly pooling and
computing `np.corrcoef` on synthetic batches of varying size and scale —
matched to 1e-16 (float64 rounding noise only).

**Cost**: this reads through all of `valid.parquet` once per epoch (every
`VALIDATE_EVERY_N_EPOCHS`, currently every epoch), which is real added
time — at the ~20-30 seq/s I/O rate observed in the full_training run,
figure roughly 60-90s per validation pass, ×15 epochs ≈ **15-25 minutes
added to total training time**. Worth it for the accuracy, per your ask,
but worth knowing going in.

**Consequence worth knowing**: since *every* sequence in `valid.parquet`
is now used for checkpoint selection in *every* future run, there's no
leftover slice of it that's genuinely untouched for a blind before/after
comparison anymore (this was already true once `full_training/train_TCN.py`
moved to using all 150 of `valid_sample.parquet`, and is now even more
true since this run uses the entire real file). That's fine for comparing
runs against each other (both this experiment and the full_training
baseline use the identical protocol — validation-only, never trained on —
so their "best val score" numbers are still a fair apples-to-apples
signal), but if you want a truly blind local check before spending a
submission, it'd need a fresh slice carved from `train.parquet` instead
(not built here — flagging it since it's a direct
consequence of this policy, not building it unless you want it).

## A second real bug found and fixed (not part of your ask, but blocking)

While verifying this experiment's 8-layer model would actually deploy
correctly, I found that `eda/scripts/tcn_streaming.py`'s `_extract_weights`
— the function that converts a trained model into the numpy weights
`solution.py` runs — read each layer's dilation from a **hardcoded
module-level constant** (`tcn_model.DILATIONS`, fixed at the 7-layer
default) instead of the actual model being extracted:

```python
for block, d in zip(model.blocks, DILATIONS):   # DILATIONS = the 7-entry default, always
```

For any model with a *different* dilation schedule — like this
experiment's 8 layers — `zip()` silently truncates to the shorter list,
**dropping the 8th layer's weights entirely** with no error. This didn't
affect anything already deployed (every model trained so far happened to
use the default dilations), but it would have silently produced a broken
model the moment this experiment's checkpoint was extracted.

Fixed to read `d = block.conv.dilation[0]` directly from each layer's own
`Conv1d` (a real PyTorch attribute, always correct for that specific
layer) instead of the external constant. Verified both directions before
trusting it:
- **Regression**: re-ran the existing 7-layer deployed model through the
  fixed code — still matches the batched PyTorch forward pass to 1e-6
  (same precision as before the fix).
- **New case**: built an untrained 8-layer test model, ran it through
  `save_checkpoint` → `extract_numpy_weights.py` → `solution.py`'s
  `StreamingCausalTCN`, and confirmed the numpy-extracted version matches
  the batched forward pass to 1.6e-6, and matches a direct (non-round-
  tripped) extraction exactly (0.0 diff).

This fix lives in `eda/scripts/tcn_streaming.py`, shared by both
`full_training/` and this folder's `extract_numpy_weights.py` — you don't
need to do anything extra, but it's worth knowing this was silently broken
for any non-default architecture until now.

## Config reference (only what differs from `full_training/`)

| Knob | Value | Notes |
|---|---|---|
| `DILATIONS` | `(1,2,4,8,16,32,64,128)` | 8 layers, RF=511 |
| `LOSS_TYPE` | `"wp_aligned"` | new default; `"weighted_mse"` available for A/B |
| `LR_FLAT_EPOCHS` | 8 | flat LR through epoch 8, cosine decay epochs 9-15 |
| `N_VALID_SEQUENCES` | `None` | `None` = all 1,873 real `valid.parquet` sequences; set an int to cap |
| `VALID_BATCH_SIZE` | 32 | sequences per streaming-validation batch (no backward pass, so can exceed `BATCH_SIZE`) |
| `N_EPOCHS` | 15 | unchanged, to isolate the effect of the other three changes |
| `RUN_NAME` | `"tcn_experiment1"` | checkpoints in `runs/tcn_experiment1_{best,latest}.pt` |
| `RESUME_FROM` | `None` | cannot resume from `full_training`'s checkpoint — different architecture (8 layers vs 7), incompatible state_dict shapes |

Everything else (streaming loader, checkpoint-per-epoch, Ctrl-C safety,
`BATCH_SIZE=16`, `GRAD_CLIP=1.0`, `DROPOUT=0.1`) is unchanged from
`full_training/train_TCN.py`.

## Before you commit to the full run

Same advice as `full_training/`: dry-run first (`N_TRAIN_SEQUENCES = 20`,
maybe `N_EPOCHS = 2`) to exercise every path in a couple minutes.

What I verified without running the actual training (per your "just
build it" ask — I did not run the full multi-hour training myself):
- Syntax-compiled and module-imported the script (no execution of
  `main()`).
- Hand-computed and confirmed `wp_aligned_loss`'s exact output on a toy
  2-sequence example with a 10x scale difference (0.3175 expected and
  got, vs. 0.7038 for the old pooled loss on the same input — concretely
  showing the fix changes behavior, not just in theory).
- Empirically confirmed the `SequentialLR` off-by-one and verified the
  fix produces exactly 8 flat epochs then 7 decaying ones.
- Built the actual 8-layer `CausalTCN`, ran a forward pass, saved/loaded
  a checkpoint, and ran it through `extract_numpy_weights.py` and
  `solution.py`'s inference class — full pipeline, verified numerically
  correct end to end (see the bug-fix section above for the precision
  numbers).

## Expected runtime

Same I/O-bound profile as `full_training/` (the actual run there measured
~118 min for 15 epochs at ~17-33 seq/s, fairly variable — likely disk/OS
contention, not something in our control). The extra layer adds ~1/8 more
per-step compute, which earlier profiling suggested is a small fraction
of total epoch time (I/O-dominated) — expect **roughly 120-135 minutes**,
but this is an estimate extrapolated from the full_training run's numbers,
not measured for this specific architecture.

## After training finishes

Same as `full_training/`: run `extract_numpy_weights.py runs/tcn_experiment1_best.pt`,
copy the resulting `.npz` into `wunderfund/solution/` alongside
`standardization.npz` (unchanged, still valid), point `solution.py`'s
`_load_weights()` at the new filename, and validate before trusting the
number — ideally via a fresh held-out slice (see the validation-set note
above) rather than `valid.parquet`'s now-fully-used 1,873 sequences.
