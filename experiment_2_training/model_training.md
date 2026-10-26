# Experiment 2: a5/a7 quantization-scale auxiliary loss

Builds on the **known-good, leaderboard-confirmed recipe**
(`full_training/runs/tcn_full_best.pt`'s exact config: 48 channels, 7
layers/dilations to 64, `weighted_mse_loss`, cosine annealing from epoch
0, 15 epochs — 0.6414 WP on the real leaderboard) — **not** the
8-layer/64-channel config currently sitting in `full_training/train_TCN.py`
on disk, which you've already found doesn't help. Isolates exactly one
change, per the lesson from experiment_1 (bundling the loss change with
architecture/schedule changes made it impossible to tell which one hurt).

## The idea

`eda/notes.md` §9 found `t0`/`t1` are per-sequence quantized — each
sequence's targets are small integer multiples of a per-sequence
`step_size` — and that `a5`/`a7` predict this step_size at corr 0.65/0.74.
This was flagged repeatedly as the one concrete, specific, still-unused
signal in the whole project. experiment_1's `wp_aligned_loss` was
*motivated* by this finding but attacked it structurally (reweighting how
the loss combines across sequences) rather than giving the model any
direct training signal about the scale itself — and it didn't help on the
real leaderboard. This experiment attacks it differently: an auxiliary
head predicting the scale directly.

## How it works

1. **Ground truth**: for every training sequence, compute the true
   `step_size` directly from its real `t0`/`t1` — the smallest nonzero
   `|t0|` or `|t1|` value in that sequence (same estimator the original
   EDA used). This only needs the true targets, which we always have
   during training.
2. **Auxiliary head**: `CausalTCNWithAux` adds one small `Conv1d(channels, 1, 1)`
   reading the *same* final hidden state the main head reads, predicting
   `log(step_size)` per row. (Log-space because step_size is strictly
   positive and spans a real range — stabilizes the regression.)
3. **Combined loss**: `main_loss (weighted_mse_loss, unchanged) + AUX_LOSS_WEIGHT * aux_loss`,
   where `aux_loss` is plain MSE between the row-averaged aux prediction
   and the true `log(step_size)` for that sequence.

This is standard multi-task auxiliary regularization: the *shared* TCN
layers get a gradient signal from both tasks, which can push them to
extract the scale-determining information (presumably derivable from
`a5`/`a7`, since that's what the EDA found) in a way that's also useful
for the main t0/t1 prediction — without the deployed model needing to do
anything new or different at inference.

## Why this needs zero deployment-side changes

The aux head only matters during training. `save_checkpoint` strips every
`aux_head.*` key from the saved `state_dict` before writing it — the
deployable checkpoint has exactly the same keys/shapes as a plain
`tcn_model.CausalTCN`, so `extract_numpy_weights.py` and `solution.py`
work completely unchanged. Verified this explicitly before trusting it:
constructed a `CausalTCNWithAux`, stripped its `aux_head.*` keys, and
loaded the result into a fresh plain `CausalTCN` — no errors, shapes
match exactly (see the `__main__`-free verification run that produced
this file; not re-derived from the docs, actually executed).

(A full resumable checkpoint — `full_state_dict_with_aux` — is also saved
alongside, used only by `RESUME_FROM`, so interrupting and resuming
doesn't lose the aux head's trained weights mid-run.)

## What was verified before trusting this

- `CausalTCNWithAux` constructs and runs a forward pass with correct
  output shapes: main head `(batch, 2, time)`, aux head `(batch, 1, time)`.
- `compute_log_step_size` checked against a hand-constructed example with
  a tricky edge case (one sequence where an entire target dimension is
  all-zero) — exact match to the manually computed expectation, including
  correctly taking the min across *both* t0 and t1 combined, not treating
  them separately.
- The state_dict-stripping/reload path (the thing that makes this
  deployable with zero extra tooling) — confirmed to work, not assumed.

Not run: the actual multi-hour training (per your usual ask — build it,
don't run it). Do a quick dry run first (`N_TRAIN_SEQUENCES=20`,
`N_EPOCHS=2`) before committing to the full run.

## Config reference (only what differs from the baseline)

| Knob | Default | Notes |
|---|---|---|
| `AUX_LOSS_WEIGHT` | 0.1 | Weight on the auxiliary loss. This is a regularizer, not the primary objective — if it shows no effect either way, try 0.05 and 0.2 before concluding the idea doesn't work; if it visibly hurts the main-task validation score, that's a real (if negative) finding worth recording, same as experiment_1's. |
| `STEP_SIZE_EPS` | 1e-6 | Floor to avoid `log(0)` on degenerate all-zero-target sequences (shouldn't occur with real data, but guarded). |

Everything else (architecture, batch size, LR, schedule, validation) is
identical to the known-good baseline, specifically to isolate this one
change's effect.

## After training finishes

Same as `full_training/`: `python3 extract_numpy_weights.py runs/tcn_exp2_aux_scale_best.pt`
— works unchanged per the verification above. Compare the resulting
`best_score` (real WP metric, same protocol as every other experiment)
against 0.6245 (the baseline's local score) before deciding whether to
submit — and remember local-score differences this small haven't
reliably predicted leaderboard differences so far, so a genuinely
positive signal here is a reason to submit and check, not a reason to
assume the leaderboard will agree outright.
