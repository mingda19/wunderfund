# Experiment 5: skip/dense connections from every layer to the head

Builds on the known-good, leaderboard-confirmed recipe exactly (48
channels, 7 layers/dilations to 64, `weighted_mse_loss`, cosine annealing
from epoch 0, 15 epochs — `full_training/runs/tcn_full_best.pt`'s config,
0.6414 WP). Isolates ONE change: the head reads the **sum of every
layer's output**, not just the last layer's — the classic WaveNet/TCN
skip-connection pattern.

## Why this, given more layers/dilation already didn't help

In plain `CausalTCN`, the head only ever sees the *last* layer's
representation — everything earlier layers computed has to survive being
transformed by every subsequent layer to reach the output. This is a
different mechanism than "add another layer": it doesn't add capacity, it
changes how the *existing* capacity's information reaches the prediction.
Depth/dilation scaling plateauing doesn't predict anything about whether
this helps, since it's not more of the same lever.

## *** Read before extracting a checkpoint from this experiment ***

`CausalTCNSkip` has **exactly the same parameter names and shapes** as
`tcn_model.CausalTCN` (same `blocks`, same `head` — only the `forward()`
method differs). I verified this concretely, not just by reasoning about
it: constructed a `CausalTCNSkip`, loaded its `state_dict` straight into
a plain `CausalTCN` — **it loaded with no error** — then compared outputs
on the same input: **max diff 9.5**, nowhere near numerical noise. A
checkpoint from this experiment run through the existing
`extract_numpy_weights.py` / `solution.py` pipeline unmodified will
silently deploy a model that computes something different from what was
trained and validated, with no crash to warn you.

`save_checkpoint` tags the checkpoint with `"architecture": "CausalTCNSkip"`
in its top-level metadata specifically so this can't be missed later —
check for that key before extracting.

**What actually needs to happen before this is deployable** (not built
yet — there's no trained checkpoint to build/verify it against, and
building unverifiable code isn't worth it):
1. A modified `extract_numpy_weights.py` path (or a flag) that still
   extracts the same per-layer `w`/`b`/`dw`/`db` arrays (those are
   identical in shape/meaning to the plain architecture — only the
   *combination* at the head differs), but the extraction itself doesn't
   need to change, since it just dumps raw layer weights regardless of
   how `forward()` uses them.
2. A modified `StreamingCausalTCN.step()` for whichever `solution.py`
   variant deploys this — instead of returning `self.head_w @ cur + self.head_b`
   using only the last layer's `cur`, it needs to accumulate a running
   sum of every layer's output across the step (same O(1)-per-layer cost,
   just one extra running-sum variable) and feed that sum to the head.
   This is a small, mechanical change, but it's new code that needs the
   same numerical verification (against the batched PyTorch forward
   pass) every other streaming-inference change in this project has had
   — ask once you have a real checkpoint and I'll build and verify it
   properly, the same way `solution_pls.py` was built and verified
   against a real trained checkpoint rather than assumed correct.

## What was verified before building this

- `CausalTCNSkip` constructs and runs a forward pass with the correct
  output shape.
- The deployment trap above — confirmed empirically (loads silently,
  produces a substantively different output), not just described.

Not run: the actual multi-hour training. Dry run first
(`N_TRAIN_SEQUENCES=20`, `N_EPOCHS=2`) before committing to the full run.

## Config reference

No new config knobs beyond the baseline's — this experiment's only
variable is the architecture class itself (`CausalTCNSkip` vs
`CausalTCN`), not a tunable parameter.

## After training finishes

**Do not** run `extract_numpy_weights.py` on this checkpoint directly —
see the warning above. Check `best_score` against 0.6245 locally first;
if it looks promising, come back and ask for the deployment-side pieces
(items 1-2 above) to be built and verified before trying to submit it.
