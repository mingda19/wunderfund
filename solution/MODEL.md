# Model: causal TCN (Phase 5 candidate)

Full experiment log lives in `eda/notes.md`; this is the summary relevant
to this specific submission package.

## Architecture

7-layer causal dilated Conv1d (kernel=3, dilations 1,2,4,8,16,32,64 →
receptive field 255 steps), 48 channels, ~64K params, head → 2 outputs
(t0, t1). Trained on the 112 raw features standardized by train-set
mean/std (computed from a full streaming pass over `train.parquet`, not
just a sample — see `eda/outputs/tables/train_full_stats.json`).

Inference: `solution.py` reimplements the trained model's forward pass in
pure numpy as true incremental (WaveNet-style) streaming — each layer
keeps a small ring buffer of just its own required past inputs
(2×dilation+1) and does an O(1) update per step, instead of replaying the
full 255-step window every row. This is **not** an approximation: it was
verified against the batched PyTorch training forward pass to ~1e-6
precision (float32 noise only) before being adopted. No torch dependency
at inference time — only numpy.

## Training recipe (final)

- Data: 600 train sequences (fresh random sample, seed=1, out of 10,607
  available — not yet the full train set, see "what's left" below).
- Loss: MSE weighted by `|clip(target, -2, 2)|` — mirrors the WP metric's
  own weighting, so the model is pushed to get large/consequential moves
  right rather than every row equally.
- 25 epochs, Adam (lr=2e-3, cosine schedule), batch size 10 sequences.
- **Checkpoint selection: best validation correlation, not final epoch.**
  Validation correlation plateaus/mildly overfits past epoch ~9-12 while
  training loss keeps falling — an earlier (unweighted-loss) run made the
  mistake of just saving the final epoch; fixed for this run.

## Fine-tuning decision log

| Change | Result | Kept? |
|---|---|---|
| 150→600 train sequences (4x) | flat-corr 0.30/0.33 → 0.33/0.35 | yes |
| Plain MSE → WP-weighted MSE loss | WP (held-out) 0.546 → 0.609 (+0.063, larger gain than the data alone would suggest — the loss reweighting targets exactly what WP rewards) | yes |
| Best-checkpoint saving instead of final-epoch | prevents the mild overfitting seen after epoch ~10 from silently costing WP | yes |
| **+20 PLS factor-compression channels** (`pls_transform.py`, supervised projection from Phase 2, 112→132 input channels) | WP 0.6087→0.6100 (+0.0013, noise-level) but latency 63.6µs→96.0µs/row (**+51%**), pushing the projected full-test-set time to **59.9 min** — essentially no headroom left in the 60-min budget | **no** — cost/benefit doesn't justify it; reverted to raw-112 input |
| LightGBM: 120→600 sequences + WP-weighted `sample_weight` | standalone WP 0.379 → 0.444 (+0.065) | yes (LightGBM improved in its own right) |
| **Ensemble TCN + LightGBM**, per-target blend weight grid-searched (0.00-1.00, step 0.05) against actual WP on 50 held-out sequences, using `run_ensemble_search.py` | best blend WP=0.6088 vs. pure-TCN WP=0.6087 — **a 0.0001 difference, i.e. noise.** Optimal weight (α_t0=0.95, α_t1=1.00) is essentially "ignore LightGBM." | **no** — LightGBM's errors aren't diversifying TCN's (both ultimately derive signal from the same raw order-book features), so blending adds ~39µs/row of pure cost for no quality gain |

**Why the ensemble didn't help, in more detail**: ensembling helps when
two models make different mistakes on different rows. Here, LightGBM's
domain+EMA features are a hand-engineered *subset/summary* of the exact
same raw signal the TCN already consumes directly (plus the TCN has real
temporal memory across the full 255-step window, LightGBM has none beyond
its EMA state). There's no independent information for LightGBM to
contribute — it's a strictly weaker view of the same data, not an
uncorrelated one. A LightGBM trained on genuinely different information
(e.g. a different feature family, or a model on a completely different
inductive bias like the `a5`/`a7` quantization-scale idea in
`eda/notes.md` §9) would be a more promising ensemble partner than the
current domain+EMA LightGBM.

## Validation results

| Set | WP | Notes |
|---|---:|---|
| 50 held-out valid sequences (seq idx 30-79, held out from all training/early-stopping) | **0.6087** | primary comparison number used throughout tuning |
| 150-sequence valid sample (full `ScorerStepByStep`, includes early-stopping sequences) | 0.6050 | end-to-end sanity check via the *actual* competition scorer, not the dev harness |
| Baseline (organizer-updated) | 0.617052 | **we are currently ~0.008-0.012 below this** |

Latency: 63.5µs/row measured → **~39.6 min projected for a 1,873-sequence
test set**, comfortably inside the 60-minute / 1-vCPU budget (run this
package's own `solution.py --validation <full valid.parquet>` for the
authoritative number on your machine).

## Known gaps / what's left before this beats the baseline

1. **Only 600 of 10,607 available train sequences used.** This is the
   most likely lever — TCN training is fast (~4.5s/epoch/150 sequences on
   MPS), full-train-set training is very plausibly affordable.
2. **The `a5`/`a7` quantization finding (`eda/notes.md` §9) is not yet
   exploited.** `t0`/`t1` are per-sequence-quantized and `a5`/`a7` predict
   the quantization step at corr 0.65/0.74 — a real, unused lever.
3. **`requirements.txt` in the starter pack does not list torch** (only
   numpy/onnxruntime/pyarrow) — irrelevant to *this* package since
   inference is pure numpy, but the *training* code (`eda/scripts/`) does
   need torch, so that's a local dev environment concern only, not a
   submission-time one.
4. Architecture/hyperparameters (48 channels, 7 layers, RF=255) were not
   swept — given the latency budget has ~20 min of headroom (39.6 min
   used of 60), a wider/deeper model is affordable and untried.

## Files in this package

- `solution.py` — `PredictionModel` class + CLI runner (matches
  `baseline/solution.py`'s `--validation` convention).
- `tcn_weights_numpy.npz` — pre-extracted per-layer conv weights (pure
  numpy, no torch needed at inference).
- `standardization.npz` — the 112-dim mean/std vectors from the full
  train-set streaming pass.
