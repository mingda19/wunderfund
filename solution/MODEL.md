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

## Weight storage format — not ONNX

Chain from training to deployment:

1. **During training** (`full_training/train_TCN.py`): checkpoints are
   plain PyTorch (`.pt`, via `torch.save`) containing the model
   `state_dict` plus optimizer/scheduler state (for resuming) and the
   architecture config — `runs/<run_name>_best.pt` / `_latest.pt`.
2. **Post-training conversion** (`full_training/extract_numpy_weights.py`):
   loads that checkpoint into a `CausalTCN`, reads out each conv layer's
   effective weight/bias/downsample tensors (PyTorch's `weight_norm`
   parametrization computes these on access — no manual g/v math needed),
   and writes them as **plain numpy arrays in a `.npz`** file.
3. **What's actually deployed**: `tcn_weights_numpy.npz` in this folder —
   a numpy archive, not a `.pt` and not `.onnx`.

**We deliberately did not export to ONNX**, unlike the baseline GRU. The
reason is architectural, not a rejection of ONNX in general: the baseline's
recurrent state is a single hidden vector, which maps directly onto ONNX's
"pass state in, get state out" per-call pattern
(`hidden_0`/`hidden_1` in `baseline/solution.py`). Our TCN's fast
inference instead depends on **7 small per-layer ring buffers** of
different sizes (the WaveNet-style trick above) — a `torch.onnx.export`
of `CausalTCN.forward()` as currently written would export the *naive*
full-window convolution (recompute over the whole 255-step receptive
field every row), because the incremental buffer logic lives in separate
hand-written code (`tcn_streaming.py`), not in the model's `forward()`.
That's exactly the slow path we moved away from (1141µs/row, ~12x over
budget — see `eda/notes.md` §8).

Getting ONNX to run the *fast* version would mean restructuring
`CausalTCN.forward()` to take the 7 ring buffers as explicit extra
inputs and return updated ones as extra outputs (the same pattern as the
baseline's hidden state, just 7 buffers instead of 2), then exporting
that. It's a legitimate option — `onnxruntime`'s C++ execution is likely
faster than our numpy buffer-shuffling, which would help if a future,
wider/deeper model eats into the latency margin — but it's real
re-engineering, not a quick swap, and the current numpy path is already
verified correct and within budget. Worth revisiting if you widen the
model enough to need the extra speed; not necessary right now.

## Dependencies — what the scoring container needs

**Just `numpy`.** `solution.py`'s only imports outside the local-testing
`__main__` block are `pathlib` (stdlib) and `numpy`. That's already in
the organizers' `requirements.txt`
(`numpy==2.2.6`, alongside `onnxruntime==1.23.2`, `pyarrow==19.0.1` — we
use neither of those two, they're just what the baseline needs). **No
email to the organizers needed for this package as it stands.**

This was a deliberate benefit of the numpy-streaming design, not an
accident: an equivalent torch-based submission would need to add `torch`
to what's installed, and while the Dockerfile's
`--extra-index-url https://download.pytorch.org/whl/cpu` strongly implies
torch is expected to be installable, it is *not* currently listed in the
starter pack's `requirements.txt` — that would need to be confirmed/
requested before relying on it (see `docs/submission_guide.md`'s "drop us
a line" note). We simply never needed to find out.

If a future change adds a dependency, here's what each would need:
- **Switching to ONNX inference** (see above): none beyond
  `onnxruntime`, already installed.
- **Re-adding the LightGBM ensemble** (currently not adopted, see below):
  would need `lightgbm`, not currently in `requirements.txt` —
  would require asking the organizers.
- **PLS factor-compression channels** (`USE_PLS`, currently off): no new
  dependency — it's a plain numpy matrix projection either way.

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
3. Architecture/hyperparameters (48 channels, 7 layers, RF=255) were not
   swept — given the latency budget has ~20 min of headroom (39.6 min
   used of 60), a wider/deeper model is affordable and untried.

## Files in this package

- `solution.py` — `PredictionModel` class + CLI runner (matches
  `baseline/solution.py`'s `--validation` convention).
- `tcn_weights_numpy.npz` — pre-extracted per-layer conv weights (pure
  numpy, no torch needed at inference).
- `standardization.npz` — the 112-dim mean/std vectors from the full
  train-set streaming pass.
- `MODEL.md` — this file (not required by `docs/submission_guide.md`, but
  shown in its example tree and directly useful for the technical-report
  requirement in `docs/prizes.md` if this solution places).

## Packaging checklist (`docs/submission_guide.md`)

Current folder already satisfies the stated requirements: `solution.py`
at what would be the zip root, defines `PredictionModel` with
`predict(self, data_point)`, all weight files load via paths relative to
`solution.py` (`Path(__file__).resolve().parent`, not absolute paths —
checked in `_load_weights()`/`_load_standardization()`). Total size ~524KB,
well under the 20MB limit.

Two things to handle when you zip:
- **Exclude `__pycache__/`** if present (created by local testing runs;
  harmless either way, just unnecessary weight in the zip).
- `solution.py`'s `if __name__ == "__main__":` block does
  `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` to reach
  `utils.py` for local `--validation` testing — that only works when this
  folder sits directly inside the starter-pack repo (where `utils.py`
  lives one level up), as it does now. It's irrelevant to actual scoring
  (the harness imports `PredictionModel` directly, never runs this file's
  `__main__` block) — only matters if you want to keep running local
  validation checks from wherever you zip from.
