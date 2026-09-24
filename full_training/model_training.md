# Full-dataset TCN training

`train_TCN.py` trains the causal TCN on some or all of `train.parquet`'s
10,607 sequences, replacing the earlier dev-scale approach (`eda/scripts/train_tcn.py`)
that loaded a several-hundred-sequence sample fully into memory. This
document explains why that approach doesn't scale, what the new script
does differently, what each config knob controls, and what to do once
training finishes. **I have not run this script** — see "before you commit
to the full run" below for how to check it works cheaply yourself first.

## Why a new script, not just a bigger sample

`train.parquet` is ~27GB (10,607 sequences × 20,000 rows × 112 features).
The dev-scale script's `load_sequences()` reads the whole parquet file
into a pandas DataFrame and filters per sequence — fine for a 150-600
sequence sample (a few GB), completely infeasible for the full file on a
16GB-RAM machine. `train_TCN.py` instead **streams**: each mini-batch is
built by reading just `BATCH_SIZE` row groups directly from disk (each
row group is exactly one sequence, per `docs/data_overview.md`), training
on them, and discarding — at any moment only one batch's worth of data
(tens of MB) is in memory, regardless of how many total sequences you
train on.

The validation set is still loaded fully into memory once (default: 100
sequences from `eda/artifacts/valid_sample.parquet`, the same held-out
sample used throughout `eda/notes.md`) since it's small and reused every
epoch — no reason to re-read it from disk each time.

## What the warnings in your earlier run were (not related to this, but you asked)

The `RuntimeWarning: divide by zero / overflow / invalid value encountered
in matmul` warnings you saw are a **false-positive from Apple's Accelerate
BLAS backend**, not a real numerical problem. Confirmed this session:

- `numpy.show_config()` on your machine shows numpy's BLAS/LAPACK backend
  is `accelerate` (Apple's framework), not OpenBLAS.
- Reproduced the exact warning with ordinary random data of the same
  shape as the flagged operation (the TCN's first-layer downsample
  matvec) — it's not specific to your data.
- The actual result contained **zero NaN/Inf**, and matched an
  independent computation path (`np.einsum`, which doesn't route through
  Accelerate's matmul kernel) to 1e-14 precision.

This is a known Accelerate-on-Apple-Silicon quirk: certain internal
vectorized code paths for small matrix-vector products momentarily raise
CPU floating-point exception flags (from internal padding/blocking, not
an actual bad value), and numpy's warning system surfaces those flags
even though the final output is fine. **You do not need to do anything
about it** — it doesn't affect correctness (every model in this project
has been numerically verified against an independent implementation
before being trusted, precisely because of quirks like this). If the
noise bothers you, wrap the affected call in `with np.errstate(divide="ignore", over="ignore", invalid="ignore"):` — purely cosmetic.

## Config reference (top of `train_TCN.py`)

### Architecture — the two you asked to tune
| Knob | Default | What it controls |
|---|---|---|
| `N_CHANNELS` | 48 | Width of every hidden conv layer. More channels = more capacity, more compute per step, more inference latency (linearly-ish). The trained-so-far model (48ch, 7 layers) runs at 63.6µs/row, using ~40 of the 60-min budget for a 1,873-seq test set — there's headroom to go wider. |
| `DILATIONS` | `(1,2,4,8,16,32,64)` | One entry per layer, so `len(DILATIONS)` = number of layers. Receptive field = `1 + 2*sum(DILATIONS)` steps (255 for the default). Adding a layer (e.g. appending `128`) roughly doubles the receptive field and adds one layer's worth of compute; removing layers shrinks both. |
| `DROPOUT` | 0.1 | Standard regularization inside each conv block. |

Kernel size is fixed at 3 in `eda/scripts/tcn_model.py` (not exposed here,
edit that file directly if you want to change it — every downstream piece,
including `extract_numpy_weights.py` and `solution.py`, reads dilation/
channel dimensions dynamically from the saved weights, so this is the only
place kernel size lives).

### Data
| Knob | Default | Notes |
|---|---|---|
| `N_TRAIN_SEQUENCES` | `None` (= all 10,607) | Set an int to cap for a faster/smaller run. |
| `N_VALID_SEQUENCES` | 100 | Out of the 150 available in `valid_sample.parquet`. |
| `TRAIN_SEED` | 0 | Shuffle order, changes every epoch regardless (re-derived from this seed + epoch via `rng.permutation`, so epochs don't repeat the same batch order). |
| `USE_PLS` | `False` | Appends the 20-dim supervised PLS projection (`eda/scripts/pls_transform.py`) as extra input channels. `eda/notes.md` §11 found this added 51% inference latency for a 0.0013 WP gain on the 600-sequence model — not worth it there, but that was a much smaller/less-trained model; worth re-testing once you have a full-data baseline to compare against. |

### Training
| Knob | Default | Notes |
|---|---|---|
| `N_EPOCHS` | 15 | See runtime estimate below before raising this much. |
| `BATCH_SIZE` | 16 | Sequences per mini-batch. Each is a full 20,000-step sequence trained in one forward pass (not sub-sampled), so this mainly trades off gradient noise vs. memory/GPU utilization, not sequence length. |
| `LEARNING_RATE` | 2e-3 | Adam, cosine-annealed to ~0 over `N_EPOCHS`. |
| `WEIGHTED_LOSS` | `True` | MSE weighted by `\|clip(target,-2,2)\|`, mirroring the WP metric. This was the single biggest lever in every experiment so far (0.546→0.609 WP on the 600-seq model) — strongly recommend leaving it on. |
| `GRAD_CLIP` | 1.0 | Set to `None` to disable. Not used in the earlier dev-scale runs; added here as cheap insurance for a long unattended run. |
| `VALIDATE_EVERY_N_EPOCHS` | 1 | Raise if validation overhead matters (it shouldn't — it's one forward pass over 100 sequences, no backward). |

### Output / resuming
| Knob | Default | Notes |
|---|---|---|
| `RUN_NAME` | `"tcn_full"` | Checkpoints land in `full_training/runs/<RUN_NAME>_latest.pt` (every epoch) and `_best.pt` (best validation score so far). |
| `RESUME_FROM` | `None` | Set to a `_latest.pt` path to continue an interrupted run — restores model, optimizer, and LR-schedule state, so it picks up the cosine schedule where it left off rather than restarting it. |
| `LOG_EVERY_N_BATCHES` | 50 | Progress print frequency within an epoch; also written to `runs/<RUN_NAME>.log` so you can `tail -f` it. |

**Interrupt safety**: Ctrl-C during training is caught — the last
completed epoch's checkpoint is already on disk (saved unconditionally
every epoch, not just on improvement), so you lose at most one epoch's
progress. Resume with `RESUME_FROM` pointed at the `_latest.pt` file.

## Before you commit to the full run

I have not executed this script (per your request, to keep CPU free), so
please do a cheap dry run first — set `N_TRAIN_SEQUENCES = 20` and
`N_EPOCHS = 2` temporarily, run it (~1-2 minutes), confirm it prints
sensible progress and produces `runs/tcn_full_best.pt` / `_latest.pt`
without errors, then set both back and run for real. This exercises every
code path (streaming reads, training step, validation, checkpointing)
without spending hours finding a typo.

I did do what I *could* check without running the heavy path: syntax-
compiled the script, imported it as a module to confirm every name at
module scope resolves (no typos in the CONFIG block or function
references), and separately verified `extract_numpy_weights.py` end-to-end
against an existing checkpoint — round-tripped its weights through the
`.npz` format and confirmed the resulting model produces **bit-identical**
predictions (0.0 max diff over 300 steps) to a direct in-memory extraction.

## Expected runtime (estimate, not measured for the full run)

Extrapolated from this session's actual measurements on this machine
(M4, MPS) at the default architecture (48 channels, 7 layers):

- **Compute**: ~17.7ms/sequence (measured: 600 sequences trained in
  10.6s/epoch once already in memory).
- **Disk I/O**: ~39ms/sequence (measured: streaming_stats.py read train
  row groups at 25.5/sec).
- Combined (this script does I/O and compute **sequentially**, not
  overlapped — see "possible speedup" below): **~58ms/sequence**.

At the defaults (all 10,607 sequences, 15 epochs): **≈10.3 min/epoch ×
15 ≈ 2.6 hours**. Scales roughly linearly with `N_EPOCHS` and with
`N_TRAIN_SEQUENCES`; scales with `N_CHANNELS`/`len(DILATIONS)` on the
compute side only (I/O cost is fixed per sequence regardless of model
size).

**Possible future speedup, not implemented**: overlapping disk I/O with
GPU compute (e.g. a PyTorch `DataLoader` with `num_workers>0` prefetching
the next batch while the current one trains) could plausibly cut the
~40ms/sequence I/O cost to near-zero wall-clock impact, i.e. close to the
~17.7ms compute-only figure — roughly 2x faster overall. Left out here
because it's meaningfully more complex (multiprocessing, per-worker file
handles) and I couldn't test it before handing this off; the current
simple sequential version is slower but has actually been verified to
work structurally. Worth revisiting if 2.6+ hours per run becomes a
bottleneck for iterating on hyperparameters.

## After training finishes

1. Check `runs/tcn_full.log` (or stdout) for the best validation score
   and which epoch it came from.
2. Convert the checkpoint to the numpy format `solution.py` needs:
   ```
   python3 extract_numpy_weights.py runs/tcn_full_best.pt
   ```
   This writes `runs/tcn_full_best_numpy.npz`.
3. Copy that file into `wunderfund/solution/`, and point
   `solution.py`'s `_load_weights()` at the new filename (currently
   `tcn_weights_numpy.npz`) — either rename the new file to match, or
   edit the one line in `solution.py`. `standardization.npz` does not
   need to change (it's derived from `train_full_stats.json`, independent
   of any specific training run).
4. **Before trusting the new number**: re-run the same held-out
   evaluation used throughout `eda/notes.md` (50 sequences, indices
   30-79 of `valid_sample.parquet`, via `eda/scripts/eval_harness.py`) to
   get a comparable WP figure against the 0.6087 baseline documented
   there, before spending time on a full `ScorerStepByStep` run against
   the real `valid.parquet` (~40+ min, longer if you widened the model).
5. Update `wunderfund/solution/MODEL.md` with the new numbers and
   training recipe (sequence count actually used, final architecture,
   epochs) — keep that file as the source of truth for what's currently
   deployed, same convention as the fine-tuning log already in there.
