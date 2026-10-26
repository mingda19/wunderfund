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

**Update, revised twice now — read this history, it matters**: validation
now streams too, and what it computes has changed.

1. *Originally*: a fixed 100-sequence sample (`eda/artifacts/valid_sample.parquet`)
   loaded fully into memory once, small enough to reuse every epoch.
2. *First revision*: switched to streaming all 1,873 sequences of the
   real `valid.parquet` (16.8GB as float32 features alone — over the 16GB
   budget held in memory at once, hence streaming `VALID_BATCH_SIZE` at a
   time), computing an exact single-pass **pooled, unweighted, unclipped**
   Pearson correlation across every `need_prediction` row. This is where
   it stopped being a strict improvement: the real submitted score with
   this checkpoint came back *slightly worse* than the 100-sequence-sample
   run's, not better.
3. *Root cause, and this revision*: the pooled-correlation proxy was never
   the same statistic as the real WP metric — it didn't restrict to
   `is_scored` rows, didn't clip to [-2,2], and pooled every row globally
   instead of computing per-sequence weighted correlation and averaging
   equally across sequences (the real metric's actual structure, per
   `docs/data_overview.md`). Validating against more data made the
   *proxy* estimate more precise, but a more precise estimate of the
   wrong quantity doesn't guarantee a better real score — which is
   exactly consistent with what was observed. Fixed by dropping the
   hand-rolled statistic entirely and feeding every sequence's
   predictions/targets straight into **`utils.BlockAccumulator`** — the
   competition's own scoring class, imported and used verbatim, not
   reimplemented. "Best val score" during training is now the literal WP
   metric on the validation set. Verified **bit-identical to 6 decimal
   places** against the independently-written `eda/scripts/eval_harness.py`
   path (which already used `BlockAccumulator` correctly) on the same
   checkpoint and sequences, before trusting this.

`val_loss` (separate from the score used for checkpoint selection) is
still computed over every `need_prediction` row, matching the training
loss exactly — `is_scored` is never available at real inference time, so
the training objective can't use it, and this keeps train/val loss
directly comparable for diagnosing over/underfitting.

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
| `N_VALID_SEQUENCES` | `None` | `None` = all 1,873 sequences in the real `valid.parquet`; set an int to cap. |
| `VALID_BATCH_SIZE` | 32 | Sequences per streaming-validation batch — no backward pass, so can exceed `BATCH_SIZE`. |
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
| `VALIDATE_EVERY_N_EPOCHS` | 1 | Now a real cost to weigh, unlike before: streaming all 1,873 validation sequences from disk once per epoch adds roughly 15-25 minutes total across a 15-epoch run (see runtime estimate below). Raise this if that's too much — e.g. `2` validates every other epoch, roughly halving the added cost, at the price of coarser checkpoint granularity. |

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
`evaluate_streaming`'s current (`BlockAccumulator`-based) scoring was
checked the same way, but for real: loaded an actual trained checkpoint
(`runs/tcn_full_best.pt`), ran it through `evaluate_streaming` on 5
validation sequences, then separately ran the exact same checkpoint
through the independently-written `eda/scripts/eval_harness.py` (which
also uses `BlockAccumulator`, but via its own streaming-inference code
path, not this file's batched one) on the same 5 sequences — both gave
**t0=0.671639, t1=0.704084, mean=0.687862**, matching to 6 decimal
places. That's two independently-written code paths agreeing exactly,
which is about as strong a correctness signal as is available without
a full submission.

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
15 ≈ 2.6 hours** for training alone. Scales roughly linearly with
`N_EPOCHS` and with `N_TRAIN_SEQUENCES`; scales with
`N_CHANNELS`/`len(DILATIONS)` on the compute side only (I/O cost is fixed
per sequence regardless of model size).

**Plus validation**, now that it streams all 1,873 real `valid.parquet`
sequences every epoch instead of a fixed 100-sequence in-memory sample: at
a similar ~20-30 seq/s I/O rate, that's roughly 60-90s per validation
pass × 15 epochs ≈ **15-25 extra minutes**, bringing the realistic total
to **~2.9-3.0 hours** at the defaults.

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
4. **Before trusting the new number**: note that the old "held-out 50
   sequences, indices 30-79 of `valid_sample.parquet`" check used
   throughout earlier `eda/notes.md` phases is **no longer a clean
   comparison** — those 50 sequences are a subset of the same
   `valid.parquet` this run now uses in full for checkpoint selection, so
   they've been "seen" (for model selection, not training) just like the
   rest. The training log's own "best val score" is now your fairest
   local signal (computed over all 1,873 sequences, the same protocol
   every run uses). For a truly independent check before spending a
   submission, either run the full `ScorerStepByStep` against real
   `valid.parquet` (~40+ min, longer if you widened the model — this is
   still not "blind" in the strict sense, but it's the authoritative
   scoring path) or carve a fresh held-out slice from `train.parquet`
   that no run has trained or validated on (not built here).
5. Update `wunderfund/solution/MODEL.md` with the new numbers and
   training recipe (sequence count actually used, final architecture,
   epochs) — keep that file as the source of truth for what's currently
   deployed, same convention as the fine-tuning log already in there.
