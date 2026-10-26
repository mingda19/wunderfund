"""Full-dataset TCN training — streams mini-batches directly from
train.parquet's row groups (each row group = one 20,000-step sequence)
instead of loading sequences into memory upfront, since the full file
(~10,607 sequences, ~27GB) does not fit in 16GB RAM the way the earlier
600-sequence dev runs did. See model_training.md for the full write-up:
why streaming is necessary, what each config knob does, expected runtime,
and what to do with the checkpoint afterward.

Validation uses all 1,873 sequences of the real valid.parquet (not a
150-sequence sample), streamed VALID_BATCH_SIZE at a time at the end of
every epoch for the same reason training streams — the full set doesn't
fit in memory either (~16.8GB as float32 features alone).

HOW TO RUN (after closing other CPU/GPU-heavy processes):
    cd full_training
    python3 train_TCN.py

Tip: before committing to the full multi-hour run, set N_TRAIN_SEQUENCES
below to something small (e.g. 20) for a two-minute dry run that exercises
every code path (data loading, training step, validation, checkpointing)
without spending hours — then set it back to None and re-run for real.
"""
from __future__ import annotations

# ============================================================================
# CONFIG — edit these, then run the script. See model_training.md for the
# reasoning behind each default and tuning guidance.
# ============================================================================

# --- Architecture (the two knobs you asked to tune: layers and channels) ---
N_CHANNELS = 64                          # width of every hidden conv layer
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)     # one entry per layer -> len(DILATIONS) layers;
                                          # receptive field = 1 + 2*sum(DILATIONS) steps
                                          # (kernel size is fixed at 3 in tcn_model.py)
DROPOUT = 0.1

# --- Data ---
N_TRAIN_SEQUENCES = None      # None = all 10,607 sequences in train.parquet; or an int to cap
N_VALID_SEQUENCES = None      # None = all 1,873 sequences in the real valid.parquet; or an int to cap.
                                # Streamed batch-by-batch at the end of every epoch (see
                                # evaluate_streaming below) — never held fully in memory. Checkpoint
                                # selection/early stopping only — never trained on. (Decided against
                                # also training on valid.parquet: it would raise the local WP estimate
                                # but risks overfitting to it with no way to detect that locally.)
TRAIN_SEED = 0                 # controls per-epoch shuffle order
USE_PLS = False                 # append the 20-dim PLS factor-compression channels;
                                 # notes.md Phase 5 found this cost +51% latency for +0.0013 WP
                                 # on the 600-sequence model — off by default, but now that you're
                                 # training on ~18x more data it may be worth re-testing

# --- Training ---
N_EPOCHS = 30
BATCH_SIZE = 16                 # sequences per mini-batch (each is a full 20,000-step sequence)
VALID_BATCH_SIZE = 32           # sequences per streaming-validation batch; can exceed BATCH_SIZE
                                 # since eval has no backward pass / gradient memory
LEARNING_RATE = 2e-3
WEIGHTED_LOSS = True             # MSE weighted by |clip(target,-2,2)| — mirrors the WP metric;
                                  # this was the single biggest lever in every experiment so far
GRAD_CLIP = 1.0                  # set to None to disable
VALIDATE_EVERY_N_EPOCHS = 1

# --- Output / resuming ---
RUN_NAME = "tcn_full_l8_64d"                    # checkpoints saved as <RUN_NAME>_latest.pt / _best.pt
RESUME_FROM = None                      # path to a previous "_latest.pt" to continue training, or None
LOG_EVERY_N_BATCHES = 200                 # progress print frequency within an epoch

# ============================================================================
# Implementation — shouldn't need to touch anything below this line.
# ============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

def _find_eda_scripts():
    """Walk up from this file's location looking for eda/scripts/common.py
    — resilient to this folder being moved to a different nesting depth
    (it was originally a sibling of wunderfund/, then moved inside it)."""
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "eda" / "scripts"
        if (candidate / "common.py").exists():
            return candidate
    raise RuntimeError(
        "could not find eda/scripts/common.py by walking up from "
        f"{Path(__file__).resolve()} — is eda/ still an ancestor-level sibling "
        "somewhere above this file? Set EDA_SCRIPTS manually if the layout changed again."
    )


EDA_SCRIPTS = _find_eda_scripts()
sys.path.insert(0, str(EDA_SCRIPTS))

from common import (TABLES_DIR, FEATURE_COLUMNS, REPO_DIR,
                     WARMUP, SEQUENCE_LENGTH, TRAIN_PATH, VALID_PATH)  # noqa: E402

sys.path.insert(0, str(REPO_DIR))
from utils import BlockAccumulator  # noqa: E402  (the competition's own scorer — reused verbatim
                                     # for validation so "best val score" during training is the
                                     # real WP metric, not an approximation of it)
from tcn_model import CausalTCN  # noqa: E402
from train_tcn import weighted_mse_loss  # noqa: E402  (pure function, safe to reuse)

if USE_PLS:
    from pls_transform import pls_scores_batch, N_PLS_COMPONENTS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COLUMNS = list(FEATURE_COLUMNS) + ["t0", "t1"]
# is_scored only exists in valid.parquet (never train.parquet, and never
# provided at real inference time either) — a separate column list for the
# validation read path so the real WP metric can be computed exactly
# (is_scored-filtered), while training reads stay identical to what the
# model actually sees at submission time.
EVAL_COLUMNS = COLUMNS + ["is_scored"]


def load_standardization():
    stats = json.loads((TABLES_DIR / "train_full_stats.json").read_text())
    means = np.array([stats["stats_need_prediction_rows"][c]["mean"] for c in FEATURE_COLUMNS])
    stds = np.array([stats["stats_need_prediction_rows"][c]["std"] for c in FEATURE_COLUMNS])
    return means, np.where(stds < 1e-8, 1.0, stds)


def n_input_channels():
    return len(FEATURE_COLUMNS) + (N_PLS_COMPONENTS if USE_PLS else 0)


def read_one_sequence(pf, group_index, means, stds):
    """One row-group -> ((n_channels, 20000) X, (2, 20000) Y), both float32."""
    table = pf.read_row_group(group_index, columns=COLUMNS)
    raw = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in FEATURE_COLUMNS]
    ).astype(np.float64)
    raw_std = (raw - means) / stds
    if USE_PLS:
        full = np.concatenate([raw_std, pls_scores_batch(raw_std)], axis=1)
    else:
        full = raw_std
    x = full.T.astype(np.float32)
    y = np.empty((2, SEQUENCE_LENGTH), dtype=np.float32)
    y[0] = table.column("t0").to_numpy(zero_copy_only=False).astype(np.float32)
    y[1] = table.column("t1").to_numpy(zero_copy_only=False).astype(np.float32)
    return x, y


def read_one_eval_sequence(pf, group_index, means, stds):
    """Like read_one_sequence, but also reads is_scored so validation can
    be restricted to exactly the rows the real metric scores — not every
    need_prediction row, which is what the training loss (necessarily)
    uses since is_scored is never available at real inference time."""
    table = pf.read_row_group(group_index, columns=EVAL_COLUMNS)
    raw = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False) for c in FEATURE_COLUMNS]
    ).astype(np.float64)
    raw_std = (raw - means) / stds
    if USE_PLS:
        full = np.concatenate([raw_std, pls_scores_batch(raw_std)], axis=1)
    else:
        full = raw_std
    x = full.T.astype(np.float32)
    y = np.empty((2, SEQUENCE_LENGTH), dtype=np.float32)
    y[0] = table.column("t0").to_numpy(zero_copy_only=False).astype(np.float32)
    y[1] = table.column("t1").to_numpy(zero_copy_only=False).astype(np.float32)
    is_scored = table.column("is_scored").to_numpy(zero_copy_only=False).astype(bool)
    return x, y, is_scored


def iter_eval_batches(pf, group_indices, means, stds, batch_size):
    """Same idea as iter_train_batches but for evaluation: sequential (no
    shuffle needed), streamed directly from disk, never holding more than
    batch_size sequences in memory. Used to stream the full 1,873-sequence
    valid.parquet at the end of every epoch without ever loading it all
    at once. Also yields the is_scored mask per sequence (see
    read_one_eval_sequence) so evaluate_streaming can score exactly the
    rows the real metric scores."""
    n_ch = n_input_channels()
    for start in range(0, len(group_indices), batch_size):
        chunk = group_indices[start:start + batch_size]
        X = np.empty((len(chunk), n_ch, SEQUENCE_LENGTH), dtype=np.float32)
        Y = np.empty((len(chunk), 2, SEQUENCE_LENGTH), dtype=np.float32)
        is_scored_batch = np.empty((len(chunk), SEQUENCE_LENGTH), dtype=bool)
        for i, gi in enumerate(chunk):
            X[i], Y[i], is_scored_batch[i] = read_one_eval_sequence(pf, gi, means, stds)
        yield torch.from_numpy(X), torch.from_numpy(Y), is_scored_batch


def iter_train_batches(pf, group_indices, means, stds, rng):
    """One epoch's worth of mini-batches, streamed directly from disk —
    never holds more than BATCH_SIZE sequences in memory at once. Shuffles
    the sequence order fresh every call (i.e. every epoch)."""
    order = rng.permutation(len(group_indices))
    shuffled = [group_indices[i] for i in order]
    n_ch = n_input_channels()
    for start in range(0, len(shuffled), BATCH_SIZE):
        chunk = shuffled[start:start + BATCH_SIZE]
        X = np.empty((len(chunk), n_ch, SEQUENCE_LENGTH), dtype=np.float32)
        Y = np.empty((len(chunk), 2, SEQUENCE_LENGTH), dtype=np.float32)
        for i, gi in enumerate(chunk):
            X[i], Y[i] = read_one_sequence(pf, gi, means, stds)
        yield torch.from_numpy(X), torch.from_numpy(Y)


def evaluate_streaming(model, loss_fn, parquet_path, group_indices, means, stds, batch_size, desc="valid"):
    """Streaming validation over a fixed set of sequences too large to hold
    in memory at once (all 1,873 of valid.parquet — ~16.8GB as float32
    features alone, over the 16GB budget). Reads batch_size sequences at a
    time, evaluates, discards, and moves on.

    The reported score is now the **real WP metric**, not an approximation
    of it: predictions/targets for every sequence are fed straight into
    `utils.BlockAccumulator` — the competition's own scoring class (clips
    to [-2,2], weights by |clipped target|, restricts to is_scored rows,
    computes weighted Pearson per sequence, averages t0/t1, then averages
    equally across sequences) — so "best val score" during training is
    now literally comparable to what the leaderboard reports, not a proxy
    that happens to correlate with it. (The previous version computed an
    unweighted, unclipped, non-is_scored-filtered pooled correlation —
    correct as *that* statistic, but not the same quantity as WP; see
    model_training.md for why that gap plausibly explains the earlier
    "more validation data, slightly worse leaderboard score" result —
    tightening an estimate of the wrong target doesn't help the real one.)

    val_loss is still computed over every need_prediction row (unweighted
    by is_scored), matching exactly what the training loss itself uses —
    is_scored is never available at real inference time, so the training
    objective can't depend on it; this keeps val_loss and train_loss
    directly comparable for diagnosing over/underfitting, while the score
    used for checkpoint selection is the metric-faithful one above.
    """
    model.eval()
    pf = pq.ParquetFile(parquet_path)
    accumulator = BlockAccumulator()

    total_loss = 0.0
    n_seq_seen = 0

    with torch.no_grad():
        for b, (xb, yb, is_scored_batch) in enumerate(
                iter_eval_batches(pf, group_indices, means, stds, batch_size)):
            xb_d = xb.to(DEVICE)
            yb_d = yb.to(DEVICE)
            pred = model(xb_d)

            batch_loss = loss_fn(pred[:, :, WARMUP:], yb_d[:, :, WARMUP:]).item()
            total_loss += batch_loss * xb.shape[0]
            n_seq_seen += xb.shape[0]

            pred_np = pred.cpu().numpy().astype(np.float32)  # (batch, 2, 20000)
            targets_np = yb.numpy()                          # (batch, 2, 20000), already float32

            for i in range(xb.shape[0]):
                accumulator.add(targets_np[i].T, pred_np[i].T, is_scored_batch[i])

            if (b + 1) % 20 == 0:
                print(f"  [{desc}] {n_seq_seen}/{len(group_indices)} sequences", flush=True)

    val_loss = total_loss / max(n_seq_seen, 1)
    result = accumulator.result()
    return val_loss, result["t0"], result["t1"]


def save_checkpoint(path, model, opt, sched, epoch, best_score, n_input):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "n_input": n_input,
        "use_pls": USE_PLS,
        "config": {
            "n_channels": N_CHANNELS, "dilations": DILATIONS, "dropout": DROPOUT,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "weighted_loss": WEIGHTED_LOSS, "grad_clip": GRAD_CLIP,
        },
    }, path)


def main():
    means, stds = load_standardization()
    n_input = n_input_channels()
    print(f"device={DEVICE}  n_input_channels={n_input}  use_pls={USE_PLS}")
    print(f"architecture: {len(DILATIONS)} layers, channels={N_CHANNELS}, "
          f"dilations={DILATIONS}")

    train_pf = pq.ParquetFile(TRAIN_PATH)
    n_train_available = train_pf.num_row_groups
    train_groups = list(range(n_train_available))
    if N_TRAIN_SEQUENCES is not None:
        train_groups = train_groups[:N_TRAIN_SEQUENCES]
    print(f"train sequences: {len(train_groups)} of {n_train_available} available")

    n_valid_available = pq.ParquetFile(VALID_PATH).num_row_groups
    valid_groups = list(range(n_valid_available))
    if N_VALID_SEQUENCES is not None:
        valid_groups = valid_groups[:N_VALID_SEQUENCES]
    print(f"validation: {len(valid_groups)} of {n_valid_available} sequences in the real valid.parquet "
          f"(streamed {VALID_BATCH_SIZE} at a time at the end of every epoch — never held fully in "
          f"memory, held out from all training)")

    model = CausalTCN(n_input=n_input, channels=N_CHANNELS, dilations=DILATIONS,
                       dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = weighted_mse_loss if WEIGHTED_LOSS else nn.MSELoss()

    start_epoch = 0
    best_score = -1.0
    if RESUME_FROM is not None:
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt["best_score"]
        print(f"resumed from {RESUME_FROM}: starting at epoch {start_epoch}, "
              f"best_score so far = {best_score:.4f}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS, last_epoch=start_epoch - 1)
    if RESUME_FROM is not None:
        sched.load_state_dict(ckpt["scheduler_state_dict"])

    latest_path = OUT_DIR / f"{RUN_NAME}_latest.pt"
    best_path = OUT_DIR / f"{RUN_NAME}_best.pt"
    log_path = OUT_DIR / f"{RUN_NAME}.log"

    rng = np.random.default_rng(TRAIN_SEED)
    t_start = time.time()

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"\n=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} — "
        f"{len(train_groups)} train seqs, {len(valid_groups)} valid seqs, "
        f"{N_EPOCHS - start_epoch} epochs remaining ===")

    try:
        for epoch in range(start_epoch, N_EPOCHS):
            model.train()
            epoch_t0 = time.time()
            total_loss = 0.0
            n_seen = 0
            for b, (xb, yb) in enumerate(iter_train_batches(train_pf, train_groups, means, stds, rng)):
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred[:, :, WARMUP:], yb[:, :, WARMUP:])
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                total_loss += loss.item() * xb.shape[0]
                n_seen += xb.shape[0]

                if (b + 1) % LOG_EVERY_N_BATCHES == 0:
                    elapsed = time.time() - epoch_t0
                    rate = n_seen / elapsed
                    eta = (len(train_groups) - n_seen) / rate if rate > 0 else float("nan")
                    log(f"  epoch {epoch+1}/{N_EPOCHS}  batch {b+1}  "
                        f"seqs {n_seen}/{len(train_groups)}  "
                        f"running_loss={total_loss/n_seen:.4f}  "
                        f"{rate:.2f} seq/s  ETA {eta/60:.1f} min")

            sched.step()
            train_loss = total_loss / max(n_seen, 1)
            epoch_elapsed = time.time() - epoch_t0

            if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0 or epoch == N_EPOCHS - 1:
                val_loss, wp_t0, wp_t1 = evaluate_streaming(
                    model, loss_fn, VALID_PATH, valid_groups, means, stds, VALID_BATCH_SIZE)
                score = (wp_t0 + wp_t1) / 2  # == BlockAccumulator's own "weighted_pearson" field
                is_best = score > best_score
                if is_best:
                    best_score = score
                    save_checkpoint(best_path, model, opt, sched, epoch, best_score, n_input)
                log(f"epoch {epoch+1}/{N_EPOCHS}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  val_wp_t0={wp_t0:.4f}  val_wp_t1={wp_t1:.4f}  "
                    f"{'*best*' if is_best else ''}  "
                    f"(epoch {epoch_elapsed/60:.1f} min, total {((time.time()-t_start)/60):.1f} min)")

            save_checkpoint(latest_path, model, opt, sched, epoch, best_score, n_input)

    except KeyboardInterrupt:
        log("\ninterrupted — latest checkpoint is already saved after the last completed epoch "
            f"(re-run with RESUME_FROM = '{latest_path}' to continue)")
        return

    log(f"\ndone. best val score (real WP metric, mean of t0/t1) = {best_score:.4f}")
    log(f"best checkpoint: {best_path}")
    log(f"latest checkpoint: {latest_path}")
    log("\nnext step: run extract_numpy_weights.py to convert the checkpoint into the "
        "pure-numpy format solution.py expects (see model_training.md).")


if __name__ == "__main__":
    main()
