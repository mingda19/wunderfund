"""Full-dataset TCN training — streams mini-batches directly from
train.parquet's row groups (each row group = one 20,000-step sequence)
instead of loading sequences into memory upfront, since the full file
(~10,607 sequences, ~27GB) does not fit in 16GB RAM the way the earlier
600-sequence dev runs did. See model_training.md for the full write-up:
why streaming is necessary, what each config knob does, expected runtime,
and what to do with the checkpoint afterward.

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
N_CHANNELS = 48                          # width of every hidden conv layer
DILATIONS = (1, 2, 4, 8, 16, 32, 64)     # one entry per layer -> len(DILATIONS) layers;
                                          # receptive field = 1 + 2*sum(DILATIONS) steps
                                          # (kernel size is fixed at 3 in tcn_model.py)
DROPOUT = 0.1

# --- Data ---
N_TRAIN_SEQUENCES = None      # None = all 10,607 sequences in train.parquet; or an int to cap
N_VALID_SEQUENCES = 100       # held-out sequences used for checkpoint selection (from valid_sample.parquet)
TRAIN_SEED = 0                 # controls per-epoch shuffle order
USE_PLS = False                 # append the 20-dim PLS factor-compression channels;
                                 # notes.md Phase 5 found this cost +51% latency for +0.0013 WP
                                 # on the 600-sequence model — off by default, but now that you're
                                 # training on ~18x more data it may be worth re-testing

# --- Training ---
N_EPOCHS = 15
BATCH_SIZE = 16                 # sequences per mini-batch (each is a full 20,000-step sequence)
LEARNING_RATE = 2e-3
WEIGHTED_LOSS = True             # MSE weighted by |clip(target,-2,2)| — mirrors the WP metric;
                                  # this was the single biggest lever in every experiment so far
GRAD_CLIP = 1.0                  # set to None to disable
VALIDATE_EVERY_N_EPOCHS = 1

# --- Output / resuming ---
RUN_NAME = "tcn_full"                    # checkpoints saved as <RUN_NAME>_latest.pt / _best.pt
RESUME_FROM = None                       # path to a previous "_latest.pt" to continue training, or None
LOG_EVERY_N_BATCHES = 50                 # progress print frequency within an epoch

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

EDA_SCRIPTS = Path(__file__).resolve().parents[1] / "eda" / "scripts"
sys.path.insert(0, str(EDA_SCRIPTS))

from common import (ARTIFACTS_DIR as EDA_ARTIFACTS_DIR, TABLES_DIR, FEATURE_COLUMNS,
                     WARMUP, SEQUENCE_LENGTH, TRAIN_PATH)  # noqa: E402
from tcn_model import CausalTCN  # noqa: E402
from train_tcn import weighted_mse_loss  # noqa: E402  (pure function, safe to reuse)

if USE_PLS:
    from pls_transform import pls_scores_batch, N_PLS_COMPONENTS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COLUMNS = list(FEATURE_COLUMNS) + ["t0", "t1"]


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


def load_fixed_set(parquet_path, group_indices, means, stds, desc=""):
    """Load a modest, fixed set of full sequences into memory once (used
    for validation — small enough to fit, and reused unchanged every
    epoch rather than re-read from disk each time)."""
    pf = pq.ParquetFile(parquet_path)
    n_ch = n_input_channels()
    X = np.empty((len(group_indices), n_ch, SEQUENCE_LENGTH), dtype=np.float32)
    Y = np.empty((len(group_indices), 2, SEQUENCE_LENGTH), dtype=np.float32)
    for i, gi in enumerate(group_indices):
        X[i], Y[i] = read_one_sequence(pf, gi, means, stds)
        if (i + 1) % 25 == 0:
            print(f"  [{desc}] loaded {i+1}/{len(group_indices)}", flush=True)
    return torch.from_numpy(X), torch.from_numpy(Y)


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


def evaluate(model, loss_fn, X_valid, Y_valid):
    model.eval()
    with torch.no_grad():
        pred = model(X_valid)
        val_loss = loss_fn(pred[:, :, WARMUP:], Y_valid[:, :, WARMUP:]).item()
        p0 = pred[:, 0, WARMUP:].reshape(-1).cpu().numpy()
        y0 = Y_valid[:, 0, WARMUP:].reshape(-1).cpu().numpy()
        p1 = pred[:, 1, WARMUP:].reshape(-1).cpu().numpy()
        y1 = Y_valid[:, 1, WARMUP:].reshape(-1).cpu().numpy()
        corr0 = float(np.corrcoef(p0, y0)[0, 1])
        corr1 = float(np.corrcoef(p1, y1)[0, 1])
    return val_loss, corr0, corr1


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

    valid_sample_path = EDA_ARTIFACTS_DIR / "valid_sample.parquet"
    n_valid_available = pq.ParquetFile(valid_sample_path).num_row_groups
    valid_groups = list(range(min(N_VALID_SEQUENCES, n_valid_available)))
    print(f"loading {len(valid_groups)} validation sequences (fixed set, held out from all training)...")
    X_valid, Y_valid = load_fixed_set(valid_sample_path, valid_groups, means, stds, desc="valid")
    X_valid = X_valid.to(DEVICE)
    Y_valid = Y_valid.to(DEVICE)

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
                val_loss, corr0, corr1 = evaluate(model, loss_fn, X_valid, Y_valid)
                score = (corr0 + corr1) / 2
                is_best = score > best_score
                if is_best:
                    best_score = score
                    save_checkpoint(best_path, model, opt, sched, epoch, best_score, n_input)
                log(f"epoch {epoch+1}/{N_EPOCHS}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  val_corr_t0={corr0:.4f}  val_corr_t1={corr1:.4f}  "
                    f"{'*best*' if is_best else ''}  "
                    f"(epoch {epoch_elapsed/60:.1f} min, total {((time.time()-t_start)/60):.1f} min)")

            save_checkpoint(latest_path, model, opt, sched, epoch, best_score, n_input)

    except KeyboardInterrupt:
        log("\ninterrupted — latest checkpoint is already saved after the last completed epoch "
            f"(re-run with RESUME_FROM = '{latest_path}' to continue)")
        return

    log(f"\ndone. best val score (mean corr) = {best_score:.4f}")
    log(f"best checkpoint: {best_path}")
    log(f"latest checkpoint: {latest_path}")
    log("\nnext step: run extract_numpy_weights.py to convert the checkpoint into the "
        "pure-numpy format solution.py expects (see model_training.md).")


if __name__ == "__main__":
    main()
