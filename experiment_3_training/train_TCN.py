"""Experiment 3 — deep ensemble via independently-seeded training runs.

Builds on the known-good, leaderboard-confirmed recipe (48 channels, 7
layers/dilations to 64, weighted_mse_loss, cosine annealing from epoch 0,
15 epochs — full_training/runs/tcn_full_best.pt's exact config, 0.6414 WP
on the real leaderboard). The ONLY thing that changes between runs of
this script is SEED — everything else, byte-for-byte, matches the
baseline recipe.

WHY THIS IS DIFFERENT FROM THE LIGHTGBM ENSEMBLE THAT DIDN'T HELP
(eda/notes.md §11): that ensemble failed because LightGBM's errors
weren't *structurally* diverse from the TCN's — same underlying
information (domain+EMA features derived from the same raw inputs), just
a structurally weaker model, so there was nothing independent for
blending to exploit. Two TCNs trained from different random seeds are a
different situation: same architecture/capacity, but different weight
initialization and different mini-batch ordering mean they converge to
different local optima of the SAME loss landscape. Their errors are
genuinely more independent (not perfectly correlated), which is exactly
the condition under which averaging predictions reduces variance — this
is standard "deep ensemble" practice, not a replay of the thing that
already failed.

HOW TO USE: run this script 2-3 times, changing SEED (and RUN_NAME, so
each run's checkpoints don't overwrite each other) each time. Averaging
the resulting models' predictions at inference is a separate, later step
(see "After training finishes" in model_training.md) — this script only
produces one ensemble member per run.

HOW TO RUN (after closing other CPU/GPU-heavy processes):
    cd experiment_3_training
    python3 train_TCN.py          # trains with SEED=1 (RUN_NAME=tcn_exp3_seed1) by default
    # edit SEED=2, RUN_NAME="tcn_exp3_seed2", run again
    # edit SEED=3, RUN_NAME="tcn_exp3_seed3", run again (optional 3rd member)

Tip: dry run first — set N_TRAIN_SEQUENCES=20 and N_EPOCHS=2 temporarily.
"""
from __future__ import annotations

# ============================================================================
# CONFIG
# ============================================================================

# --- Architecture — matches the known-good baseline exactly ---
N_CHANNELS = 48
DILATIONS = (1, 2, 4, 8, 16, 32, 64)     # 7 layers, RF=255 — same as tcn_full_best.pt
DROPOUT = 0.1

# --- The one change this experiment tests: seed ---
SEED = 1                 # controls BOTH torch weight init (torch.manual_seed) and the per-epoch
                          # data-shuffle order (TRAIN_SEED below just mirrors this for clarity).
                          # Run this script once per ensemble member with SEED=1, 2, 3 (and a
                          # matching RUN_NAME each time) to build a 2-3-model deep ensemble.
TRAIN_SEED = SEED         # kept as a separate name only for consistency with the other experiments'
                          # config layout — always equals SEED here, not independently meaningful.

# --- Data ---
N_TRAIN_SEQUENCES = None      # None = all 10,607 sequences in train.parquet; or an int to cap
N_VALID_SEQUENCES = None      # None = all 1,873 sequences in the real valid.parquet; or an int to cap
USE_PLS = False

# --- Training — matches the known-good baseline exactly ---
N_EPOCHS = 15
BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
LEARNING_RATE = 2e-3
WEIGHTED_LOSS = True
GRAD_CLIP = 1.0
VALIDATE_EVERY_N_EPOCHS = 1

# --- Output / resuming ---
RUN_NAME = "tcn_exp3_seed1"    # CHANGE THIS each time you change SEED, or you'll overwrite the
                                # previous ensemble member's checkpoints
RESUME_FROM = None
LOG_EVERY_N_BATCHES = 200

# ============================================================================
# Implementation — identical to full_training/train_TCN.py's known-good
# version, plus torch.manual_seed(SEED) at the top of main().
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
from utils import BlockAccumulator  # noqa: E402
from tcn_model import CausalTCN  # noqa: E402
from train_tcn import weighted_mse_loss  # noqa: E402

if USE_PLS:
    from pls_transform import pls_scores_batch, N_PLS_COMPONENTS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COLUMNS = list(FEATURE_COLUMNS) + ["t0", "t1"]
EVAL_COLUMNS = COLUMNS + ["is_scored"]


def load_standardization():
    stats = json.loads((TABLES_DIR / "train_full_stats.json").read_text())
    means = np.array([stats["stats_need_prediction_rows"][c]["mean"] for c in FEATURE_COLUMNS])
    stds = np.array([stats["stats_need_prediction_rows"][c]["std"] for c in FEATURE_COLUMNS])
    return means, np.where(stds < 1e-8, 1.0, stds)


def n_input_channels():
    return len(FEATURE_COLUMNS) + (N_PLS_COMPONENTS if USE_PLS else 0)


def read_one_sequence(pf, group_index, means, stds):
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

            pred_np = pred.cpu().numpy().astype(np.float32)
            targets_np = yb.numpy()

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
            "weighted_loss": WEIGHTED_LOSS, "grad_clip": GRAD_CLIP, "seed": SEED,
        },
    }, path)


def main():
    torch.manual_seed(SEED)  # the one thing this experiment changes between runs

    means, stds = load_standardization()
    n_input = n_input_channels()
    print(f"device={DEVICE}  n_input_channels={n_input}  SEED={SEED}  RUN_NAME={RUN_NAME}")
    print(f"architecture: {len(DILATIONS)} layers, channels={N_CHANNELS}, dilations={DILATIONS}")

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
    print(f"validation: {len(valid_groups)} of {n_valid_available} sequences")

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

    log(f"\n=== run start {time.strftime('%Y-%m-%d %H:%M:%S')} — SEED={SEED} — "
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
                    log(f"  epoch {epoch+1}/{N_EPOCHS}  batch {b+1}  seqs {n_seen}/{len(train_groups)}  "
                        f"running_loss={total_loss/n_seen:.4f}  {rate:.2f} seq/s  ETA {eta/60:.1f} min")

            sched.step()
            train_loss = total_loss / max(n_seen, 1)
            epoch_elapsed = time.time() - epoch_t0

            if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0 or epoch == N_EPOCHS - 1:
                val_loss, wp_t0, wp_t1 = evaluate_streaming(
                    model, loss_fn, VALID_PATH, valid_groups, means, stds, VALID_BATCH_SIZE)
                score = (wp_t0 + wp_t1) / 2
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
    log(f"\nIf this is not your last ensemble member: edit SEED and RUN_NAME above, run again.")
    log("next step once you have 2-3 members: run extract_numpy_weights.py on each, then see "
        "model_training.md's 'After training finishes' for how to average them at inference.")


if __name__ == "__main__":
    main()
