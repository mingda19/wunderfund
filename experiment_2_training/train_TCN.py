"""Experiment 2 — a5/a7 quantization-scale auxiliary loss.

Builds on full_training's known-good, leaderboard-confirmed recipe
(N_CHANNELS=48, DILATIONS up to 64 / 7 layers, weighted_mse_loss, cosine
annealing from epoch 0, 15 epochs — the exact config of
full_training/runs/tcn_full_best.pt, which scored 0.6414 WP on the real
leaderboard — NOT the 8-layer/64-channel config full_training/train_TCN.py
currently has on disk, which you've already found doesn't help). Isolates
ONE change: an auxiliary loss term that trains the network to predict the
per-sequence target quantization scale, motivated by eda/notes.md §9 (the
`a5`/`a7` features correlate 0.65/0.74 with that scale) and never actually
exploited by any experiment so far — experiment_1's wp_aligned_loss
attacked the scale-variance problem structurally (reweighting the loss),
this attacks it by giving the model a direct training signal about the
scale itself.

HOW IT WORKS: `t0`/`t1` are per-sequence quantized — roughly
`k * step_size(sequence)` for small integers k, where step_size is
constant within a sequence. We compute the TRUE step_size for every
training sequence directly from its real targets (the smallest nonzero
|target| across t0/t1 in that sequence — same estimator used in the
original EDA, eda/scripts/canonical_order.py's spirit). A small auxiliary
head reads from the TCN's shared final hidden state (the same
representation the main head reads from) and predicts log(step_size).
Its loss is added to the main loss with a small weight (AUX_LOSS_WEIGHT).

This is standard auxiliary-task / multi-task regularization: the shared
TCN layers get gradient signal from both tasks, which can push them to
represent the scale-determining information (presumably derivable from
a5/a7, since that's what the EDA found correlates with it) more usefully
for the main t0/t1 prediction too — without requiring the inference-time
model to do anything new. The aux head is TRAINING-ONLY: at inference
nothing reads it, so it's stripped from the saved checkpoint's state_dict
before saving (see save_checkpoint) and the deployed model is bit-for-bit
the same CausalTCN architecture extract_numpy_weights.py and
solution.py already know how to handle — no deployment-side changes
needed.

HOW TO RUN (after closing other CPU/GPU-heavy processes):
    cd experiment_2_training
    python3 train_TCN.py

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

# --- The one change this experiment tests ---
AUX_LOSS_WEIGHT = 0.1    # weight on the auxiliary scale-prediction loss, added to the main loss.
                          # Start modest — this is a regularizer, not the primary objective; too
                          # large and it could hurt the main task. 0.05-0.2 is a reasonable range
                          # to sweep if the default doesn't show a clear effect either way.
STEP_SIZE_EPS = 1e-6      # floor for step_size estimation / log(), avoids log(0) on degenerate rows

# --- Data ---
N_TRAIN_SEQUENCES = None      # None = all 10,607 sequences in train.parquet; or an int to cap
N_VALID_SEQUENCES = None      # None = all 1,873 sequences in the real valid.parquet; or an int to cap
TRAIN_SEED = 0
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
RUN_NAME = "tcn_exp2_aux_scale"
RESUME_FROM = None
LOG_EVERY_N_BATCHES = 200

# ============================================================================
# Implementation
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
from tcn_model import CausalConvBlock  # noqa: E402  (reused for the aux-head model below)
from train_tcn import weighted_mse_loss  # noqa: E402

if USE_PLS:
    from pls_transform import pls_scores_batch, N_PLS_COMPONENTS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COLUMNS = list(FEATURE_COLUMNS) + ["t0", "t1"]
EVAL_COLUMNS = COLUMNS + ["is_scored"]


class CausalTCNWithAux(nn.Module):
    """Same backbone as tcn_model.CausalTCN (reuses CausalConvBlock
    verbatim) plus a small auxiliary head reading the same final hidden
    state as the main head. Parameter names/shapes for `blocks` and
    `head` are identical to plain CausalTCN, so a checkpoint saved
    *without* the aux_head keys (see save_checkpoint) loads into a plain
    CausalTCN via extract_numpy_weights.py with no changes needed there."""

    def __init__(self, n_input, channels, dilations, n_output=2, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = n_input
        for d in dilations:
            layers.append(CausalConvBlock(in_ch, channels, d, dropout=dropout))
            in_ch = channels
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Conv1d(channels, n_output, 1)
        self.aux_head = nn.Conv1d(channels, 1, 1)  # predicts log(step_size) per row

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return self.head(h), self.aux_head(h)


def compute_log_step_size(y, eps=STEP_SIZE_EPS):
    """y: (batch, 2, time) true targets (already WARMUP-sliced by caller).
    Returns (batch,) log(step_size) — the smallest nonzero |t0 or t1|
    value in that sequence, the same per-sequence quantization-step
    estimator the original EDA used. Sequences with no nonzero targets in
    the sliced region (shouldn't happen with real data, but guarded)
    fall back to eps."""
    abs_y = y.abs()
    nonzero = abs_y > eps
    masked = torch.where(nonzero, abs_y, torch.full_like(abs_y, float("inf")))
    step = masked.reshape(y.shape[0], -1).amin(dim=1)
    step = torch.where(torch.isfinite(step), step, torch.full_like(step, eps))
    step = torch.clamp(step, min=eps)
    return torch.log(step)


def combined_loss(pred, aux_pred, target):
    """pred, aux_pred, target already WARMUP-sliced. Main loss unchanged
    (weighted_mse_loss, matching the known-good baseline exactly);
    auxiliary loss is plain MSE between the per-row aux prediction
    (averaged over the sequence, since the true scale is constant per
    sequence) and the true log-step-size."""
    main = weighted_mse_loss(pred, target)
    true_log_step = compute_log_step_size(target)          # (batch,)
    pred_log_step = aux_pred.squeeze(1).mean(dim=1)         # (batch,) — average over time
    aux = nn.functional.mse_loss(pred_log_step, true_log_step)
    return main + AUX_LOSS_WEIGHT * aux, main.item(), aux.item()


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


def evaluate_streaming(model, parquet_path, group_indices, means, stds, batch_size, desc="valid"):
    """Scoring uses only the main head's output (matching real inference,
    which never sees the aux head) — the real WP metric via
    utils.BlockAccumulator, same as full_training/train_TCN.py."""
    model.eval()
    pf = pq.ParquetFile(parquet_path)
    accumulator = BlockAccumulator()

    total_main_loss = 0.0
    total_aux_loss = 0.0
    n_seq_seen = 0

    with torch.no_grad():
        for b, (xb, yb, is_scored_batch) in enumerate(
                iter_eval_batches(pf, group_indices, means, stds, batch_size)):
            xb_d = xb.to(DEVICE)
            yb_d = yb.to(DEVICE)
            pred, aux_pred = model(xb_d)

            _, main_l, aux_l = combined_loss(
                pred[:, :, WARMUP:], aux_pred[:, :, WARMUP:], yb_d[:, :, WARMUP:])
            total_main_loss += main_l * xb.shape[0]
            total_aux_loss += aux_l * xb.shape[0]
            n_seq_seen += xb.shape[0]

            pred_np = pred.cpu().numpy().astype(np.float32)
            targets_np = yb.numpy()

            for i in range(xb.shape[0]):
                accumulator.add(targets_np[i].T, pred_np[i].T, is_scored_batch[i])

            if (b + 1) % 20 == 0:
                print(f"  [{desc}] {n_seq_seen}/{len(group_indices)} sequences", flush=True)

    val_main_loss = total_main_loss / max(n_seq_seen, 1)
    val_aux_loss = total_aux_loss / max(n_seq_seen, 1)
    result = accumulator.result()
    return val_main_loss, val_aux_loss, result["t0"], result["t1"]


def save_checkpoint(path, model, opt, sched, epoch, best_score, n_input):
    full_state = model.state_dict()
    # strip aux_head.* so the checkpoint loads into a plain CausalTCN via
    # the existing extract_numpy_weights.py / solution.py with zero changes
    deployable_state = {k: v for k, v in full_state.items() if not k.startswith("aux_head.")}
    torch.save({
        "state_dict": deployable_state,
        "full_state_dict_with_aux": full_state,  # kept for RESUME_FROM only
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
            "aux_loss_weight": AUX_LOSS_WEIGHT,
        },
    }, path)


def main():
    means, stds = load_standardization()
    n_input = n_input_channels()
    print(f"device={DEVICE}  n_input_channels={n_input}  aux_loss_weight={AUX_LOSS_WEIGHT}")
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

    model = CausalTCNWithAux(n_input=n_input, channels=N_CHANNELS, dilations=DILATIONS,
                              dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_score = -1.0
    if RESUME_FROM is not None:
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["full_state_dict_with_aux"])
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
        f"{N_EPOCHS - start_epoch} epochs remaining — aux_loss_weight={AUX_LOSS_WEIGHT} ===")

    try:
        for epoch in range(start_epoch, N_EPOCHS):
            model.train()
            epoch_t0 = time.time()
            total_main, total_aux = 0.0, 0.0
            n_seen = 0
            for b, (xb, yb) in enumerate(iter_train_batches(train_pf, train_groups, means, stds, rng)):
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                opt.zero_grad()
                pred, aux_pred = model(xb)
                loss, main_l, aux_l = combined_loss(
                    pred[:, :, WARMUP:], aux_pred[:, :, WARMUP:], yb[:, :, WARMUP:])
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                total_main += main_l * xb.shape[0]
                total_aux += aux_l * xb.shape[0]
                n_seen += xb.shape[0]

                if (b + 1) % LOG_EVERY_N_BATCHES == 0:
                    elapsed = time.time() - epoch_t0
                    rate = n_seen / elapsed
                    eta = (len(train_groups) - n_seen) / rate if rate > 0 else float("nan")
                    log(f"  epoch {epoch+1}/{N_EPOCHS}  batch {b+1}  seqs {n_seen}/{len(train_groups)}  "
                        f"main_loss={total_main/n_seen:.4f}  aux_loss={total_aux/n_seen:.4f}  "
                        f"{rate:.2f} seq/s  ETA {eta/60:.1f} min")

            sched.step()
            train_main_loss = total_main / max(n_seen, 1)
            train_aux_loss = total_aux / max(n_seen, 1)
            epoch_elapsed = time.time() - epoch_t0

            if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0 or epoch == N_EPOCHS - 1:
                val_main, val_aux, wp_t0, wp_t1 = evaluate_streaming(
                    model, VALID_PATH, valid_groups, means, stds, VALID_BATCH_SIZE)
                score = (wp_t0 + wp_t1) / 2
                is_best = score > best_score
                if is_best:
                    best_score = score
                    save_checkpoint(best_path, model, opt, sched, epoch, best_score, n_input)
                log(f"epoch {epoch+1}/{N_EPOCHS}  train_main={train_main_loss:.4f}  "
                    f"train_aux={train_aux_loss:.4f}  val_main={val_main:.4f}  val_aux={val_aux:.4f}  "
                    f"val_wp_t0={wp_t0:.4f}  val_wp_t1={wp_t1:.4f}  {'*best*' if is_best else ''}  "
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
        "pure-numpy format solution.py expects — this works unchanged, the aux_head was "
        "stripped from state_dict before saving (see save_checkpoint).")


if __name__ == "__main__":
    main()
