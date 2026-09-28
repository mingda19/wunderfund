"""Experiment 1 — WP-aligned loss + one more TCN layer + delayed LR decay.

Builds on full_training/train_TCN.py (same streaming-from-parquet approach
— see that folder's model_training.md for why streaming is necessary at
all). Three changes here, see model_training.md in this folder for the
full reasoning behind each:

1. **`wp_aligned_loss`** (new) — replaces the old pooled `weighted_mse_loss`.
   Motivated by the `a5`/`a7` per-sequence target-quantization finding
   (`eda/notes.md` §9): sequences have very different target scales
   (~5x range observed), and the old loss pooled every row across the
   whole batch into one weighted average, so higher-scale sequences
   (bigger |target|, hence bigger weight) silently dominated the gradient
   — even though the WP *metric* itself weights every sequence equally
   and only weights *rows within* a sequence by magnitude. This loss
   mirrors the metric's actual structure instead: normalize by each
   sequence's own weight sum first, average t0/t1 per sequence, *then*
   average equally across sequences in the batch.
2. **One more layer** — `DILATIONS` extended to `(...,128)`, doubling the
   receptive field (255->511 steps) for ~1/8 more compute.
3. **Delayed LR decay** — cosine annealing now only starts at
   `LR_FLAT_EPOCHS` (default 8) instead of from epoch 0, since the
   previous full run's validation plateaued right around epoch 7-8 while
   the LR was already well into its decay — this gives the model more
   full-strength steps before the schedule starts winding down.
4. **Validation set** — uses all 1,873 sequences of the real `valid.parquet`
   (not the 150-sequence `valid_sample.parquet` used previously), streamed
   `VALID_BATCH_SIZE` at a time at the end of every epoch so the full set
   never needs to fit in memory at once (~16.8GB as float32 features alone
   — over the 16GB budget). Purely for checkpoint selection/early
   stopping — none of it is trained on, by deliberate choice.

HOW TO RUN (after closing other CPU/GPU-heavy processes):
    cd experiment_1_training
    python3 train_TCN.py

Tip: as before, do a cheap dry run first (N_TRAIN_SEQUENCES = 20) to
exercise every code path before committing to the full multi-hour run.
"""
from __future__ import annotations

# ============================================================================
# CONFIG — edit these, then run the script. See model_training.md for the
# reasoning behind each default and tuning guidance.
# ============================================================================

# --- Architecture ---
N_CHANNELS = 48                              # unchanged from full_training/
DILATIONS = (1, 2, 4, 8, 16, 32, 64)     # NOTE: this is the same 7-layer schedule as full_training/ —
                                              # if you meant to keep the "+1 layer" change from the
                                              # original plan, add 128 back in (see model_training.md)
                                              # receptive field = 1 + 2*sum(DILATIONS) = 255 steps
DROPOUT = 0.1

# --- Data ---
N_TRAIN_SEQUENCES = None      # None = all 10,607 sequences in train.parquet; or an int to cap
N_VALID_SEQUENCES = None       # None = all 1,873 sequences in the real valid.parquet; or an int to cap.
                                # Streamed batch-by-batch at the end of every epoch (see
                                # evaluate_streaming below) — never held fully in memory: 1,873 seqs
                                # x 20,000 x 112 x 4 bytes = ~16.8GB, over the 16GB budget on its own.
                                # Checkpoint selection only — never trained on.
TRAIN_SEED = 0
USE_PLS = False

# --- Training ---
N_EPOCHS = 15                    # unchanged, per the plan to isolate the loss/architecture/schedule effect
BATCH_SIZE = 16
VALID_BATCH_SIZE = 32            # sequences per streaming-validation batch; can be larger than
                                  # BATCH_SIZE since eval has no backward pass / gradient memory
LEARNING_RATE = 2e-3             # peak LR during the flat phase (unchanged from full_training/)
LOSS_TYPE = "wp_aligned"         # "wp_aligned" (new, default) | "weighted_mse" (old pooled version,
                                  # for A/B comparison) | "plain_mse"
LR_FLAT_EPOCHS = 8               # LR stays at LEARNING_RATE for epochs 0..LR_FLAT_EPOCHS-1, then
                                  # cosine-decays to ~0 over the remaining (N_EPOCHS - LR_FLAT_EPOCHS) epochs
GRAD_CLIP = 1.0
VALIDATE_EVERY_N_EPOCHS = 1

# --- Output / resuming ---
RUN_NAME = "tcn_experiment1_7"
RESUME_FROM = None                # a different architecture from full_training/'s checkpoint (8 layers,
                                   # not 7) — cannot resume from tcn_full_best.pt, only from a run of
                                   # this same script
LOG_EVERY_N_BATCHES = 50

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
    — resilient to this folder being moved to a different nesting depth."""
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

from common import (TABLES_DIR, FEATURE_COLUMNS,
                     WARMUP, SEQUENCE_LENGTH, TRAIN_PATH, VALID_PATH)  # noqa: E402
from tcn_model import CausalTCN  # noqa: E402
from train_tcn import weighted_mse_loss  # noqa: E402  (kept importable for LOSS_TYPE="weighted_mse")

if USE_PLS:
    from pls_transform import pls_scores_batch, N_PLS_COMPONENTS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "runs"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COLUMNS = list(FEATURE_COLUMNS) + ["t0", "t1"]
METRIC_CLIP = 2.0


def wp_aligned_loss(pred, target, eps=1e-6):
    """Per-sequence-normalized, WP-metric-shaped loss.

    pred, target: (batch, 2, time) — already sliced to the need_prediction
    region (WARMUP: onward) by the caller, matching how the real metric
    only scores those rows.

    For each (sequence, target) pair independently: weight rows by
    |clip(target,-2,2)| (matching the metric's row weighting within a
    sequence) and normalize by *that pair's own* weight sum (not the whole
    batch's) — this is the fix for the scale-variance problem in §9 of
    eda/notes.md: sequences with a larger quantization step have
    systematically larger |target|, hence larger weights, so a
    batch-pooled weighted average lets them dominate the gradient even
    though the WP metric treats every sequence equally. Then average t0
    and t1 per sequence (mirrors the metric's `s_b = (rho_0+rho_1)/2`),
    and average equally across sequences in the batch (mirrors the
    metric's `S = mean over b`).
    """
    w = torch.clamp(target, -METRIC_CLIP, METRIC_CLIP).abs()          # (batch, 2, time)
    num = (w * (pred - target) ** 2).sum(dim=-1)                       # (batch, 2)
    den = w.sum(dim=-1).clamp_min(eps)                                 # (batch, 2)
    per_seq_target = num / den                                        # (batch, 2)
    per_seq = per_seq_target.mean(dim=1)                               # (batch,) — average t0, t1
    return per_seq.mean()                                              # equal weight across sequences


def build_loss_fn():
    if LOSS_TYPE == "wp_aligned":
        return wp_aligned_loss
    if LOSS_TYPE == "weighted_mse":
        return weighted_mse_loss
    if LOSS_TYPE == "plain_mse":
        return nn.MSELoss()
    raise ValueError(f"unknown LOSS_TYPE={LOSS_TYPE!r}")


def build_scheduler(opt, start_epoch):
    """Flat LR for the first LR_FLAT_EPOCHS epochs, then cosine decay to
    ~0 over the remaining epochs. Always constructed fresh (last_epoch=-1)
    and overwritten via load_state_dict when resuming — SequentialLR's
    constructor-time last_epoch handling is fragile to reconstruct by
    hand, but state_dict round-trips correctly.

    NOTE: `milestones=[LR_FLAT_EPOCHS - 1]`, not `[LR_FLAT_EPOCHS]` — verified
    empirically (see experiment_1_training/model_training.md) that
    SequentialLR switches sub-schedulers one epoch later than the milestone
    value suggests, so this off-by-one correction is required to get
    exactly LR_FLAT_EPOCHS flat epochs before decay begins."""
    flat = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=LR_FLAT_EPOCHS)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(N_EPOCHS - LR_FLAT_EPOCHS, 1))
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[flat, decay], milestones=[LR_FLAT_EPOCHS - 1])


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


def iter_eval_batches(pf, group_indices, means, stds, batch_size):
    """Same idea as iter_train_batches but for evaluation: sequential (no
    shuffle needed — order doesn't affect a pooled correlation), streamed
    directly from disk, never holding more than batch_size sequences in
    memory. Used to stream the full 1,873-sequence valid.parquet at the
    end of every epoch without ever loading it all at once."""
    n_ch = n_input_channels()
    for start in range(0, len(group_indices), batch_size):
        chunk = group_indices[start:start + batch_size]
        X = np.empty((len(chunk), n_ch, SEQUENCE_LENGTH), dtype=np.float32)
        Y = np.empty((len(chunk), 2, SEQUENCE_LENGTH), dtype=np.float32)
        for i, gi in enumerate(chunk):
            X[i], Y[i] = read_one_sequence(pf, gi, means, stds)
        yield torch.from_numpy(X), torch.from_numpy(Y)


def iter_train_batches(pf, group_indices, means, stds, rng):
    """One epoch's worth of mini-batches, streamed directly from disk —
    never holds more than BATCH_SIZE sequences in memory at once."""
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
    in memory at once (e.g. all 1,873 of valid.parquet — ~16.8GB as float32
    features alone, over the 16GB budget). Reads batch_size sequences at a
    time, evaluates, discards, and moves on.

    Correlation is NOT computed by averaging per-batch correlations — that
    would be a statistically different (and worse) number than the true
    pooled correlation across every scored row in the whole set. Instead
    this accumulates the six running sums (n, sum_p, sum_y, sum_p*y,
    sum_p^2, sum_y^2) needed for a single-pass exact Pearson correlation —
    the same one-pass approach already used in
    eda/scripts/streaming_stats.py for the full-dataset EDA — and computes
    the real pooled correlation from those sums once every batch has been
    seen.
    """
    model.eval()
    pf = pq.ParquetFile(parquet_path)

    total_loss = 0.0
    n_seq_seen = 0
    n = np.zeros(2, dtype=np.float64)
    sum_p = np.zeros(2, dtype=np.float64)
    sum_y = np.zeros(2, dtype=np.float64)
    sum_py = np.zeros(2, dtype=np.float64)
    sum_p2 = np.zeros(2, dtype=np.float64)
    sum_y2 = np.zeros(2, dtype=np.float64)

    with torch.no_grad():
        for b, (xb, yb) in enumerate(iter_eval_batches(pf, group_indices, means, stds, batch_size)):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            p = pred[:, :, WARMUP:]
            y = yb[:, :, WARMUP:]

            batch_loss = loss_fn(p, y).item()
            total_loss += batch_loss * xb.shape[0]
            n_seq_seen += xb.shape[0]

            for j in range(2):
                pj = p[:, j].reshape(-1).cpu().numpy().astype(np.float64)
                yj = y[:, j].reshape(-1).cpu().numpy().astype(np.float64)
                n[j] += pj.shape[0]
                sum_p[j] += pj.sum()
                sum_y[j] += yj.sum()
                sum_py[j] += (pj * yj).sum()
                sum_p2[j] += (pj * pj).sum()
                sum_y2[j] += (yj * yj).sum()

            if (b + 1) % 20 == 0:
                print(f"  [{desc}] {n_seq_seen}/{len(group_indices)} sequences", flush=True)

    val_loss = total_loss / max(n_seq_seen, 1)
    corrs = []
    for j in range(2):
        num = n[j] * sum_py[j] - sum_p[j] * sum_y[j]
        den = np.sqrt(max((n[j] * sum_p2[j] - sum_p[j] ** 2) * (n[j] * sum_y2[j] - sum_y[j] ** 2), 0.0))
        corrs.append(float(num / den) if den > 1e-12 else 0.0)
    return val_loss, corrs[0], corrs[1]


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
            "loss_type": LOSS_TYPE, "lr_flat_epochs": LR_FLAT_EPOCHS, "grad_clip": GRAD_CLIP,
        },
    }, path)


def main():
    means, stds = load_standardization()
    n_input = n_input_channels()
    print(f"device={DEVICE}  n_input_channels={n_input}  use_pls={USE_PLS}  loss_type={LOSS_TYPE}")
    print(f"architecture: {len(DILATIONS)} layers, channels={N_CHANNELS}, "
          f"dilations={DILATIONS}")
    print(f"lr schedule: flat at {LEARNING_RATE} for epochs 1-{LR_FLAT_EPOCHS}, "
          f"then cosine decay over the remaining {N_EPOCHS - LR_FLAT_EPOCHS} epochs")

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
    loss_fn = build_loss_fn()

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

    sched = build_scheduler(opt, start_epoch)
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
        f"{N_EPOCHS - start_epoch} epochs remaining — "
        f"loss={LOSS_TYPE}, layers={len(DILATIONS)}, lr_flat_epochs={LR_FLAT_EPOCHS} ===")

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
                    cur_lr = sched.get_last_lr()[0]
                    log(f"  epoch {epoch+1}/{N_EPOCHS}  batch {b+1}  "
                        f"seqs {n_seen}/{len(train_groups)}  "
                        f"running_loss={total_loss/n_seen:.4f}  lr={cur_lr:.2e}  "
                        f"{rate:.2f} seq/s  ETA {eta/60:.1f} min")

            sched.step()
            train_loss = total_loss / max(n_seen, 1)
            epoch_elapsed = time.time() - epoch_t0

            if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0 or epoch == N_EPOCHS - 1:
                val_loss, corr0, corr1 = evaluate_streaming(
                    model, loss_fn, VALID_PATH, valid_groups, means, stds, VALID_BATCH_SIZE)
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
