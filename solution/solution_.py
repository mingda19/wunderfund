"""Alpha Connectome submission — causal TCN, pure-numpy streaming inference.

Architecture: 7-layer causal dilated Conv1d (kernel=3, dilations 1..64,
receptive field 255 steps, 48 channels), trained on the standardized raw
112 features with an MSE loss weighted by |clip(target,-2,2)|

Inference is pure numpy (no torch at runtime): each conv layer keeps a
small ring buffer of just its own required past inputs (2*dilation+1) and
does an O(1) update per step — true incremental (WaveNet-style) streaming,
not a windowed replay — verified against the batched PyTorch training
forward pass to ~1e-6 precision.

Each layer's 3 dilated-conv taps are also fused into a single matmul
(concatenate the 3 tap vectors, reshape the kernel weight accordingly —
exact same linear algebra as 3 separate matvecs, 1 BLAS call instead of
3): profiling found the per-step bottleneck was numpy/BLAS call-dispatch
overhead on many tiny matvecs, not FLOPs, buffer size, dict lookups, or
dtype (float32 vs float64 made no measurable difference — all ruled out
by direct measurement before landing on this fix). Verified exact
(not approximate) against the unfused 3-matvec form — diff ~1e-15, pure
float64 rounding noise — before adopting it.

Measured 44.9us/row on this dev machine (1 CPU thread, down from 66.9us/row
pre-fusion), projecting to ~28 min for a 1,873-sequence test set, well
inside the 60-minute budget.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KERNEL_SIZE = 3
_ARTIFACT_DIR = Path(__file__).resolve().parent


def _load_weights():
    d = np.load(_ARTIFACT_DIR / "tcn_full_best_numpy.npz")
    n_layers = int(d["n_layers"])
    layers = []
    for i in range(n_layers):
        dw = d[f"dw{i}"]
        layers.append({
            "w": d[f"w{i}"], "b": d[f"b{i}"],
            "dw": dw if dw.size > 0 else None,
            "db": d[f"db{i}"] if dw.size > 0 else None,
            "dilation": int(d[f"dilation{i}"]),
            "in_ch": d[f"w{i}"].shape[1],
        })
    return layers, d["head_w"], d["head_b"]


def _load_standardization():
    d = np.load(_ARTIFACT_DIR / "standardization.npz")
    return d["mean"], d["std"]


class StreamingCausalTCN:
    """True incremental causal-conv inference — O(1) compute per step per
    layer via a small ring buffer, not a windowed full-receptive-field
    replay. See notes.md Phase 4a for the derivation/verification.

    Each layer's 3 dilated-conv taps are fused into a single matmul: the
    kernel's 3 (out_ch, in_ch) tap-slices are concatenated into one
    (out_ch, 3*in_ch) matrix, applied to the 3 tap vectors concatenated
    into one (3*in_ch,) vector. Mathematically identical to
    w[:,:,0]@tap_old + w[:,:,1]@tap_mid + w[:,:,2]@tap_new (a block-matmul
    identity, not an approximation — verified to ~1e-15 against the
    original 3-matvec form) — just 1 BLAS call per layer instead of 3,
    which measurably matters since per-call dispatch overhead on these
    tiny matvecs, not FLOPs, was the actual bottleneck (~33% faster
    end-to-end)."""

    def __init__(self, layers, head_w, head_b):
        self.layers = layers
        self.head_w = head_w
        self.head_b = head_b
        self.ws_fused = [layer["w"].transpose(0, 2, 1).reshape(layer["w"].shape[0], -1)
                          for layer in layers]
        self.reset()

    def reset(self):
        self.buffers = []
        for layer in self.layers:
            buf_len = (KERNEL_SIZE - 1) * layer["dilation"] + 1
            self.buffers.append(np.zeros((buf_len, layer["in_ch"]), dtype=np.float64))

    def step(self, x):
        cur = x
        for i, layer in enumerate(self.layers):
            buf = self.buffers[i]
            buf[:-1] = buf[1:]
            buf[-1] = cur

            d = layer["dilation"]
            taps = np.concatenate([buf[0], buf[d], buf[-1]])
            conv_out = self.ws_fused[i] @ taps + layer["b"]
            conv_out = np.maximum(conv_out, 0.0)

            if layer["dw"] is not None:
                res = layer["dw"] @ cur + layer["db"]
            else:
                res = cur
            cur = np.maximum(conv_out + res, 0.0)

        return self.head_w @ cur + self.head_b


class PredictionModel:
    def __init__(self):
        layers, head_w, head_b = _load_weights()
        self.mean, self.std = _load_standardization()
        self.tcn = StreamingCausalTCN(layers, head_w, head_b)
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.tcn.reset()

        raw = np.asarray(data_point.state, dtype=np.float64)
        std_row = (raw - self.mean) / self.std
        pred = self.tcn.step(std_row)

        if not data_point.need_prediction:
            return None
        return pred.astype(np.float32)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
