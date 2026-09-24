"""Alpha Connectome submission — causal TCN, pure-numpy streaming inference.

Architecture: 7-layer causal dilated Conv1d (kernel=3, dilations 1..64,
receptive field 255 steps, 48 channels), trained on the standardized raw
112 features with an MSE loss weighted by |clip(target,-2,2)| (mirrors the
WP scoring metric's own weighting). See MODEL.md for the full write-up:
training data/recipe, why PLS factor-compression and a LightGBM ensemble
were tried and NOT adopted, and validation numbers.

Inference is pure numpy (no torch at runtime): each conv layer keeps a
small ring buffer of just its own required past inputs (2*dilation+1) and
does an O(1) update per step — true incremental (WaveNet-style) streaming,
not a windowed replay — verified against the batched PyTorch training
forward pass to ~1e-6 precision. Measured 63.6us/row on this dev machine
(1 CPU thread), projecting to ~40 min for a 1873-sequence test set, well
inside the 60-minute budget.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KERNEL_SIZE = 3
_ARTIFACT_DIR = Path(__file__).resolve().parent


def _load_weights():
    d = np.load(_ARTIFACT_DIR / "tcn_weights_numpy.npz")
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
    replay. See notes.md Phase 4a for the derivation/verification."""

    def __init__(self, layers, head_w, head_b):
        self.layers = layers
        self.head_w = head_w
        self.head_b = head_b
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
            tap_old, tap_mid, tap_new = buf[0], buf[d], buf[-1]
            w = layer["w"]
            conv_out = w[:, :, 0] @ tap_old + w[:, :, 1] @ tap_mid + w[:, :, 2] @ tap_new + layer["b"]
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
