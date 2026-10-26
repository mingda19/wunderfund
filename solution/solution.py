"""Alpha Connectome submission — PLS-augmented causal TCN variant.

Same pure-numpy streaming TCN as solution.py (fused-matmul incremental
inference, see that file's docstring for the derivation/verification of
the streaming math and the matmul fusion), but this checkpoint was
trained on 132 input channels: the 112 standardized raw features plus 20
supervised PLS factor-compression components (eda/notes.md §6/§11).

solution.py's predict() never computed this projection — it only ever
fed the 112 raw-standardized features into the TCN — so it cannot run a
PLS-trained checkpoint at all (hard shape mismatch at the first layer's
fused matmul, not a silent wrong-answer bug; confirmed by trying it
directly). This file is the correct home for that checkpoint: it adds the
one extra step (a single (20,112) matvec, ~2.3us measured — negligible)
before handing off to the identical TCN code.

Measured with the current (fused) implementation on this exact
checkpoint (8 layers, dilations 1..128, 132-channel input):
  - TCN compute only: ~54.9us/row
  - PLS projection: ~2.3us/row
  - projected total for a 1,873-sequence test set: ~36 min — comfortably
    inside the 60-minute budget. (The original fear that PLS would blow
    the budget was based on a latency figure measured before the matmul
    fusion existed; re-measured with the current code before building
    this file, not assumed.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KERNEL_SIZE = 3
_ARTIFACT_DIR = Path(__file__).resolve().parent


def _load_weights():
    d = np.load(_ARTIFACT_DIR / "tcn_full_PLS_best_numpy.npz")
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


def _load_pls():
    d = np.load(_ARTIFACT_DIR / "pls_weights.npz")
    return d["x_mean"], d["rotations"]  # (112,), (112, 20)


class StreamingCausalTCN:
    """Identical to solution.py's class — see there for the fused-matmul
    derivation/verification. Duplicated here (not imported) so this file
    stays self-contained and renameable to solution.py on its own for a
    submission zip, exactly like solution.py itself doesn't depend on
    anything outside this directory."""

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
        self.pls_x_mean, self.pls_rotations = _load_pls()
        self.tcn = StreamingCausalTCN(layers, head_w, head_b)
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.tcn.reset()

        raw = np.asarray(data_point.state, dtype=np.float64)
        std_row = (raw - self.mean) / self.std
        pls_row = (std_row - self.pls_x_mean) @ self.pls_rotations  # (20,)
        full_row = np.concatenate([std_row, pls_row])  # (132,)
        pred = self.tcn.step(full_row)

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
