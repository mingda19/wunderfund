"""Convert a train_TCN.py checkpoint into the pure-numpy weight format
solution.py expects (no torch dependency at inference time — see
wunderfund/solution/solution.py and eda/scripts/tcn_streaming.py for why:
true incremental streaming inference is reimplemented in plain numpy).

Usage:
    python3 extract_numpy_weights.py runs/tcn_full_best.pt

Writes <checkpoint_stem>_numpy.npz next to the checkpoint. Copy that file
(and a standardization.npz — same one already in wunderfund/solution/
works unchanged, since standardization is computed from train_full_stats.json,
independent of any particular training run) into wunderfund/solution/ and
update solution.py's filename if you rename it, to deploy a newly trained
model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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

from tcn_model import CausalTCN  # noqa: E402
from tcn_streaming import _extract_weights  # noqa: E402  (already verified against the batched forward pass — see notes.md Phase 4a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to a train_TCN.py checkpoint (_best.pt or _latest.pt)")
    ap.add_argument("--out", default=None, help="output .npz path (default: <checkpoint_stem>_numpy.npz)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        n_input = ckpt["n_input"]
        config = ckpt.get("config", {})
        n_channels = config.get("n_channels", 48)
        dilations = tuple(config.get("dilations", (1, 2, 4, 8, 16, 32, 64)))
        print(f"checkpoint metadata: n_input={n_input}  n_channels={n_channels}  "
              f"dilations={dilations}  use_pls={ckpt.get('use_pls')}  "
              f"epoch={ckpt.get('epoch')}  best_score={ckpt.get('best_score')}")
    else:
        raise ValueError(
            "this looks like a raw state_dict, not a train_TCN.py checkpoint — "
            "pass --n-input/--n-channels/--dilations manually or adapt this script")

    model = CausalTCN(n_input=n_input, channels=n_channels, dilations=dilations)
    model.load_state_dict(state_dict)
    model.eval()

    layers, head_w, head_b = _extract_weights(model)

    save_dict = {"head_w": head_w, "head_b": head_b, "n_layers": len(layers)}
    for i, layer in enumerate(layers):
        save_dict[f"w{i}"] = layer["w"]
        save_dict[f"b{i}"] = layer["b"]
        save_dict[f"dilation{i}"] = layer["dilation"]
        if layer["dw"] is not None:
            save_dict[f"dw{i}"] = layer["dw"]
            save_dict[f"db{i}"] = layer["db"]
        else:
            save_dict[f"dw{i}"] = np.array([])
            save_dict[f"db{i}"] = np.array([])

    out_path = Path(args.out) if args.out else ckpt_path.with_name(ckpt_path.stem + "_numpy.npz")
    np.savez(out_path, **save_dict)
    print(f"wrote {out_path}  ({len(layers)} layers, channels={n_channels})")
    print("\nnext: copy this file (and standardization.npz, unchanged) into "
          "wunderfund/solution/, point solution.py's _load_weights() at the "
          "new filename, and re-run the eval_harness / ScorerStepByStep check "
          "before trusting the new numbers.")


if __name__ == "__main__":
    main()
