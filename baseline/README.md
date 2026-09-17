# Vanilla GRU ONNX baseline

This is an inference-only stateful baseline. It scores **0.589595 WP** on the
complete validation set.

From the starterpack root, run:

```bash
python baseline/solution.py --validation datasets/valid.parquet
```

The model updates its recurrent state on every row, including warm-up rows,
and resets state between sequences. ONNX Runtime is configured for CPU-only
sequential execution with one intra-op and one inter-op thread.

`baseline_submission.zip` contains `solution.py` and `baseline.onnx` at the
archive root and is ready to upload. The baseline does not include training
code or a training checkpoint.
