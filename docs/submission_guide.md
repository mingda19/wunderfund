# How to make a submission

This page covers the technical requirements for your submission, including the code format, how to package your files, and the resource limits.

## What to submit

This is a code competition. You'll submit a `.zip` file containing all the code and artifacts needed to generate predictions.

The key requirements are:
- The zip file must contain a `solution.py` file at its root.
- Your `solution.py` must define a class named `PredictionModel`.
- This class must have a `predict(self, data_point)` method.


## The `PredictionModel` class

Your `solution.py` must define a no-argument `PredictionModel` class with a
`predict(self, data_point)` method. This example shows the interface;
replace the placeholder output with your model's predictions.

```python
import numpy as np
from utils import DataPoint


class PredictionModel:
    def __init__(self):
        self.seq_ix = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            # Reset your model's sequence state here.

        # Process data_point.state and update model state on every call.
        if not data_point.need_prediction:
            return None

        # Replace with your predictions for t0 and t1.
        return np.zeros(2, dtype=np.float32)
```

## Inputs, `DataPoint` and outputs

Your `predict` method will receive a `DataPoint` object on each call.
Your code should return `None` if `need_prediction` is `False`,otherwise return two finite float32-compatible values with shape `(2,)`, in `[t0, t1]` order.

The `DataPoint` object comes from the provided `utils.py` file and has the following attributes:
* `seq_ix: int`: The ID for the current sequence
* `step_in_seq: int`: The step number within the sequence: 0 to 19,999
* `need_prediction: bool`: Whether a prediction is required for this point
* `state: np.ndarray`: 112 float32 features in the order listed in
  [Data overview](data_overview.md).

> **Note:** Remember to handle the model's internal state. When you encounter a new sequence (a new `seq_ix`), you must reset the state.

Process every row in the provided order, not dropping any.


## How to package your solution
You need to package all your files into a single `.zip` archive with `solution.py` in the root of the archive.

Include your weights, helper modules and configuration files. Load them using
paths relative to `solution.py`.


```shell
submission.zip
├── solution.py
├── model.onnx
├── your_own_helpers
├   └── lib.py
└── MODEL.md
```

From a directory containing your submission files:

```bash
zip -r ../submission.zip .
```

## Evaluation environment and limits

Your code will run in an isolated Linux environment with Python 3.11 with the following constraints:

| Resource | Limit |
|---|---|
| CPU | 1 vCPU |
| RAM | 16 GB |
| Inference time | 60 minutes for the entire test set |
| Submission size | 20 MB for the uploaded ZIP |
| GPU | Not available; CPU inference only |
| Internet access | Disabled |


## How submissions are scored
### Docker
When you submit a solution, a scoring Docker container is deployed.
The intricacies of the scoring process is not important here, but some details might be useful for you.
- The container image is based on `python:3.11-slim-bookworm`
- The environment variables used by some ML libraries are configured to prevent them from attempting to access the network

Below is the part of Dockerfile. You might want to use it locally for debug purposes.

```dockerfile
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get full-upgrade -y
RUN apt-get install -y curl libgomp1 p7zip-full build-essential && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && python -m pip install --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
 && python -m pip install orbax-checkpoint \
 && python -m pip check && pip cache purge

# Keep heavy libs strictly offline at runtime 
ENV HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_DISABLED=1 MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false \
# Redirect all caches into /app
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/app/.cache/huggingface/datasets \
    NUMBA_CACHE_DIR=/app/.cache/numba

### R̴̨̋E̴̟͝S̸̪̚T̸̢͘ ̶̜̈́I̴͖͗S̷̢͗ ̴̘̂C̶̕͜E̶̋͜N̵̼̓S̴̙͠O̴̘͐R̶̼͑Ě̵͕Ḓ̵̋ ###
```

### Libs and packages
You may notice a `requirements.txt` is mentioned in Dockerfile above.
We tried to install some reasonable set of popular libs often used in ML.

> **Note:** If you need any package added to the scorer docker image — please drop us a line in Discord or email: [get help](get_help.md)


## Before uploading

- Check that `solution.py` is at the ZIP root and required files are included.
- Check output shape, warm-up handling and sequence-state resets.
- Confirm identical predictions across repeated runs.
- Check that the solution meets the execution and submission-size limits above.

The [Rules](rules.md) are also worth reading.
