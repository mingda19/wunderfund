# Quick start guide

This guide will help you get started with the **Alpha Connectome ML Competition**.


## 1. Get the starter pack

```shell 
curl -L https://files.wundernn.io/wnn_connectome_starterpack.tar.gz | \ 
tar -xz && \ 
cd competition_package 
```

or download the archive manually:

- [wnn_connectome_starterpack.zip](https://files.wundernn.io/wnn_connectome_starterpack.zip)
- [wnn_connectome_starterpack.tar.gz](https://files.wundernn.io/wnn_connectome_starterpack.tar.gz)
- _slow download?_ try [mirror-1](https://files-alternative.wundernn.io/wnn_connectome_starterpack.zip)/[mirror-2](https://storage.yandexcloud.net/files-alternative.wundernn.io/wnn_connectome_starterpack.zip)

### What's inside
```shell
wnn_connectome_starterpack/
├── datasets
├   ├── train.parquet
├   └── valid.parquet
├── baseline
├   ├── baseline.onnx
├   └── solution.py
├── README.md
└── utils.py
```


## 2. Set up your environment and packages
We strongly recommend using a virtual environment to keep your project dependencies tidy and avoid conflicts. Probably any Python 3.10+ will do. Submissions will be evaluated in a Docker container based on `python:3.11-slim-bookworm` image.

Set up the env and then install the required packages:
```bash
pip install -r requirements.txt
```

## 3. Explore the data

The dataset is provided in Parquet format.
* `datasets/train.parquet`: 10,607 sequences.
* `datasets/valid.parquet`: 1,873 sequences.

Each sequence contains 20,000 rows. Steps 0–98 are warm-up; predictions are required at steps 99–19,999. Inputs contain 112 features.

```python
import pyarrow.parquet as pq

dataset = pq.ParquetFile('datasets/train.parquet')
sequence = dataset.read_row_group(0)
print(sequence.shape)
print(sequence.column_names)
```

The details are in [Data overview](data_overview.md).

## 4. Run the example solution
We provided a baseline solution in `baseline` folder.

```bash
python baseline/solution.py --validation datasets/valid.parquet
```
This script will load the validation data and the pre-trained `baseline.onnx` model, run inference, and calculate the **Weighted Pearson** score.

> **Fun facts:**
> * This baseline is a stateful vanilla GRU.
> * It scores **0.589595 WP** on the complete validation set.
> * The ONNX Runtime session uses one CPU compute thread.

## 5. Submit
Zip your `solution.py` and model artifacts:
```bash
cd baseline
zip -r submission.zip solution.py baseline.onnx
```
Go to the [submit](https://wundernn.io/connectome/submit) page and send your solution for scoring.

> 🎉 ta-daa, you're awesome

---

## 6. Create your own model
1. Create your own `solution.py` implementing the `PredictionModel` interface. Use [Submission guide](submission_guide.md) for reference.
2. Train your model using `datasets/train.parquet`.
3. Validate using `datasets/valid.parquet`. 

The data and metric is described in details in [Data overview](data_overview.md). The starterpack's `METRIC.md` contains the full scoring reference.


Good luck with the challenge! We're excited to see what you build.
