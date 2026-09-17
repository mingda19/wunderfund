# FAQ
Here are answers to some common questions. This list will be updated during the competition.

## Competition mechanics

**What is the goal of this competition?**  
Your goal is to predict `t0` and `t1`, two indicators of future price movements of a trading instrument.

**What is the evaluation metric?**  
Weighted Pearson correlation (WP), averaged across the two targets and then
equally across eligible sequences. On the supplied scoring mask, targets and
predictions are clipped to `[-2.0, 2.0]` before WP. See
[Data overview](data_overview.md) for the formula. A higher score is better.

**Can I work in a team?**  
Yes, you can participate as an individual or as part of a team.
Currently there's no way to team-up in the interface.

**How many submissions can I make per day?**  
You can make up to 5 submissions per day.

## Data questions

**What data is provided?**  
`datasets/train.parquet` and `datasets/valid.parquet`. Dataset sizes, columns
and sequence format are described in [Data overview](data_overview.md).

**Can I use external data?**  
No. All solutions must be trained using only the provided `train.parquet` and `valid.parquet` datasets.

**Can I use pre-trained models?**  
Yes, you may use open-source pre-trained models, provided they are publicly available and do not contain external market data.

**Why are the features anonymized?**  
The features are anonymized to focus the competition on the modeling task itself, rather than on domain-specific feature engineering.

## Technical questions

**When should I reset model state?**  
Reset when `seq_ix` changes. Process every row, including warm-up rows.

**What are the execution limits?**  
Your code will run in a Linux container with:
- 1 CPU core
- 16 GB of RAM
- No GPU — CPU inference only
- A **60-minute** time limit for the entire test set

**Must my solution be deterministic?**  
Yes. Repeated runs on the same input must produce identical predictions.

**Can my submission contain multiple files?**  
Yes. Include the required code and model files in one ZIP, with `solution.py`
at the root. See [Submission guide](submission_guide.md).
