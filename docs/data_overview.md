# Data overview

Each row of the data is one moment in time for two trading instruments: `i0` and `i1`.
There are 112 input features: orderbook- and trades-related features for both instruments plus eight additional features.

The task is to predict `t0` and `t1`, two anon indicators of the future price movement of instument `i0`. Submissions are scored with **Weighted Pearson** correlation (WP), computed per sequence on a subset of rows marked `is_scored` and averaged over sequences.


## Data files
The participant dataset contains:

- `datasets/train.parquet` — training sequences with targets `t0`, `t1`.
- `datasets/valid.parquet` — validation sequences with targets and an additional scoring mask column `is_scored`.
- `datasets/valid_mask.parquet` — the very same validation scoring mask as a standalone three-column table (`seq_ix`, `step_in_seq`, `is_scored`).


| File | Sequences | Rows (= Seq × 20,000) |
|---|---:|---:|
| `train.parquet` | 10,607 | 212,140,000 |
| `valid.parquet` | 1,873 | 37,460,000 |
| `valid_mask.parquet` | 1,873 | 37,460,000 |

> **Tip — reading data:** Each Parquet row group is exactly one sequence: 20,000 rows
> with a single `seq_ix`. `ParquetFile.read_row_group(i)` therefore returns one
> complete sequence, which is the simplest way to iterate over the data without
> loading a whole file (see [Quick Start](quick_start.md)).


Test data has the same format and sequence structure as `valid.parquet`, but
is never distributed to participants: the platform feeds it to your solution's callback row by row during the evalutation.

## Sequences

### Structure
- A sequence is 20,000 consecutive rows with one `seq_ix` and
  `step_in_seq` = 0 … 19,999.
- Steps 0–98 are warm-up: `need_prediction` is false and no output is expected.
  They give your model 99 rows of context before the first prediction.
- Steps 99–19,999: `need_prediction` is true; two finite predictions are
  required on every one of these steps.
- Only some of the required steps are scored (`is_scored`; see Evaluation).


### Rules for your model
- Process rows in the given order and update your state on every row —
  including warm-up rows and rows where `is_scored` is false.
- Reset the state when `seq_ix` changes. Sequences are independent: do not
  join them or carry state across them.
- `seq_ix` values are shuffled: consecutive ids are not adjacent in time.
  Do not use `seq_ix` as a feature or to order sequences.
- A prediction may use only the current row and earlier rows of the same
  sequence.


## Row columns

| Column | train | valid | valid_mask | Meaning |
|---|:-:|:-:|:-:|---|
| `seq_ix` | ✓ | ✓ | ✓ | Sequence id. Not a feature; a larger id does not mean later in time |
| `step_in_seq` | ✓ | ✓ | ✓ | Row position in the sequence, 0 … 19,999 |
| `need_prediction` | ✓ | ✓ | × | False on steps 0–98, true on steps 99–19,999 |
| `is_scored` | × | ✓ | ✓ | True on rows used for evaluation |
| 112 feature columns | ✓ | ✓ | × | float32, in the order of the table below |
| `t0`, `t1` | ✓ | ✓ | × | float32 targets for `i0` |

Column layout of each file:

```text
train.parquet       seq_ix, step_in_seq, need_prediction,            <112 features>, t0, t1   (117 columns)
valid.parquet       seq_ix, step_in_seq, need_prediction, is_scored, <112 features>, t0, t1   (118 columns)
valid_mask.parquet  seq_ix, step_in_seq, is_scored
```
`valid_mask.parquet` carries the same mask as the `is_scored` column of
`valid.parquet`; join the two on (`seq_ix`, `step_in_seq`).

At test time your `predict` callback receives a `DataPoint` with `seq_ix`,
`step_in_seq`, `need_prediction` and `state` — the 112 features as a float32
array in the order below. It never receives `is_scored` or the targets. See [Submission guide](submission_guide.md).



## Target columns

`t0` and `t1` are two indicators of the future price movement of instument `i0`; their
exact definition and nature is not disclosed. There are no targets for instument `i1`.

- Both are float32 columns present in `train.parquet` and `valid.parquet`
  only.
- Your callback returns them as a length-2 array in `[t0, t1]` order.
- The metric clips targets and predictions to `[-2, 2]` during scoring.


## Feature columns
Features describe two instruments: `i0`, whose future price movement you
predict, and `i1`, a second instrument provided as input only (it has no
targets). 

Column names follow `i<instrument>_<group><index>` naming, where the group
is `p` (price-like), `v` (volume-like), `dp` (trade price) or `dv` (trade
volume).

In other words, the data contains **L11** order books (top-11 bid and ask prices+volumes) for 2 instruments, plus history of trades, plus 8 additional features.


The table below is the exact order of the 112 features in the files and in
`state`. Within each range indices increase numerically (`p0, p1, …, p10`).
Do not sort column names alphabetically: that would place `i0_p10` right after
`i0_p1`, before `i0_p2`. Note that bid and ask are told apart by index only
(0–10 bid, 11–21 ask).

| Group | Columns | Count | Position in `state` |
|---|---|---:|---:|
| `i0` bid price-like | `i0_p0` … `i0_p10` | 11 | 0–10 |
| `i0` ask price-like | `i0_p11` … `i0_p21` | 11 | 11–21 |
| `i0` bid volume-like | `i0_v0` … `i0_v10` | 11 | 22–32 |
| `i0` ask volume-like | `i0_v11` … `i0_v21` | 11 | 33–43 |
| `i0` trade price | `i0_dp0` … `i0_dp3` | 4 | 44–47 |
| `i0` trade volume | `i0_dv0` … `i0_dv3` | 4 | 48–51 |
| `i1` bid price-like | `i1_p0` … `i1_p10` | 11 | 52–62 |
| `i1` ask price-like | `i1_p11` … `i1_p21` | 11 | 63–73 |
| `i1` bid volume-like | `i1_v0` … `i1_v10` | 11 | 74–84 |
| `i1` ask volume-like | `i1_v11` … `i1_v21` | 11 | 85–95 |
| `i1` trade price | `i1_dp0` … `i1_dp3` | 4 | 96–99 |
| `i1` trade volume | `i1_dv0` … `i1_dv3` | 4 | 100–103 |
| Additional | `a0` … `a7` | 8 | 104–111 |

> **The index is an identifier, not a depth level.** `p0` is not necessarily
> the best bid and `p10` is not necessarily the deepest level; the same holds
> for volume-like features. An index only guarantees that the same component
> of the book representation appears in the same column on every row. Do not
> assume any ordering by price or by distance from the best quote.



## Evaluation

### Which rows are scored
Only rows with `is_scored = true` participate in the scoring; formally
`need_prediction AND is_scored` (the mask never selects warm-up rows, so the
`AND` here is a safeguard).

For validation the mask is given (`is_scored` / `valid_mask.parquet`) so that
you can reproduce the score locally. 
For test it is hidden and applied by the platform after inference. Your model therefore cannot know which rows count: return a prediction on every required row as if it were scored.

### The metric: Weighted Pearson correlation
The metric is **Weighted Pearson correlation (WP)**. Higher is better.

For each 20,000-row sequence, calculate WP separately for `t0` and `t1`,
using only rows where `need_prediction AND is_scored` is true. On those rows,
both target and prediction values are clipped to `[-2.0, 2.0]`:

$$
y_i=\operatorname{clip}(t_i,-2,2),\qquad
\hat y_i=\operatorname{clip}(p_i,-2,2),\qquad
w_i=|y_i|.
$$

For each target, the weighted correlation is

$$
\rho_w=
\frac{\sum_i w_i(y_i-\bar y_w)(\hat y_i-\bar{\hat y}_w)}
{\sqrt{\sum_i w_i(y_i-\bar y_w)^2\;\sum_i w_i(\hat y_i-\bar{\hat y}_w)^2}},
\qquad
\bar y_w=\frac{\sum_i w_i y_i}{\sum_i w_i},\quad
\bar{\hat y}_w=\frac{\sum_i w_i \hat y_i}{\sum_i w_i}.
$$

If `B` is the set of eligible sequences, the final score is

$$
s_b=\frac{\rho_{b,0,w}+\rho_{b,1,w}}{2},\qquad
S=\frac{1}{|B|}\sum_{b\in B}s_b.
$$

Each eligible sequence has equal weight in the final score. See the baseline
kit and scorer documentation (`METRIC.md`) for the full evaluation specification.
