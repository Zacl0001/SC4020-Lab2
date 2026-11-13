# Published Task 2 results

[`train_validation_test_metrics.csv`](train_validation_test_metrics.csv) contains
the verified evaluation results from [`task2.ipynb`](../task2.ipynb): two datasets,
four model families and three splits, for 24 rows.

| Column | Meaning |
|---|---|
| `dataset` | `davis` or `kiba` |
| `model` | `global_mean`, `protein_cf`, `drug_cf` or `biased_mf` |
| `split` | `train`, `val` or `test` |
| `mse` | Mean squared error on the supplied affinity scale; lower is better |
| `ci` | Global concordance index over observation pairs with unequal actual scores |
| `hit_at_1`, `hit_at_5` | Per-drug hit rate for recovering an actual-best measured candidate |
| `hit_at_5_candidates_gt_5` | Hit@5 restricted to drugs with more than five measured candidates |
| `fallback_fraction` | Fraction of predictions using the KNN mean fallback; zero for the other families |

Higher CI and hit rates are better. DAVIS and KIBA use different score scales,
so their raw MSE values are not directly comparable. Training rows are in-sample;
validation selects models and fixed settings are then evaluated on test data.
The notebook documents tie handling, candidate-set effects and prior test exposure.

## Reproduce the CSV

Follow the environment and Run All instructions in [`task2/README.md`](../README.md).
The notebook calculates metrics from the eight saved models and writes:

```text
task2/results/task2_submission/review/train_validation_test_metrics.csv
```

The published CSV is byte-identical to that generated file in the pinned
environment. The generated directory remains Git-ignored and also contains the
case summaries, full candidate rankings, experiment tables, figures and reload
checks. The published CSV is an output snapshot, not a notebook input.

The combined notebook preserves the five-stage MF experiment, all 73 fit records,
3,275 epoch rows and the two separately labelled historical lambda=100 results.
It regenerates the diagnostic figures without external result files. The model
implementation and full training workflow remain in
[`utils/biased_mf.py`](../utils/biased_mf.py) and
[`run_biased_mf.py`](../run_biased_mf.py).
