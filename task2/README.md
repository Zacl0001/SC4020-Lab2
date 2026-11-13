# Task 2 — Drug–Target Affinity Prediction

**Open [task2.ipynb](task2.ipynb) and Run All.** This is the current complete Task 2
submission: one checkpoint-review workflow, with no training modes or switches.

The notebook prepares/checks canonical-ID data, loads the eight saved models,
presents the completed hyperparameter experiments, and recomputes validation,
test and recommendation results. It includes all 13 assignment sections and
three fixed test-ligand cases for each dataset.

Jiaxin implemented Global Mean, Protein-Based KNN and Drug-Based KNN;
WeiW0917 implemented Biased Matrix Factorization. The earlier baseline notebooks
remain as working records. The MF experiments and their historical supplementary
evidence are consolidated in `task2.ipynb`, the current entry point for all four
models and their canonical-ID checkpoints.

## Run from the repository root

Use Python 3.12 and the dependencies already provided on `main`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1200 \
  task2/task2.ipynb
```

Alternatively, select that environment in Jupyter and Run All interactively.
Generated figures, tables, full case rankings and checkpoint copies go to
`task2/results/task2_submission/review/`, which is ignored by Git. The production
checkpoints in `task2/saved_models/` are not overwritten.

The verified [final metrics CSV](results/train_validation_test_metrics.csv) is
published separately for direct access on GitHub. It contains 24 rows covering
both datasets, all four models and the training/validation/test splits. Run All
regenerates the same CSV under the ignored review directory. See the
[results guide](results/README.md) for columns and provenance.

For the independent 23-file lecturer package, see [SUBMISSION.md](SUBMISSION.md).
`requirements-submission.txt` contains its smaller dependency list. No old
notebook, private artifact or external results folder is required.

## Data and model definitions

- `shared_data.py` validates raw cleaning and reconstructs deterministic canonical
  mappings from each complete cleaned dataset before applying the fixed splits.
- Existing `data/splits/{dataset}_{train,val,test}.csv` filenames are retained.
  Their columns include numeric IDs, original identifiers and measured affinity.
- DAVIS uses supplied pKd without another logarithmic conversion. Missing KIBA
  interactions remain unobserved; genuine zero scores remain valid observations.
- Model parameters use training observations only. Validation selects settings
  and the model family before any test metrics are calculated.
- The selected configurations are Protein KNN K=10/5 and Drug KNN K=5/10 for
  DAVIS/KIBA, with shrinkage10 and minimum overlap2. MF uses F128, lambda1,
  learning rate .001, batch2048, seed42 and patience10.

The notebook contains full-precision records for the 48-setting KNN search and
73-fit MF campaign, including all 3,275 MF epoch rows. These are clearly labelled
as recorded experiments. The figures are regenerated from those records; current
model comparison and case metrics are computed from checkpoint predictions.
Section 7 also preserves the regularization/rate learning curves, extension
prefix checks and two historical lambda=100 results. Those two supplementary
results are outside the 73-fit campaign and do not participate in model selection.
Prior test exposure is disclosed, and test results do not change model selection.

## Verification

```bash
python -m unittest discover -s task2/tests -p 'test*.py' -v
```

This branch includes the 18 MF numerical/model/integration tests, covering
gradients, Adam updates, stopping, validation-only selection and checkpoint
reloads. The notebook itself checks data/split integrity and all eight model
reloads. [submission_verification.json](submission_verification.json) records
the independently verified submission and the checks performed for this branch.

The main notebook has been independently executed without other notebooks,
private references or intermediate experiment runs. Its current validation-MSE
selection chooses MF for both datasets, while showing the other methods' actual
results and ranking-metric trade-offs.

## Training reproduction — separate from notebook review

To refit all four model families and reproduce the KNN validation grid:

```bash
python -m task2.run_models --knn-search --output-dir .local_artifacts/model_refit
```

The output directory must be new. This uses the existing model fitters, saves
eight new states and checks reload equality. It reads training/validation only.

To rerun the full MF campaign independently:

```bash
OUT=.local_artifacts/mf_reproduction
python -m task2.run_biased_mf --stage prepare --output-dir "$OUT"
python -m task2.run_biased_mf --stage tests --output-dir "$OUT"
for STAGE in capacity interaction rates seeds summarize verify; do
  python -m task2.run_biased_mf --stage "$STAGE" --output-dir "$OUT" || break
done
```

These commands generate their own evidence in the new output directory. No
separate historical CSV bundle is needed; the recorded study is also embedded
in the main notebook for convenient review. The full campaign is not part of
ordinary Run All execution.
