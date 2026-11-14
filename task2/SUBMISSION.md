# Task 2 submission — open and Run All

`task2/task2.ipynb` is the lecturer-facing review notebook. It has one workflow:

**Prepare/check data → load saved models → review recorded experiments → calculate
validation results → select models → verify checkpoint reloads → calculate test
results → show six recommendation cases.**

There are no execution modes, training switches or environment-variable options.
The models were trained beforehand. Running the notebook recomputes evaluation
and recommendations; it does not retrain or rerun hyperparameter searches.

## Exact submission files

Copy the notebook to `task2.ipynb` at the package root, and copy
`task2/requirements-submission.txt` to `requirements.txt`. Retain the other paths:

```text
submission/
├── task2.ipynb
├── requirements.txt
├── SUBMISSION.md
├── shared_data.py
├── data/
│   ├── davis_all.csv
│   ├── kiba_all.csv
│   └── kiba_clean.csv
└── task2/
    ├── data/
    │   ├── davis_clean_task2.csv
    │   ├── davis_split_assignment.csv
    │   └── kiba_split_assignment.csv
    ├── utils/
    │   ├── global_mean.py
    │   ├── protein_cf.py
    │   ├── drug_cf.py
    │   ├── biased_mf.py
    │   └── evaluation.py
    └── saved_models/
        ├── davis_global_mean.json
        ├── davis_protein_cf.npz
        ├── davis_drug_cf.npz
        ├── davis_biased_mf.npz
        ├── kiba_global_mean.json
        ├── kiba_protein_cf.npz
        ├── kiba_drug_cf.npz
        └── kiba_biased_mf.npz
```

This is **23 files including these instructions**. No other notebook, result
folder, image, Git history or private artifact is required.

The six input CSVs are sufficient for the actual preparation checks. The raw data
supports cleaning/count verification; the validated cleaned files preserve score
serialization; the compact assignments preserve the original split rows and
affinity tokens. Canonical mappings and indexed splits are regenerated, so they
do not need to be shipped as extra copies. The exploratory analysis writes a
separate `data/davis_clean.csv`; this package uses the validated Task 2 reference
at `task2/data/davis_clean_task2.csv` instead.

## Run the notebook

Use Python 3.12 and install the pinned dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Select that Python environment in Jupyter, open `task2.ipynb`, and **Run All**.
For command-line execution from the submission root:

```bash
export PATH="$PWD/.venv/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1200 task2.ipynb
```

Tables, plots, full case rankings and checkpoint copies are written to
`task2/results/task2_submission/review/`. Re-execution refreshes those generated
outputs. The supplied checkpoints in `task2/saved_models/` are not overwritten.

## How to interpret the results

- The data, IDs and fixed 70/15/15 pair assignments are shared by all four methods.
- The KNN search contains 48 previously measured validation configurations.
- The MF study contains 73 recorded fits and 3,275 epoch rows. The notebook embeds
  these verified tables, labels them as recorded experiments and regenerates
  their figures. It does not present them as new fits. Two historical lambda=100
  supplementary results and their provenance are embedded separately; they are
  outside the 73-fit campaign and excluded from selection.
- Training/validation/test metrics and all six recommendation cases are computed
  from actual predictions during Run All.
- Model selection uses validation MSE. Those choices are kept unchanged when
  test performance is reported; test results do not select another winner.
- Prior test exposure is disclosed. The cases are the three smallest canonical
  test drug IDs per dataset with at least ten measured candidates, not cases
  chosen for favorable predictions.

## Training reproduction is separate

The repository retains the actual fitting functions and experiment runners.
They are not needed to review this package. Developers can reproduce training
from the project repository using:

```bash
# Selected models plus the KNN validation search, into a new output directory.
python -m task2.run_models --knn-search --output-dir .local_artifacts/model_refit

# Full MF campaign: follow the prepare/tests/capacity/interaction/rates/seeds/
# summarize/verify commands in task2/README.md.
```

All 13 assignment sections remain in the notebook. The report, video, agreed
contribution statement and Task 3 modeling remain separate deliverables.
