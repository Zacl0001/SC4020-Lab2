# Prospective campaign protocol

**GitHub scope:** the runner, core model, tests, README/report/protocol, executed
reporting notebook and two final MF checkpoints are included. The notebook embeds
recorded tables and learning traces and verifies selected states without fitting.
Separate historical CSVs, PNGs, logs and intermediate checkpoints remain local.
Historical comparison CSVs are optional for a fresh full reproduction. The
counts/results below refer to the already-completed campaign, not notebook reruns.

This is the retained scientific design of the completed 73-fit campaign.
Consolidation changes operational names, not the numerical rules below. The
official runner is `task2/run_biased_mf.py`; retained local evidence is in
`task2/results/biased_mf/` and selected models in `task2/saved_models/`.
Original training-source hashes remain in `provenance/original_manifest.json`;
the active manifest records the refactored source separately.

Research question: how do latent dimension and L2 affect accuracy, overfitting
and validation generalization in DAVIS and KIBA?

Fixed: the original observed-only objective, NumPy Adam, batch 2048, patience 10,
unchanged score scales and shared metrics. No final-test data. No rank above 128.

1. Capacity: F=8,16,32,64,128 at lambda=0 and .01; alpha=.005, seed=42, cap=100.
2. Overfitting: examine full raw histories, not gaps alone. Operational flag:
   final training MSE < best-epoch training MSE by >1e-6 AND final validation
   MSE > best validation MSE by >1e-6. This is descriptive, not a hypothesis test.
3. Interaction: cross all five F with lambda=0,.01,.1,1,10, alpha=.005,
   seed=42, cap=100. Reuse all matching campaign states from stage 1.
4. Rates: pick the two distinct F-lambda pairs with smallest stage-3 validation
   MSE per dataset (ties: F, lambda ascending). Try alpha=.001,.003,.005,.01.
   Every rate starts with cap=100. If it reaches cap with a new validation best
   in its final 10 epochs, rerun from the same initialization with cap=300,
   unchanged patience. Preserve both runs; compare the extension result.
   Apply this identical trigger to every rate, including .005. No >300 runs;
   label any continuing cap censoring. Fitter has no optimizer resume state, so
   extensions replay the prefix; verify exact equality of the initial history.
5. Robustness: freeze the best TWO full F-lambda-alpha configurations per dataset
   from stage 4, ties F/lambda/alpha ascending. Run seeds 42,123,2026 with the
   same 100->300 policy; reuse verified seed-42 results. Report sample std (ddof=1).
   Final hyperparameter selection: lowest mean validation MSE across all three
   seeds, ties F/lambda/alpha ascending. This is further validation selection,
   not an independent test. Always hand off seed 42, never the luckiest seed.

Stage 1 unregularized results are scientific comparisons, not an automatic rank
selection. Stage-3 joint validation results determine refinement candidates.
CI and Hit@K are descriptive secondary metrics, not selection criteria.
Underfitting flags describe higher training AND validation errors relative to
an appropriate peer; they are not inferred from a small gap alone.

Reuse: historical artifacts have no source/data hashes; most lack parameters.
Re-run historical configurations needed for this campaign, explicitly labeled.
Compare historical metrics and available parameter states as reproduction checks.
Existing lambda=100 results are historical supplementary evidence only.
Within this campaign, require matching config, source/data hashes, mappings,
mean, checkpoint fields, saved predictions and recomputed metrics before reuse.

Budget before extensions: 50 core fits (20 capacity + 30 interaction additions),
12 additional rate fits, 8 additional seed fits = 70 physical fits. One existing
DAVIS F=8, alpha=.005, lambda=.01 run is the runner smoke check and is counted
in the 20 capacity fits. Core: 14 historical repeats + 36 new configurations.
Additional extension fits are conditional and retained separately. Tests use
synthetic data and are not counted as scientific campaign fits.

The campaign made no dataset/model/shared metric changes, old-artifact
overwrites, test evaluation, training+validation refits or commits. Subsequent
authorized local consolidation archived obsolete MF working copies and promoted
the already-verified final models; it did not repeat the campaign.

## Current operational commands

Read-only verification from the repository root:

This CLI command requires the separate local aggregate CSVs/manifests. For
verification using only published artifacts, open the notebook instead:

```bash
.venv/bin/jupyter notebook task2/biased_mf.ipynb
```

The notebook loads the two published checkpoints and the shared training/validation
splits, and compares predictions/metrics with its embedded full-precision evidence.
It has no training cells and does not read test data.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m task2.run_biased_mf --stage verify
```

Deliberate reproduction uses a fresh output directory and the stages `prepare`,
`smoke`, `tests`, `capacity`, `interaction`, `rates`, `seeds`, `summarize`, `verify`:

```bash
OUT=task2/results/biased_mf_reproductions/run1
for STAGE in prepare smoke tests capacity interaction rates seeds summarize verify; do
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m task2.run_biased_mf --stage "$STAGE" --output-dir "$OUT" || break
done
```

This trains new models; it is unnecessary for loading or validating published
results. Full intermediate states belong to ignored local reproduction folders.
See `RETENTION.md` for what is shipped and how the original archive is preserved.
