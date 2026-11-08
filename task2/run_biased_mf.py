"""Official Biased MF campaign and read-only compact-evidence verifier.

With local experiment artifacts available, verify without fitting or writing:
    python -m task2.run_biased_mf --stage verify

To deliberately reproduce experiments, use a NEW output directory and stages:
    prepare, smoke, tests, capacity, interaction, rates, seeds, summarize, verify
For example: --output-dir task2/results/biased_mf_reproductions/run1

The mathematical model and shared metrics are imported unchanged. Each fresh
physical fit has its own immutable directory. GitHub includes the final models
and a self-contained reporting notebook; separate historical/result CSVs remain
optional local artifacts. The notebook verifies published models without them.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import zipfile
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .utils.biased_mf import (
    DRUG, PROTEIN, PAIR_KEYS, PARAMETERS, HISTORY_COLUMNS, fit_biased_mf, predict_biased_mf,
    save_biased_mf, load_biased_mf,
)
from .utils.evaluation import mean_squared_error, concordance_index, hit_at_k

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results" / "biased_mf"
OUT = EVIDENCE
SCORE_SCALES = {
    "davis": "pKd; supplied values, no additional log transformation",
    "kiba": "transformed KIBA score; supplied values, no additional transformation",
}
FACTORS = (8, 16, 32, 64, 128)
PENALTIES = (0.0, 0.01, 0.1, 1.0, 10.0)
RATES = (0.001, 0.003, 0.005, 0.01)
SEEDS = (42, 123, 2026)
FIXED = {"batch_size": 2048, "patience": 10}
CONFIG_COLUMNS = ["dataset", "n_factors", "learning_rate", "regularization", "seed", "max_epochs"]
METRICS = ["train_mse", "val_mse", "val_ci", "val_hit_at_1", "val_hit_at_5",
           "val_hit_at_5_candidates_gt_5"]


def load_selection_splits(root, dataset):
    """Load exactly training and validation; verify their pair/ID contracts."""
    if dataset not in SCORE_SCALES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    frames = {
        part: pd.read_csv(Path(root) / "data" / "splits" / f"{dataset}_{part}.csv",
                          float_precision="round_trip")
        for part in ("train", "val")
    }
    train, val = frames["train"], frames["val"]
    for part, frame in frames.items():
        if frame.empty or frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{dataset} {part}: empty or duplicate observations")
        if not np.isfinite(frame["affinity"].to_numpy(dtype=float)).all():
            raise ValueError(f"{dataset} {part}: non-finite labels")
    train_pairs = pd.MultiIndex.from_frame(train[PAIR_KEYS])
    if pd.MultiIndex.from_frame(val[PAIR_KEYS]).isin(train_pairs).any():
        raise ValueError("Training and validation pairs overlap")
    for column in PAIR_KEYS:
        if not val[column].isin(train[column]).all():
            raise ValueError(f"Validation has IDs absent from training: {column}")
    return frames


def candidate_distribution(validation, dataset):
    """Exact observed-validation candidate-count histogram and summary."""
    counts = validation.groupby(DRUG)[PROTEIN].nunique()
    histogram = counts.value_counts().sort_index().rename_axis("candidate_count").reset_index(name="ligands")
    histogram.insert(0, "dataset", dataset)
    histogram["ligand_fraction"] = histogram["ligands"] / len(counts)
    summary = {
        "dataset": dataset, "ligands": len(counts), "observed_candidates": int(counts.sum()),
        "min": int(counts.min()), "p25": float(counts.quantile(.25)),
        "median": float(counts.median()), "p75": float(counts.quantile(.75)),
        "p90": float(counts.quantile(.90)), "max": int(counts.max()),
        "mean": float(counts.mean()), "singleton_ligands": int((counts == 1).sum()),
        "ligands_at_most_5": int((counts <= 5).sum()),
        "ligands_more_than_5": int((counts > 5).sum()),
    }
    return histogram, summary


def evaluate_validation(model, training, validation):
    train_predictions = predict_biased_mf(model, training[PAIR_KEYS])
    predictions = predict_biased_mf(model, validation[PAIR_KEYS])
    one, five = hit_at_k(validation, predictions, 1), hit_at_k(validation, predictions, 5)
    eligible = validation.groupby(DRUG)[PROTEIN].transform("size").to_numpy() > 5
    supplementary = hit_at_k(validation.loc[eligible], predictions[eligible], 5)
    return {
        "train_mse": mean_squared_error(training["affinity"], train_predictions),
        "val_mse": mean_squared_error(validation["affinity"], predictions),
        "val_ci": concordance_index(validation["affinity"], predictions),
        "val_hit_at_1": one["hit_rate"], "val_hit_at_5": five["hit_rate"],
        "val_eligible_ligands": five["eligible_ligands"],
        "val_hit_at_5_candidates_gt_5": supplementary["hit_rate"],
        "val_eligible_ligands_candidates_gt_5": supplementary["eligible_ligands"],
    }


def select_best(rows):
    """Primary criterion: validation MSE; deterministic parameter ties."""
    successful = pd.DataFrame(rows)
    successful = successful.loc[successful["status"].eq("ok")]
    if successful.empty:
        raise RuntimeError("No stable configurations to select")
    return successful.sort_values(
        ["val_mse", "n_factors", "learning_rate", "regularization"], kind="stable").iloc[0]


def _curve(model, path, title):
    """Single-model curve used by the explicit-grid compatibility API."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = model["history"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history[:, 0], history[:, 1], label="Training MSE")
    ax.plot(history[:, 0], history[:, 2], label="Validation MSE")
    ax.axvline(int(model["best_epoch"]), color="gray", linestyle="--", label="Best epoch")
    ax.set(xlabel="Epoch", ylabel="MSE", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_search(root, grid, max_epochs=100, reuse_baseline=False):
    """Explicit small-grid API retained for programmatic/synthetic test clients.

    There is no implicit old grid, baseline cache, or CLI search mode. Requires
    fresh output paths, and uses the same fitter and metrics as the campaign.
    """
    if reuse_baseline:
        raise ValueError("The obsolete baseline cache is no longer supported")
    grid = tuple(grid)
    if not grid:
        raise ValueError("Supply at least one explicit configuration")
    root = Path(root)
    results, models = root / "results", root / "saved_models"
    result_path = results / "biased_mf_explicit_grid.csv"
    if result_path.exists() or any((models / f"{ds}_biased_mf.npz").exists() for ds in SCORE_SCALES):
        raise FileExistsError("Use fresh output paths; existing results/checkpoints are protected")
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    all_rows, selected = [], []
    for dataset in SCORE_SCALES:
        parts = load_selection_splits(root, dataset)
        trained, rows = {}, []
        for factors, rate, penalty in grid:
            model = fit_biased_mf(parts["train"], parts["val"], n_factors=factors,
                                  learning_rate=rate, regularization=penalty, max_epochs=max_epochs,
                                  **FIXED, seed=42, dataset=dataset, score_scale=SCORE_SCALES[dataset])
            row = dict(dataset=dataset, n_factors=factors, learning_rate=rate,
                       regularization=penalty, status="ok", best_epoch=int(model["best_epoch"]),
                       **evaluate_validation(model, parts["train"], parts["val"]))
            trained[(factors, rate, penalty)] = model
            rows.append(row)
        best = select_best(rows)
        model = trained[(int(best.n_factors), float(best.learning_rate), float(best.regularization))]
        path = models / f"{dataset}_biased_mf.npz"
        save_biased_mf(model, path)
        restored = load_biased_mf(path)
        for frame in parts.values():
            np.testing.assert_array_equal(predict_biased_mf(model, frame[PAIR_KEYS]),
                                          predict_biased_mf(restored, frame[PAIR_KEYS]))
        _curve(model, results / f"{dataset}_explicit_grid.png", dataset.upper())
        selected.append(best.to_dict())
        all_rows.extend(rows)
    write_csv(result_path, pd.DataFrame(all_rows))
    return pd.DataFrame(selected)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, frame):
    frame.to_csv(path, index=False, mode="x")


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def config(dataset, factors, rate, penalty, seed=42, cap=100):
    return dict(dataset=str(dataset), n_factors=int(factors), learning_rate=float(rate),
                regularization=float(penalty), seed=int(seed), max_epochs=int(cap))


def row_config(row):
    return config(row["dataset"], row["n_factors"], row["learning_rate"],
                  row["regularization"], row["seed"], row["max_epochs"])


def run_id(cfg):
    number = lambda x: f"{x:g}".replace(".", "p")
    return (f"{cfg['dataset']}_f{cfg['n_factors']}_a{number(cfg['learning_rate'])}"
            f"_l{number(cfg['regularization'])}_s{cfg['seed']}_e{cfg['max_epochs']}")


def current_fingerprints():
    sources = [ROOT / "utils/biased_mf.py", ROOT / "utils/evaluation.py", Path(__file__)]
    data = [ROOT / "data/splits" / f"{ds}_{part}.csv"
            for ds, part in product(("davis", "kiba"), ("train", "val"))]
    return {str(path.relative_to(ROOT)): sha(path) for path in sources + data}


def historical_rows():
    """Optional local comparisons; a fresh checkout needs no historical CSVs."""
    path = EVIDENCE / "provenance/historical_results.csv"
    if path.is_file():
        return read_csv(path)
    return pd.DataFrame(columns=CONFIG_COLUMNS + METRICS + ["historical_source"])


def matching_history(cfg):
    rows = historical_rows()
    for key in CONFIG_COLUMNS:
        rows = rows.loc[rows[key].eq(cfg[key])]
    return rows


PROTOCOL = """# Prospective campaign protocol

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

Reuse: optional historical artifacts have no source/data hashes; most lack parameters.
Re-run historical configurations needed for this campaign, explicitly labeled.
Compare historical metrics as reproduction checks; the original campaign's
parameter-state reproduction evidence is retained in the compact audit.
Existing lambda=100 results are historical supplementary evidence only.
When the historical comparison CSV is absent, all required
conditions are fitted normally, and historical comparison checks are marked
unavailable rather than fabricated. This does not change the experiment grid.
Within this campaign, require matching config, source/data hashes, mappings,
mean, checkpoint fields, saved predictions and recomputed metrics before reuse.

Budget before extensions: 50 core fits (20 capacity + 30 interaction additions),
12 additional rate fits, 8 additional seed fits = 70 physical fits. One existing
DAVIS F=8, alpha=.005, lambda=.01 run is the runner smoke check and is counted
in the 20 capacity fits. The original recorded campaign had 14 historical
repeats + 36 new core configurations; a fresh run may have no such references.
Additional extension fits are conditional and retained separately. Tests use
synthetic data and are not counted as scientific campaign fits.

No dataset/model/shared metric changes, no old-artifact overwrites, no test
evaluation, no automatic final model refit on training+validation, no commits.
"""


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ("runs", "figures", "checkpoints"):
        (OUT / name).mkdir()
    fingerprints = current_fingerprints()
    manifest = dict(mode="full_campaign", fingerprints=fingerprints, factors=FACTORS, penalties=PENALTIES,
                    rates=RATES, seeds=SEEDS, fixed=FIXED, initial_cap=100, extended_cap=300,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
                    final_seed_policy="always 42", selection="three-seed mean validation MSE")
    write_json(OUT / "manifest.json", manifest)
    (OUT / "PROTOCOL.md").write_text(PROTOCOL)
    data_rows, audit, candidate_stats = [], [], []
    historical = historical_rows()
    for ds in ("davis", "kiba"):
        parts = load_selection_splits(ROOT, ds)
        for part, frame in parts.items():
            if not frame[PAIR_KEYS].map(lambda x: isinstance(x, str) and bool(x.strip())).all().all():
                raise ValueError("Missing/nonstring identifiers")
            data_rows.append(dict(dataset=ds, split=part, rows=len(frame),
                                  drugs=frame[PAIR_KEYS[0]].nunique(), targets=frame[PAIR_KEYS[1]].nunique(),
                                  min_affinity=float(frame.affinity.min()), max_affinity=float(frame.affinity.max()),
                                  sha256=fingerprints[f"data/splits/{ds}_{part}.csv"],
                                  pair_uniqueness="passed", training_coverage="passed",
                                  train_validation_disjoint="passed"))
        _, candidate = candidate_distribution(parts["val"], ds)
        candidate_stats.append(candidate)
        for _, old in historical.loc[historical.dataset.eq(ds)].iterrows():
            cfg = row_config(old)
            audit.append({**cfg, "historical_source": old.historical_source,
                          "checkpoint_available": False,
                          "metadata_mappings_predictions_metrics_checked": False,
                          "historical_source_data_hashes_available": False,
                          "decision": "rerun_if_required; historical provenance unverified"})
    write_csv(OUT / "data_integrity.csv", pd.DataFrame(data_rows))
    write_csv(OUT / "candidate_summary.csv", pd.DataFrame(candidate_stats))
    audit_columns = CONFIG_COLUMNS + ["historical_source", "checkpoint_available",
                                    "metadata_mappings_predictions_metrics_checked",
                                    "historical_source_data_hashes_available", "decision"]
    write_csv(OUT / "historical_reuse_audit.csv", pd.DataFrame(audit, columns=audit_columns))
    plan = []
    for ds, f, penalty in product(("davis", "kiba"), FACTORS, PENALTIES):
        cfg = config(ds, f, .005, penalty)
        old = matching_history(cfg)
        plan.append({**cfg, **FIXED, "first_stage": "capacity" if penalty in (0, .01) else "interaction",
                     "historical_match": not old.empty,
                     "action": "historical_unverified_rerun" if not old.empty else "new_configuration"})
    write_csv(OUT / "core_plan.csv", pd.DataFrame(plan))
    write_csv(OUT / "historical_lambda100_supplement.csv", historical.loc[historical.regularization.eq(100)])
    print(pd.DataFrame(data_rows).to_string(index=False))
    print("Plan: 50 core fits, 12 additional rate fits, 8 additional seed fits; extensions conditional.")
    print(f"Optional historical matches in core plan: {sum(row['historical_match'] for row in plan)}")


class Campaign:
    def __init__(self):
        self.manifest = json.loads((OUT / "manifest.json").read_text())
        if self.manifest.get("mode") == "consolidated_evidence":
            raise ValueError("Published evidence is read-only; use --stage verify or a fresh --output-dir")
        if current_fingerprints() != self.manifest["fingerprints"]:
            raise ValueError("Source or train/validation data fingerprint changed; do not reuse this campaign")
        self.provenance = json.dumps(self.manifest["fingerprints"], sort_keys=True)
        self.parts = {}

    def data(self, ds):
        if ds not in self.parts:
            self.parts[ds] = load_selection_splits(ROOT, ds)
        return self.parts[ds]

    def verify(self, row):
        cfg = row_config(row)
        folder = OUT / "runs" / row["run_id"]
        model = load_biased_mf(folder / "model.npz")
        assert model["campaign_provenance"].item() == self.provenance
        assert sha(folder / "model.npz") == row["checkpoint_sha256"]
        for key, value in {**cfg, **FIXED}.items():
            assert model[key].item() == value
        parts = self.data(cfg["dataset"])
        for key, column in (("drugs", PAIR_KEYS[0]), ("targets", PAIR_KEYS[1])):
            np.testing.assert_array_equal(model[key], np.sort(parts["train"][column].unique()))
        assert float(model["global_mean"]) == float(parts["train"].affinity.mean())
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            for split in ("train", "val"):
                predicted = predict_biased_mf(model, parts[split][PAIR_KEYS])
                np.testing.assert_array_equal(predicted, saved[split])
                assert array_sha(predicted) == row[f"{split}_predictions_sha256"]
        metrics = evaluate_validation(model, parts["train"], parts["val"])
        for key, value in metrics.items():
            np.testing.assert_allclose(value, row[key], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(value, float(model[f"metric_{key}"]), atol=1e-12, rtol=1e-12)
        best = int(model["best_epoch"])
        assert best == int(np.argmin(model["history"][:, 2])) == row["best_epoch"]
        np.testing.assert_allclose(model["history"][best, 1:3], [row["train_mse"], row["val_mse"]],
                                   atol=1e-12, rtol=1e-12)
        return model

    def trial(self, cfg, stage):
        if cfg["n_factors"] not in FACTORS or cfg["max_epochs"] not in (100, 300):
            raise ValueError("Configuration outside the planned capacity/budget limits")
        rid = run_id(cfg)
        folder = OUT / "runs" / rid
        if (folder / "result.json").exists():
            row = json.loads((folder / "result.json").read_text())
            if row["status"] == "ok":
                self.verify(row)
            print(f"REUSE verified campaign run {rid}", flush=True)
            return row
        folder.mkdir(exist_ok=False)
        parts = self.data(cfg["dataset"])
        start = perf_counter()
        prior = matching_history(cfg)
        row = {**cfg, **FIXED, "run_id": rid, "first_stage": stage,
               "origin": "historical_unverified_rerun" if not prior.empty else "new_configuration"}
        try:
            fit_cfg = {key: value for key, value in cfg.items() if key != "dataset"}
            model = fit_biased_mf(parts["train"], parts["val"], **fit_cfg, **FIXED,
                                  dataset=cfg["dataset"], score_scale=SCORE_SCALES[cfg["dataset"]])
            fit_seconds = perf_counter() - start
            metrics = evaluate_validation(model, parts["train"], parts["val"])
            trace = model["history"]
            best, last = int(model["best_epoch"]), int(model["epochs_run"])
            np.testing.assert_allclose(trace[best, 1:3], [metrics["train_mse"], metrics["val_mse"]],
                                       atol=1e-12, rtol=1e-12)
            assert best == int(np.argmin(trace[:, 2]))
            batches = math.ceil(len(parts["train"]) / FIXED["batch_size"])
            row.update(metrics, status="ok", numerical_issue="", best_epoch=best, epochs_run=last,
                       stop_reason=str(model["stop_reason"]), fit_seconds=fit_seconds,
                       parameter_count=sum(model[key].size for key in PARAMETERS),
                       parameter_squared_norm=sum(float(np.sum(model[key] ** 2)) for key in PARAMETERS),
                       train_val_mse_gap=metrics["val_mse"] - metrics["train_mse"],
                       train_mse_last=float(trace[-1, 1]), val_mse_last=float(trace[-1, 2]),
                       post_best_val_increase=float(trace[-1, 2] - trace[best, 2]),
                       post_best_train_decrease=float(trace[best, 1] - trace[-1, 1]),
                       overfit_curve_evidence=bool(trace[-1, 2] > trace[best, 2] + 1e-6
                                                  and trace[-1, 1] < trace[best, 1] - 1e-6),
                       cap_still_improving=bool(model["stop_reason"] == "max_epochs" and best > last - 10),
                       total_optimizer_updates=last * batches, updates_to_best=best * batches)
            if not prior.empty:
                old = prior.iloc[0]
                row["historical_metrics_reproduced"] = bool(all(
                    np.isclose(metrics[key], old[key], rtol=1e-12, atol=1e-12) for key in METRICS))
            model["campaign_provenance"] = self.provenance
            model["campaign_run_id"] = rid
            model.update({f"metric_{key}": value for key, value in metrics.items()})
            predictions = {part: predict_biased_mf(model, frame[PAIR_KEYS]) for part, frame in parts.items()}
            for part, predicted in predictions.items():
                row[f"{part}_predictions_sha256"] = array_sha(predicted)
            save_biased_mf(model, folder / "model.npz")
            np.savez_compressed(folder / "predictions.npz", **predictions)
            row["checkpoint_sha256"] = sha(folder / "model.npz")
            history = pd.DataFrame(trace, columns=HISTORY_COLUMNS)
            history["epoch"] = history.epoch.astype(int)
            for key, value in {**cfg, "run_id": rid}.items():
                history[key] = value
            write_csv(folder / "history.csv", history)
            self.verify(row)
            print(f"FIT {rid}: train={row['train_mse']:.6f} val={row['val_mse']:.6f} "
                  f"best={best}/{last} {row['stop_reason']} ({fit_seconds:.1f}s)", flush=True)
        except FloatingPointError as exc:
            row.update(status="numerical_failure", numerical_issue=str(exc), fit_seconds=perf_counter() - start)
            print(f"FAIL {rid}: {exc}", flush=True)
        write_json(folder / "result.json", row)
        return row

    def extended_trial(self, cfg, stage):
        initial = self.trial({**cfg, "max_epochs": 100}, stage)
        if initial["status"] == "ok" and initial["cap_still_improving"]:
            extended = self.trial({**cfg, "max_epochs": 300}, stage + "_extension")
            if extended["status"] == "ok":
                a = load_biased_mf(OUT / "runs" / initial["run_id"] / "model.npz")["history"]
                b = load_biased_mf(OUT / "runs" / extended["run_id"] / "model.npz")["history"]
                np.testing.assert_array_equal(a, b[:len(a)])
            return {**extended, "initial_run_id": initial["run_id"], "extension_applied": True}
        return {**initial, "initial_run_id": initial["run_id"], "extension_applied": False}


def successful(frame):
    return frame.loc[frame.status.eq("ok")].copy()


def sort_results(frame):
    return successful(frame).sort_values(["val_mse", "n_factors", "regularization", "learning_rate"], kind="stable")


def peer_evidence(frame, interaction=False):
    frame = frame.copy()
    frame["underfit_evidence"] = "not established"
    for _, group in successful(frame).groupby(["dataset", "n_factors" if interaction else "regularization"]):
        best = sort_results(group).iloc[0]
        for index, row in group.iterrows():
            if row.cap_still_improving:
                label = "budget-limited; insufficient evidence"
            elif row.train_mse > best.train_mse + 1e-6 and row.val_mse > best.val_mse + 1e-6:
                label = ("both errors higher than same-F best; possible excessive penalty"
                         if interaction and row.regularization > best.regularization
                         else "both errors higher than validation-best peer; possible capacity/optimization limitation")
            else:
                label = "not established"
            frame.loc[index, "underfit_evidence"] = label
    return frame


def curve_plot(rows, filename, title):
    import matplotlib.pyplot as plt
    rows = successful(pd.DataFrame(rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for _, row in rows.iterrows():
        trace = read_csv(OUT / "runs" / row.run_id / "history.csv")
        label = f"F={row.n_factors}, L={row.regularization:g}, a={row.learning_rate:g}, s={row.seed}"
        for ax, metric in zip(axes, ("train_mse", "val_mse")):
            line, = ax.plot(trace.epoch, trace[metric], label=label)
            point = trace.loc[trace.epoch.eq(row.best_epoch)].iloc[0]
            ax.scatter(point.epoch, point[metric], color=line.get_color(), s=20)
            ax.set(xlabel="Epoch", ylabel=metric.replace("_", " ").upper())
    axes[1].legend(fontsize=7)
    fig.suptitle(title + " (dots: restored epochs)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / filename, dpi=140)
    plt.close(fig)


def stage_capacity(campaign):
    if not (OUT / "software_tests.json").exists():
        raise RuntimeError("Run the software-test stage first")
    rows = [campaign.trial(config(ds, f, .005, penalty), "capacity")
            for ds, f, penalty in product(("davis", "kiba"), FACTORS, (0., .01))]
    frame = peer_evidence(pd.DataFrame(rows))
    write_csv(OUT / "stage1_capacity.csv", frame)
    write_csv(OUT / "stage2_curve_diagnostics.csv", frame)
    import matplotlib.pyplot as plt
    for ds in ("davis", "kiba"):
        subset = successful(frame.loc[frame.dataset.eq(ds)])
        fig, ax = plt.subplots(figsize=(7, 4))
        for penalty, group in subset.groupby("regularization"):
            ax.plot(group.n_factors, group.train_mse, "o--", label=f"Training, lambda={penalty:g}")
            ax.plot(group.n_factors, group.val_mse, "o-", label=f"Validation, lambda={penalty:g}")
        ax.set(xlabel="Latent factors", ylabel="MSE at best validation epoch", title=ds.upper())
        ax.set_xticks(FACTORS)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{ds}_capacity.png", dpi=140)
        plt.close(fig)
        for penalty in (0., .01):
            curve_plot(subset.loc[subset.regularization.eq(penalty)],
                       f"{ds}_capacity_curves_l{penalty:g}.png", f"{ds.upper()} capacity, lambda={penalty:g}")
    print(frame[["dataset", "n_factors", "regularization", "train_mse", "val_mse",
                 "best_epoch", "epochs_run", "overfit_curve_evidence"]].to_string(index=False))


def stage_interaction(campaign):
    read_csv(OUT / "stage1_capacity.csv")  # Enforce sequential workflow.
    rows = [campaign.trial(config(ds, f, .005, penalty), "interaction")
            for ds, f, penalty in product(("davis", "kiba"), FACTORS, PENALTIES)]
    frame = peer_evidence(pd.DataFrame(rows), interaction=True)
    write_csv(OUT / "stage3_interaction.csv", frame)
    ordered = sort_results(frame)
    write_csv(OUT / "stage3_best_lambda_per_f.csv", ordered.groupby(["dataset", "n_factors"]).head(1))
    write_csv(OUT / "stage3_best_f_per_lambda.csv", ordered.groupby(["dataset", "regularization"]).head(1))
    import matplotlib.pyplot as plt
    for ds in ("davis", "kiba"):
        subset = successful(frame.loc[frame.dataset.eq(ds)])
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ("train_mse", "val_mse")):
            table = subset.pivot(index="n_factors", columns="regularization", values=metric).reindex(index=FACTORS, columns=PENALTIES)
            image = ax.imshow(table, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(PENALTIES)), [f"{x:g}" for x in PENALTIES])
            ax.set_yticks(range(len(FACTORS)), FACTORS)
            ax.set(xlabel="Lambda", ylabel="Factors", title=metric.replace("_", " ").upper())
            for i, j in product(range(len(FACTORS)), range(len(PENALTIES))):
                ax.text(j, i, f"{table.iloc[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(image, ax=ax)
        fig.suptitle(f"{ds.upper()} F-lambda interaction: restored-best MSE")
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{ds}_interaction_heatmaps.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for f, group in subset.groupby("n_factors"):
            ax.plot(range(len(group)), group.sort_values("regularization").val_mse, "o-", label=f"F={f}")
        ax.set_xticks(range(len(PENALTIES)), [f"{x:g}" for x in PENALTIES])
        ax.set(xlabel="Lambda (categorical spacing, including zero)", ylabel="Validation MSE", title=ds.upper())
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{ds}_regularization.png", dpi=140)
        plt.close(fig)
        for f in (8, 64, 128):
            curve_plot(subset.loc[subset.n_factors.eq(f) & subset.regularization.isin([0, 1, 10])],
                       f"{ds}_regularization_curves_f{f}.png", f"{ds.upper()} regularization, F={f}")
    print(ordered.groupby("dataset").head(3)[["dataset", "n_factors", "regularization", "train_mse", "val_mse"]].to_string(index=False))


def stage_rates(campaign):
    interaction = read_csv(OUT / "stage3_interaction.csv")
    candidates = sort_results(interaction).groupby("dataset", sort=False).head(2)
    write_csv(OUT / "stage4_candidates.csv", candidates)
    plan = [config(row.dataset, row.n_factors, rate, row.regularization)
            for _, row in candidates.iterrows() for rate in RATES]
    write_csv(OUT / "stage4_plan.csv", pd.DataFrame(plan))
    rows = [campaign.extended_trial(cfg, "rates") for cfg in plan]
    frame = pd.DataFrame(rows)
    write_csv(OUT / "stage4_rates.csv", frame)
    for ds in ("davis", "kiba"):
        for number, (_, pair) in enumerate(candidates.loc[candidates.dataset.eq(ds)].iterrows(), 1):
            subset = frame.loc[frame.dataset.eq(ds) & frame.n_factors.eq(pair.n_factors)
                               & frame.regularization.eq(pair.regularization)]
            curve_plot(subset, f"{ds}_rate_curves_candidate{number}.png", f"{ds.upper()} learning-rate candidate {number}")
    print(sort_results(frame)[["dataset", "n_factors", "regularization", "learning_rate", "val_mse", "best_epoch",
                              "total_optimizer_updates", "extension_applied", "cap_still_improving"]].to_string(index=False))


def stage_seeds(campaign):
    rates = read_csv(OUT / "stage4_rates.csv")
    candidates = sort_results(rates).groupby("dataset", sort=False).head(2)
    write_csv(OUT / "stage5_candidates_frozen.csv", candidates)
    plan = [config(row.dataset, row.n_factors, row.learning_rate, row.regularization, seed)
            for _, row in candidates.iterrows() for seed in SEEDS]
    write_csv(OUT / "stage5_plan.csv", pd.DataFrame(plan))
    frame = pd.DataFrame([campaign.extended_trial(cfg, "seeds") for cfg in plan])
    write_csv(OUT / "stage5_seed_results.csv", frame)
    summaries = []
    keys = ["dataset", "n_factors", "regularization", "learning_rate"]
    for key, group in frame.groupby(keys):
        row = dict(zip(keys, key))
        ok = successful(group)
        row["successful_seeds"] = len(ok)
        row["selection_eligible"] = len(ok) == len(SEEDS)
        for metric in METRICS + ["best_epoch", "train_val_mse_gap"]:
            row[f"{metric}_mean"] = float(ok[metric].mean())
            row[f"{metric}_std"] = float(ok[metric].std(ddof=1))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    write_csv(OUT / "stage5_robustness_summary.csv", summary)
    eligible = summary.loc[summary.selection_eligible].sort_values(["val_mse_mean", "n_factors", "regularization", "learning_rate"], kind="stable")
    chosen = eligible.groupby("dataset", sort=False).head(1)
    if len(chosen) != 2:
        raise RuntimeError("Need a fully evaluated three-seed candidate for each dataset")
    final = []
    import matplotlib.pyplot as plt
    for ds in ("davis", "kiba"):
        sub = summary.loc[summary.dataset.eq(ds)]
        fig, ax = plt.subplots(figsize=(7, 4))
        for number, (_, row) in enumerate(sub.iterrows()):
            group = successful(frame.loc[frame.dataset.eq(ds) & frame.n_factors.eq(row.n_factors)
                                        & frame.regularization.eq(row.regularization)
                                        & frame.learning_rate.eq(row.learning_rate)])
            ax.scatter(np.full(len(group), number), group.val_mse, label="Individual seeds" if number == 0 else None)
            ax.errorbar(number, row.val_mse_mean, yerr=row.val_mse_std, fmt="s", color="black", capsize=5,
                        label="Mean +/- sample SD" if number == 0 else None)
        ax.set_xticks(range(len(sub)), [f"F={r.n_factors}, L={r.regularization:g}\na={r.learning_rate:g}" for _, r in sub.iterrows()])
        ax.set(ylabel="Validation MSE", title=f"{ds.upper()} seed robustness (n=3)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{ds}_seed_robustness.png", dpi=140)
        plt.close(fig)
        winner = chosen.loc[chosen.dataset.eq(ds)].iloc[0]
        seed42 = frame.loc[frame.dataset.eq(ds) & frame.n_factors.eq(winner.n_factors)
                           & frame.regularization.eq(winner.regularization)
                           & frame.learning_rate.eq(winner.learning_rate) & frame.seed.eq(42)].iloc[0].to_dict()
        model = campaign.verify(seed42)
        model["final_selection_policy"] = "lowest three-seed mean validation MSE; fixed handoff seed 42"
        model["robustness_seeds"] = np.asarray(SEEDS)
        model["robustness_val_mse_mean"] = winner.val_mse_mean
        model["robustness_val_mse_std"] = winner.val_mse_std
        checkpoint = OUT / "checkpoints" / f"{ds}_biased_mf.npz"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        save_biased_mf(model, checkpoint)
        restored = load_biased_mf(checkpoint)
        for part, data in campaign.data(ds).items():
            predicted = predict_biased_mf(restored, data[PAIR_KEYS])
            assert array_sha(predicted) == seed42[f"{part}_predictions_sha256"]
        final.append({**seed42, "robustness_val_mse_mean": winner.val_mse_mean,
                      "robustness_val_mse_std": winner.val_mse_std,
                      "selected_checkpoint": str(checkpoint.relative_to(OUT)),
                      "selected_checkpoint_sha256": sha(checkpoint)})
    write_csv(OUT / "final_selection.csv", pd.DataFrame(final))
    print(summary.to_string(index=False))
    print("FINAL (fixed seed42):")
    print(pd.DataFrame(final)[["dataset", "n_factors", "regularization", "learning_rate", "seed", "val_mse",
                              "val_ci", "val_hit_at_1", "val_hit_at_5", "best_epoch",
                              "robustness_val_mse_mean", "robustness_val_mse_std"]].to_string(index=False))


def verify_all(campaign):
    rows = [json.loads(path.read_text()) for path in sorted((OUT / "runs").glob("*/result.json"))]
    for row in rows:
        if row["status"] == "ok":
            campaign.verify(row)
    inventory = pd.DataFrame(rows)
    write_csv(OUT / "all_physical_runs.csv", inventory)
    histories = [read_csv(OUT / "runs" / row["run_id"] / "history.csv") for row in rows if row["status"] == "ok"]
    write_csv(OUT / "all_training_histories.csv", pd.concat(histories, ignore_index=True))
    write_json(OUT / "verification.json", {
        "physical_fits": len(rows), "successful": int(inventory.status.eq("ok").sum()),
        "failures": int(inventory.status.ne("ok").sum()),
        "extended_cap_fits": int(inventory.max_epochs.eq(300).sum()),
        "checkpoint_and_prediction_reloads": "exact equality for every successful run on train and validation",
        "fingerprints_match": current_fingerprints() == campaign.manifest["fingerprints"],
        "test_splits_loaded": False,
    })
    print(inventory.groupby(["first_stage", "status"]).size().to_string())
    print(f"Verified {len(rows)} physical runs; no final-test data used.")


def verify_published(evidence=EVIDENCE):
    """Read-only validation of compact evidence and selected prediction states.

    No fitting, output writes or intermediate-run archive is needed. Historical
    source hashes remain historical; consolidation has its own source manifest.
    """
    evidence = Path(evidence)
    manifest = json.loads((evidence / "manifest.json").read_text())
    if current_fingerprints() != manifest["fingerprints"]:
        raise ValueError("Current source/data hashes differ from the active evidence manifest")
    consolidated = manifest.get("mode") == "consolidated_evidence"
    if consolidated:
        original = json.loads((evidence / "provenance/original_manifest.json").read_text())
        provenance = json.dumps(original["fingerprints"], sort_keys=True)
        archive = evidence / "provenance/original_sources.zip"
        with zipfile.ZipFile(archive) as sources:
            for name, expected in original["fingerprints"].items():
                if name.startswith("data/"):
                    assert current_fingerprints()[name] == expected
                else:
                    assert hashlib.sha256(sources.read(name)).hexdigest() == expected
        for name in ("utils/biased_mf.py", "utils/evaluation.py"):
            assert manifest["fingerprints"][name] == original["fingerprints"][name]
    else:
        provenance = json.dumps(manifest["fingerprints"], sort_keys=True)

    selection = read_csv(evidence / "final_selection.csv")
    runs = read_csv(evidence / "all_physical_runs.csv")
    histories = read_csv(evidence / "all_training_histories.csv")
    seeds = read_csv(evidence / "stage5_seed_results.csv")
    summary = read_csv(evidence / "stage5_robustness_summary.csv")
    assert set(selection.dataset) == {"davis", "kiba"} and len(selection) == 2
    verified = []
    for _, row in selection.iterrows():
        checkpoint = (evidence / row.selected_checkpoint).resolve()
        assert sha(checkpoint) == row.selected_checkpoint_sha256
        model = load_biased_mf(checkpoint)
        assert model["campaign_provenance"].item() == provenance
        for key, value in {**row_config(row), **FIXED}.items():
            assert model[key].item() == value
        if consolidated:
            assert (int(model["n_factors"]), float(model["regularization"]),
                    float(model["learning_rate"]), int(model["seed"])) == (128, 1., .001, 42)
            assert int(model["best_epoch"]) == (74 if row.dataset == "davis" else 34)
        parts = load_selection_splits(ROOT, row.dataset)
        for key, column in (("drugs", DRUG), ("targets", PROTEIN)):
            np.testing.assert_array_equal(model[key], np.sort(parts["train"][column].unique()))
        assert float(model["global_mean"]) == float(parts["train"].affinity.mean())
        for split, data in parts.items():
            predictions = predict_biased_mf(model, data[PAIR_KEYS])
            assert array_sha(predictions) == row[f"{split}_predictions_sha256"]
        measured = evaluate_validation(model, parts["train"], parts["val"])
        for key, value in measured.items():
            np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(value, float(model[f"metric_{key}"]), rtol=1e-12, atol=1e-12)
        history = model["history"]
        best = int(model["best_epoch"])
        assert best == int(np.argmin(history[:, 2])) == row.best_epoch
        np.testing.assert_allclose(history[best, 1:3], [row.train_mse, row.val_mse], atol=1e-12, rtol=1e-12)

        # All trajectories/optimizer budgets remain independently checkable
        # from compact tables, even when intermediate checkpoints are excluded.
        for _, run in successful(runs.loc[runs.dataset.eq(row.dataset)]).iterrows():
            trace = histories.loc[histories.run_id.eq(run.run_id)]
            assert len(trace) == run.epochs_run + 1
            trace_best = trace.loc[trace.val_mse.idxmin()]
            assert trace_best.epoch == run.best_epoch
            np.testing.assert_allclose([trace_best.train_mse, trace_best.val_mse],
                                       [run.train_mse, run.val_mse], atol=1e-12, rtol=1e-12)
            n = len(parts["train"])
            count = (len(model["drugs"]) + len(model["targets"])) * (run.n_factors + 1)
            assert run.parameter_count == count
            assert run.total_optimizer_updates == math.ceil(n / FIXED["batch_size"]) * run.epochs_run
            np.testing.assert_allclose(trace_best.train_objective,
                                       .5 * run.train_mse + run.regularization * run.parameter_squared_norm / (2*n),
                                       atol=1e-12, rtol=1e-12)
        for _, candidate in summary.loc[summary.dataset.eq(row.dataset)].iterrows():
            group = seeds.loc[seeds.dataset.eq(row.dataset) & seeds.n_factors.eq(candidate.n_factors)
                              & seeds.regularization.eq(candidate.regularization)
                              & seeds.learning_rate.eq(candidate.learning_rate)]
            assert set(group.seed) == set(SEEDS)
            for metric in METRICS + ["best_epoch", "train_val_mse_gap"]:
                np.testing.assert_allclose([candidate[f"{metric}_mean"], candidate[f"{metric}_std"]],
                                           [group[metric].mean(), group[metric].std(ddof=1)], atol=1e-12, rtol=1e-12)
        best_candidate = summary.loc[summary.dataset.eq(row.dataset) & summary.selection_eligible].sort_values(
            ["val_mse_mean", "n_factors", "regularization", "learning_rate"], kind="stable").iloc[0]
        for key in ("n_factors", "regularization", "learning_rate"):
            assert row[key] == best_candidate[key]
        assert row.seed == 42
        verified.append(dict(dataset=row.dataset, checkpoint=str(checkpoint.relative_to(ROOT)),
                             bytes=checkpoint.stat().st_size, best_epoch=best,
                             exact_train_validation_prediction_hashes=True, **measured))
    print(pd.DataFrame(verified).to_string(index=False))
    print(f"Verified {len(runs)} retained physical-run histories and both selected checkpoints; no fitting performed.")
    return {"mode": "read_only", "physical_run_histories_verified": len(runs),
            "source_and_data_hashes_match": True, "selected_checkpoints": verified,
            "intermediate_checkpoints_required": False, "test_data_accessed": False}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "tests", "capacity", "interaction", "rates", "seeds", "summarize", "verify"), required=True)
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE,
                        help="Published evidence for verify; use a fresh directory for deliberate reproduction")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if args.stage != "prepare" and not (OUT / "manifest.json").is_file():
        parser.error("The separate full-evidence CSV/manifest bundle is not available here. "
                     "Use task2/biased_mf.ipynb to inspect recorded tables and verify the published models. "
                     "For deliberate full reproduction use --stage prepare with a fresh --output-dir, "
                     "or point --output-dir to an existing completed run. Verification never trains a model.")
    if args.stage == "verify":
        verify_published(OUT)
        return
    if args.stage == "prepare":
        prepare()
        return
    campaign = Campaign()
    if args.stage == "smoke":
        row = campaign.trial(config("davis", 8, .005, .01), "smoke")
        assert row["status"] == "ok"
        if "historical_metrics_reproduced" in row:
            assert row["historical_metrics_reproduced"]
        write_json(OUT / "smoke_check.json", {
            "run_id": row["run_id"],
            "historical_comparison_available": "historical_metrics_reproduced" in row,
            "historical_metrics_reproduced": row.get("historical_metrics_reproduced"),
            "reload_predictions_verified": True})
    elif args.stage == "tests":
        completed = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "task2/tests",
                                    "-p", "test_biased_mf.py", "-v"], cwd=ROOT.parent,
                                   text=True, capture_output=True, check=False)
        with (OUT / "software_tests.log").open("x") as stream:
            stream.write(completed.stdout + completed.stderr)
        print(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError("Software tests failed")
        write_json(OUT / "software_tests.json", {"command": "python -m unittest discover -s task2/tests -p test_biased_mf.py -v",
                                                 "returncode": completed.returncode})
    else:
        import matplotlib
        matplotlib.use("Agg")
        {"capacity": stage_capacity, "interaction": stage_interaction,
         "rates": stage_rates, "seeds": stage_seeds, "summarize": verify_all}[args.stage](campaign)


if __name__ == "__main__":
    main()
