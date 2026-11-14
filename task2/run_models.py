"""Refit the four selected models into a new directory using train/validation only."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared_data import PAIR_KEYS, load_task2_split
from .run_biased_mf import config, FIXED, SCORE_SCALES
from .utils import global_mean, protein_cf, drug_cf, biased_mf, evaluation

MODELS = {"global_mean": global_mean, "protein_cf": protein_cf,
          "drug_cf": drug_cf, "biased_mf": biased_mf}
NEIGHBORS = {"protein_cf": {"davis": 10, "kiba": 5},
             "drug_cf": {"davis": 5, "kiba": 10}}


def run_models(output_dir, datasets=("davis", "kiba"), knn_search=False):
    """Use the published settings and existing fitters; never overwrite a run."""
    if not datasets or any(dataset not in SCORE_SCALES for dataset in datasets):
        raise ValueError("Choose davis and/or kiba")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = output_dir / "saved_models"
    checkpoints.mkdir()
    rows, search_rows = [], []
    for dataset in datasets:
        parts = {split: load_task2_split(dataset, split) for split in ("train", "val")}
        for kind, module in MODELS.items():
            if kind == "global_mean":
                model = global_mean.fit_global_mean(parts["train"], dataset)
            elif kind == "biased_mf":
                model = biased_mf.fit_biased_mf(parts["train"], parts["val"],
                    **config(dataset, 128, .001, 1.), **FIXED, score_scale=SCORE_SCALES[dataset])
            else:
                model = getattr(module, "fit_" + kind)(parts["train"], shrinkage=10.)
                model.update(k=NEIGHBORS[kind][dataset], min_common=2,
                             dataset=dataset, score_scale=SCORE_SCALES[dataset])
                if knn_search:
                    validation = parts["val"]
                    for k in (5, 10, 20, 30):
                        for minimum in (2, 5, 10):
                            predicted, fallback = getattr(module, "predict_" + kind)(
                                model, validation[PAIR_KEYS], k=k, min_common=minimum)
                            search_rows.append(dict(dataset=dataset, model=kind, k=k, min_common=minimum,
                                validation_mse=evaluation.mean_squared_error(validation.affinity, predicted),
                                fallback_fraction=float(fallback.mean())))
                    best = min(search_rows[-12:], key=lambda row: (row["validation_mse"], row["k"], row["min_common"]))
                    model.update(k=best["k"], min_common=best["min_common"])
            suffix = ".json" if kind == "global_mean" else ".npz"
            path = checkpoints / f"{dataset}_{kind}{suffix}"
            getattr(module, "save_" + kind)(model, path)
            restored = getattr(module, "load_" + kind)(path)
            for split, frame in parts.items():
                prediction = getattr(module, "predict_" + kind)(model, frame[PAIR_KEYS])
                reloaded = getattr(module, "predict_" + kind)(restored, frame[PAIR_KEYS])
                if isinstance(prediction, tuple):
                    np.testing.assert_array_equal(prediction[1], reloaded[1])
                    prediction, reloaded = prediction[0], reloaded[0]
                np.testing.assert_array_equal(prediction, reloaded)
                row = dict(dataset=dataset, model=kind, split=split,
                    mse=evaluation.mean_squared_error(frame.affinity, prediction),
                    ci=evaluation.concordance_index(frame.affinity, prediction),
                    hit_at_1=evaluation.hit_at_k(frame, prediction, 1)["hit_rate"],
                    hit_at_5=evaluation.hit_at_k(frame, prediction, 5)["hit_rate"],
                    checkpoint=str(path.relative_to(output_dir)))
                if kind == "biased_mf":
                    row.update(best_epoch=int(model["best_epoch"]), stopping_epoch=int(model["epochs_run"]))
                elif kind in NEIGHBORS:
                    row.update(k=int(model["k"]), min_common=int(model["min_common"]))
                rows.append(row)
            print(f"Fitted and reloaded {dataset}/{kind}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "model_comparison.csv", index=False)
    if search_rows:
        pd.DataFrame(search_rows).to_csv(output_dir / "knn_validation_search.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for checkpoints and comparison CSV")
    parser.add_argument("--dataset", choices=("all", "davis", "kiba"), default="all")
    parser.add_argument("--knn-search", action="store_true", help="Reproduce the K/minimum-overlap validation grid")
    args = parser.parse_args()
    datasets = ("davis", "kiba") if args.dataset == "all" else (args.dataset,)
    print(run_models(args.output_dir, datasets, knn_search=args.knn_search).to_string(index=False))


if __name__ == "__main__":
    main()
