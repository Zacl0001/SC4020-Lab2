"""Jiaxin's constant training-mean model with canonical-ID input validation."""
import json
from pathlib import Path
import numpy as np
from shared_data import DRUG, PROTEIN, SCHEMA, ids, mapping_metadata


def fit_global_mean(training, dataset):
    ids(training, DRUG)
    ids(training, PROTEIN)
    if training.empty or not np.isfinite(training.affinity.to_numpy(dtype=float)).all():
        raise ValueError("Require nonempty finite training observations")
    metadata = {key: value.tolist() for key, value in mapping_metadata(training).items()
                if key not in ("drug_strings", "protein_strings")}
    return dict(model="Global Mean", dataset=dataset, training_mean=float(training.affinity.mean()),
                training_observations=len(training), score_scale="pKd" if dataset == "davis" else "supplied transformed KIBA",
                random_seed=42, canonical_schema=SCHEMA, **{k:v for k,v in metadata.items() if k != "canonical_schema"})


def predict_global_mean(model, pairs):
    ids(pairs, DRUG)
    ids(pairs, PROTEIN)
    return np.full(len(pairs), float(model["training_mean"]), dtype=float)


def save_global_mean(model, path):
    Path(path).write_text(json.dumps(model, indent=2, allow_nan=False)+"\n")


def load_global_mean(path):
    return json.loads(Path(path).read_text())
