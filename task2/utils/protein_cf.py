"""Protein-based CF using observed training affinities only."""

import numpy as np
from shared_data import (DRUG, PROTEIN, SCHEMA, training_axes, positions, axis,
                         mapping_metadata)


def _validate_axes(model):
    if str(model.get("canonical_schema", "")) != SCHEMA:
        raise ValueError("Use a freshly fitted canonical-ID CF checkpoint")
    axis(model["drugs"], "drugs")
    axis(model["proteins"], "proteins")


def fit_protein_cf(training, shrinkage=10.0, n_drugs=None, n_proteins=None):
    drugs, proteins = training_axes(training, n_drugs, n_proteins)
    if not np.isfinite(training.affinity.to_numpy(dtype=float)).all():
        raise ValueError("Observed training affinities must be finite")
    matrix = training.pivot(index=DRUG, columns=PROTEIN, values="affinity").reindex(index=drugs, columns=proteins)
    values = matrix.to_numpy(dtype=float)
    observed = np.isfinite(values)
    counts = observed.astype(np.int64).T @ observed.astype(np.int64)
    # pandas computes pairwise Pearson using only jointly observed rows.
    correlations = matrix.corr(method="pearson", min_periods=2).to_numpy()
    similarities = np.nan_to_num(correlations, nan=0.0)
    similarities = np.clip(similarities, 0, 1) * counts / (counts + shrinkage)
    np.fill_diagonal(similarities, 0)
    return {
        "values": values,
        "drugs": drugs,
        "proteins": proteins,
        "canonical_schema": SCHEMA,
        **mapping_metadata(training),
        "similarities": similarities,
        "common_counts": counts,
        "protein_means": np.nanmean(values, axis=0),
        "global_mean": float(training["affinity"].mean()),
        "shrinkage": float(shrinkage),
    }


def predict_protein_cf(model, pairs, k=None, min_common=None):
    """Read numeric IDs only; unseen nonnegative IDs retain mean fallbacks."""
    _validate_axes(model)
    k = int(model["k"] if k is None else k)
    min_common = int(model["min_common"] if min_common is None else min_common)
    if k < 1 or min_common < 2:
        raise ValueError("Require k >= 1 and min_common >= 2")
    drugs = positions(pairs, DRUG, len(model["drugs"]), fallback=True)
    proteins = positions(pairs, PROTEIN, len(model["proteins"]), fallback=True)
    predictions = np.full(len(pairs), float(model["global_mean"]))
    fallback = np.ones(len(pairs), dtype=bool)
    known_target = proteins >= 0
    predictions[known_target] = model["protein_means"][proteins[known_target]]
    similarity = np.where(model["common_counts"] >= min_common, model["similarities"], 0)
    # Stable ties follow sorted protein IDs. Self-similarity stays zero.
    order = np.argsort(-similarity, axis=1, kind="stable")
    valid = np.flatnonzero((drugs >= 0) & known_target)
    for start in range(0, len(valid), 2048):
        rows = valid[start:start + 2048]
        targets = proteins[rows]
        neighbours = order[targets]
        affinities = model["values"][drugs[rows, None], neighbours]
        weights = similarity[targets[:, None], neighbours]
        suitable = np.isfinite(affinities) & (weights > 0)
        selected = suitable & (np.cumsum(suitable, axis=1) <= k)
        weights = np.where(selected, weights, 0)
        deviations = np.where(selected, affinities - model["protein_means"][neighbours], 0)
        total = weights.sum(axis=1)
        usable = total > 0
        adjustment = np.divide((weights * deviations).sum(axis=1), total,
                               out=np.zeros(len(rows)), where=usable)
        predictions[rows] += adjustment
        fallback[rows] = ~usable
    assert np.isfinite(predictions).all()
    return predictions, fallback


def save_protein_cf(model, path):
    _validate_axes(model)
    np.savez_compressed(path, **model)


def load_protein_cf(path):
    with np.load(path, allow_pickle=False) as saved:
        model = {name: saved[name].copy() for name in saved.files}
    _validate_axes(model)
    return model
