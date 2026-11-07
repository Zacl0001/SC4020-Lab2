"""Shared metrics for observed drug–target scores."""

import numpy as np
import pandas as pd


def _scores(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("Scores must be equal-length one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Scores must be finite")
    return actual, predicted


def mean_squared_error(actual, predicted):
    return float(np.mean((np.asarray(actual, dtype=float) - predicted) ** 2))


def concordance_index(actual, predicted):
    """Tie-aware CI in O(n log n); return NaN if no actual scores differ."""
    actual, predicted = _scores(actual, predicted)
    ranks = np.unique(predicted, return_inverse=True)[1] + 1
    order = np.argsort(actual, kind="stable")
    tree = np.zeros(int(ranks.max()) + 1 if len(ranks) else 1, dtype=np.int64)

    def count(rank):
        total = 0
        while rank > 0:
            total += int(tree[rank])
            rank -= rank & -rank
        return total

    points, comparable, seen = 0.0, 0, 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and actual[order[end]] == actual[order[start]]:
            end += 1
        # Query before inserting this group to exclude equal actual scores.
        for index in order[start:end]:
            rank = int(ranks[index])
            lower = count(rank - 1)
            equal = count(rank) - lower
            points += lower + 0.5 * equal
            comparable += seen
        for index in order[start:end]:
            rank = int(ranks[index])
            while rank < len(tree):
                tree[rank] += 1
                rank += rank & -rank
        seen += end - start
        start = end
    return points / comparable if comparable else float("nan")


def hit_at_k(candidates, predicted, k=1, drug_col="compound_iso_smiles",
             target_col="target_sequence", score_col="affinity"):
    """Macro Hit@K over ligands with >=1 observed candidate.

    Rank by prediction descending, then target ID as a string ascending.
    All targets tied for best actual score count as true best. Use min(k, n)
    candidates for a ligand with n candidates; singletons are eligible.
    Return the hit rate and eligible ligand count (NaN, 0 for empty input).
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    actual, predicted = _scores(candidates[score_col], predicted)
    if candidates[[drug_col, target_col]].isna().any().any():
        raise ValueError("Candidate IDs must be present")
    frame = candidates[[drug_col, target_col]].reset_index(drop=True).copy()
    frame[target_col] = frame[target_col].astype(str)
    if frame.duplicated([drug_col, target_col]).any():
        raise ValueError("Candidate drug–target pairs must be unique")
    frame["_prediction"] = predicted
    frame["_actual"] = actual
    # Actual scores are not sorting keys.
    best = frame.groupby(drug_col)["_actual"].transform("max")
    frame["_best"] = frame["_actual"].eq(best)
    ranked = frame.sort_values(["_prediction", target_col], ascending=[False, True])
    top = ranked.groupby(drug_col, sort=False).head(k)
    hits = top.groupby(drug_col)["_best"].any()
    return {"hit_rate": float(hits.mean()) if len(hits) else float("nan"),
            "eligible_ligands": len(hits)}
