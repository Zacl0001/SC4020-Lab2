"""Biased MF fitted to observed training pairs, with validation early stopping.

For N training observations the objective is
    (sum(error**2) + reg * sum(parameter**2)) / (2 * N).
The fixed training mean is not regularized. All biases and factors are.
A batch of B observations uses mean(data gradients) + reg/N * parameter.
Thus the regularization coefficient does NOT depend on B. The size-weighted
average of batch gradients equals the full gradient at fixed parameters,
including a smaller final batch. Batch size still changes Adam's trajectory
and number of updates per epoch; identical fitted models are not expected.

No label transformation, score clipping, matrix imputation, or cold-start
inference is performed. Unknown IDs raise ValueError. Ordered Unicode ID
arrays define the integer mappings and permit pickle-free checkpoints.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DRUG = "compound_iso_smiles"
PROTEIN = "target_sequence"
PAIR_KEYS = [DRUG, PROTEIN]
PARAMETERS = ("drug_bias", "target_bias", "drug_factors", "target_factors")
HISTORY_COLUMNS = ("epoch", "train_mse", "val_mse", "train_objective")


def _check_pairs(pairs, *, labels=False):
    required = PAIR_KEYS + (["affinity"] if labels else [])
    if not isinstance(pairs, pd.DataFrame) or not set(required) <= set(pairs.columns):
        raise ValueError(f"Expected a DataFrame with columns {required}")
    for column in PAIR_KEYS:
        if not pairs[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{column} must contain nonblank string IDs")
    if labels:
        if pairs.empty:
            raise ValueError("Training and validation observations must be nonempty")
        try:
            scores = pairs["affinity"].to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("Affinities must be finite numeric scores") from exc
        if not np.isfinite(scores).all():
            raise ValueError("Affinities must be finite numeric scores")
        return scores


def _indices(model, pairs):
    drug = pd.Index(model["drugs"]).get_indexer(pairs[DRUG])
    target = pd.Index(model["targets"]).get_indexer(pairs[PROTEIN])
    if (drug < 0).any() or (target < 0).any():
        raise ValueError(
            f"Unknown training IDs: {(drug < 0).sum()} drug rows, "
            f"{(target < 0).sum()} target rows; Biased MF requires known IDs"
        )
    return drug, target


def _predict_indices(model, drug, target):
    # Chunk gathers to avoid an observations-by-factors temporary for all pairs.
    result = np.empty(len(drug), dtype=np.float64)
    for start in range(0, len(drug), 8192):
        d, t = drug[start:start + 8192], target[start:start + 8192]
        result[start:start + 8192] = (
            float(model["global_mean"]) + model["drug_bias"][d]
            + model["target_bias"][t]
            + np.einsum("ij,ij->i", model["drug_factors"][d], model["target_factors"][t])
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite Biased MF predictions")
    return result


def _objective(model, drug, target, scores, regularization):
    residual = _predict_indices(model, drug, target) - scores
    penalty = sum(float(np.sum(model[key] ** 2)) for key in PARAMETERS)
    value = (float(residual @ residual) + regularization * penalty) / (2 * len(scores))
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite Biased MF objective")
    return value


def _batch_gradients(model, drug, target, scores, regularization, n_training):
    """Unbiased batch gradient of the documented full-training objective.

    np.bincount sums every occurrence of an ID, unlike fancy-index +=.
    Gradients for both factor matrices use the same pre-update parameters.
    L2 is coupled to the objective (ordinary Adam), not AdamW weight decay.
    """
    error = (_predict_indices(model, drug, target) - scores) / len(scores)
    nd, nt = len(model["drugs"]), len(model["targets"])
    gradients = {key: (regularization / n_training) * model[key] for key in PARAMETERS}
    gradients["drug_bias"] += np.bincount(drug, weights=error, minlength=nd)
    gradients["target_bias"] += np.bincount(target, weights=error, minlength=nt)
    p, q = model["drug_factors"][drug], model["target_factors"][target]
    for factor in range(p.shape[1]):
        gradients["drug_factors"][:, factor] += np.bincount(
            drug, weights=error * q[:, factor], minlength=nd)
        gradients["target_factors"][:, factor] += np.bincount(
            target, weights=error * p[:, factor], minlength=nt)
    return gradients


def _adam_update(model, gradients, first, second, step, learning_rate):
    for key in PARAMETERS:
        gradient = gradients[key]
        if not np.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite gradient: {key}")
        first[key] *= 0.9
        first[key] += 0.1 * gradient
        second[key] *= 0.999
        second[key] += 0.001 * gradient ** 2
        model[key] -= learning_rate * (first[key] / (1 - 0.9 ** step)) / (
            np.sqrt(second[key] / (1 - 0.999 ** step)) + 1e-8)
        if not np.isfinite(model[key]).all():
            raise FloatingPointError(f"Non-finite parameter: {key}")


def _validate_model(model):
    try:
        if int(model["format_version"]) != 1:
            raise ValueError("Unsupported Biased MF checkpoint version")
        for key in ("drugs", "targets"):
            ids = np.asarray(model[key])
            if (ids.ndim != 1 or not len(ids) or ids.dtype.kind != "U"
                    or len(np.unique(ids)) != len(ids)
                    or any(not value.strip() for value in ids)):
                raise ValueError(f"Invalid {key} mapping")
        rank = int(model["n_factors"])
        if rank < 1:
            raise ValueError("Invalid factor count")
        nd, nt = len(model["drugs"]), len(model["targets"])
        for key, shape in zip(PARAMETERS, ((nd,), (nt,), (nd, rank), (nt, rank))):
            array = np.asarray(model[key])
            if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(f"Invalid {key}: expected finite floating array {shape}")
        if not np.isfinite(float(model["global_mean"])):
            raise ValueError("Invalid global mean")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid or incomplete Biased MF checkpoint") from exc


def fit_biased_mf(training, validation, n_factors=16, learning_rate=0.01,
                  regularization=0.001, max_epochs=100, patience=10,
                  batch_size=2048, seed=42, dataset="unspecified",
                  score_scale="supplied affinity; unchanged"):
    """Fit using training labels only; restore minimum-validation-MSE epoch.

    Validation labels select the stopping/checkpoint epoch, never gradients.
    Epoch zero is eligible, so a worsening run returns its initialization.
    Returned history includes every attempted epoch, even after the best one.
    Scores remain on their supplied scale; global_mean stays fixed.
    """
    for name, value in (("n_factors", n_factors), ("max_epochs", max_epochs),
                        ("patience", patience), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    y = _check_pairs(training, labels=True)
    vy = _check_pairs(validation, labels=True)
    rng = np.random.default_rng(seed)
    drugs = np.sort(training[DRUG].unique()).astype(str)
    targets = np.sort(training[PROTEIN].unique()).astype(str)
    model = {
        "format_version": 1, "drugs": drugs, "targets": targets,
        "global_mean": float(np.mean(y)),
        "drug_bias": np.zeros(len(drugs)), "target_bias": np.zeros(len(targets)),
        "drug_factors": rng.normal(0, 0.01, (len(drugs), n_factors)),
        "target_factors": rng.normal(0, 0.01, (len(targets), n_factors)),
        "n_factors": n_factors, "learning_rate": learning_rate,
        "regularization": regularization, "max_epochs": max_epochs,
        "patience": patience, "batch_size": batch_size, "seed": seed,
        "training_observations": len(y), "validation_observations": len(vy),
        "dataset": str(dataset), "score_scale": str(score_scale),
        "label_transformation": "none; supplied split affinities unchanged",
        "optimizer": "Adam(beta1=0.9,beta2=0.999,epsilon=1e-8); coupled L2",
        "objective": "(sum_training_squared_error + regularization * sum_parameter_squares)/(2*N)",
        "unknown_id_policy": "error", "selection_metric": "validation MSE",
        "history_columns": np.asarray(HISTORY_COLUMNS),
    }
    d, t = _indices(model, training)
    vd, vt = _indices(model, validation)
    first = {key: np.zeros_like(model[key]) for key in PARAMETERS}
    second = {key: np.zeros_like(model[key]) for key in PARAMETERS}
    best, best_mse, best_epoch, stale, step = None, np.inf, 0, 0, 0
    history = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for epoch in range(max_epochs + 1):
                if epoch:
                    order = rng.permutation(len(y))
                    for start in range(0, len(y), batch_size):
                        rows = order[start:start + batch_size]
                        gradients = _batch_gradients(
                            model, d[rows], t[rows], y[rows], regularization, len(y))
                        step += 1
                        _adam_update(model, gradients, first, second, step, learning_rate)
                train_mse = float(np.mean((_predict_indices(model, d, t) - y) ** 2))
                val_mse = float(np.mean((_predict_indices(model, vd, vt) - vy) ** 2))
                objective = _objective(model, d, t, y, regularization)
                if not np.isfinite([train_mse, val_mse, objective]).all():
                    raise FloatingPointError("Non-finite epoch metrics")
                history.append((epoch, train_mse, val_mse, objective))
                if val_mse < best_mse:
                    best = {key: model[key].copy() for key in PARAMETERS}
                    best_mse, best_epoch, stale = val_mse, epoch, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
    except FloatingPointError as exc:
        raise FloatingPointError(
            f"Unstable Biased MF at epoch {epoch}; lr={learning_rate}, "
            f"rank={n_factors}, reg={regularization}. {exc}"
        ) from exc
    model.update(best)
    model.update(history=np.asarray(history), best_epoch=best_epoch,
                 best_val_mse=best_mse, epochs_run=epoch,
                 stop_reason="patience" if stale >= patience else "max_epochs")
    _validate_model(model)
    return model


def predict_biased_mf(model, pairs):
    """Return one finite score per row, using ONLY SMILES and sequence columns.

    Missing/unknown IDs fail clearly; no negative-index wraparound or fallback.
    Empty, correctly structured input returns shape (0,).
    """
    _validate_model(model)
    _check_pairs(pairs)
    drug, target = _indices(model, pairs)
    return _predict_indices(model, drug, target)


def save_biased_mf(model, path):
    """Save numeric/Unicode arrays only; callers create the parent directory."""
    _validate_model(model)
    arrays = {key: np.asarray(value) for key, value in model.items()}
    if any(value.dtype.kind not in "biufU" for value in arrays.values()):
        raise ValueError("Checkpoint fields must be numeric or Unicode, never objects")
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Checkpoint path must end in .npz")
    np.savez_compressed(path, **arrays)


def load_biased_mf(path):
    """Load without pickle and validate ID mappings and parameter dimensions."""
    with np.load(path, allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in saved.files}
    _validate_model(model)
    return model
