"""Drug-based CF: reuse item CF with the training matrix transposed."""

import numpy as np
from .protein_cf import DRUG, PROTEIN, fit_protein_cf, predict_protein_cf, _validate_axes
from shared_data import training_axes


def fit_drug_cf(training, shrinkage=10.0, n_drugs=None, n_proteins=None):
    swapped = training.rename(columns={DRUG: PROTEIN, PROTEIN: DRUG})
    drugs, proteins = training_axes(training, n_drugs, n_proteins)
    swapped.attrs = dict(training.attrs)
    for a, b in (("n_drugs", "n_proteins"), ("drug_strings", "protein_strings"),
                 ("drug_mapping_sha256", "protein_mapping_sha256")):
        if a in swapped.attrs and b in swapped.attrs:
            swapped.attrs[a], swapped.attrs[b] = swapped.attrs[b], swapped.attrs[a]
    model = fit_protein_cf(swapped, shrinkage, n_drugs=len(proteins), n_proteins=len(drugs))
    model["drugs"], model["proteins"] = model["proteins"], model["drugs"]
    for a, b in (("drug_strings", "protein_strings"), ("drug_mapping_sha256", "protein_mapping_sha256")):
        if a in model and b in model:
            model[a], model[b] = model[b], model[a]
    model["drug_means"] = model.pop("protein_means")
    model["orientation"] = "protein rows, drug columns"
    return model


def predict_drug_cf(model, pairs, k=None, min_common=None):
    item_model = dict(model)
    item_model["drugs"] = model["proteins"]
    item_model["proteins"] = model["drugs"]
    item_model["protein_means"] = model["drug_means"]
    swapped = pairs.rename(columns={DRUG: PROTEIN, PROTEIN: DRUG})
    return predict_protein_cf(item_model, swapped, k, min_common)


def save_drug_cf(model, path):
    _validate_axes(model)
    np.savez_compressed(path, **model)


def load_drug_cf(path):
    with np.load(path, allow_pickle=False) as saved:
        model = {name: saved[name].copy() for name in saved.files}
    _validate_axes(model)
    return model
