"""Drug-based CF: reuse item CF with the training matrix transposed."""

import numpy as np
from .protein_cf import DRUG, PROTEIN, fit_protein_cf, predict_protein_cf


# Build the protein-by-drug matrix and learn drug similarities and means from training data.
def fit_drug_cf(training, shrinkage=10.0):
    # Swapping columns makes rows proteins and items drugs.
    swapped = training.rename(columns={DRUG: PROTEIN, PROTEIN: DRUG})
    model = fit_protein_cf(swapped, shrinkage)
    model["drugs"], model["proteins"] = model["proteins"], model["drugs"]
    model["drug_means"] = model.pop("protein_means")
    model["orientation"] = "protein rows, drug columns"
    return model


# Predict using similar drugs; return affinity predictions and flags marking fallback use.
def predict_drug_cf(model, pairs, k=None, min_common=None):
    # Map drug items to the existing item-CF engine; labels are not used.
    item_model = dict(model)
    item_model["drugs"] = model["proteins"]
    item_model["proteins"] = model["drugs"]
    item_model["protein_means"] = model["drug_means"]
    swapped = pairs.rename(columns={DRUG: PROTEIN, PROTEIN: DRUG})
    return predict_protein_cf(item_model, swapped, k, min_common)


# Save the trained Drug CF model to a compressed .npz file.
def save_drug_cf(model, path):
    np.savez_compressed(path, **model)


# Load a saved Drug CF model so it can make predictions again.
def load_drug_cf(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}
