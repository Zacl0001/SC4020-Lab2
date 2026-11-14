"""Affinity model shared by the open-task notebooks."""

import numpy as np
import torch
from torch import nn


class AffinityPredictor(nn.Module):
    """Concatenate the frozen drug and protein vectors for a pair, then regress the affinity."""

    def __init__(self, drug_vectors, protein_vectors, hidden_sizes, dropout):
        super().__init__()
        self.register_buffer("drug_vectors", drug_vectors)
        self.register_buffer("protein_vectors", protein_vectors)

        layers = []
        size = drug_vectors.shape[1] + protein_vectors.shape[1]
        for hidden in hidden_sizes:
            layers += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout)]
            size = hidden
        layers.append(nn.Linear(size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, drug_ids, protein_ids):
        pair = torch.cat([self.drug_vectors[drug_ids], self.protein_vectors[protein_ids]], dim=1)
        return self.mlp(pair).squeeze(-1)


def predict(model, loader, device):
    """Return (predictions, targets) over the loader, in loader order."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for drug_ids, protein_ids, affinities in loader:
            output = model(drug_ids.to(device), protein_ids.to(device))
            predictions.append(output.cpu().numpy())
            targets.append(affinities.numpy())

    return np.concatenate(predictions), np.concatenate(targets)
