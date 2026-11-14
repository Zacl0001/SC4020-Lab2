"""Helpers shared across the open-task notebooks."""


def create_model_name(dataset, holdout):
    """Checkpoint name for one training run, shared by the training and evaluation notebooks."""
    return f"{dataset}_holdout_{holdout}"
