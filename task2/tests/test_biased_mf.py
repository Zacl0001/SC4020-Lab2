"""Focused correctness tests; all observations are synthetic.

Run: python -m unittest discover -s task2/tests -p 'test_biased_mf.py' -v
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from task2.utils import biased_mf as mf
from task2 import run_biased_mf as experiments


def synthetic_splits():
    rng = np.random.default_rng(123)
    p, q = rng.normal(size=(8, 2)), rng.normal(size=(7, 2))
    rows = [(f"d{i}", f"t{j}", 5 + .3 * i - .2 * j + .3 * p[i] @ q[j])
            for i in range(8) for j in range(7)]
    frame = pd.DataFrame(rows, columns=mf.PAIR_KEYS + ["affinity"])
    validation = np.arange(len(frame)) % 5 == 0
    return frame.loc[~validation].reset_index(drop=True), frame.loc[validation].reset_index(drop=True)


def gradient_fixture():
    rng = np.random.default_rng(12)
    model = {
        "drugs": np.array(["d0", "d1", "d2", "d3"]),
        "targets": np.array(["t0", "t1", "t2"]), "global_mean": 2.3,
        "drug_bias": rng.normal(0, .2, 4), "target_bias": rng.normal(0, .2, 3),
        "drug_factors": rng.normal(0, .3, (4, 2)),
        "target_factors": rng.normal(0, .3, (3, 2)),
    }
    # Repeated drugs and targets; d3 is absent, so its gradient is pure L2.
    drug = np.array([0, 0, 1, 1, 2, 2, 0])
    target = np.array([0, 1, 0, 2, 1, 2, 2])
    scores = np.array([1.3, 2.9, 2.1, 3.2, 1.8, 2.2, 3.4])
    return model, drug, target, scores


class GradientTests(unittest.TestCase):
    def test_central_finite_differences_full_and_repeated_id_batches(self):
        model, drug, target, scores = gradient_fixture()
        reg, epsilon = .37, 1e-6
        for indices in (np.arange(len(scores)), np.array([0, 1, 2, 6])):
            d, t, y = drug[indices], target[indices], scores[indices]
            actual = mf._batch_gradients(model, d, t, y, reg, len(scores))

            def objective():
                # Independent scalar reference, with global N in the penalty.
                errors = [model["global_mean"] + model["drug_bias"][i]
                          + model["target_bias"][j]
                          + sum(model["drug_factors"][i] * model["target_factors"][j]) - value
                          for i, j, value in zip(d, t, y)]
                return (sum(e * e for e in errors) / (2 * len(y))
                        + reg / (2 * len(scores))
                        * sum(np.sum(model[key] ** 2) for key in mf.PARAMETERS))

            for key in mf.PARAMETERS:
                numerical = np.zeros_like(model[key])
                for index in np.ndindex(model[key].shape):
                    original = model[key][index]
                    model[key][index] = original + epsilon
                    plus = objective()
                    model[key][index] = original - epsilon
                    minus = objective()
                    model[key][index] = original
                    numerical[index] = (plus - minus) / (2 * epsilon)
                np.testing.assert_allclose(actual[key], numerical, atol=1e-8, rtol=1e-6)
                error = np.max(np.abs(actual[key] - numerical))
                print(f"Gradient check B={len(y)} {key}: max absolute error={error:.3e}")

    def test_batch_size_weighted_gradient_and_l2_invariance(self):
        model, drug, target, scores = gradient_fixture()
        reg, n = .37, len(scores)
        full = mf._batch_gradients(model, drug, target, scores, reg, n)
        for batch_size in (1, 2, 3, 4, 7):
            aggregate = {key: np.zeros_like(model[key]) for key in mf.PARAMETERS}
            for start in range(0, n, batch_size):
                section = slice(start, start + batch_size)
                d, t, y = drug[section], target[section], scores[section]
                gradients = mf._batch_gradients(model, d, t, y, reg, n)
                unregularized = mf._batch_gradients(model, d, t, y, 0, n)
                for key in mf.PARAMETERS:
                    np.testing.assert_allclose(gradients[key] - unregularized[key],
                                               reg / n * model[key], atol=1e-15, rtol=1e-13)
                    aggregate[key] += len(y) / n * gradients[key]
            for key in mf.PARAMETERS:
                np.testing.assert_allclose(aggregate[key], full[key], atol=1e-15, rtol=1e-13)
        print("Batch-size/L2 invariance: B=1,2,3,4,7; weighted gradients agree within 1e-15 absolute tolerance")

    def test_first_adam_step_uses_all_preupdate_gradients(self):
        model, drug, target, scores = gradient_fixture()
        gradients = mf._batch_gradients(model, drug, target, scores, .37, len(scores))
        original = {key: model[key].copy() for key in mf.PARAMETERS}
        first = {key: np.zeros_like(model[key]) for key in mf.PARAMETERS}
        second = {key: np.zeros_like(model[key]) for key in mf.PARAMETERS}
        mf._adam_update(model, gradients, first, second, 1, .01)
        for key in mf.PARAMETERS:
            expected = original[key] - .01 * gradients[key] / (np.abs(gradients[key]) + 1e-8)
            np.testing.assert_allclose(model[key], expected, atol=1e-14)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.train, self.val = synthetic_splits()

    def fit(self, **kwargs):
        settings = dict(n_factors=3, max_epochs=8, patience=10, batch_size=11, seed=42)
        settings.update(kwargs)
        return mf.fit_biased_mf(self.train, self.val, **settings)

    def test_prediction_shape_finiteness_and_empty(self):
        model = self.fit()
        predictions = mf.predict_biased_mf(model, self.val[mf.PAIR_KEYS])
        self.assertEqual(predictions.shape, (len(self.val),))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(mf.predict_biased_mf(model, self.val.iloc[:0][mf.PAIR_KEYS]).shape, (0,))

    def test_predictions_ignore_labels(self):
        model = self.fit()
        poisoned = self.val.assign(affinity=np.nan)
        np.testing.assert_array_equal(mf.predict_biased_mf(model, poisoned),
                                      mf.predict_biased_mf(model, self.val[mf.PAIR_KEYS]))

    def test_validation_labels_never_change_training_updates(self):
        # Allow the same epochs regardless of validation quality. Returned best
        # checkpoints may legitimately differ; compare every actual update.
        original_update = mf._adam_update
        trajectories = []
        for validation in (self.val, self.val.assign(affinity=self.val.affinity + 100)):
            trajectory = []

            def record(model, *args):
                original_update(model, *args)
                trajectory.append(np.concatenate([model[key].ravel().copy() for key in mf.PARAMETERS]))

            with patch.object(mf, "_adam_update", side_effect=record):
                fitted = mf.fit_biased_mf(self.train, validation, n_factors=3,
                                         max_epochs=4, patience=10, batch_size=11)
            self.assertEqual(fitted["global_mean"], self.train.affinity.mean())
            np.testing.assert_array_equal(fitted["drugs"], np.sort(self.train[mf.DRUG].unique()))
            np.testing.assert_array_equal(fitted["targets"], np.sort(self.train[mf.PROTEIN].unique()))
            trajectories.append(trajectory)
        np.testing.assert_array_equal(trajectories[0], trajectories[1])

    def test_validation_cannot_expand_training_mapping(self):
        validation = self.val.copy()
        validation.loc[0, mf.DRUG] = "validation-only-ID"
        with self.assertRaisesRegex(ValueError, "Unknown training IDs"):
            mf.fit_biased_mf(self.train, validation, max_epochs=1)

    def test_seed_is_deterministic_and_inputs_are_unchanged(self):
        before_train, before_val = self.train.copy(deep=True), self.val.copy(deep=True)
        first, second = self.fit(), self.fit()
        for key in (*mf.PARAMETERS, "history"):
            np.testing.assert_array_equal(first[key], second[key])
        pd.testing.assert_frame_equal(before_train, self.train)
        pd.testing.assert_frame_equal(before_val, self.val)

    def test_pickle_free_round_trip(self):
        model = self.fit(dataset="synthetic", score_scale="unchanged synthetic scale")
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "model.npz"
            mf.save_biased_mf(model, path)
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype.kind != "O" for key in archive.files))
            loaded = mf.load_biased_mf(path)
            for key in model:
                np.testing.assert_array_equal(model[key], loaded[key])
            np.testing.assert_array_equal(mf.predict_biased_mf(model, self.val),
                                          mf.predict_biased_mf(loaded, self.val))

    def test_training_reduces_synthetic_loss(self):
        model = self.fit(max_epochs=100, patience=15, learning_rate=.02)
        history = model["history"]
        best = int(model["best_epoch"])
        self.assertLess(history[best, 1], .2 * history[0, 1])
        self.assertLess(history[best, 3], .2 * history[0, 3])

    def test_best_epoch_restoration_and_patience(self):
        snapshots = []
        real_update = mf._adam_update

        def record(model, *args):
            real_update(model, *args)
            snapshots.append({key: model[key].copy() for key in mf.PARAMETERS})

        # One update per epoch. Synthetic validation has a strict minimum at
        # epoch 2, then worsens; scores used for training remain real.
        original_predict = mf._predict_indices
        calls = 0

        def controlled_validation(model, drug, target):
            nonlocal calls
            if len(drug) == len(self.val):
                errors = [3., 2., 1., 2., 3., 4.]
                value = self.val.affinity.to_numpy() + errors[calls]
                calls += 1
                return value
            return original_predict(model, drug, target)

        with patch.object(mf, "_adam_update", side_effect=record), \
                patch.object(mf, "_predict_indices", side_effect=controlled_validation):
            model = self.fit(batch_size=1000, max_epochs=20, patience=2)
        self.assertEqual(model["best_epoch"], 2)
        self.assertEqual(model["epochs_run"], 4)
        self.assertEqual(model["stop_reason"], "patience")
        for key in mf.PARAMETERS:
            np.testing.assert_array_equal(model[key], snapshots[1][key])

    def test_unknown_and_missing_ids_fail_clearly(self):
        model = self.fit()
        for column in mf.PAIR_KEYS:
            for value in ("unseen", None, ""):
                with self.subTest(column=column, value=value):
                    bad = self.val.copy()
                    bad.loc[0, column] = value
                    with self.assertRaises(ValueError):
                        mf.predict_biased_mf(model, bad)

    def test_nonfinite_scores_parameters_and_bad_shapes_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for part in ("train", "val"):
                train, val = self.train.copy(), self.val.copy()
                (train if part == "train" else val).loc[0, "affinity"] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    mf.fit_biased_mf(train, val, max_epochs=1)
        model = self.fit()
        for key, value in (("drug_bias", np.zeros(1)),
                           ("global_mean", np.nan),
                           ("drugs", np.repeat(model["drugs"][:1], len(model["drugs"]))),
                           ("target_factors", np.full_like(model["target_factors"], np.inf))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                mf.predict_biased_mf({**model, key: value}, self.val)

    def test_invalid_hyperparameters_rejected(self):
        for settings in ({"learning_rate": np.inf}, {"learning_rate": 0},
                         {"regularization": -1}, {"regularization": np.nan},
                         {"n_factors": 0}, {"batch_size": 0}, {"patience": 0},
                         {"max_epochs": 1.5}, {"seed": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.fit(**settings)

    def test_numerical_overflow_fails_with_configuration(self):
        with self.assertRaisesRegex(FloatingPointError, "Unstable Biased MF at epoch"):
            self.fit(learning_rate=1e308)


class ExperimentTests(unittest.TestCase):
    def test_candidate_counts_and_supplementary_hit(self):
        counts = [1, 3, 5, 6, 9]
        frame = pd.DataFrame([(f"d{i}", f"t{j}", float(j))
                              for i, n in enumerate(counts) for j in range(n)],
                             columns=mf.PAIR_KEYS + ["affinity"])
        histogram, summary = experiments.candidate_distribution(frame, "synthetic")
        self.assertEqual(histogram.candidate_count.tolist(), counts)
        self.assertEqual(summary["singleton_ligands"], 1)
        self.assertEqual(summary["ligands_at_most_5"], 3)
        self.assertEqual(summary["ligands_more_than_5"], 2)
        with patch.object(experiments, "predict_biased_mf", side_effect=lambda m, p: frame.affinity.to_numpy()):
            row = experiments.evaluate_validation({}, frame, frame)
        self.assertEqual(row["val_hit_at_5"], 1)
        self.assertEqual(row["val_hit_at_5_candidates_gt_5"], 1)
        self.assertEqual(row["val_eligible_ligands_candidates_gt_5"], 2)

    def test_selection_uses_validation_mse_and_deterministic_ties(self):
        rows = [dict(status="ok", val_mse=mse, train_mse=train, n_factors=rank,
                     learning_rate=.01, regularization=.001)
                for mse, train, rank in [(2., .01, 8), (1., .5, 16), (1., .9, 8)]]
        best = experiments.select_best(rows)
        self.assertEqual(best["n_factors"], 8)
        self.assertEqual(best["train_mse"], .9)

    def test_search_reads_only_training_and_validation(self):
        train, val = synthetic_splits()
        real_read_csv, opened = pd.read_csv, []

        def guarded_read(path, *args, **kwargs):
            path = Path(path)
            self.assertIn(path.name, {f"{ds}_{part}.csv" for ds in ("davis", "kiba")
                                     for part in ("train", "val")})
            opened.append(path.name)
            return real_read_csv(path, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            splits = root / "data" / "splits"
            splits.mkdir(parents=True)
            for dataset in ("davis", "kiba"):
                train.to_csv(splits / f"{dataset}_train.csv", index=False)
                val.to_csv(splits / f"{dataset}_val.csv", index=False)
            # No test file exists; any attempted CSV access outside the four
            # allowed files fails. Exercise actual fitting, selection and saves.
            with patch.object(experiments.pd, "read_csv", side_effect=guarded_read), \
                    patch.object(experiments, "_curve"):
                selected = experiments.run_search(
                    root, grid=((2, .01, .001), (3, .005, .01)),
                    max_epochs=3, reuse_baseline=False)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(opened), 4)
            for dataset in ("davis", "kiba"):
                self.assertTrue((root / "saved_models" / f"{dataset}_biased_mf.npz").exists())


if __name__ == "__main__":
    unittest.main()
