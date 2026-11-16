"""Small behavioral tests, runnable with stdlib unittest (no extra dependency)."""
from pathlib import Path
import os
import pickle
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import EXPECTED_FEATURES, TARGET, load_folds, load_raw
from src.preprocessing import FeatureValidator, IQRClipper, build_preprocessor
from src.preprocessing.evaluate import evaluate_cv, validate_oof


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.default_rng(42).normal(size=(40, 16)),
                              columns=EXPECTED_FEATURES, index=np.arange(100, 140))

    def test_validation_extremes_do_not_change_fitted_parameters(self):
        for scaling in ("standard", "minmax"):
            with self.subTest(scaling=scaling):
                pipe = build_preprocessor(scaling, "iqr_clip").fit(self.X)
                clipper = pipe.named_steps["outliers"]
                scaler = pipe.named_steps["scale"]
                before_iqr = [getattr(clipper, a).copy() for a in
                              ("q1_", "q3_", "iqr_", "lower_bounds_", "upper_bounds_")]
                attrs = ("mean_", "var_", "scale_") if scaling == "standard" else (
                    "data_min_", "data_max_", "scale_", "min_")
                before_scale = [getattr(scaler, a).copy() for a in attrs]
                extreme = self.X.iloc[:5].copy() * 1e9
                pipe.transform(self.X.iloc[:5])
                pipe.transform(extreme)
                clipper.summarize(self.X, extreme)
                for attr, expected in zip(
                        ("q1_", "q3_", "iqr_", "lower_bounds_", "upper_bounds_"), before_iqr):
                    np.testing.assert_array_equal(getattr(clipper, attr), expected)
                for attr, expected in zip(attrs, before_scale):
                    np.testing.assert_array_equal(getattr(scaler, attr), expected)
                np.testing.assert_allclose(pipe.transform(self.X),
                                           build_preprocessor(scaling, "iqr_clip").fit_transform(self.X))

    def test_order_shape_and_input_immutability(self):
        original = self.X.copy(deep=True)
        for scaling in ("none", "standard", "minmax"):
            for method in ("none", "iqr_clip"):
                pipe = build_preprocessor(scaling, method).fit(self.X)
                whole = pipe.transform(self.X)
                shuffled = self.X.iloc[[8, 2, 17, 1]]
                actual = pipe.transform(shuffled.loc[:, EXPECTED_FEATURES[::-1]])
                self.assertEqual(actual.shape, (4, 16))
                np.testing.assert_allclose(actual, whole[[8, 2, 17, 1]])
                array = shuffled.to_numpy(copy=True)
                before = array.copy()
                np.testing.assert_allclose(pipe.transform(array), actual)
                np.testing.assert_array_equal(array, before)
        pd.testing.assert_frame_equal(self.X, original)
        none = build_preprocessor("none").fit_transform(self.X)
        none[0, 0] = 123456
        pd.testing.assert_frame_equal(self.X, original)

    def test_zero_iqr_preserves_valid_discrete_values(self):
        X = self.X.copy()
        X[EXPECTED_FEATURES[0]] = [0.] * 39 + [99.]
        clipper = IQRClipper().fit(X)
        self.assertTrue(clipper.skip_mask_[0])
        self.assertFalse(clipper.outlier_mask(X)[:, 0].any())
        valid = X.iloc[[39, 0]].copy()
        valid.iloc[0, 0] = 200
        np.testing.assert_array_equal(clipper.transform(valid)[:, 0], [200, 0])
        row = clipper.summarize(X, valid).iloc[0]
        self.assertEqual(row.train_outlier_count, 0)
        self.assertEqual(row.valid_outlier_count, 0)
        self.assertIn("IQR=0", row.skip_reason)

    def test_transform_is_independent_of_validation_batch(self):
        """One sample must get the same result alone and beside an extreme row."""
        for scaling in ("none", "standard", "minmax"):
            for method in ("none", "iqr_clip"):
                with self.subTest(scaling=scaling, method=method):
                    pipe = build_preprocessor(scaling, method).fit(self.X)
                    one = self.X.iloc[[3]].copy()
                    batch = pd.concat([one, one * 1e9, one * -1e9])
                    np.testing.assert_array_equal(pipe.transform(one)[0], pipe.transform(batch)[0])

    def test_iqr_boundaries_are_inclusive_and_audit_is_non_mutating(self):
        X = np.repeat(np.arange(5.)[:, None], 16, axis=1)
        clipper = IQRClipper().fit(X)
        boundaries = np.vstack([np.full(16, -2.), np.full(16, 6.)])
        before = boundaries.copy()
        self.assertFalse(clipper.outlier_mask(boundaries).any())
        np.testing.assert_array_equal(clipper.transform(boundaries), boundaries)
        audit = clipper.summarize(X, boundaries)
        audit.loc[:, "q1"] = -999
        np.testing.assert_array_equal(clipper.q1_, np.ones(16))
        np.testing.assert_array_equal(boundaries, before)

    def test_clipping_uses_training_fences(self):
        X = pd.DataFrame(np.repeat(np.arange(5.)[:, None], 16, axis=1), columns=EXPECTED_FEATURES)
        clipper = IQRClipper().fit(X)
        np.testing.assert_array_equal(clipper.q1_, np.ones(16))
        np.testing.assert_array_equal(clipper.q3_, np.full(16, 3.))
        valid = np.vstack([np.full(16, -100.), np.full(16, 100.)])
        np.testing.assert_array_equal(clipper.transform(valid),
                                      np.vstack([np.full(16, -2.), np.full(16, 6.)]))
        self.assertTrue(clipper.outlier_mask(valid).all())

    def test_rejects_forbidden_missing_extra_duplicate_and_invalid_values(self):
        invalid = [
            self.X.assign(**{"Row#": np.arange(len(self.X))}),
            self.X.assign(**{TARGET: 1}), self.X.drop(columns=EXPECTED_FEATURES[0]),
            self.X.assign(unexpected=1), self.X.assign(clonesize="1.0"),
            self.X.assign(clonesize=True), self.X.assign(clonesize=1+2j),
            self.X.assign(clonesize=np.nan), self.X.assign(clonesize=np.inf),
            self.X.assign(clonesize=-np.inf),
            self.X.assign(clonesize=pd.Series(pd.NA, index=self.X.index, dtype="Float64")),
            pd.concat([self.X, self.X.iloc[:, :1]], axis=1),
            np.zeros((3, 15)), np.zeros((3, 17)), np.zeros(16),
            np.full((3, 16), "1"), np.full((3, 16), np.inf),
        ]
        for X in invalid:
            for transformer in (FeatureValidator(), IQRClipper(), build_preprocessor()):
                with self.subTest(shape=X.shape, cls=type(transformer).__name__):
                    with self.assertRaises(ValueError):
                        transformer.fit(X)
                    fitted = clone(transformer).fit(self.X)
                    with self.assertRaises(ValueError):
                        fitted.transform(X)

    def test_clone_pickle_fitted_state_and_feature_names(self):
        for transformer in (FeatureValidator(), IQRClipper(), build_preprocessor(),
                            build_preprocessor("standard", "iqr_clip")):
            with self.assertRaises(NotFittedError):
                transformer.transform(self.X)
            fitted = clone(transformer).fit(self.X)
            restored = pickle.loads(pickle.dumps(fitted))
            np.testing.assert_array_equal(restored.transform(self.X), fitted.transform(self.X))
            np.testing.assert_array_equal(fitted.get_feature_names_out(), EXPECTED_FEATURES)
            with self.assertRaises(NotFittedError):
                clone(fitted).transform(self.X)
        model = Pipeline([("preprocess", build_preprocessor()), ("ridge", Ridge(alpha=1.0))])
        model.fit(self.X, np.arange(len(self.X)))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(self.X),
                                      model.predict(self.X))

    def test_scalers_match_sklearn_defaults_and_invalid_options_fail(self):
        for scaling, expected in (("standard", StandardScaler()), ("minmax", MinMaxScaler())):
            np.testing.assert_array_equal(build_preprocessor(scaling).fit_transform(self.X),
                                          expected.fit_transform(self.X.to_numpy()))
        minmax = build_preprocessor("minmax").fit(self.X)
        self.assertGreater(minmax.transform(self.X.iloc[:1] * 1000).max(), 1.)
        for kwargs in ({"scaling": "bad"}, {"outlier_method": "drop"},
                       {"iqr_factor": 0}, {"iqr_factor": -1}, {"iqr_factor": np.inf},
                       {"iqr_factor": np.nan}, {"iqr_factor": True}):
            with self.assertRaises(ValueError):
                build_preprocessor(**kwargs)


class RealCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get("BLUEBERRY_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
        cls.df = load_raw(str(root / "data/raw"))
        cls.folds = load_folds(cls.df, str(root / "data/processed/folds.csv"))

    def test_complete_oof_and_holdout_isolation(self):
        df, folds = self.df, self.folds
        result = evaluate_cv(df, folds)
        holdout = folds["fold_id"].to_numpy() == -1
        self.assertEqual(len(result.oof_predictions), 621)
        self.assertEqual(len(result.outlier_summary), 320)
        self.assertEqual(len(result.fold_metrics), 20)
        np.testing.assert_array_equal(result.oof_predictions["id"], df.loc[~holdout, "Row#"])
        # If evaluation inspects holdout features or labels, strict validation
        # or training will fail. Equivalent results prove their independence.
        poisoned = df.copy(deep=True)
        poisoned.loc[holdout, EXPECTED_FEATURES + [TARGET]] = np.nan
        isolated = evaluate_cv(poisoned, folds)
        for scheme, oof in result.oof.items():
            self.assertEqual(oof.shape, (777,))
            self.assertTrue(np.isfinite(oof[~holdout]).all())
            self.assertTrue(np.isnan(oof[holdout]).all())
            np.testing.assert_array_equal(oof, isolated.oof[scheme])
            damaged = oof.copy()
            damaged[np.flatnonzero(~holdout)[0]] = np.nan
            with self.assertRaises(ValueError):
                validate_oof(damaged, folds)
            damaged = oof.copy()
            damaged[np.flatnonzero(holdout)[0]] = 0
            with self.assertRaises(ValueError):
                validate_oof(damaged, folds)
        # Standard preprocessing is numerically equivalent to team baseline
        # composition, without calling the baseline script or writing reports.
        from src.data.loader import iter_cv, oof_mae_cv
        reference = np.full(777, np.nan)
        y = np.full(777, np.nan)
        y[~holdout] = df.loc[~holdout, TARGET]
        for tr, va in iter_cv(folds, k=5):
            model = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
            model.fit(df.iloc[tr][EXPECTED_FEATURES].to_numpy(), y[tr])
            reference[va] = model.predict(df.iloc[va][EXPECTED_FEATURES].to_numpy())
        np.testing.assert_allclose(result.oof["B"][~holdout], reference[~holdout], atol=1e-8)
        self.assertAlmostEqual(result.comparison.set_index("scheme").loc["B", "oof_mae"],
                               oof_mae_cv(y, reference, folds), places=8)

    def test_rejects_bad_fold_metadata_and_alignment(self):
        bad_tag = self.folds.copy()
        bad_tag.loc[0, "split"] = "wrong"
        bad_dtype = self.folds.copy()
        bad_dtype["fold_id"] = bad_dtype["fold_id"].astype(float)
        bad_size = self.folds.copy()
        pos = np.flatnonzero(bad_size["fold_id"].to_numpy() == 0)[0]
        bad_size.loc[pos, "fold_id"] = 1
        bad_alignment = self.folds.iloc[::-1].reset_index(drop=True)
        for folds in (bad_tag, bad_dtype, bad_size, bad_alignment, self.folds.drop(columns="split")):
            with self.assertRaises(ValueError):
                evaluate_cv(self.df, folds)


if __name__ == "__main__":
    unittest.main()
