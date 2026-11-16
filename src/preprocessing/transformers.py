"""Strict, non-mutating feature validation and train-fitted preprocessing.

Arrays must already follow loader.EXPECTED_FEATURES order. Targets are never
inspected: the optional y argument only supports sklearn's estimator protocol.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data.loader import EXPECTED_FEATURES, ID_COL, TARGET


def _feature_array(X):
    """Validate without coercing strings, filling values, or dropping columns."""
    if isinstance(X, pd.DataFrame):
        forbidden = [c for c in (ID_COL, TARGET) if c in X.columns]
        if forbidden:
            raise ValueError(f"标识列/目标列不可进入预处理: {forbidden}; 请只传 EXPECTED_FEATURES")
        if X.columns.has_duplicates:
            raise ValueError("特征列名重复，请提供唯一的 16 个 EXPECTED_FEATURES")
        missing = [c for c in EXPECTED_FEATURES if c not in X.columns]
        extra = [c for c in X.columns if c not in EXPECTED_FEATURES]
        if missing or extra:
            raise ValueError(f"特征列不符合约定: 缺列={missing}, 多余列={extra}")
        bad = [c for c in EXPECTED_FEATURES
               if not is_numeric_dtype(X[c].dtype)
               or is_bool_dtype(X[c].dtype) or is_complex_dtype(X[c].dtype)]
        if bad:
            raise ValueError(f"特征必须为实数数值类型，非数值/布尔/复数列: {bad}")
        values = X.loc[:, EXPECTED_FEATURES].to_numpy(
            dtype=np.float64, copy=True, na_value=np.nan)
    elif isinstance(X, np.ndarray):
        if X.ndim != 2 or X.shape[1] != len(EXPECTED_FEATURES):
            raise ValueError(f"数组必须为二维且恰好 16 列，实际 shape={X.shape}")
        if X.dtype.kind not in "iuf":
            raise ValueError("数组必须为实数数值类型，不接受字符串、object、布尔或复数")
        values = X.astype(np.float64, copy=True)
    else:
        raise TypeError("X 必须为 pandas DataFrame 或二维 numpy 数组")
    if not len(values):
        raise ValueError("X 至少需要一行样本")
    bad_columns = np.flatnonzero(~np.isfinite(values).all(axis=0))
    if len(bad_columns):
        names = [EXPECTED_FEATURES[i] for i in bad_columns]
        raise ValueError(f"特征含 NaN 或 inf（含 -inf）: {names}; 不会自动填补或替换")
    return values


def _validate_iqr_factor(iqr_factor):
    if (isinstance(iqr_factor, (bool, np.bool_))
            or not isinstance(iqr_factor, Real)
            or not np.isfinite(iqr_factor) or iqr_factor <= 0):
        raise ValueError("iqr_factor 必须为有限的正实数")


class FeatureValidator(TransformerMixin, BaseEstimator):
    """Enforce the 16-column contract and return an independent ordered array."""

    def fit(self, X, y=None):
        values = _feature_array(X)
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(EXPECTED_FEATURES, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        return _feature_array(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is not None and not np.array_equal(
                np.asarray(input_features, dtype=object), self.feature_names_in_):
            raise ValueError("input_features 必须按 EXPECTED_FEATURES 的约定顺序")
        return self.feature_names_in_.copy()


class IQRClipper(FeatureValidator):
    """Learn IQR fences on fit; skip zero-IQR columns in flagging and clipping."""

    def __init__(self, iqr_factor=1.5):
        self.iqr_factor = iqr_factor

    def fit(self, X, y=None):
        _validate_iqr_factor(self.iqr_factor)
        values = _feature_array(X)
        self.q1_, self.q3_ = np.quantile(values, [0.25, 0.75], axis=0, method="linear")
        self.iqr_ = self.q3_ - self.q1_
        self.lower_bounds_ = self.q1_ - self.iqr_factor * self.iqr_
        self.upper_bounds_ = self.q3_ + self.iqr_factor * self.iqr_
        self.skip_mask_ = self.iqr_ == 0
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(EXPECTED_FEATURES, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_", "skip_mask_"])
        values = _feature_array(X)
        active = ~self.skip_mask_
        values[:, active] = np.clip(values[:, active], self.lower_bounds_[active],
                                    self.upper_bounds_[active])
        return values

    def outlier_mask(self, X):
        """Flags from stored training fences; this method never fits or clips."""
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_", "skip_mask_"])
        values = _feature_array(X)
        return ((values < self.lower_bounds_) | (values > self.upper_bounds_)) & ~self.skip_mask_

    def summarize(self, X_train, X_valid):
        """Counts refer to original, unscaled feature values and train fences."""
        train_flags = self.outlier_mask(X_train)
        valid_flags = self.outlier_mask(X_valid)
        return pd.DataFrame({
            "feature": EXPECTED_FEATURES,
            "n_train": len(X_train), "n_valid": len(X_valid),
            "q1": self.q1_, "q3": self.q3_, "iqr": self.iqr_,
            "lower_bound": self.lower_bounds_, "upper_bound": self.upper_bounds_,
            "train_outlier_count": train_flags.sum(axis=0),
            "valid_outlier_count": valid_flags.sum(axis=0),
            "skip_reason": np.where(self.skip_mask_, "IQR=0：跳过标记与截断", ""),
        })


def build_preprocessor(scaling="standard", outlier_method="none", iqr_factor=1.5):
    """Return a cloneable Pipeline: strict validation -> optional clip -> scale.

    No operation drops/reorders rows or changes the number of features. All
    numerical parameters are learned by fit, never recomputed by transform.
    MinMaxScaler retains sklearn defaults, including clip=False.
    """
    if scaling not in ("standard", "minmax", "none"):
        raise ValueError("scaling 必须为 'standard'、'minmax' 或 'none'")
    if outlier_method not in ("none", "iqr_clip"):
        raise ValueError("outlier_method 必须为 'none' 或 'iqr_clip'")
    _validate_iqr_factor(iqr_factor)
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
              "none": "passthrough"}[scaling]
    return Pipeline([
        ("validate", FeatureValidator()),
        ("outliers", IQRClipper(iqr_factor) if outlier_method == "iqr_clip" else "passthrough"),
        ("scale", scaler),
    ])
