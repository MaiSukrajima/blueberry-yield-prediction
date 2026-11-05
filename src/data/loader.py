#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一数据加载入口 —— 全组唯一的数据读取口。

它做三件事：
  1. 读原始 CSV（带 md5 指纹校验）
  2. 读 ``folds.csv``，并做**指纹守卫**：折与手上的数据对不上就直接报错
  3. 给出 ``iter_cv()``，保证每个人拿到的训练/验证划分完全一样

典型用法
--------
    import sys; sys.path.insert(0, ".")
    import numpy as np
    from src.data.loader import load_raw, load_folds, iter_cv, oof_mae, EXPECTED_FEATURES, TARGET

    df = load_raw()
    folds = load_folds(df)

    X = df[EXPECTED_FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    oof = np.full(len(df), np.nan)

    for tr_pos, va_pos in iter_cv(folds, k=5):
        model.fit(X[tr_pos], y[tr_pos])          # 预处理务必写在 Pipeline 里
        oof[va_pos] = model.predict(X[va_pos])

    print("OOF MAE =", oof_mae(y, oof))
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

CSV_NAME = "WildBlueberryPollinationSimulationData.csv"
CSV_MD5 = "feffa3af1cbb3c8860f4ae34241adc3d"

ID_COL = "Row#"
TARGET = "yield"

#: 唯一允许入模的 16 个特征列（顺序即契约顺序）
EXPECTED_FEATURES = [
    "clonesize", "honeybee", "bumbles", "andrena", "osmia",
    "MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
    "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange",
    "RainingDays", "AverageRainingDays",
    "fruitset", "fruitmass", "seeds",
]

DEFAULT_DATA_DIR = os.path.join("data", "raw")
DEFAULT_FOLDS = os.path.join("data", "processed", "folds.csv")


class DataContractError(RuntimeError):
    """数据与契约不符时抛出，宁可报错也不要静默跑出不可比的分数。"""


def md5_of_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def load_raw(data_dir: str = DEFAULT_DATA_DIR, csv_name: str = CSV_NAME,
             strict: bool = True) -> pd.DataFrame:
    """读原始 CSV 并校验指纹。"""
    path = os.path.join(data_dir, csv_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"缺少 {path}。数据由维护人随压缩包分发；也可运行 "
            f"`python src/data/fetch_dataset.py` 从 Kaggle 公开数据集重新获取。"
        )
    with open(path, "rb") as f:
        raw = f.read()
    got = md5_of_bytes(raw)
    if strict and got != CSV_MD5:
        raise DataContractError(
            f"{csv_name} 的 md5 为 `{got}`，与契约值 `{CSV_MD5}` 不一致。\n"
            f"说明你手上的文件被改过或不是同一份数据。\n"
            f"请重新解压数据包；确需在新数据上跑，请用 strict=False。"
        )

    import io
    df = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in EXPECTED_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise DataContractError(f"缺少契约列：{missing}")
    return df


def load_folds(df: pd.DataFrame | None = None,
               folds_path: str = DEFAULT_FOLDS,
               strict: bool = True) -> pd.DataFrame:
    """读折划分，并校验它与数据一致（指纹守卫）。"""
    if not os.path.isfile(folds_path):
        raise FileNotFoundError(
            f"缺少 {folds_path}。运行 `python src/data/make_folds.py` 生成。"
        )
    folds = pd.read_csv(folds_path)

    need = {"id", "fold_id", "split"}
    if not need.issubset(folds.columns):
        raise DataContractError(f"folds.csv 缺列：{need - set(folds.columns)}")

    meta_path = os.path.splitext(folds_path)[0] + ".meta.json"
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        if strict and meta.get("csv_md5") != CSV_MD5:
            raise DataContractError(
                f"folds.meta.json 记录的 csv_md5 是 `{meta.get('csv_md5')}`，"
                f"与契约值 `{CSV_MD5}` 不一致，说明折是在另一份数据上生成的。"
            )

    if df is not None:
        if len(folds) != len(df):
            raise DataContractError(
                f"折表 {len(folds)} 行，数据 {len(df)} 行，数量不匹配。"
            )
        if strict and not np.array_equal(
            np.sort(folds["id"].to_numpy()), np.sort(df[ID_COL].to_numpy())
        ):
            raise DataContractError("折表的 id 集合与数据的 Row# 集合不一致。")

        # 按数据顺序重排，使位置索引与 df 的行位置严格对应
        pos = pd.Series(np.arange(len(df)), index=df[ID_COL].to_numpy())
        folds = folds.copy()
        folds["_pos"] = pos.loc[folds["id"].to_numpy()].to_numpy()
        folds = folds.sort_values("_pos").reset_index(drop=True)

    return folds


def iter_cv(folds: pd.DataFrame, k: int = 5):
    """依次产出 (train_pos, valid_pos)，均为整数位置索引，可直接喂给 numpy 数组。

    使用位置索引而非 id，是为了避免 pandas 索引与 numpy 下标不一致这个常见坑。
    """
    fold_ids = sorted(int(v) for v in folds["fold_id"].unique() if v >= 0)
    if len(fold_ids) != k:
        raise DataContractError(f"折表里有 {len(fold_ids)} 个 CV 折，与 k={k} 不符")
    ids = folds["fold_id"].to_numpy()
    for f in fold_ids:
        va = np.where(ids == f)[0]
        tr = np.where((ids >= 0) & (ids != f))[0]   # CV 区其余四折，不含 holdout
        yield tr, va


def holdout_pos(folds: pd.DataFrame) -> np.ndarray:
    """holdout 的整数位置索引。"""
    return np.where(folds["fold_id"].to_numpy() == -1)[0]


def cv_pos(folds: pd.DataFrame) -> np.ndarray:
    """CV 区全部行的整数位置索引。"""
    return np.where(folds["fold_id"].to_numpy() >= 0)[0]


def oof_mae(y_true, oof) -> float:
    """统一分数口径：把各折验证预测拼成完整一列后整体算一次 MAE。

    ``oof`` 里未覆盖的位置（NaN）会被跳过；若覆盖率不足 100%，直接报错而不是给出乐观分数。
    """
    y_true = np.asarray(y_true, dtype=float)
    oof = np.asarray(oof, dtype=float)
    mask = ~np.isnan(oof)
    cov = mask.sum()
    if cov == 0:
        raise DataContractError("oof 全是 NaN，没有任何预测")
    if cov < len(oof):
        # 允许 holdout 段为 NaN，其余必须覆盖
        raise DataContractError(
            f"OOF 仅覆盖 {cov}/{len(oof)} 行（{cov/len(oof):.1%}）。"
            f"若 holdout 段为 NaN 属正常，请先用 cv_pos() 取出 CV 区再算。"
        )
    return float(np.mean(np.abs(y_true[mask] - oof[mask])))


def oof_mae_cv(y_true, oof, folds) -> float:
    """只在 CV 区计算 OOF MAE，holdout 段留 NaN 也不会报错。"""
    pos = cv_pos(folds)
    y_true = np.asarray(y_true, dtype=float)[pos]
    oof = np.asarray(oof, dtype=float)[pos]
    if np.isnan(oof).any():
        n = int(np.isnan(oof).sum())
        raise DataContractError(f"CV 区仍有 {n} 行未预测，折没跑完就出分")
    return float(np.mean(np.abs(y_true - oof)))


def fold_mae(y_true, oof, folds) -> pd.DataFrame:
    """逐折 MAE，用于报告「均值 ± 标准差」。"""
    y_true = np.asarray(y_true, dtype=float)
    oof = np.asarray(oof, dtype=float)
    ids = folds["fold_id"].to_numpy()
    rows = []
    for f in sorted(v for v in set(ids) if v >= 0):
        m = ids == f
        rows.append({"fold": int(f), "n": int(m.sum()),
                     "mae": float(np.mean(np.abs(y_true[m] - oof[m])))})
    return pd.DataFrame(rows)
