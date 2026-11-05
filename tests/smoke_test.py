#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冒烟测试 —— 验证数据管道是否真的能跑通、接口是否守约。

两组测试：
  A. **离线合成数据**：不依赖真实数据，验证 loader / make_folds / oof_mae 的逻辑与守卫
  B. **真实数据**：若 ``data/raw/`` 与 ``data/processed/`` 就位，验证指纹、折完整性、基线可复现

用法
----
    python tests/smoke_test.py
    python tests/smoke_test.py --skip-real     # 只跑离线部分

退出码：0 = 全部通过；1 = 有失败。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data import loader as L  # noqa: E402

PASS = 0
FAIL = 0
FAILED = []


def check(name: str, cond, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"[FAIL] {name}  {detail}")


def make_synthetic(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in L.EXPECTED_FEATURES})
    df[L.ID_COL] = np.arange(n)
    df[L.TARGET] = df["fruitset"] * 5000 + rng.normal(scale=100, size=n) + 6000
    return df[L.ID_COL] .to_frame().join(df[L.EXPECTED_FEATURES + [L.TARGET]])


def test_offline(tmpdir: str):
    print("\n--- A. 离线合成数据 ---")
    df = make_synthetic()
    check("合成数据形状", df.shape == (200, len(L.EXPECTED_FEATURES) + 2), str(df.shape))
    check("合成数据无缺失", df.isna().sum().sum() == 0)

    # 指纹守卫：写入合成 CSV，strict 模式必须抛错
    d = os.path.join(tmpdir, "raw")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, L.CSV_NAME)
    df.to_csv(p, index=False)
    try:
        L.load_raw(d, strict=True)
        check("指纹守卫拦截篡改文件", False, "应抛 DataContractError 但没有")
    except L.DataContractError:
        check("指纹守卫拦截篡改文件", True)
    got = L.load_raw(d, strict=False)
    check("strict=False 可放行", len(got) == 200)

    # fold 表构造与 iter_cv
    ids = df[L.ID_COL].to_numpy()
    fold_id = np.array([-1] * 30 + [i % 5 for i in range(170)])
    folds = pd.DataFrame({"id": ids, "fold_id": fold_id,
                          "split": ["holdout" if f == -1 else "cv" for f in fold_id]})
    folds.to_csv(os.path.join(tmpdir, "folds.csv"), index=False)
    fr = L.load_folds(df, os.path.join(tmpdir, "folds.csv"), strict=False)

    check("load_folds 行数一致", len(fr) == len(df))
    check("load_folds 位置索引对齐", fr["_pos"].tolist() == list(range(len(df))))

    pairs = list(L.iter_cv(fr, k=5))
    check("iter_cv 产出 5 折", len(pairs) == 5, str(len(pairs)))
    covered = np.concatenate([va for _, va in pairs])
    check("iter_cv 验证集恰好覆盖 CV 区", len(covered) == 170 and len(set(covered)) == 170)
    check("iter_cv 验证集互不重叠", len(covered) == len(set(covered)))
    check("iter_cv 训练集不含 holdout",
          all((fold_id[tr] != -1).all() for tr, _ in pairs))
    check("iter_cv 训练集不含自身验证折",
          all(len(set(tr) & set(va)) == 0 for tr, va in pairs))

    check("holdout_pos 长度", len(L.holdout_pos(fr)) == 30)
    check("cv_pos 长度", len(L.cv_pos(fr)) == 170)

    # oof_mae
    y = df[L.TARGET].to_numpy()
    oof = np.full(len(y), np.nan)
    for tr, va in L.iter_cv(fr, k=5):
        oof[va] = y[tr].mean()
    check("oof_mae_cv 可计算", abs(L.oof_mae_cv(y, oof, fr)) < 5000)
    try:
        L.oof_mae(y, oof)
        check("oof_mae 覆盖率不足时报警", False, "应抛错但没有")
    except L.DataContractError:
        check("oof_mae 覆盖率不足时报警", True)

    fm = L.fold_mae(y, oof, fr)
    check("fold_mae 返回 5 行", len(fm) == 5, str(len(fm)))

    # 特征契约
    check("EXPECTED_FEATURES 为 16 列", len(L.EXPECTED_FEATURES) == 16)
    check("EXPECTED_FEATURES 不含标识列",
          L.ID_COL not in L.EXPECTED_FEATURES and "id" not in L.EXPECTED_FEATURES)
    check("EXPECTED_FEATURES 不含目标", L.TARGET not in L.EXPECTED_FEATURES)


def test_real():
    print("\n--- B. 真实数据 ---")
    raw = os.path.join(ROOT, "data", "raw")
    fp = os.path.join(ROOT, "data", "processed", "folds.csv")
    if not (os.path.isfile(os.path.join(raw, L.CSV_NAME)) and os.path.isfile(fp)):
        print("[SKIP] 真实数据或折划分不存在，跳过")
        return

    df = L.load_raw(raw)
    check("真实数据 777 行", len(df) == 777, str(len(df)))
    check("真实数据 18 列", df.shape[1] == 18, str(df.shape[1]))
    check("真实数据 0 缺失", df.isna().sum().sum() == 0)
    check("yield 极值吻合", abs(df[L.TARGET].min() - 1637.704022) < 1e-5
          and abs(df[L.TARGET].max() - 8969.401842) < 1e-5)

    folds = L.load_folds(df)
    check("折表 777 行", len(folds) == len(df))
    check("折表 id 唯一", folds["id"].is_unique)
    check("holdout 156 行", int((folds["fold_id"] == -1).sum()) == 156,
          str(int((folds["fold_id"] == -1).sum())))
    check("CV 区 621 行", int((folds["fold_id"] >= 0).sum()) == 621,
          str(int((folds["fold_id"] >= 0).sum())))
    check("fold_id 取值合法",
          set(folds["fold_id"].unique()) <= {0, 1, 2, 3, 4, -1})

    pairs = list(L.iter_cv(folds, k=5))
    check("真实折 5 折齐全", len(pairs) == 5)
    sizes = [len(va) for _, va in pairs]
    check("每折 124–126 行", all(123 <= s <= 126 for s in sizes), str(sizes))

    # 基线可复现（只跑 Ridge，快）
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[L.EXPECTED_FEATURES].to_numpy(dtype=float)
    y = df[L.TARGET].to_numpy(dtype=float)
    oof = np.full(len(y), np.nan)
    for tr, va in L.iter_cv(folds, k=5):
        m = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    mae = L.oof_mae_cv(y, oof, folds)
    check("Ridge OOF MAE ≈ 98.79", abs(mae - 98.79) < 1.0, f"{mae:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        test_offline(td)
    if not args.skip_real:
        test_real()

    print(f"\n合计：通过 {PASS} / 失败 {FAIL}")
    if FAIL:
        print("失败项：" + ", ".join(FAILED))
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
