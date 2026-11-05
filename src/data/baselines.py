#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基线模型与门槛分数 —— 在统一折划分上实测，供全组对标。

跑的是「任何人都能复现」的最小集合：
    1. 训练折均值（常数基线，下界）
    2. Ridge（StandardScaler + alpha=1，线性参考）
    3. RandomForest（默认参数，树模型参考）
    4. HistGradientBoosting（默认参数，强基线）

所有预处理都写在 Pipeline 里，fit 只看当折训练集。

用法
----
    python src/data/baselines.py
    python src/data/baselines.py --json reports/baselines.json

输出：控制台表格 + 可选 JSON；同时写 ``reports/baselines.md``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.loader import (  # noqa: E402
    EXPECTED_FEATURES, TARGET, fold_mae, iter_cv, load_folds, load_raw, oof_mae_cv,
)


def run_model(name, make_model, X, y, folds, k):
    oof = np.full(len(y), np.nan)
    t0 = time.perf_counter()
    for tr, va in iter_cv(folds, k=k):
        m = make_model()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    fit_time = time.perf_counter() - t0

    score = oof_mae_cv(y, oof, folds)
    per = fold_mae(y, oof, folds)
    return {
        "model": name,
        "oof_mae": score,
        "fold_mean": float(per["mae"].mean()),
        "fold_std": float(per["mae"].std(ddof=1)),
        "per_fold": [float(v) for v in per["mae"]],
        "fit_seconds": fit_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join("data", "raw"))
    ap.add_argument("--folds", default=os.path.join("data", "processed", "folds.csv"))
    ap.add_argument("--out", default=os.path.join("reports", "baselines.md"))
    ap.add_argument("--json", default=os.path.join("reports", "baselines.json"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    df = load_raw(args.data_dir)
    folds = load_folds(df, args.folds)
    X = df[EXPECTED_FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    results = []

    # 常数基线：每个训练折的均值
    oof_const = np.full(len(y), np.nan)
    for tr, va in iter_cv(folds, k=args.k):
        oof_const[va] = y[tr].mean()
    per = fold_mae(y, oof_const, folds)
    results.append({
        "model": "常数基线（训练折均值）",
        "oof_mae": oof_mae_cv(y, oof_const, folds),
        "fold_mean": float(per["mae"].mean()),
        "fold_std": float(per["mae"].std(ddof=1)),
        "per_fold": [float(v) for v in per["mae"]],
        "fit_seconds": 0.0,
    })

    results.append(run_model(
        "Ridge（StandardScaler, alpha=1）",
        lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        X, y, folds, args.k))

    results.append(run_model(
        "RandomForest（默认, n_estimators=100）",
        lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        X, y, folds, args.k))

    results.append(run_model(
        "HistGradientBoosting（默认参数）",
        lambda: HistGradientBoostingRegressor(random_state=42),
        X, y, folds, args.k))

    base = results[0]["oof_mae"]
    L = []
    L.append("# 基线分数 baselines")
    L.append("")
    L.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 数据：{len(df)} 行 × {len(EXPECTED_FEATURES)} 特征")
    L.append(f"- 折：{args.k} 折 CV，seed=42，按 yield 十分位分层")
    L.append("")
    L.append("## 主表（OOF MAE，越低越好）")
    L.append("")
    L.append("| 模型 | OOF MAE | 相对常数基线 | 每折均值 ± 标准差 | 5 折耗时 |")
    L.append("| --- | --- | --- | --- | --- |")
    for r in results:
        rel = (r["oof_mae"] / base - 1) * 100
        rel_s = "—" if r is results[0] else f"{rel:+.1f}%"
        L.append(f"| {r['model']} | **{r['oof_mae']:.2f}** | {rel_s} | "
                 f"{r['fold_mean']:.2f} ± {r['fold_std']:.2f} | {r['fit_seconds']:.1f}s |")

    L.append("")
    L.append("## 逐折明细")
    L.append("")
    L.append("| 模型 | " + " | ".join(f"折{i}" for i in range(args.k)) + " |")
    L.append("| --- | " + " | ".join("---" for _ in range(args.k)) + " |")
    for r in results:
        L.append(f"| {r['model']} | " + " | ".join(f"{v:.1f}" for v in r["per_fold"]) + " |")

    L.append("")
    L.append("## 怎么用这张表")
    L.append("")
    ridge = results[1]["oof_mae"]
    L.append(f"- **门槛 1**：低于常数基线 `{base:.2f}` 才算模型学到了东西。")
    L.append(f"- **门槛 2**：低于 Ridge `{ridge:.2f}` 才算超过线性模型。"
             f"⚠️ 注意本数据 `fruitset`/`seeds`/`fruitmass` 与 yield 相关性达 0.93–0.98，"
             f"线性模型就能做得很好，**不要因为分数「看起来低」就以为有问题**。")
    L.append(f"- **门槛 3**：追求更优请对比 HistGradientBoosting `{results[3]['oof_mae']:.2f}`。")
    L.append("- 报分一律报 OOF MAE（拼接后整体算），不要报 5 个折 MAE 的算术平均。")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_rows": int(len(df)),
            "n_features": len(EXPECTED_FEATURES),
            "k": args.k,
            "seed": 42,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    for r in results:
        print(f"{r['model']:<42s} OOF MAE = {r['oof_mae']:8.2f}   "
              f"({r['fold_mean']:.2f} ± {r['fold_std']:.2f})")
    print(f"\n[写出] {args.out}\n[写出] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
