#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""折划分生成：777 行 → 5 折 CV（80%）+ 冻结 holdout（20%）。

为什么必须由脚本统一生成
------------------------
8 个人各自 ``train_test_split``，验证集不同，分数就不可比。折必须唯一、可复现、可审计。

协议
----
- 全量 777 行先按 ``yield`` 十分位分层切出 **holdout**（``test_size=0.20``，约 156 行）
- 剩余约 621 行做 **5 折分层交叉验证**（每折验证约 124 行）
- 随机种子统一 ``42``；分层依据为 ``yield`` 的 ``qcut(q=10)`` 分箱
- 幂等：绑定 CSV 的 md5，输入不变则跳过；``--force`` 强制重算

用法
----
    python src/data/make_folds.py
    python src/data/make_folds.py --force
    python src/data/make_folds.py --holdout-frac 0.15 --n-splits 5

输出
----
``data/processed/folds.csv``（id / fold_id / split）与 ``folds.meta.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

CSV_NAME = "WildBlueberryPollinationSimulationData.csv"
CSV_MD5 = "feffa3af1cbb3c8860f4ae34241adc3d"
ID_COL = "Row#"
TARGET = "yield"
SEED = 42
VERSION = "folds-777-v2.0"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("data", "raw", CSV_NAME))
    ap.add_argument("--out", default=os.path.join("data", "processed", "folds.csv"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[FAIL] 找不到 {args.csv}")
        print("       数据由维护人随压缩包分发，或运行 python src/data/fetch_dataset.py")
        return 1

    cur_md5 = md5_of(args.csv)
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"

    # 幂等：参数与输入都没变就跳过
    if os.path.isfile(args.out) and os.path.isfile(meta_path) and not args.force:
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            same = (
                meta.get("csv_md5") == cur_md5
                and meta.get("n_splits") == args.n_splits
                and abs(meta.get("holdout_frac", -1) - args.holdout_frac) < 1e-9
                and meta.get("seed") == args.seed
            )
            if same:
                print(f"[SKIP] 折已存在且参数一致（{args.out}），如需重算加 --force")
                return 0
        except Exception:
            pass

    if cur_md5 != CSV_MD5:
        print(f"[WARN] CSV md5 `{cur_md5}` 与契约值 `{CSV_MD5}` 不同，折将按当前文件重算")

    df = pd.read_csv(args.csv)
    if ID_COL not in df.columns or TARGET not in df.columns:
        print(f"[FAIL] 缺少 {ID_COL} 或 {TARGET} 列")
        return 1
    if df[ID_COL].duplicated().any():
        print(f"[FAIL] {ID_COL} 存在重复，不能作为对齐键")
        return 1

    y = df[TARGET].to_numpy()
    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")

    # L2：冻结 holdout
    idx = df.index.to_numpy()
    cv_idx, ho_idx = train_test_split(
        idx, test_size=args.holdout_frac, random_state=args.seed, stratify=bins
    )

    # L1：CV 内 5 折
    fold_id = pd.Series(-1, index=df.index, dtype=int)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for f, (_, va) in enumerate(skf.split(cv_idx, bins[cv_idx])):
        fold_id.loc[cv_idx[va]] = f

    if (fold_id.loc[ho_idx] != -1).any():
        print("[FAIL] holdout 行被误划入 CV 折")
        return 1

    out = pd.DataFrame({
        "id": df[ID_COL].to_numpy(),
        "fold_id": fold_id.to_numpy(),
        "split": ["holdout" if f == -1 else "cv" for f in fold_id.to_numpy()],
    })

    # 完整性自检
    assert len(out) == len(df), "折表行数与数据不一致"
    assert out["id"].is_unique, "折表 id 不唯一"
    assert set(out["fold_id"].unique()) <= set(range(args.n_splits)) | {-1}, "出现非法 fold_id"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    sizes = {int(k): int(v) for k, v in out["fold_id"].value_counts().sort_index().items()}
    meta = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "csv": os.path.basename(args.csv),
        "csv_md5": cur_md5,
        "n_rows": int(len(df)),
        "id_col": ID_COL,
        "target": TARGET,
        "n_splits": args.n_splits,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "stratify": "yield qcut(10)",
        "sizes": {"holdout": int((out["fold_id"] == -1).sum()),
                  "cv_total": int((out["fold_id"] >= 0).sum()),
                  "per_fold": sizes},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 共 {len(out)} 行 → CV {meta['sizes']['cv_total']} 行 / "
          f"holdout {meta['sizes']['holdout']} 行")
    print(f"     每折大小：{sizes}")
    print(f"[写出] {args.out}")
    print(f"[写出] {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
