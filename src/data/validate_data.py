#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据完整性校验：缺列 / 类型 / 缺失 / 重复 / 取值域 / 共线性 / 指纹。

校验对象是 Kaggle 公开数据集 Wild Blueberry Yield 的原始 CSV：
    777 行 × 18 列 = Row# + 16 个特征 + yield

用法
----
    python src/data/validate_data.py
    python src/data/validate_data.py --data-dir data/raw --out reports

退出码：0 = 无失败项（可能有警告）；1 = 存在失败项。
输出：``reports/data_profile.md``
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

CSV_NAME = "WildBlueberryPollinationSimulationData.csv"
CSV_MD5 = "feffa3af1cbb3c8860f4ae34241adc3d"
N_ROWS = 777

ID_COL = "Row#"
TARGET = "yield"

FEATURES = [
    "clonesize", "honeybee", "bumbles", "andrena", "osmia",
    "MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
    "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange",
    "RainingDays", "AverageRainingDays",
    "fruitset", "fruitmass", "seeds",
]
EXPECTED_COLS = [ID_COL] + FEATURES + [TARGET]

# 已知的离散取值特征（来自模拟实验设计网格）
DISCRETE = ["clonesize", "honeybee", "bumbles", "andrena", "osmia",
            "MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
            "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange",
            "RainingDays", "AverageRainingDays"]

TEMP_COLS = ["MaxOfUpperTRange", "MinOfUpperTRange", "AverageOfUpperTRange",
             "MaxOfLowerTRange", "MinOfLowerTRange", "AverageOfLowerTRange"]

YIELD_MIN = 1637.704022
YIELD_MAX = 8969.401842


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, name, detail):
        self.items.append((level, name, detail))

    def ok(self, name, detail):   self.add("PASS", name, detail)
    def warn(self, name, detail): self.add("WARN", name, detail)
    def fail(self, name, detail): self.add("FAIL", name, detail)

    def counts(self):
        c = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for lv, _, _ in self.items:
            c[lv] += 1
        return c


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--csv", default=None, help="直接指定文件，默认取 data-dir 下的标准文件名")
    args = ap.parse_args()

    path = args.csv or os.path.join(args.data_dir, CSV_NAME)
    r = Report()

    # ---- 0. 文件与指纹
    if not os.path.isfile(path):
        print(f"[FAIL] 找不到 {path}")
        print("       先从 Kaggle 公开数据集获取：python src/data/fetch_dataset.py")
        return 1
    got = md5_of(path)
    if got == CSV_MD5:
        r.ok("文件指纹", f"md5 `{got}` 与 Kaggle 数据集原件一致")
    else:
        r.fail("文件指纹", f"md5 `{got}` ≠ 契约值 `{CSV_MD5}`，文件被改动过")

    df = pd.read_csv(path)

    # ---- 1. 形状
    if df.shape == (N_ROWS, len(EXPECTED_COLS)):
        r.ok("行列规模", f"{df.shape[0]} 行 × {df.shape[1]} 列")
    else:
        r.fail("行列规模", f"实际 {df.shape[0]}×{df.shape[1]}，应为 {N_ROWS}×{len(EXPECTED_COLS)}")

    # ---- 2. 列名与顺序
    if list(df.columns) == EXPECTED_COLS:
        r.ok("列名与顺序", "18 列与契约完全一致")
    else:
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_COLS]
        if missing or extra:
            r.fail("列名与顺序", f"缺列 {missing}；多列 {extra}")
        else:
            r.warn("列名与顺序", "列集合一致，但顺序与契约不同")

    # ---- 3. 类型
    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        r.fail("列类型", f"非数值列 {non_num}")
    else:
        r.ok("列类型", "18 列全为数值型，无需类别编码")

    # ---- 4. 缺失
    nmiss = int(df.isna().sum().sum())
    if nmiss == 0:
        r.ok("缺失值", "全表 0 缺失")
    else:
        r.fail("缺失值", f"共 {nmiss} 个缺失")

    # ---- 5. 重复行
    ndup = int(df.duplicated().sum())
    if ndup == 0:
        r.ok("完全重复行", "0 条")
    else:
        r.warn("完全重复行", f"{ndup} 条完全重复")

    # ---- 6. 数值有限性
    num = df[FEATURES + [TARGET]].to_numpy(dtype=float)
    if np.isfinite(num).all():
        r.ok("数值有限性", "无 inf / -inf")
    else:
        r.fail("数值有限性", "存在 inf 或 -inf")

    # ---- 7. Row# 性质
    if ID_COL in df.columns:
        s = df[ID_COL]
        uniq = s.nunique()
        is_cont = (uniq == len(s)) and (s.min() == 0) and (s.max() == len(s) - 1)
        if is_cont:
            r.ok("Row# 性质", f"0..{len(s)-1} 连续整数，{uniq} 个唯一值 = 行数，即原始行号")
        else:
            r.warn("Row# 性质", f"{uniq} 个唯一值（行数 {len(s)}），非 0..n-1 连续")
        r.ok("Row# 用法", "仅为行标识，不入模：它唯一对应每一行，入模等于把答案的钥匙交给模型")

    # ---- 8. 目标变量
    if TARGET in df.columns:
        y = df[TARGET].astype(float)
        if abs(y.min() - YIELD_MIN) < 1e-5 and abs(y.max() - YIELD_MAX) < 1e-5:
            r.ok("yield 极值", f"min={y.min():.6f} max={y.max():.6f}，与原件一致")
        else:
            r.warn("yield 极值", f"min={y.min():.6f} max={y.max():.6f}，与契约值有出入")
        if (y <= 0).any():
            r.fail("yield 有效性", f"存在 {int((y <= 0).sum())} 个非正值")
        else:
            r.ok("yield 有效性", f"全为正，均值 {y.mean():.4f} 标准差 {y.std():.4f}")

    # ---- 9. 常量列
    const_cols = [c for c in FEATURES if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        r.fail("常量列", f"{const_cols} 只有单一取值，无信息量")
    else:
        r.ok("常量列", "16 个特征均有取值变化")

    # ---- 10. 离散特征取值域
    disc_info = []
    for c in DISCRETE:
        if c in df.columns:
            disc_info.append((c, df[c].nunique(), float(df[c].min()), float(df[c].max())))
    bad = [c for c, k, _, _ in disc_info if k > 50]
    if disc_info and bad:
        r.warn("离散特征", f"{bad} 取值种类 >50，可能不是实验设计网格值")
    elif disc_info:
        r.ok("离散特征", f"{len(disc_info)} 列取值种类 3–{max(k for _, k, _, _ in disc_info)}，符合模拟实验设计")

    # ---- 11. 温度六列共线
    corr_t = df[TEMP_COLS].corr().abs()
    np.fill_diagonal(corr_t.values, 0)
    mx = float(corr_t.values.max())
    pair = corr_t.stack().idxmax()
    if mx > 0.95:
        r.warn("温度共线", f"最高 |r|={mx:.4f}（{pair[0]} vs {pair[1]}），建议降维或交给树模型")
    else:
        r.ok("温度共线", f"最高 |r|={mx:.4f}，未达 0.95 预警线")

    # ---- 12. 特征-目标相关性
    corrs = [(c, float(df[[c, TARGET]].corr().iloc[0, 1])) for c in FEATURES]
    corrs.sort(key=lambda x: -abs(x[1]))
    top = corrs[0]
    if abs(top[1]) > 0.9:
        r.warn("特征相关", f"`{top[0]}` 与 yield 的 |r|={abs(top[1]):.4f} 偏高，注意是否属泄漏型特征")
    else:
        r.ok("特征相关", f"最强特征 `{top[0]}` |r|={abs(top[1]):.4f}")

    # ---- 13. 量纲差异
    span = df[FEATURES].max() - df[FEATURES].min()
    ratio = float(span.max() / max(span.min(), 1e-12))
    if ratio > 100:
        r.warn("特征量纲", f"各特征极差相差 {ratio:.0f} 倍，线性/神经网络方案必须先标准化")
    else:
        r.ok("特征量纲", f"各特征极差相差 {ratio:.1f} 倍")

    # ================================ 输出
    c = r.counts()
    os.makedirs(args.out, exist_ok=True)
    prof = os.path.join(args.out, "data_profile.md")

    L = []
    L.append("# 数据画像 data_profile")
    L.append("")
    L.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 文件：`{os.path.basename(path)}`（{os.path.getsize(path):,} 字节）")
    L.append(f"- 形状：**{df.shape[0]} 行 × {df.shape[1]} 列**")
    L.append(f"- md5：`{got}`")
    L.append(f"- 校验汇总：**通过 {c['PASS']} / 警告 {c['WARN']} / 失败 {c['FAIL']}**")
    L.append("")
    L.append("## 1. 逐项校验")
    L.append("")
    L.append("| 级别 | 检查项 | 说明 |")
    L.append("| --- | --- | --- |")
    for lv, name, detail in r.items:
        mark = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[lv]
        L.append(f"| {mark} {lv} | {name} | {detail} |")

    L.append("")
    L.append("## 2. 逐列统计")
    L.append("")
    L.append("| 列 | 角色 | 类型 | 非空 | 唯一值 | 最小 | 中位 | 最大 | 均值 | 标准差 |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for col in EXPECTED_COLS:
        role = "标识" if col == ID_COL else ("目标" if col == TARGET else "特征")
        s = df[col]
        L.append(
            f"| `{col}` | {role} | {s.dtype} | {int(s.notna().sum())} | {s.nunique()} | "
            f"{s.min():.6g} | {s.median():.6g} | {s.max():.6g} | {s.mean():.6g} | {s.std():.6g} |"
        )

    L.append("")
    L.append("## 3. 特征与 yield 的相关性（Pearson，按绝对值降序）")
    L.append("")
    L.append("| 排名 | 特征 | r |")
    L.append("| --- | --- | --- |")
    for i, (col, v) in enumerate(corrs, 1):
        L.append(f"| {i} | `{col}` | {v:+.4f} |")

    L.append("")
    L.append("## 4. 结论")
    L.append("")
    if c["FAIL"] == 0:
        L.append("**全部结构校验通过（无失败项），可用于建模。**")
        if c["WARN"]:
            L.append("")
            L.append("警告项不阻塞使用，但写报告时需说明：")
            L.append("")
            for lv, name, detail in r.items:
                if lv == "WARN":
                    L.append(f"- **{name}**：{detail}")
    else:
        L.append("**存在失败项，禁止用于建模，先按上表修复。**")

    with open(prof, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    for lv, name, detail in r.items:
        mark = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[lv]
        print(f"{mark} {name}: {detail}")
    print()
    print(f"合计：通过 {c['PASS']} / 警告 {c['WARN']} / 失败 {c['FAIL']}")
    print(f"[写出] {prof}")
    return 0 if c["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
