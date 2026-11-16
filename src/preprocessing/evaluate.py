"""Fixed Ridge preprocessing comparison on existing CV folds; never use holdout.

Standalone delivery: python run.py evaluate
Integrated team project: python -m src.preprocessing.evaluate
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.data.loader import (
    CSV_NAME, EXPECTED_FEATURES, ID_COL, TARGET,
    fold_mae, iter_cv, load_folds, load_raw, oof_mae_cv,
)
from .transformers import IQRClipper, build_preprocessor


SCHEMES = (
    ("A", "none", "none"),
    ("B", "standard", "none"),
    ("C", "minmax", "none"),
    ("D", "standard", "iqr_clip"),
)


@dataclass
class EvaluationResult:
    comparison: pd.DataFrame
    fold_metrics: pd.DataFrame
    outlier_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof: dict[str, np.ndarray]


def validate_oof(oof, folds):
    """Fail before scoring if CV coverage or holdout isolation is broken."""
    values = np.asarray(oof)
    cv_mask = folds["fold_id"].to_numpy() >= 0
    if values.shape != (len(folds),):
        raise ValueError("OOF 必须是一维数组，长度与 folds 相同")
    if not np.isfinite(values[cv_mask]).all():
        raise ValueError("CV 区 OOF 预测必须全部有限且完整覆盖")
    if not np.isnan(values[~cv_mask]).all():
        raise ValueError("holdout 的 OOF 位置必须保持 NaN")


def evaluate_cv(df, folds):
    """Only index CV feature/target values, even though load_raw reads the CSV.

    Full-length y is an alignment buffer containing NaN at every holdout row.
    Neither feature validation nor descriptive statistics is run on holdout.
    """
    required = {"id", "fold_id", "split"}
    if not required.issubset(folds.columns):
        raise ValueError(f"折表缺少必需列: {sorted(required - set(folds.columns))}")
    if not pd.api.types.is_integer_dtype(folds["fold_id"].dtype):
        raise ValueError("fold_id 必须是整数类型")
    ids = folds["fold_id"].to_numpy()
    if (len(df) != 777 or len(folds) != 777
            or set(ids) != {-1, 0, 1, 2, 3, 4}
            or np.count_nonzero(ids >= 0) != 621
            or np.count_nonzero(ids == -1) != 156
            or folds["id"].duplicated().any()
            or not np.array_equal(folds["id"].to_numpy(), df[ID_COL].to_numpy())):
        raise ValueError("数据/折对齐不符合固定 777 行、621 CV + 156 holdout 的约定")
    expected_sizes = {-1: 156, 0: 125, 1: 124, 2: 124, 3: 124, 4: 124}
    if folds["fold_id"].value_counts().to_dict() != expected_sizes:
        raise ValueError("各折行数与既有固定折划分不一致，不得重新划分")
    if not np.array_equal(folds["split"].to_numpy(), np.where(ids >= 0, "cv", "holdout")):
        raise ValueError("折表 split 标记与 fold_id 不一致")
    cv = np.flatnonzero(ids >= 0)
    y = np.full(len(df), np.nan)
    y[cv] = df.iloc[cv][TARGET].to_numpy(dtype=float)
    if not np.isfinite(y[cv]).all():
        raise ValueError("CV 区 yield 必须为有限数值；目标不做填补或变换")

    prediction_table = pd.DataFrame({
        "id": folds.iloc[cv]["id"].to_numpy(),
        "fold_id": ids[cv], "y_true": y[cv],
    })
    comparisons, per_fold_tables, audits, predictions = [], [], [], {}
    for scheme, scaling, method in SCHEMES:
        oof = np.full(len(df), np.nan)
        coverage = np.zeros(len(df), dtype=int)
        fit_seconds, predict_seconds = 0.0, 0.0
        for tr, va in iter_cv(folds, k=5):
            # Select rows first: never inspect holdout features or labels.
            X_train = df.iloc[tr].loc[:, EXPECTED_FEATURES]
            X_valid = df.iloc[va].loc[:, EXPECTED_FEATURES]
            model = Pipeline([
                ("preprocess", build_preprocessor(scaling, method, iqr_factor=1.5)),
                ("model", Ridge(alpha=1.0)),
            ])
            start = time.perf_counter()
            model.fit(X_train, y[tr])
            fit_seconds += time.perf_counter() - start
            start = time.perf_counter()
            oof[va] = model.predict(X_valid)
            predict_seconds += time.perf_counter() - start
            coverage[va] += 1

            # D reuses the clipper actually fitted inside this fold's pipeline.
            # A/B/C fit a diagnostic-only checker on the same training features.
            if method == "iqr_clip":
                checker = model.named_steps["preprocess"].named_steps["outliers"]
            else:
                checker = IQRClipper(iqr_factor=1.5).fit(X_train)
            audit = checker.summarize(X_train, X_valid)
            audit.insert(0, "fold_id", int(ids[va[0]]))
            audit.insert(0, "scheme", scheme)
            audit["outlier_method"] = method
            audit["iqr_factor"] = 1.5
            # Flag counts are diagnostic even in schemes that keep all values.
            audit["train_clipped_count"] = audit["train_outlier_count"] if method == "iqr_clip" else 0
            audit["valid_clipped_count"] = audit["valid_outlier_count"] if method == "iqr_clip" else 0
            audits.append(audit)

        validate_oof(oof, folds)
        if not ((coverage[cv] == 1).all() and (coverage[ids == -1] == 0).all()):
            raise ValueError("每条 CV 记录必须恰好预测一次，holdout 不得预测")
        score = oof_mae_cv(y, oof, folds)
        per = fold_mae(y, oof, folds).rename(columns={"fold": "fold_id", "n": "n_valid"})
        per.insert(0, "scheme", scheme)
        per["n_train"] = len(cv) - per["n_valid"]
        per_fold_tables.append(per)
        comparisons.append({
            "scheme": scheme, "scaling": scaling, "outlier_method": method,
            "iqr_factor": 1.5, "ridge_alpha": 1.0,
            "oof_mae": score, "cv_rmse": float(np.sqrt(np.mean((y[cv] - oof[cv]) ** 2))),
            "fold_mae_mean": float(per["mae"].mean()),
            "fold_mae_std": float(per["mae"].std(ddof=1)),
            "std_ddof": 1, "n_cv": len(cv), "n_holdout": int(np.sum(ids == -1)),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        })
        prediction_table[f"pred_{scheme}"] = oof[cv]
        predictions[scheme] = oof
    return EvaluationResult(pd.DataFrame(comparisons), pd.concat(per_fold_tables, ignore_index=True),
                            pd.concat(audits, ignore_index=True), prediction_table, predictions)


def _fingerprint(path, algorithm="sha256"):
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


def _code_revision(root):
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                               text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "当前交付目录没有可用 Git commit；使用下列源文件 SHA256 追溯。"
    return f"代码基于 Git commit `{revision}`；工作区{'有未提交改动，精确代码以下列 SHA256 为准' if dirty else '干净'}。"


def write_reports(result, project_root, out_dir):
    """Write only this module's reports; raw data and team reports are read-only."""
    project_root, out_dir = Path(project_root), Path(out_dir)
    # Prevent accidental overwrites of another team's existing output area.
    code_root = Path(__file__).resolve().parents[2]
    expected_output = (code_root / "reports" / "preprocessing").resolve()
    if out_dir.resolve() != expected_output:
        raise ValueError("输出目录必须是本模块交付目录内 reports/preprocessing/")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("comparison", "fold_metrics", "outlier_summary", "oof_predictions"):
        getattr(result, name).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    version = json.loads((project_root / "reports/data_version.json").read_text(encoding="utf-8"))
    fold_meta = json.loads((project_root / "data/processed/folds.meta.json").read_text(encoding="utf-8"))
    raw_path = project_root / "data/raw" / CSV_NAME
    fold_path = project_root / "data/processed/folds.csv"
    baseline = json.loads((project_root / "reports/baselines.json").read_text(encoding="utf-8"))
    references = [r for r in baseline["results"] if r["model"].startswith("Ridge")]
    standard_score = float(result.comparison.set_index("scheme").loc["B", "oof_mae"])
    delta_clip = float(result.comparison.set_index("scheme").loc["D", "oof_mae"]) - standard_score
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    lines = [
        "# 数据清洗与预处理实验结果", "", "负责人：李佳曦。", "",
        f"- 运行时间：{now}（Asia/Shanghai）",
        f"- 数据版本：`{version['data_version']}`；折版本：`{fold_meta['version']}`。",
        f"- 数据组输入目录（只读）：`{os.path.relpath(project_root, code_root)}`。",
        "- 输出保存在本任务交付目录的 `reports/preprocessing/`，不写入数据组目录。",
        "- 数据规模：777 行 × 18 列；16 个特征。现有五折：621 CV，156 holdout；seed=42。",
        f"- 原始文件：`data/raw/{CSV_NAME}`；{raw_path.stat().st_size} 字节。",
        f"- 原始文件 MD5：`{_fingerprint(raw_path, 'md5')}`（load_raw strict=True 校验）。",
        f"- 原始文件 SHA256：`{_fingerprint(raw_path)}`。",
        f"- folds.csv SHA256：`{_fingerprint(fold_path)}`。",
        f"- 原 loader.py SHA256：`{_fingerprint(project_root / 'src/data/loader.py')}`。",
        f"- 原 requirements.txt SHA256：`{_fingerprint(project_root / 'requirements.txt')}`。",
        "- " + _code_revision(code_root),
    ]
    for relative in ("src/preprocessing/__init__.py", "src/preprocessing/transformers.py",
                     "src/preprocessing/evaluate.py"):
        lines.append(f"  - `{relative}`：`{_fingerprint(code_root / relative)}`")
    lines.extend([
        "", "## 实际运行环境", "",
        f"Python {platform.python_version()}；numpy {np.__version__}；pandas {pd.__version__}；"
        f"scikit-learn {sklearn.__version__}；scipy {scipy.__version__}。",
        "依赖遵循 requirements.txt；Python 生成环境原为 3.13.9，实际解释器版本以上述记录为准。",
        "", "## 固定 Ridge(alpha=1.0) 的五折结果", "",
        "每折重新建立完整 Pipeline；顺序：输入校验 → 可选 IQR 截断 → 缩放 → Ridge。",
        "总体 OOF MAE 由 oof_mae_cv 计算；逐折 MAE 由 fold_mae 计算；标准差 ddof=1。",
        "总体 OOF MAE 与逐折 MAE 均值是两个不同统计量，后者只用于报告波动。", "",
        "| 方案 | 缩放 | 截断 | OOF MAE | CV RMSE | 折 MAE 均值 ± 标准差 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for r in result.comparison.to_dict("records"):
        lines.append(f"| {r['scheme']} | {r['scaling']} | {r['outlier_method']} | "
                     f"{r['oof_mae']:.6f} | {r['cv_rmse']:.6f} | "
                     f"{r['fold_mae_mean']:.6f} ± {r['fold_mae_std']:.6f} |")
    lines.extend(["", "| 方案 | 折0 MAE | 折1 MAE | 折2 MAE | 折3 MAE | 折4 MAE |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for scheme, _, _ in SCHEMES:
        scores = result.fold_metrics.query("scheme == @scheme").sort_values("fold_id")["mae"]
        lines.append(f"| {scheme} | " + " | ".join(f"{x:.6f}" for x in scores) + " |")
    if references:
        ref = float(references[0]["oof_mae"])
        lines.extend(["", f"交付 baselines.json 中 Ridge 参考值为 {ref:.9f}；"
                      f"本次 B 为 {standard_score:.9f}，差值 {standard_score-ref:+.9f}。",
                      "参考值仅用于核对复现，未作为计算结果或训练参数。"])
    if references and abs(standard_score - ref) > 0.1:
        lines.append("参考复现存在偏差：请结合上述实际版本、输入列顺序、文件指纹和折内处理顺序核查；未修改数据追分。")
    lines.extend([
        "", "## 默认规则与异常值解释", "",
        "默认保留全部有限、有效样本，不删除行、不填补、不编码、不处理 yield。"
        "线性模型默认 standard；树模型通常用 none；缩放不消除温度特征共线性。",
        "IQR 检查使用训练折的 Q1/Q3（线性插值），边界为 Q1−1.5×IQR 与 Q3+1.5×IQR。"
        "严格小于下界或大于上界才标记；验证折仅应用训练阈值。",
        "仿真中的固定网格、低频蜂密度或温度取值可能合理。统计规则标记的极端值不等于确认的数据错误。",
        "IQR=0 的列跳过标记与截断，计数为 0 并给出 skip_reason；报告中的上下界保留理论值供审计，实际不应用。",
        f"D 相对 B 的 OOF MAE 差为 {delta_clip:+.6f}（正数表示变差）；"
        + ("截断未改善，本项目保留不截断作为默认。" if delta_clip >= 0 else
           "截断在这次固定配置下有改善，但不能据此确认数据有误，仍保留不截断作为默认。"),
        "alpha 固定时缩放也改变了 Ridge 对各原始特征的有效惩罚强度；这些结果只适用于该配置，"
        "不能宣称某种缩放方法对所有模型最优。", "",
        "## 输出与隔离验证", "",
        "四个内存 OOF 数组均长 777：621 个 CV 位置各有且只有一次有限预测，156 个 holdout 位置始终 NaN。",
        "oof_predictions.csv 仅含 621 行 CV：id 对应原始 Row#，另含 fold_id、y_true、pred_A/B/C/D。",
        "outlier_summary.csv 每方案×5 折×16 特征，共 320 行；先在原始特征上统计，再截断/缩放。"
        "A/B/C 的阈值只是诊断；实际截断数量为 0。各方案的相同阈值和计数不能相加当作独立样本数。",
        "holdout 未参与本次实验：load_raw 按既有接口载入文件并核验指纹，之后只索引 CV 特征和标签；"
        "没有对 holdout 做分布统计、预处理拟合、训练、预测或评分。holdout_usage.md 未改动。",
        "未输出全局清洗后数据集；原始 CSV、folds、loader、队友基线和数据组报告均保持原样。", "",
        "独立目录复现：`python run.py evaluate`；测试：`python run.py test`；完整核验：`python run.py all`。",
        "合入团队工程后也可使用 `python -m src.preprocessing.evaluate`。", "",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    code_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path,
                        default=Path(os.environ.get("BLUEBERRY_PROJECT_ROOT", code_root)),
                        help="原数据组工程根目录；独立交付请使用 run.py 自动配置")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    df = load_raw(str(root / "data/raw"))
    folds = load_folds(df, str(root / "data/processed/folds.csv"))
    result = evaluate_cv(df, folds)
    write_reports(result, root, code_root / "reports/preprocessing")
    print(result.comparison.to_string(index=False))
    print("\n已生成 reports/preprocessing/ 下五个报告；holdout 未使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
