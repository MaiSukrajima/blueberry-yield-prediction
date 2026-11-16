"""独立交付入口：只读引用数据组工程，测试和报告留在本交付目录。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import io
import os
from pathlib import Path
import re
import sys
import unittest

# Importing the team's loader must not create __pycache__ in its directory.
sys.dont_write_bytecode = True
DELIVERY_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = DELIVERY_ROOT.parent / "蓝莓数据层交付_v2.1_20261103"


def _snapshot(root):
    result = {}
    for parent, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d not in {".git", ".venv", "__pycache__"}]
        for name in files:
            path = Path(parent) / name
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _configure(project):
    if project == DELIVERY_ROOT:
        raise ValueError("独立入口需引用外部数据组工程；合入团队工程时直接运行预处理模块")
    required = ("src/data/loader.py", "requirements.txt", "data/processed/folds.csv",
                "data/processed/folds.meta.json", "data/raw/WildBlueberryPollinationSimulationData.csv",
                "reports/data_version.json", "reports/baselines.json")
    missing = [name for name in required if not (project / name).is_file()]
    if missing:
        raise FileNotFoundError(f"数据组目录缺少 {missing}；请用 --project-root 指定完整交付目录")
    expected = dict(re.findall(r"^([\w-]+)==([^\s#]+)",
                              (project / "requirements.txt").read_text(encoding="utf-8"), re.MULTILINE))
    actual = {name: version(name) for name in expected}
    mismatch = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
    if mismatch:
        raise RuntimeError(f"依赖版本不符合数据组锁定要求（要求值, 实际值）: {mismatch}；"
                           "请使用本目录 .venv 或按 requirements.txt 安装，不要升级原环境")
    # Local src is a namespace extension. src.preprocessing comes from this
    # directory; src.data.loader comes from the unmodified team directory.
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(DELIVERY_ROOT))
    os.environ["BLUEBERRY_PROJECT_ROOT"] = str(project)
    return actual


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _run_tests(out):
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(DELIVERY_ROOT / "tests"), pattern="test_preprocessing.py")
    result = unittest.TextTestRunner(stream=_Tee(sys.stdout, log), verbosity=2).run(suite)
    (out / "test_results.txt").write_text(log.getvalue(), encoding="utf-8")
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("测试失败或存在跳过项，请查看 reports/preprocessing/test_results.txt")
    return result.testsRun


def _check_reports(out):
    """Check saved CSVs independently, including reconstruction of metrics."""
    import numpy as np
    import pandas as pd

    comparison = pd.read_csv(out / "comparison.csv").set_index("scheme")
    per = pd.read_csv(out / "fold_metrics.csv")
    audit = pd.read_csv(out / "outlier_summary.csv")
    oof = pd.read_csv(out / "oof_predictions.csv")
    if (len(comparison) != 4 or len(per) != 20 or len(audit) != 320 or len(oof) != 621
            or oof["id"].duplicated().any() or (oof["fold_id"] < 0).any()
            or not np.isfinite(oof[["y_true", "pred_A", "pred_B", "pred_C", "pred_D"]]).all().all()):
        raise ValueError("导出表的规模、CV 范围或预测完整性不符合约定")
    for scheme in "ABCD":
        errors = oof["y_true"] - oof[f"pred_{scheme}"]
        actual = comparison.loc[scheme]
        np.testing.assert_allclose(actual["oof_mae"], errors.abs().mean(), atol=1e-9)
        np.testing.assert_allclose(actual["cv_rmse"], np.sqrt(np.mean(errors ** 2)), atol=1e-9)
        folds = per.loc[per["scheme"] == scheme].set_index("fold_id").sort_index()
        for fold_id, rows in oof.groupby("fold_id"):
            mae = (rows["y_true"] - rows[f"pred_{scheme}"]).abs().mean()
            np.testing.assert_allclose(folds.loc[fold_id, "mae"], mae, atol=1e-9)
            if folds.loc[fold_id, "n_valid"] != len(rows):
                raise ValueError("逐折指标的验证样本数不正确")
        np.testing.assert_allclose(actual["fold_mae_mean"], folds["mae"].mean(), atol=1e-9)
        np.testing.assert_allclose(actual["fold_mae_std"], folds["mae"].std(ddof=1), atol=1e-9)
    for part in ("train", "valid"):
        flagged = audit[f"{part}_outlier_count"]
        clipped = audit[f"{part}_clipped_count"]
        if ((flagged < 0) | (flagged > audit[f"n_{part}"])).any():
            raise ValueError("越界计数超出对应样本数")
        expected = flagged.where(audit["scheme"] == "D", 0)
        np.testing.assert_array_equal(clipped, expected)
        if (flagged[audit["iqr"] == 0] != 0).any():
            raise ValueError("IQR=0 的列不应标记或截断")
    # The same train fences must appear in each scheme, regardless of scaling.
    columns = ["q1", "q3", "iqr", "lower_bound", "upper_bound",
               "train_outlier_count", "valid_outlier_count"]
    tables = [audit[audit["scheme"] == s].set_index(["fold_id", "feature"])[columns].sort_index()
              for s in "ABCD"]
    for other in tables[1:]:
        pd.testing.assert_frame_equal(tables[0], other)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("all", "test", "evaluate"), default="all")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT,
                        help="数据组工程根目录，默认读取旁边的 v2.1 交付目录")
    args = parser.parse_args()
    project = args.project_root.resolve()
    actual_versions = _configure(project)
    before = _snapshot(project)
    out = DELIVERY_ROOT / "reports/preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    test_count = None
    try:
        if args.command in ("all", "test"):
            test_count = _run_tests(out)
        if args.command in ("all", "evaluate"):
            from src.preprocessing.evaluate import main as evaluate_main
            evaluate_main(["--project-root", str(project)])
            _check_reports(out)
    finally:
        if before != _snapshot(project):
            raise RuntimeError("检测到原数据组目录文件变化，请检查；此次核验不能标为通过")

    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    rows = ["# 独立交付核验记录", "", "负责人：李佳曦。", "",
            f"- 核验时间：{now}。", f"- 实际命令：`python run.py {args.command}`。",
            f"- 测试：{test_count} 项全部通过，无跳过。" if test_count is not None else "- 本次命令未运行测试。",
            f"- 数据组目录：`{project.name}`；运行前后 {len(before)} 个文件 SHA256 完全一致。",
            "- Python 导入关闭字节码写入，原目录不产生本模块的缓存。",
            "- 实际直接依赖：" + "、".join(f"{k} {v}" for k, v in actual_versions.items()) + "。"]
    if args.command in ("all", "evaluate"):
        rows.extend(["- 实验：4 组 × 5 折，全部完成。",
                     "- CSV 二次核对：4 条对比、20 条逐折指标、320 条特征异常统计、621 条 CV OOF。",
                     "- 从导出的预测重新计算总体 MAE、RMSE、逐折 MAE 和 ddof=1 标准差，均与报告一致。",
                     "- 四组的训练 IQR 阈值和诊断计数一致；只有 D 实际截断；IQR=0 的列未被标记。",
                     "- holdout 未训练、预测或评分；完整 OOF 和 NaN 隔离另由测试和评估器内部验证。"])
    rows.extend(["", "本文件由实际核验命令生成。测试详细输出见 test_results.txt；分数与解释见 summary.md。", ""])
    (out / "verification.md").write_text("\n".join(rows), encoding="utf-8")
    print(f"\n核验完成：数据组目录 {len(before)} 个文件未改变；结果位于 {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
