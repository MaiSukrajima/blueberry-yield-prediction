# 蓝莓产量预测基线模型交付

本 PR 依赖团队数据层 PR #1，请先合并 https://github.com/MaiSukrajima/blueberry-yield-prediction/pull/1 后再运行。本 PR 不重复提交数据层源码，CSV 与模型权重保留在本地交付包中。上传的 Word 报告来自已验证的本地源码快照，仓库合入后重新训练可记录团队 commit。

本模块完成普通线性回归和随机森林，使用训练折均值作为常数对照，复现 Ridge 线性参考。主指标为 621 行拼接 OOF MAE，同时报告 RMSE、R²、逐折均值与样本标准差、训练和推理耗时。结果见 `reports/baseline_models/metrics.json` 与 `reports/baseline_models/蓝莓基线模型实验报告.docx`。

## 数据与边界

- 数据版本 `wildblueberry-777-v2.1`，原始数据 777 行，16 个特征，MD5 `feffa3af1cbb3c8860f4ae34241adc3d`。
- 直接读取数据组的 `folds.csv`，不重新切分。621 行用于固定五折 CV，156 行 holdout 保持冻结。
- `Row#` 仅用于对齐；`yield` 仅作标签；标准化只在当折训练样本上 fit。
- 本模块没有参数搜索或特征工程，没有访问竞赛接口、外部数据或团队远程仓库。
- `models/*_cv621.joblib` 为仅使用 621 行训练的开发检查点。团队确定最终方案后，应按契约终检 holdout 并登记，再训练 all777 最终模型。本次没有执行这两步，也没有宣称已获得终检分数。
- 无官方 `test.csv`，所以交付 OOF 预测及通用推理接口，不生成虚构的 Kaggle submission。

## 安装与运行

推荐 Python 3.12 或 3.13，锁定建模依赖与数据组一致。实测环境版本保存在 `metrics.json`。命令在解压后的项目根目录执行。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-baseline.txt
python -m src.models.train_baseline
python -m unittest discover -s tests -p test_baseline.py -v
python tests/smoke_test.py
```

原始数据和 folds 已随包提供，训练无需联网，也不要运行 `make_folds.py` 改写团队折划分。安装依赖时需网络。为尽量稳定，随机森林使用 `n_jobs=1`；这只改变并行度，其他参数与数据组的默认 100 棵树一致。

## 推理

输入为 16 个特征，可选带 `Row#`，不含 `yield`。列的顺序可以不同，接口会按契约重排。输出保留输入顺序并包含 `predicted_yield`。示例只取 CV 区五行且移除了标签，仅验证接口，不是测试分数。

```powershell
python -m src.models.predict_baseline --model reports/baseline_models/models/linear_regression_cv621.joblib --input reports/baseline_models/inference_example.csv --output reports/baseline_models/example_predictions.csv
```

模型采用 joblib 格式，应只加载自己训练或可信来源的文件，并使用锁定的 scikit-learn 版本。缺列、额外列、缺失值或无穷值会报错。

## 交付目录

| 路径 | 内容 |
| --- | --- |
| `src/models/train_baseline.py` | 四组模型固定五折训练、评估、保存 |
| `src/models/predict_baseline.py` | 无标签 CSV 推理接口 |
| `configs/baseline.json` | 种子、固定参数、数据与折版本指纹 |
| `reports/baseline_models/scores.csv` | OOF MAE、RMSE、R²和耗时汇总 |
| `reports/baseline_models/fold_metrics.csv` | 每折训练与验证指标、样本量、耗时 |
| `reports/baseline_models/oof_predictions.csv` | 621 行 Row#、fold_id、真实值和各模型 OOF 预测 |
| `reports/baseline_models/metrics.json` | 完整参数、环境、来源 commit 与 SHA256、复现比对 |
| `reports/baseline_models/models/` | 20 个折模型及 4 个 CV621 检查点 |
| `reports/baseline_models/figures/` | OOF MAE 对比、预测与残差图 |
| `reports/baseline_models/*_coefficients.csv` | 线性回归和 Ridge 标准化特征系数 |
| `tests/test_baseline.py` | 折隔离、指标重算、模型恢复、输入校验等测试 |
| `reports/baseline_models/verification.json` | 实测验收记录 |
| `reports/baseline_models/蓝莓基线模型实验报告.docx` | 可供项目汇总的实验报告 |

数据组原有 `src/data`、`reports/baselines.*` 和交接文档保持原样，以免把本模块结果覆盖到数据组参考结果中。根目录原 README 为数据交接包说明；本文件为新增基线模块入口。

结果目录的 .gitignore 排除逐行 CSV 和模型权重；数据层的数据忽略规则由团队 PR #1 提供。

## 分数与追溯

每个 OOF 预测来自未见过该行标签的折模型；主分数对完整 621 行计算，不能把折 MAE 简单平均当作主分数。折标准差使用 `ddof=1`。计时分别测 fit 和验证集 predict，不含训练集回测、序列化与绘图；重复执行的耗时会改变。

`metrics.json` 记录本地源码快照的真实 commit 与源码 SHA256。该 commit 是本地交付版本，不表示向团队仓库提交、推送或合并。若合入团队仓库，需重新运行并记录团队 commit。数据、预测 CSV 与模型文件仅随本地交付包分发，不加入 Git；新增结果目录也由 `.gitignore` 保护。

本模块实现由 AI 辅助生成并在交付数据上执行、核验。课程或团队若要求 AI 使用说明，应如实保留，不要改称完全独立手写。
