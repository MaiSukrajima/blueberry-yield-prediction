# 数据清洗与预处理实验结果

负责人：李佳曦。

- 运行时间：2026-11-16T09:42:46+08:00（Asia/Shanghai）
- 数据版本：`wildblueberry-777-v2.1`；折版本：`folds-777-v2.0`。
- 数据组输入目录（只读）：`..\蓝莓数据层交付_v2.1_20261103`。
- 输出保存在本任务交付目录的 `reports/preprocessing/`，不写入数据组目录。
- 数据规模：777 行 × 18 列；16 个特征。现有五折：621 CV，156 holdout；seed=42。
- 原始文件：`data/raw/WildBlueberryPollinationSimulationData.csv`；85259 字节。
- 原始文件 MD5：`feffa3af1cbb3c8860f4ae34241adc3d`（load_raw strict=True 校验）。
- 原始文件 SHA256：`287dad940a0829512b52fbaa76c5c8dbed7015d1e70b028898e8a985216d1cad`。
- folds.csv SHA256：`8cc561ee6c8b6413b9107df65ca36f27c0fc3d5ed3796abbddc7d7d41e812e48`。
- 原 loader.py SHA256：`9a7e7141ace7b8600ef52b5a74e8c8f025649831dc0139e0d5c9adaa487fde4e`。
- 原 requirements.txt SHA256：`74f4798d942cbbb37c88e873ce1e6327c3ff9585ec918683a8f7921d76b19c69`。
- 代码基于 Git commit `367a7b44e6060a7742296bddab9a8d307172a227`；工作区有未提交改动，精确代码以下列 SHA256 为准。
  - `src/preprocessing/__init__.py`：`0f7224c270e67c1fb47d133a794c95187e9ac8232fde294500566b95a635b8a7`
  - `src/preprocessing/transformers.py`：`8ebe02939f2cc3fcad3e47d11f68ee3121e9a37db695fb5d968f5d47cdce60d7`
  - `src/preprocessing/evaluate.py`：`dd044a90f2b8b44f5d1021ec8e41251b3c36a13b1492aea54e234b710cc71e18`

## 实际运行环境

Python 3.11.16；numpy 2.3.5；pandas 2.3.3；scikit-learn 1.7.2；scipy 1.16.3。
依赖遵循 requirements.txt；Python 生成环境原为 3.13.9，实际解释器版本以上述记录为准。

## 固定 Ridge(alpha=1.0) 的五折结果

每折重新建立完整 Pipeline；顺序：输入校验 → 可选 IQR 截断 → 缩放 → Ridge。
总体 OOF MAE 由 oof_mae_cv 计算；逐折 MAE 由 fold_mae 计算；标准差 ddof=1。
总体 OOF MAE 与逐折 MAE 均值是两个不同统计量，后者只用于报告波动。

| 方案 | 缩放 | 截断 | OOF MAE | CV RMSE | 折 MAE 均值 ± 标准差 |
| --- | --- | --- | ---: | ---: | ---: |
| A | none | none | 120.293230 | 233.470077 | 120.307795 ± 15.807254 |
| B | standard | none | 98.785064 | 129.838684 | 98.784215 ± 6.577427 |
| C | minmax | none | 117.053063 | 154.139001 | 117.059417 ± 3.664034 |
| D | standard | iqr_clip | 102.498779 | 138.218195 | 102.496518 ± 6.134433 |

| 方案 | 折0 MAE | 折1 MAE | 折2 MAE | 折3 MAE | 折4 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 111.263066 | 114.672586 | 125.401039 | 145.275642 | 104.926642 |
| B | 99.311149 | 95.398806 | 103.464520 | 106.172299 | 89.574301 |
| C | 113.114028 | 115.022645 | 118.883924 | 122.455937 | 115.820549 |
| D | 103.900555 | 97.076125 | 108.056113 | 108.323127 | 95.126673 |

交付 baselines.json 中 Ridge 参考值为 98.785063504；本次 B 为 98.785063504，差值 +0.000000000。
参考值仅用于核对复现，未作为计算结果或训练参数。

## 默认规则与异常值解释

默认保留全部有限、有效样本，不删除行、不填补、不编码、不处理 yield。线性模型默认 standard；树模型通常用 none；缩放不消除温度特征共线性。
IQR 检查使用训练折的 Q1/Q3（线性插值），边界为 Q1−1.5×IQR 与 Q3+1.5×IQR。严格小于下界或大于上界才标记；验证折仅应用训练阈值。
仿真中的固定网格、低频蜂密度或温度取值可能合理。统计规则标记的极端值不等于确认的数据错误。
IQR=0 的列跳过标记与截断，计数为 0 并给出 skip_reason；报告中的上下界保留理论值供审计，实际不应用。
D 相对 B 的 OOF MAE 差为 +3.713716（正数表示变差）；截断未改善，本项目保留不截断作为默认。
alpha 固定时缩放也改变了 Ridge 对各原始特征的有效惩罚强度；这些结果只适用于该配置，不能宣称某种缩放方法对所有模型最优。

## 输出与隔离验证

四个内存 OOF 数组均长 777：621 个 CV 位置各有且只有一次有限预测，156 个 holdout 位置始终 NaN。
oof_predictions.csv 仅含 621 行 CV：id 对应原始 Row#，另含 fold_id、y_true、pred_A/B/C/D。
outlier_summary.csv 每方案×5 折×16 特征，共 320 行；先在原始特征上统计，再截断/缩放。A/B/C 的阈值只是诊断；实际截断数量为 0。各方案的相同阈值和计数不能相加当作独立样本数。
holdout 未参与本次实验：load_raw 按既有接口载入文件并核验指纹，之后只索引 CV 特征和标签；没有对 holdout 做分布统计、预处理拟合、训练、预测或评分。holdout_usage.md 未改动。
未输出全局清洗后数据集；原始 CSV、folds、loader、队友基线和数据组报告均保持原样。

独立目录复现：`python run.py evaluate`；测试：`python run.py test`；完整核验：`python run.py all`。
合入团队工程后也可使用 `python -m src.preprocessing.evaluate`。
