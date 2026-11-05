# 蓝莓产量预测 —— 数据层交付 v2.1

> 期末作业项目（回归任务：用环境、生物、气候特征预测蓝莓产量）。
> `data/`、`src/data/`、`docs/`、`reports/` 由数据组维护，其余目录归各组所有。

> **v2.0 已替换数据源**：原 `DeepBlueberry Challenge 2025` 竞赛数据为邀请制、无法获取，
> 改用 Kaggle 公开数据集。**规模从 15,000 行变为 777 行，v1.x 所有数字作废。**
> 详见 `docs/DATA_CONTRACT.md` 第 8 节。
>
> **v2.1 更正两处表述**：① 本数据与竞赛版是「同源、不同运行批次」，不是「扩增生成」；
> ② `Row#` 有组结构但实测**未**造成泄漏。数据与分数**均未变动**，只改描述。
> 详见 `docs/方案对比分析.md`。

---

## 一、这里有什么

数据工作已全部完成并实测跑通。拿到手就能开工，**不需要 Kaggle 账号、不需要下载**。

| 你现在需要的 | 位置 | 状态 |
| --- | --- | --- |
| 原始数据（1 个 CSV，777 行） | `data/raw/WildBlueberryPollinationSimulationData.csv` | 已有，已核验 |
| 全组共用的折划分 | `data/processed/folds.csv` | 已生成 |
| 数据画像（逐列统计 + 校验结论） | `reports/data_profile.md` | 已生成 |
| 数据交接约定 | `docs/DATA_CONTRACT.md` | 已定稿 |
| 报告用「数据说明」章节 | `docs/数据说明.md` | 已定稿 |
| 基线分数与门槛 | `reports/baselines.md` | 已实测 |
| 统一读取入口 | `src/data/loader.py` | 已验证 |

**数据在哪？** `data/` 不进 Git，随项目压缩包发。

- 拿到**压缩包** → 直接开工，`data/raw/` 里已有 CSV；
- 只 **clone 了 Git 仓库** → 找数据维护人（朱宸宇）要压缩包；
- 或者**自己跑一条命令**重新获取（不需要账号）：

```bash
python src/data/fetch_dataset.py
```

---

## 二、三条命令跑通

环境（全组必须一致，否则折划分会对不上）：

```bash
pip install -r requirements.txt
```

```bash
# 1. 校验数据（应输出 13 通过 / 3 警告 / 0 失败）
python src/data/validate_data.py

# 2. 生成折划分（已存在且参数不变时自动跳过）
python src/data/make_folds.py

# 3. 冒烟测试（31 项断言）
python tests/smoke_test.py
```

**只要第 1 条出现 `0 失败`，你就可以信任这份数据。**

想复现基线分数：`python src/data/baselines.py`

---

## 三、怎么在你的代码里用

```python
import sys; sys.path.insert(0, ".")          # 在项目根目录下运行
import numpy as np
from src.data.loader import (
    load_raw, load_folds, iter_cv, oof_mae_cv,
    EXPECTED_FEATURES, TARGET,
)

df = load_raw()                                   # 带 md5 指纹守卫
folds = load_folds(df)                            # 带折/数据一致性守卫

X = df[EXPECTED_FEATURES].to_numpy(dtype=float)   # 16 列，不含 Row#
y = df[TARGET].to_numpy(dtype=float)

oof = np.full(len(df), np.nan)
for tr_pos, va_pos in iter_cv(folds, k=5):
    # 任何预处理（标准化、编码）只能在这里面 fit
    model.fit(X[tr_pos], y[tr_pos])
    oof[va_pos] = model.predict(X[va_pos])

print("OOF MAE =", oof_mae_cv(y, oof, folds))     # 这就是全组统一的分数口径
```

`oof` 里 holdout 的 156 行保持 `NaN` 是正常的，用 `oof_mae_cv()` 会自动只算 CV 区。

---

## 四、数据从哪来

一句话：**2020 年发表于学术期刊、托管在 Mendeley Data 的 777 条仿真数据，被两位不同的 Kaggle 用户独立公开，两份 CSV 逐字节相同。**

| 层 | 内容 | 证据 |
| --- | --- | --- |
| ① 学术源头 | Obsie, Qu, Drummond (2020)，缅因大学蓝莓授粉模拟，聚合成 777 条 | Mendeley DOI `10.17632/p5hvjzsvn8.1`，CC BY 4.0 |
| ② 公开发布 | Kaggle Datasets 两个独立上传者都收录了这份 CSV | `saurabhshahane/…` 与 `shashwatwork/…`，md5 完全相同 |
| ③ 本地落地 | `data/raw/` 那份 CSV | md5 `feffa3af1cbb3c8860f4ae34241adc3d`，85,259 字节 |

完整溯源与自证方法见 `reports/provenance.md`，可自己跑：

```bash
python src/data/verify_provenance.py          # 完整溯源，需联网
python src/data/verify_provenance.py --no-net # 只校验本地指纹
```

预期末行：`合计：通过 6 / 警告 0 / 失败 0`。

### 4.1 为什么不用原计划的竞赛数据

`DeepBlueberry Challenge 2025` 是**邀请制竞赛**，数据接口需参赛者认证，非受邀者无权下载。

本数据集与它的关系，经逐行比对确认为：**同一套仿真模型的两批独立运行结果，不是母本与副本**。
按 `Row#` 对齐后，16 个特征完全相同的行只有 **1 / 15,000**；同一 `Row#` 下 `fruitset` 的一致率仅 **0.4%**。

| 项 | 竞赛版 | 本版 |
| --- | --- | --- |
| 行数 | 15,000（train） | **777** |
| 列数 | 19（含 `id`、`Row#`） | **18**（含 `Row#`，无 `id`） |
| 测试集 | 有 `test.csv` 10,000 行 | **无**，需自行划分 |
| 数据性质 | 约 777 个场景 × 每场景约 19 次独立仿真 | 每个场景一条原始记录 |
| 温度与 `yield` 的相关 | ≈ −0.004（信号被重复运行稀释） | **≈ −0.18** |

**写报告时请如实写明数据来源，不要把本版数字和竞赛版混用。**

---

## 五、五条硬约束（不遵守则分数不可比）

1. **`Row#` 不是特征。** 它是 0–776 的原始行号，唯一对应每一行，入模等于把样本序号交给模型。
   可用特征只有 `EXPECTED_FEATURES` 里的 16 列。

2. **所有预处理必须在折内 `fit`。** 标准化、编码都不许在 `iter_cv` 外面做，
   否则验证折的信息漏进训练，分数虚高。

3. **分数一律报 OOF MAE。** 不要把 5 个折的 MAE 求平均，那是另一种统计量。

4. **holdout 只在定模时用，全组限 3 次。** 用了要登记到 `reports/holdout_usage.md`。

5. **不要自己重新切分。** 从 `data/processed/folds.csv` 读折。

---

## 六、基线门槛（你至少要超过这个）

真实数据、本仓库折划分上实测的 OOF MAE（CV 区 621 行）：

| 模型 | OOF MAE | 相对常数基线 |
| --- | --- | --- |
| 训练折均值（常数基线） | 1103.77 | — |
| **Ridge（标准化, alpha=1）** | **98.79** | −91.1% |
| HistGradientBoosting（默认） | 116.49 | −89.4% |
| RandomForest（默认, 100 棵） | 123.14 | −88.8% |

**如果你的 OOF MAE 高于 98.79，说明还没超过线性模型。**

⚠️ **一个反直觉但重要的事实**：线性模型比两种树模型都好。原因一是 `fruitset` 与 `yield`
近似线性（r=0.984），线性模型已吃掉主要信号；二是每折训练集只有约 496 行，树模型容易过拟合。
**这不是你的代码写错了，是数据的性质。** 想超过 98.79 要动特征工程，不是换更复杂的模型。

⚠️ **每折验证只有 124 行**，折间波动约 ±6–9。**两个模型差距小于约 5 时不要当作有差异。**

---

## 七、三个必须先知道的数据特性

1. **`fruitset` 与 `yield` 的 Pearson r = 0.9841**，`seeds` 0.9612，`fruitmass` 0.9304。
   信号高度集中在果实类特征上。它们不是泄漏，是真实的生物学前驱，**不要删**。

2. **温度六列近乎完全共线**（最高 |r| = 0.9999）。正则化模型和树模型不受影响，
   普通最小二乘的系数会不稳定。

3. **量纲差异 147 倍**。线性模型与神经网络必须先标准化。

---

## 八、目录结构

```
blueberry-handover/
├── data/
│   ├── raw/                                     # 原始 CSV，只读
│   │   └── WildBlueberryPollinationSimulationData.csv
│   ├── processed/                               # folds.csv + folds.meta.json
│   └── _archive_competition_v1/                 # 已作废的 15000 行竞赛版，仅留档
├── docs/
│   ├── DATA_CONTRACT.md                         # 数据交接约定（进仓库，能 diff）
│   ├── 数据说明.md                               # 报告用章节
│   └── 数据交接约定_v2.0.docx                     # Word 版，给不用 Git 的同学
├── reports/
│   ├── data_profile.md                          # 数据画像
│   ├── provenance.md                            # 溯源报告
│   ├── dataset_fetch.md                         # 获取记录
│   ├── baselines.md                             # 基线分数
│   ├── holdout_usage.md                         # holdout 登记
│   ├── checksums.md5                            # 文件指纹
│   └── data_version.json                        # 数据版本标识
├── src/data/
│   ├── fetch_dataset.py                         # 从 Kaggle 公开数据集获取（无需认证）
│   ├── validate_data.py                         # 结构校验 + 生成画像
│   ├── make_folds.py                            # 折划分（幂等）
│   ├── loader.py                                # 统一读取入口
│   ├── baselines.py                             # 基线模型
│   └── verify_provenance.py                     # 溯源验证
├── tests/smoke_test.py                          # 31 项断言
├── requirements.txt
└── README.md
```

---

## 九、遇到问题看这里

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `FileNotFoundError: data/raw/...csv` | 数据没放对位置（CSV 不进 Git） | 解压项目压缩包，或跑 `python src/data/fetch_dataset.py` |
| `md5 与契约值不一致` | 文件被改过或不是同一份 | 重新解压；确认没加 BOM、没在 Excel 里另存过 |
| `折表的 id 集合与数据不一致` | 折和数据不是同一批 | `python src/data/make_folds.py --force` |
| `CV 区仍有 N 行未预测` | 折没跑完就出分 | 检查 `iter_cv` 的 5 折是否都填充了预测 |
| `折表里有 N 个 CV 折，与 k=5 不符` | 折文件损坏 | `--force` 重新生成 |
| 分数比同学高很多 | 预处理泄漏或用了 `Row#` | 见第五节第 1、2 条 |
| 树模型跑不过 Ridge | **正常现象**，不是 bug | 见第六节说明 |

---

## 十、版本

| 版本 | 日期 | 变化 |
| --- | --- | --- |
| v1.0–v1.2 | 2026-11-02 | 基于 15,000 行竞赛数据版本（**已作废**，留档于 `data/_archive_competition_v1/`） |
| **v2.0** | 2026-11-03 | 数据源替换为 Kaggle 公开数据集 777 行；折划分、基线、画像全部重做 |
| **v2.1** | 2026-11-03 | 更正与竞赛版关系的表述（"扩增生成"→同源不同批次，依据逐行比对）；更正 `Row#` 泄漏表述（实测随机折与分组折差 0.03%，不构成泄漏）；新增 `docs/方案对比分析.md` |
