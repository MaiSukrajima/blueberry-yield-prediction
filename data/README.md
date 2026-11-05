# data/ 说明

本目录的 CSV **不入库**（见 `.gitignore` 中的 `data/raw/`、`data/processed/`），只留在各自电脑本地。
原因有两个：仓库体积，以及"数据是可以一条命令重新获取的"——存进 Git 反而没有意义。

> ⚠️ **数据源已于 2026-11 变更。**
> 原选题 `DeepBlueberry Challenge 2025` 是**邀请制竞赛**，非受邀者无权下载（接口返回 401）。
> 现改用 Kaggle Datasets 上的公开数据集。**任何基于 15000 行旧数据得出的结论均已作废**，详见 `docs/DATA_CONTRACT.md`。

## 获取数据

### 方式一：跑脚本（推荐，无需 Kaggle 账号）

```bash
python src/data/fetch_dataset.py
```

会自动下载 → 校验 md5 → 落盘到 `data/raw/`，并生成 `reports/dataset_fetch.md`。
支持 `--check-only`（只校验本地文件，不联网）和 `--proxy http://127.0.0.1:7897`（走本地代理）。

> 脚本配了**主用 + 备用两个镜像**，任一可用即可。镜像偶尔会返回 404，重跑或换网络即可，不是脚本问题。

### 方式二：解压项目交付包

`蓝莓数据层交付_v2.1_20261103.zip` 内已含 `data/raw/` 与 `data/processed/`，解压到仓库根目录即可。
（压缩包在群文件，仓库里没有。）

### 方式三：手动下载

- 主：<https://www.kaggle.com/datasets/saurabhshahane/wild-blueberry-yield-prediction>
- 备：<https://www.kaggle.com/datasets/shashwatwork/wild-blueberry-yield-prediction-dataset>

下载后把 `WildBlueberryPollinationSimulationData.csv` 放到 `data/raw/`。

## 数据来源与许可

| 项 | 内容 |
| --- | --- |
| 学术出处 | Obsie, Qu & Drummond (2020)，*Wild blueberry yield prediction using a combination of computer simulation and machine learning* |
| 原始发布 | Mendeley Data，DOI [`10.17632/p5hvjzsvn8.1`](https://doi.org/10.17632/p5hvjzsvn8.1) |
| 许可 | **CC BY 4.0**（可商用、可修改，**须注明出处**） |
| 文件 md5 | `feffa3af1cbb3c8860f4ae34241adc3d`（85259 字节） |

引用时请注明上述论文与 Mendeley 数据集。

## 形状与字段

**777 行 × 18 列**（`Row#` 为行号索引，`yield` 为目标变量，其余 **16 列为特征**）：

| 分组 | 字段 | 数量 |
| --- | --- | --- |
| 行号（非特征） | `Row#` | 1 |
| 场地 / 植株 | `clonesize`、`fruitset`、`fruitmass`、`seeds` | 4 |
| 蜂类（生物） | `honeybee`、`bumbles`、`andrena`、`osmia` | 4 |
| 温度区间 | `MaxOfUpperTRange`、`MinOfUpperTRange`、`AverageOfUpperTRange`、`MaxOfLowerTRange`、`MinOfLowerTRange`、`AverageOfLowerTRange` | 6 |
| 降雨 | `RainingDays`、`AverageRainingDays` | 2 |
| **目标** | **`yield`**（蓝莓产量，回归目标） | 1 |

- **无缺失值**（原始论文声明，已被 `validate_data.py` 复核通过）
- **无 `id` 列，也没有 `test.csv`** —— 与旧竞赛版不同，测试集需自行划分

## 拿到的第一件事：跑校验

```bash
python src/data/validate_data.py
```

预期输出：**13 通过 / 3 警告 / 0 失败**。若与你看到的不一致，先别动数据，去开 issue 或直接找我。

完整校验逻辑见 `src/data/validate_data.py`，口径定义见 `docs/DATA_CONTRACT.md`。

## 折划分

**不要自己切分。** 全组必须用同一套折，否则分数不可比：

```bash
python src/data/make_folds.py     # 生成 data/processed/folds.csv（幂等，重复跑结果一致）
```

或直接读 `data/processed/folds.csv`（列：`id,fold_id,split`）。
`src/data/loader.py` 在读取时会做折一致性守卫，用错版本会直接报错。
