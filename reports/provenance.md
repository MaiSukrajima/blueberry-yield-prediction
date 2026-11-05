# 数据溯源报告 provenance

- 生成时间：2026-11-03 20:45:10
- 校验对象：`WildBlueberryPollinationSimulationData.csv`
- 汇总：**通过 6 / 警告 0 / 失败 0**

## 一、溯源链条

| 层 | 内容 | 证据 |
| --- | --- | --- |
| ① 学术源头 | Obsie, Qu, Drummond (2020), Computers and Electronics in Agriculture：缅因大学用蓝莓授粉模拟模型跑仿真，聚合成 **777 条**记录 | Mendeley Data，DOI `10.17632/p5hvjzsvn8.1`，CC BY 4.0 |
| ② 公开发布 | Kaggle Datasets 两个独立上传者的数据集都收录了这份 777 行 CSV | `saurabhshahane/wild-blueberry-yield-prediction`（含 Data in Brief 说明文档）、`shashwatwork/wild-blueberry-yield-prediction-dataset` |
| ③ 本地落地 | `data/raw/WildBlueberryPollinationSimulationData.csv` | md5 `feffa3af1cbb3c8860f4ae34241adc3d`，85,259 字节 |

**一句话**：这是 2020 年发表于学术期刊、托管在 Mendeley Data 的仿真数据集，被两位不同用户在 Kaggle 上独立公开，两者内含的 CSV 逐字节相同——数据不是本地生成或编造的。

## 二、逐项证据

| 级别 | 编号 | 检查项 | 说明 |
| --- | --- | --- | --- |
| ✅ PASS | P01 | 本地文件指纹 | md5 `feffa3af1cbb3c8860f4ae34241adc3d`、85,259 字节，与契约值一致 |
| ✅ PASS | P02 | 源 saurabhshahane/wild-blueberry-yield-prediction | 主源（含 Data in Brief 说明文档），CSV md5 `feffa3af1cbb3c8860f4ae34241adc3d` 与本地**逐字节相同** |
| ✅ PASS | P02 | 源 shashwatwork/wild-blueberry-yield-prediction-dataset | 备源（独立上传者），CSV md5 `feffa3af1cbb3c8860f4ae34241adc3d` 与本地**逐字节相同** |
| ✅ PASS | P03 | 数据规模 | 777 行 × 18 列，与该数据集声明一致 |
| ✅ PASS | P04 | 离散特征取值网格 | 13 列取值种类均在 2–20 之间（如 clonesize=6, honeybee=7, bumbles=10, andrena=12, osmia=12…），符合计算机模拟实验设计的固定网格，随机生成的数据不会具备这种结构 |
| ✅ PASS | P05 | yield 极值吻合 | min=1637.704022 / max=8969.401842，与学术原件精确到小数点后 6 位一致 |

## 三、获取方式（无需账号）

```bash
python src/data/fetch_dataset.py      # 下载 + 校验 + 落地
python src/data/verify_provenance.py  # 溯源验证（需联网）
```

背后的接口是 Kaggle **Datasets** 的公开下载端点，**不需要登录、不需要 API Token**：

```
GET https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}
```

> 为什么不用 `deep-blueberry-challenge-2025` 竞赛数据：该竞赛为**邀请制**，其数据接口需要认证且非参与者无权下载。本次改用公开 Dataset，内容是这批 777 条学术记录的原始版本；它与竞赛版属同一模型的**不同运行批次**（逐行精确匹配 1/15,000），不是母本与副本。任务（用环境/生物/气候特征回归预测 yield）与字段（16 特征 + yield）完全一致，但两版信噪比不同，分数不可直接比较。

## 四、与竞赛版的差异（写报告时请说明）

| 项 | 竞赛版 DeepBlueberry 2025 | 本版公开数据集 |
| --- | --- | --- |
| 行数 | 15,000（train） | **777** |
| 列数 | 19（含 `id`、`Row#`） | **18**（含 `Row#`，无 `id`） |
| 数据性质 | 约 777 个场景 × 每场景约 19 次独立仿真 | 每个场景一条原始记录 |
| 测试集 | 有 `test.csv` 10,000 行 | **无**，需自行划分 |
| 主要风险 | `Row#` 有组结构，实测未产生泄漏 | 无此问题，但样本量小 |

本报告所有结论均针对**本版 777 行数据**，不要与竞赛版的任何数字混用。
