# 数据获取报告 dataset_fetch

- 时间：2026-11-03 20:02:58
- 主源：`https://www.kaggle.com/datasets/saurabhshahane/wild-blueberry-yield-prediction`
- 备源：`https://www.kaggle.com/datasets/shashwatwork/wild-blueberry-yield-prediction-dataset`

## 1. 获取方式

Kaggle **Datasets** 下载接口，**无需登录 / 无需 API Token**：

```
GET https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}
```

> 注意：Kaggle **Competitions**（竞赛）的数据接口必须认证，且 `deep-blueberry-challenge-2025` 为邀请制，无法下载。本题改用公开 Dataset，属同源数据。

- `saurabhshahane/wild-blueberry-yield-prediction` → zip 361,235 字节，含 1 个 CSV
  - 采用 `Data in Brief/Data in Brief/WildBlueberryPollinationSimulationData.csv`，md5 `feffa3af1cbb3c8860f4ae34241adc3d` 与预期一致 ✅

## 2. 文件校验

| 项 | 预期 | 实际 | 结果 |
| --- | --- | --- | --- |
| md5 | `feffa3af1cbb3c8860f4ae34241adc3d` | `feffa3af1cbb3c8860f4ae34241adc3d` | ✅ |
| 字节数 | `85,259` | `85,259` | ✅ |
| 形状 | `777×18` | `777×18` | ✅ |
| 缺失值 | 0 | 0 | ✅ |

列名：

```
Row#, clonesize, honeybee, bumbles, andrena, osmia, MaxOfUpperTRange, MinOfUpperTRange, AverageOfUpperTRange, MaxOfLowerTRange, MinOfLowerTRange, AverageOfLowerTRange, RainingDays, AverageRainingDays, fruitset, fruitmass, seeds, yield
```

## 3. 复现方式

```bash
python src/data/fetch_dataset.py          # 重新下载并校验
python src/data/validate_data.py          # 结构校验
python src/data/make_folds.py             # 生成折划分
python tests/smoke_test.py                # 冒烟测试
```
