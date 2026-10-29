# data/ 说明

本目录的 CSV **不入库**（见根目录 `.gitignore`），数据只留在各自电脑本地，避免仓库体积过大、也避免竞赛数据被转发。

## 获取数据

1. 打开竞赛页：https://www.kaggle.com/competitions/deep-blueberry-challenge-2025
2. 进入 **Data** 标签页，下载 `train.csv`、`test.csv`、`sample_submission.csv`
3. 三个文件放到本目录 `data/` 下即可

## 字段与目标

- 目标变量：`yield`（蓝莓产量，回归目标）
- 共 19 列：环境 / 生物 / 气候类特征 + `yield`
- 官方声明：**无缺失值**

下载后建议先跑一次完整性自检（形状、缺失值、重复行），确认与页面描述一致，再进入 EDA 与建模。
