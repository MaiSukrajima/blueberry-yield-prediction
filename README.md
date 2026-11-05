# 蓝莓产量预测 · DeepBlueberry Challenge 2025

团队协作仓库（8 人）。任务：用环境、生物、气候类特征预测蓝莓产量 `yield`（回归）。

- 数据源：Kaggle 公开数据集 **Wild Blueberry Yield Prediction**（Obsie, Qu & Drummond 2020，CC BY 4.0）
- 数据文件：`data/raw/WildBlueberryPollinationSimulationData.csv`，**777 行 × 18 列**（16 特征 + `Row#` + 目标 `yield`），**无缺失值**
- 获取方式见 [`data/README.md`](data/README.md)，一条命令即可：`python src/data/fetch_dataset.py`（**无需 Kaggle 账号**）
- 评估指标：RMSE / MAE

> ⚠️ **数据源已变更（2026-11）。** 原选题 `DeepBlueberry Challenge 2025` 为邀请制竞赛、无法下载，已改用上述公开数据集。
> 规模由 15000×19 变为 777×18，**基于旧数据的折划分与基线分数全部作废**。数据口径以 [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) 为准。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `data/` | 本地数据（CSV 不入库，仅放本机） |
| `notebooks/` | 探索性分析与建模笔记 |
| `src/` | 可复用脚本 |
| `reports/` | 图表、结果与技术报告 |

## 分支规范

每人用自己的名字建分支，例如 `feat/zhuchenyu-eda`，完成后向 `main` 提交 Pull Request。

## 推送流程（Fork + Pull Request）

本仓库为**公开**仓库，无需邀请。先 Fork 到自己账号，再改动、提 PR：

```bash
# 1) 在仓库网页点右上角 Fork，复制到自己账号；然后克隆自己的 fork
git clone https://github.com/你的用户名/blueberry-yield-prediction.git
cd blueberry-yield-prediction

# 2) 建自己的分支（不要在 main 上直接改）
git checkout -b feat/你的名字-模块

# 3) 改动后提交（data/ 里的 CSV 已被 .gitignore 忽略，不会上传）
git add .
git commit -m "feat: 简单说明这次做了什么"

# 4) 推送到自己的 fork
git push -u origin feat/你的名字-模块

# 5) 回到原仓库网页，点 "Compare & pull request"，向 main 提 PR
```

> **首次推送登录**：用户名填 GitHub 用户名，密码填 **Personal Access Token（PAT）**，不是登录密码。
> 生成 PAT：GitHub → Settings → Developer settings → Personal access tokens → 勾选 `repo` 权限。
>
> **国内网络访问 GitHub 不稳**时，可配置本地代理（示例为 7897 端口，按自己代理改）：
> `git config --global http.proxy http://127.0.0.1:7897`
