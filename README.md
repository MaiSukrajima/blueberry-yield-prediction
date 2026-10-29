# 蓝莓产量预测 · DeepBlueberry Challenge 2025

团队协作仓库（8 人）。任务：用环境、生物、气候类特征预测蓝莓产量 `yield`（回归）。

- 竞赛主页与数据下载：https://www.kaggle.com/competitions/deep-blueberry-challenge-2025
- 数据文件：`train.csv`（15000 × 19，含目标 `yield`）/ `test.csv` / `sample_submission.csv`，官方声明**无缺失值**
- 评估指标：RMSE / MAE

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
