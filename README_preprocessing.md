# 数据清洗与预处理 · 李佳曦

本模块负责严格输入检查、折内 IQR 异常检查、可选截断、标准化和归一化，以及 sklearn Pipeline 接入。**默认保留全部有效样本、不截断；线性模型默认 standard，树模型通常 none。**

## 运行

本仓库目前只有团队目录骨架，数据组 loader、数据与折表尚未合入。`run.py` 会只读引用外部数据组工程，同时加载本仓库 `src/preprocessing/`。不用把新增文件塞回原数据组目录。

当前电脑，在本仓库根目录执行：

```powershell
& '..\李佳曦_数据清洗与预处理交付\.venv\Scripts\python.exe' -X utf8 run.py all
```

数据组目录改变时：

```powershell
& '..\李佳曦_数据清洗与预处理交付\.venv\Scripts\python.exe' -X utf8 run.py all --project-root '数据组工程的完整路径'
```

其他电脑先建立自己的环境，不复制本机虚拟环境：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -X utf8 -m pip install -r requirements.txt --progress-bar off
& '.\.venv\Scripts\python.exe' -X utf8 run.py all --project-root '数据组工程的完整路径'
```

`all` 依次运行 11 项测试、四方案五折实验和导出 CSV 的独立复算；`test` 和 `evaluate` 可单独运行。入口会检查依赖版本，并比较原数据组工程运行前后的文件 SHA256。Python 字节码写入关闭，原目录不产生本模块缓存。

## 交付文件

| 文件 | 用途 |
| --- | --- |
| `src/preprocessing/transformers.py` | `build_preprocessor`、严格特征检查、IQRClipper |
| `src/preprocessing/evaluate.py` | 固定 Ridge(alpha=1.0) 的现有五折评估 |
| `tests/test_preprocessing.py` | 泄漏、批次独立性、行序、IQR=0、clone/序列化、OOF 等测试 |
| [docs/preprocessing.md](docs/preprocessing.md) | 中文公式、规则、线性/树模型和外层 CV 接入示例 |
| `run.py` | 外部数据组目录连接、环境检查、测试和核验入口 |
| `requirements.txt` | 与原数据组相同的四项直接依赖 |
| `reports/preprocessing/` | 实际运行结果和核验记录 |

报告包括四方案对比、20 条逐折指标、320 条训练/验证特征 IQR 记录、621 条 CV OOF、summary.md、测试原始日志和 verification.md。`id` 对应原 Row#；不导出 holdout 或全局清洗后的数据集。

## 实际结果

| 方案 | 总体 OOF MAE | 折 MAE 标准差（ddof=1） |
| --- | ---: | ---: |
| A 不截断 + 不缩放 | 120.293230 | 15.807254 |
| B 不截断 + StandardScaler | **98.785064** | 6.577427 |
| C 不截断 + MinMaxScaler | 117.053063 | 3.664034 |
| D IQR 截断 + StandardScaler | 102.498779 | 6.134433 |

11 项关键测试全部通过。B 复现约 98.79 的交付参考值；D 未改善，默认仍不截断。alpha 固定，结论只适用于这次 Ridge 配置。完整环境、文件指纹和逐折结果见 [summary.md](reports/preprocessing/summary.md)，导出结果复算及原目录完整性见 [verification.md](reports/preprocessing/verification.md)。

## 队友调用

团队数据层合入同一工程后，可直接调用：

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing import build_preprocessor

linear = make_pipeline(build_preprocessor(scaling="standard"), Ridge(alpha=1.0))
tree = make_pipeline(build_preprocessor(scaling="none"),
                     RandomForestRegressor(random_state=42))
```

X 只传 `EXPECTED_FEATURES`；DataFrame 列顺序会固定，数组必须遵守这 16 列顺序。误传 Row#、yield、缺列、多列、非数值、NaN 或 inf 会报错。每折 clone 完整模型，仅训练折 fit，验证折 predict。验证集不能另行 fit。

`src/__init__.py` 当前用于连接两个目录的包路径；往队友已有工程复制时，只复制 `src/preprocessing/`，不要覆盖队友原有的 src/__init__.py。数据层到位后也支持 `python -m src.preprocessing.evaluate`。

## 提交范围

工作分支：`feat/lijiaxi-preprocessing`。本模块只新增自己的代码、专属说明、测试和报告，不修改原有 README、data/README、.gitignore 或其他同学的文件。不要提交原始数据、folds 或 `.venv`；主分支通过团队约定的 PR 流程合入。

用一句话解释：我实现了只用训练数据学习刻度的“尺子”，让队友在每一折公平处理特征；把异常值先记账而非机械删除，并用真实测试和实验确认接口可靠、holdout 不参与。
