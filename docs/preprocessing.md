# 数据清洗与预处理模块交接

负责人：**李佳曦**。本模块供曹麟一、左永乐等建模同学在自己的交叉验证中调用。

本文件现属于独立的“李佳曦_数据清洗与预处理交付”目录。`run.py` 只读引用旁边的数据组交付目录，实验报告仅写入本任务目录。原始数据、loader、基线和 folds 未复制或改动。

## 1. 分工和交付物

交付可复现的预处理函数、IQR 异常值检查、可选截断、标准化与归一化、Pipeline 示例、关键测试和固定 Ridge 的验证结果。

- `src/preprocessing/__init__.py`：对外导出接口。
- `src/preprocessing/transformers.py`：严格输入检查、`IQRClipper`、`build_preprocessor`。
- `src/preprocessing/evaluate.py`：读取现有折划分，完成四组固定配置实验。
- `tests/test_preprocessing.py`：用标准库 unittest 测试，无需新增测试依赖。
- `reports/preprocessing/`：对比、逐折指标、IQR 统计、CV 区 OOF 表和实验总结，共五个报告。

以 `reports/data_version.json` 的 `wildblueberry-777-v2.1` 和实际 loader 为准。DATA_CONTRACT 标题仍有 v2.0 的历史表述，但末尾已有 v2.1 修订；loader 顶部旧示例也不能替代实际的 `oof_mae_cv` 接口。

原始数据是 777 行、18 列。`Row#` 只是对齐标识，没有生物学含义，不入模；不能仅因它是行号就断言已经证实数据泄漏。`yield` 是目标，不做特征预处理。固定 16 个输入特征不增不减，不进行编码、特征工程、筛选、PCA 或共线性删列。

## 2. 清洗规则：严格检查，默认保留

既有 `reports/data_profile.md` 已确认无缺失、完全重复行或无穷值，且全部特征为数值。因此没有类别编码的对象，也不需要缺失值填补；本模块不重复 EDA。遇到新的 NaN、inf 或非数值输入直接报错，便于查明数据来源，不自动用 0、均值或其他值替换。

`build_preprocessor(scaling="standard", outlier_method="none", iqr_factor=1.5)` 返回可 clone 的 sklearn Pipeline，处理顺序固定为：

1. 检查输入并按契约固定特征顺序。
2. 可选地应用训练折学到的 IQR 截断。
3. 用训练折参数做 standard、minmax 或不缩放。

支持 pandas DataFrame 和二维 numpy 数组；输出为独立的浮点 numpy 数组，行数、行顺序、16 列特征均保持不变。输入对象不会被原地修改。DataFrame 的索引不随 numpy 输出携带，调用者用原有行位置或 `Row#` 对齐；模块不会排序样本或删除重复记录。

DataFrame 必须只包含 `EXPECTED_FEATURES` 的全部 16 列，传入顺序可以不同，模块会按契约重排。误传 `Row#`、`yield`、其他多余列、缺列、重复列名、非实数类型、NaN 或 inf 都会明确报错。不会默默丢掉多余列。先由调用方明确选择 `df.loc[:, EXPECTED_FEATURES]`。

数组必须是数值类型且形状为 `(n_samples, 16)`。其列顺序约定如下，建议始终通过 `df[EXPECTED_FEATURES].to_numpy(dtype=float)` 构造：

```text
clonesize, honeybee, bumbles, andrena, osmia,
MaxOfUpperTRange, MinOfUpperTRange, AverageOfUpperTRange,
MaxOfLowerTRange, MinOfLowerTRange, AverageOfLowerTRange,
RainingDays, AverageRainingDays, fruitset, fruitmass, seeds
```

数组不带列名，程序只能验证列数、类型和有限性，无法识别调用方是否把某一列偷偷换成了目标或打乱顺序。需要调用方遵守这个约定；带列名的 DataFrame 更易检查。

自定义变换器遵循 `BaseEstimator`、`TransformerMixin`，在 `fit` 学习参数，在 `transform` 使用 fitted 状态检查。sklearn 协议中的可选 `y` 参数仅为 Pipeline 兼容，不会被读取或处理；目标不能包含在 X 中。

## 3. 异常值检查与可选处理

默认保留全部有限且有效的数值样本。本数据来自仿真，蜂密度、降雨、温度等经常取固定网格值。低频出现或落在 IQR 界外，只说明“被统计规则标记为极端值”，不代表“已经确认的数据错误”。机械删行会损失有效实验条件，还会破坏固定 OOF 对齐。

IQR 检查只用当折训练特征，采用 numpy 的线性插值分位数：

```text
Q1 = 训练特征的 25% 分位数
Q3 = 训练特征的 75% 分位数
IQR = Q3 - Q1
下界 = Q1 - iqr_factor × IQR
上界 = Q3 + iqr_factor × IQR
```

严格小于下界或大于上界才标记。验证折只应用训练折阈值，不重新计算自己的分位数。`IQR=0` 时整列跳过标记和截断，计数记为 0 并记录原因；理论边界仍显示在报告里供审计，但不执行。否则，许多有效离散值会被错误压成常数。

`outlier_method="iqr_clip"` 把非跳过列的界外数值压到训练边界；训练和验证使用同一组边界。不删行，不处理 yield。默认 `"none"` 仍可用独立的 `IQRClipper` 做诊断：

```python
from src.preprocessing import IQRClipper

checker = IQRClipper(iqr_factor=1.5).fit(X_train)
audit = checker.summarize(X_train, X_valid)
# audit 是检查结果；这两行不会修改 X_train 或 X_valid。
```

`outlier_summary.csv` 包括方案、折号、特征、训练和验证样本数、Q1、Q3、IQR、上下界、两部分越界数量、跳过原因以及实际截断数量。检查在缩放前的原始特征尺度上进行。A/B/C 同样报告检查结果，但其实际截断数为 0；D 复用 Pipeline 中已 fit 的截断器。不同方案的相同计数不能相加当作新的独立样本。

## 4. 标准化和归一化有什么不同

`StandardScaler` 对每列使用训练集均值 μ 和标准差 σ：

```text
z = (x - μ_train) / σ_train
σ_train = sqrt(mean((x_train - μ_train)²))   # ddof=0
```

训练集上变成均值接近 0、方差接近 1 的尺度，常量列由 sklearn 以安全方式处理。它不会限制结果的上下界。这里的缩放标准差 `ddof=0`，与报告折间 MAE 标准差使用的 `ddof=1` 不是同一件事。

验证集标准化后均值不一定为 0，因为用的是训练集的 μ 和 σ，验证集的分布可能不同。不能为了让验证均值为 0 再对验证集单独 fit，那样训练和验证就用了两把不同的尺子。

`MinMaxScaler` 默认使用训练最小值与最大值：

```text
z = (x - min_train) / (max_train - min_train)
```

非恒定训练列通常映射到 `[0, 1]`。本模块保持 sklearn 默认行为（`feature_range=(0, 1)`、`clip=False`）。如果新数据超出训练最小值或最大值，结果可以小于 0 或大于 1，这不是错误。它对训练极值较敏感；不会偷偷额外截断到 `[0, 1]`。两种 scaler 都不能消除温度六列之间的共线性。

**标准化是本项目线性模型的默认方案；树模型通常无需缩放。** 对有正则化的 Ridge，缩放会改变原始特征受到的有效惩罚，因此固定 alpha 的分数不能推广到所有模型，也不能代替各模型自己的合理验证。

## 5. 给曹麟一、左永乐的接入示例

线性模型：

```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor

linear_model = Pipeline([
    ("preprocess", build_preprocessor(scaling="standard", outlier_method="none")),
    ("model", Ridge(alpha=1.0)),
])
# linear_model.fit(X_train, y_train)
# predictions = linear_model.predict(X_valid)
```

树模型用同一接口，保留严格检查，通常无需缩放：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.preprocessing import build_preprocessor

tree_model = Pipeline([
    ("preprocess", build_preprocessor(scaling="none", outlier_method="none")),
    ("model", RandomForestRegressor(random_state=42)),
])
```

队友数据层到位、将 `src/preprocessing/` 合入工程后，在外层固定五折中应每折新建或 clone **完整模型 Pipeline**。下面是可直接运行的线性模型接入例子；树模型只需把 `prototype` 换成上面的 `tree_model`，无需改折划分。保留队友原有的 `src/__init__.py`，不要用独立交付的包路径支持文件覆盖它。

```python
import numpy as np
from sklearn.base import clone
from src.data.loader import (
    load_raw, load_folds, iter_cv, oof_mae_cv, fold_mae,
    EXPECTED_FEATURES, TARGET,
)

df = load_raw()
folds = load_folds(df)
cv = folds["fold_id"].to_numpy() >= 0
y = np.full(len(df), np.nan)
y[cv] = df.loc[cv, TARGET].to_numpy(dtype=float)
oof = np.full(len(df), np.nan)
prototype = linear_model  # 由上面的线性模型代码创建

for tr, va in iter_cv(folds, k=5):
    model = clone(prototype)
    model.fit(df.iloc[tr][EXPECTED_FEATURES], y[tr])
    oof[va] = model.predict(df.iloc[va][EXPECTED_FEATURES])

assert np.isfinite(oof[cv]).all()
assert np.isnan(oof[~cv]).all()
print("总体 OOF MAE:", oof_mae_cv(y, oof, folds))
per = fold_mae(y, oof, folds)
print(per)
print("逐折均值和样本标准差:", per["mae"].mean(), per["mae"].std(ddof=1))
```

“折内 fit”就是先确定训练/验证索引，再只把训练部分交给完整 Pipeline 的 `fit`。预测验证集时使用训练学到的均值、极值、分位数等参数。先在全部 777 行甚至全部 621 行上缩放，再做五折，会提前使用当前验证折的信息，造成数据泄漏。把预处理放进 Pipeline 后，交叉验证的每次 fit 都会同时重学预处理和模型。

不要把模块预先 fit 好后输出全局清洗表给大家训练；不同模型的每一折都应自己拟合。若队友之后增加内层交叉验证，也将完整 Pipeline 交给内层搜索，不能在外层训练集统一预处理后再给内层验证用。

## 6. 运行命令与实际验证

当前独立交付目录中，推荐使用统一入口（自动定位旁边的数据组目录）：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 run.py all
```

本次为避免改动现有 ml 环境，独立交付目录自带本机 `.venv`，只安装 requirements.txt 原有的锁定依赖。可分开测试或评估；数据组目录移动时使用 `--project-root`：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 run.py test
& '.\.venv\Scripts\python.exe' -X utf8 run.py evaluate
& '.\.venv\Scripts\python.exe' -X utf8 run.py all --project-root '数据组工程的完整路径'
```

合入包含数据层的团队工程后，原来的 `python -m src.preprocessing.evaluate` 和 `python -m unittest discover -s tests -p test_preprocessing.py -v` 也可使用。当前 GitHub 骨架尚未包含数据层时，使用 run.py 引用外部工程。

四组统一固定 Ridge(alpha=1.0)、现有五折、seed=42，无模型搜索：A 不截断且不缩放；B 不截断且标准化；C 不截断且 MinMax；D IQR 截断后标准化。

本次实际运行环境：Python 3.11.16、numpy 2.3.5、pandas 2.3.3、scikit-learn 1.7.2、scipy 1.16.3。四项直接依赖与 requirements.txt 完全一致；Python 版本与数据组原生成环境的 3.13.9 不同，已记录，没有改动其环境。

| 方案 | 总体 OOF MAE | CV RMSE | 逐折 MAE 均值 ± 标准差（ddof=1） |
| --- | ---: | ---: | ---: |
| A 不截断 + 不缩放 | 120.293230 | 233.470077 | 120.307795 ± 15.807254 |
| B 不截断 + StandardScaler | 98.785064 | 129.838684 | 98.784215 ± 6.577427 |
| C 不截断 + MinMaxScaler | 117.053063 | 154.139001 | 117.059417 ± 3.664034 |
| D IQR 截断 + StandardScaler | 102.498779 | 138.218195 | 102.496518 ± 6.134433 |

B 复现交付参考值约 98.79，逐折 MAE 为 99.311149、95.398806、103.464520、106.172299、89.574301。D 比 B 增加约 3.713716，截断未改善，默认保持不截断。这里没有检验差异的统计显著性；结论仅限这套固定 Ridge 配置。本次折 4 的 bumbles 出现 IQR=0，已跳过并记录，未把整列截成常数。

实际分数、逐折波动、运行版本与原始文件指纹见 `../reports/preprocessing/summary.md` 和 `comparison.csv`。主指标是把 621 个验证预测拼接后计算的总体 OOF MAE，不能用五个折 MAE 的均值替代。完整 OOF 数组长 777，holdout 始终 NaN；CSV 只写 CV 记录，其 id 对应原始 Row#。

锁定版本环境中实跑 **11 项测试全部通过**。测试覆盖验证集极端值不改变训练参数、验证批次不改变单条样本结果、行序和行数保持、IQR=0 保护、边界行为、非法特征拒绝、clone/序列化恢复、sklearn 默认缩放行为、错误折元数据拒绝、真实 OOF 覆盖，以及把 holdout 特征和标签替换成 NaN 后实验预测仍完全一致。run.py 还从保存的 CSV 独立复算各项指标，并比较数据组目录运行前后的文件 SHA256；实际记录保存在 verification.md。

holdout 未参与规则制定、训练、预测或评分，未修改 holdout_usage.md。没有修改原始 CSV、现有 folds、loader、基线代码或覆盖数据组已有报告。
