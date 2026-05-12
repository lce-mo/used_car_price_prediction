# Used Car Price Prediction

天池二手车交易价格预测项目。项目目标是通过二手车价格预测竞赛，学习完整的机器学习建模流程：数据读取、清洗、特征工程、交叉验证、OOF 评估、模型训练、预测提交、实验复盘和文档沉淀。

核心评价指标是 MAE：

```text
MAE = mean(abs(y_true - y_pred))
```

## 1. 项目结构

```text
.
├── data/
│   ├── raw/                 # 原始数据，不上传 GitHub
│   ├── interim/             # 清洗后的中间数据，不上传生成文件
│   └── processed/           # 特征处理后的数据，不上传生成文件
├── docs/
│   ├── project_state.md
│   ├── data_dictionary.md
│   ├── feature_design.md
│   ├── experiment_log.md
│   ├── modeling_strategy.md
│   └── 过去日志/
├── outputs/
│   ├── models/              # 模型产物，不上传 pkl
│   ├── predictions/         # 验证/测试预测，不上传 csv
│   ├── submissions/         # 提交文件，不上传 csv
│   ├── reports/             # 运行报告，不上传生成文件
│   └── figures/
├── scripts/                 # 命令入口
├── src/
│   ├── data/                # 数据读取、清洗、校验、切分
│   ├── features/            # 特征工程
│   ├── models/              # 训练、预测、CV、评估、输出
│   └── utils/               # IO、日志、指标
├── Makefile
├── requirements.txt
└── AGENTS.md
```

## 2. 数据准备

请将比赛数据放到 `data/raw/`：

```text
data/raw/used_car_train_20200313.csv
data/raw/used_car_testB_20200421.csv
```

可选文件：

```text
data/raw/used_car_testA_20200313.csv
data/raw/used_car_train_first50000_correct.csv
```

重要：原始数据必须使用单空格分隔读取。

```python
pd.read_csv(path, sep=" ")
```

不要使用：

```python
pd.read_csv(path, sep=r"\s+")
```

原始文件中存在空字段，错误分隔符会造成字段错位，导致本地 CV、OOF、误差分析和线上结论失真。

## 3. 安装依赖

```bash
make setup
```

等价于：

```bash
python -m pip install -r requirements.txt
```

主要依赖：

- pandas
- numpy
- scikit-learn
- lightgbm

## 4. 常用命令

```bash
make data
make features
make train
make predict
make submit
```

命令说明：

- `make data`：读取 `data/raw/`，清洗后输出到 `data/interim/`。
- `make features`：构造特征，输出到 `data/processed/`。
- `make train`：训练模型、生成 OOF、生成测试预测、写出标准 `outputs/` 产物。
- `make predict`：当前复用训练入口生成预测产物。
- `make submit`：从 `outputs/predictions/test_predictions.csv` 生成提交文件。
- `make clean`：清理 Python 缓存和 notebook checkpoint。

## 5. 标准输出

运行 `make train` 或 `make predict` 后，标准产物如下：

```text
outputs/
├── models/
│   ├── baseline_model.pkl
│   └── best_model.pkl
├── predictions/
│   ├── valid_predictions.csv
│   └── test_predictions.csv
├── submissions/
│   ├── submission_001_baseline.csv
│   └── submission_002_improved.csv
└── reports/
    ├── eda_report.md
    ├── feature_report.md
    ├── error_analysis.md
    └── model_summary.md
```

说明：

- `valid_predictions.csv` 包含 `SaleID`, `price`, `prediction`, `abs_error`。
- `test_predictions.csv` 包含 `SaleID`, `price`。
- `submission_001_baseline.csv` 和 `submission_002_improved.csv` 当前由同一标准预测生成。
- `baseline_model.pkl` 和 `best_model.pkl` 当前都保存最近一次标准训练入口的全量模型。

## 6. 当前默认训练结果

最近一次默认标准训练：

- 模型：LightGBM
- target mode：`log1p(price)`
- CV：`repeated_stratified`, 5 folds x 3 repeats
- fold MAE：`592.2295 ± 6.0433`
- aggregated OOF MAE：`578.8037`

注意：这是当前标准训练入口结果，不是历史线上最佳融合结果。

历史线上最佳来自 E017 8 模型 `power_age` 扩展融合：

- online MAE：`462.9080`
- OOF MAE：`475.7776`
- meta-CV MAE：`475.8581`

后续计划是将 E017 融合流程迁移到标准输出体系。

## 7. 文档入口

推荐阅读顺序：

1. `docs/project_state.md`
2. `docs/data_dictionary.md`
3. `docs/feature_design.md`
4. `docs/modeling_strategy.md`
5. `docs/experiment_log.md`

这些文档描述了当前项目状态、字段含义、特征设计、建模策略和实验结论。

## 8. GitHub 上传说明

`.gitignore` 已配置为不上传数据和产物：

- `data/raw/*`
- `data/interim/*`
- `data/processed/*`
- `outputs/models/*.pkl`
- `outputs/predictions/*.csv`
- `outputs/submissions/*.csv`
- `outputs/reports/*`
- `outputs/test/*`
- `__pycache__/`

只保留目录占位文件 `.gitkeep`，便于克隆后保留标准结构。

提交前建议检查：

```bash
git status --short --ignored
```

确认大数据、模型、预测文件和提交文件都显示为 ignored。

## 9. 后续方向

优先事项：

1. 将 E017 融合流程迁移到标准 `outputs/` 结构。
2. 增加模型注册与晋级机制，区分 baseline、candidate、best。
3. 将 `make predict` 改成加载 `outputs/models/best_model.pkl` 直接预测。
4. 固化高价老车误差切片到 `outputs/reports/error_analysis.md`。
5. 继续验证 `power_age`、target transform、smoothing 多样性的融合收益。
