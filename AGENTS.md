# 项目协作说明

## 1. 项目目标

本项目用于学习和实践天池二手车价格预测的完整机器学习流程。目标是根据车辆基础信息、匿名特征、车龄、里程、品牌、车型等字段预测二手车交易价格 `price`。

核心评价指标是 MAE：

```text
MAE = mean(abs(y_true - y_pred))
```

协作重点不是盲目堆特征或刷分，而是让每一步实验都能说明：

- 为什么这样做。
- 验证什么假设。
- 本地 OOF、fold MAE、线上 MAE 如何变化。
- 结果是否支持进入下一阶段。

## 2. 当前工程状态

项目已经从旧的扁平脚本结构重构为模块化结构。以下旧文件删除是预期行为：

- `src/features.py`
- `src/tune_lightgbm.py`
- `src/analyze_*.py`
- `src/evaluate_*.py`
- `src/summarize_experiments.py`
- `experiments/` 下的旧日志文件

旧实验日志已迁移到 `docs/过去日志/`，当前主文档位于 `docs/`。

当前主要模块：

- `src/config.py`：路径、随机种子、默认训练配置。
- `src/data/`：数据读取、清洗、校验、切分。
- `src/features/`：特征构造、类别处理、日期/车龄、power-age、target encoding。
- `src/models/`：训练、预测、CV、融合搜索、模型注册与标准输出。
- `src/utils/`：IO、日志、指标。
- `scripts/`：Makefile 调用的命令入口。
- `docs/`：项目状态、数据字典、特征设计、实验日志、建模策略。

## 3. 最重要的数据读取约束

原始数据必须使用单空格分隔读取：

```python
pd.read_csv(path, sep=" ")
```

禁止使用：

```python
pd.read_csv(path, sep=r"\s+")
```

原因：天池原始文件中存在空字段，使用正则空白分隔会破坏字段对齐，导致 CV、OOF、误差切片、线上判断全部失真。

当前代码中相关防护：

- `src/config.py` 中定义 `RAW_DATA_SEPARATOR = " "`。
- `src/data/validate_data.py` 和 `src/models/train_model.py` 会检查关键字段范围，识别疑似列错位。

## 4. 标准目录约定

### 4.1 输入数据

原始数据放在 `data/raw/`，但不上传 GitHub。

常用文件：

- `data/raw/used_car_train_20200313.csv`
- `data/raw/used_car_testB_20200421.csv`
- `data/raw/used_car_testA_20200313.csv`
- `data/raw/used_car_train_first50000_correct.csv`

### 4.2 中间数据

- `data/interim/`：清洗后的中间数据。
- `data/processed/`：特征处理后的数据。

这些目录只上传 `.gitkeep`，不上传生成的 CSV。

### 4.3 标准输出

标准输出目录如下：

```text
outputs/
  models/
    baseline_model.pkl
    best_model.pkl
  predictions/
    valid_predictions.csv
    test_predictions.csv
  submissions/
    submission_001_baseline.csv
    submission_002_improved.csv
  reports/
    eda_report.md
    feature_report.md
    error_analysis.md
    model_summary.md
  figures/
```

GitHub 只上传这些目录下的 `.gitkeep`，不上传模型、预测、提交或报告产物。

## 5. 常用命令

```bash
make setup
make data
make features
make train
make predict
make submit
make eda
make clean
```

命令说明：

- `make setup`：安装 `requirements.txt`。
- `make data`：读取 raw 并生成 `data/interim/train_clean.csv`、`data/interim/test_clean.csv`。
- `make features`：生成 `data/processed/train_features.csv`、`data/processed/test_features.csv`。
- `make train`：训练、OOF 评估、全量预测，并写出标准 `outputs/` 产物。
- `make predict`：当前复用训练入口生成预测产物，不是快速加载 pickle 推理。
- `make submit`：从 `outputs/predictions/test_predictions.csv` 生成标准提交文件。
- `make clean`：清理 Python 缓存和 notebook checkpoint。

## 6. 实验记录规则

主实验日志：

- `docs/experiment_log.md`

旧日志：

- `docs/过去日志/`

记录要求：

- 线上 MAE 必须来自真实提交，不能伪造或推测。
- 本地效果优先记录 aggregated OOF MAE。
- 如记录 fold 指标，写成 `OOF / fold mean ± std`。
- 每个关键实验必须说明假设、配置、结果、结论和下一步判断。
- 错误读取数据时期的实验只能作为历史参考，不能作为当前结论依据。

## 7. 当前历史最佳

历史线上最佳是 E017 `priority1_power_age_extended_blend`：

- online MAE：`462.9080`
- OOF MAE：`475.7776`
- meta-CV MAE：`475.8581`

重要说明：

- 当前 `outputs/models/best_model.pkl` 保存的是最近一次标准训练入口的全量模型。
- 它目前不等同于 E017 历史最优融合模型。
- 后续需要将 E017 融合流程迁移到标准输出体系，并增加模型晋级机制。

## 8. 已验证有效方向

可信方向：

- `model target encoding`
- `group stats`
- `log1p` 与 `sqrt` target transform 的互补
- 不同 smoothing 的融合多样性
- `power_age` 特征族
- 多模型融合池扩展

暂缓方向：

- 错误读取时期的所有校准和专家模型结论
- hard-routed segmented expert
- 简单 OOF 后处理校准
- `age_detail` 单独方向
- `pow075 + power_age` 近期 50k 筛选偏弱

## 9. GitHub 上传规则

不上传：

- 原始数据：`data/raw/*`
- 中间数据：`data/interim/*`
- 特征数据：`data/processed/*`
- 模型：`*.pkl`, `*.joblib`, `*.model`
- 预测和提交：`outputs/predictions/*`, `outputs/submissions/*`
- 训练报告产物：`outputs/reports/*`
- 大量实验输出：`outputs/test/*`, `outputs/main_lgbm_m2_q3/*`
- Python 缓存：`__pycache__/`, `*.pyc`

可以上传：

- 源码
- 脚本入口
- 文档
- `.gitkeep` 目录占位文件
- `requirements.txt`
- `Makefile`
- `.gitignore`

提交前必须检查：

```bash
git status --short --ignored
git check-ignore -v data/raw/used_car_train_20200313.csv
git check-ignore -v outputs/models/best_model.pkl
git check-ignore -v outputs/predictions/test_predictions.csv
```

## 10. 协作约束

- 不允许修改、覆盖或删除 `data/raw/` 中的原始数据。
- 不允许为了跑通流程删除核心逻辑。
- 新增复杂逻辑应放在 `src/`，`scripts/` 只做入口。
- 新路径必须从 `src/config.py` 获取，避免写死绝对路径。
- 关键实验和工程变更需要同步更新 `docs/`。
- 不伪造实验指标；未知指标使用 `pending` 或 `null`。
