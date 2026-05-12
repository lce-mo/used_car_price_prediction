# 项目状态说明

## 1. 项目目标

本项目用于学习和实践二手车价格预测的完整机器学习流程，目标是根据车辆基础信息、匿名连续特征、车龄、里程、品牌、车型等字段预测二手车成交价格 `price`。

核心目标不是单纯堆叠特征刷分，而是建立可复盘的建模流程：每次实验都需要说明改动原因、验证假设、指标变化和下一步判断。

## 2. 核心指标与评价方式

主要评价指标是 MAE：

```text
MAE = mean(abs(y_true - y_pred))
```

当前使用三类评估口径：

- 本地 fold MAE：每折验证集 MAE，用于观察训练稳定性。
- aggregated OOF MAE：所有训练样本的 out-of-fold 预测聚合后计算 MAE，当前判断本地效果时优先看这个指标。
- 线上 MAE：提交比赛平台后得到的真实成绩，只记录真实提交结果。

当前标准训练产物中，最近一次默认 `make train` 的结果为：

- fold MAE：`592.2295 ± 6.0433`
- aggregated OOF MAE：`578.8037`
- 标准验证预测：`outputs/predictions/valid_predictions.csv`
- 标准测试预测：`outputs/predictions/test_predictions.csv`

历史最佳线上结果仍来自 E017 8 模型 `power_age` 扩展融合：

- 线上 MAE：`462.9080`
- OOF MAE：`475.7776`
- meta-CV MAE：`475.8581`

注意：标准目录中的当前 `baseline_model.pkl` / `best_model.pkl` 保存的是最近一次标准训练入口产物，不等同于历史 E017 融合最优模型。

## 3. 当前数据源

原始数据位于 `data/raw/`：

- `used_car_train_20200313.csv`：训练集，150000 行，包含 `price`。
- `used_car_testB_20200421.csv`：当前默认测试集，50000 行，不包含 `price`。
- `used_car_testA_20200313.csv`：历史测试集。
- `used_car_train_first50000_correct.csv`：50k 快速筛选训练子集。

最重要的数据读取约束：

```python
pd.read_csv(path, sep=" ")
```

不能使用 `sep=r"\s+"`。原始文件中存在空字段，正则空白分隔会破坏列对齐，导致 CV、OOF、切片分析和线上判断失真。

## 4. 数据流向与 Pipeline

当前项目支持两条相关流程。

### 4.1 文件式数据处理流程

```text
data/raw/
  -> data/interim/
  -> data/processed/
```

命令入口：

- `make data`：读取 raw，执行保守清洗，输出 `data/interim/train_clean.csv` 和 `data/interim/test_clean.csv`。
- `make features`：读取 clean 数据，构造基础特征，输出 `data/processed/train_features.csv` 和 `data/processed/test_features.csv`。

### 4.2 当前训练主流程

```text
data/raw/
  -> 内存中构造 PreparedData
  -> CV 训练与 OOF 预测
  -> 全量训练与 testB 预测
  -> outputs/ 标准产物
```

命令入口：

- `make train`：训练、OOF 评估、全量预测，并写出标准产物。
- `make predict`：当前复用训练入口生成预测产物。
- `make submit`：从标准测试预测生成标准提交文件。

标准输出目录：

- 模型文件：
  - `outputs/models/baseline_model.pkl`
  - `outputs/models/best_model.pkl`
- 预测文件：
  - `outputs/predictions/valid_predictions.csv`
  - `outputs/predictions/test_predictions.csv`
- 提交文件：
  - `outputs/submissions/submission_001_baseline.csv`
  - `outputs/submissions/submission_002_improved.csv`
- 报告文件：
  - `outputs/reports/eda_report.md`
  - `outputs/reports/feature_report.md`
  - `outputs/reports/error_analysis.md`
  - `outputs/reports/model_summary.md`

## 5. 当前瓶颈与注意事项

主要瓶颈：

- 高价老车仍是主要误差来源，尤其是高价格桶和长车龄组合。
- 单模型效果明显弱于融合结果，当前最强方向仍是多模型融合池多样性。
- `power_age` 特征族已经验证有效，但默认标准训练入口尚未等价复现 E017 融合配置。

工程注意事项：

- 原始数据读取必须固定为 `sep=" "`。
- target encoding 必须在 fold 内构造，不能用全量训练目标直接生成训练特征。
- `fold mean MAE` 和 aggregated `OOF MAE` 不同，不能混用。
- 当前 `outputs/models/baseline_model.pkl` 和 `outputs/models/best_model.pkl` 都保存最近一次全量训练模型；项目还没有独立的模型注册和最优模型晋级机制。
- 当前 `make predict` 会重新运行训练流程，不是加载 pickle 后快速推理。

## 6. 下一步计划

优先级建议：

1. 将 E017 8 模型融合配置迁移到标准输出体系，使历史最佳可以稳定产出到 `outputs/` 标准目录。
2. 为 `best_model.pkl` 增加明确晋级规则，例如只有 OOF MAE 或 meta-CV MAE 优于当前记录时才覆盖。
3. 将 `make predict` 改为加载 `outputs/models/best_model.pkl` 后直接预测，避免重复训练。
4. 继续围绕 `power_age`、target transform、smoothing 多样性扩展融合池。
5. 对高价老车误差切片形成固定报告，输出到 `outputs/reports/error_analysis.md`。
