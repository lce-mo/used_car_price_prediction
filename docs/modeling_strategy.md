# 建模策略说明

## 1. 模型选择

当前主模型是 LightGBM，备用模型是 GradientBoostingRegressor。

### 1.1 LightGBM

选择理由：

- 适合表格数据。
- 能处理非线性关系和高阶交互。
- 对缺失值和不同尺度数值特征较鲁棒。
- 训练速度适合多折 CV 和多实验迭代。

当前默认参数：

- `model_name`: `lightgbm`
- `learning_rate`: `0.08`
- `n_estimators`: `400`
- `num_leaves`: `63`
- `subsample`: `0.8`
- `colsample_bytree`: `0.8`
- `objective`: `regression`
- `target_mode`: `log1p(price)`

### 1.2 GBRT

GBRT 作为轻量备用模型保留，适合小样本 smoke test 或依赖 LightGBM 不可用时快速验证流程。

## 2. 目标变换

支持的目标变换：

- `price`：直接预测原始价格。
- `log1p`：训练 `log1p(price)`，预测后 `expm1` 还原。
- `sqrt`：训练 `sqrt(price)`，预测后平方还原。
- `pow075`：训练 `price^0.75`，预测后反变换。

历史结论：

- `log1p` 是稳定基础口径。
- `sqrt` 与 `log1p` 存在融合互补。
- `pow075 + power_age` 近期 50k 筛选偏弱，暂不优先推进。

## 3. 验证方式

当前验证体系包括：

- KFold：基础交叉验证。
- repeated stratified CV：当前默认 CV 策略。
- OOF：每个样本只使用不包含自身的 fold 模型预测。
- meta-CV：用于融合权重搜索，评估融合权重是否稳定。

当前默认 CV：

- `cv_strategy`: `repeated_stratified`
- `n_splits`: `5`
- `cv_repeats`: `3`
- `total_folds`: `15`
- `stratify_price_bins`: `5`

当前标准训练结果：

- fold MAE：`592.2295 ± 6.0433`
- aggregated OOF MAE：`578.8037`
- 标准验证预测：`outputs/predictions/valid_predictions.csv`

注意：

- fold mean MAE 和 aggregated OOF MAE 不是同一指标。
- 对线上潜力判断时，优先看 aggregated OOF MAE、融合 OOF MAE、meta-CV MAE 和真实线上 MAE。

## 4. 重要建模开关

### 4.1 特征开关

常用开关：

- `use_group_stats`
- `use_power_age`
- `use_age_detail`
- `use_model_age_group_stats`
- `use_brand_target_encoding`
- `use_model_target_encoding`
- `use_model_age_target_encoding`
- `use_model_backoff_target_encoding`

历史结论：

- `power_age` 是当前最重要的有效方向之一。
- `model target encoding` 和 `group stats` 仍然可信。
- `age_detail` 单独方向暂缓。

### 4.2 Target Encoding Smoothing

重要参数：

- `target_encoding_smoothing`
- `model_backoff_min_count`
- `model_low_freq_min_count`

历史结论：

- 不同 smoothing 之间有融合价值。
- `s10/s20/s50` 及其 `power_age` 变体都曾进入主线融合池。
- 近期 50k 筛选显示 `log_s30_power_age`、`log_s5_power_age`、`log_s15_power_age` 可以作为候选继续验证。

### 4.3 样本权重与专家模型

当前默认：

- `use_sample_weighting`: `false`
- `sample_weight_mode`: `none`
- `use_segmented_modeling`: `false`

历史结论：

- hard-routed segmented expert 曾明显恶化整体 MAE。
- 简单高价老车校准收益小，不宜作为主线。

## 5. 模型与预测输出路径

标准模型文件：

- `outputs/models/baseline_model.pkl`
- `outputs/models/best_model.pkl`

说明：

- 当前两者都保存最近一次标准训练入口的全量拟合模型。
- 项目尚未实现独立模型注册和自动晋级机制，因此 `best_model.pkl` 暂时代表“当前标准训练产物”，不是历史线上最优 E017 融合模型。

标准预测文件：

- `outputs/predictions/valid_predictions.csv`
  - 列：`SaleID`, `price`, `prediction`, `abs_error`
  - 用于本地 OOF 误差分析。
- `outputs/predictions/test_predictions.csv`
  - 列：`SaleID`, `price`
  - 用于生成提交文件。

标准提交文件：

- `outputs/submissions/submission_001_baseline.csv`
- `outputs/submissions/submission_002_improved.csv`

标准报告文件：

- `outputs/reports/model_summary.md`
- `outputs/reports/error_analysis.md`
- `outputs/reports/feature_report.md`
- `outputs/reports/eda_report.md`

## 6. 下一步建模方向

建议优先级：

1. 将 E017 8 模型融合迁移到标准输出体系。
2. 增加模型注册表，明确 baseline、candidate、best 的晋级标准。
3. 让 `make predict` 加载 `outputs/models/best_model.pkl` 直接预测，减少重复训练。
4. 对 `power_age` 新 smoothing 候选做全量融合验证。
5. 固化高价老车误差切片报告，作为每次实验的必看指标。
