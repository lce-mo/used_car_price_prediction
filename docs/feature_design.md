# 特征设计说明

## 1. 总体思路

二手车价格由品牌车型、车龄、里程、车况、动力、地区和匿名连续特征共同决定。当前特征设计遵循三个原则：

1. 优先使用可解释的非目标特征。
2. 对价格 proxy 和 target encoding 保持 fold-safe，避免 target leakage。
3. 保留已经验证有效的 `power_age`、group stats、不同 target transform 和 smoothing 多样性方向。

当前标准特征报告输出：

- `outputs/reports/feature_report.md`

当前标准预测输出：

- `outputs/predictions/valid_predictions.csv`
- `outputs/predictions/test_predictions.csv`

## 2. 基础特征

基础特征直接来自原始字段：

- 数值/连续字段：`power`, `kilometer`, `v_0`-`v_14`
- 类别字段：`model`, `brand`, `bodyType`, `fuelType`, `gearbox`, `notRepairedDamage`, `regionCode`
- 低价值或常量倾向字段：`seller`, `offerType` 当前不进入最终模型特征。

处理方式：

- `power` 被裁剪到 `[0, 600]`。
- 额外生成 `power_outlier_flag` 和 `power_is_zero`。
- 类别字段统一缺失 token 后进行 ordinal encoding。
- 匿名字段 `v_0`-`v_14` 作为数值特征直接使用。

## 3. 时间与车龄特征

来源字段：

- `regDate`
- `creatDate`

构造字段：

- 注册日期拆分：`reg_year`, `reg_month`, `reg_day`, `reg_weekday`
- 创建日期拆分：`create_year`, `create_month`, `create_day`, `create_weekday`
- 车龄：`car_age_days`, `car_age_years`
- 车龄分桶：`0_1y`, `1_3y`, `3_5y`, `5_8y`, `8y_plus`

风险点：

- 日期可能非法，需要 `errors="coerce"`。
- 负车龄或异常车龄不能简单删除，当前主要由缺失处理和模型鲁棒性承接。

## 4. 统计与交叉特征

### 4.1 Count Encoding

常用频次特征：

- `name_count`
- `brand_count`
- `regionCode_count`
- `model_count`

这些特征描述样本在整体 train/test 分布中的稀有程度。它们不使用 `price`，因此不属于 target leakage。

### 4.2 Group Stats

非目标 group stats 使用可观测字段聚合：

- 按 `brand` 聚合 `power` 和 `kilometer`
- 按 `model` 聚合 `power`

示例字段：

- `brand_power_mean`
- `brand_power_median`
- `brand_kilometer_mean`
- `model_power_mean`

注意：这些统计只使用非目标字段，不使用 `price`。

### 4.3 Power-Age 特征

`power_age` 是当前已经验证有效的关键方向之一，用于表达车辆动力与车龄之间的相对关系。

代表字段：

- `power_per_age_year`
- `power_per_age_year_sqrt`
- `power_age_product`
- `power_age_ratio`
- `log_power_per_age`
- `kilometer_per_age_year`
- `log_kilometer_per_age`
- `age_kilometer_product`

历史实验显示，`power_age` 在单模型和融合中均有价值，尤其作为融合成员能够提供稳定互补信息。

## 5. Target Encoding 与 Price Proxy

target encoding 用于将高基数类别与价格关系表达为数值特征，例如：

- `brand_target_mean`
- `model_target_mean`
- `brand_age_target_mean`
- `model_age_target_mean`
- `model_backoff_target_mean`
- `model_power_age_backoff_target_mean`

必须遵守：

- 在 CV 内部，每个验证 fold 只能使用训练 fold 的目标统计。
- 对测试集预测时，可以使用全训练集拟合 encoder 后应用到测试集。
- 低频 `model` 需要回退到 `brand` 或全局统计，降低稀有类别噪声。

禁止：

- 用全量训练集目标直接生成训练行的 target encoding 后再做 CV。

## 6. 历史变动与结论

已验证有效：

- `model target encoding`
- `group stats`
- `target-mode log1p`
- `target-mode sqrt`
- 不同 smoothing 的模型融合
- `power_age` 特征族

阶段性降级：

- `age_detail` 单独方向表现偏负。
- hard-routed 高价老车专家模型曾恶化整体 MAE。
- 简单 OOF 后处理校准收益很小，不应作为主线。
- `pow075 + power_age` 近期 50k 单模型偏弱，暂不优先全量推进。

## 7. 当前标准产物关联

最近一次标准训练输出：

- 模型：`outputs/models/baseline_model.pkl`
- 当前模型：`outputs/models/best_model.pkl`
- 验证预测：`outputs/predictions/valid_predictions.csv`
- 测试预测：`outputs/predictions/test_predictions.csv`
- 特征报告：`outputs/reports/feature_report.md`
