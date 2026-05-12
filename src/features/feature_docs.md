# 特征工程文档

本文档说明 `src/features/` 下的模块化特征工程。默认接口风格是：

```text
DataFrame -> DataFrame
```

每个函数复制输入表，新增或规范化特征列，然后返回新的 `DataFrame`。

## 类别特征

| 特征名 | 来源字段 | 计算方法 | 风险 / 注意事项 |
|---|---|---|---|
| `model` | 原始 `model` | 转为稳定字符串 key，缺失填充为 `__MISSING__` | 旧训练主线也会把它转成数值；做类别编码和 target encoding 前要保持 key 口径一致。 |
| `brand` | 原始 `brand` | 转为稳定字符串 key，缺失填充为 `__MISSING__` | 会参与 group stats 和 target encoding；类型不一致会导致分组变化。 |
| `bodyType` | 原始 `bodyType` | 转为稳定字符串 key，缺失填充为 `__MISSING__` | 缺失值要显式保留，不能静默变成普通数值缺失。 |
| `fuelType` | 原始 `fuelType` | 转为稳定字符串 key，缺失填充为 `__MISSING__` | 后续 ordinal encoding 不代表真实大小顺序。 |
| `gearbox` | 原始 `gearbox` | 转为稳定字符串 key，缺失填充为 `__MISSING__` | 同样不能把编码大小理解成业务顺序。 |
| `notRepairedDamage` | 原始 `notRepairedDamage` | 将 `-` 和空字符串视为缺失，将 `0.0/1.0` 规范为 `0/1` | 原始 `-` 代表缺失，不应作为普通类别直接建模。 |
| `regionCode` | 原始 `regionCode` | 转为稳定字符串 key，缺失填充为 `__MISSING__` | 高基数类别；target encoding 时尤其容易泄漏或过拟合。 |
| `name_count` | 原始 `name` | 统计当前输入表中相同 `name` 的出现次数 | 如果 train/test 合并后计算，属于 transductive 统计，实验记录里要写清楚口径。 |
| `brand_count` | 原始 `brand` | 统计相同 `brand` 的出现次数 | 非目标统计，不泄漏价格，但受计算范围影响。 |
| `regionCode_count` | 原始 `regionCode` | 统计相同 `regionCode` 的出现次数 | 高基数频数特征，train/test 计算范围变化会改变取值。 |
| `model_count` | 原始 `model` | 统计相同 `model` 的出现次数 | 被 model-age group stats 和低频 model 逻辑复用。 |
| `model_frequency_bin` | `model_count` | 将 model 频数分桶为 `f1`、`f2`、`f3_5`、...、`f51_plus` | 用于 CV 分层或误差切片复用；默认不加入训练特征，避免改变主线特征集合。 |

## 时间特征

| 特征名 | 来源字段 | 计算方法 | 风险 / 注意事项 |
|---|---|---|---|
| `reg_year` | `regDate` | 按 `YYYYMMDD` 解析注册日期，取年份 | 非法日期会变成缺失。 |
| `reg_month` | `regDate` | 按 `YYYYMMDD` 解析注册日期，取月份 | 同上。 |
| `reg_day` | `regDate` | 按 `YYYYMMDD` 解析注册日期，取日 | 同上。 |
| `reg_weekday` | `regDate` | 按 `YYYYMMDD` 解析注册日期，取星期 | 多数情况下只是弱代理特征。 |
| `create_year` | `creatDate` | 按 `YYYYMMDD` 解析上架日期，取年份 | 原始字段名拼写是 `creatDate`，特征名沿用旧主线的 `create_*`。 |
| `create_month` | `creatDate` | 按 `YYYYMMDD` 解析上架日期，取月份 | 注意字段名与特征名前缀不同。 |
| `create_day` | `creatDate` | 按 `YYYYMMDD` 解析上架日期，取日 | 非法日期会变成缺失。 |
| `create_weekday` | `creatDate` | 按 `YYYYMMDD` 解析上架日期，取星期 | 主要是上架时间代理。 |
| `car_age_days` | `creatDate`, `regDate` | `(creatDate - regDate).days` | 为保持旧主线等价，负车龄不在这里强行修正；后续交互特征按需 clip。 |
| `car_age_years` | `car_age_days` | `car_age_days / 365.25` | 核心车龄特征，会被交互、分桶、切片、target encoding 共用。 |
| `car_age_months` | `car_age_days` | `clip(car_age_days, lower=0) / 30.4` | 可选 age-detail 特征；不是默认主线特征。 |
| `is_nearly_new_1y` | `car_age_years` | 车龄 clip 后小于等于 1 年记为 `1` | 小样本新车切片容易不稳定。 |
| `is_nearly_new_3y` | `car_age_years` | 车龄 clip 后小于等于 3 年记为 `1` | 需要关注 Q1/Q2 和年轻车切片是否受损。 |
| age bin 标签 | `car_age_years` | 分桶为 `0_1y`、`1_3y`、`3_5y`、`5_8y`、`8y_plus` | CV、误差切片、target encoding key 必须复用同一实现，避免口径漂移。 |

## 折旧与交互特征

| 特征名 | 来源字段 | 计算方法 | 风险 / 注意事项 |
|---|---|---|---|
| `power_outlier_flag` | 原始 `power` | `power < 0` 或 `power > 600` 记为 `1` | 必须在 power clip 前生成。 |
| `power` | 原始 `power` | 转为数值后 clip 到 `[0, 600]` | clip 是旧主线继承下来的建模假设。 |
| `power_is_zero` | clip 后 `power` | clip 后 power 等于 0 记为 `1` | 捕捉缺失或异常 power 的代理信号。 |
| `power_bin` | clip 后 `power` | 按 `0,60,90,120,150,200,300,600` 分桶 | 可选特征，也被 power-age target encoding key 使用。 |
| `kilometer_per_year` | `kilometer`, `car_age_years` | 正车龄下计算 `kilometer / car_age_years` | 保守折旧代理；模块化流水线可选，不是旧主线默认核心。 |
| `age_kilometer_interaction` | `kilometer`, `car_age_years` | `clip(age, lower=0) * kilometer` | 捕捉累计使用程度。 |
| `is_old_car_10y` | `car_age_years` | 车龄大于等于 10 年记为 `1` | 粗粒度老车标记，需观察高龄切片表现。 |
| `power_per_age_year` | `power`, `car_age_years` | `power / (clip(age, lower=0) + 1)` | E017 相关有效方向，主要作为融合成员特征族。 |
| `power_per_age_year_sqrt` | `power`, `car_age_years` | `power / sqrt(clip(age, lower=0) + 1)` | 对年龄分母做缓和。 |
| `power_age_product` | `power`, `car_age_years` | `power * clip(age, lower=0)` | 可能产生较大值，inf 会替换为缺失。 |
| `power_age_ratio` | `power`, `car_age_years` | 与 `power_per_age_year` 相同的旧主线比例特征 | 为保持旧实验等价保留，存在冗余。 |
| `log_power_per_age` | `power`, `car_age_years` | `log1p(clip(power, lower=0)) / (age + 1)` | 相比原始 power 比率更抗极端值。 |
| `kilometer_per_age_year` | `kilometer`, `car_age_years` | `kilometer / (age + 1)` | 同年龄车的使用强度代理。 |
| `log_kilometer_per_age` | `kilometer`, `car_age_years` | `log1p(clip(kilometer, lower=0)) / (age + 1)` | 更稳健的使用强度代理。 |
| `age_kilometer_product` | `kilometer`, `car_age_years` | `clip(age, lower=0) * kilometer` | 旧 power-age 实验特征。 |
| `power_minus_brand_power_mean` | `power`, `brand_power_mean` | 当前 power 减去品牌平均 power | 依赖 group stats 先构建。 |
| `power_ratio_brand_power_mean` | `power`, `brand_power_mean` | 当前 power 除以品牌平均 power | 分母加 `1e-6`，inf 会替换为缺失。 |
| `kilometer_minus_brand_kilometer_mean` | `kilometer`, `brand_kilometer_mean` | 当前 mileage 减去品牌平均 mileage | 受 group stats 计算范围影响。 |
| `kilometer_ratio_brand_kilometer_mean` | `kilometer`, `brand_kilometer_mean` | 当前 mileage 除以品牌平均 mileage | 分母加 `1e-6`，inf 会替换为缺失。 |
| `brand_car_age_years_mean` | `brand`, `car_age_years` | 品牌内平均车龄 | 非目标统计，但计算范围要记录。 |
| `car_age_minus_brand_age_mean` | `car_age_years`, `brand_car_age_years_mean` | 当前车龄减去品牌平均车龄 | 对品牌内部车龄位置建模。 |

## 非目标分组统计特征

| 特征名 | 来源字段 | 计算方法 | 风险 / 注意事项 |
|---|---|---|---|
| `brand_power_mean` | `brand`, `power` | 品牌内 power 均值 | 不使用价格，不泄漏 target；但 train/test 合并计算时要记录口径。 |
| `brand_power_median` | `brand`, `power` | 品牌内 power 中位数 | 同上。 |
| `brand_kilometer_mean` | `brand`, `kilometer` | 品牌内 kilometer 均值 | 同上。 |
| `model_power_mean` | `model`, `power` | model 内 power 均值 | 同上。 |
| `model_age_count` | `model`, age bin | model-age 组合出现次数 | 依赖统一 age bin。 |
| `model_age_count_ratio` | `model_age_count`, `model_count` | `model_age_count / model_count` | 低频 model 上波动较大。 |
| `model_age_power_median` | model-age/model/global stats | 达到最小样本数时用 model-age power 中位数，否则回退 | 回退可降低小样本 group 不稳定。 |
| `model_age_kilometer_median` | model-age/model/global stats | 达到最小样本数时用 model-age mileage 中位数，否则回退 | 同上。 |
| `model_age_power_mean` | model-age/model/brand/global stats | power 均值层级回退 | 低频 model-age 组合需重点关注。 |
| `model_age_power_std` | model-age/model/brand/global stats | power 标准差层级回退 | 单样本 std 缺失会被填充。 |
| `model_age_kilometer_mean` | model-age/model/brand/global stats | mileage 均值层级回退 | 同上。 |
| `model_age_kilometer_std` | model-age/model/brand/global stats | mileage 标准差层级回退 | 单样本 std 缺失会被填充。 |
| `power_minus_model_age_power_mean` | `power`, `model_age_power_mean` | 当前 power 减去 model-age peer 均值 | 依赖 model-age group stats。 |
| `power_ratio_model_age_power_mean` | `power`, `model_age_power_mean` | 当前 power 除以 model-age peer 均值 | 分母加 `1e-6`，inf 会替换为缺失。 |
| `kilometer_minus_model_age_kilometer_mean` | `kilometer`, `model_age_kilometer_mean` | 当前 mileage 减去 model-age peer 均值 | 依赖 model-age group stats。 |
| `kilometer_ratio_model_age_kilometer_mean` | `kilometer`, `model_age_kilometer_mean` | 当前 mileage 除以 model-age peer 均值 | 分母加 `1e-6`，inf 会替换为缺失。 |

## Target Encoding / Price Proxy 特征

所有 target encoding 训练特征必须使用 fold-safe 口径：

1. 只在当前 fold 的训练行上拟合映射；
2. 将映射应用到当前 fold 的验证行；
3. 只有生成 holdout/test 特征时，才允许使用全量训练集拟合映射。

`target_space` 支持：

- `raw`：编码原始 `price`
- `log1p`：编码 `log1p(price)`
- `sqrt`：编码 `sqrt(price)`

| 特征名 | 来源字段 | 计算方法 | 风险 / 注意事项 |
|---|---|---|---|
| `brand_price_mean` / 自定义后缀 | `brand`, target | 按品牌做平滑均值编码 | 训练行必须 OOF 构造。 |
| `model_price_mean` / 自定义后缀 | `model`, target | 按 model 做平滑均值编码 | 高基数字段，smoothing 很重要。 |
| `brand_target_mean` | `brand`, target | 品牌级平滑 target mean | 必须记录 target_space。 |
| `model_target_mean` | `model`, target | model 级平滑 target mean | 有效但最容易因全量拟合导致验证泄漏。 |
| `brand_age_target_mean` | `brand`, age bin, target | 对 `brand|age_bin` 做平滑编码 | 组合 key 稀疏，需要监控 rare bin。 |
| `model_age_target_mean` | `model`, age bin, target | 对 `model|age_bin` 做平滑编码，并带回退 | 稀疏组合必须使用 backoff。 |
| `model_backoff_target_mean` | `model`, `brand`, target | model 计数足够时用 model 编码，否则回退到 brand 编码 | 最小计数阈值会影响特征行为。 |
| `model_power_age_backoff_target_mean` | `model`, power bin, age bin, target | 对 `model|power_bin|age_bin` 编码，并用 model/brand 回退 | 组合非常稀疏，只能 fold-safe 使用。 |
| `power_age_bin_target_mean` | power bin, age bin, target | 对 `power_bin|age_bin` 编码，并用 power-bin 回退 | 不依赖 model 的价格代理。 |
| `model_low_freq_flag` | model 频数 | 当前 fit rows 中 model 计数低于阈值记为 `1` | fold 内特征，不应为 CV 全局预先计算。 |

## 构建顺序

`build_features.py` 按以下顺序调用模块：

1. 数值字段规范化和 power 清洗；
2. 日期和车龄特征；
3. 频数统计特征；
4. 非目标 group stats；
5. 可选 brand-relative 特征；
6. 可选折旧与 power-age 交互特征；
7. 可选 age-detail 特征；
8. 可选 model-age group stats；
9. 可选轻量交互特征；
10. 只有显式请求时，才构造 fold-safe target encoding。

默认 `add_price_proxy_features()` 保持 no-op，避免误把全量价格统计加入训练特征导致 target leakage。
