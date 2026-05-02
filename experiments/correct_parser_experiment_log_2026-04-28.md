# 2026-04-28：正确读取口径后的实验日志

## 背景

之前的本地 OOF / CV 指标存在严重口径问题：部分实验曾使用 `pd.read_csv(path, sep=r"\s+")` 读取天池二手车原始数据。这个写法会把空字段附近的空白也当成分隔符，导致列错位，进而让 `price`、`kilometer`、`regionCode` 等字段出现不合理值。

正确读取方式：

```python
pd.read_csv(path, sep=" ")
```

因此，旧日志中的 `400+ / 450+ / 486` 等本地 MAE 不再作为实验判断依据。线上分数仍然有效，但本地验证体系必须重建。

## 当前原则

- 先修复和验证读取口径，再做模型优化。
- 旧 OOF、旧切片、旧融合搜索结果全部标记为“错误读取口径下的参考产物”，不能继续用于决策。
- 新实验必须同时记录：
  - 数据读取方式
  - 原始字段范围校验
  - CV MAE
  - OOF MAE
  - OOF 与真实 `price` 的一致性检查
  - 线上分数，如有提交

## 已确认事项

- `src/train.py` 使用 `RAW_DATA_SEPARATOR = " "`。
- `src/train.py` 已包含 `validate_raw_dataframe`，用于检查明显列错位：
  - `price < 0`
  - `kilometer > 15`
  - `bodyType / fuelType / gearbox` 超范围
  - `regionCode` 超范围
- `src/analyze_oof_slices.py` 使用 `sep=" "` 读取 train。
- `src/evaluate_oof_calibration.py` 使用 `sep=" "` 读取 train。
- `src/analyze_anomalies.py` 使用 `sep=" "` 读取 train。
- `src/analyze_price_age_loss.py` 使用 `sep=" "` 读取 train。
- `data/test/clean.py` 与 `data/test/diagnose.py` 使用 `sep=" "`。

## 实施计划

### Step 1：数据读取校验

目标：

- 用正确读取方式检查 train / testA / testB 的 shape 和关键字段范围。
- 确认本地统计与 `experiments/correct.md` 中的结论一致。

通过标准：

- train shape = `(150000, 31)`
- testB shape = `(50000, 30)`
- train `price min >= 0`
- `kilometer max <= 15`
- `bodyType / fuelType / gearbox` 不出现大规模异常取值

### Step 2：重建固定 50000 子集

目标：

- 基于原始 train 文件前 50000 行重建一个正确读取口径的固定子集。
- 保留旧文件，新增 corrected 文件，避免误删历史实验依赖。

计划输出：

- `data/raw/used_car_train_first50000_correct.csv`

### Step 3：smoke run

目标：

- 使用正确 50k 子集跑一个小样本 smoke，确认训练、CV、OOF、保存流程全部可用。

建议配置：

- `--train-path data/raw/used_car_train_first50000_correct.csv`
- `--train-sample-size 1000`
- `--n-splits 2`
- `--cv-repeats 1`
- `--cv-strategy repeated_stratified`
- `--predict-test false`

### Step 4：重跑正确 50k baseline

目标：

- 在正确读取口径下重新建立 50k baseline。
- 旧的 `first50000_modelte_s50_repeated` 不再作为 baseline。

建议配置：

- LightGBM
- `learning_rate=0.07`
- `n_estimators=2150`
- `num_leaves=287`
- `use_group_stats=true`
- `target_mode=log1p`
- `use_model_target_encoding=true`
- `target_encoding_smoothing=50`
- `repeated_stratified 3x3`

### Step 5：重跑全量 baseline

目标：

- 在正确读取口径下重新建立全量 baseline。
- 本地 MAE 应与线上 `700+` 在量级上更接近。

### Step 6：重新切片与融合

目标：

- 用正确 OOF 重新分析误差来源。
- 只有在正确 OOF 上复现有效后，才继续融合扩展。

优先复现：

- `log_s50`
- `sqrt_s50`
- `log_s10`
- `log_s20`

### 2026-04-28 Step 6：正确口径下复现四模型融合

状态：完成。

新增成员输出：

- `outputs/test/correct_full150000_sqrt_s50_repeated_predict`
- `outputs/test/correct_full150000_log_s10_repeated_predict`
- `outputs/test/correct_full150000_log_s20_repeated_predict`

融合输出：

- `outputs/correct_full150000_four_model_blend_search`
- `outputs/correct_full150000_four_model_blend_error_slices`
- `outputs/correct_final_four_model_blend_predict`

单模型结果：

| run | fold MAE | aggregated OOF MAE |
| --- | ---: | ---: |
| `log_s50` | `515.2566 ± 4.0556` | `489.2927` |
| `sqrt_s50` | `515.2788 ± 3.1902` | `489.5090` |
| `log_s10` | `515.0496 ± 2.6956` | `489.1857` |
| `log_s20` | `514.8378 ± 2.7084` | `488.8541` |

四模型融合搜索：

- 搜索步长：`0.025`
- 搜索组合数：`12341`
- 最优 OOF MAE：`479.7973`
- meta-CV MAE：`479.8267`

最优权重：

| member | weight |
| --- | ---: |
| `log_s50` | `0.175` |
| `sqrt_s50` | `0.425` |
| `log_s10` | `0.200` |
| `log_s20` | `0.200` |

Meta-CV 权重稳定性：

| meta_fold | valid_mae | w_log_s50 | w_sqrt_s50 | w_log_s10 | w_log_s20 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `475.3022` | `0.175` | `0.425` | `0.175` | `0.225` |
| 2 | `480.7008` | `0.175` | `0.425` | `0.200` | `0.200` |
| 3 | `484.8916` | `0.175` | `0.425` | `0.200` | `0.200` |
| 4 | `481.9227` | `0.200` | `0.425` | `0.200` | `0.175` |
| 5 | `476.3164` | `0.200` | `0.425` | `0.200` | `0.175` |

切片对比：

| run | target5 MAE | target5 abs_err_share | Q5 MAE | Q1 MAE | Q2 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `correct_log_s50` | `710.7830` | `0.6705` | `1204.9849` | `119.4533` | `208.8042` |
| `correct_four_blend` | `694.1257` | `0.6677` | `1172.1342` | `123.2663` | `207.7073` |

提交候选：

- `outputs/correct_final_four_model_blend_predict/submission.csv`

提交文件统计：

- rows：`50000`
- price min：`11.3163`
- price median：`3196.7905`
- price max：`91540.1338`

结论：

- 在正确读取口径下，四模型融合仍然明显优于单模型。
- OOF 从最佳单模型 `488.8541` 降到 `479.7973`，提升约 `9.06 MAE`。
- meta-CV 同步改善，说明这不是明显的 OOF 权重过拟合。
- 改善主要来自 Q5 高价桶与目标五组高误差切片。
- Q1 低价桶轻微恶化，需要线上验证确认是否可接受。

### 2026-04-28 Step 7：50k 中规模特征与参数体检

状态：完成。

目标：

- 以正确读取口径和当前线上有效模型为基准，快速检查现有特征、参数是否仍然可信。
- 本轮使用 `data/raw/used_car_train_first50000_correct.csv`，不生成 testB 提交。

输出：

- 特征健康报告：`outputs/correct_feature_health_50k`
- 特征重要性：`outputs/correct_feature_importance_50k_log_s50`
- 消融汇总：`outputs/correct_health_check_50k/ablation_summary.csv`
- `s20` 切片对比：`outputs/correct_health_check_50k/log_s20_slice_comparison.csv`

#### 数据与特征健康检查

配置：

- train：`data/raw/used_car_train_first50000_correct.csv`
- test：`data/raw/used_car_testB_20200421.csv`
- feature config：当前 baseline 特征，`use_group_stats=true`

结果：

- raw train shape：`(50000, 31)`
- raw testB shape：`(50000, 30)`
- prepared train features：`(50000, 44)`
- prepared test features：`(50000, 44)`
- numeric / categorical feature count：`37 / 7`
- target min / median / max：`11 / 3299 / 93900`
- `car_age_years` min / median / max：`0.27 / 12.08 / 25.24`
- `car_age_years` missing rate：`0.0750`
- raw `power > 600` rate：`0.0860%`
- `notRepairedDamage == "-"` rate：`15.9560%`

train/testB 分布差异：

- 最大 numeric standardized mean difference 约 `0.0103`。
- train/testB 分布非常接近，没有发现明显数据漂移。

结论：

- 未发现类似“列错位”的严重输入问题。
- 车龄、功率、缺失率都在可解释范围内。
- `features.py` 中 frequency / group statistics 是基于 train+test 拼接后的非标签统计，竞赛中可接受，但需要记住这是 transductive feature engineering；它没有使用 `price`，不属于标签泄漏。

#### 特征重要性检查

配置：

- 50k 全量 fit
- `target_mode=log1p`
- `use_group_stats=true`
- `use_model_target_encoding=true`
- `target_encoding_smoothing=50`
- LightGBM 参数同 baseline

Top gain features：

| rank | feature | gain_share |
| ---: | --- | ---: |
| 1 | `num__v_3` | `0.5647` |
| 2 | `num__v_0` | `0.1669` |
| 3 | `num__v_12` | `0.1609` |
| 4 | `num__v_10` | `0.0243` |
| 5 | `num__v_8` | `0.0147` |
| 6 | `num__car_age_days` | `0.0075` |
| 7 | `num__v_6` | `0.0054` |
| 8 | `num__v_1` | `0.0050` |
| 9 | `num__kilometer` | `0.0045` |
| 10 | `cat__notRepairedDamage` | `0.0045` |

观察：

- Top features 没有出现 `SaleID`、原始日期串等明显泄漏字段。
- `model_target_mean` gain share 只有约 `0.0011`，单看重要性不高，因此必须靠消融判断它是否有效。

#### 50k 消融结果

统一配置：

- train：`data/raw/used_car_train_first50000_correct.csv`
- CV：`repeated_stratified`, `3 folds x 3 repeats`
- LightGBM：`learning_rate=0.07`, `n_estimators=2150`, `num_leaves=287`
- 默认 baseline：`log1p`, `use_group_stats=true`, `use_model_target_encoding=true`, `target_encoding_smoothing=50`

| run | fold MAE | aggregated OOF MAE | 结论 |
| --- | ---: | ---: | --- |
| `correct_first50000_log_s20_repeated` | `582.2050 ± 5.6393` | `552.2890` | 当前 50k 最好 |
| `correct_first50000_log_s50_repeated` | `584.6844 ± 4.8174` | `555.2518` | baseline |
| `correct_first50000_no_group_stats_repeated` | `586.4587 ± 4.2369` | `556.2572` | 去掉 group stats 变差 |
| `correct_first50000_sqrt_s50_repeated` | `587.5206 ± 6.2285` | `557.1374` | sqrt 单模型变差，但融合仍可能有价值 |
| `correct_first50000_no_model_te_repeated` | `587.6160 ± 5.0739` | `557.7790` | 去掉 model TE 变差 |

相对 baseline：

- `smoothing=20`：fold MAE 改善 `2.4794`，aggregated OOF 改善 `2.9628`。
- 去掉 `group stats`：fold MAE 变差 `1.7743`。
- 去掉 `model target encoding`：fold MAE 变差 `2.9317`。
- `sqrt_s50` 单模型：fold MAE 变差 `2.8363`。

#### `s20` 切片对比

| run | target5 MAE | target5 abs_err_share | Q5 MAE | Q1 MAE | Q2 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `log_s50_baseline` | `804.2284` | `0.6664` | `1410.5663` | `135.2911` | `223.2398` |
| `log_s20` | `799.9718` | `0.6664` | `1397.1422` | `135.2163` | `223.6286` |

结论：

- `s20` 的提升主要来自 Q5 和目标五组高误差切片。
- Q1 基本不变，Q2 轻微变差但幅度很小。
- 这支持后续把 `s20` 作为更强单模型候选，而不是只作为融合成员。

#### 本轮体检结论

- 当前特征没有发现明显泄漏或列错位残留问题。
- `group stats` 和 `model target encoding` 在正确口径下仍然有效。
- `target_encoding_smoothing=20` 在 50k 上优于 `50`，值得作为下一轮全量主线候选。
- `sqrt` 单模型较弱，但全量融合中权重高，说明它的价值主要是误差多样性，不是单模型精度。
- 后续中规模实验建议优先围绕 `smoothing` 细化，而不是先大改特征。

### 2026-04-28 Step 8：50k smoothing 网格

状态：完成。

目标：

- 在正确 50k 子集上细化 `model target encoding` 的 smoothing。
- 固定除 smoothing 外的所有配置，判断车型历史均价应该被信任到什么程度。

统一配置：

- train：`data/raw/used_car_train_first50000_correct.csv`
- model：LightGBM
- `learning_rate=0.07`
- `n_estimators=2150`
- `num_leaves=287`
- `target_mode=log1p`
- `use_group_stats=true`
- `use_model_target_encoding=true`
- CV：`repeated_stratified`, `3 folds x 3 repeats`
- `predict_test=false`

输出：

- `outputs/correct_first50000_smoothing_grid/smoothing_grid_summary.csv`
- `outputs/correct_first50000_smoothing_grid/smoothing_slice_comparison.csv`

整体结果：

| smoothing | fold MAE | aggregated OOF MAE |
| --- | ---: | ---: |
| `s10` | `581.2692 ± 7.1309` | `551.2101` |
| `s15` | `581.4553 ± 5.9866` | `551.5935` |
| `s20` | `582.2050 ± 5.6393` | `552.2890` |
| `s30` | `582.3707 ± 6.0203` | `552.8523` |
| `s5` | `582.3849 ± 5.7255` | `552.6112` |
| `s100` | `583.4095 ± 5.7252` | `553.9009` |
| `s50` | `584.6844 ± 4.8174` | `555.2518` |

相对旧 50k baseline `s50`：

- `s10` fold MAE 改善 `3.4151`
- `s10` aggregated OOF 改善 `4.0417`
- `s15` fold MAE 改善 `3.2291`
- `s20` fold MAE 改善 `2.4794`

切片对比：

| smoothing | target5 MAE | target5 abs_err_share | Q5 MAE | Q1 MAE | Q2 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s10` | `797.9309` | `0.6660` | `1396.1065` | `134.9254` | `222.9898` |
| `s15` | `799.2116` | `0.6666` | `1395.8962` | `135.2491` | `222.8111` |
| `s20` | `799.9718` | `0.6664` | `1397.1422` | `135.2163` | `223.6286` |
| `s50` | `804.2284` | `0.6664` | `1410.5663` | `135.2911` | `223.2398` |

结论：

- `s10` 是本轮 50k smoothing 网格的整体最优。
- `s15` 与 `s10` 很接近，Q5 MAE 略低于 `s10`，但整体和 target5 不如 `s10`。
- `s5` 比 `s10/s15/s20` 差，说明过度信任车型均价会开始过拟合。
- `s50/s100` 明显偏保守，削弱了车型信息。
- `s10` 的提升没有通过牺牲 Q1/Q2 得到，低价桶基本稳定。

下一步建议：

- 全量优先跑 `correct_full150000_log_s10_repeated_predict`，但这个实验此前已完成：
  - fold MAE：`515.0496 ± 2.6956`
  - aggregated OOF MAE：`489.1857`
- 现在应补全量 `s15`，因为它在 50k 上和 `s10` 非常接近，可能提供新的融合多样性。
- 暂不优先全量跑 `s5/s30/s100`，除非后续需要更多融合成员。

### 2026-04-28 Step 9：当前最佳提交的 OOF 误差深挖

状态：完成。

背景：

- 当前线上最佳提交：`outputs/correct_final_four_model_blend_predict/submission.csv`
- 线上 MAE：`468.1260`
- 对应 OOF：`outputs/correct_full150000_four_model_blend_search/best_blend_oof_predictions.csv`
- OOF MAE：`479.7973`

输出：

- `outputs/correct_best_blend_error_deep_dive`
- 核心文件：
  - `deep_dive_summary.json`
  - `report.md`
  - `price_age_grid.csv`
  - `brand_summary.csv`
  - `model_summary_min200.csv`
  - `kilometer_bucket_summary.csv`
  - `power_bucket_summary.csv`
  - `top_abs_error_samples.csv`
  - `error_concentration.csv`

#### 误差集中度

| sample group | sample count | abs_err_share | min abs err |
| --- | ---: | ---: | ---: |
| top `0.1%` | `150` | `4.04%` | `12324.44` |
| top `0.5%` | `750` | `10.95%` | `6331.75` |
| top `1%` | `1500` | `16.56%` | `4706.50` |
| top `5%` | `7500` | `41.13%` | `2067.66` |
| top `10%` | `15000` | `57.87%` | `1272.60` |
| top `20%` | `30000` | `76.85%` | `650.30` |

结论：

- 误差非常集中。
- 前 `10%` 样本贡献接近 `58%` 的绝对误差。
- 后续优化应该围绕高误差群体，而不是平均用力。

#### 价格与车龄

目标五组：

- `Q5 x 8y_plus`
- `Q4 x 8y_plus`
- `Q5 x 5_8y`
- `Q3 x 8y_plus`
- `Q5 x 3_5y`

结果：

- sample share：`46.31%`
- abs_err_share：`66.89%`
- target5 MAE：`692.91`

Q5 高价桶：

- sample share：约 `20%`
- abs_err_share：`48.82%`
- Q5 MAE：`1171.17`

Top price-age cells：

| cell | n | MAE | abs_err_share | signed_err_mean | under_pred_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q5 x 8y_plus` | `8056` | `1381.01` | `15.46%` | `-757.40` | `63.60%` |
| `Q4 x 8y_plus` | `19042` | `583.62` | `15.44%` | `-121.57` | `55.75%` |
| `Q5 x 5_8y` | `9796` | `1029.56` | `14.01%` | `-294.67` | `55.59%` |
| `Q3 x 8y_plus` | `25838` | `342.93` | `12.31%` | `-23.03` | `53.34%` |
| `Q5 x 3_5y` | `6739` | `1031.69` | `9.66%` | `-171.44` | `50.87%` |

结论：

- 最大误差来源仍然是高价老车。
- 尤其 `Q5 x 8y_plus` 存在明显系统性低估，平均低估约 `757`。
- `Q4 x 8y_plus` 单样本 MAE 不如 Q5 高，但样本数大，因此总贡献也很高。

#### 品牌与车型

品牌层面：

| brand | n | MAE | abs_err_share | signed_err_mean |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `31480` | `424.26` | `18.56%` | `-62.56` |
| `4` | `16737` | `670.97` | `15.60%` | `-74.60` |
| `10` | `14249` | `713.38` | `14.12%` | `-116.69` |
| `1` | `13794` | `729.02` | `13.97%` | `-68.93` |
| `24` | `772` | `3073.98` | `3.30%` | `-492.42` |

车型层面，min_count >= 200：

| model | n | MAE | abs_err_share | signed_err_mean |
| ---: | ---: | ---: | ---: | ---: |
| `19` | `9573` | `573.53` | `7.63%` | `-108.06` |
| `4` | `8445` | `532.28` | `6.25%` | `-76.50` |
| `0` | `11762` | `375.64` | `6.14%` | `-47.29` |
| `40` | `4502` | `587.72` | `3.68%` | `-68.42` |
| `13` | `3762` | `660.70` | `3.45%` | `-108.33` |
| `44` | `2195` | `943.80` | `2.88%` | `-242.25` |

结论：

- brand `0/4/10/1` 是主要贡献品牌，既有样本多也有误差大。
- brand `24` 样本少但 MAE 极高，是高价值专项检查对象。
- model `19/4/0` 的总贡献最高，model `44` 单样本误差更大且明显低估。

#### 里程、功率、维修状态

里程：

- `14_plus` 样本最多，贡献 `51.47%` 绝对误差。
- 低里程样本单样本 MAE 更高：
  - `<=3` MAE：`854.81`
  - `3_6` MAE：`829.95`

功率：

| power_bucket | n | MAE | abs_err_share | signed_err_mean |
| --- | ---: | ---: | ---: | ---: |
| `120_150` | `29567` | `522.53` | `21.47%` | `-63.12` |
| `150_200` | `20194` | `736.06` | `20.65%` | `-83.28` |
| `200_300` | `10703` | `1097.06` | `16.32%` | `-154.13` |
| `300_600` | `2905` | `1985.04` | `8.01%` | `-348.87` |

维修状态：

- `notRepairedDamage=0.0` 贡献 `86.56%` 的绝对误差。
- 这主要是因为该组样本占比高、价格更高，不代表这个字段本身有问题。

#### 本轮误差分析结论

- 当前误差不是均匀分布，而是高度集中。
- 最主要影响样本是：
  - Q5 高价车
  - 8 年以上老车
  - 高价中老车 `Q5 x 5_8y / 3_5y`
  - 高功率车，尤其 `200+` power
  - 低里程高价车
  - 部分品牌和车型，如 brand `24`、model `44`
- 主要系统性偏差是高价车低估，尤其老车和高功率车。

下一步建议：

- 不急着继续 smoothing。
- 先围绕“高价车系统性低估”设计校准实验：
  - 按 `price_pred_bucket x age_bucket x power_bucket` 做 OOF 残差校准。
  - 或者只对 Q5 候选区域做轻量乘法/加法校准。
  - 必须用 OOF 验证，避免盲目把所有高价 test 预测抬高。

线上结果：

- 提交文件：`outputs/correct_final_four_model_blend_predict/submission.csv`
- 线上 MAE：`468.1260`

对比旧提交：

| submission | online MAE |
| --- | ---: |
| 旧四模型融合，错误读取口径产物 | `726.6409` |
| 正确读取口径四模型融合 | `468.1260` |

线上提升：

- `258.5149 MAE`

结论更新：

- 读取口径修复是当前最大收益来源。
- 旧 submission 虽然也做了融合，但因为训练 / 预测数据读取错位，线上表现严重受损。
- 正确读取口径下，本地 OOF `479.7973` 与线上 `468.1260` 已经处于同一量级，说明新的验证体系明显更可信。
- 后续实验必须全部基于正确读取口径和本日志的 baseline / blend 结果继续推进。

## 实验记录

### 2026-04-28 Step 0：停止旧融合扩展

状态：完成。

说明：

- 已停止继续做 `pow075 / pow090 / s5 / s100` 扩展实验。
- 刚才临时加入的 `pow090` target mode 已回滚。
- 新增实验产物若存在，暂不删除，只标记为旧验证口径下的无效参考，不进入后续决策。

### 2026-04-28 Step 1：数据读取校验

状态：完成。

读取方式：

```python
pd.read_csv(path, sep=" ")
```

关键结果：

| dataset | shape | SaleID range | key checks |
| --- | ---: | ---: | --- |
| train | `(150000, 31)` | `0-149999` | `price min=11`, `price median=3250`, `kilometer max=15`, `regionCode max=8120` |
| testA | `(50000, 30)` | `150000-199999` | `kilometer max=15`, `regionCode max=8121` |
| testB | `(50000, 30)` | `200000-249999` | `kilometer max=15`, `regionCode max=8120` |

结论：

- 正确读取后字段范围合理。
- 本地验证需要以这个读取口径为唯一标准。

### 2026-04-28 Step 2：重建固定 50000 子集

状态：完成。

输出：

- `data/raw/used_car_train_first50000_correct.csv`

校验结果：

- shape：`(50000, 31)`
- `SaleID range = 0-49999`
- `price min = 11`
- `price median = 3299`
- `kilometer max = 15`
- `regionCode max = 8103`

说明：

- 保留旧 `data/raw/used_car_train_first50000.csv`，避免破坏历史产物依赖。
- 后续正确口径下的 50k 实验统一使用 `used_car_train_first50000_correct.csv`。

### 2026-04-28 Step 3：smoke run

状态：完成。

输出：

- `outputs/test/correct_parser_smoke_1000`

配置：

- train：`data/raw/used_car_train_first50000_correct.csv`
- sample：`1000`
- model：LightGBM
- `n_estimators=100`
- `num_leaves=31`
- `target_mode=log1p`
- `use_group_stats=true`
- `use_model_target_encoding=true`
- `target_encoding_smoothing=50`
- CV：`repeated_stratified`, `2 folds x 1 repeat`

结果：

- fold MAE：`1061.9628`, `1248.3262`
- fold mean：`1155.1445 ± 93.1817`
- OOF MAE：`1155.1445`
- OOF rows：`1000`
- `cv_prediction_count min/max = 1/1`

结论：

- 正确读取口径下训练流程可跑通。
- 本地 MAE 已回到 `1000+` 的合理量级，不再出现旧口径下异常的 `400/500`。

### 2026-04-28 Step 4：正确 50k baseline

状态：完成。

输出：

- `outputs/test/correct_first50000_log_s50_repeated`
- `outputs/correct_first50000_log_s50_repeated_error_slices`

配置：

- train：`data/raw/used_car_train_first50000_correct.csv`
- model：LightGBM
- `learning_rate=0.07`
- `n_estimators=2150`
- `num_leaves=287`
- `target_mode=log1p`
- `use_group_stats=true`
- `use_model_target_encoding=true`
- `target_encoding_smoothing=50`
- CV：`repeated_stratified`, `3 folds x 3 repeats`
- `predict_test=false`

结果：

- fold MAE：
  - `582.4742`
  - `590.2674`
  - `578.8825`
  - `593.2086`
  - `578.9902`
  - `587.1194`
  - `583.6541`
  - `587.3485`
  - `580.2145`
- fold mean：`584.6844 ± 4.8174`
- aggregated OOF MAE：`555.2518`
- OOF rows：`50000`
- `cv_prediction_count min/max = 3/3`
- `oof_predictions.csv` 中的 `price` 与正确读取的原始 `price` 完全一致：
  - `price_mismatch_count = 0`

Top price x age cells：

| cell | n | MAE | abs_err_share | signed_err_mean | under_pred_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q5 x 8y_plus` | `2680` | `1593.7204` | `0.1538` | `-923.2882` | `0.6485` |
| `Q4 x 8y_plus` | `6283` | `648.6720` | `0.1468` | `-160.0159` | `0.5779` |
| `Q5 x 5_8y` | `3273` | `1241.9633` | `0.1464` | `-382.3768` | `0.5707` |
| `Q3 x 8y_plus` | `8555` | `385.5636` | `0.1188` | `-38.3322` | `0.5328` |
| `Q5 x 3_5y` | `2214` | `1260.6393` | `0.1005` | `-231.3724` | `0.5068` |

结论：

- 正确 50k baseline 已建立。
- 误差来源仍然集中在高价与老车组合，和旧口径下的方向大体一致。
- 但后续所有数值必须以本节结果为新起点，不能再引用旧 50k baseline。

### 2026-04-28 Step 5：正确全量 baseline

状态：完成。

输出：

- `outputs/test/correct_full150000_log_s50_repeated_predict`
- `outputs/correct_full150000_log_s50_repeated_error_slices`

配置：

- train：`data/raw/used_car_train_20200313.csv`
- test：默认 `data/raw/used_car_testB_20200421.csv`
- model：LightGBM
- `learning_rate=0.07`
- `n_estimators=2150`
- `num_leaves=287`
- `target_mode=log1p`
- `use_group_stats=true`
- `use_model_target_encoding=true`
- `target_encoding_smoothing=50`
- CV：`repeated_stratified`, `3 folds x 3 repeats`
- `predict_test=true`

结果：

- fold MAE：
  - `510.5645`
  - `513.9698`
  - `518.4917`
  - `521.7715`
  - `517.2616`
  - `512.5896`
  - `518.1802`
  - `508.1432`
  - `516.3372`
- fold mean：`515.2566 ± 4.0556`
- aggregated OOF MAE：`489.2927`
- OOF rows：`150000`
- `cv_prediction_count min/max = 3/3`
- `price_mismatch_count = 0`
- submission rows：`50000`
- submission price：
  - min：`13.6183`
  - median：`3175.4217`
  - max：`92896.6012`

说明：

- `fold mean` 是每个验证 fold 的 MAE 均值。
- `aggregated OOF MAE` 是 repeated CV 下同一样本多次 OOF 预测平均后的 MAE。
- 因为平均预测会降低方差，所以 aggregated OOF MAE 低于 fold mean，这是正常现象。

Top price x age cells：

| cell | n | MAE | abs_err_share | signed_err_mean | under_pred_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q5 x 8y_plus` | `8013` | `1416.5274` | `0.1547` | `-790.0656` | `0.6297` |
| `Q4 x 8y_plus` | `18836` | `600.6891` | `0.1542` | `-143.7968` | `0.5623` |
| `Q5 x 5_8y` | `9765` | `1056.8477` | `0.1406` | `-315.4601` | `0.5580` |
| `Q3 x 8y_plus` | `25888` | `349.9878` | `0.1234` | `-39.6270` | `0.5414` |
| `Q5 x 3_5y` | `6732` | `1064.2502` | `0.0976` | `-195.8119` | `0.5201` |

结论：

- 正确全量 baseline 已建立。
- 相比旧日志中的 `486.0493 ± 7.5572`，正确读取口径下 fold mean 上升到 `515.2566 ± 4.0556`。
- 主要误差来源仍然是高价老车组合，尤其 `Q5 x 8y_plus`、`Q4 x 8y_plus`、`Q5 x 5_8y`。
- 后续融合复现应以 `outputs/test/correct_full150000_log_s50_repeated_predict` 为新的 `log_s50` 基准。
