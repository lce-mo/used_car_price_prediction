# 二手车预测实验日志

## 记录规则

- 本日志从“正确读取口径”之后开始作为主日志。
- 旧日志中的本地 MAE 多数基于错误读取口径，不再作为决策依据。
- 本地 MAE 优先记录 `aggregated OOF MAE`；如需要同时记录 fold mean，则写成 `OOF / fold`。
- 线上 MAE 只记录真实提交成绩。
- “变化”默认相对上一条主线线上成绩或当前基准说明。

## 实验总表

| 日期 | ID | 版本 | 核心改动 | 本地MAE | 线上MAE | 变化 | 结论 |
|---|---|---|---|---:|---:|---|---|
| 4/28 | E001 | parser_fix | 修正原始数据读取：`sep=r"\s+"` 改为 `sep=" "`，重建正确口径四模型融合 | 479.80 | 468.1260 | 相比旧融合线上 `726.6409`，线上 -258.5149 | 主线，验证体系重建 |
| 4/28 | E002 | correct_log_s50 | 正确口径全量单模型：`log1p + model TE s50 + group stats` | 489.29 / fold 515.26 | - | 新 baseline | 保留 |
| 4/28 | E003 | correct_4blend | 正确口径四模型融合：`0.175 log_s50 + 0.425 sqrt_s50 + 0.2 log_s10 + 0.2 log_s20` | 479.80 | 468.1260 | 相比 E002 本地 -9.50 | 当前线上最优 |
| 4/28 | E004 | health_50k | 50k 体检：特征健康、重要性、消融；确认 group stats 和 model TE 仍有效 | 555.25 baseline | - | 发现 `s20` 优于 `s50` | 诊断完成 |
| 4/28 | E005 | smoothing_50k | 50k smoothing 网格：`s5/s10/s15/s20/s30/s50/s100` | best `s10=551.21` | - | 相比 50k `s50=555.25`，本地 -4.04 | `s10` 进入候选 |
| 4/28 | E006 | error_deep_dive | 基于当前最佳 OOF 做误差深挖：价格×车龄、品牌、车型、功率、里程、误差集中度 | 479.80 | 468.1260 | 定位主误差来源 | 下一步做高价低估校准 |
| 4/29 | E007 | pred_age_add_calib | `pred_bucket x age_bucket` 加法残差校准，5-fold meta-CV | 479.69 | - | 本地 -0.11 | 收益过小，不提交 |
| 4/29 | E008 | pred_age_power_add_calib | `pred_bucket x age_bucket x power_bucket` 加法残差校准，5-fold meta-CV | 479.68 | - | 本地 -0.12 | 收益过小，不提交 |
| 4/29 | E009 | ratio_age_power_calib | `pred_bucket x age_bucket x power_bucket` 乘法 ratio 校准，5-fold meta-CV | 479.63 | - | 本地 -0.17 | 四组最佳，但不提交 |
| 4/29 | E010 | selective_uplift | 高价候选区域 selective uplift，5-fold meta-CV | 479.69 | - | 本地 -0.11 | 收益过小，不提交 |
| 4/29 | E011 | s10_power_age_50k | 50k：`s10 + power_age`，增加 `power_per_age_year` 等功率车龄交互 | 549.74 | - | 相比 50k s10 本地 -1.47 | 进入全量候选 |
| 4/29 | E012 | s10_age_detail_50k | 50k：`s10 + age_detail`，增加车龄月数/新车标记 | 554.35 | - | 相比 50k s10 本地 +3.14 | 废弃 |
| 4/29 | E013 | s10_power_age_age_detail_50k | 50k：`s10 + power_age + age_detail` | 550.89 | - | 相比 50k s10 本地 -0.32 | 只改善切片，不进主线 |
| 4/29 | E014 | s10_model_age_group_stats_50k | 50k：`s10 + model_age_group_stats` | 550.24 | - | 相比 50k s10 本地 -0.97 | 候选但优先级低于 E011 |
| 4/29 | E015 | s10_power_age_full | 全量：`s10 + power_age`，生成 OOF 和 testB submission | 488.13 / fold 514.43 | 485.6350 | 相比 E003 线上 +17.5090；相比全量 s10 OOF -1.05 | 单模型不提交，作为融合候选 |

## 关键基准

当前线上最优：

- ID：`E003`
- submission：`outputs/correct_final_four_model_blend_predict/submission.csv`
- online MAE：`468.1260`
- OOF MAE：`479.7973`

当前正确口径单模型基准：

- ID：`E002`
- output：`outputs/test/correct_full150000_log_s50_repeated_predict`
- OOF MAE：`489.2927`
- fold mean：`515.2566 ± 4.0556`

当前正确 50k 基准：

- output：`outputs/test/correct_first50000_log_s50_repeated`
- OOF MAE：`555.2518`
- fold mean：`584.6844 ± 4.8174`

## 当前误差结论

基于 `E003` 当前最佳融合 OOF：

- OOF MAE：`479.7973`
- 线上 MAE：`468.1260`
- top `10%` 样本贡献 `57.87%` 绝对误差。
- Q5 高价桶贡献 `48.82%` 绝对误差，Q5 MAE `1171.17`。
- 目标五组样本占比 `46.31%`，贡献 `66.89%` 绝对误差：
  - `Q5 x 8y_plus`
  - `Q4 x 8y_plus`
  - `Q5 x 5_8y`
  - `Q3 x 8y_plus`
  - `Q5 x 3_5y`
- 主要系统偏差是高价车低估，尤其 `Q5 x 8y_plus` 平均低估约 `757`。

## 后续计划

优先级 1：

- 基于 OOF 设计高价低估校准。
- 候选切片：`pred_price_bucket x age_bucket x power_bucket`。
- 目标：降低 Q5 / target5 MAE，同时不明显伤害 Q1/Q2。

优先级 2：

- 全量补跑 `s15` 或直接在融合池里验证 smoothing 多样性。
- 但在完成高价低估校准前，不优先继续 smoothing 网格。

优先级 3：

- 对 brand `24`、model `44` 做专项检查。
- 先做分析，不直接上专家模型。

## 4/29 高价老车校准实验结论

实验目标：

- 验证当前最佳模型是否能通过规则校准修正高价老车低估。
- 使用当前最佳融合 OOF：`outputs/correct_full150000_four_model_blend_search/best_blend_oof_predictions.csv`
- 基准 OOF MAE：`479.7973`
- 验证方式：5-fold meta-CV。每一折只用训练折学习校准规则，再应用到验证折。

输出目录：

- `outputs/correct_tail_calibration_stage1`
- 脚本：`src/evaluate_tail_calibration.py`

四组最佳结果：

| ID | 实验 | 最佳规则 | OOF MAE | Q5 MAE | target5 MAE | 结论 |
|---|---|---|---:|---:|---:|---|
| E007 | 加法：`pred_bucket x age_bucket` | `alpha=0.75`, `smoothing=0` | `479.6909` | `1170.2846` | `692.7154` | 有效但很小 |
| E008 | 加法：`pred_bucket x age_bucket x power_bucket` | `alpha=0.75`, `smoothing=1000` | `479.6810` | `1170.4203` | `692.7165` | 有效但很小 |
| E009 | 乘法：`pred_bucket x age_bucket x power_bucket` | `alpha=1.0`, `smoothing=1000`, `ratio_clip=0.95-1.05` | `479.6307` | `1170.1573` | `692.6064` | 本轮最佳 |
| E010 | selective uplift | `pred>=P90`, `age>=5`, `power>=120`, `alpha=1.0` | `479.6869` | `1170.5626` | `692.6604` | 有效但很小 |

最佳 E009 相对 E003：

- overall OOF：`479.7973 -> 479.6307`，改善 `0.1665`
- Q5 MAE：`1171.1699 -> 1170.1573`，改善 `1.0126`
- target5 MAE：`692.9142 -> 692.6064`，改善 `0.3078`

按标准切片重新比较：

| run | target5 MAE | Q5 MAE | Q4 MAE | Q1 MAE | Q2 MAE |
|---|---:|---:|---:|---:|---:|
| E003 base | `694.1257` | `1172.1342` | `553.9885` | `123.2663` | `207.7073` |
| E009 best calib | `693.8087` | `1171.0982` | `554.2770` | `123.1638` | `207.7905` |

结论：

- “高价老车低估”这个假设成立，所有四类规则都有正向但很小的 OOF 改善。
- 简单后处理校准的收益不足，最佳仅 `-0.17 MAE`，低于提交阈值。
- 暂不生成 submission，避免用线上次数验证微小收益。
- 下一步如果继续高价老车方向，应从训练/特征侧入手，而不是继续调简单校准规则。

## 4/29 高价老车特征实验 50k

实验目标：

- 验证高价老车低估是否来自模型缺少 `age x power`、更细车龄、或 `model x age` 统计表达。
- 使用正确 50k 子集：`data/raw/used_car_train_first50000_correct.csv`
- 新 50k 基准：`correct_first50000_log_s10_repeated`
  - fold MAE：`581.2692 ± 7.1309`
  - aggregated OOF MAE：`551.2101`
  - target5 MAE：`797.9309`
  - Q5 MAE：`1396.1065`

输出目录：

- `outputs/correct_high_price_old_feature_experiments_50k`

整体结果：

| ID | run | fold MAE | OOF MAE | 相比 s10 OOF | 结论 |
|---|---|---:|---:|---:|---|
| E011 | `correct_first50000_s10_power_age_repeated` | `580.0897 ± 8.0021` | `549.7409` | `-1.4692` | 最稳，进入全量候选 |
| E012 | `correct_first50000_s10_age_detail_repeated` | `584.0521 ± 4.4780` | `554.3491` | `+3.1390` | 明显负收益，废弃 |
| E013 | `correct_first50000_s10_power_age_age_detail_repeated` | `581.3846 ± 6.0960` | `550.8866` | `-0.3235` | 整体收益小 |
| E014 | `correct_first50000_s10_model_age_group_stats_repeated` | `580.2065 ± 6.3172` | `550.2437` | `-0.9664` | 有效但不如 E011 |

切片结果：

| run | target5 MAE | Q5 MAE | Q4 MAE | Q1 MAE | Q2 MAE |
|---|---:|---:|---:|---:|---:|
| `s10_base` | `797.9309` | `1396.1065` | `621.2235` | `134.9254` | `222.9898` |
| `power_age` | `795.4706` | `1392.3725` | `620.4384` | `134.6380` | `221.7911` |
| `age_detail` | `802.5740` | `1409.7480` | `621.0180` | `135.5660` | `224.3682` |
| `power_age + age_detail` | `795.4190` | `1392.6694` | `622.9027` | `135.9974` | `223.2595` |
| `model_age_group_stats` | `796.8612` | `1397.7233` | `617.8200` | `133.7152` | `222.5437` |

结论：

- `power_age` 是本轮最稳的特征方向：
  - overall OOF 改善 `1.47`
  - target5 改善 `2.46`
  - Q5 改善 `3.73`
  - Q1/Q2 也改善，没有低价桶代价
- `age_detail` 单独明显负收益，说明“更细车龄”没有帮助，可能只是增加噪声。
- `power_age + age_detail` 的 target5 略优于单独 `power_age`，但 overall 和 Q1/Q2 变差，不进主线。
- `model_age_group_stats` overall 有效，但 Q5 变差；它更像改善 Q4/低价桶，不是高价主线首选。

下一步：

- 优先全量复验 E011：`correct_full150000_s10_power_age_repeated_predict`。
- 如果全量 OOF 改善，并且 Q5/target5 仍改善，再把它加入融合池。
- 暂不推进 `age_detail`。

## 4/29 E015 全量复验：s10 + power_age

状态：完成。

输出：

- `outputs/test/correct_full150000_s10_power_age_repeated_predict`
- `outputs/correct_full150000_s10_power_age_error_slices`
- `outputs/correct_full150000_s10_power_age_analysis`

配置：

- train：`data/raw/used_car_train_20200313.csv`
- test：`data/raw/used_car_testB_20200421.csv`
- model：LightGBM
- `learning_rate=0.07`
- `n_estimators=2150`
- `num_leaves=287`
- `target_mode=log1p`
- `use_group_stats=true`
- `use_model_target_encoding=true`
- `target_encoding_smoothing=10`
- `use_power_age=true`
- CV：`repeated_stratified`, `3 folds x 3 repeats`
- `predict_test=true`

整体结果：

| run | fold MAE | OOF MAE |
|---|---:|---:|
| `s10_base` | `515.0496 ± 2.6956` | `489.1857` |
| `s10_power_age` | `514.4319 ± 3.0353` | `488.1344` |
| `s50_base` | `515.2566 ± 4.0556` | `489.2927` |

相对全量 `s10_base`：

- fold MAE 改善：`0.6177`
- aggregated OOF 改善：`1.0513`

切片对比：

| run | target5 MAE | target5 abs_err_share | Q5 MAE | Q4 MAE | Q1 MAE | Q2 MAE |
|---|---:|---:|---:|---:|---:|---:|
| `s10_base` | `710.5360` | `0.6704` | `1205.0052` | `565.6203` | `119.4977` | `208.7470` |
| `s10_power_age` | `706.9148` | `0.6684` | `1202.8496` | `563.4494` | `119.5034` | `208.6300` |
| `four_blend_E003` | `694.1257` | `0.6677` | `1172.1342` | `553.9885` | `123.2663` | `207.7073` |

结论：

- `power_age` 在全量上复现了 50k 的正收益。
- 它改善整体 OOF，也改善 target5 / Q5 / Q4，且 Q1/Q2 基本不受伤害。
- 单模型仍明显弱于当前四模型融合 E003，因此不应单独提交。
- 下一步应把 `s10_power_age` 作为第五个成员加入融合池，重新做 OOF 融合搜索。

线上结果：

- 提交文件：`outputs/test/correct_full150000_s10_power_age_repeated_predict/submission.csv`
- 线上 MAE：`485.6350`

对比：

| run | online MAE |
|---|---:|
| E003 四模型融合 | `468.1260` |
| E015 `s10_power_age` 单模型 | `485.6350` |

结论更新：

- E015 单模型线上明显弱于 E003，下降 `17.5090 MAE`。
- 这不否定 `power_age` 特征，因为 E015 的正确定位是“融合候选成员”，不是替代四模型融合的单模型。
- 后续不要再单独提交弱于融合池的单模型；应先做 OOF 融合搜索，再决定是否生成 submission。
