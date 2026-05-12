# 二手车预测实验日志

## 1. 记录规则

本日志记录正确读取口径之后的主要实验。旧的错误读取口径结论只能作为历史参考，不能作为当前建模判断依据。

记录要求：

- 本地指标优先记录 aggregated OOF MAE。
- 如需记录 fold mean，写成 `OOF / fold mean ± std`。
- 线上 MAE 只记录真实提交成绩。
- 每次实验必须说明核心改动、结果和下一步判断。
- 当前标准产物统一使用 `outputs/` 标准目录。

标准产物：

- 模型：`outputs/models/baseline_model.pkl`
- 当前模型：`outputs/models/best_model.pkl`
- 验证预测：`outputs/predictions/valid_predictions.csv`
- 测试预测：`outputs/predictions/test_predictions.csv`
- 提交文件：`outputs/submissions/submission_001_baseline.csv`
- 改进提交：`outputs/submissions/submission_002_improved.csv`
- 报告目录：`outputs/reports/`

## 2. 当前标准训练产物

最近一次标准 `make train` / `make predict` 输出：

| 项目 | 值 |
|---|---:|
| model | `lightgbm` |
| target mode | `log1p(price)` |
| CV | `repeated_stratified`, 5 folds x 3 repeats |
| fold MAE | `592.2295 ± 6.0433` |
| aggregated OOF MAE | `578.8037` |
| 验证预测 | `outputs/predictions/valid_predictions.csv` |
| 测试预测 | `outputs/predictions/test_predictions.csv` |
| 模型文件 | `outputs/models/best_model.pkl` |

注意：当前标准产物是工程标准化后的默认训练入口结果，不是历史线上最佳 E017 融合结果。

## 3. 历史实验总表

| ID | 版本 | 核心改动 | 本地 MAE | 线上 MAE | 结论 |
|---|---|---|---:|---:|---|
| E001 | parser_fix | 修正原始数据读取：`sep=r"\s+"` 改为 `sep=" "`，重建正确口径 | `479.80` | `468.1260` | 验证体系重建，错误读取时期结论废弃 |
| E002 | correct_log_s50 | 正确口径全量单模型：`log1p + model TE s50 + group stats` | `489.2927 / 515.2566 ± 4.0556` | - | 正确读取后的单模型基线 |
| E003 | correct_4blend | 四模型融合：`log_s50`, `sqrt_s50`, `log_s10`, `log_s20` | `479.7973` | `468.1260` | 曾为线上最优 |
| E004 | health_50k | 50k 特征健康、重要性、消融 | `555.25` | - | 确认 group stats 和 model TE 仍有效 |
| E005 | smoothing_50k | 50k smoothing 网格：`s5/s10/s15/s20/s30/s50/s100` | best `s10=551.2101` | - | `s10` 进入候选 |
| E006 | error_deep_dive | 基于当前最佳 OOF 做误差深挖 | `479.7973` | `468.1260` | 定位高价老车为主要误差来源 |
| E007 | pred_age_add_calib | `pred_bucket x age_bucket` 加法残差校准 | `479.6909` | - | 收益过小，不提交 |
| E008 | pred_age_power_add_calib | `pred_bucket x age_bucket x power_bucket` 加法校准 | `479.6810` | - | 收益过小，不提交 |
| E009 | ratio_age_power_calib | `pred_bucket x age_bucket x power_bucket` ratio 校准 | `479.6307` | - | 简单校准最佳但收益仍小 |
| E010 | selective_uplift | 高价候选区域 selective uplift | `479.6869` | - | 收益过小，不提交 |
| E011 | s10_power_age_50k | 50k：`s10 + power_age` | `549.7409` | - | `power_age` 进入全量候选 |
| E012 | s10_age_detail_50k | 50k：`s10 + age_detail` | `554.3491` | - | 明显负收益，废弃 |
| E013 | s10_power_age_age_detail_50k | 50k：`power_age + age_detail` | `550.8866` | - | 整体收益小，不进主线 |
| E014 | s10_model_age_group_stats_50k | 50k：`model_age_group_stats` | `550.2437` | - | 有效但优先级低于 `power_age` |
| E015 | s10_power_age_full | 全量：`s10 + power_age` | `488.1344 / 514.4319 ± 3.0353` | `485.6350` | 单模型不提交，作为融合候选 |
| E016 | five_model_power_age_blend | E003 四模型融合池加入 `s10_power_age` | `477.7254 / meta-CV 477.7597` | `465.4393` | 曾为线上最优 |
| E017 | priority1_power_age_extended_blend | 补齐多个 `power_age` 变体，8 模型融合 | `475.7776 / meta-CV 475.8581` | `462.9080` | 当前历史线上最优 |
| E018 | feature_power_age_model_age_full | 全量：`power_age + model_age + brand_age` 新特征与交叉 TE | `487.7743 / 514.7009 ± 3.4618` | - | 单模型候选，不替代 E017 |
| E019 | modular_feature_refactor_full_recheck | 模块化 `src/features/` 复跑 E018 配置 | `487.7743 / 514.7009 ± 3.4618` | - | 重构等价验证通过 |
| E020 | standard_outputs | 补齐标准模型、预测、提交和报告输出 | `578.8037 / 592.2295 ± 6.0433` | - | 工程产物标准化完成 |

## 4. 当前最佳结果

历史线上最佳仍为 E017：

- 线上 MAE：`462.9080`
- OOF MAE：`475.7776`
- meta-CV MAE：`475.8581`
- 结论：扩展 `power_age` 融合成员在线上和 OOF 上都成立。

E017 融合成员权重：

| member | weight |
|---|---:|
| `log_s50` | `0.0873` |
| `sqrt_s50` | `0.1823` |
| `log_s10` | `0.0854` |
| `log_s20` | `0.0694` |
| `log_s10_power_age` | `0.0847` |
| `sqrt_s50_power_age` | `0.2619` |
| `log_s20_power_age` | `0.1568` |
| `log_s50_power_age` | `0.0721` |

## 5. 关键结论

已验证有效：

- 正确数据读取是所有结论的前提。
- `model target encoding` 有效。
- `group stats` 有效。
- `log1p` 与 `sqrt` 目标变换存在互补。
- `power_age` 是当前最重要的有效特征方向。
- 多模型融合池多样性比单模型微调更重要。

已降级或暂缓：

- 错误读取时期的所有指标和结论。
- hard-routed segmented expert。
- 简单 OOF 后处理校准。
- `age_detail` 单独方向。
- `pow075 + power_age` 近期 50k 单模型结果偏弱。

## 6. 下一步判断

建议后续实验顺序：

1. 将 E017 融合流程迁移到标准输出目录，使其稳定产出：
   - `outputs/models/best_model.pkl`
   - `outputs/predictions/valid_predictions.csv`
   - `outputs/predictions/test_predictions.csv`
   - `outputs/submissions/submission_002_improved.csv`
2. 补充模型注册机制，区分 baseline、candidate、best。
3. 继续验证 `log_s30_power_age`、`log_s5_power_age`、`log_s15_power_age` 是否能进入融合池。
4. 固化高价老车误差切片进入 `outputs/reports/error_analysis.md`。
