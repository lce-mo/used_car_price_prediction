# 项目协作说明

## 1. 协作与记录规则

### 1.1 项目目标

这个项目的主要目标是通过二手车价格预测竞赛学习完整的机器学习建模流程。后续每一步实验都要说明为什么这样做、验证什么假设、结果如何解释，而不是只追求盲目堆特征或刷分。

### 1.2 实验记录

- 主实验日志使用 `experiments/experiment_log_2.md`。
- 旧日志和旧输出只能作为历史参考，不能直接作为当前结论依据。
- 每次有线上分数、关键 OOF 结果、重要失败方向或阶段性结论，都要写入实验日志。

### 1.3 输出目录

- 常规实验输出放在 `outputs/test/` 下。
- 融合搜索、最终预测等特殊输出可以放在 `outputs/` 下的独立目录。
- 目录名要能看出数据版本、核心特征/模型、CV 口径和是否生成预测文件。

## 2. 项目当前状态

### 2.1 最重要的状态变化：数据读取已修正

此前项目中存在严重的数据读取问题：原始数据应该使用单空格分隔读取，即 `pd.read_csv(path, sep=" ")`，不能使用 `sep=r"\s+"`。旧读取方式会破坏字段对齐，导致此前相当一部分 CV、OOF、误差切片和实验结论失真。

当前 `src/train.py` 已加入正确读取约束：

- 使用 `RAW_DATA_SEPARATOR = " "`。
- 通过 `validate_raw_dataframe` 检查价格、里程等关键字段，防止再次读错。
- 后续实验必须基于修正后的数据读取逻辑。

因此，所有“错误读取数据”时期的结论只能作为思路参考，不能作为现在的主判断依据。

### 2.2 当前最佳线上结果

当前最佳提交来自正确读取数据后的四模型融合：

- 实验版本：E003 `correct_4blend`
- 提交文件：`outputs/correct_final_four_model_blend_predict/submission.csv`
- 线上 MAE：`468.1260`
- OOF MAE：`479.7973`
- 融合成员与权重：
  - `0.175 * log_s50`
  - `0.425 * sqrt_s50`
  - `0.200 * log_s10`
  - `0.200 * log_s20`
- OOF 文件：`outputs/correct_full150000_four_model_blend_search/best_blend_oof_predictions.csv`

后续所有优化都应该以这个融合结果作为主线基准，而不是只和单模型对比。

### 2.3 当前可信基线

正确读取数据后的主要基线如下：

| 实验 | 输出目录 | OOF MAE | fold mean MAE | 备注 |
|---|---|---:|---:|---|
| `correct_log_s50` | `outputs/test/correct_full150000_log_s50_repeated_predict` | `489.2927` | `515.2566 ± 4.0556` | 修正读取后的单模型基线 |
| `correct_log_s10` | `outputs/test/correct_full150000_log_s10_repeated_predict` | `489.1857` | `515.0496 ± 2.6956` | 略优于 s50 |
| `correct_first50000_log_s10` | `outputs/test/correct_first50000_log_s10_repeated` | `551.2101` | `581.2692 ± 7.1309` | 中规模快速验证基线 |
| `correct_4blend` | `outputs/correct_full150000_four_model_blend_search` | `479.7973` | - | 当前主线最佳 OOF |

注意：`fold mean MAE` 和 `OOF MAE` 不是同一个指标。当前判断线上潜力时优先看 OOF MAE、融合后的 OOF MAE，以及最终线上 MAE。

### 2.4 已验证有效的方向

目前仍然可信、可以继续推进的方向：

- `model target encoding` 仍然有效。
- `group stats` 仍然有效。
- `target-mode log1p` 和 `sqrt` 可以形成互补。
- 不同 smoothing 的模型有融合价值，当前 `s10/s20/s50` 都进入过主线融合。
- 高价老车仍是主要误差来源：
  - Q5 价格桶贡献了约 `48.82%` 的绝对误差。
  - 重点五组贡献了约 `66.89%` 的绝对误差。
  - 重点五组是：
    - `Q5 × 8y_plus`
    - `Q4 × 8y_plus`
    - `Q5 × 5_8y`
    - `Q3 × 8y_plus`
    - `Q5 × 3_5y`

### 2.5 最近一个有效但未超过主线的方向

`s10 + power_age` 在全量复验中是有效的单模型改进：

- 输出目录：`outputs/test/correct_full150000_s10_power_age_repeated_predict`
- OOF MAE：`488.1344`
- fold mean MAE：`514.4319 ± 3.0353`
- 线上 MAE：`485.6350`

结论：它比普通 `s10` 单模型更好，但远弱于当前四模型融合 `468.1260`。因此它不应该单独作为提交主线，而应该作为新的融合候选成员继续验证。

### 2.6 已经降级或暂缓的方向

以下方向暂时不要作为优先主线：

- 基于错误读取数据时期的所有 OOF 校准、专家模型、软权重结论。
- hard-routed segmented expert，高价老车专家模型曾明显恶化整体 MAE。
- 简单 OOF 校准规则，正确数据下收益很小；最佳记录约为 OOF `479.6307`，只比四模型融合提升约 `0.1665`。
- `age_detail` 单独方向，中规模验证表现偏负。
- `s10 + power_age + age_detail` 虽有小幅变化，但低价桶不稳定。
- `model_age_group_stats` 在 5 万样本上整体有帮助，但 Q5 变差，优先级低于 `power_age` 融合扩展。
