# 2026-04-28：优先级 1/2 融合实验记录

## 目的

在已经验证 `log1p` 与 `sqrt(price)` 融合有效之后，继续执行两个优先级：

1. 对 `log1p + sqrt` 的融合权重做细搜。
2. 加入 `log1p` 不同 target encoding smoothing 成员，验证多模型融合是否继续降低 OOF MAE。

验证口径保持不变：

- 数据：全量 150000 训练集。
- CV：`repeated_stratified`，`3 folds x 3 repeats`。
- 模型：LightGBM，`learning_rate=0.07`，`n_estimators=2150`，`num_leaves=287`。
- 特征：`use_group_stats=true`，`use_model_target_encoding=true`。
- 主 baseline：`outputs/test/full150000_modelte_s50_repeated`。

## 优先级 1：log/sqrt 权重细搜

输入 OOF：

- `log_s50`：`outputs/test/full150000_modelte_s50_repeated/oof_predictions.csv`
- `sqrt`：`outputs/test/full150000_target_sqrt_repeated/oof_predictions.csv`

结果：

- 原 `0.5 * log_s50 + 0.5 * sqrt`：OOF MAE `456.4789`
- 细搜最优 `0.55 * log_s50 + 0.45 * sqrt`：OOF MAE `456.4064`

结论：

- 细搜有正收益，但幅度很小，约 `0.07 MAE`。
- 单纯调整 `log/sqrt` 比例已经接近边际收益区间。

## 优先级 2：加入 log1p smoothing 成员

新增全量训练：

| run | target encoding smoothing | fold MAE |
| --- | ---: | ---: |
| `outputs/test/full150000_log_s10_repeated_predict` | 10 | `486.1884 ± 7.3176` |
| `outputs/test/full150000_log_s20_repeated_predict` | 20 | `486.4410 ± 7.3455` |

四模型融合输入：

- `log_s50`
- `sqrt`
- `log_s10`
- `log_s20`

OOF 网格搜索：

- 搜索步长：`0.025`
- 组合数：`12341`
- 输出目录：`outputs/full150000_multi_blend_search`

最优权重：

| member | weight |
| --- | ---: |
| `log_s50` | `0.225` |
| `sqrt` | `0.375` |
| `log_s10` | `0.225` |
| `log_s20` | `0.175` |

结果：

| run | OOF MAE |
| --- | ---: |
| `log_s50` baseline | `462.5448` |
| `sqrt` | `465.2253` |
| `log_s10` | `462.4706` |
| `log_s20` | `462.9385` |
| `0.5 log_s50 + 0.5 sqrt` | `456.4789` |
| `0.55 log_s50 + 0.45 sqrt` | `456.4064` |
| four-model blend | `455.1386` |

Meta-CV 验证：

- 5 折 meta-CV MAE：`455.1662`
- 各折最优权重大体稳定，`sqrt` 权重集中在 `0.375` 附近。

这说明多模型融合的提升不是明显的 OOF 权重过拟合。

## 切片结果

目标五组：

- `Q5 x 8y_plus`
- `Q4 x 8y_plus`
- `Q5 x 5_8y`
- `Q3 x 8y_plus`
- `Q5 x 3_5y`

| run | target5 MAE | Q5 MAE | Q1 MAE | Q2 MAE |
| --- | ---: | ---: | ---: | ---: |
| `log_s50` baseline | `685.0101` | `1184.4244` | `62.4515` | `188.4038` |
| `0.5 log_s50 + 0.5 sqrt` | `673.1385` | `1160.1645` | `64.8171` | `190.0827` |
| four-model blend | `672.0718` | `1158.3115` | `64.0400` | `188.6714` |

结论：

- 四模型融合继续改善目标高误差切片。
- Q5 高价桶继续下降。
- Q1/Q2 低价桶没有明显恶化，Q2 基本回到 baseline 附近。

## 生成文件

- OOF 搜索结果：`outputs/full150000_multi_blend_search/blend_results.csv`
- OOF 搜索摘要：`outputs/full150000_multi_blend_search/blend_summary.json`
- 最优融合 OOF：`outputs/full150000_multi_blend_search/best_blend_oof_predictions.csv`
- meta-CV 选择记录：`outputs/full150000_multi_blend_search/blend_cv_choices.csv`
- 切片对比：`outputs/full150000_multi_blend_search/slice_comparison.csv`
- 最终提交：`outputs/final_multi_blend_log_s50_sqrt_s10_s20_predict/submission.csv`
- 带组件提交：`outputs/final_multi_blend_log_s50_sqrt_s10_s20_predict/submission_with_components.csv`

## 当前判断

优先级 1/2 都完成。真正有效的是优先级 2 的多成员融合，而不是单纯细调 `log/sqrt` 比例。

建议把 `outputs/final_multi_blend_log_s50_sqrt_s10_s20_predict/submission.csv` 作为下一次线上验证候选。若线上优于 `730.4803`，说明 smoothing 多样性确实提供了额外泛化收益；若线上持平或变差，则说明这部分 OOF 收益可能有一部分来自训练分布内的融合偏差。

## 线上结果

提交文件：

- `outputs/final_multi_blend_log_s50_sqrt_s10_s20_predict/submission.csv`

线上 MAE：

- `726.6409`

对比上一版：

- 上一版 `0.5 * log_s50 + 0.5 * sqrt`：`730.4803`
- 四模型融合：`726.6409`
- 线上提升：`3.8394 MAE`

结论：

- 这次 OOF 改善成功转化为线上收益。
- `log1p` 不同 smoothing 成员确实提供了有效多样性。
- 当前主线应从“单个模型继续硬挖特征”转为“稳定扩展高质量、多样但不过度相关的模型成员，再用 OOF 控制融合权重”。
