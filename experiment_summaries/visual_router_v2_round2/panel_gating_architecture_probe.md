# Visual Router V2 Round2 panel-aware gating architecture probe

生成时间：2026-06-22 15:29:50 CST

## 目的

本探针在不继续高维 panel concat 的前提下，验证 panel 信息能否作为 `global_mean_patch` 的轻量 gate / residual 调制信号。它只检查 architecture contract 和表示尺度，不给出 35k 性能结论。

## 背景

35k panel concat screening 中，selection raw-soft 主指标仍由 baseline `film_mean_patch_aux` 最优：`film_global_panel_mean_aux` 的 selection raw-soft MAE delta 为 +0.000578，`film_panel_mean_aux` 为 +0.001768。两种 panel concat 在 diagnostic/test_small 有局部收益，但 test_small 不能用于选择设计；selection 的 q5 high-error、PatchTST 与 CrossFormer strata 存在退化。因此直接用 2304/3072 维 panel concat 替代或扩展 visual representation 不进入 65k。

## 最小设计

- baseline `film_mean_patch_aux`：继续使用 768 维 `global_mean_patch`，RevIN aux 仍通过 FiLM 注入。
- `film_panel_gated_mean_aux`：panel stack 只生成 3 个 sigmoid gate；panel residual 为 `sum_i gate_i * (panel_i - global)` 的归一化加权和；最终 `visual = global + alpha * residual`，输出 768 维。
- `film_panel_lowrank_aux`：2304 维 panel concat 只进入 256 维 bottleneck adapter，再生成 768 维 residual，与 global residual merge。
- `film_panel_attention_aux`：三 panel token 只做极小 softmax attention，输出 768 维 residual，不引入复杂 transformer。

这些设计把 panel 信息限制为 residual 调制，保留 global mean_patch fallback，可降低高维 concat 带来的过拟合、尺度膨胀和 seed 间专家分配波动风险。

## Smoke 输入

- feature_manifest=/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_35k_features/round2_panel_pooling_35k_feature_manifest.csv
- sample_set=round2_selection_small，max_samples=32，loaded_samples=32。
- global_mean_patch_shape=[32, 768]，panel_mean_stack_shape=[32, 3, 768]，revin_aux_shape=[32, 6]。
- finite_check=True，saved_pseudo_image_tensor=false，rerun_vit=false，full_scale_validation=false。

## Smoke 结果

| variant | visual_shape | finite | alpha | visual_delta/global mean | panel_residual/global mean |
| --- | --- | --- | --- | ---: | ---: |
| film_mean_patch_aux | [32, 768] | True | n/a | 0.000000 | 0.091685 |
| film_panel_gated_mean_aux | [32, 768] | True | 0.10000000149011612 | 0.009169 | 0.091685 |
| film_panel_lowrank_aux | [32, 768] | True | 0.10000000149011612 | 0.000737 | 0.007371 |
| film_panel_attention_aux | [32, 768] | True | 0.10000000149011612 | 0.009169 | 0.091685 |

Gate / attention 检查：

- `film_panel_gated_mean_aux` gate range=[0.500000, 0.500000], mean=0.500000, valid=True。
- `film_panel_attention_aux` attention range=[0.333333, 0.333333], row_sum_max_abs_error=0.000000e+00。

## 判断

small smoke 证明三个 panel-aware residual/gating candidate 都能在既有 35k panel pooling cache 上产生 finite 的 768 维 visual representation，gate/attention 范围合法，`alpha=0.1` 下 visual delta 相对 global norm 较小，未观察到表示尺度爆炸。

下一步不应直接进入 65k 或 full-scale。若继续推进，建议另开目标只在 35k small screening 上比较 `film_panel_gated_mean_aux` 与 baseline 的 selection raw-soft MAE/MSE/regret，并重点审计 q5 high-error、PatchTST、CrossFormer strata；只有 selection 主指标和关键 strata 同时不退化，才考虑更大规模验证。

