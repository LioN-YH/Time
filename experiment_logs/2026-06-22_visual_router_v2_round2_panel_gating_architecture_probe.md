# Visual Router V2 Round2 panel-aware gating architecture probe

日志日期：2026-06-22 15:30:18 CST

## 目的

在 Round2 panel-wise pooling 35k screening 未支持高维 panel concat 升级后，设计并验证一个更轻量的 panel-aware gating / residual architecture probe，用于判断 panel 信息是否可以作为 `global_mean_patch` 的辅助调制信号，而不是直接替代或高维拼接 visual representation。

## 背景

当前 Round2 主线仍是 `spatial_panel_3view + film_mean_patch_aux`。已完成的 35k panel pooling screening 显示：

- selection raw-soft MAE 最优仍为 baseline `film_mean_patch_aux=0.310385`。
- `film_global_panel_mean_aux` 相对 baseline 的 selection raw-soft MAE delta 为 `+0.000578`。
- `film_panel_mean_aux` 相对 baseline 的 selection raw-soft MAE delta 为 `+0.001768`。
- panel variants 在 diagnostic/test_small 有局部收益，但 test_small 不能用于选择设计。
- selection 的 q5 high-error、PatchTST、CrossFormer strata 存在退化。

因此本步骤不继续直接高维 panel concat，不进入 65k，不做 full-scale validation，只在既有 35k panel pooling feature cache 上做小样本 architecture smoke。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/visual_router_v2_panel_gating.py`。
   - 定义 `film_panel_gated_mean_aux`：panel stack 生成 3 个 sigmoid gate，按 `panel_i - global` 形成归一化 residual，最终输出 768 维。
   - 定义 `film_panel_lowrank_aux`：2304 维 panel concat 只进入 256 维 bottleneck adapter，再输出 768 维 residual。
   - 定义 `film_panel_attention_aux`：三 panel token 做极小 softmax attention，输出 768 维 residual。
   - 三者均保留 `global_mean_patch` 作为主表示，`alpha` 默认 0.1，RevIN aux 后续仍按 FiLM 注入。

2. 新增 `visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_panel_gating.py`。
   - 默认读取 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_35k_features/round2_panel_pooling_35k_feature_manifest.csv`。
   - 默认使用 `round2_selection_small` 的 32 条样本。
   - 从既有 `.npz` shard 读取 `global_mean_patch`、`panel_mean_concat`、`revin_aux`，将 `panel_mean_concat` reshape 为 `[B,3,768]`。
   - 只做 shape、finite、gate range、alpha、residual norm 和 visual delta norm 检查。

3. 使用 `quito` 环境完成语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/visual_router_v2_panel_gating.py \
     visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_panel_gating.py
   ```

4. 使用 `quito` 环境运行 small smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_panel_gating.py \
     --overwrite
   ```

5. 将轻量结果写入：
   - `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_gating_architecture_probe/`
   - `experiment_summaries/visual_router_v2_round2/panel_gating_architecture_probe.md`
   - `experiment_summaries/visual_router_v2_round2/panel_gating_metadata.json`

## 结果

smoke 输入与边界：

- sample_set：`round2_selection_small`
- max_samples：32
- `global_mean_patch_shape=[32,768]`
- `panel_mean_stack_shape=[32,3,768]`
- `revin_aux_shape=[32,6]`
- finite_check：`true`
- rerun_vit：`false`
- saved_pseudo_image_tensor：`false`
- trained_router：`false`
- full_scale_validation：`false`
- launched_65k：`false`
- test_small_used_for_design_selection：`false`

关键 smoke 指标：

| variant | visual shape | finite | alpha | visual_delta/global mean | panel_residual/global mean |
| --- | --- | --- | ---: | ---: | ---: |
| `film_mean_patch_aux` | `[32,768]` | true | n/a | 0.000000 | 0.091685 |
| `film_panel_gated_mean_aux` | `[32,768]` | true | 0.1 | 0.009169 | 0.091685 |
| `film_panel_lowrank_aux` | `[32,768]` | true | 0.1 | 0.000737 | 0.007371 |
| `film_panel_attention_aux` | `[32,768]` | true | 0.1 | 0.009169 | 0.091685 |

gate / attention 检查：

- `film_panel_gated_mean_aux` gate range 为 `[0.5,0.5]`，gate_range_valid=true。
- `film_panel_attention_aux` attention range 为 `[0.333333,0.333333]`，row_sum_max_abs_error=0。

## 结论

本步骤完成了 panel-aware gating / residual 的最小 architecture design 和 small smoke。三个候选均能在既有 35k panel pooling feature cache 上产生 finite 的 768 维 visual representation；gate/attention 范围合法；`alpha=0.1` 下 visual delta 相对 global norm 较小，未观察到表示尺度爆炸。

本结果只是 architecture smoke，不是性能结论。它证明 gating/residual 设计具备进入 35k small screening 的工程条件，但不支持直接进入 65k 或 full-scale。

## 下一步方案

1. 不启动 65k，不做 full-scale validation，不修改 full-scale pipeline。
2. 若继续推进，应另开目标只在 35k small screening 上比较 `film_panel_gated_mean_aux` 与 `film_mean_patch_aux` 的 selection raw-soft MAE/MSE/regret。
3. 后续 35k 若运行，必须重点审计 q5 high-error、PatchTST、CrossFormer strata，并继续禁止用 test_small 选择设计。
4. 只有 selection 主指标和关键 strata 同时不退化，才考虑更大规模验证。
