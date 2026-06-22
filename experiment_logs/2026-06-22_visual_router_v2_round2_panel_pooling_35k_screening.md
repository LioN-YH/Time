# Visual Router V2 Round2 panel pooling 35k small screening

日志日期：2026-06-22 14:56:17 CST

## 目的

在 Round2 35k small 样本上验证 `spatial_panel_3view` 的 panel-wise / view-region pooling 是否相对当前 global `mean_patch` baseline 带来稳定收益，并判断该方向是否进入 65k expanded validation。

## 背景

Round2 已确认 `spatial_panel_3view + film_mean_patch_aux` 是当前 mainline candidate。本步骤只在 exploration 分支执行 35k small screening，不做 full-scale validation，不启动 1M/116M 长跑，不修改 full-scale pipeline，不保存 pseudo image tensor。

候选固定为：

- `film_mean_patch_aux`：使用 `global_mean_patch`，作为当前 baseline。
- `film_panel_mean_aux`：使用 `panel_mean_concat`，只输入 line/fold/FFT panel mean concat。
- `film_global_panel_mean_aux`：使用 `global_plus_panel_mean`，保留 global fallback 并加入 panel-specific 信息。

## 操作

1. 修正 `probe_visual_router_v2_panel_pooling.py` 的 `--max-samples-per-set` 默认值，使默认读取完整 Round2 small manifest；显式传入整数时才做 smoke 限制。
2. 使用 `quito` 环境在 GPU0 构建 35k panel pooling feature cache：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_panel_pooling.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_35k_features \
     --artifact-prefix round2_panel_pooling_35k \
     --embedding-batch-size 64 \
     --shard-size 512 \
     --device cuda:0 \
     --local-files-only
   ```

3. 校验 feature manifest：四个 sample set 数量为 20000/5000/5000/5000，70 个 shard，finite check 通过，未保存 pseudo image tensor。
4. 在 35k screening 目录构建 prediction subset SQLite index，覆盖 35000 个 sample_key 和 175000 条五专家 prediction 记录。
5. 并行运行 9 个训练任务：3 个 variants × seeds 16/17/18 × 3 epochs，按 GPU0-3 轮转分配。
6. 运行 aggregate，生成 selection / diagnostic / test_small summary、selected_model counts、stratified summary 和 metadata。
7. 归档轻量结果到 `experiment_summaries/visual_router_v2_round2/panel_pooling_35k_*`，并更新 `WORKSPACE_STRUCTURE.md`。

## 结果

feature cache 输出目录：

- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_35k_features/`

training/screening 输出目录：

- `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_35k_screening/`

关键完整性结果：

- `round2_train_small=20000`
- `round2_selection_small=5000`
- `round2_diagnostic_balanced_small=5000`
- `round2_test_small=5000`
- feature 维度：global mean 768，panel concat 2304，global+panel 3072，RevIN aux 6。
- prediction index：35000 个 sample_key，175000 条五专家记录。
- 9 个 variant/seed task 均 `completed`，日志未发现 traceback/OOM/error 关键词。

selection raw-soft 结果：

| variant | MAE mean | MSE mean | regret mean | MAE std |
| --- | ---: | ---: | ---: | ---: |
| `film_mean_patch_aux` | 0.310385 | 3.329199 | 0.046935 | 0.008199 |
| `film_global_panel_mean_aux` | 0.310962 | 3.544773 | 0.047512 | 0.005712 |
| `film_panel_mean_aux` | 0.312153 | 3.580640 | 0.048703 | 0.005447 |

diagnostic/test_small 方向：

- diagnostic 上 panel variants raw-soft MAE/MSE 好于 baseline，但该集合只用于解释。
- test_small 上 panel variants raw-soft MAE 略好于 baseline，但 test_small 只允许 frozen screening，不能用于选择 variant。
- selection 上两种 panel variants 的 raw-soft MAE/MSE/regret 均未优于 baseline，因此不能升级。

关键 strata：

- selection 的 `oracle_model=PatchTST` 和 `error_gap_quantile=q5` 上，panel variants 相对 baseline 明显变差。
- `oracle_model=DLinear`、`ES` 和 `LOW_LOW_HIGH` 上有局部改善，但不足以抵消 selection overall 与 high-error tail 的退化。
- selected_model ratio 没有塌缩到单一专家，但 panel/global+panel 的 seed 间分配波动更明显。

## 结论

本轮 35k small screening 不支持将 panel-wise pooling 推进到 65k expanded validation。

最终判断为 B：Keep as side branch。

理由是 selection raw-soft MAE/MSE/regret 的主选择指标仍由 baseline `film_mean_patch_aux` 最优；panel variants 在 diagnostic/test_small 的局部收益不能作为升级依据，且 selection high-error tail 与 PatchTST strata 有退化。

full-scale 并行主线不受本结果影响，仍应继续使用 `spatial_panel_3view + film_mean_patch_aux`。

## 下一步方案

1. 本轮 panel-wise pooling 暂不进入 65k expanded validation。
2. 后续若继续探索 view-region 信息，优先考虑更轻量的 panel attention/gating，而不是直接用高维 panel concat 替代 current global mean fallback。
3. 保留本轮 35k feature cache、screening 输出目录和轻量 summary，供后续架构分支复核。
