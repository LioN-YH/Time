# Visual Router V2 Round2d period continuity diagnostic

日志日期：2026-06-22 01:23:01 CST

## 目的

新增并运行 Round2d period continuity diagnostic，用轻微历史输入扰动检查周期相关 layout 的不连续风险，回答 hard top1 fold、top3fold、pseudo image、ViT embedding、router weight、selected model 和 fused prediction 是否发生跳变。

## 背景

Round2c 35k layout feature cache 与固定 FiLM screening 已完成，best layout 为 `spatial_panel_3view`。进入 65k expanded validation 前，需要先诊断 `current_rgb_3view` 的 hard top1 period fold 和 `top3fold_period_layout` 的 hard top-k period fold 是否对轻微扰动敏感。本步骤要求不训练新 router、不重建 35k feature cache、不做 65k validation、不保存大规模 pseudo image tensor。

## 操作

1. 新增脚本 `visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py`。
   - 支持 `--run-single`、`--aggregate-only`、`--overwrite`。
   - 支持 `--devices cuda:0,cuda:1,cuda:2,cuda:3`，默认按 layout × seed × sample_set 做进程级并行。
   - 单任务输出隔离到 `tasks/<layout>_seed<seed>_<sample_set>/`。
   - 复用 Round2c checkpoint、Round2c prediction SQLite、Round2 small manifest、Round2 layout registry、Quito 历史窗口 loader、冻结 ViT encoder 和 FiLMRouter。
   - 只在运行内生成 pseudo image / ViT embedding，不落盘大规模 tensor 或 embedding cache。

2. 使用 `quito` 环境做脚本编译检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py
   ```

3. 先运行 4 sample smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py \
     --run-single \
     --layout current_rgb_3view \
     --seed 16 \
     --sample-sets round2_selection_small \
     --max-samples-per-set 4 \
     --perturbation-sigma-list 0.001 \
     --num-perturbations 1 \
     --embedding-batch-size 2 \
     --device cuda:0 \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity_smoke \
     --overwrite
   ```

4. smoke 首次发现 SQLite 相对数组路径解析错误：误把 prediction array path 按 Round2c 输出目录解析。修复为通过 `--prediction-manifest-path` 的 parent 解析已有 Round2c SQLite 中的相对路径，不扫描 116M manifest。

5. 正式运行多 GPU 诊断：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/diagnose_visual_router_v2_round2_period_continuity.py \
     --sample-sets round2_selection_small round2_diagnostic_balanced_small \
     --max-samples-per-set 512 \
     --layouts current_rgb_3view,top3fold_period_layout \
     --seeds 16 17 18 \
     --devices cuda:0,cuda:1,cuda:2,cuda:3 \
     --perturbation-sigma-list 0.001,0.005,0.01 \
     --num-perturbations 3 \
     --embedding-batch-size 16 \
     --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity \
     --summary-copy-dir /home/shiyuhong/Time-visual-router-v2/experiment_summaries/visual_router_v2_round2/period_continuity \
     --overwrite
   ```

6. 正式运行后重跑 aggregation 修正 metadata 中 `devices_used` 字段，并强化 summary 对指定 strata 的文字回答。未重跑 worker。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_period_continuity/
```

轻量 summary 已复制到：

```text
experiment_summaries/visual_router_v2_round2/period_continuity/
```

必需输出均已生成：

- `round2_period_continuity_raw_results.csv`
- `round2_period_selection_stability.csv`
- `round2_period_image_continuity.csv`
- `round2_period_embedding_continuity.csv`
- `round2_period_router_weight_continuity.csv`
- `round2_period_fused_prediction_continuity.csv`
- `round2_period_stratified_summary.csv`
- `round2_period_high_change_examples.csv`
- `round2_period_continuity_metadata.json`
- `round2_period_continuity_summary.md`

覆盖范围：

- layouts：`current_rgb_3view`、`top3fold_period_layout`
- seeds：16、17、18
- sample_sets：`round2_selection_small`、`round2_diagnostic_balanced_small`
- 每个 layout × seed × sample_set 为 512 samples × 3 sigma × 3 perturbations = 4,608 rows
- 总 raw rows = 55,296
- devices_used = `cuda:0,cuda:1,cuda:2,cuda:3`

核心均值：

| layout | top1_changed | top3_jaccard | image_cos | mean_patch_cos | weight_js | selected_flip | fused_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current_rgb_3view | 0.095775 | 0.914157 | 0.012659 | 0.060437 | 0.020471 | 0.042969 | 0.003293 |
| top3fold_period_layout | 0.095775 | 0.914157 | 0.000262 | 0.005865 | 0.001715 | 0.037507 | 0.003129 |

metadata 关键约束：

- `round2_stage=period_continuity_diagnostic`
- `trained_new_model=false`
- `built_feature_cache=false`
- `ran_vit_for_embedding_diagnostic=true`
- `saved_pseudo_image_tensor=false`
- `used_test_small_for_selection=false`
- `loaded_116m_prediction_manifest_to_memory=false`
- `parallel_backend=process_per_layout_seed_sample_set`
- `period_soft_mixture_implemented=false`
- `next_step_recommendation=direct_65k_with_top3fold`

## 结论

1. `current_rgb_3view` 的 hard top1 fold 对轻微扰动敏感：top1 period changed ratio 约 9.58%，mean-patch embedding cosine distance 和 router JS divergence 明显高于 top3fold。
2. `top3fold_period_layout` 比 current hard top1 fold 更连续：周期选择本身同样会 flip，但 top3fold 的 image cosine、mean-patch embedding cosine、router weight JS 均显著更低。
3. 周期选择变化会显著放大变化：changed 样本的 image cosine mean 约 0.061950，unchanged 约 0.000583；changed 样本的 router JS mean 约 0.099124，unchanged 约 0.001769。
4. top3fold 的 diagnostic-balanced 优势可能来自更稳定的周期表达，但 q1、低周期 bucket 等 strata 仍存在 selected model flip 风险，需要在 65k 继续监控。
5. 不建议把 `period_soft_mixture` 作为 65k 前置硬门槛；它应作为 Round2e / 后续 soft-period 改进项。
6. 建议让 `top3fold_period_layout` 与 `spatial_panel_3view`、`current_rgb_3view` 一起进入 65k expanded validation。

## 下一步方案

1. 进入 65k expanded validation，候选 layout 保持 Round2c 建议：`spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。
2. 在 65k 中继续跟踪 `top1_period_bucket`、`error_gap_quantile`、CrossFormer/PatchTST、strong seasonality 等 strata。
3. 后续单独设计 `period_soft_mixture` smoke，不阻塞当前 65k；若 65k 中 period layout 出现明显不稳定或泛化回退，再升级为完整 head 对照。
4. panelized top3fold 和 period tokens 保留为后续表示增强候选。
