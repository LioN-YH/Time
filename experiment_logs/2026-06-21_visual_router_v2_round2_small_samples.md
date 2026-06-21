# Visual Router V2 Round2 view layout small screening 样本构建

日志日期：2026-06-21 21:20:46 CST

## 目的

启动 Visual Router V2 Round2 view layout / pseudo image small screening 的第一步，冻结小样本集合、layout candidate 设计和 top3fold 复用边界，为后续 small feature cache screening 做准备。

## 背景

Round1 global summary 确认 `film_mean_patch_aux` 是 frozen `pilot_test` 当前最强变体，raw-soft MAE/MSE/regret 为 `0.417824 / 183.353985 / 0.077539`。Round2 不应同时改变多个因素，因此本步只调整 pseudo image / view layout 协议资产，后端 router/head 后续固定为 `mean_patch_embedding + revin_aux FiLM modulation`。

## 操作

1. 读取用户提供的任务说明，确认本步禁止训练 router、禁止运行 ViT、禁止生成 full feature cache、禁止启动 200k P2a-style cache。
2. 复核已有 P0 样本构建脚本 `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_pilot_samples.py`，确认可复用稳定 hash、流式 oracle parquet 扫描、TSF enrichment join 和 validation 逻辑。
3. 复核已有 top3fold 实现，确认位于：
   - `visual_router_experiments/common/pseudo_imageization.py::select_fft_periods`
   - `visual_router_experiments/common/pseudo_imageization.py::imageize_top3fold`
   - `visual_router_experiments/common/vit_embedding_utils.py::make_pseudo_images`
4. 新增 `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_small_samples.py`，用于构建 Round2 small sample sets、layout candidates、top3fold reuse audit、metadata、validation、summary 和轻量 summary 副本。
5. 使用 Quito 环境运行语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_small_samples.py
   ```

6. 使用 Quito 环境运行正式样本构建：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_small_samples.py
   ```

7. 新增 `experiment_summaries/visual_router_v2_round2/round2_small_screening_protocol.md`，记录样本协议、layout candidates、top3fold 复用边界和后续耗时估计。
8. 将轻量 summary 复制到 `experiment_summaries/visual_router_v2_round2/small_samples/`。
9. 更新 `WORKSPACE_STRUCTURE.md`，登记新脚本、外部输出目录和 Round2 summary 目录。

## 结果

外部输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/
```

生成文件包括：

- `round2_train_small_sample_keys.csv`
- `round2_selection_small_sample_keys.csv`
- `round2_diagnostic_balanced_small_sample_keys.csv`
- `round2_test_small_sample_keys.csv`
- `round2_small_sample_manifest.csv`
- `round2_layout_candidates.json`
- `round2_top3fold_reuse_audit.md`
- `round2_small_sample_metadata.json`
- `round2_small_screening_summary.md`
- `round2_coverage_summary.csv`
- `round2_validation_summary.json`

样本规模验证：

| sample_set | split | count | duplicate |
| --- | --- | ---: | ---: |
| `round2_train_small` | vali | 20,000 | 0 |
| `round2_selection_small` | vali | 5,000 | 0 |
| `round2_diagnostic_balanced_small` | vali | 5,000 | 0 |
| `round2_test_small` | test | 5,000 | 0 |

`round2_small_sample_manifest.csv` 共 35,000 行，跨集合 sample_key 重复数为 0，`round2_train_small ∩ round2_selection_small = 0`。metadata 记录：

- `round2_stage=small_sample_builder`
- `trained_model=false`
- `built_feature_cache=false`
- `ran_vit=false`
- `saved_pseudo_image_tensor=false`
- `used_pilot_test_for_selection=false`
- `test_small_used_for_selection=false`
- `loaded_116m_prediction_manifest_to_memory=false`
- `top3fold_existing_implementation_found=true`

第一轮默认 layout set 已写入 `round2_layout_candidates.json`：

- `current_rgb_3view`
- `spatial_panel_3view`
- `line_only`
- `line_difference_band`
- `fft_absolute_energy`
- `top3fold_period_layout`

第一轮暂缓：

- `period_soft_mixture`
- `independent_view_encoder`

P0 overlap 已计算并写入 metadata。示例结果：`round2_train_small` 与 P0 任意集合 overlap ratio 为 `0.0223`，`round2_selection_small` 为 `0.0212`，`round2_diagnostic_balanced_small` 为 `0.0264`，`round2_test_small` 为 `0.0046`。

## 结论

Round2 small screening 的样本边界和 layout candidate 协议已冻结。本步没有生成 feature cache，没有训练 router/head，没有运行 frozen ViT，没有保存 pseudo image tensor，也没有读取 116M prediction manifest 到内存。后续可以在该 35k fixed sample set 上实现小规模 layout feature cache screening。

## 下一步方案

1. 实现 Round2 layout registry / adapter，使 `current_rgb_3view`、`spatial_panel_3view`、`line_only`、`line_difference_band`、`fft_absolute_energy` 和 `top3fold_period_layout` 可由同一 feature-cache builder 参数化调用。
2. 先对 35k small samples 做单 layout smoke，确认 sample_key/order_index、ViT feature shape、mean_patch pooling、RevIN aux 和 metadata 门禁。
3. 再按默认 layout set 生成 small feature cache；如资源允许，使用 3 张 GPU 做进程级并行，预计 wall time 约 2-3 小时。
4. 后续 router/head 固定为 `film_mean_patch_aux`，只用 `round2_train_small` 训练、`round2_selection_small` 选择，`round2_diagnostic_balanced_small` 诊断，`round2_test_small` frozen screening only。
