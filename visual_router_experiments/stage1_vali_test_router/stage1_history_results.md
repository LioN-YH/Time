# Stage 1 History Results

本文档保存 `stage1_vali_test_router/README.md` 中拆出的历史实验结果索引。README 只保留当前主线入口和文件职责；历史 smoke、1k、dry-run 与 full-scale 状态在这里留痕。

更新日期：2026-06-15 23:07:21 CST

## TimeFuse-Style Fusor Baseline

`evaluate_router_baselines.py` 是当前统一 baseline 入口：保留 global/dataset/TSF-cell/dataset+TSF-cell 等统计规则 baseline，并在 feature cache 与 prediction manifest 可对齐时训练 TimeFuse-style 单层 fusor。

TimeFuse-style fusor 口径：

- `nn.Linear(input_dim, output_dim)` 输出 logits；
- softmax 得到五专家权重；
- 用权重融合五专家 `y_pred`；
- 对 fused prediction 和 `y_true` 使用 `SmoothL1Loss(beta=0.01)` 训练；
- 旧 `timefuse_single_variable_logistic_regression` 只保留为 `legacy/deprecated` hard-label 分类历史口径。

120 sample_key pilot 代表输出目录：

```text
experiment_logs/run_outputs/2026-06-15_014918_visual_router_stage1_timefuse_fusor_baseline_pilot/
```

代表结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `timefuse_style_fusor` hard top-1 | 1.490870 | 0.805392 | 0.685478 | 0.216667 | 0.576704 | 0.619892 |
| `timefuse_style_fusor_raw_soft_fusion` | 1.509144 | 0.805392 | 0.703752 | NA | 0.576704 | 0.619892 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |

## 120 Sample 离线 Embedding Smoke

早期离线 smoke 使用 `96_48_S` 五专家 pilot 的 120 个 `metric=mae` sample_key。该路径会生成 embedding `.npy`，目前只作为历史对照和小规模调试保留。

代表输出目录：

```text
experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/
experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/
```

旧版分类 router 结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy |
| --- | ---: | ---: | ---: | ---: |
| `visual_router_mlp_v1_classification` hard top-1 | 1.013099 | 0.805392 | 0.207707 | 0.350000 |
| `visual_router_mlp_v1_classification_soft_fusion` | 1.022590 | 0.805392 | 0.217198 | NA |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 |

## Fusion Router Smoke

`train_visual_router.py` 之后默认使用 `fusion_huber_kl`：用五专家预测加权融合的 SmoothL1 主损失训练权重，同时用 soft oracle KL 作为辅助监督。

代表输出目录：

```text
experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/
```

代表结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `visual_router_mlp_v2_fusion_huber_kl` hard top-1 | 0.982425 | 0.805392 | 0.177033 | 0.466667 | 0.757180 | 0.483784 |
| `visual_router_mlp_v2_fusion_huber_kl_soft_fusion` | 1.085451 | 0.805392 | 0.280059 | NA | 0.757180 | 0.483784 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `timefuse_single_variable_logistic_regression` legacy/deprecated | 1.079743 | 0.805392 | 0.274351 | 0.466667 | NA | NA |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |

结论：hard top-1 在 60 个 test window 上超过旧分类 router、`global_best_single` 和结构特征 LogisticRegression；raw soft fusion 仍弱，说明权重校准还不稳定。

## Online Visual Router Smoke

`train_visual_router_online.py` 证明了 120 sample_key 下 online embedding 路线可复现离线 embedding 代表结果，并且不生成 `.npy`、`embeddings/` 或伪图像 tensor cache。

代表输出目录：

```text
experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/
```

关键验证：

| 检查项 | 结果 |
| --- | --- |
| `online_embedding_manifest.csv` | `120 x 19`，覆盖 labels 的 120 个 `metric=mae` sample_key |
| `visual_router_predictions.csv` | `60 x 22`，覆盖 test split 的 60 个 sample_key |
| `visual_router_soft_fusion_predictions.csv` | `60 x 36` |
| 设备与 dtype | GPU 3，metadata 中 device=`cuda`，forward_dtype=`float16` |
| 落盘缓存 | 未生成 `.npy`、未生成 `embeddings/` 目录、未生成伪图像 tensor cache |

指标对齐：

| 方法 | hard top-1 MAE | raw soft fusion MAE | oracle MAE | global_best_single |
| --- | ---: | ---: | ---: | ---: |
| online in-memory ViT | 0.982425 | 1.085451 | 0.805392 | 1.055190 |
| offline embedding reference | 0.982425 | 1.085451 | 0.805392 | 1.055190 |

## Soft Fusion Calibration Smoke

`evaluate_soft_fusion_calibration.py` 用已有 router test 权重和 prediction cache，评估 raw soft、temperature scaling 和 top-k 截断。

代表输出目录：

```text
experiment_logs/run_outputs/2026-06-14_032303_451482_visual_router_stage1_soft_fusion_calibration_smoke/
```

旧代表 router 的校准结果：

| 策略 | test MAE | oracle MAE | normalized weight entropy | mean max weight | 相对 global_best_single |
| --- | ---: | ---: | ---: | ---: | ---: |
| `top1_hard` | 0.982425 | 0.805392 | 0.000000 | 1.000000 | +6.895970% |
| `top2_fusion_T0p25` | 0.999014 | 0.805392 | 0.232572 | 0.822011 | +5.323768% |
| `soft_T0p25` | 1.000585 | 0.805392 | 0.295159 | 0.799357 | +5.174969% |
| `raw_soft` | 1.085451 | 0.805392 | 0.757180 | 0.483784 | -2.867833% |
| `global_best_single` | 1.055190 | 0.805392 | NA | NA | 0.000000% |

结论：温度 sharpen 和 top-k 截断可以把 soft fusion 拉回到超过 `global_best_single`，但仍弱于 hard top-1；当前 router 排序信号比概率幅度更可靠。

## Fixed Candidates Embedding 对照

使用 `fixed_candidates` 周期桶重新生成同一批 120 个 sample_key 后，图像化 latency 改善，但小样本指标弱于旧代表 embedding。

代表输出目录：

```text
experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/
experiment_logs/run_outputs/2026-06-14_032518_167365_visual_router_stage1_visual_router_smoke/
experiment_logs/run_outputs/2026-06-14_032647_499280_visual_router_stage1_soft_fusion_calibration_smoke/
```

关键结论：

- 去掉首批 warm-up 后，fixed_candidates 图像化均值为 `0.222156 ms/window`，旧 run 为 `0.469106 ms/window`；
- 端到端每窗口均值从 `1.692935 ms` 降到 `1.405437 ms`；
- 新 embedding hard top-1 MAE=`1.011773`，弱于旧代表 hard top-1 MAE=`0.982425`；
- 120 个样本中 22 个 embedding 的最大绝对差异超过 `1e-6`，test hard top-1 有 13 个 sample_key 改变专家选择。

## `96_48_S` 1k 中等规模链路

1k manifest-only 样本清单：

```text
experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/
```

抽样口径：

- `vali=500`、`test=500`；
- 每个 split 下 `TEST_DATA_MIN=250`、`TEST_DATA_HOUR=250`；
- 每个 dataset 选 50 个 item，每个 item 选 ch0 的 5 个中心等距 window；
- item 使用 TSF cell 均衡轮转后在 cell 内等距抽样。

链路状态：

- 1k 五专家 prediction cache、merge、oracle/TSF/baseline、online router 和 calibration 均已完成；
- 1k online router 保留为中等规模实证结果；
- 历史 1k ViT embedding launcher 不启动，仅作为已生成但废弃的离线 cache 对照入口留痕。

1k 关键结果：

- oracle top-1 MAE=`0.356273`；
- `global_best_single` MAE=`0.467657`；
- `dataset_tsf_cell` MAE=`0.439672`；
- online visual hard top-1 MAE=`0.459729`；
- raw soft fusion MAE=`0.437221`；
- best deployable calibration 为 `calibration_top3_fusion`，MAE=`0.436033`。

最终汇总日志：

```text
experiment_logs/2026-06-14_stage1_96_48_s_1k_final_summary.md
```

## Full-Scale Dry-Run

首个 full-scale 框架 dry-run 已完成：

```text
experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/
```

验证结果：

- merged manifest 为 `20` 行，覆盖 `4` 个 sample_key，每个 sample_key 五专家完整；
- prediction cache 全部使用 `array_storage=packed_npy_v1`；
- streaming router 输出 `2` 条 test prediction，权重行和约为 `1.0`；
- calibration 输出 `raw_soft`、`soft_T0p5`、`top1_hard`、`top2_fusion`、`top2_fusion_T0p5` 共 `5` 个策略；
- streaming router 目录未生成 `.npy`、`embeddings/` 或 embedding shard 文件；
- dry-run 根目录写入 `main.log`、`status.json` 和 `metadata.json`，可作为 full-scale 长任务模板。

## Full-Scale 正式长跑

正式输出根目录：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/
```

已完成 sample manifest：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/
```

关键信息：

- sample_count：`23,275,170`；
- sample_shard_count：`64`；
- selection_strategy：`all_candidate_windows`；
- shard index：`sample_manifest_full_scale/sample_manifest_shard_index.csv`。

prediction cache launcher：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/
```

截至 `2026-06-15 22:25:26 CST`，只读检查确认：

- completed=`320`；
- running=`0`；
- failed=`0`；
- ES 已完成 64/64；
- 五专家全部完成；
- 原 worker、backfill 和 accelerator 均已结束。

后续应进入：

```text
merge prediction cache
-> 完整性校验
-> oracle labels
-> TSF cell enrichment
-> evaluate_router_baselines.py
-> train_visual_router_online_streaming.py
-> evaluate_soft_fusion_calibration.py
```

注意：`launcher_compat_check/` 只是 dry-run manifest 的兼容性检查目录，不是正式 prediction cache 结果。
