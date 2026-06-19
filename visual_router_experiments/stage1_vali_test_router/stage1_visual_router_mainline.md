# Stage 1 视觉路由主线梳理

记录日期：2026-06-19 11:42:04 CST

## 目的

本文档只梳理 Stage 1 visual router 主线，用于后续把 `96_48_S` 的经验扩展到 `576_288_S`、`1024_512_S` 或其它 config。TimeFuse-style fusor 是 baseline 支线，不再混入视觉路由的执行顺序。

## 主线定义

Stage 1 的正式视觉路线固定为：

```text
历史窗口 x
-> 在线伪图像 tensor
-> frozen ViT
-> router 权重
-> 五专家 hard top-1 / soft fusion / calibrated fusion
```

关键约束：

- 每个 router 只服务一个 `config_name`，动作空间是同 config 下的五个专家；
- 训练 split 固定为 `vali`，评估 split 固定为 `test`；
- 路由样本粒度为 `config_name + split + dataset_name + item_id + channel_id + window_index`；
- full-scale 不长期保存伪图像 tensor 或 ViT embedding `.npy`；
- full-scale prediction cache 使用 `packed_npy_v1` 或等价少文件 shard 格式；
- 所有可部署指标必须和 `oracle_top1`、`global_best_single`、统计 baseline 同 config 比较。

## `96_48_S` 已沉淀出的正确路线

### 1. 生成全候选 sample manifest

入口：

```text
build_full_scale_sample_manifest.py
```

当前 `96_48_S` 正式产物：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/
```

已确认：

- `sample_count=23,275,170`；
- `sample_shard_count=64`；
- `selection_strategy=all_candidate_windows`；
- manifest 是后续 prediction cache、oracle labels、router 和 calibration 的共同样本来源。

### 2. 生成五专家 prediction cache

入口：

```text
launch_full_scale_prediction_cache.py
build_prediction_cache_from_manifest.py
merge_prediction_cache_shards.py
```

当前 `96_48_S` 正式产物：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/
```

已确认：

- 五专家 320 个 shard 全部完成；
- merged cache `record_count=116,375,850`；
- `sample_count=23,275,170`；
- 五专家各覆盖 `23,275,170` 个 sample；
- 完整性校验 `passed=true`；
- `array_storage=packed_npy_v1`；
- `sample_key + model_name` 唯一、五专家完整、共享 `y_true` 一致。

### 3. 生成 oracle labels 和 TSF enrichment

入口：

```text
build_full_scale_window_oracle_labels.py
build_full_scale_tsf_enrichment.py
validate_full_scale_oracle_tsf_outputs.py
```

当前 `96_48_S` 正式产物：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/tsf_enrichment_full_scale_2026-06-16/
```

已确认：

- oracle labels 覆盖 `23,275,170` 个唯一 sample_key；
- MAE/MSE 两个 metric 共 `46,550,340` 行；
- TSF enrichment 覆盖同一批 sample_key；
- oracle 与 TSF sample_key 集合差异为 0。

### 4. 训练 full-scale streaming visual router

入口：

```text
train_visual_router_online_streaming.py
```

当前 `96_48_S` 已完成训练产物：

```text
/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/
```

已确认：

- `status=completed`；
- `phase=train_only_done`；
- `completed_epochs=1`；
- latest checkpoint 为 `checkpoints/latest_96_48_S.pt`；
- 训练使用 streaming online 路线，不保存长期 ViT embedding 或伪图像 tensor。

重要经验：

- 旧 `_1epoch/` 目录因 full-scale manifest lookup OOM 失败，不应引用；
- 正确实现是 SQLite 磁盘索引 + batch 查询，并保留 `packed_npy_v1` 的 row index；
- full-scale 训练建议保留 `--train-only` 和独立 eval-only 两段式，便于 checkpoint 复用和追加 epoch。

### 5. 用 checkpoint 执行 eval-only

入口：

```text
train_visual_router_online_streaming.py --resume-checkpoint <latest_ckpt> --epochs 0
```

当前 `96_48_S` 已完成评估产物：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/
```

已确认 full-scale test 指标：

| 方法 | MAE |
| --- | ---: |
| visual hard top-1 | 0.5615367653135453 |
| raw soft fusion | 0.5174675759559787 |
| oracle top-1 | 0.33862214116809347 |

其它关键结果：

- `router_predictions=13,924,650`；
- `oracle_label_accuracy=0.4621166779775434`；
- 输出包含 `visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_soft_fusion_predictions.csv` 和相关 metadata。

### 6. 执行 soft fusion calibration

入口：

```text
evaluate_soft_fusion_calibration.py
```

当前状态：

- 1k calibration 已跑通；
- full-scale calibration 还需要确认脚本采用 streaming/SQLite 读取，不能全量加载 116M 行 manifest；
- calibration 只能使用 router test 权重做固定 temperature/top-k 策略，不允许读取 test oracle error 做逐样本动态调权。

建议 full-scale calibration 报告至少包含：

- `top1_hard`；
- `raw_soft`；
- `soft_T0p25` / `soft_T0p5` 等固定温度；
- `top2_fusion` / `top3_fusion`；
- `top2_fusion_T*` / `top3_fusion_T*`；
- 与 `global_best_single` 和 `oracle_top1` 的同表比较。

## 不再作为视觉主线的路线

以下路线只保留为历史、调试或对照，不作为 full-scale visual router 主线：

- 离线 ViT embedding cache：`pilot/build_vit_embeddings_pilot.py`；
- 1k 离线 embedding launcher：`pilot/launch_96_48_s_1k_vit_embedding_pilot.py`；
- full-scale 使用 `train_visual_router_online.py`：该入口会暂存全量 embedding，只适合 120/1k；
- 旧 OOM 目录：`/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch/`；
- TimeFuse-style fusor：作为 baseline 支线单独追踪，不影响 visual router 扩配置执行顺序。

## 扩展到其它 Config 的标准步骤

对每个新 config，例如 `576_288_S` 或 `1024_512_S`，按以下顺序复制主线：

1. 生成该 config 的 full-scale sample manifest，并确认 `sample_key` 中的 `config_name` 正确；
2. 用同 config 的五专家 checkpoint / 统计模型生成 prediction cache shard；
3. merge 前校验 `sample_key + model_name` 唯一、五专家完整、共享 `y_true` 一致；
4. 在 merged cache 上生成该 config 的 oracle labels 和 TSF enrichment；
5. 用 `train_visual_router_online_streaming.py` 训练该 config 的独立 router；
6. 使用 `--resume-checkpoint ... --epochs 0` 在独立目录执行 eval-only；
7. 对 eval-only 输出运行 calibration；
8. 最终报告按 config 分表，再给 macro average 总览。

扩配置时不要复用另一个 config 的 router head 作为正式可部署结论。跨 config encoder 共享、leave-one-config-out 或 zero-shot head 迁移应放入 Stage 1B，而不是 Stage 1 主实验。

## 当前下一步

只关注视觉路由时，当前最自然的下一步是：

1. 检查 full-scale calibration 脚本是否已经避免全量加载 merged manifest；
2. 若需要，给 `evaluate_soft_fusion_calibration.py` 增加 SQLite/streaming 读取路径；
3. 对已完成的 `96_48_S` eval-only 输出运行 full-scale calibration；
4. 生成视觉主线报告，先汇总 `oracle_top1`、`global_best_single`、visual hard top-1、raw soft、best calibrated soft；
5. 再决定是追加 `96_48_S` epoch，还是复制主线到下一个 config。
