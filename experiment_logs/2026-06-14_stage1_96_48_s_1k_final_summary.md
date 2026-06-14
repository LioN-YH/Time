# Stage 1 `96_48_S` 1k 中等规模实验最终汇总

日志日期：2026-06-14 17:59:35 CST

## 目的

对 `96_48_S` 1k 中等规模实验做最终收口，串联 preflight、五专家 prediction cache、merge、oracle/TSF/baseline、online visual router、soft fusion calibration 的真实结果，给出可复核的最终产物索引。

## 背景

本轮目标要求实验可恢复、分阶段推进，不主动联网下载模型、依赖或数据；ViT/HF 路径必须使用本地 cache 和 `--local-files-only`；不长期保存伪图像 tensor 或 ViT embedding `.npy`。最终应得到统一的中文实验日志、README 总览和结果汇总表。

## 执行时间线

1. `17:27:04 CST` 完成 preflight 检查，确认 sample manifest 完整、GPU 空闲、`quito` 环境可用、五专家 shard 尚不存在。
2. `17:43:17 CST` 完成五专家 prediction cache；原 launcher 后台方式未保住子进程，随后用同命令直接会话补跑并获得 5 个完成 shard。
3. `17:45:12 CST` 完成 merge，得到 5000 行五专家 manifest 和共享 `y_true` 合并 cache。
4. `17:46:53 CST` 完成 oracle labels、TSF cell enrichment 和非视觉 baseline。
5. `17:52:36 CST` 完成 online visual router 首次运行，并补充 `local_files_only` 产物字段后重跑得到最终自证版本。
6. `17:54:38 CST` 完成 soft fusion calibration，得到最佳 calibrated soft 策略。
7. `17:56:00 CST` 更新 `WORKSPACE_STRUCTURE.md`，补充本轮正式 1k run 目录口径。

## 关键结果

### 五专家 prediction cache

- 五个 shard 均为 `completed`。
- 每个 shard 1000 行、1000 个 sample_key、仅含单专家。
- `sample_key + model_name` 无重复。
- 前 25 条数组重算 MAE/MSE 与 manifest 一致。

### Merge 与 cache contract

- 合并后 manifest 行数：5000。
- sample_key 唯一数：1000。
- 每个 sample_key 覆盖五专家。
- 共享 `y_true_path` 唯一且数组内容一致。
- 所有 `y_pred_path` 文件存在。

### Oracle / TSF / baseline

- `window_oracle_labels.csv` 行数：2000。
- `window_oracle_labels_with_tsf_cell.csv` 行数：2000，TSF 字段无缺失。
- `dataset_tsf_cell` baseline test MAE：0.439672。
- `global_best_single` baseline test MAE：0.467657。
- `oracle_top1` MAE：0.356273。

### Online visual router

- 最终 online run 目录：

```text
experiment_logs/run_outputs/2026-06-14_175036_visual_router_stage1_online_visual_router_96_48_s_1k_local_only/
```

- `local_files_only = True`。
- `embedding_storage = in_memory_only`。
- `persistent_embedding_npy_written = False`。
- `persistent_pseudo_image_tensor_written = False`。
- `online_embedding_manifest.csv` 行数：1000。
- hard top-1 MAE：0.459729。
- raw soft fusion MAE：0.437221。

### Soft fusion calibration

- calibration 输出 19 个策略、9500 条预测。
- 每个策略覆盖 500 个 test sample。
- 最佳 non-oracle calibration：`calibration_top3_fusion`，MAE=0.436033。
- 它优于 raw soft fusion 0.437221、online hard top-1 0.459729、`dataset_tsf_cell` 0.439672。

## 结果索引

最终汇总表：

```text
experiment_logs/run_outputs/2026-06-14_175338_visual_router_stage1_soft_fusion_calibration_96_48_s_1k/stage1_96_48_s_1k_final_summary.csv
```

产物索引：

```text
experiment_logs/run_outputs/2026-06-14_175338_visual_router_stage1_soft_fusion_calibration_96_48_s_1k/stage1_96_48_s_1k_artifact_index.csv
```

## 结论

`96_48_S` 1k 中等规模实验已按阶段完成并验证闭环。最终可部署结果中，`calibration_top3_fusion` 是当前最优的融合策略，MAE=0.436033；online raw soft fusion 次之，MAE=0.437221。整个流程满足本地 cache、自恢复、逐步留痕和不长期保存 embedding/tensor 的要求。

## 下一步方案

1. 若要进一步推进，可基于该 1k 结果继续扩展到更大样本或其它 config。
2. 若要做结果复盘，可直接引用 `stage1_96_48_s_1k_final_summary.csv` 和本次各阶段日志。
