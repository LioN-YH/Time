# Stage 1 Full-Scale Streaming Dry-Run

日志日期：2026-06-14 21:58:47 CST

## 目的

在保持 Stage 1 online 主线 `x -> pseudo image -> frozen ViT -> router` 不变的前提下，补齐 `96_48_S` 面向 QuitoBench vali/test 全量规模的可恢复执行框架，并用小样本 full-scale dry-run 验证 manifest、prediction cache、merge、oracle/baseline、streaming online router 和 calibration 的闭环。

## 背景

此前 `train_visual_router_online.py` 已经证明 120 sample_key 和 1k sample_key 的 online ViT router 可行，但该入口会在运行内维护 `sample_key -> embedding` 字典，不适合真正全量窗口。专家 prediction cache 旧路径也主要使用 per-sample `.npy` 小文件，百万级窗口会造成文件数量膨胀。因此本次需要把全量执行框架升级为：

- prediction cache 允许 shard 落盘，但默认使用 packed array；
- router 训练和评估使用 streaming online ViT，不保存伪图像 tensor 或 ViT embedding；
- 长任务保留 `main.log`、`status.json` 和 metadata，便于失败后定位和重跑。

## 操作

1. 新增 `visual_router_experiments/common/prediction_array_io.py`，统一读取 `per_sample_npy` 和 `packed_npy_v1` prediction arrays。
2. 扩展 `prediction_cache_schema.py`，允许 manifest 记录 `array_storage`、`y_true_row_index`、`y_pred_row_index`，并在 packed 模式下校验共享 y_true row index。
3. 扩展 `build_prediction_cache_from_manifest.py`，新增 `--array-storage packed_npy_v1`，在 shard 内按 split/dataset/model 写 packed `.npy`。
4. 扩展 `merge_prediction_cache_shards.py`，支持合并 packed shard，并在 merged cache 中重建共享 y_true packed 文件。
5. 更新 `train_visual_router.py` 和 `evaluate_soft_fusion_calibration.py`，改用统一数组读取层，兼容 packed manifest。
6. 新增 `train_visual_router_online_streaming.py`，使用 `StandardScaler.partial_fit` 遍历 vali embedding 流，每个训练 epoch 重新在线生成 vali embedding，test split 流式 forward；输出标准 `visual_router_predictions.csv` 和 `visual_router_metadata.json`。
7. 新增 `launch_full_scale_prediction_cache.py`，生成 full-scale prediction cache launcher，默认 DLinear/PatchTST/CrossFormer 绑定 GPU，ES/NaiveForecaster 走 CPU 独立进程。
8. 新增 `run_full_scale_dry_run.py`，执行小样本闭环 dry-run。首轮开发目录为 `2026-06-14_stage1_full_scale_dry_run_dev/`，后续重复 merge 时暴露出根 `status.json` 被失败状态覆盖、根目录缺少 `main.log` 的问题。
9. 修复 full-scale dry-run / merge / launcher 的恢复与留痕问题：

   - `merge_prediction_cache_shards.py` 在目标数组已存在且内容不一致时改为覆盖半成品，支持同一批 shard 在同一输出目录下可恢复重跑；
   - `run_full_scale_dry_run.py` 新增根级 `main.log`，最终 `status.json` 以完整闭环为准，并在 `--skip-existing` 时可从已完成 `metadata.json` 恢复根状态；
   - `launch_full_scale_prediction_cache.py` 新增根级 `main.log` 和 `metadata.json`；
   - `train_visual_router_online_streaming.py` 的 ViT 加载增加有限指数退避重试，用于应对 429、503、timeout 等临时网络或远端缓存错误。

10. 使用修复后的入口重新运行有效 dry-run。实际运行命令为：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/run_full_scale_dry_run.py \
     --samples-per-split 2 \
     --sample-shard-count 2 \
     --embedding-batch-size 2 \
     --router-epochs 1 \
     --device auto \
     --local-files-only \
     --output-dir experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2
   ```

11. 同步更新 `stage1_protocol_and_plan.md`、`stage1_cache_contract.md`、`README.md`、`WORKSPACE_STRUCTURE.md` 和 `AGENTS.md` 中 full-scale online / packed cache 相关口径。

## 结果

dry-run 输出目录：

```text
experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/
```

关键验证结果：

- root `status.json` 为 `completed`；
- root `main.log` 存在，并记录了 prediction cache、merge、oracle、enrichment、baseline、streaming router 和 calibration 的逐步 start/completed 状态；
- merged prediction manifest 为 `20` 行，覆盖 `4` 个 sample_key 和五个专家；
- 每个 sample_key 的专家数为 `5`，`array_storage` 全部为 `packed_npy_v1`；
- `window_oracle_labels_with_tsf_cell.csv` 为 `8` 行，覆盖 `mae` 和 `mse` 两种 metric；
- streaming router 输出 `2` 条 test prediction，split 全部为 `test`，五专家权重行和约为 `1.0`；
- calibration summary 输出 `5` 个策略：`raw_soft`、`soft_T0p5`、`top1_hard`、`top2_fusion`、`top2_fusion_T0p5`；
- `streaming_online_router/` 下没有 `.npy`、`embeddings/` 或 embedding shard 文件，只保存 manifest、latency、router predictions、summary、metadata 和日志。
- 首轮 `2026-06-14_stage1_full_scale_dry_run_dev/` 的根 `status.json` 曾因重复 merge 到已有目录而显示 failed；该目录只作为开发调试留痕，正式 dry-run 证据以 `v2` 目录为准。

已通过的验证命令：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
  visual_router_experiments/common/prediction_array_io.py \
  visual_router_experiments/common/prediction_cache_schema.py \
  visual_router_experiments/stage1_vali_test_router/build_prediction_cache_from_manifest.py \
  visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py \
  visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
  visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py \
  visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
  visual_router_experiments/stage1_vali_test_router/launch_full_scale_prediction_cache.py \
  visual_router_experiments/stage1_vali_test_router/run_full_scale_dry_run.py
```

## 结论

本次已经完成首个 full-scale 框架 dry-run，并补齐恢复语义与根级日志留痕，证明新的 packed prediction cache、packed merge、oracle/baseline 后处理、streaming online router 和 calibration ABI 可以闭合。dry-run 只用于验证执行框架和字段契约，不作为正式性能指标引用。

当前正式路线应固定为：

- 专家 prediction cache：允许 shard 落盘，full-scale 默认 `packed_npy_v1`；
- ViT / 伪图像：只在运行时生成，不落盘、不长期缓存；
- router：full-scale 使用 `train_visual_router_online_streaming.py`，不依赖全量 in-memory embedding 字典；
- calibration：继续读取 `visual_router_predictions.csv` 和 prediction manifest，使用固定 temperature/top-k sweep。

## 下一步方案

1. 根据服务器 GPU/CPU 资源确定真实全量的 `sample_shard_count`、输出根目录和并发上限。
2. 用 `launch_full_scale_prediction_cache.py` 生成正式 full-scale prediction cache launcher，并先跑一个较大 shard smoke。
3. 在 merged cache 上运行 oracle、TSF enrichment、baseline 后，使用 `train_visual_router_online_streaming.py` 做正式 streaming router。
4. 补统一 final summarizer，将 baseline、hard top-1、raw soft、best calibrated soft 和 oracle 汇总到最终报告表。
