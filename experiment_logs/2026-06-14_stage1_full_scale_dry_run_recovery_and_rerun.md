# Stage 1 Full-Scale Dry-Run Recovery And Rerun

日志日期：2026-06-14 23:13:57 CST

## 目的

修复首个 full-scale dry-run 在重复 merge 时暴露出的恢复语义问题，补齐 dry-run / launcher 根目录的 `main.log`、`metadata.json` 与 `status.json` 留痕，然后重新验证 `96_48_S` full-scale 可恢复流水线模板是否仍能闭环。

## 背景

上一轮 `run_full_scale_dry_run.py` 已经完成了首个闭环 dry-run，但我在同一输出目录上重复执行 `merge_prediction_cache_shards.py` 时，触发了半成品目标数组冲突，导致根 `status.json` 被覆盖成失败状态。同时，dry-run 根目录和 launcher 根目录都还缺少长任务要求的 `main.log` 与 `metadata.json`，不够适合作为后续正式 full-scale 的模板。

## 操作

1. 修改 `merge_prediction_cache_shards.py` 的数组复制逻辑：当目标文件已存在且内容一致时跳过，若是同一批 shard 重跑留下的半成品则允许覆盖，保证同目录可恢复。
2. 修改 `run_full_scale_dry_run.py`：
   - 增加根级 `main.log`；
   - 在根级 `status.json` 中记录 `main_log`；
   - 在 `--skip-existing` 时可从已完成的 `metadata.json` 恢复根状态；
   - 为每个子步骤继续保留独立 `main.log`、`status.json`。
3. 修改 `launch_full_scale_prediction_cache.py`：
   - 增加根级 `main.log`；
   - 增加根级 `metadata.json`；
   - 保持 launcher 只生成不自动启动的默认行为。
4. 修改 `train_visual_router_online_streaming.py`：
   - 为 ViT 加载增加有限指数退避重试；
   - 仅把 429、503、timeout、连接抖动等临时错误视为可重试。
5. 重新运行 dry-run，最终输出目录改为：

   ```text
   experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/
   ```

6. 生成一次仅产出不启动的 launcher 模板，确认 launcher 根目录同样写入 `main.log`、`metadata.json`、`status.json`、`launcher.sh` 和 `launch_plan.md`。

## 结果

- `run_full_scale_dry_run.py --skip-existing` 可从完成的 `metadata.json` 恢复根状态，不再把已完成目录误判为失败；
- `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/` 根目录存在 `main.log`、`metadata.json`、`status.json`；
- `merged_cache/manifest.csv` 为 `20` 行，覆盖 `4` 个 sample_key，五专家完整，`array_storage=packed_npy_v1`；
- `streaming_online_router/` 只保留 manifest、latency、router predictions、summary、metadata 和日志，没有 `.npy`、`embeddings/` 或伪图像 tensor cache；
- `soft_fusion_calibration/` 完成 5 个策略的输出；
- `launch_full_scale_prediction_cache.py` 生成的 launcher 目录也写入了根级 `main.log`、`metadata.json` 与 `status.json`。

## 结论

full-scale dry-run 的执行框架已经不是一次性脚本，而是可恢复、可续跑、带根级留痕的模板。当前可以把它作为后续正式全量长跑的起点：`packed_npy_v1` prediction cache + streaming online router + calibration 的 ABI 已经稳定。

## 下一步方案

1. 基于正式资源分配决定真实 full-scale 的 `sample_shard_count` 和 launcher 输出根目录。
2. 用 `launch_full_scale_prediction_cache.py` 生成正式 launcher 并启动五专家 shard。
3. 在 merged cache 上继续跑 oracle、TSF enrichment、baseline、streaming router 和 calibration。
4. 如果正式长跑也出现半成品目录，再按这次的恢复语义继续精确重跑对应 shard 或子步骤。
