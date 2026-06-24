# Visual Router V2 Round2 fullscale streaming FiLM smoke 与 shard00 启动

日志日期：2026-06-24 06:04:30 CST

## 目的

在 1M staged seed16 gate 通过后，补齐 `spatial_panel_3view + film_mean_patch_aux` 的 full-scale online streaming 入口，并先用极小 smoke 验证 `x -> pseudo image -> frozen ViT -> mean_patch embedding -> FiLMRouter` 链路，再启动 seed16 shard-aware fullscale 后台任务。

## 背景

已有 `train_visual_router_online_streaming.py` 属于旧 Stage 1 pure visual MLP 入口，只支持 `variant_a_3view/variant_b_top3fold`，不支持 Round2 layout registry 和 RevIN aux FiLM 后端，不能冒充 Round2 主线。Round2 staged 固定 FiLM 脚本依赖离线 feature shard，也不满足 full-scale online 主线“不落盘伪图像 tensor / embedding cache”的约束。

## 操作

1. 新增脚本 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_fullscale_streaming_film.py`。
2. 脚本固定使用 `spatial_panel_3view` layout registry，ViT pooling 固定为 `mean_patch`，RevIN aux 通过 FiLM gamma/beta 调制 visual hidden representation。
3. prediction 读取使用 SQLite subset index 和 batch 查询，不加载 116M 行 manifest 到 Python dict。
4. 使用 `quito` 环境完成语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_fullscale_streaming_film.py
   ```

5. 运行极小 smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_fullscale_streaming_film.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_smoke_seed16 \
     --max-samples-per-split 2 --epochs 1 --embedding-batch-size 2 --batch-size 2 --eval-batch-size 2 \
     --device auto --local-files-only --status-update-interval 1 --print-rows 2
   ```

6. 启动 seed16 shard00/64 后台任务，限制到物理 GPU3：

   ```text
   /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/launch.sh
   ```

## 结果

- smoke 已完成，输出目录为：
  `/data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_smoke_seed16/`
- smoke 生成了 `checkpoint`、`online_embedding_manifest.csv`、`online_embedding_latency_summary.csv`、`visual_router_predictions.csv`、`visual_router_soft_fusion_predictions.csv`、`visual_router_round2_fullscale_summary.csv` 和 `visual_router_metadata.json`。
- smoke 的 test 样本数为 2，raw-soft MAE 为 `0.082888`，hard top1 MAE 为 `0.082888`。该结果只用于链路验证，不作为正式指标。
- smoke 的 subset prediction index 在扫描到 `115,800,000` 行时收齐 `20/20` 条目标记录，说明 subset builder 可提前停止，但随手选取的 sample_key 仍可能位于 manifest 靠后位置。
- 正式 shard00/64 后台任务已启动：
  - run dir: `/data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/`
  - launcher PID: `352708`
  - Python child PID: `352712`
  - PGID: `352708`
  - main log: `/data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/main.log`
  - status: `/data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/status.json`
  - stop command: `kill -TERM -352708`
- 启动后短间隔健康检查显示 Python 子进程仍在运行，CPU 和内存有活动，`status.json` 当前为 `running/init`。

## 结论

Round2 fullscale streaming FiLM 的最小端到端链路已经跑通；新入口满足 Round2 主线的 layout、mean_patch pooling、RevIN aux FiLM 和 runtime-only pseudo image / embedding 约束。当前正式任务只是 seed16 的 shard00/64 后台启动状态，尚未完成，不能作为 fullscale 结果引用。

## 下一步方案

1. 继续监控 shard00/64 的 `main.log`、`status.json`、GPU 占用和 checkpoint 写出情况。
2. 如果 shard00 能完成，再按同一入口扩展其它 shard，或设计多 shard launcher 聚合完整 fullscale。
3. 若 init 阶段读取全量 labels parquet 成为明显瓶颈，应改为从 sample shard 或 parquet batch 中先构建 shard-specific labels，避免每个 shard 重复 materialize 全量 metric 行。
