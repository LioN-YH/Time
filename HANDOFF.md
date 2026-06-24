# Handoff: Visual Router V2 Round2 1M gate passed, fullscale shard00 completed

日志日期：2026-06-24 07:01:24 CST

## 当前目标

Visual Router V2 Round2 fullscale 主线：先完成 `spatial_panel_3view,current_rgb_3view + film_mean_patch_aux` 的 1M staged seed16 gate；若 gate 无明显失败，启动 `spatial_panel_3view + film_mean_patch_aux` fullscale seed16，用于后续和 TimeFuse fullscale 做 single-seed first pass 对比。

当前上下文已经超过项目规范的 handoff 阈值。本文件是强制交接点，后续建议在新窗口继续，不要依赖当前对话上下文。

## 已完成

- 已读取用户任务文件：
  `/home/shiyuhong/.codex-tianyu/attachments/01c25011-06bf-4489-812d-d1b1633bce95/pasted-text-1.txt`
- 1M staged seed16 gate 已完成并通过：
  - summary: `experiment_summaries/visual_router_v2_round2/1m_staged_seed16_gate/round2_staged_fullscale_validation_summary.md`
  - best layout: `spatial_panel_3view`
  - feature manifest: passed
  - prediction lookup: `5242880/5242880`
  - `staged_test` 未用于选择。
- 已新增/修改 fullscale streaming FiLM 入口：
  `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_fullscale_streaming_film.py`
- fullscale seed16 `spatial_panel_3view + film_mean_patch_aux` shard00/64 已完成：
  - run dir: `/data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64`
  - status: `completed/done`
  - checkpoint: `checkpoints/round2_film_seed16_epoch0001.pt`
  - prediction index: `prediction_manifest_index.sqlite`
  - predictions: `visual_router_predictions.csv`
  - soft fusion predictions: `visual_router_soft_fusion_predictions.csv`
  - summary: `visual_router_round2_fullscale_summary.csv`
  - metadata: `visual_router_metadata.json`

## 关键结果快照

1M staged gate 的 selection 基准：

```text
best_layout=spatial_panel_3view
backend_style=film_mean_patch_aux
selected_from_sample_set=staged_selection
staged_selection raw_soft_MAE=0.2998578451288766
staged_selection raw_soft_MSE=1.186940034708284
staged_selection raw_soft_regret=0.0338153726784321
```

1M staged gate 的 staged_test 对比：

```text
spatial_panel_3view raw_soft_MAE=0.4128115751322085
current_rgb_3view   raw_soft_MAE=0.4208501357718179
```

fullscale seed16 shard00/64 test 结果：

```text
sample_count=217573
raw_soft_MAE=0.4651163217929314
raw_soft_MSE=191.82048624897772
raw_soft_regret=0.1186626195232913
hard_top1_MAE=0.4882601586607196
hard_top1_MSE=191.94987953017548
hard_top1_regret=0.1418064563910796
oracle_label_accuracy=0.5146686399507292
weight_entropy=1.295631112864455
mean_max_weight=0.42796135904762694
```

注意：shard00/64 不是完整 fullscale 汇总，不能当作最终 fullscale 结论；它只说明 fullscale shard 链路和一片数据已跑通。

## 正在运行

有一个局部 TimeFuse/统计对比任务仍在运行：

```text
pid=509709
cmd=/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/compare_p0_275k_spatial_timefuse_statistical.py --overwrite --device auto --batch-size 2048 --output-dir /data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison
output_dir=/data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison
status_json=/data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison/status.json
latest_status=running, stage=load_timefuse_features, updated_at=2026-06-24 07:00:42 CST
partial_output=p0_275k_statistical_policy_mapping.partial.csv
```

监控命令：

```text
ps -p 509709 -o pid,ppid,pgid,stat,etime,pcpu,pmem,cmd
cat /data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison/status.json
find /data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison -maxdepth 2 -type f | sort
```

停止命令（仅在用户要求或确认需要停止时使用）：

```text
kill -TERM 509709
```

## fullscale shard00 复核命令

```text
cat /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/status.json
cat /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/visual_router_round2_fullscale_summary.csv
cat /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/visual_router_metadata.json
tail -120 /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_seed16_shard00_of64/main.log
```

## 当前工作树

当前有未提交改动，包含实验脚本、结构文档和中文实验日志。不要还原这些改动，除非用户明确要求。

```text
M HANDOFF.md
M WORKSPACE_STRUCTURE.md
M experiment_logs/README.md
M visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_staged_samples.py
M visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_staged_validation_parallel.py
M visual_router_experiments/stage1_vali_test_router/summarize_visual_router_v2_round2_staged_validation.py
?? experiment_logs/2026-06-24_one_shard_staged_validation_local_history_check.md
?? experiment_logs/2026-06-24_spatial_panel_timefuse_subset_comparison_check.md
?? experiment_logs/2026-06-24_visual_router_v2_round2_1m_staged_seed16_gate.md
?? experiment_logs/2026-06-24_visual_router_v2_round2_fullscale_streaming_film_launch.md
?? experiment_summaries/visual_router_v2_round2/1m_staged_seed16_gate/
?? visual_router_experiments/stage1_vali_test_router/compare_p0_275k_spatial_timefuse_statistical.py
?? visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_fullscale_streaming_film.py
```

## 下一步建议

1. 在新窗口先检查 `p0_275k_spatial_timefuse_statistical_comparison` 是否完成；若完成，记录其输出指标并补中文实验日志/README。
2. 复核 fullscale shard00 输出，确认 `visual_router_metadata.json` 中：
   - `stream_shard_index=0`
   - `stream_shard_count=64`
   - `embedding_storage=batch_runtime_only_not_saved`
   - `pseudo_image_tensor_storage=not_saved`
   - `train_sample_count=146102`
   - `test_sample_count=217573`
3. 若 shard00 结果和资源状态可接受，再写或启动多 shard launcher，扩展到 64 shards；每个 shard 应避免重复扫描 116M manifest，优先复用/优化 subset index 或预分 shard-specific keys。
4. 完整 fullscale 完成后再聚合 all-shard summary，并输出 overall / strata / tail / router behavior / metadata / 中文 summary。
5. fullscale 完成后再与 TimeFuse fullscale 做第一版 MAE/MSE 对比；目前 shard00 指标不能作为 fullscale 对比结论。

## 边界条件

- 不新建分支、不切分支、不合并其他分支。
- 不做新的 layout search。
- 不加入 period branch、panel-wise pooling、calibration，也不改 router head 或 imageization 语义。
- 不把 1M staged gate 写成最终 fullscale 结论。
- 不把 shard00/64 写成完整 fullscale 结论。
- `test` 只能做 frozen eval，不用于选择 layout、seed、epoch 或超参。
