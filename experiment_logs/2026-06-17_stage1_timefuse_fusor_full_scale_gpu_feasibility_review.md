# Stage 1 TimeFuse-style Fusor Full-Scale GPU 可行性复核

日志日期：2026-06-17 19:27:23 CST

## 目的

复核当前 GPU2/GPU3 空闲情况下，是否适合启动 `96_48_S` full-scale TimeFuse-style fusor baseline 实验。

## 背景

用户观察到 GPU2 和 GPU3 仍有较大空余，希望判断能否并行运行 TimeFuse-style Fusor Baseline。该 baseline 的正式 full-scale 前置包括五专家 merged prediction cache、oracle labels、TSF enrichment 和 TimeFuse-derived feature cache。

## 操作

1. 使用 `nvidia-smi` 检查四张 RTX 3090 的显存和进程占用。
2. 检查 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/` 下 full-scale 产物状态。
3. 读取 TimeFuse feature cache launcher 的 shard 状态，确认 64 个 feature shard 是否完成。
4. 检查当前占用 GPU2/GPU3 的 PID `919803`，确认它属于 `train_visual_router_online_streaming.py` 的 full-scale visual router 1 epoch v2 训练。
5. 阅读 `evaluate_router_baselines.py` 与 `fusion_utils.py` 的输入读取方式，判断现有 TimeFuse-style fusor 入口是否可直接承载 full-scale 数据规模。

## 结果

1. GPU2/GPU3 当前显存占用较低，分别约 1303MiB 和 981MiB，但 PID `919803` 正在使用 GPU2/GPU3 运行 full-scale streaming visual router 训练。
2. full-scale prediction cache 已完成并校验通过：`record_count=116,375,850`、`sample_count=23,275,170`。
3. full-scale oracle labels 与 TSF enrichment 已完成并通过 validation。
4. TimeFuse feature cache 根级 `status.json` 仍是早期 `running` 快照，但实际 64/64 个 shard 的 `status.json` 均为 `completed`，历史日志也已记录总行数为 `23,275,170`。
5. 当前 `evaluate_router_baselines.py` 仍主要按单文件 CSV 读取：
   - `load_labels()` 使用 `pd.read_csv()`，而 full-scale oracle labels 当前是 Parquet；
   - `load_feature_cache()` 使用单个 `feature_cache.csv`，而 full-scale feature cache 是 64 个 shard；
   - `load_prediction_lookup()` 会一次性读取五专家 manifest，并建立 lookup；full-scale manifest 为 `116,375,850` 行，不适合原样一次性载入内存。

## 结论

可以做 `96_48_S` full-scale TimeFuse-style fusor baseline，但不建议直接用现有 `evaluate_router_baselines.py` 原样启动。当前更稳妥的路径是新增或改造一个 streaming/shard-aware full-scale fusor 入口：按 sample shard 读取 feature cache、MAE oracle labels 和 packed prediction arrays，训练阶段只聚合/流式读取 vali，评估阶段按 test shard 流式输出 summary。

GPU 资源层面，GPU2/GPU3 显存足够，但它们正在跑 visual router；若不希望干扰现有长跑，应优先考虑 GPU1 或等待 visual router 完成。TimeFuse-style fusor 的计算量主要在特征/预测数组 I/O 和 streaming 对齐，GPU 不是唯一瓶颈。

## 下一步方案

1. 不要在当前入口上直接启动 full-scale fusor。
2. 先实现 full-scale streaming fusor 脚本，输入固定为：
   - oracle labels Parquet；
   - `timefuse_feature_cache_full_scale_launcher/shards/sample_shard_*/feature_cache.csv`；
   - `prediction_cache_full_scale_launcher/merged_cache/manifest.csv` 与 packed arrays。
3. 做一个 1 到 2 个 shard 的压力测试，记录显存、内存、I/O 吞吐和输出字段。
4. 压力测试通过后，用后台 launcher 启动正式 full-scale fusor，并写入独立 `status.json`、PID、主日志、停止命令和恢复命令。
