# Stage 1 `96_48_S` Full-Scale 状态复盘与任务拆分

日志日期：2026-06-15 12:56:57 CST

## 目的

在继续推进 Stage 1 `96_48_S` 正式 full-scale 全候选窗口实验前，先复盘当前任务真实完成位置，避免把 prediction cache 监控、merge、oracle、streaming router 和 calibration 混在同一个过大的 goal 中继续推进，降低上下文膨胀和接手风险。

## 背景

用户指出当前 goal 范围过大，容易导致执行过程中上下文爆炸，希望先拆解当前完成情况和剩余任务。此前最新 handoff 固定快照为 `2026-06-15 04:02:18 CST`，当时状态为 `completed=44`、`running=9`、`failed=0`，四个非 ES 专家仍在低编号 shard，`ES` 原 worker 和 backfill 刚开始并行推进。

正式输出根目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/
```

## 操作

1. 只读检查当前工作区和运行状态，未启动新的 shard、未执行 merge、未生成 oracle labels、未启动 streaming router，也未运行 calibration。
2. 检查 Git 状态和当前提交，确认 `HEAD` 仍为 `1829af6`，但工作区存在多项既有未提交改动和新增日志/脚本文件。
3. 检查 tmux 会话，确认仍存在：
   - `stage1_fullscale_launcher_shell`
   - `stage1_es_backfill_0016_0063`
   - `stage1_mid`
4. 检查存储和 GPU 状态：
   - `/data2` 可用约 `2.6T`；
   - `/home` 仅剩约 `18G`，仍接近满盘；
   - 四张 RTX 3090 基本空闲，说明当前剩余慢路径主要是 CPU 上的 `ES`。
5. 使用 `quito` 环境 Python 读取 prediction cache shard `status.json`，汇总五专家完成、运行和缺失状态。
6. 检查正式输出根目录下是否已经存在 merged cache、oracle、router 或 calibration 结果目录。

## 结果

截至 `2026-06-15 12:56:17 CST`，正式 prediction cache 最新状态为：

```text
status_files=297
completed=292
running=5
failed=0
```

各专家状态：

| 专家 | completed | running | failed | 备注 |
| --- | ---: | --- | ---: | --- |
| `DLinear` | 64/64 | 无 | 0 | 已完成全部 sample shard |
| `PatchTST` | 64/64 | 无 | 0 | 已完成全部 sample shard |
| `CrossFormer` | 64/64 | 无 | 0 | 已完成全部 sample shard |
| `NaiveForecaster` | 64/64 | 无 | 0 | 已完成全部 sample shard |
| `ES` | 36/64 | `0008,0044,0045,0046,0047` | 0 | 剩余慢路径 |

`ES` 当前缺失 shard 为：

```text
0009,0010,0011,0012,0013,0014,0015,
0048,0049,0050,0051,0052,0053,0054,0055,
0056,0057,0058,0059,0060,0061,0062,0063
```

进程层面确认当前仍有 5 个 `ES` Python 子进程运行：原 worker 在 `sample_shard_0008_of_0064`，backfill 四个 lane 在 `sample_shard_0044_of_0064` 到 `sample_shard_0047_of_0064`。

正式输出根目录当前约 `88G`。只读检查未发现已经生成的正式 merged cache、oracle、streaming router 或 calibration 输出目录，因此原大 goal 的后半部分尚未开始。

## 结论

原始 8 项任务当前可拆解为以下完成状态：

1. Preflight：已完成。
2. 正式 full-scale manifest 和 prediction cache launcher：已完成。
3. 五专家 prediction cache shards：进行中，四个非 ES 专家已完成，`ES` 尚未完成，当前不能 merge。
4. 合并 prediction cache 和完整校验：未开始。
5. merged cache 上生成 oracle labels：未开始。
6. 正式 streaming visual router 1 epoch：未开始。
7. soft fusion calibration：未开始。
8. 正式结果汇总、handoff 和日志闭环：部分进行中，最终结果闭环未完成。

当前不应继续把后续所有步骤塞进同一个 goal。更合适的拆分方式是先单独完成 prediction cache 收尾，再为 merge、oracle、router、calibration 分别开小目标。

## 下一步方案

建议后续拆成 5 个较小目标：

1. **Prediction cache 收尾目标**：只监控 `ES`，直到 `completed=320` 且 `failed=0`；必要时只精确重跑失败的 `ES` shard，不做 merge。
2. **Merge 与完整性校验目标**：执行 launcher `status.json` 中的 `merge_command`，并校验 `sample_key + model_name` 唯一、五专家完整、共享 `y_true_path + y_true_row_index` 一致、`array_storage=packed_npy_v1`。
3. **Oracle/TSF 元信息目标**：在 merged cache 上生成 oracle labels 和必要 TSF enrichment，确保后续 router 监督和分层评估输入稳定。
4. **Streaming visual router 目标**：只使用 `train_visual_router_online_streaming.py` 跑 1 epoch 正式视觉主线，确认输出 `visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_metadata.json`，并检查没有 `.npy`、`embeddings/` 或伪图像 tensor cache。
5. **Calibration 与最终汇总目标**：运行 soft fusion calibration，生成 summary/comparison，最后更新正式 handoff、中文实验日志和总览表。

下一窗口或下一小目标建议先继续监控 `ES`，不要在 `completed=320` 前执行 merge。
