# 默认配置单 GPU 并发 baseline 启动

日志日期：2026-06-10 04:15:54 CST

## 目的

根据 GPU 利用率观察结果，将默认配置 baseline 从“每组实验 4 卡顺序执行”切换为“单 GPU 多任务并发执行”，更快获得 3 个模型在 3 个目标设置上的单 seed 训练评估结果，并继续保存 cluster 分析所需结果。

## 背景

4 卡顺序 baseline 启动后，首个 `dlinear_96_48_S_seed16` 任务虽然能正常训练，但每张 GPU 显存占用只有数百 MB，GPU 利用率约 5%-6%。这说明 DLinear 这类小模型在 4 卡 DDP 下主要受数据加载和 DDP 开销限制，资源利用率很低。

用户确认采用单 GPU 多任务并发策略，并建议如果 4 组并发后某些 GPU 仍明显空闲，可以继续向空闲 GPU 填入额外任务。

## 操作

1. 清理已中止的 4 卡顺序 baseline 半成品：
   - 停止原后台 PID `2945456` 及其训练子进程。
   - 删除正式 4 卡顺序 run 目录：`experiment_logs/run_outputs/2026-06-10_035629_default_baseline_finetune_eval`
   - 删除对应主日志：`experiment_logs/run_outputs/default_baseline_finetune_eval_setsid_2026-06-10_035629.log`
   - 删除半成品模型输出：`quito/outputs/default_baseline`

2. 更新项目规范：
   - 在 `AGENTS.md` 中新增“中止实验和半成品清理规范”。
   - 规范说明：策略快速变化时，如果用户确认中止实验，应停止进程并精确删除对应半成品输出和运行日志，同时更新 README 状态。

3. 修复单 GPU 并发启动问题：
   - 初次 4 lane 并发时发现 `num_processes=1` 仍通过 `quito-cli finetune` 调用 `torchrun`，多个 lane 会争用默认 rendezvous 端口，导致非首个 lane 快速失败。
   - 修改 `experiment_scripts/run_default_baseline_finetune_eval.py`：
     - `num_processes=1` 时直接调用 `quito/quito/scripts/finetune.py`，绕过 `torchrun`。
     - run 目录时间戳改为微秒级，避免多个 lane 同秒启动时共用一个 `status.json`。
   - 完成 `py_compile` 和单 GPU dry-run 验证。

4. 清理失败的 4 lane 并发半成品：
   - 删除 `experiment_logs/run_outputs/2026-06-10_041031_default_baseline_finetune_eval`
   - 删除 `default_baseline_single_gpu_lane*_2026-06-10_041031.log`
   - 删除 `quito/outputs/default_baseline`

5. 第一次稳定启动 4 lane 后，发现 GPU0 因 DLinear 仍低利用率，因此清理 `041247` 轮半成品并重排任务，最终改为 5 lane：
   - GPU0 同时运行 DLinear 队列和 CrossFormer 576 队列。
   - GPU1/GPU2/GPU3 分别运行其余队列。

## 当前运行任务

正式 5 lane 启动时间戳：`2026-06-10_041445`

| Lane | GPU | PID | 任务队列 | 主日志 |
| --- | --- | --- | --- | --- |
| lane0a | 0 | `2957862` | `dlinear:96_48_S,dlinear:576_288_S,dlinear:1024_512_S` | `experiment_logs/run_outputs/default_baseline_single_gpu_lane0a_2026-06-10_041445.log` |
| lane0b | 0 | `2957863` | `crossformer:576_288_S` | `experiment_logs/run_outputs/default_baseline_single_gpu_lane0b_2026-06-10_041445.log` |
| lane1 | 1 | `2957864` | `patchtst:96_48_S,patchtst:1024_512_S` | `experiment_logs/run_outputs/default_baseline_single_gpu_lane1_2026-06-10_041445.log` |
| lane2 | 2 | `2957865` | `crossformer:96_48_S,crossformer:1024_512_S` | `experiment_logs/run_outputs/default_baseline_single_gpu_lane2_2026-06-10_041445.log` |
| lane3 | 3 | `2957866` | `patchtst:576_288_S` | `experiment_logs/run_outputs/default_baseline_single_gpu_lane3_2026-06-10_041445.log` |

每个任务设置：

- `num_processes=1`
- 训练 `batch_size=1024`
- 评估 `eval_batch_size=512`
- `seed=16`
- `learning_rate=0.001`
- `num_epochs=5`

## 结果

5 lane 启动后均保持运行。启动约 30 秒后的 GPU 状态：

- GPU0：约 `99%` 利用率，约 `10.6 GB` 显存，占用来自 CrossFormer 576 和 DLinear。
- GPU1：约 `92%` 利用率，约 `2.0 GB` 显存，运行 PatchTST 96。
- GPU2：约 `29%` 利用率，约 `0.6 GB` 显存，运行 CrossFormer 96。
- GPU3：约 `99%` 利用率，约 `6.7 GB` 显存，运行 PatchTST 576。

当前 9 个目标任务已经全部排入队列，不再额外添加重复任务。

## 结论

单 GPU 并发 baseline 已替代 4 卡顺序 baseline。该策略更适合当前“快速获得单 seed default baseline”的目标；最终论文或正式报告中若需要严格对齐 4 卡训练制度，后续仍可对 tuned 配置或关键 baseline 重新做 4 卡验证。

## 下一步方案

1. 持续监控 5 个 lane 的主日志和各自 `status.json`。
2. 若出现 OOM，优先将对应任务的训练 `batch_size` 从 `1024` 降至 `512` 后只重跑失败任务。
3. 若出现 Ray evaluate 多实例冲突，先完成训练，再单独排队执行 evaluate。
4. 所有 lane 完成后，汇总各 run 目录中的 `baseline_summary.csv`、`per_item_results.csv` 和 `cluster_metrics.csv`。
