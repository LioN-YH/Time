# 默认配置单 seed baseline 训练评估启动

日志日期：2026-06-10 03:56:46 CST

## 目的

优先获得 PatchTST、CrossFormer、DLinear 在 `96_48_S`、`576_288_S`、`1024_512_S` 三个设置上的单 seed baseline 训练评估结果，作为后续超参搜索完成前的临时性能表和 cluster 分析基础。

## 背景

此前正在运行 PatchTST / CrossFormer 的单卡并发超参粗搜，但预计耗时较长。用户确认当前不需要 3 个 seed，先跑单 seed 即可；同时决定暂停超参搜索，优先完成当前默认配置 baseline 实验。

由于每个 baseline 实验都要求 4 卡并行，而机器总共 4 张 GPU，因此 baseline 任务采用“每组实验占满 4 卡、9 组顺序执行”的方式，不能再同时并发多组 4 卡实验。

## 操作

1. 停止当前单卡并发超参粗搜相关进程：
   - 原后台编排 PID：`2914604`
   - 对应 `quito-cli tune` 子进程和 Ray worker 已不再出现在进程表中。

2. 删除用户指定的半成品输出：
   - 删除 `experiment_logs/run_outputs/` 下的 `*patchtst_crossformer_single_gpu_screen*` 运行目录和主日志。
   - 删除 `quito/outputs/single_gpu_screen`。
   - 保留 smoke test、cluster 数据和中文实验日志。

3. 新增 baseline 编排脚本：
   - 脚本路径：`experiment_scripts/run_default_baseline_finetune_eval.py`
   - 覆盖任务：`DLinear / PatchTST / CrossFormer` x `96_48_S / 576_288_S / 1024_512_S`
   - 固定 seed：`16`
   - 每组任务使用 `num_processes=4`，GPU 为 `0,1,2,3`
   - 模型架构参数沿用官方 `configs/finetune` 和 `configs/evaluate`
   - 训练侧参数统一为：
     - 每卡 `batch_size=256`
     - 全局 batch size 为 `1024`
     - `eval_batch_size=256`
     - `learning_rate=0.001`
     - `num_epochs=5`

4. 脚本增加 cluster 分析产物：
   - 原始 Quito 评估 JSON 保持不变。
   - 每组评估完成后额外输出：
     - `per_item_results.csv`
     - `cluster_metrics.csv`
     - `summary.json`
   - 汇总根目录位于本次 run 目录的 `cluster_analysis/` 下。

5. 完成 dry-run 验证：
   - dry-run 运行目录：`experiment_logs/run_outputs/2026-06-10_035521_default_baseline_finetune_eval`
   - 验证通过：9 组临时配置、命令、日志路径和状态文件均能生成。

6. 启动正式后台任务：
   - 后台 PID：`2945456`
   - 主日志：`experiment_logs/run_outputs/default_baseline_finetune_eval_setsid_2026-06-10_035629.log`
   - 正式运行目录：`experiment_logs/run_outputs/2026-06-10_035629_default_baseline_finetune_eval`
   - 状态文件：`experiment_logs/run_outputs/2026-06-10_035629_default_baseline_finetune_eval/status.json`
   - 当前首个任务：`dlinear_96_48_S_seed16`

## 结果

当前 baseline 编排脚本已通过语法检查和 dry-run 验证，并已正式后台启动。启动时 GPU 上只剩此前无法清理的 `[Not Found]` 驱动残留记录，没有活跃 tuning/Ray worker。

正式任务会按如下顺序执行：

1. `dlinear_96_48_S_seed16`
2. `patchtst_96_48_S_seed16`
3. `crossformer_96_48_S_seed16`
4. `dlinear_576_288_S_seed16`
5. `patchtst_576_288_S_seed16`
6. `crossformer_576_288_S_seed16`
7. `dlinear_1024_512_S_seed16`
8. `patchtst_1024_512_S_seed16`
9. `crossformer_1024_512_S_seed16`

## 结论

超参粗搜已暂停并清理半成品结果；当前实验路线切换为单 seed default baseline。该 baseline 不代表最终 tuned 性能，但能先产出完整训练、评估和 cluster 分析所需的基础结果。

## 下一步方案

1. 持续观察 `status.json`、主日志和各任务独立日志，确认第一组 DLinear 能完成训练与评估闭环。
2. 若出现 OOM，可优先将每卡 `batch_size` 从 `256` 降到 `128` 后续跑失败任务。
3. 9 组 baseline 完成后，读取 `baseline_summary.csv` 和 `cluster_analysis/` 目录汇总整体指标与 cluster 分层表现。
4. 后续阶段再恢复超参搜索，或直接用 tuned top 配置重跑同一套 finetune/evaluate 流程。
