# Default baseline MSE-best 复盘与续训脚本修正

## 日志日期

2026-06-10 14:35:19 CST

## 目的

记录上午围绕 QuitoBench default baseline 的 protocol 复盘、脚本修正和补评估工作，避免后续混淆 `validation MAE` best checkpoint 与论文 protocol 中的 `validation MSE` best checkpoint。

## 背景

当前 `quito/outputs/default_baseline/` 中已有的 baseline 结果来自早先生成的配置：

- seed：16
- 训练 epoch：5
- batch size：1024
- learning rate：1e-3
- best checkpoint 选择指标：`validation MAE`

需要注意：这不是严格意义的 early stopping，因为配置里 `enable_early_stopping: false`，训练不会提前停止；但 checkpointing 仍会依据 `es_metric: mae` 标记 `best_epoch=..._MAE=...ckpt`，后续 evaluate 也使用该 MAE-best checkpoint。

论文 protocol 与主流 LTSF 实现更一致的做法是按 `validation MSE` 做模型选择。因此本次工作的目标是：

1. 保留原始 `outputs/default_baseline/` 中的 MAE-best 结果。
2. 从已有训练 event 和 per-epoch checkpoint 中复盘 MSE-best epoch。
3. 仅在 MSE-best 与 MAE-best epoch 不同时补跑 test evaluate。
4. 修正后续 baseline 编排脚本，使默认 checkpoint 选择指标改为 MSE，并支持断点续训。

## 操作

### 1. 修正 baseline 编排脚本

检查并确认 `experiment_scripts/run_default_baseline_finetune_eval.py` 已包含以下能力：

- 新增 `--selection-metric`，默认值为 `mse`。
- 在临时 finetune YAML 中写入 `early_stopping.es_metric = selection_metric`。
- 新增 `--resume-checkpoint`。
- 新增 `--resume-mode strict|model_only`：
  - `strict`：写入 `resume.checkpoint_path`，恢复完整训练状态。
  - `model_only`：写入 `model.checkpoint_path`，只加载模型权重并重建 optimizer/scheduler。
- 限制 `--resume-checkpoint` 只能配合单个 `--only` 任务使用，避免 checkpoint 套错实验。

### 2. 新增并修正 MSE-best 复盘脚本

新增脚本：

`experiment_scripts/rescore_default_baseline_ckpts_by_mse.py`

核心逻辑：

- 从 `FINE_TUNE/ver_0/events.out.tfevents.*` 读取 `valid/MSE_epoch` 与 `valid/MAE_epoch`。
- 从 `checkpoints/ckpt_epoch=...` 建立 epoch 到 checkpoint 的映射。
- 按 validation MSE 找出 MSE-best epoch。
- 读取当前 `best_epoch=...` 的 MAE-best epoch。
- 只在两者 epoch 不同时补跑 test evaluate。

上午修正了两个关键问题：

1. 复盘脚本遇到仍在运行或尚不完整的任务时，不再退出整个流程。
   - 可标注状态：`running`、`no_events`、`no_validation_yet`、`no_checkpoints`、`no_matching_epochs`。
2. `ckpt_epoch=...` 与 `best_epoch=...` 是两个不同文件，即使表示同一轮也不能按路径判断差异；已改为按 epoch 判断 MSE-best 与 MAE-best 是否不同。

### 3. 运行 dry-run 复盘

命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  experiment_scripts/rescore_default_baseline_ckpts_by_mse.py \
  --dry-run
```

结果：

- 脚本正常完成。
- 当时仍在运行的任务被标记为 `running`。
- 已完成任务中只有 `PatchTST 576_288_S` 出现 MSE-best 与 MAE-best epoch 不同。

### 4. 对不同 epoch 的任务补跑 MSE-best test evaluate

命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  experiment_scripts/rescore_default_baseline_ckpts_by_mse.py \
  --evaluate-different \
  --gpu-ids 3 \
  --num-processes 1 \
  --eval-batch-size 512
```

输出目录：

`experiment_logs/run_outputs/2026-06-10_110642_042704_default_baseline_mse_best_rescore/`

补评估结果目录：

`quito/outputs/default_baseline_mse_best/patchtst/576_288_S/seed_16/EVALUATE/ver_0/`

## 结果

复盘汇总文件：

`experiment_logs/run_outputs/2026-06-10_110642_042704_default_baseline_mse_best_rescore/mse_best_summary.csv`

已完成任务的 MSE-best 与 MAE-best 对比：

| 模型 | 配置 | 状态 | MSE-best epoch | MAE-best epoch | 是否不同 |
| --- | --- | --- | ---: | ---: | --- |
| CrossFormer | 96_48_S | completed | 4 | 4 | 否 |
| DLinear | 1024_512_S | completed | 4 | 4 | 否 |
| DLinear | 576_288_S | completed | 4 | 4 | 否 |
| DLinear | 96_48_S | completed | 4 | 4 | 否 |
| PatchTST | 576_288_S | completed | 3 | 4 | 是 |
| PatchTST | 96_48_S | completed | 4 | 4 | 否 |

当时仍在运行的任务：

| 模型 | 配置 | 状态 | 当前可见 MSE-best epoch | 当前可见 MAE-best epoch |
| --- | --- | --- | ---: | ---: |
| CrossFormer | 1024_512_S | running | 0 | 0 |
| CrossFormer | 576_288_S | running | 0 | 0 |
| PatchTST | 1024_512_S | running | 0 | 0 |

`PatchTST 576_288_S` 补评估结果：

| checkpoint 口径 | epoch | test MSE | test MAE | test MASE |
| --- | ---: | ---: | ---: | ---: |
| MAE-best | 4 | 70.642736 | 0.430489 | 740.461706 |
| MSE-best | 3 | 76.565120 | 0.461386 | 1462.475403 |

在这组 5-epoch single-seed 快速 baseline 上，切到 validation MSE-best 口径后，`PatchTST 576_288_S` 的 test MAE 变差。但这只说明该快速 baseline 的单点行为，不改变论文 protocol 应按 validation MSE 做 model selection 的事实。

## 结论

1. `validation MAE` 与 `validation MSE` 在这里不影响训练 loss、梯度传播、optimizer 更新或学习率调度；它们只影响最终选择哪个 checkpoint 做 test evaluate。
2. 因为每轮 checkpoint 都已保存，修正现有 5-epoch baseline 的 MSE-best 口径不需要重训，只需要复盘 event 并按需补 evaluate。
3. 当前 `quito/outputs/default_baseline/` 应继续视为原始 MAE-best baseline 输出，包括 finetune 和 evaluate。
4. 新补的 MSE-best evaluate 单独放在 `quito/outputs/default_baseline_mse_best/`，未覆盖原始结果。
5. 若发现训练不充分，可以从已有 checkpoint 继续训练；但从 5-epoch 配置 strict resume 再加 15 epoch 不等价于从头训练 20 epoch，主要因为原始 cosine scheduler 的 `T_max=5` 已按 5 epoch 走完。

## 下一步方案

1. 等仍在运行的旧 baseline 任务结束后，再重新执行 MSE-best 复盘。
2. 若只做探索，可用 `model_only` 从当前 best 或 last checkpoint 继续训练 15 epoch，观察 validation 曲线是否继续改善。
3. 若做正式可比实验，应从头按目标 epoch 数训练，并使用 `selection_metric=mse`、正确的 scheduler 总长度、必要的 tuning 和多 seed。
4. 后续不要把 MAE-best 原始结果与 MSE-best 补评估结果混在同一个输出根目录中；继续使用 `outputs/default_baseline_mse_best/` 或新的明确命名目录。
