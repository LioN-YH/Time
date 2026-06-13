# PatchTST / CrossFormer 4 卡 tuning 编排脚本与 dry-run 验证

日志日期：2026-06-10 03:12:46 CST

## 目的

为 PatchTST 和 CrossFormer 的 6 组架构超参数搜索任务建立统一编排脚本，并在正式启动前验证命令生成、日志路径、状态文件和 GPU 可见性。

## 背景

前序 smoke test 已确认 DLinear、PatchTST、CrossFormer 的最小训练与评估闭环可运行。下一步需要对 PatchTST 和 CrossFormer 运行默认架构搜索空间，任务范围为 `96_48_S`、`576_288_S`、`1024_512_S`。

用户已确认采用“每个 tuning job 独占 4 张 GPU、6 组顺序执行”的方案，以尽量匹配后续 4 卡 finetune 的训练条件，避免单卡 tuning 和 4 卡 finetune 之间因有效 batch size 变化产生额外不一致。

## 操作

1. 检查项目级规范 `AGENTS.md`、实验日志总览 `experiment_logs/README.md`、以及 Quito 的 `configs/tune/patchtst/tuning_config.yaml` 和 `configs/tune/crossformer/tuning_config.yaml`。
2. 审查已有脚本 `experiment_scripts/run_patchtst_crossformer_tuning.py`，发现旧版本实现为“多组单卡任务并发”，与最终方案不一致。
3. 重写 `experiment_scripts/run_patchtst_crossformer_tuning.py`：
   - 默认顺序执行 6 组任务。
   - 每组任务传入 `--num_processes 4`。
   - 每组任务设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`。
   - 每组任务合并记录 stdout/stderr 到独立日志文件。
   - 写入 `status.json`，记录任务状态、命令、返回码、开始/结束时间和预期输出根目录。
   - 增加 `--dry-run`、`--only`、`--continue-on-failure`，方便验证和从失败点续跑。
4. 执行语法检查：

   ```bash
   python -m py_compile experiment_scripts/run_patchtst_crossformer_tuning.py
   ```

5. 检查 4 张 GPU 可见性：

   ```bash
   nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
   ```

6. 执行 dry-run：

   ```bash
   python experiment_scripts/run_patchtst_crossformer_tuning.py --dry-run
   ```

## 结果

1. Python 语法检查通过。
2. 4 张 GPU 均可见，型号均为 NVIDIA GeForce RTX 3090，执行前显存占用较低：
   - GPU 0：714 MiB / 24576 MiB
   - GPU 1：363 MiB / 24576 MiB
   - GPU 2：10 MiB / 24576 MiB
   - GPU 3：355 MiB / 24576 MiB
3. dry-run 成功生成 6 组任务命令，运行目录为：

   ```text
   /home/shiyuhong/Time/experiment_logs/run_outputs/2026-06-10_031235_patchtst_crossformer_4gpu_tuning
   ```

4. dry-run 状态文件为：

   ```text
   /home/shiyuhong/Time/experiment_logs/run_outputs/2026-06-10_031235_patchtst_crossformer_4gpu_tuning/status.json
   ```

5. dry-run 中 6 组任务均为 `planned`，返回码均为 0。该 dry-run 只用于验证命令编排，不是正式实验结果；相关 `run_outputs` 目录已在 2026-06-11 的废弃半成品清理中删除，保留本日志用于记录脚本验证结论。示例命令如下：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/quito-cli tune --config_path configs/tune/patchtst/96_48_S.yaml --tuning_config_path configs/tune/patchtst/tuning_config.yaml --num_processes 4 --num_samples 10 --use_gpu 1
   ```

## 结论

4 卡顺序 tuning 编排脚本已准备完成，dry-run 证明 6 组任务的命令、日志文件和状态文件可以正常生成。脚本当前符合“每个 tuning job 独占 4 张 GPU、6 组顺序执行”的实验方案。

## 下一步方案

启动正式 tuning：

```bash
python experiment_scripts/run_patchtst_crossformer_tuning.py
```

正式运行完成后，读取每组 `status.json`、stdout/stderr 日志和 Quito 输出目录，整理 6 组 tuning 的返回码、最佳参数和后续 finetune 配置衔接方案，并新增独立实验日志。
