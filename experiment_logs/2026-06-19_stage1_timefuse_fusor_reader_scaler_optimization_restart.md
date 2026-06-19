# Stage 1 TimeFuse-style fusor reader/scaler 优化与 GPU2/3 重启

日志日期：2026-06-19 11:55:23 CST

## 目的

为 Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline 提供一个可继续长跑的版本，解决 index/scaler/训练首批次阶段持续过慢的问题，并保持正式公平比较要求：训练使用 `CUDA_VISIBLE_DEVICES=2,3` 双卡 GPU。

## 背景

旧 GPU2/3 正式任务已经完成 64 个 shard-local oracle SQLite index 和 64 个 prediction SQLite index，但长时间停留在 `scaler_partial_fit`。检查后确认，旧 scaler 复用了完整 streaming reader，导致只需要 17 维 feature 的 StandardScaler 阶段也反复读取 oracle labels、五专家 `y_pred/y_true` 和 expert errors。随后 feature-only scaler 版本虽然快速完成，但训练首个 batch 又暴露两个 reader 问题：split 过滤发生在 prediction arrays 读取之后，且 packed `.npy` 在 batch 内按 sample 反复 `np.load`。

## 操作

1. 停止旧 GPU2/3 慢进程，保留同一正式输出目录，不删除已有 smoke/pressure 结果和已完成 index。
2. 在 `train_timefuse_fusor_streaming.py` 中增加 shard-local SQLite index 复用判断：若 oracle/prediction index 行数匹配当前 shard 的 sample_key 数和五专家完整性，则跳过重建。
3. 将 `fit_scaler_streaming()` 改为 feature-only streaming：只读取 feature CSV 的 `split` 与 17 维 feature 列，只在 vali split 上 `partial_fit`，不读取 oracle/prediction arrays。
4. 在 `stage1_timefuse_fusor_streaming_reader.py` 中增加 `split_filter`，把 train/eval 的 split 过滤下推到 feature CSV 层，避免先读取随后会被丢弃的非目标 split prediction arrays。
5. 将 batch 内 packed `.npy` 读取改为按数组路径分组读取：同一 batch 中同一路径只 `np.load(mmap_mode="r")` 一次，再按 row index 批量切片；legacy per-sample 小文件仍走原统一读取接口。
6. 将 reader 的原始 CSV 读取改为大块读取，split 过滤后再切成稳定 `batch_size`，避免过滤后产生几十个样本的小 batch。
7. 使用 Quito 环境执行语法检查，并用 shard 0000 的现有 SQLite index 做极小 vali reader 验证。
8. 通过同一正式目录的 `resume.sh` 恢复 GPU2/3 后台任务。

## 结果

- 语法检查通过。
- 极小 reader 验证通过：`split_filter='vali'` 后首个 batch 全部为 `vali`，`y_pred_shape=[16, 5, 48, 1]`，`y_true_shape=[16, 48, 1]`。
- 正式目录 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` 已复用 64/64 shard-local SQLite index。
- feature-only scaler 在本轮恢复中约从 `11:50:32 CST` 到 `11:51:49 CST` 完成 64 个 shard，覆盖 `9,350,520` 个 vali sample。
- 后台任务已恢复并进入训练：
  - PID/PGID：`1845436/1845436`
  - Python 子进程：`1845438`
  - `CUDA_VISIBLE_DEVICES=2,3`
  - `main.log` 出现 `启用 DataParallel 双卡训练 logical_devices=[0, 1]`
  - `status.json` 于 `2026-06-19 11:55:05 CST` 显示 `phase=train`、`epoch=1`、`current_shard=sample_shard_0001_of_0064`、`train_batches=600`、`train_samples=153526`、`latest_loss=0.21573381125926971`
  - 收尾复核于 `2026-06-19 11:58:19 CST` 显示已继续推进到 `current_shard=sample_shard_0002_of_0064`、`train_batches=1200`、`train_samples=307052`、`latest_loss=0.11619359254837036`
  - GPU2/GPU3 均有 PyTorch 显存占用，`nvidia-smi` 约为 `441 MiB / 441 MiB`

## 结论

用户关于“超过 24 小时仍在 index/scaler 阶段不合理”的判断是正确的。主要问题不是 TimeFuse fusor 模型计算本身，而是数据读取路径把 scaler 和 split 过滤做在了过重的 reader 之后，并且 packed arrays 在 batch 内重复打开。当前版本已经把 scaler 降为 feature-only，把 index 变为可复用，把 split 过滤下推到数组读取前，并把 packed npy 改成 batch-level grouped loading；正式 GPU2/3 任务已进入可监控的 train 阶段。

## 下一步方案

1. 继续轻量监控正式目录的 `status.json`、`main.log`、PID/PGID 和 GPU2/GPU3 占用，不要扫描 full merged `manifest.csv`。
2. 等待 `checkpoints/latest_timefuse_fusor.pt` 写出；若训练完成后进入 test eval，再检查 `timefuse_fusor_summary.csv`、raw soft summary、selected counts 和 sample predictions。
3. 停止命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/stop.sh
```

4. 恢复命令：

```bash
bash /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/resume.sh
```
