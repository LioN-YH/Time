# Stage 1 `96_48_S` Streaming Visual Router OOM 修复审查与 v2 重启

日志日期：2026-06-17 09:46:07 CST

## 目的

审查 qoder 对 `96_48_S` full-scale streaming visual router OOM 的修复是否真正解决问题；如果修复可靠，则在物理 GPU2/GPU3 上重新启动 `--epochs 1 --train-only` 训练，并监控是否穿过上一轮 OOM 发生的 manifest lookup 压力区间。

## 背景

上一轮正式训练在 2026-06-16 22:24:57 CST 被 Linux OOM killer 杀死。失败发生在 manifest lookup 阶段，当时扫描到约 100M 行、匹配约 4000 万条记录，Python 进程 `anon-rss` 约 117GB。qoder 判断根因是预加载全量 `prediction_lookup` dict，并把代码改成“轻量级路径索引 + 按需读取”。

复查时发现 qoder 的轻量级路径索引仍有两个关键风险：

1. full-scale cache 使用 `packed_npy_v1`，读取单个样本必须保留 `array_storage`、`y_true_row_index` 和 `y_pred_row_index`；只保存路径会导致读取整块 packed `.npy` 或触发 shape/内存错误。
2. 即使只保存路径，约 4675 万条 `(sample_key, model_name)` 仍然是千万级 Python dict/string 对象，不能可靠把 RSS 降到 <1GB；旧 OOM 路径只是被推迟，不是被根治。

## 操作

1. 阅读 `HANDOFF.md`、当前 git diff、`train_visual_router_online_streaming.py`、`fusion_utils.py` 和 `prediction_array_io.py`。
2. 将 qoder 的内存修复改为 SQLite 磁盘索引：
   - 新增 `SQLitePredictionIndex`，用 SQLite 保存本次所需的 `sample_key/model_name/y_true_path/y_pred_path/mae/mse/array_storage/y_true_row_index/y_pred_row_index`。
   - `build_lightweight_prediction_index()` 分块扫描 manifest，只把当前 chunk 匹配行写入 SQLite，不在 Python 内存中保留千万级 dict。
   - `load_prediction_tensors_from_lightweight_index()` 每个训练 batch 只从 SQLite 查询当前 batch 的五专家记录，并通过 `load_prediction_array()` 按 packed row index 读取单行。
   - soft fusion 分支也改为 batch 级 SQLite 查询，避免后续 eval-only 再退回全量 lookup。
   - `required_prediction_sample_keys()` 去掉 full-scale sample_key 排序，保留原始出现顺序并 `drop_duplicates()`，减少启动阶段临时列表和排序开销。
3. 使用 Quito 环境执行语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
   ```

4. 基于正式 merged manifest 前 5 行构造绝对路径 mini manifest，验证 SQLite 索引和 packed row index 读取：
   - `y_pred_shape=(1, 5, 48, 1)`
   - `y_true_shape=(1, 48, 1)`
   - 第一个专家复算 MAE `1.6258877515792847`，与 manifest `1.625887751579285` 对齐。
5. 检查旧 v2 launcher 残留：
   - 旧 `main.log` 显示 `ModuleNotFoundError: No module named 'numpy'`，原因是 launcher 激活了错误 conda 路径。
   - 旧 PID `899001` 已不存在。
6. 重写 `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/launcher.sh`：
   - 固定使用 `/home/shiyuhong/application/miniconda3/envs/quito/bin/python`。
   - 固定 `CUDA_VISIBLE_DEVICES=2,3`。
   - 显式 `--output-dir` 指向 v2 输出目录。
7. 首次 v2 重启因 handoff 中错误 config path `/home/shiyuhong/Time/quito/configs/evaluate/default_baseline.yaml` 失败；已保留为 `main_failed_bad_config_path_2026-06-17_0837.log`。
8. 改用实际配置 `/home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml` 后重新后台启动：
   - PID/PGID：`919803 / 919803`
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/`
   - 主日志：`main.log`
   - 状态文件：`status.json`
9. 监控 GPU 和内存：
   - 物理 GPU0 有既有 `python3.10` 占用约 17GB，不是本训练。
   - 本训练通过 `CUDA_VISIBLE_DEVICES=2,3` 限制到物理 GPU2/GPU3；manifest/SQLite 阶段主要是 CPU/I/O，当前只在 GPU2 建立约 693MiB CUDA 上下文，GPU3 要到 ViT DataParallel forward 阶段才会明显使用。
10. 启动 DeepSeek sidecar 做只读监控。首轮 sidecar 误执行 `wc -l` 大 manifest，已中断；随后重启轻量版 `monitor-stage1-lite`，仅允许 `ps/tail main.log/stat index/status/nvidia-smi/free/sleep`，禁止扫描 52GB manifest。

## 结果

截至 2026-06-17 09:46 CST：

- 训练进程仍在运行：

  ```text
  PID 919803, PGID 919803, elapsed 约 01:05, RSS 约 17.2GB
  ```

- `main.log` 已穿过旧 OOM 压力区间：

  ```text
  [manifest_index] chunks=25 rows_seen=25000000 matched_rows=9769890 target_sample_keys=9350520
  [manifest_index] chunks=50 rows_seen=50000000 matched_rows=19723770 target_sample_keys=9350520
  [manifest_index] chunks=75 rows_seen=75000000 matched_rows=29950910 target_sample_keys=9350520
  [manifest_index] chunks=100 rows_seen=100000000 matched_rows=40167530 target_sample_keys=9350520
  ```

- 上一轮 OOM 在约 `rows_seen=100M / matched_rows≈40M` 时发生；本轮同等压力下 RSS 仍约 16-17GB，没有随 matched rows 线性涨到 117GB。
- SQLite 临时索引仍在构建，文件约 14.3GB：

  ```text
  prediction_manifest_index.sqlite.tmp 约 14.3GB
  ```

- `status.json` 仍显示 `phase=init`，这是脚本当前只在后续阶段更新状态文件；manifest index 进度以 `main.log` 和 SQLite 文件为准。
- 当前 GPU 状态：

  ```text
  GPU0: 17342 MiB, 0% util（既有 python3.10）
  GPU1:    12 MiB, 0% util
  GPU2:   693 MiB, 0% util（本训练 CUDA 上下文）
  GPU3:    12 MiB, 0% util
  ```

## 结论

qoder 原“只保存路径”的轻量级 Python dict 修复不够可靠：它丢失 packed row index，并且仍会把千万级 Python 对象常驻内存。本轮已改为 SQLite 磁盘索引 + batch 级查询，mini packed 验证通过，并且正式运行已经穿过上一轮 OOM 发生的 `100M rows / 40M matched` 压力区间，RSS 稳定在约 16-17GB。当前证据表明 OOM 根因已经被有效切断。

训练尚未完成，目前仍在 manifest SQLite index 构建阶段。GPU3 未明显使用是正常现象，因为当前阶段不是 ViT forward；进入 `scaler_fit` / `train_epoch_1` 后 DataParallel 才会同时使用物理 GPU2/GPU3。

## 下一步方案

1. 继续轻量监控，等待 `prediction_manifest_index.sqlite.tmp` 原子替换为 `prediction_manifest_index.sqlite`，并观察日志是否出现 `sqlite_index_ready`。
2. 索引完成后确认进入 `scaler_fit`，检查 `online_embedding_latency_summary.csv` 和 GPU2/GPU3 利用率。
3. 当前从 09:46 CST 起粗略估计：
   - SQLite index 剩余约 10-25 分钟；
   - `scaler_fit + train_epoch_1` 预计还需约 4-7 小时；
   - 总体完成窗口保守估计为 2026-06-17 14:00-17:00 CST。
4. 完成后检查：
   - `status.json` 为 `completed` / `train_only_done`；
   - `checkpoints/latest_96_48_S.pt` 存在；
   - `checkpoints/latest_checkpoint_index.json` 指向有效 checkpoint。
5. 若需要停止，优先对 PID `919803` 发送 `SIGTERM`，不要直接 `SIGKILL`。
