# Stage 1 `96_48_S` 正式 full-scale prediction cache 监控与抽检

日志日期：2026-06-15 02:59:03 CST

## 目的

继续推进 Stage 1 的 `96_48_S` 正式 full-scale 全候选窗口实验，在 prediction cache 长跑过程中确认当前 shard 进度、运行进程、磁盘和已完成 shard 的 packed cache 口径，判断是否已经满足 merge 条件。

## 背景

正式 full-scale sample manifest 已完成，共 `23,275,170` 个 sample_key、`64` 个 sample shard。prediction cache launcher 位于：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/
```

该阶段必须等待五个专家各完成 64 个 sample shard，即 `320` 个 completed shard 后，才能执行 `status.json` 中的 merge command。当前实验仍只做视觉主线，后续 router 必须使用 `x -> pseudo image -> frozen ViT -> router` 的 streaming online 路径，不落盘 ViT embedding 或伪图像 tensor cache。

## 操作

1. 检查当前仓库与资源状态：
   - `git rev-parse --short HEAD`
   - `git status --short`
   - `date '+%F %T %Z'`
   - `nvidia-smi --query-gpu=...`
   - `df -h /data2 /home`
   - `du -sh /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale`
2. 读取正式输出根目录下的 `HANDOFF.md`，确认当前任务仍是正式 full-scale prediction cache 长跑，不是 1k、dry-run 或 launcher 兼容性检查。
3. 读取 launcher 的 `status.json` 和各 shard 的 `status.json`，统计 completed、running、failed 数量。
4. 使用 `pgrep -af 'build_prediction_cache_from_manifest.py|prediction_cache_full_scale_launcher'` 核对五个 worker 对应的 Python 子进程。
5. 抽检当前已完成的 8 个 shard 的 `manifest.csv`：
   - 行数；
   - `sample_key + model_name` 重复；
   - `array_storage`；
   - `model_name`；
   - 跨已完成专家的 `y_true_path + y_true_row_index` 一致性；
   - 每个 sample_key 当前已完成专家数量。
6. 更新正式输出根目录的 `HANDOFF.md`，把最新进度和接手口径落盘。

## 结果

1. 当前 commit 仍为 `1829af6`；工作区存在前序实验和日志相关未提交改动，本轮未回退或清理这些既有改动。
2. 资源状态：
   - `/data2` 可用空间约 `2.6T`；
   - `/home` 仍接近满盘，只剩约 `18G`；
   - 正式 full-scale 输出根目录当前约 `6.5G`；
   - GPU 0/1/2/3 均未出现异常满显存，DLinear、PatchTST、CrossFormer 仍按 launcher 绑定 GPU 0/1/2。
3. shard 状态：
   - `status_files=13`；
   - `completed=8`；
   - `running=5`；
   - `failed=0`。
4. 各专家进度：
   - `DLinear`：已完成 `sample_shard_0000_of_0064` 和 `sample_shard_0001_of_0064`，正在运行 `sample_shard_0002_of_0064`；
   - `PatchTST`：已完成 `sample_shard_0000_of_0064` 和 `sample_shard_0001_of_0064`，正在运行 `sample_shard_0002_of_0064`；
   - `CrossFormer`：已完成 `sample_shard_0000_of_0064` 和 `sample_shard_0001_of_0064`，正在运行 `sample_shard_0002_of_0064`；
   - `NaiveForecaster`：已完成 `sample_shard_0000_of_0064` 和 `sample_shard_0001_of_0064`，正在运行 `sample_shard_0002_of_0064`；
   - `ES`：仍在运行 `sample_shard_0000_of_0064`。
5. 已完成 shard 抽检结果：
   - 8 个 completed shard 每个均为 `363,675` 行；
   - 单 shard 内 `sample_key + model_name` 重复数均为 `0`；
   - 全部 `array_storage` 均为 `packed_npy_v1`；
   - 当前 8 个 completed shard 合并抽检覆盖 `727,350` 个 sample_key；
   - 跨已完成专家的 `global_duplicate_sample_model=0`；
   - `shared_y_true_violations=0`；
   - 每个 sample_key 当前已有 4 个专家结果，分布为 `{4: 727350}`，这是因为 ES 尚未完成对应 shard。
6. 当前 `completed=8`，距离 merge 需要的 `completed=320` 仍未满足，因此本轮没有执行 merge、oracle、streaming router 或 calibration。

## 结论

prediction cache 正式长跑健康推进，四个非 ES 专家已经从 shard 0001 推进到 shard 0002，ES 仍在首个 shard。已完成的 packed cache 口径与正式协议一致，暂未发现重复键、array storage 异常或共享 y_true 对齐问题。

当前阶段仍是 prediction cache 生产和监控，不应启动 merge。必须继续等待五专家 64 个 sample shard 全部完成后，再进入合并和后续 oracle / streaming router / calibration。

## 下一步方案

1. 继续监控 `prediction_cache_full_scale_launcher/shards/*/sample_shard_*/status.json`，直到 completed shard 数达到 `320`。
2. 如果出现 failed shard，只精确定位并重跑失败 shard，不删除已完成 shard。
3. completed 达到 `320` 后，执行 launcher `status.json` 中的 `merge_command`。
4. merge 后校验 `sample_key + model_name` 唯一、每个 sample_key 五专家完整、共享 `y_true_path + y_true_row_index` 一致、`array_storage=packed_npy_v1`。
5. 再生成 oracle labels、TSF enrichment 和 baseline，随后使用 `train_visual_router_online_streaming.py` 跑 1 epoch 正式 streaming visual router，并执行 soft fusion calibration。

### 2026-06-15 03:01:31 CST handoff 追加

完成本轮监控小目标后，按项目 65% 上下文阈值规则触发 handoff。最后一次轻量状态检查显示 `completed=11`、`running=5`、`failed=0`：`DLinear`、`PatchTST`、`NaiveForecaster` 已进入 shard 0003，`CrossFormer` 仍在 shard 0002，`ES` 仍在 shard 0000。严格 CSV 抽检仍以上文 8 个 completed shard 为准；新窗口接手后应优先复核新增 completed 的 shard 0002。

已同步更新：

- 根目录 `HANDOFF.md`
- `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md`

### 2026-06-15 03:06:09 CST 续接抽检与 handoff 追加

续接后复核 launcher 实时状态，当前 `status_files=17`、`completed=12`、`running=5`、`failed=0`。`DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 均已完成 shard 0000、0001、0002，并正在运行 shard 0003；`ES` 仍在 shard 0000。

使用 `quito` 环境对 12 个 completed shard 重新做严格 CSV 抽检，结果如下：

- 12 个 completed shard 每个均为 `363,675` 行；
- 单 shard 内 `sample_key + model_name` 重复数均为 `0`；
- 全部 `array_storage` 均为 `packed_npy_v1`；
- 合并抽检覆盖 `1,091,025` 个 sample_key；
- 跨已完成专家的 `global_duplicate_sample_model=0`；
- `shared_y_true_violations=0`；
- 当前每个 sample_key 已有 4 个专家结果，分布为 `{4: 1091025}`，原因是 ES 尚未完成首 shard。

当前距离 merge 需要的 `completed=320` 仍很远，本轮没有执行 merge、oracle、streaming router 或 calibration。完成该小目标后再次触发 65% 上下文 handoff 条件，已更新根目录 `HANDOFF.md` 与正式输出根目录 `HANDOFF.md`。

### 2026-06-15 03:10:09 CST 轻量状态复核与 handoff 追加

由于当前上下文已远超 65% 阈值，本次只做最小轻量状态复核，没有执行新增 shard 的严格 CSV 抽检，也没有执行 merge、oracle、streaming router 或 calibration。

轻量状态如下：

- `status_files=21`；
- `completed=16`；
- `running=5`；
- `failed=0`；
- `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 已完成 shard 0000、0001、0002、0003，并正在运行 shard 0004；
- `ES` 仍在 shard 0000。

严格 CSV 抽检仍以上一节的 12 个 completed shard 为准，新增 shard 0003 尚未严格抽检。已同步更新根目录 `HANDOFF.md` 和正式输出根目录 `HANDOFF.md`。

### 2026-06-15 03:12:36 CST 再次轻量确认

再次复核 launcher 与 worker 进程，当前状态未变化，仍为 `status_files=21`、`completed=16`、`running=5`、`failed=0`。四个非 ES 专家仍在 `sample_shard_0004_of_0064`，ES 仍在 `sample_shard_0000_of_0064`。本次没有新增严格 CSV 抽检，仍不满足 merge 条件。

### 2026-06-15 03:19:02 CST 轻量状态更新

再次复核 launcher 与 worker 进程，当前轻量状态为 `status_files=25`、`completed=20`、`running=5`、`failed=0`。`DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 均已完成 shard 0000 到 0004，并正在运行 shard 0005；`ES` 仍在 shard 0000。由于当前上下文已远超 65% 阈值，本次没有对新增 shard 0003/0004 做严格 CSV 抽检，也没有执行 merge、oracle、streaming router 或 calibration。

### 2026-06-15 03:21:21 CST 再次轻量确认

再次复核 launcher 与 worker 进程，当前状态未变化，仍为 `status_files=25`、`completed=20`、`running=5`、`failed=0`。四个非 ES 专家仍在 `sample_shard_0005_of_0064`，ES 仍在 `sample_shard_0000_of_0064`。本次没有新增严格 CSV 抽检，仍不满足 merge 条件。

### 2026-06-15 03:35:58 CST completed shard 严格只读 audit

续接后复核实时状态，并使用 `quito` 环境执行只读 audit，输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/audits/2026-06-15_033558_completed_manifest_audit/
```

audit 快照状态如下：

- `status_files=33`；
- `completed=28`；
- `running=5`；
- `failed=0`；
- 四个非 ES 专家均已完成 `sample_shard_0000_of_0064` 到 `sample_shard_0006_of_0064`，并运行 `sample_shard_0007_of_0064`；
- `ES` 仍在 `sample_shard_0000_of_0064`。

严格 audit 覆盖快照时所有 28 个 `status=completed` shard，补上了此前未严格抽检的 shard 0003/0004，并额外覆盖 shard 0005/0006。检查结果：

- 共读取并校验 `10,182,900` 行 completed shard manifest；
- 按 sample shard 合并覆盖检查 `2,545,725` 个 sample_key-shard 组合；
- 每个 completed shard 的 `array_storage` 均为 `packed_npy_v1`；
- `global_duplicate_sample_model=0`；
- `shared_y_true_violations=0`；
- `stable_metadata_violations=0`；
- 每个已 audit 的 sample shard 当前均只有四个非 ES 专家结果，分布为 `{4: 363675}`；这是因为 ES 尚未完成首个 shard，不是 cache 对齐错误。

本次仍未执行 merge、oracle、streaming router 或 calibration。原因是当前 `completed=28`，距离正式 merge 要求的 `completed=320` 仍很远。

### 2026-06-15 03:41:31 CST audit 后轻量状态确认

audit 完成后再次执行轻量状态快照，当前状态已前进为：

- `status_files=37`；
- `completed=32`；
- `running=5`；
- `failed=0`；
- `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 已完成到 `sample_shard_0007_of_0064`，并运行 `sample_shard_0008_of_0064`；
- `ES` 仍在 `sample_shard_0000_of_0064`。

运行进程仍健康存在，最新 Python 子进程为 `DLinear=461965`、`PatchTST=463302`、`CrossFormer=464446`、`ES=427585`、`NaiveForecaster=462203`。正式输出根目录约 `14G`，`/data2` 仍约 `2.6T` 可用，`/home` 仍接近满盘。已同步更新根目录 `HANDOFF.md` 和正式输出根目录 `HANDOFF.md`。

### 2026-06-15 03:45:47 CST 最新轻量状态修正

写入 handoff 后再次做轻量 sanity check，现场状态继续前进：

- `status_files=39`；
- `completed=34`；
- `running=5`；
- `failed=0`；
- `DLinear`、`NaiveForecaster` 已完成到 `sample_shard_0008_of_0064`，并运行 `sample_shard_0009_of_0064`；
- `PatchTST`、`CrossFormer` 已完成到 `sample_shard_0007_of_0064`，并运行 `sample_shard_0008_of_0064`；
- `ES` 仍在 `sample_shard_0000_of_0064`。

本次只是轻量状态修正，没有新增严格 CSV audit。严格 audit 仍以 `2026-06-15 03:35:58 CST` 的 28 个 completed shard 为准。
