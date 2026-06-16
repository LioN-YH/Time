# Stage 1 `96_48_S` Full-Scale Handoff

## 最新状态（2026-06-16 02:33:22 CST）

本节记录当前窗口完成的 prediction cache merge 与完整性校验。下方 `2026-06-16 00:52:36 CST` 的 TimeFuse feature cache 状态是另一条并行任务的 handoff 信息，保留供接手时参考。

- 当前任务边界：本轮只做 Stage 1 `96_48_S` full-scale prediction cache merge 与完整性校验；没有启动 oracle labels、streaming visual router 或 calibration。
- 正式 merged cache 已完成：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/
```

- 最终 merge 日志：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_retry_compact_2026-06-16_011812.log
```

- `merged_cache/status.json` / `metadata.json` 核心结果：

```text
status: completed
generated_at: 2026-06-16 02:09:58 CST
sample_count: 23,275,170
record_count: 116,375,850
array_storage: packed_npy_v1
merge_strategy: packed_npy_v1_streaming_by_sample_shard
shared_y_true_path: true
```

- 完整性校验已通过：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/integrity_summary.json
```

校验摘要：

```text
passed: true
actual_record_count: 116,375,850
actual_sample_key_unique_count: 23,275,170
sample_group_count: 23,275,170
model_counts: each expert = 23,275,170
sample_key_model_uniqueness_violations: 0
expert_completeness_violations: 0
shared_y_true_violations: 0
stable_metadata_violations: 0
array_storage_violations: 0
array_path_violations: 0
```

- 处理过的 merge 失败点：
  - `sample_shard=0014`：ES packed `y_true/y_pred` 文件重复尾部，manifest 只引用前半段；已按 manifest row index 处理。
  - `sample_shard=0054`：ES manifest 对部分 split/dataset 引用重复追加后的后半段 row index；已在 merge 中按各专家 row index 读取内容，并在 merged cache 中重写共享 y_true 与 compact y_pred row index。
- 已修改 `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py`，并用 Quito 环境 `py_compile`、0014/0054 隔离回归和 full-scale validation 验证通过。
- 没有删除任何已完成 prediction cache shard；没有进入 oracle、router 或 calibration。

后续接力：如继续 Stage 1 主线，下一步可以在 completed `merged_cache/` 上生成 oracle labels 和 TSF enrichment，再进入 baseline / TimeFuse-style fusor / streaming visual router / calibration。最终 merge/validation 结果以本节路径为准，不要引用早期 failed retry 作为最终结果。

## 最新状态（2026-06-16 00:52:36 CST）

本节是当前接手时的最新状态；下方 `2026-06-15 23:49:51 CST` 及更早内容是 prediction cache merge 历史记录。当前新目标是 **TimeFuse-derived 单变量 feature cache full-scale 预计算**。

- 当前任务边界：只监控和收尾 TimeFuse feature cache 预计算；不要在 feature cache、merged prediction cache 和 oracle labels 齐全前启动 TimeFuse fusor 训练。
- 已新增正式脚本：
  - `visual_router_experiments/stage1_vali_test_router/build_timefuse_feature_cache_from_manifest.py`
  - `visual_router_experiments/stage1_vali_test_router/launch_timefuse_feature_cache_full_scale.py`
- 语法检查已通过；64 行 builder smoke、1024 行 builder smoke 和 2-shard launcher smoke 均通过。
- 特征口径：17 维 TimeFuse-derived 单变量元特征，只使用历史窗口 `x`；不读取未来 `y`、专家预测、prediction cache manifest、oracle label 或 TSF label 文件。
- GPU 策略：未使用 GPU。ADF、ACF、AutoReg、periodogram 等小窗口统计是 `numpy/scipy/statsmodels` CPU 计算，GPU 没有实际收益。

### 正在运行

| 项目 | 值 |
| --- | --- |
| 任务 | full-scale TimeFuse feature cache 预计算 |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/` |
| launcher | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/launcher.sh` |
| 根状态 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/status.json` |
| lane 日志 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/logs/lane_*.log` |
| shard 输出 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_XXXX_of_0064/` |
| 启动命令 | 见本节下方“恢复/重启命令” |

当前 lane PID/PGID：

```text
lane_00: 720659 / 720659
lane_01: 720661 / 720661
lane_02: 720662 / 720662
lane_03: 720664 / 720664
lane_04: 720665 / 720665
lane_05: 720666 / 720666
lane_06: 720667 / 720667
lane_07: 720668 / 720668
```

当前首批 builder PID：

```text
sample_shard_0000_of_0064: PID 720690
sample_shard_0001_of_0064: PID 720686
sample_shard_0002_of_0064: PID 720692
sample_shard_0003_of_0064: PID 720687
sample_shard_0004_of_0064: PID 720693
sample_shard_0005_of_0064: PID 720691
sample_shard_0006_of_0064: PID 720688
sample_shard_0007_of_0064: PID 720694
```

截至 `2026-06-16 00:52:36 CST`：

```text
status_files=8
running=8
rows_written_or_completed=363,489
sum_builder_rows_per_second≈1,179.24
eta_hours_at_current_sum_rps≈5.40
当前只看到首批 8 个 shard status；后续 lane 会顺序启动 shard 0008、0016、... 等分配到各 lane 的任务。
```

### 监控命令

```bash
cd /home/shiyuhong/Time
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher

/home/shiyuhong/application/miniconda3/envs/quito/bin/python - <<'PY'
import collections, json, pathlib, statistics
root = pathlib.Path('/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards')
files = sorted(root.glob('sample_shard_*/status.json'))
c = collections.Counter()
rows = 0
rps = []
for p in files:
    s = json.loads(p.read_text())
    status = s.get('status', 'unknown')
    c[status] += 1
    done = int(s.get('rows_written', s.get('sample_count', 0)) or 0)
    if status == 'completed':
        done = int(s.get('sample_count', done) or done)
    rows += done
    if float(s.get('rows_per_second', 0) or 0) > 0:
        rps.append(float(s['rows_per_second']))
print({'status_files': len(files), **dict(c), 'rows_done_or_written': rows, 'sum_rows_per_second': sum(rps)})
PY

tail -n 80 "$ROOT/logs/lane_00.log"
tail -n 80 "$ROOT/shards/sample_shard_0000_of_0064/main.log"
du -sh "$ROOT"
```

### 停止命令

```bash
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher
for p in "$ROOT"/pids/*.pid; do
  kill -TERM -- -$(cat "$p") 2>/dev/null || kill -TERM $(cat "$p") 2>/dev/null || true
done
```

### 恢复/重启命令

```bash
cd /home/shiyuhong/Time
bash /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/launcher.sh
```

恢复时会跳过 `status=completed` 的 shard；单 shard 中断时 builder 的 `--resume` 会保留完整 item 组并续跑。

### 后续验证口径

feature cache 64 个 shard 全部完成后，至少检查：

- 64 个 `feature_cache.csv` 行数总和为 `23,275,170`；
- 全局 `sample_key` 唯一；
- 每行 `feature_dim=17`，`feature_version=timefuse_single_variable_meta_v1`；
- 17 个特征列全部有限；
- `sample_key` 与 `config_name/split/dataset_name/item_id/channel_id/window_index` 一致；
- 每个 shard 的 `metadata.json` / `status.json` 均为 `completed`。

正式 TimeFuse fusor 训练仍需等待 feature cache、merged prediction cache 和 oracle labels 全部齐全。

## 最新状态（2026-06-15 23:49:51 CST）

本节是当前接手时的最新状态；下方 `2026-06-15 22:25:26 CST` 及更早内容是历史监控记录。

- 当前任务边界：只做 full-scale prediction cache merge 与完整性校验；不要启动 oracle labels、streaming visual router 或 calibration。
- 五专家 prediction cache 已完成：`status_files=320`、`completed=320`、`running=0`、`failed=0`，五个专家各 `64/64` 个 sample shard。
- 资源启动前检查：`/data2` 约 `2.5T` 可用、inode 充足；CPU/I/O 空闲；4 张 RTX 3090 基本空闲。本轮 merge 是 CPU/I/O-bound，没有改成 GPU 计算。
- `prediction_cache_full_scale_launcher/status.json` 中的 `merge_command` 已读取并用于正式启动；该字段为 list，`argc=325`，输入 `320` 个 shard，输出目录为 `prediction_cache_full_scale_launcher/merged_cache`。
- merge 前只读预检：五专家各 `23,275,170` 行，总 manifest 行数 `116,375,850`，预期 sample_key 数 `23,275,170`；首行 `array_storage` 检查全部为 `packed_npy_v1`；缺失/坏 shard 数均为 `0`。
- 已修改正式脚本 `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py`：新增 `packed_npy_v1_streaming_by_sample_shard` 分支，避免旧 packed 分支对 full-scale 逐行反复 `np.load`。执行入口仍是 status.json 的同一条 `merge_command`。
- 语法检查已通过；历史 dry-run 的 2 个 sample shard 回归通过，结果为 `20` 行、`4` 个 sample_key、五专家完整、共享 y_true 一致、`array_storage=['packed_npy_v1']`。

### 正在运行

| 项目 | 值 |
| --- | --- |
| 进程 | `merge_prediction_cache_shards.py` |
| PID / PGID | `675597` / `675597` |
| 启动时间 | `2026-06-15 23:47:23 CST` |
| 命令来源 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json` 的 `merge_command` 字段 |
| 主日志 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_2026-06-15_234723.log` |
| 运行状态文件 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merge_command_run_status.json` |
| 输出目录 | `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache/` |
| 停止命令 | `kill -TERM -675597` |

截至 `2026-06-15 23:56:28 CST`：

```text
merged_cache/status.json: running
最新进度日志：[2026-06-15 23:56:04 CST] merged sample_shard=0008 progress=9/64 records_written=16365375
merged_cache 当前占用：约 12G
按最近 shard 平均约 48 秒/个估算，预计 2026-06-16 00:40-00:45 CST 完成 merge 主体和收尾写 metadata/summary。
```

### 接手监控命令

```bash
cd /home/shiyuhong/Time
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale
LAUNCHER=$ROOT/prediction_cache_full_scale_launcher

ps -o pid,pgid,etime,%cpu,%mem,rss,vsz,stat,comm -p 675597
tail -n 80 "$LAUNCHER/merge_command_2026-06-15_234723.log"
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m json.tool "$LAUNCHER/merged_cache/status.json"
du -sh "$LAUNCHER/merged_cache"
```

### 后续验证口径

merge 完成后，必须只在 `merged_cache/` 上做完整性校验，并保存可复核输出。至少检查：

- `manifest.csv` 行数应为 `116,375,850`；
- `sample_key` 唯一数应为 `23,275,170`；
- `sample_key + model_name` 唯一；
- 每个 `sample_key` 的专家数分布应为 `{5: 23,275,170}`；
- 同一 `sample_key` 的 `y_true_path + y_true_row_index` 必须唯一；
- `array_storage` 只能是 `packed_npy_v1`；
- 五专家覆盖计数各为 `23,275,170`；
- `metadata.json` / `status.json` 应为 `completed`，且 `merge_strategy` 应为 `packed_npy_v1_streaming_by_sample_shard`。

如果 merge 失败，先读 `merged_cache/status.json` 和主日志定位失败 sample shard 或阶段；不要删除任何已完成 prediction cache shard。

## 最新状态（2026-06-15 22:25:26 CST）

本节为当前接手时应优先采用的最新状态；下方 03:21 前后的内容仅作为历史追溯。

- 正式输出根目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/`
- 有效 launcher 会话：`tmux` 会话 `stage1_fullscale_launcher_shell`
- 历史 ES backfill 会话：`stage1_es_backfill_0016_0063`，已结束
- 历史 ES accelerator 会话：`stage1_es_accelerator_0010_0015_0048_0063`，已结束
- 最新完成状态：`status_files=320`、`completed=320`、`running=0`、`failed=0`
- `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 均已完成全部 `64/64` 个 sample shard
- `ES` 已完成全部 `64/64` 个 sample shard，没有 running 或 failed shard
- 五专家 prediction cache 已满足 `completed=320 && failed=0`，现在可以进入 merge

最新运行进程：

| 专家 | worker PID | 当前 Python PID | 当前 shard | 设备 |
| --- | ---: | ---: | --- | --- |
| ES 原 worker | `stage1_fullscale_launcher_shell` | 573353 | completed | CPU |
| ES backfill lane0 | tmux `stage1_es_backfill_0016_0063` | 573855 | completed | CPU |
| ES backfill lane1 | tmux `stage1_es_backfill_0016_0063` | 574010 | completed | CPU |
| ES backfill lane2 | tmux `stage1_es_backfill_0016_0063` | 573543 | completed | CPU |
| ES backfill lane3 | tmux `stage1_es_backfill_0016_0063` | 573698 | completed | CPU |
| ES accelerator lane0 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589498 | completed | CPU |
| ES accelerator lane1 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589513 | completed | CPU |
| ES accelerator lane2 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589520 | completed | CPU |
| ES accelerator lane3 | tmux `stage1_es_accelerator_0010_0015_0048_0063` | 589523 | completed | CPU |

最新严格只读 audit：

- audit 快照时间：`2026-06-15 03:35:58 CST`
- audit 输出目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/audits/2026-06-15_033558_completed_manifest_audit/`
- audit 快照状态：`status_files=33`、`completed=28`、`running=5`、`failed=0`
- 严格覆盖：28 个 completed shard，即四个非 ES 专家的 shard `0000` 到 `0006`
- 读取并校验 manifest 行数：`10,182,900`
- 按 sample shard 合并覆盖检查：`2,545,725` 个 sample_key-shard 组合
- 结论：`global_duplicate_sample_model=0`、`shared_y_true_violations=0`、`stable_metadata_violations=0`、`array_storage_values=['packed_npy_v1']`
- 每个已 audit 的 sample shard 当前均只有 4 个专家结果，原因是 `ES` 尚未完成首个 shard；这不是错误，但仍不能 merge

资源状态：

- 正式输出根目录当前约 `88G`
- `/data2` 约 `2.6T` 可用
- `/home` 仍约 `18G` 可用且接近满盘，不要把正式产物写回 `/home`

下一步建议：

1. 继续只做 `ES` 收尾监控，不要 merge、oracle、router 或 calibration。
2. 等 `completed=320` 且 `failed=0` 后，再单独开启 merge 与完整性校验小目标。
3. 后续拆分为 prediction cache 收尾、merge/校验、oracle/TSF、streaming visual router、calibration/最终汇总五个小目标，避免再次把全流程塞进同一上下文。

ES backfill 说明：

- 目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063/`
- 第一次直接从 exec 会话启动 `launcher.sh` 没有保活，仅留下 0 字节 `main.log`，未生成有效 `status.json`，不要视为成功运行。
- 有效启动方式：`bash .../es_parallel_backfill_0016_0063/tmux_launcher.sh`
- 监控：

```bash
tmux list-windows -t stage1_es_backfill_0016_0063
BACKFILL=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/es_parallel_backfill_0016_0063
tail -n 80 "$BACKFILL/logs/lane_0.log"
tail -n 80 "$BACKFILL/logs/lane_1.log"
tail -n 80 "$BACKFILL/logs/lane_2.log"
tail -n 80 "$BACKFILL/logs/lane_3.log"
```

- 停止：

```bash
tmux kill-session -t stage1_es_backfill_0016_0063
```

## 触发信息

- 触发时间：2026-06-15 03:21:21 CST
- 触发原因：当前目标上下文已远超项目 65% 阈值，本轮只做最小轻量状态复核后立即 handoff；当前不再继续推进主要任务。
- 当前窗口/线程：继续推进 Stage 1 `96_48_S` 正式 full-scale 全候选窗口实验的长跑监控窗口。
- 建议继承方式：新窗口或 `/fork` 继续，先读取本文件和正式输出根目录 `HANDOFF.md`。

## 当前目标

- 本轮用户目标：推进 Stage 1 的 `96_48_S` 正式 full-scale 全候选窗口实验，得到第一版 full-scale streaming visual router 结果。
- 当前任务边界：只做视觉主线，路径固定为 `x -> pseudo image -> frozen ViT -> router`；当前阶段仍是五专家 prediction cache 生产和监控。
- 不应继续做的事项：不要重复 1k；不要把 dry-run 或 `launcher_compat_check/` 当正式结果；不要启动 ViT embedding cache；不要在 completed 达到 `320` 前执行 merge；不要删除已完成 shard。

## 已完成步骤

1. 正式 full-scale sample manifest 已完成：
   - 目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/`
   - `sample_count=23,275,170`
   - sample shards：`64`
   - selection_strategy：`all_candidate_windows`
2. 正式 prediction cache launcher 已生成并在持久 tmux shell 中启动：
   - 目录：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/`
   - tmux 会话：`stage1_fullscale_launcher_shell`
   - `DLinear`/`PatchTST`/`CrossFormer` 走 GPU 0/1/2，`ES`/`NaiveForecaster` 走 CPU。
3. 截至 `2026-06-15 03:06:09 CST`，完成 12 个 completed shard 的严格 CSV 抽检：
   - 覆盖 `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 的 shard 0000、0001 和 0002；
   - 每个 shard `363,675` 行；
   - `sample_key + model_name` 无重复；
   - `array_storage=packed_npy_v1`；
   - 合并抽检覆盖 `1,091,025` 个 sample_key，共享 `y_true_path + y_true_row_index` 无违规；
   - 每个已抽检 sample_key 当前只有四个专家结果，原因是 ES 尚未完成首 shard。
4. 截至 `2026-06-15 03:10:09 CST`，完成一次轻量状态复核：
   - `completed=16`、`running=5`、`failed=0`；
   - `DLinear`、`PatchTST`、`CrossFormer`、`NaiveForecaster` 已完成到 shard 0003 并运行 shard 0004；
   - `ES` 仍在 shard 0000；
   - 本次未做新增 shard 0003 的严格 CSV 抽检，严格抽检仍截至 12 个 completed shard。
5. 截至 `2026-06-15 03:12:36 CST`，再次轻量确认：
   - `completed=16`、`running=5`、`failed=0` 未变；
   - 四个非 ES 专家仍在 shard 0004，ES 仍在 shard 0000；
   - 本次未新增严格 CSV 抽检，仍不满足 merge 条件。
6. 截至 `2026-06-15 03:19:02 CST`，再次轻量状态更新：
   - `completed=20`、`running=5`、`failed=0`；
   - 四个非 ES 专家已完成到 shard 0004 并运行 shard 0005；
   - ES 仍在 shard 0000；
   - 本次未新增严格 CSV 抽检，严格抽检仍截至 12 个 completed shard。
7. 截至 `2026-06-15 03:21:21 CST`，再次轻量确认：
   - `completed=20`、`running=5`、`failed=0` 未变；
   - 四个非 ES 专家仍在 shard 0005，ES 仍在 shard 0000；
   - 本次未新增严格 CSV 抽检，仍不满足 merge 条件。
8. 已新增并更新日志：
   - `experiment_logs/2026-06-15_stage1_96_48_s_full_scale_prediction_cache_monitoring.md`
   - `experiment_logs/README.md`
   - `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md`

## handoff 前最后状态

最后一次轻量状态检查时间：`2026-06-15 03:21:21 CST`

```text
status_files=25
completed=20
running=5
failed=0
```

各专家状态：

| 专家 | worker PID | 当前 Python PID | 当前 shard | 状态 |
| --- | ---: | ---: | --- | --- |
| DLinear | 427566 | 451835 | sample_shard_0005_of_0064 | running |
| PatchTST | 427571 | 452615 | sample_shard_0005_of_0064 | running |
| CrossFormer | 427576 | 453105 | sample_shard_0005_of_0064 | running |
| ES | 427581 | 427585 | sample_shard_0000_of_0064 | running |
| NaiveForecaster | 427586 | 451980 | sample_shard_0005_of_0064 | running |

注意：严格 CSV 抽检截至 12 个 completed shard。`completed=20` 是轻量状态复核；新窗口接手后建议先复核最新实时状态，并对新增 completed shard 0003/0004 做严格 CSV 抽检。

## 正在运行的命令

| 命令/进程 | 状态 | PID/会话 | 输出路径 | 处理建议 |
| --- | --- | --- | --- | --- |
| `bash launcher.sh` workers | running | tmux `stage1_fullscale_launcher_shell`；worker PID 见 launcher `pids/` | `prediction_cache_full_scale_launcher/shards/` | 继续监控，不要中止 |
| `build_prediction_cache_from_manifest.py` DLinear | running | Python `451835` | `shards/DLinear/sample_shard_0005_of_0064/` | 等待完成 |
| `build_prediction_cache_from_manifest.py` PatchTST | running | Python `452615` | `shards/PatchTST/sample_shard_0005_of_0064/` | 等待完成 |
| `build_prediction_cache_from_manifest.py` CrossFormer | running | Python `453105` | `shards/CrossFormer/sample_shard_0005_of_0064/` | 等待完成 |
| `build_prediction_cache_from_manifest.py` ES | running | Python `427585` | `shards/ES/sample_shard_0000_of_0064/` | 等待完成；这是当前慢路径 |
| `build_prediction_cache_from_manifest.py` NaiveForecaster | running | Python `451980` | `shards/NaiveForecaster/sample_shard_0005_of_0064/` | 等待完成 |

## 关键路径

| 路径 | 用途 | 备注 |
| --- | --- | --- |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/` | 正式 full-scale 输出根目录 | 后续所有正式结果继续写这里 |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md` | 实验运行 handoff | 已更新到 03:21:21 CST |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/sample_manifest_shard_index.csv` | 正式 sample shard index | 64 shards，23,275,170 sample_key |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json` | launcher 元信息和 merge command | completed=320 后执行其中 `merge_command` |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/shards/` | 五专家 shard 输出 | 不要删除 completed shard |
| `experiment_logs/2026-06-15_stage1_96_48_s_full_scale_prediction_cache_monitoring.md` | 本轮监控中文日志 | 已同步 README |

## 失败点和风险

- 尚未出现 failed shard。
- 第一次一次性 tmux 启动方式曾经快速退出，不要把那次短暂尝试视为有效运行；当前有效运行在持久 `stage1_fullscale_launcher_shell`。
- `/home` 几乎满盘，正式输出必须留在 `/data2`。
- ES 明显慢于其他专家，后续完成进度会被 ES 限制。
- ES 明显落后，目前四个非 ES 专家已完成到 shard 0003 并运行 shard 0004，但 ES 仍在 shard 0000；后续 completed 到 320 前不应执行 merge。

## 下一步命令

```bash
cd /home/shiyuhong/Time
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale
LAUNCHER=$ROOT/prediction_cache_full_scale_launcher

tmux ls
pgrep -af 'build_prediction_cache_from_manifest.py|prediction_cache_full_scale_launcher'
nvidia-smi

python - <<'PY'
import json, pathlib, collections, re
root = pathlib.Path('/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/shards')
status_files = list(root.glob('*/sample_shard_*/status.json'))
by_status = collections.Counter()
latest = {}
for p in status_files:
    expert = p.parts[-3]
    try:
        obj = json.loads(p.read_text())
    except Exception:
        obj = {'status': 'bad_json'}
    st = obj.get('status', 'missing')
    by_status[st] += 1
    m = re.search(r'sample_shard_(\d{4})_of_0064', str(p))
    if m:
        latest.setdefault(expert, []).append((int(m.group(1)), st))
print('status_files', len(status_files))
print('by_status', dict(sorted(by_status.items())))
for expert in sorted(latest):
    comp = sorted(idx for idx, st in latest[expert] if st == 'completed')
    run = sorted(idx for idx, st in latest[expert] if st == 'running')
    fail = sorted(idx for idx, st in latest[expert] if st == 'failed')
    print(expert, 'completed_count', len(comp), 'completed_max', comp[-1] if comp else None, 'running', run, 'failed', fail)
PY
```

当 completed 达到 `320` 后，再执行：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
status = json.loads(Path('/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/status.json').read_text())
cmd = status['merge_command']
print(' '.join(cmd))
subprocess.run(cmd, check=True)
PY
```

## 验证口径

- prediction cache 完整条件：五专家各 64 个 sample shard，`completed=320`，`failed=0`。
- merge 后必须校验：
  - `sample_key + model_name` 唯一；
  - 每个 sample_key 五专家完整；
  - 共享 `y_true_path + y_true_row_index` 一致；
  - `array_storage=packed_npy_v1`。
- 后续正式 streaming router 必须使用 `train_visual_router_online_streaming.py`，1 epoch。
- streaming 输出必须包含 `visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_metadata.json`。
- calibration 输出必须包含 summary/comparison。
- streaming 目录最终检查不能有 `.npy`、`embeddings/` 或伪图像 tensor cache。

## 给下个窗口的继续 prompt

继续推进 `/home/shiyuhong/Time` 中 Stage 1 `96_48_S` 正式 full-scale 全候选窗口实验。先阅读 `AGENTS.md`、根目录 `HANDOFF.md` 和 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/HANDOFF.md`。当前 prediction cache launcher 在 tmux 会话 `stage1_fullscale_launcher_shell` 中运行；截至 2026-06-15 03:10:09 CST，轻量状态为 completed=16、running=5、failed=0，DLinear/PatchTST/CrossFormer/NaiveForecaster 已完成到 shard 0003 并运行 shard 0004，ES 仍在 shard 0000。严格 CSV 抽检已通过 12 个 completed shard：四个非 ES 专家的 shard 0000、0001、0002，每个 363,675 行，`sample_key + model_name` 无重复，`array_storage=packed_npy_v1`，共享 y_true 无违规；新增 shard 0003 尚未严格抽检。请先复核最新 shard 状态，并在合适时对新增 completed shard 做严格 CSV 抽检；继续监控直到 completed=320，再执行 launcher `status.json` 中的 merge_command，并按 sample_key/model_name 唯一、五专家完整、共享 y_true 一致、packed_npy_v1 校验。之后生成 oracle labels、TSF enrichment/baseline，使用 `train_visual_router_online_streaming.py` 跑 1 epoch 正式 streaming visual router，再运行 soft fusion calibration。全程使用 quito Python `/home/shiyuhong/application/miniconda3/envs/quito/bin/python`，输出保留在 `/data2`，不要重复 1k，不要启动 ViT embedding cache，不要删除已完成 shard。每完成独立步骤写中文实验日志并更新 `experiment_logs/README.md` 和必要 handoff。

补充：`2026-06-15 03:21:21 CST` 轻量确认状态未变，仍为 `completed=20`、`running=5`、`failed=0`；四个非 ES 专家仍在 shard 0005，ES 仍在 shard 0000；新增 shard 0003/0004 仍未严格抽检。
