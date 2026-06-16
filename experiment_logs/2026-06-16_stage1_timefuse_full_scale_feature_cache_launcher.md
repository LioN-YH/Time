# Stage 1 `96_48_S` Full-Scale TimeFuse Feature Cache 启动

日志日期：2026-06-16 00:51:00 CST

## 目的

为 Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline 预计算 TimeFuse-derived 单变量元特征 cache。该特征 cache 只依赖 sample manifest 和历史窗口 `x`，不需要等待 prediction cache merge、oracle labels 或 TSF enrichment。

## 背景

已有 pilot `pilot/build_structure_feature_cache_pilot.py` 能生成 17 维 TimeFuse-derived 单变量元特征，但它依赖 oracle label 文件提供样本清单，并且一次性攒结果，不适合 23,275,170 个 full-scale sample_key。full-scale sample manifest shard index 已存在：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/sample_manifest_shard_index.csv
```

该 index 指向 64 个 sample manifest shard，每个 shard 约 363,674 到 363,675 个 sample_key，字段包含 `sample_key/config_name/split/dataset_name/item_id/channel_id/window_index/history_length/pred_length` 等，足够重新加载 Quito 历史窗口 `x` 并提取元特征。

## 操作

1. 阅读项目规范、TimeFuse feature pilot、full-scale prediction cache builder/launcher、sample manifest builder、Stage 1 README、cache contract、TimeFuse readiness 日志和原始 `TimeFuse/meta_feature.py`。
2. 新增正式 builder：
   - `visual_router_experiments/stage1_vali_test_router/build_timefuse_feature_cache_from_manifest.py`
   - 输入单个 sample manifest shard；
   - 输出 `feature_cache.csv`、`metadata.json`、`status.json`、`main.log`、`latency_summary.csv`；
   - 保留 17 维 TimeFuse-derived 单变量特征定义；
   - 只访问 batch 的 `x`，不访问未来 `y`、专家预测或 oracle label；
   - 支持 `--resume`，completed shard 直接跳过，未完成 shard 保留完整 item 组后续跑。
3. 新增正式 launcher：
   - `visual_router_experiments/stage1_vali_test_router/launch_timefuse_feature_cache_full_scale.py`
   - 输入 `sample_manifest_shard_index.csv`；
   - 按 lane 生成 `lane_scripts/lane_*.sh`，用 `setsid bash ... < /dev/null` 后台保活；
   - 每个 lane 顺序处理若干 sample shard；
   - 根目录写 `launcher.sh`、`launch_plan.md`、`status.json`、`metadata.json`、`main.log`、`pids/`、`logs/`。
4. 在 Quito conda 环境执行语法检查：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
  visual_router_experiments/stage1_vali_test_router/build_timefuse_feature_cache_from_manifest.py \
  visual_router_experiments/stage1_vali_test_router/launch_timefuse_feature_cache_full_scale.py
```

5. 执行 builder smoke：
   - `timefuse_feature_cache_smoke_max64/`：64 行，`--resume` 二次运行能跳过 completed shard；
   - `timefuse_feature_cache_smoke_max1024/`：1024 行，特征全有限，无重复 sample_key，内部吞吐约 74.14 rows/s。
6. 执行 launcher smoke：
   - `timefuse_feature_cache_launcher_smoke/`；
   - 2 个 sample shard，每 shard `--max-samples 32`；
   - 第一次发现短命 shell 后台 lane 不稳定，随后将 launcher 改为 `setsid` lane script；
   - 复测后 2 个 shard 均 `completed`，各 32 行，无重复 sample_key，17 维特征全有限。
7. 启动正式 full-scale 后台任务：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/launch_timefuse_feature_cache_full_scale.py \
  --sample-manifest-shard-index-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/sample_manifest_shard_index.csv \
  --output-dir /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher \
  --lane-count 8 \
  --batch-size 512 \
  --num-workers 0 \
  --resume \
  --auto-start
```

## 结果

新增正式脚本均已通过语法检查，small smoke 和 launcher smoke 均通过。正式 full-scale 输出目录为：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/
```

截至 `2026-06-16 00:50:44 CST`，正式任务状态为：

```text
status_files=8
running=8
rows_done_or_written=255,080
declared_sample_count_seen=2,909,400
mean_builder_rows_per_second=145.90
sum_builder_rows_per_second=1,167.24
eta_hours_at_sum_rps=5.48
```

当前 8 个 lane PID/PGID：

```text
lane_00: PID/PGID 720659
lane_01: PID/PGID 720661
lane_02: PID/PGID 720662
lane_03: PID/PGID 720664
lane_04: PID/PGID 720665
lane_05: PID/PGID 720666
lane_06: PID/PGID 720667
lane_07: PID/PGID 720668
```

当前首批 builder PID：

```text
sample_shard_0000_of_0064: PID 720690, rows_written=31,885
sample_shard_0001_of_0064: PID 720686, rows_written=31,885
sample_shard_0002_of_0064: PID 720692, rows_written=31,885
sample_shard_0003_of_0064: PID 720687, rows_written=31,885
sample_shard_0004_of_0064: PID 720693, rows_written=31,885
sample_shard_0005_of_0064: PID 720691, rows_written=31,885
sample_shard_0006_of_0064: PID 720688, rows_written=31,885
sample_shard_0007_of_0064: PID 720694, rows_written=31,885
```

当前输出目录占用约 `104M`；`/data2` 仍约 `2.4T` 可用，`/home` 约 `64G` 可用但仍接近满盘。GPU 没有用于该任务，原因是特征提取由 `numpy/scipy/statsmodels` 的 ADF、ACF、AutoReg、periodogram 等小窗口 CPU 统计组成，搬到 GPU 没有实际收益，反而会增加数据搬运和调度开销。

监控命令：

```text
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher

/home/shiyuhong/application/miniconda3/envs/quito/bin/python -c "import collections,json,pathlib; root=pathlib.Path('$ROOT/shards'); files=list(root.glob('sample_shard_*/status.json')); c=collections.Counter(); rows=0
for p in files:
    s=json.loads(p.read_text()); c[s.get('status','unknown')]+=1; rows+=int(s.get('rows_written', s.get('sample_count',0)) or 0)
print({'status_files':len(files), **dict(c), 'rows_seen': rows})"

tail -n 80 "$ROOT/logs/lane_00.log"
tail -n 80 "$ROOT/shards/sample_shard_0000_of_0064/main.log"
```

停止命令：

```text
ROOT=/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher
for p in "$ROOT"/pids/*.pid; do
  kill -TERM -- -$(cat "$p") 2>/dev/null || kill -TERM $(cat "$p") 2>/dev/null || true
done
```

恢复命令：

```text
bash /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/launcher.sh
```

恢复时 launcher 会跳过 `status=completed` 的 shard；单个失败 shard 也可以重跑 `status.json` 中对应 task 的 builder 命令，builder 的 `--resume` 会保留完整 item 组。

## 结论

正式 full-scale TimeFuse-derived 单变量 feature cache 预计算已经实现、smoke 验证通过，并已按 8 lane CPU 后台启动。该任务与 oracle label 和专家 prediction cache 解耦，只使用历史 `x`，当前没有启动 full-scale TimeFuse fusor 训练。

## 下一步方案

1. 持续监控 `timefuse_feature_cache_full_scale_launcher/shards/*/status.json`，直到 64 个 sample shard 全部 `completed`。
2. 完成后做完整性校验：
   - 64 个 `feature_cache.csv` 行数总和应为 `23,275,170`；
   - `sample_key` 全局唯一；
   - 每个 shard 的 `feature_dim=17`；
   - 17 个特征列全部有限；
   - sample_key 与 `config_name/split/dataset_name/item_id/channel_id/window_index` 一致。
3. 不要在 feature cache 完成前启动 full-scale TimeFuse fusor 训练；即使 feature cache 完成，也还需要 prediction cache merged 结果和 oracle labels 齐全后再训练 fusor。
4. 将本次新增脚本和输出目录继续保留在 Stage 1 正式路径下，不回退到 pilot 入口。
