# Visual Router V2 Round 1 Feature Cache Handoff

## 最新状态（2026-06-21 01:34:47 CST）

### 当前结论：Round 1 sharded pilot feature cache 已完成并通过全量验收

本轮目标是为 Visual Router V2 Round 1 RevIN aux 与 pooling 消融构建 sharded pilot feature cache。builder 与共享 feature schema 已新增，128 样本 smoke、resume/skip 验证和正式 P2a 全量构建均已完成；正式输出已通过逐 shard 独立核验。

| 项目 | 值 |
| --- | --- |
| 当前 worktree | `/home/shiyuhong/Time-visual-router-v2` |
| 分支 | `exp/visual-router-v2-pilot` |
| 新增 builder | `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py` |
| 新增 feature helper | `visual_router_experiments/stage1_vali_test_router/visual_router_v2_features.py` |
| P0 sample set | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/` |
| P1 Round 0 | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/` |
| Visual checkpoint | `/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt` |
| Smoke 输出目录 | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_smoke/` |
| 正式输出目录 | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/` |
| 正式 PID/PGID | `2469769 / 2469769`，已退出 |
| 正式 Python 子进程 | `2469772`，已退出 |
| 启动方式 | `setsid bash command.sh > main.log 2>&1 < /dev/null &` |
| CUDA 策略 | `CUDA_VISIBLE_DEVICES=1,2,3`，`--device cuda --vit-data-parallel` |
| 当前状态 | `status=completed`，`phase=done`，`processed_count=200000`，`completed_shards=100`，`failed_reason=null` |
| 已写 shard | `pilot_train=75`、`pilot_selection=15`、`diagnostic_balanced=10`，合计 100 个 shard |
| 总缓存大小 | `546.1020946502686 MB` |
| pilot_test | 未生成，`features/pilot_test/` 不存在 |

## 已完成实现

1. `visual_router_v2_features.py`
   - 定义 `FEATURE_SCHEMA_VERSION=visual_router_v2_round1_feature_cache_v1`。
   - 定义 `AUX_FEATURE_COLUMNS = mean, log_std, min, max, range, clip_ratio`。
   - 支持 P0 CSV 校验：`sample_set`、`order_index` 连续、`sample_key` 唯一。
   - 支持从历史 `x` 计算 RevIN aux，不读取 future y、oracle error、expert prediction。
   - 支持 shard feature shape/dtype/finite 校验。
   - 支持 `.npz` tmp 文件 + atomic rename 写出。
   - 支持已有 shard 的 sample_key/order_index/shape/finite 校验后 skip。

2. `build_visual_router_v2_round1_features.py`
   - 从 checkpoint `embedding_metadata` 自动读取实际 Visual 口径：`variant_a_3view`、`revin_aux`、`hf_vit_0_5`、`fixed_candidates`、候选周期 `[2,3,4,5,6,8,10,12,16,24,32,48,64,96]`。
   - 默认只处理 `pilot_train`、`pilot_selection`、`diagnostic_balanced`，默认拒绝 `pilot_test`。
   - 每个 shard 写 `sample_key`、`order_index`、`cls_embedding`、`mean_patch_embedding`、`revin_aux`。
   - `mean_patch_embedding` 使用 `last_hidden_state[:, 1:, :].mean(dim=1)`，明确排除 CLS。
   - 不 fit scaler，不训练 router/head/encoder，不读取 prediction manifest，不保存 pseudo image tensor。
   - 写出 `status.json`、`round1_feature_metadata.json`、`round1_feature_manifest.csv`、`round1_feature_cache_size_summary.csv`、`round1_feature_summary.md` 和 latency CSV。

## 已完成验证

语法检查：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall \
  visual_router_experiments/stage1_vali_test_router/visual_router_v2_features.py \
  visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py
```

128 样本 smoke 命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py \
  --max-samples-per-set 128 \
  --shard-size 64 \
  --embedding-batch-size 8 \
  --device cpu \
  --dtype fp32 \
  --local-files-only \
  --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_smoke
```

smoke 结果：

```text
status=completed
sample_counts={"pilot_train":128,"pilot_selection":128,"diagnostic_balanced":128}
manifest_rows=6
total_samples=384
size_mb=1.6356325149536133
cls_embedding shape=(N,768)
mean_patch_embedding shape=(N,768)
revin_aux shape=(N,6)
all finite=true
dtype=float32
sample_key/order_index 与 P0 CSV 前 128 行完全一致
```

resume/skip 验证：

```text
同一 smoke 命令重复运行，不加 --overwrite。
round1_feature_metadata.json: skipped_shards=6, completed_shards=6, elapsed_sec≈0.81。
```

## 正式运行命令

正式命令写在：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/command.sh
```

内容：

```bash
cd /home/shiyuhong/Time-visual-router-v2
export CUDA_VISIBLE_DEVICES=1,2,3
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py \
  --sample-sets pilot_train pilot_selection diagnostic_balanced \
  --shard-size 2000 \
  --embedding-batch-size 48 \
  --device cuda \
  --local-files-only \
  --vit-data-parallel \
  --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features
```

## 最终验收命令

```bash
cat /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/status.json
cat /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/round1_feature_metadata.json
find /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/features -maxdepth 3 -type f -name 'shard_*.npz' | wc -l
```

最终核验结果：

```text
status=completed
sample_counts:
  pilot_train=150000
  pilot_selection=30000
  diagnostic_balanced=20000
completed_shards=100
total sample_count=200000
pilot_test 不存在
manifest_rows=100
total_cache_size_mb=546.1020946502686
all shard dtype=float32, finite=true
cls_embedding shape=(N,768)
mean_patch_embedding shape=(N,768)
revin_aux shape=(N,6)
sample_key/order_index 与 P0 CSV 全量完全一致
```

建议核验脚本：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

out=Path('/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features')
p0=Path('/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples')
manifest=pd.read_csv(out/'round1_feature_manifest.csv')
size=pd.read_csv(out/'round1_feature_cache_size_summary.csv')
assert set(manifest['sample_set'])=={'pilot_train','pilot_selection','diagnostic_balanced'}
assert int(manifest['sample_count'].sum())==200000
assert len(manifest)==100
assert len(size)==len(manifest)
assert abs(size['file_size_mb'].sum()-manifest['file_size_mb'].sum())<1e-9
expected_counts={'pilot_train':150000,'pilot_selection':30000,'diagnostic_balanced':20000}
assert manifest.groupby('sample_set')['sample_count'].sum().astype(int).to_dict()==expected_counts
for sample_set, group in manifest.groupby('sample_set', sort=False):
    src=pd.read_csv(p0/f'{sample_set}_sample_keys.csv')
    keys=[]; orders=[]
    for row in group.sort_values('start_order_index').itertuples(index=False):
        with np.load(row.shard_path, allow_pickle=True) as data:
            shard_keys=[str(x) for x in data['sample_key'].tolist()]
            shard_orders=data['order_index'].astype(np.int64)
            cls=data['cls_embedding']; mean=data['mean_patch_embedding']; aux=data['revin_aux']
        assert cls.shape==(row.sample_count, 768), (sample_set, cls.shape)
        assert mean.shape==(row.sample_count, 768), (sample_set, mean.shape)
        assert aux.shape==(row.sample_count, 6), (sample_set, aux.shape)
        assert cls.dtype==np.float32 and mean.dtype==np.float32 and aux.dtype==np.float32
        assert np.isfinite(cls).all() and np.isfinite(mean).all() and np.isfinite(aux).all()
        assert np.array_equal(shard_orders, np.arange(row.start_order_index, row.end_order_index+1))
        keys.extend(shard_keys); orders.extend(shard_orders.tolist())
    assert keys==src['sample_key'].astype(str).tolist(), sample_set
    assert orders==src['order_index'].astype(int).tolist(), sample_set
status=json.loads((out/'status.json').read_text())
meta=json.loads((out/'round1_feature_metadata.json').read_text())
assert status['status']=='completed'
assert meta['feature_constraints']['mean_patch_excludes_cls'] is True
assert meta['feature_constraints']['read_prediction_manifest'] is False
assert meta['feature_constraints']['train_router_or_encoder'] is False
assert meta['default_excludes_pilot_test'] is True
assert not (out/'features'/'pilot_test').exists()
print({'status':'passed','manifest_rows':len(manifest),'total_samples':int(manifest['sample_count'].sum()),'size_mb':float(size['file_size_mb'].sum())})
PY
```

本次已执行上述核验，输出：

```text
{'status': 'passed', 'checked': [('diagnostic_balanced', 20000), ('pilot_selection', 30000), ('pilot_train', 150000)], 'manifest_rows': 100, 'total_samples': 200000, 'size_mb': 546.1020946502686}
```

## 文档与日志状态

已新增/更新：

```text
visual_router_experiments/stage1_vali_test_router/visual_router_v2_features.py
visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py
experiment_logs/2026-06-20_visual_router_v2_round1_feature_cache_builder.md
experiment_logs/README.md
WORKSPACE_STRUCTURE.md
HANDOFF.md
```

`experiment_logs/2026-06-20_visual_router_v2_round1_feature_cache_builder.md` 已追加 2026-06-21 01:34:47 CST 的最终查收记录；`experiment_logs/README.md` 中该行状态已改为“完成”；`WORKSPACE_STRUCTURE.md` 已把正式目录更新为 completed 产物。

## 禁止事项

```text
不要启动重复正式 P2a。
不要处理 pilot_test，除非用户明确要求 final_test_only。
不要读取 116M prediction manifest。
不要训练 router/head/encoder。
不要删除已完成 smoke 或正式 shard。
不要把本 cache 表述为 full-scale embedding cache。
```
