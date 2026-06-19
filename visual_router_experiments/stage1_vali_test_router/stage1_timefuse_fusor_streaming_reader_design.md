# Stage 1 TimeFuse Fusor Streaming Reader 设计

记录日期：2026-06-18 00:27:39 CST

## 目的

本文档固定 `96_48_S` full-scale TimeFuse-style fusor 的数据读取层契约。当前阶段只实现 reader 和 smoke test，不启动正式 fusor 训练。

## 输入输出契约

输入：

- feature shard：`timefuse_feature_cache_full_scale_launcher/shards/sample_shard_XXXX_of_0064/feature_cache.csv`，每行一个 `sample_key`，17 维 `timefuse_single_variable_meta_v1` 数值特征；
- oracle labels：`window_oracle_labels.parquet`，按 `metric=mae/mse` 过滤；
- prediction manifest：优先读取五专家同编号 shard manifest，也兼容单个 merged `manifest.csv`；
- packed arrays：通过 manifest 的 `array_storage`、`y_true_row_index`、`y_pred_row_index` 读取单行。

每个 reader batch 输出：

| 字段 | 形状/内容 | 说明 |
| --- | --- | --- |
| `sample_keys` | `List[str]` | 与 feature batch 原始顺序一致 |
| `features` | `[B, 17] float32` | fusor 输入，只来自历史窗口结构特征 |
| `labels` | batch 顺序 oracle rows | 用于 hard 选择评估或训练后诊断，不作为输入特征 |
| `y_pred` | `[B, 5, pred_len, channels] float32` | 五专家预测，专家顺序固定为 `MODEL_COLUMNS` |
| `y_true` | `[B, pred_len, channels] float32` | 当前 sample 共享真实未来 |
| `expert_errors` | `[B, 5] float32` | 从数组按指定 metric 复算，用于一致性检查和软目标构造 |

## 内存策略

1. feature CSV 按 batch/chunk 读取，不把 64 个 feature shard 拼成一个全量 DataFrame。
2. oracle parquet 只针对当前 feature shard 的 `sample_key` 和单一 metric 建 shard-local SQLite。
3. prediction manifest 只针对当前 feature shard 的 `sample_key` 建 shard-local SQLite；若传入 merged 116M manifest，也只把命中行写入 SQLite。
4. reader 运行时只持有当前 batch 和至多一个预取 batch，内存峰值由 `batch_size`、`prefetch_batches` 和当前 batch arrays 决定。
5. 不使用全量 manifest lookup、全量 labels join 或全量 feature-label-prediction merge。

## 并行策略

- 默认使用五专家同编号 shard manifest，避免为了单 shard smoke 扫描 52GB merged manifest。
- 当前 batch 内可用 `prediction_num_workers` 多线程读取 packed npy row；线程只处理当前 batch 的 sample，不扩大到全量 shard。
- `prefetch_batches` 当前限制为 0 或 1。预取线程可与主线程 feature CSV 读取重叠；SQLite 查询使用连接锁串行化，数组读取仍可并行。
- GPU 不参与 reader smoke；后续若训练 fusor 并使用 GPU，应按任务要求限制 `CUDA_VISIBLE_DEVICES=2,3`。

## 已验证

2026-06-18 00:27 CST 使用真实 full-scale shard 运行 smoke：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py \
  --max-rows 16 \
  --batch-size 8 \
  --smoke-batches 2 \
  --prediction-num-workers 2 \
  --prefetch-batches 1
```

输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-18_002728_217172_stage1_timefuse_fusor_streaming_reader_smoke/
```

验证结果：

- oracle SQLite index：16 条；
- prediction SQLite index：80 条，即 16 个 sample_key × 5 专家；
- batch 1/2 均输出 `feature_shape=[8,17]`、`y_pred_shape=[8,5,48,1]`、`y_true_shape=[8,48,1]`、`expert_errors_shape=[8,5]`；
- DLinear 数组复算 MAE 与 reader `expert_errors` 的最大差异为 `0.0`；
- 未启动训练，未使用 GPU。

## 下一步

1. 在该 reader 上接入 streaming StandardScaler 和单层 `TimeFuseFusor` 训练循环。
2. 先做单 shard train/eval smoke，再扩展为 64 shard launcher。
3. 正式训练前记录输出目录、PID、status、停止命令和恢复策略。
