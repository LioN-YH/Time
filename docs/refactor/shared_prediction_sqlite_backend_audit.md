# Stage 1 P10a Shared Prediction SQLite Backend Audit

日志日期：2026-06-20 03:26:14 CST

## 1. 目标

本文审计 Visual Router 与 TimeFuse-style fusor 两条正式入口中的 prediction / oracle SQLite backend、index prepare、packed array batch loading 和 `ExpertProvider` 边界。

本阶段只做文档化审计，不抽共享代码，不修改正式入口行为，不修改 launcher，不新增 provider/head/runtime 代码。

## 2. Visual Router 当前 SQLite Prediction Path

`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 当前仍保留一条入口私有的 full-scale SQLite prediction path：

| 环节 | 当前实现 | 归属判断 |
| --- | --- | --- |
| required keys | `required_prediction_sample_keys(labels_df, args)` 根据 `router_mode`、`epochs`、`train_only`、`skip_soft_fusion` 推导 vali/test 需要的 sample_key | runtime / split strategy，不属于 provider |
| index prepare | `build_lightweight_prediction_index(...)` 分块扫描 full merged manifest，只写 required sample_keys 的五专家记录 | 可抽 shared index prepare helper |
| index artifact | `output_dir / "prediction_manifest_index.sqlite"` | runtime artifact lifecycle |
| query backend | `SQLitePredictionIndex.fetch_records(sample_keys)` 返回当前 batch 的 lookup-like records | 可抽 shared prepared backend |
| packed array loading | `load_prediction_tensors_from_lightweight_index(...)` 按 SQLite record 读取 packed row，返回 `y_pred/y_true/expert_errors` | 读取 `y_pred/y_true` 可共享；`expert_errors` 属于 Visual training loss |
| training loss | `fusion_huber_kl` 使用 `expert_errors` 构造 soft oracle，同时用 `y_pred/y_true` 计算 weighted fusion Huber | Visual Router branch-specific loss |
| eval soft lookup | test batch 仍通过 lookup-like records 支撑 raw soft fusion 与 adapter bypass 校验 | legacy evaluation bridge，短期保留 |

关键结论：Visual Router 的 SQLite path 不是 `PredictionCacheExpertProvider` 的直接等价物。它同时承担 required sample_key 推导、run_dir 内 SQLite artifact、index metadata、batch query、packed row index 读取、training loss 需要的 `expert_errors` 和 eval soft lookup 兼容。

## 3. TimeFuse-Style Fusor 当前 SQLite Path

`train_timefuse_fusor_streaming.py` 与 `stage1_timefuse_fusor_streaming_reader.py` 已经把 full-scale TimeFuse-style fusor 推到 shard-aware SQLite 读取：

| 环节 | 当前实现 | 归属判断 |
| --- | --- | --- |
| feature key source | `collect_feature_sample_keys(...)` 只读 feature shard 的 `sample_key` 列 | TimeFuse runtime / feature shard strategy |
| split 下推 | `_iter_feature_frames(...)` 在读取 oracle/prediction arrays 前按 split 过滤 feature CSV | TimeFuse reader 运行策略 |
| oracle index | `build_oracle_sqlite_index(...)` 从 oracle parquet 按 metric + sample_keys 构建 `oracle_index` | oracle supervision backend，不进入 deployable feature provider |
| prediction index | `build_prediction_sqlite_index(...)` 从一个或多个 prediction manifest 构建 shard-local `prediction_index` | 可抽 shared prediction index prepare helper |
| index reuse | `reusable_shard_indexes(...)` 按 oracle/prediction 表行数判断可复用 | runtime artifact lifecycle |
| query backend | `PredictionSQLiteIndex.fetch_records(sample_keys)` batch 查询当前 batch 五专家 records | 可抽 shared prepared backend |
| packed loading | `_load_array_grouped(...)` 对 batch 内 packed npy 按路径分组 mmap 切片 | 应与 `prediction_array_io.load_prediction_arrays_grouped(...)` 收敛 |
| scaler | `fit_scaler_streaming(...)` 只读 17 维 feature，不读取 oracle/prediction arrays | TimeFuse feature/training logic，不属于 prediction backend |
| training/eval | `Stage1TimeFuseFusorStreamingReader` 组装 feature、oracle labels、`y_pred/y_true/expert_errors` | 当前入口私有 reader；长期应拆出 provider 边界 |

关键结论：TimeFuse 路线已经证明 shard-local SQLite + batch query + grouped packed mmap 是可运行的 full-scale backend 形态，但它仍混合了 TimeFuse feature batch、oracle labels、training reader 和 prediction backend。P10a 不应把这一整套 reader 抽成 shared provider。

## 4. Shared Backend / Index Prepare 应承担的内容

未来可共享的是 prediction backend implementation，而不是完整 framework interface。最小 shared prediction SQLite backend 应只覆盖：

- manifest chunk scan：按 `chunk_read_rows` 分块读取 prediction manifest，不构建全量 Python lookup。
- required / target sample_keys：接收调用方已经确定的 sample_keys；不自行推导 split、train/eval 或 shard。
- SQLite 子集索引：写入 `(sample_key, model_name)` 主键、array path、metric、storage 和 row index。
- batch fetch records：`fetch_records(sample_keys)` 返回当前 batch 的 records，保持输入 batch 后续可按调用方顺序重排。
- packed row index lineage：保留 `y_true_row_index`、`y_pred_row_index`、`array_storage`、`manifest_dir` 或等价路径解析信息。
- grouped mmap loading 边界：可以提供按 path 分组读取 packed npy 的 helper，但只负责数组读取，不计算 loss。
- index metadata：记录 `expected_records`、`actual_records`、`chunk_read_rows`、`manifest_dir` / `prediction_manifest_paths`、`target_sample_keys`、created time。
- atomic replace / cleanup：使用临时 SQLite 文件构建，成功后原子替换；失败时不留下可被误复用的半成品。

这层可以服务 Visual Router 和 TimeFuse-style fusor，但调用方必须显式传入 sample_keys、index path、manifest path、chunk rows 和复用策略。

## 5. 不应放进 Shared Backend 的内容

以下逻辑不属于 shared prediction SQLite backend：

- Visual Quito history window、pseudo image tensor、ViT encoder、embedding scaler。
- TimeFuse 17 维 feature streaming、feature-only scaler、feature shard subset 生成。
- training loss、optimizer、scheduler、checkpoint、resume。
- `fusion_huber_kl` 的 soft oracle、TimeFuse SmoothL1 weighted fusion loss。
- `status.json`、`metadata.json`、CSV、Markdown summary、comparison、sample predictions 写出。
- launcher、Bash、`nohup`、tmux、monitor、GPU 绑定和 `/data2` 路径绑定。
- oracle label 生成、oracle baseline、upper-bound、diagnostic report。
- deployable `FeatureProvider` 的 test-time 动态特征。

这些职责应留给 runtime、branch-specific FeatureProvider、RouterHead、Evaluator、training loop 或 launcher。

## 6. ExpertProvider 边界

长期 `ExpertProvider` 的最小边界仍应是：

```python
load_batch(sample_keys) -> ExpertBatch
```

它应保证：

- `sample_keys` 输出顺序与输入顺序一致。
- `model_columns` 与 `y_pred` 专家维度一致。
- 读取并返回 `y_pred/y_true`。
- 校验 shape、finite、共享 `y_true` 和必要的 sample/model 完整性。
- 保留 row index lineage，例如 `row_index_metadata`。
- 可以消费 runtime 已准备好的 SQLite backend 或 reader backend。

它不应：

- 创建 `run_dir`。
- 写 status/metadata/checkpoint/CSV/summary。
- 推导 required sample_keys。
- 扫描全量 manifest 创建 index。
- 知道 Bash、launcher、`/data2` 或 GPU 资源策略。
- 计算 Visual Router `expert_errors` 或 TimeFuse loss。

因此，prepared backend 可以作为 provider 的输入依赖；provider 不能反向拥有 runtime index prepare。

## 7. Oracle 与 Prediction 边界

Prediction backend 可以进入 `ExpertProvider`，因为它提供专家预测和共享真实值。

Oracle backend 只能用于监督、诊断、baseline、upper-bound 和训练标签读取：

- Visual Router 的 `oracle_model/oracle_value` 当前来自 labels，用于 classification 或评估诊断。
- TimeFuse-style fusor 当前 reader 同时查询 oracle 和 prediction SQLite，是入口实现上的组合，不代表 oracle 应进入 deployable `FeatureProvider`。
- 长期接口应区分 `PredictionExpertProvider` 与 `OracleLabelProvider` / supervision reader。
- oracle label 不进入 test-time 可部署 feature，不进入 `TimeFuseFeatureCacheProvider` 或 Visual pseudo image / ViT 输入。

## 8. 后续路线

建议 P10 后续按风险从低到高推进：

1. **P10b：先抽 shared index prepare smoke helper**
   只抽 prediction SQLite prepare / fetch / metadata / atomic replace 的最小 helper，并用仓库内 packed fixture 或小 shard smoke 锁定 `sample_key + model_name` 完整性、row index lineage 和 grouped loading 行为。

   当前状态（2026-06-20）：已完成 smoke-only helper，详见 `docs/refactor/prediction_sqlite_backend.md`。该 helper 未接 Visual Router / TimeFuse 正式入口，也未修改 provider、reader、adapter、launcher、loss 或输出 schema。

2. **P10c：整理 launcher / run scripts 边界**
   在 provider 正式替换前，先把 run_dir、index artifact、status、metadata、resume、monitor 和停止命令边界文档化或轻量收束，避免 provider 接手 runtime 职责。

3. **Stage 1.5 / Stage 2：再评估 provider prepared backend**
   只有 shared backend smoke、Visual training/evaluation bypass、TimeFuse protocol chain 和小规模 pressure 都稳定后，才考虑让 `PredictionCacheExpertProvider` 接收 prepared SQLite backend 并进入正式入口。

明确推迟：

- 不直接用 `PredictionBatchReader` 替换 Visual Router SQLite path。
- 不把 TimeFuse `Stage1TimeFuseFusorStreamingReader` 整体上收为 shared provider。
- 不把 oracle backend 放进 deployable FeatureProvider。
- 不改正式 output schema、loss、optimizer、scaler、checkpoint/resume。

## 9. P10a 结论

Visual Router 和 TimeFuse-style fusor 已经收敛出同一个 full-scale prediction backend implementation 方向：调用方先确定 sample_keys，再构建 SQLite 子集索引，训练/评估 batch 只查询当前 sample_keys，并用 packed row index 分组 mmap 读取 `y_pred/y_true`。

但 shared backend 的边界必须窄：它是 prepared prediction SQLite backend / index prepare，不是 runtime、provider、feature、loss 或 launcher。下一步可以先抽 smoke-only shared index prepare helper；真正让 `PredictionCacheExpertProvider` 消费 prepared backend 应推迟到 Stage 1.5 / Stage 2。
