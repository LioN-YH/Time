# Visual Router Prediction Cache Provider Gap Audit

日志日期：2026-06-20 03:02:17 CST

## 1. 目标

P9e 审计 `PredictionCacheExpertProvider` / `PredictionBatchReader` 与 Visual
Router 正式入口中 `SQLitePredictionIndex`、`build_lightweight_prediction_index(...)`
和 `load_prediction_tensors_from_lightweight_index(...)` 的 full-scale 能力差距。

本阶段只做架构审计和迁移计划，不替换正式入口，不新增 full-scale provider
实现，不修改 provider、reader、Visual Router 训练入口或 evaluation adapter。

## 2. 当前 PredictionCacheExpertProvider 已具备的能力

`time_router.experts.PredictionCacheExpertProvider` 当前是 P6a 的 smoke-only
adapter。它内部复用 `PredictionBatchReader`，对外只暴露：

```python
expert_batch = provider.load_batch(sample_keys, verify_metrics=True)
```

已具备能力：

| 能力 | 当前状态 | 证据 |
| --- | --- | --- |
| 显式 sample_keys batch 输入 | 已支持，且拒绝空列表和重复 key | `PredictionCacheExpertProvider.load_batch(...)` |
| 保持 sample_key 顺序 | 已支持，输出 `ExpertBatch.sample_keys` 为调用方顺序 | provider smoke 与 reader `ordered_keys` |
| 固定五专家 `model_columns` | 已支持，默认 `DEFAULT_MODEL_COLUMNS`，输出 tuple | `PredictionBatchReader.model_columns` |
| 读取 `y_pred` / `y_true` | 已支持，输出 `ExpertBatch.y_pred/y_true` | reader `PredictionBatch` |
| `row_index_metadata` | 已支持，来自 reader `row_indices_by_sample_model` | `ExpertBatch.row_index_metadata` |
| `verify_metrics` | 已支持，沿用 reader 对 manifest MAE/MSE 的复算校验 | `load_batch(..., verify_metrics=True)` |
| `packed_npy_v1` / `per_sample_npy` | 已支持，由 `PredictionBatchReader` 和 `prediction_array_io` 负责 | `load_prediction_arrays_grouped(...)` |
| `ExpertBatch` 输出 | 已支持，是 provider 的唯一正式输出对象 | `time_router.protocols.ExpertBatch` |
| 不创建 run_dir / 不写 status metadata CSV | 已满足 | provider 只包装 reader 输出 |

这些能力足够支撑小规模 fixture、golden smoke、evaluation adapter smoke 和未来
batch-level adapter 对齐测试。

## 3. Visual Router 正式 SQLite prediction path 仍承担的能力

`train_visual_router_online_streaming.py` 的正式 full-scale prediction path 不只是
读取 batch arrays，还承担 runtime 索引准备、split 下推、训练 loss 输入和运行产物
记录等职责。

当前能力包括：

| 能力 | 当前归属 | 为什么不能直接替换为 provider |
| --- | --- | --- |
| `required_prediction_sample_keys(...)` | Visual Router runtime / split 逻辑 | 需要根据 `labels_df`、`router_mode`、`epochs`、`train_only`、`skip_soft_fusion` 推导 vali/test 所需 sample_key；provider 不应知道训练模式 |
| `build_lightweight_prediction_index(...)` | runtime index prepare | full-scale manifest 约 52GB，必须 chunk scan，只为 required sample_keys 建 SQLite 子集索引 |
| 大 manifest chunk scan | index prepare | 属于 run 初始化和资源控制，不应在 `load_batch(...)` 里隐式触发 |
| `prediction_manifest_index.sqlite` | run artifact | 写在 `output_dir` 下，属于 runtime artifact lifecycle |
| `index_metadata` | index prepare / runtime metadata | 包含 `target_sample_keys`、`expected_records`、`actual_records`、`chunk_read_rows`、`manifest_dir` |
| `SQLitePredictionIndex.fetch_records(...)` | batch query backend | 每个 embedding batch 只查询当前 sample_keys，避免千万级 Python dict 常驻内存 |
| `load_prediction_tensors_from_lightweight_index(...)` | training loss batch reader | 为 `fusion_huber_kl` 训练 loss 同时返回 `y_pred`、`y_true` 和 `expert_errors` |
| batch-level packed row index 单行读取 | SQLite path / array IO | 正式入口当前按 SQLite record 读取 packed row，控制 IO 与内存峰值 |
| eval raw soft fusion 所需 `soft_lookup` | evaluation legacy path | `add_soft_fusion_metrics(...)` 当前仍消费 lookup-like records，不消费 provider |
| checkpoint/resume 相关索引重建口径 | runtime | index 位于 run_dir，随本次 run 初始化，provider 不应决定复用/重建策略 |

其中 `expert_errors` 是关键差距：`PredictionCacheExpertProvider` 输出 `ExpertBatch`
只包含专家预测和共享真实值，不输出训练 loss 直接使用的 `[sample, expert]`
误差矩阵。该矩阵属于 Visual Router `fusion_huber_kl` loss 的分支逻辑，不应强行塞进
通用 `ExpertBatch` contract；若需要旁路校验，应由训练分支从 `ExpertBatch.y_pred/y_true`
显式复算。

## 4. 应属于 provider 的职责

未来若让 Visual Router 正式入口消费 ExpertProvider，provider 的安全边界应保持为：

- `load_batch(sample_keys) -> ExpertBatch`。
- 保证 `sample_keys` 输出顺序与输入顺序一致。
- 保证 `model_columns` 与专家维度顺序一致。
- 读取 `y_pred/y_true`，并校验 shape、有限值和共享 `y_true`。
- 保留 row index lineage，例如 `row_index_metadata`。
- 可以暴露轻量 `extra`，记录 provider 名称、array storage、reader/index backend 和当前 batch lineage。
- 不创建 `run_dir`。
- 不写 `status.json`、`metadata.json`、CSV、summary 或 checkpoint。
- 不决定 split、训练模式、loss、evaluation 输出 schema 或 launcher 行为。

## 5. 不应属于 provider 的职责

以下能力应留在 runtime、SplitStrategy、index prepare helper 或 launcher/report 层：

- 从 `labels_df` 和 `router_mode` 推导 `required_prediction_sample_keys(...)`。
- 选择 stream shard、config、vali/test split 和样本限制。
- 创建 `output_dir` / `run_dir`。
- 决定 SQLite index 路径，例如 `output_dir / "prediction_manifest_index.sqlite"`。
- 写 index metadata、run metadata、status、summary、comparison、CSV。
- 管理 run artifact lifecycle、checkpoint/resume、清理旧输出和恢复策略。
- full-scale preflight、磁盘检查、GPU/CPU 资源策略。
- `/data2` 路径绑定或 launcher 后台运行策略。
- 为 Visual Router 特定 loss 计算 `expert_errors`。
- 选择 `skip_soft_fusion`、`train_only`、`eval-only` 等训练/评估分支。

provider 可以消费 runtime 准备好的 backend，但不应反向承担这些 runtime 决策。

## 6. 三种迁移方案比较

### 6.1 方案 A：保留 Visual Router SQLitePredictionIndex，只在 batch 后包装 ExpertBatch

做法：

- 保留 `required_prediction_sample_keys(...)`。
- 保留 `build_lightweight_prediction_index(...)`。
- 保留 `SQLitePredictionIndex.fetch_records(...)`。
- 保留 `load_prediction_tensors_from_lightweight_index(...)`。
- 在 training loss 或 evaluation batch 拿到 legacy `y_pred/y_true` 后，构造
  `ExpertBatch` 做旁路校验。

优点：

- 风险最低，P9d evaluation bypass 已经证明该方向可行。
- 不改变 full-scale SQLite 子集索引、manifest chunk scan、packed row index 和当前 IO 峰值。
- 不改变 `fusion_huber_kl` loss、CSV、metadata、status、checkpoint schema。

缺点：

- 正式入口仍保留 legacy SQLite path，provider 只作为旁路 contract。
- 共享 provider 没有真正承载 full-scale backend。

适用结论：

- P9f 最适合先做 training loss `ExpertBatch` bypass check：从当前
  `load_prediction_tensors_from_lightweight_index(...)` 的输出包装 `ExpertBatch`，
  再从 `ExpertBatch.y_pred/y_true` 复算 `expert_errors`，与 legacy
  `expert_errors` 做内存一致性校验。该校验应默认关闭，不改变训练 loss。

### 6.2 方案 B：给 PredictionCacheExpertProvider 增加可选 prepared index / batch query 后端

做法：

- runtime 继续负责 required sample_keys 推导和 prepared index 创建。
- provider 构造时接收显式 prepared index 或 reader backend。
- provider 的 `load_batch(sample_keys)` 通过 backend batch query 返回 `ExpertBatch`。
- provider 不自己创建 run_dir，不扫描全量 manifest，不写 index metadata。

优点：

- 让正式入口逐步消费 `ExpertProvider` contract。
- 可以复用 Visual Router 当前 SQLite full-scale 经验，避免回退到高内存 lookup。
- index prepare 和 provider loading 职责可分开测试。

风险：

- 需要新增 backend interface，必须避免把 Visual Router runtime 细节塞进 provider。
- 需要 smoke、small pressure 和 full-scale preflight，证明不会增加内存/IO 峰值。
- 需要处理 `soft_lookup`、`expert_errors` 和 evaluation legacy helper 的过渡。

适用结论：

- 可作为 Stage 1.5 的正式迁移候选，但 P9e 不应实现。
- 实现前应先抽 shared prediction index prepare helper，明确 prepared backend 的最小
  public API。

### 6.3 方案 C：直接用 PredictionBatchReader 替换 Visual Router SQLite path

做法：

- 在正式入口中直接构造 `PredictionBatchReader(manifest_path=...)`。
- 每个训练/eval batch 调用 `reader.load(sample_keys)`。

风险：

- reader 当前按 manifest chunk scan 查找显式 sample_keys；若每个 embedding batch 都
  重新扫 52GB manifest，会造成极高 IO。
- 若改成一次加载全部 required sample_keys，又可能回到高内存 / 高 Python object 风险。
- reader 没有 run_dir 下 prepared SQLite artifact、index metadata、resume lifecycle 语义。
- 不能直接替代 `required_prediction_sample_keys(...)`、`soft_lookup` 和 `fusion_huber_kl`
  `expert_errors` 训练口径。

适用结论：

- 默认不建议直接做。
- 只适合 fixture、小规模 smoke 或已经由外部 backend 限制过 manifest 子集的场景。

## 7. P9f / P10 后续建议

建议顺序：

1. **P9f：training loss ExpertBatch bypass check**
   继续采用方案 A。默认关闭 flag，只在 `fusion_huber_kl` training batch 内把 legacy
   SQLite arrays 包装为 `ExpertBatch`，从 `ExpertBatch.y_pred/y_true` 复算
   `expert_errors` 并与 legacy `expert_errors` 比较；不改变训练 loss、optimizer 或输出。

2. **P10a：抽 shared prediction index prepare helper**
   把 `build_lightweight_prediction_index(...)` 的 index prepare 语义文档化或抽到共享位置前，
   先明确它属于 runtime/index prepare，不属于 provider。抽取时要保持 SQLite metadata、
   primary key、chunk scan、atomic replace 和 batch query 行为不变。

3. **P10b：launcher/run script 层整理**
   在 provider 替换前，先让 full-scale run 的 output_dir、index artifact、metadata、
   status、resume 和监控命令边界更清楚。这样 provider 后续只接收 prepared backend，
   不承担 launcher 职责。

4. **Stage 1.5 / Stage 2：再评估真正 provider 替换**
   只有当 evaluation bypass、training bypass、shared index prepare 和 small pressure
   都稳定后，才考虑让 Visual Router 正式入口直接调用 `ExpertProvider.load_batch(...)`。

推迟项：

- 推迟直接替换 `SQLitePredictionIndex`。
- 推迟把 `PredictionBatchReader` 直接迁入正式入口。
- 推迟将 `/data2`、run_dir、status 或 full-scale preflight 放入 provider。
- 推迟改 `fusion_huber_kl` loss、checkpoint/resume 或正式 CSV schema。

## 8. P9e 结论

`PredictionCacheExpertProvider` 当前已经具备 canonical `ExpertBatch` 包装能力，
但尚不具备 Visual Router 正式 full-scale SQLite path 的 runtime/index prepare 能力。
二者的差距不是简单 reader API 缺口，而是 runtime 职责边界：正式入口当前需要根据 split
和训练模式准备 SQLite 子集索引、记录 index artifact metadata、按 batch 查询 packed row
record，并为训练 loss 计算 `expert_errors`。

因此，短期应继续保留 Visual Router SQLitePredictionIndex，优先做 batch 后
`ExpertBatch` 旁路校验；中期再抽 shared prediction index prepare helper；长期才考虑
prepared backend 形式的 `PredictionCacheExpertProvider` full-scale 接入。
