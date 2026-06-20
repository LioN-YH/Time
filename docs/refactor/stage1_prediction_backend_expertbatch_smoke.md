# Stage 1 P13d Prediction Backend -> ExpertBatch Small Smoke

创建日期：2026-06-20

## 1. 目标

P13d 在 P13c 的 backend/provider connection audit 之后，新增一个只面向小型
fixture 的 prediction backend -> `ExpertBatch` smoke。该 smoke 使用 P13b
`tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 的行顺序作为 ordered
sample_keys，用 P13b `expert_predictions.json` 作为数值参考，在 tempfile 内构造小型
`packed_npy_v1` prediction cache manifest、数组和 SQLite backend，然后经
`PredictionBatchReader` / `PredictionCacheExpertProvider` 输出 `ExpertBatch`。

本阶段只验证小链路，不迁移正式入口，不访问 `/data2`，不启动训练、pressure 或
full-scale。

## 2. 实现范围

新增 smoke：

```text
tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
```

验证链路：

```text
P13b sample_manifest.csv ordered sample_keys
  + P13b expert_predictions.json 数值参考
  -> tempfile packed_npy_v1 prediction manifest / arrays
  -> shared prediction SQLite backend fetch records
  -> grouped prediction array loading
  -> PredictionBatchReader
  -> PredictionCacheExpertProvider
  -> ExpertBatch
```

检查项：

- `ExpertBatch.sample_keys` 与 P13b sample manifest 行顺序一致；
- `ExpertBatch.model_columns` 与 P13b `expert_predictions.json` 一致；
- `y_pred` shape 为 `[sample, expert, pred_len, channel]`；
- `y_true` shape 为 `[sample, pred_len, channel]`；
- `y_pred` / `y_true` 数值与 P13b JSON 参考一致；
- shared SQLite backend metadata 保留 target sample key 顺序、model columns、record count 和 missing report；
- SQLite fetch records 经 `records_to_ordered_rows(...)` 与 grouped array IO 可读回参考数组；
- `ExpertBatch.row_index_metadata` 存在；
- `ExpertBatch.extra` 记录 provider name、`array_storage=packed_npy_v1`、reader manifest path 和 schema 校验开关。

## 3. P13b JSON 的边界

P13b `expert_predictions.json` 仍只是 small fixture 的数值参考，不是正式 prediction
backend schema。P13d 不把该 JSON 升级为正式 cache schema，也不要求后续正式入口读取
该 JSON。

P13d 在 tempfile 内构造的 manifest 只是为了把 P13b 参考数组转换为 backend 可读形式；
正式 full-scale prediction cache 仍应使用 canonical prediction manifest schema、固定五专家
完整性、`packed_npy_v1` shard 和 Runtime/backend prepare 层的索引策略。

## 4. 非 Canonical Sample Key 处理

P13b real-derived fixture 的 sample_key 使用 schema-style `::` 字符串，不是
`PredictionCacheKey.as_string()` 生成的 canonical 双下划线格式。为避免重写 P13b fixture
或伪造 sample_key，P13d 给 `PredictionBatchReader` 和 `PredictionCacheExpertProvider`
增加显式参数：

```python
validate_manifest_schema=False
```

默认值仍为 `True`，正式 cache 和既有 smoke 默认继续走 strict canonical schema 校验。
P13d 只在临时 backend fixture 上关闭 strict schema，并保留最小校验：

- manifest 必须包含 reader 所需字段；
- `(sample_key, model_name)` 唯一；
- 每个 sample 覆盖调用方显式传入的 `model_columns`；
- 数组路径、`array_storage`、row index、MAE/MSE 仍由 reader/provider 和 grouped array IO 校验。

该开关是 smoke-only / bridge-only 能力，不代表正式 prediction cache 可以跳过 canonical schema。

## 5. 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 6. 验收命令

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
```

本阶段回归验收：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 7. 后续连接

P13e 可继续做 TimeFuse 17 维 `FeatureProvider` small smoke，验证 feature-only provider
按 P13b / future sample manifest 的 ordered sample_keys 输出 `FeatureBatch`。Visual history
window / pseudo image / frozen ViT provider 仍应先进入 P14a 插入点审计，不应在 P13d 后直接
接正式 Visual 入口。
