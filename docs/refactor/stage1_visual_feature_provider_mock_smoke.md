# Stage 1 P14b Visual FeatureProvider Mock Smoke

创建日期：2026-06-20

## 1. 目标

P14b 新增 smoke-only Visual `FeatureProvider` mock，用仓库内小型 fixture 和 deterministic
encoder stub 验证 Visual-style provider 可以按 manifest ordered `sample_keys` 输出
`FeatureBatch`。

本阶段不是正式 ViT provider，不迁移 `train_visual_router_online_streaming.py`，不加载真实
Hugging Face ViT，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 2. 新增文件

- `time_router/features/visual_mock.py`
  - `DeterministicVisualEncoderStub`
  - `VisualMockFeatureProvider`
- `tests/fixtures/stage1_visual_feature_mock/README.md`
- `tests/fixtures/stage1_visual_feature_mock/history_windows.json`
- `tests/smoke/stage1_visual_feature_provider_mock_smoke.py`

## 3. Fixture 口径

smoke 继续使用 P13b real-derived manifest 作为 ordered sample_keys 来源：

```text
tests/fixtures/stage1_real_derived_small/sample_manifest.csv
```

Visual history fixture 为：

```text
tests/fixtures/stage1_visual_feature_mock/history_windows.json
```

该 JSON 只保存：

```text
sample_key -> history_window_x
```

它不包含 future `y`、`y_true`、oracle、expert error、prediction cache path、run_dir、
metadata、status、checkpoint 或 `/data2` 路径。

## 4. Provider Contract

`VisualMockFeatureProvider.load_batch(sample_keys)` 的最小 contract：

- 调用方必须显式传入非空 `sample_keys`。
- `sample_keys` 不允许重复。
- provider 按传入顺序读取内存 `history_windows`，不排序、不从 fixture 文件重读。
- deterministic encoder stub 输出 `[sample, 8]` 的 `numpy.float32` embedding。
- 返回 `FeatureBatch(sample_keys, features, feature_schema, extra)`。

`feature_schema` 记录：

- `feature_schema_name = visual_mock_history_encoder_v1`
- `feature_dim = 8`
- `history_source = stage1_visual_feature_mock_history_window_x`
- `pseudo_image.variant = mock_not_materialized`
- `encoder_stub.name = deterministic_visual_history_stats_stub_v1`
- `encoder_stub.loads_real_vit = False`
- `encoder_stub.uses_gpu = False`
- `encoder_stub.uses_huggingface_cache = False`
- `storage = batch_runtime_only_not_saved`

`extra` 只记录轻量 lineage：

- `provider_name`
- `source`
- `num_available_rows`

## 5. 明确不做

- 不接 Visual RouterHead。
- 不接 `EvaluationInputAdapter`。
- 不写 canonical `run_dir`。
- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不抽正式 ViT provider。
- 不加载 Hugging Face ViT。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual RouterHead adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 6. Smoke 边界检查

`tests/smoke/stage1_visual_feature_provider_mock_smoke.py` 在 provider 阶段 patch：

- `builtins.open`
- `Path.open`
- `Path.read_text`
- `np.load`

因此 manifest 和 history fixture 只能在 provider 外部读取。provider 初始化与
`load_batch(...)` 阶段若尝试读取任何文件都会失败，覆盖 prediction cache、oracle/error、
`y_true`、run_dir、metadata、status、checkpoint 和 `/data2` 边界。

smoke 同时检查 `experiment_logs/run_outputs/` 一层目录集合不变，证明 provider 不创建
canonical run_dir。

## 7. 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
```

回归 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 8. 后续连接

P14b 只证明 Visual-style `FeatureProvider -> FeatureBatch` 的最小 mock contract。P14c 可继续
做 Visual eval-only canonical bypass plan，规划 legacy SQLite batch arrays 如何与 Visual
`FeatureBatch`、Visual head 和 evaluator 对齐，但仍不替换正式入口、不改正式输出 schema。

