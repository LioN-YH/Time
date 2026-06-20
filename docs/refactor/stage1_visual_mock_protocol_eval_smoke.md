# Stage 1 P14d Visual Mock Protocol Eval Smoke

创建日期：2026-06-20

## 1. 目标

P14d 新增 Visual mock `FeatureBatch + mock RouterHead + EvaluationInputAdapter`
protocol smoke。该 smoke 用 P13b/P14b tiny fixture 和 P13b expert JSON 数值参考，
验证 Visual 分支可以在内存中从 `FeatureBatch + ExpertBatch` 走到
`RouterOutput + evaluation summary/rows`。

本阶段只做 smoke，不改正式入口，不加载真实 ViT，不接 legacy MLP，不写 canonical
`run_dir`，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 2. 新增文件

- `tests/smoke/stage1_visual_mock_protocol_eval_smoke.py`

本轮没有新增正式 `time_router.models` 文件。mock RouterHead 只定义在 smoke 内，避免把
smoke-only 逻辑误注册为正式 Visual RouterHead adapter。

## 3. 输入与连接边界

P14d 的输入来自三个仓库内 tiny fixture：

- `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`
  - 读取 ordered sample_keys。
  - manifest 行顺序是本 smoke 的唯一样本顺序来源。
- `tests/fixtures/stage1_visual_feature_mock/history_windows.json`
  - 由 P14b `VisualMockFeatureProvider` 消费。
  - 只包含 history window `x`，不包含 future `y`、`y_true`、prediction、oracle 或 run artifact。
- `tests/fixtures/stage1_real_derived_small/expert_predictions.json`
  - 只作为 P13b/P13d small fixture 数值参考，用于在 smoke 内构造最小 `ExpertBatch`。
  - 它不是正式 prediction backend schema；正式路径仍应由 prediction backend /
    `ExpertProvider` / `ExpertBatch` 提供。

内存链路为：

```text
P13b sample_manifest.csv ordered sample_keys
  -> P14b VisualMockFeatureProvider
  -> FeatureBatch(sample_keys, features)
  -> P13b expert_predictions.json reference
  -> ExpertBatch(sample_keys, model_columns, y_pred, y_true)
  -> smoke-only DeterministicVisualMockRouterHead
  -> RouterOutput(sample_keys, model_columns, logits, weights)
  -> EvaluationInputAdapter
  -> in-memory summary / per-sample rows
```

`FeatureBatch` 与 `ExpertBatch` 只通过 ordered `sample_keys` 对齐；`RouterOutput` 的
`model_columns` 必须等于 `ExpertBatch.model_columns`。

## 4. Mock RouterHead 口径

`DeterministicVisualMockRouterHead` 只存在于 smoke 文件内：

- 输入 `FeatureBatch.features`。
- 输出 `RouterOutput`。
- `sample_keys` 保持 `FeatureBatch.sample_keys`。
- `model_columns` 使用 `ExpertBatch.model_columns`。
- `weights` shape 为 `[num_samples, num_models]`。
- `weights` 每行 softmax 归一化，行和接近 1。
- 不读取 prediction cache、oracle/error、run_dir、checkpoint。
- 不代表正式 Visual MLP adapter，不加载真实 ViT。

当前 P13b small fixture 的 `expert_predictions.json` 是三专家数值参考；P14d 只验证该
small fixture 上的 protocol 连接，不声称覆盖正式五专家 full-scale 路径。

## 5. Evaluation 验证

smoke 调用 `EvaluationInputAdapter.evaluate(...)`，并检查：

- `EvaluationInput.sample_keys` 保持 manifest 顺序。
- `EvaluationInput.model_columns` 等于 `ExpertBatch.model_columns`。
- `EvaluationInput.y_pred/y_true` 原样引用 `ExpertBatch`，不复制、不回读。
- `EvaluationInput.weights` 原样引用 `RouterOutput.weights`。
- summary 中 `num_samples` 等于 manifest 行数。
- hard top-1 与 raw soft fusion 的 MAE/MSE 均可生成且为有限值。
- `selected_counts` 可生成，专家集合等于 `model_columns`，总数等于 sample_count。
- per-sample rows 数量等于 sample_count，sample_key 顺序保持 manifest 顺序。

## 6. 明确不做

- 不修改 `scripts/run_stage1_canonical_small.py`。
- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增正式 `VisualFeatureProvider`。
- 不抽真实 ViT provider。
- 不接 legacy `VisualMLPRouter`。
- 不新增正式 Visual RouterHead adapter。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不写 canonical `run_dir`。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 7. 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

截至本文件创建时，新增 P14d smoke 已单独通过；完整回归命令记录在对应中文实验日志中。

## 8. 后续连接

1. P14e 已完成：Visual eval-only legacy MLP adapter audit，见
   `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md`。已明确 legacy MLP eval-only
   adapter 的输入、输出、不做范围，以及 scaler/checkpoint/device/dtype/DataParallel
   归 Runtime/entrypoint 管理。
2. P14f：Visual legacy MLP adapter smoke，重点检查
   `FeatureBatch -> legacy MLP -> RouterOutput` 的 sample/model 保序、dtype/device 和
   checkpoint 边界，继续使用 tiny fixture，不接正式入口。
3. P15：根据 P13d/P13e/P14a/P14b/P14c/P14d/P14e/P14f 结果决定是否新增 branch-specific
   small entrypoint。
