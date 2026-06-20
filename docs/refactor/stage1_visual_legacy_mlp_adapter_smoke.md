# Stage 1 P14f Visual Legacy MLP Adapter Smoke

创建日期：2026-06-20

## 1. 目标

P14f 新增 smoke-only Visual legacy MLP thin adapter 验证。它只证明 head-ready
`FeatureBatch.features` 可以经一个已加载的小型 torch MLP 输出 canonical
`RouterOutput`，并继续被 `EvaluationInputAdapter` 消费。

本阶段不是正式 Visual RouterHead adapter 迁移，不接正式 `VisualMLPRouter`，不加载真实
checkpoint，不抽真实 ViT provider，不改正式入口，不访问 `/data2`，不启动训练、pressure
或 full-scale。

## 2. Smoke 链路

新增脚本：

```text
tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py
```

链路为：

```text
P13b sample_manifest.csv ordered sample_keys
  -> P14b VisualMockFeatureProvider
  -> FeatureBatch(sample_keys, head-ready float32 features)
  -> P13b expert_predictions.json 数值参考
  -> ExpertBatch(sample_keys, model_columns, y_pred, y_true)
  -> smoke-only loaded torch MLP state_dict fixture
  -> SmokeOnlyLegacyMLPAdapter
  -> RouterOutput(sample_keys, model_columns, logits, weights, extra)
  -> EvaluationInputAdapter
  -> in-memory summary / rows
```

`expert_predictions.json` 只作为 small fixture 数值参考，不是正式 prediction backend
schema。正式专家预测路径仍应由 prediction backend / `ExpertProvider` / `ExpertBatch`
提供。

## 3. Adapter 边界

P14f 的 adapter 定义在 smoke 脚本内部，名称为 `SmokeOnlyLegacyMLPAdapter`。它不是正式
`time_router.models` 中的 Visual RouterHead adapter，也不代表正式入口已迁移。

adapter 只做：

- 接收已准备好的 `FeatureBatch`。
- 接收显式 `model_columns`。
- 接收 Runtime/test fixture 已加载好的 torch MLP。
- 在 `torch.inference_mode()` 下 forward。
- 对 logits 做 `softmax(dim=1)` 得到 weights。
- 输出 `RouterOutput`。
- 检查 head-ready features 为 float32、shape 对齐、logits/weights shape、有限值、
  权重非负与 row sum 近似为 1。

adapter 不做：

- 不读取 prediction cache、oracle/error、run_dir、status、metadata 或 `/data2`。
- 不读取 `ExpertBatch.y_pred/y_true`。
- 不读取 checkpoint path，不调用 `torch.load`。
- 不 fit/load scaler。
- 不决定全局 device/dtype/DataParallel。
- 不写 evaluation CSV、summary、prediction rows 或 canonical artifacts。

## 4. Runtime 责任

P14e 的边界在 P14f 中继续保持：

- scaler fit/checkpoint state 属于 Runtime/entrypoint。
- checkpoint loading、resume 和 signature 校验属于 Runtime/entrypoint。
- device/dtype/DataParallel 策略属于 Runtime/entrypoint。
- adapter 的输入是 head-ready float32 `FeatureBatch.features`；如果未来输入 raw ViT
  embedding，必须在 adapter 前增加显式 pre-head transform step。

P14f 的 state_dict fixture 是内存构造并用固定 seed 初始化的小型 MLP，只用于确定性 smoke。
它不读真实 checkpoint，也不改变正式 checkpoint schema。

## 5. 检查项

新增 smoke 覆盖：

- `RouterOutput.sample_keys == FeatureBatch.sample_keys`。
- `RouterOutput.model_columns == ExpertBatch.model_columns`。
- logits/weights shape 为 `[num_samples, num_models]`。
- logits/weights dtype 为 float32 且全为有限值。
- weights 每行非负且 row sum 接近 1。
- `RouterOutput.extra` 只记录 adapter/head lineage。
- `EvaluationInputAdapter.evaluate(...)` 可生成 summary 和 per-sample rows。
- EvaluationInput / per-sample rows 保持 manifest sample_key 顺序。
- adapter/evaluator 阶段 patch 文件 IO、`np.load`、`np.save` 和 `torch.load`，确认不读取
  checkpoint、prediction、oracle、run_dir 或 `/data2`。
- smoke 不写 canonical `run_dir`。

## 6. 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 7. 后续连接

P14f 完成后，P15a 可以进入 branch-specific small entrypoint decision。决策依据应包括
P13d/P13e/P14a/P14b/P14c/P14d/P14e/P14f 的结果，重点判断是否新增 Visual-specific /
TimeFuse-specific small entrypoint，而不是把 branch-specific feature/head 逻辑塞回通用
small canonical CLI。
