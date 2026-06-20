# Stage 1 P14f Visual legacy MLP adapter smoke

日志日期：2026-06-20 22:01:57 CST

## 目的

新增 Visual legacy MLP adapter smoke，验证 head-ready float32 `FeatureBatch.features`
可以经 smoke-only 小型 torch MLP / loaded state_dict fixture 输出 canonical
`RouterOutput`，并被 `EvaluationInputAdapter` 消费。

## 背景

P14e 已完成 Visual eval-only legacy MLP adapter audit，结论是 future adapter 的最小边界为：

```text
FeatureBatch(sample_keys, head-ready features)
+ model_columns
+ runtime-loaded MLP
-> RouterOutput(sample_keys, model_columns, logits, weights, extra)
```

scaler fit/checkpoint state、checkpoint loading、resume、device/dtype/DataParallel 仍归
Runtime/entrypoint 管理。P14f 只做 smoke，不接正式 `VisualMLPRouter`，不加载真实 checkpoint，
不抽真实 ViT provider，不改正式入口，不访问 `/data2`。

## 操作

1. 新增 `tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py`。
2. 在 smoke 内定义 `SmokeOnlyLegacyMLP` 和 `SmokeOnlyLegacyMLPAdapter`，不新增正式
   `time_router.models` adapter 文件。
3. 使用 P13b `sample_manifest.csv` 读取 ordered sample_keys。
4. 使用 P14b `VisualMockFeatureProvider` 生成 head-ready float32
   `FeatureBatch(features=(4, 8))`。
5. 使用 P13b `expert_predictions.json` 构造 `ExpertBatch` 数值参考，并在代码与文档中明确
   该 JSON 只是 small fixture 参考，不是正式 prediction backend schema。
6. 使用固定 seed 构造内存 state_dict fixture，加载到 smoke-only 小型 torch MLP。
7. 在 adapter/evaluator 阶段 patch 文件 IO、`np.load`、`np.save` 和 `torch.load`，确认不读取
   checkpoint、prediction、oracle、run_dir 或 `/data2`。
8. 新增 `docs/refactor/stage1_visual_legacy_mlp_adapter_smoke.md`。
9. 更新 `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md`、
   `docs/refactor/stage1_visual_eval_canonical_bypass_plan.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md`、
   `docs/refactor/stage1_refactor_roadmap.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

已新增 P14f smoke 和文档。smoke 覆盖以下检查：

- `RouterOutput.sample_keys == FeatureBatch.sample_keys`。
- `RouterOutput.model_columns == ExpertBatch.model_columns`。
- logits/weights shape 为 `[num_samples, num_models]`。
- logits/weights dtype 为 float32 且全为有限值。
- weights 每行非负且 row sum 接近 1。
- `RouterOutput.extra` 只记录 adapter/head lineage、checkpoint/scaler/device 责任归属。
- `EvaluationInputAdapter.evaluate(...)` 可生成 summary 和 per-sample rows。
- EvaluationInput 和 per-sample rows 保持 manifest sample_key 顺序。
- 不写 canonical `run_dir`。

实际验收结果：

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

以上命令均已通过。新增 smoke 输出 `hard_mae=0.125000000`、
`raw_soft_mae=0.079372019`，并确认 adapter/evaluator 阶段未调用文件 IO、`np.load`、
`np.save` 或 `torch.load`。

## 结论

P14f 将 P14e 的 audit 边界落实为最小 smoke：head-ready `FeatureBatch` 可以通过
smoke-only loaded torch MLP adapter 输出 `RouterOutput`，再进入现有 evaluation adapter。
该实现没有新增正式 Visual RouterHead adapter，也没有修改正式入口或 checkpoint/scaler schema。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续进入 P15a branch-specific small entrypoint decision。
