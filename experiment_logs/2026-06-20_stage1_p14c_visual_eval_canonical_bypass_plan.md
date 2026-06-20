# Stage 1 P14c Visual eval-only canonical bypass plan

日志日期：2026-06-20 20:08:06 CST

## 目的

冻结 Visual eval-only canonical bypass 方案，明确 legacy SQLite batch arrays、Visual
`FeatureBatch`、Visual head / legacy MLP、`RouterOutput`、`EvaluationInputAdapter` 和 future
Runtime artifact writer 的连接边界。

## 背景

P14a 已完成 Visual FeatureProvider insertion audit，确认 future Visual provider 的最小输出是
`FeatureBatch`。P14b 已完成 Visual mock provider smoke，证明 mock provider 可以按 P13b
manifest ordered sample_keys 输出 `FeatureBatch`。此前 P9d/P9f 已证明 Visual evaluation/training
legacy SQLite batch arrays 可以包装为 `ExpertBatch` 做旁路校验。

P14c 只做文档审计和迁移方案冻结，不改正式入口，不新增正式 provider/head 代码，不访问
`/data2`，不启动训练、pressure 或 full-scale。

## 操作

- 新增 `docs/refactor/stage1_visual_eval_canonical_bypass_plan.md`。
- 在 P14c 文档中冻结 future eval-only 链路：
  `SampleManifest ordered sample_keys -> VisualFeatureProvider / mock provider / legacy embedding path
  -> FeatureBatch -> legacy SQLite prediction arrays 或 PredictionCacheExpertProvider
  -> ExpertBatch -> Visual RouterHead / legacy MLP adapter -> RouterOutput
  -> EvaluationInputAdapter -> Evaluator summary/rows -> future Runtime artifact writer`。
- 明确 `ExpertBatch` 只提供 `sample_keys`、`model_columns`、`y_pred`、`y_true` 和轻量 lineage；
  不读取视觉 history、pseudo image、ViT feature、oracle/error 或 run_dir。
- 明确 `FeatureBatch` 可来自 P14b mock provider、future VisualFeatureProvider 或 legacy embedding
  path；只保存 router/head 所需视觉特征，不读取 prediction cache、oracle/error 或 run_dir。
- 明确 Visual RouterHead adapter 尚未抽取，后续可先做 mock head smoke 或 legacy MLP thin
  adapter；head 输出 `RouterOutput(sample_keys, model_columns, weights/logits/extra)`。
- 明确 `EvaluationInputAdapter` 消费 `ExpertBatch + RouterOutput`，Evaluator 产生内存
  summary/rows；Runtime artifact writer 后续才写 future canonical `evaluation/` 与
  `predictions/`。
- 同步更新 `docs/refactor/stage1_visual_feature_provider_insertion_audit.md`、
  `docs/refactor/stage1_visual_feature_provider_mock_smoke.md`、
  `docs/refactor/stage1_entrypoint_migration_plan.md`、
  `docs/refactor/stage1_refactor_roadmap.md` 和 `WORKSPACE_STRUCTURE.md`。
- 更新 `experiment_logs/README.md` 总览追踪表。
- 运行 P14c 指定 smoke 与 compileall 验证。

## 结果

- P14c 文档已明确本轮不修改 `train_visual_router_online_streaming.py`、
  `train_timefuse_fusor_streaming.py` 和 `launch_timefuse_fusor_full_scale.py`。
- P14c 文档已明确本轮不新增正式 VisualFeatureProvider、不抽真实 ViT provider、不新增 Visual
  RouterHead adapter、不接 `PredictionCacheExpertProvider` 到正式入口、不替换 Visual
  `SQLitePredictionIndex`、不改 legacy CSV/summary/metadata/status/checkpoint schema。
- P14c 文档已给出后续小步：P14d 做 Visual mock `FeatureBatch + mock RouterHead +
  EvaluationInputAdapter` protocol smoke；P14e 做 Visual eval-only legacy MLP adapter audit or
  smoke；P15 决定 branch-specific small entrypoint。
- 验证结果：
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py`：通过，确认 Visual mock provider 输出 `FeatureBatch(features=(4, 8), dtype=float32)`，且 provider 阶段不读文件、prediction/oracle/y_true/run_dir。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py`：通过，确认 TimeFuse 17 维 `FeatureBatch` 保序和数值一致。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py`：通过，确认 shared SQLite backend、`PredictionBatchReader` 和 `PredictionCacheExpertProvider` 输出 `ExpertBatch` 并对齐 P13b 参考。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py`：通过，确认 P13b real-derived fixture 可由 P12b small entrypoint 写出 canonical run_dir。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py`：通过，确认 `SampleManifest -> ExpertBatch -> FeatureBatch -> RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer` tiny 链路保序且只有 Runtime writer 写 run_dir。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router`：通过。

## 结论

P14c 已把 Visual eval-only 的 canonical bypass 口径冻结为纯内存协议链路。短期仍保留
legacy SQLite prediction path 和 legacy 输出 schema，只在 batch arrays 已存在后包装为
`ExpertBatch`。Visual 特征、专家预测、head 输出、evaluation adapter 和 runtime artifact writer
各自边界清晰，正式入口迁移仍留到后续小步。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续进入 P14d：Visual mock `FeatureBatch + mock RouterHead + EvaluationInputAdapter` protocol
   smoke，继续不迁移正式入口、不访问 `/data2`。
