# Stage 1 P13c backend/provider 连接审计

日志日期：2026-06-20 15:03:56 CST

## 目的

在 P13a/P13b 已完成真实小规模输入映射审计和 real-derived small fixture smoke 后，继续审计真实
small batch 后续如何接入 prediction backend / `ExpertProvider` / `FeatureProvider`。本步骤只做文档审计和连接方案冻结，不迁移正式入口，不新增真实数据脚本，不访问 `/data2`，不启动训练或 full-scale。

## 背景

P12b/P13b 的 canonical small entrypoint 已能读取 `sample_manifest.csv`、`features.csv` 和
`expert_predictions.json`，并验证 manifest 保序、feature/expert join 以及 canonical `run_dir`
写出。但该路径仍是 fixture-driven path：`expert_predictions.json` 不是正式 prediction backend，
三列 `features.csv` 也不是 TimeFuse 17 维 feature cache 或 Visual online ViT feature。

## 操作

1. 读取当前任务要求、分支状态、`HANDOFF.md`、`experiment_logs/README.md` 和 P13a/P13b 相关文档，确认本轮只允许文档审计、smoke 验证、提交和 push。
2. 新增 `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`，冻结 P13c 连接方案。
3. 更新 `docs/refactor/stage1_real_derived_small_fixture.md` 和
   `docs/refactor/stage1_real_small_input_mapping_audit.md`，补充 P13c 已完成的后续连接结论。
4. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md` 和
   `docs/refactor/stage1_refactor_roadmap.md`，把 P13c 状态和 P13d/P13e/P14/P15 小步建议写入路线。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P13c 审计文档和 roadmap 口径变化。

## 结果

P13c 文档明确：

- P13b `expert_predictions.json` 后续应由 prediction backend / `ExpertProvider` / `ExpertBatch` 替换。
- shared prediction SQLite backend 适合 Runtime/backend prepare 层；`PredictionBatchReader` 适合底层 cache reader；`PredictionCacheExpertProvider` 适合作为 smoke-only prediction-cache adapter。
- prediction cache path、SQLite path、packed npy path 不进入 `SampleManifest`。
- P13b 三列 `features.csv` 只是 schema-style fixture；TimeFuse 17 维 feature 后续应通过 `TimeFuseFeatureCacheProvider` 或 branch-specific small provider smoke 输出 `FeatureBatch`。
- Visual history window / pseudo image / frozen ViT embedding 属于 Visual `FeatureProvider` 插入点；本轮不抽 Visual online ViT provider。
- `scripts/run_stage1_canonical_small.py` 继续作为 generic thin CLI；branch-specific feature/head 验证另走 smoke 或 small entrypoint。
- 后续建议拆为 P13d prediction backend -> `ExpertBatch` small smoke、P13e TimeFuse 17 维 feature provider small smoke、P14a Visual feature provider insertion audit、P14b Visual eval-only canonical bypass plan 和 P15 branch-specific small entrypoint decision。

验收已在 `2026-06-20 15:07:46 CST` 前完成，全部通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 结论

P13c 已把真实 small batch 的 backend/provider 连接边界从 P13b fixture 中拆清楚：generic small CLI 继续负责 fixture contract 和 canonical artifact 验证，真实 prediction 与 feature 连接应通过 smoke-only provider/backend 或 branch-specific small entrypoint 小步推进。指定 smoke 和 compileall 均通过，且本轮未改正式训练入口、未访问 `/data2`、未启动训练或 full-scale。

## 下一步方案

1. 检查 git diff，确认只包含文档、结构索引和实验日志。
2. 使用提交信息 `docs: audit stage1 backend provider connection` 提交并 push 到远程 `refactor/stage1-route-audit` 分支。
3. 后续 P13d 优先做 prediction backend -> `ExpertBatch` small smoke，对照 P13b JSON fixture，不接正式入口。
