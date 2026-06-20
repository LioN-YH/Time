# Stage 1 P13a real small-input mapping audit

日志日期：2026-06-20 14:35:32 CST

## 目的

审计真实 Visual Router 与 TimeFuse-style 小规模输入如何映射到 P12b canonical small
fixture contract，明确 SampleManifest、SupervisionProvider、FeatureProvider 和 prediction
backend 的字段边界。

## 背景

P12b 已让 `scripts/run_stage1_canonical_small.py` 支持显式
`sample_manifest.csv`、`features.csv` 和 `expert_predictions.json`，但这些仍是 tiny fixture，
不是真实 Visual Router 或 TimeFuse-style 输入。本步骤只做 mapping audit / 文档冻结，不迁移
正式入口，不新增真实数据脚本，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 操作

1. 读取用户粘贴目标文件，确认 P13a 验收范围和禁止范围。
2. 复核 `docs/refactor/stage1_canonical_small_fixture_contract.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_sample_manifest_physical_schema.md` 和 `WORKSPACE_STRUCTURE.md`。
3. 复核 `time_router/data/visual_labels_adapter.py`、
   `time_router/data/timefuse_supervision_adapter.py`、
   `time_router/features/timefuse_cache.py`、
   `time_router/experts/prediction_cache.py` 和
   `time_router/io/prediction_sqlite_backend.py` 的当前字段和职责边界。
4. 新增 `docs/refactor/stage1_real_small_input_mapping_audit.md`，记录 Visual / TimeFuse
   真实字段到 `stage1_sample_manifest_v1` 的映射表、supervision 边界、feature 边界、expert
   prediction 边界和 P13b 后续真实小规模 fixture 建议。
5. 更新 P12b fixture contract、entrypoint migration plan、roadmap 和结构索引，补充 P13a 状态。

## 结果

- 新增 P13a 审计文档，明确 `sample_key`、`split`、`config_name`、`dataset_name`、
  `item_id`、`channel_id`、`window_index`、`seq_len`、`pred_len` 和 `lineage` 的真实来源映射。
- 明确 oracle label、oracle value 和 per-model error 属于 `SupervisionProvider` /
  `SupervisionBatch` / evaluation diagnostics，不进入 `SampleManifest` 或 deployable
  `FeatureProvider`。
- 明确 TimeFuse 17 维 feature、Visual Quito history window、pseudo image 和 ViT feature 属于
  branch-specific `FeatureProvider`，P13a 不抽 Visual online ViT provider。
- 明确 P12b `expert_predictions.json` 只是 tiny fixture；正式路径仍走 prediction backend /
  `ExpertProvider` / `ExpertBatch`，不把 prediction cache path 或 SQLite index path 放进
  `SampleManifest`。
- P13a 指定 smoke 与 compileall 已在 `quito` 环境下通过，确认纯文档审计没有破坏代码行为。

已执行：

```bash
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

结果：全部通过。

## 结论

P13a 已完成真实小规模输入到 P12b fixture contract 的文档化 mapping audit。该结论只冻结字段
和职责边界，不代表正式 Visual Router 或 TimeFuse-style fusor 入口已经迁移。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续 P13b 可从已有 golden fixture、小规模真实样本或 P10f/P10g smoke fixture 派生真实小规模
   `sample_manifest.csv`、branch-specific feature fixture 和 expert fixture/backend smoke，用
   P12b entrypoint 验证真实字段映射与保序 join。
