# Stage 1 P11b canonical SampleManifest physical schema

日志日期：2026-06-20 12:27:14 CST

## 目的

冻结 Stage 1 canonical `SampleManifest` 的物理存储 schema、schema version、split summary
schema，以及 canonical `run_dir/inputs/` 中 manifest snapshot/reference 的保存方式，为后续
P11c Runtime artifact writer/helper 和正式入口迁移提供稳定文档依据。

## 背景

P11a 已完成 canonical run artifact schema，冻结 future canonical `run_dir` 的
`run_metadata.json`、`run_status.json`、`inputs/`、`indexes/`、`predictions/`、`evaluation/`、
`checkpoints/` 和 `logs/` 结构。P10e/P10f/P10g 已有 `SampleManifest` / `SupervisionBatch`
protocol skeleton 与 Visual labels、TimeFuse feature/oracle smoke adapter，但还没有冻结
`SampleManifest` 的物理字段、版本号、split summary schema 和 `inputs/` 引用方式。

本步骤按用户要求优先文档冻结；未迁移正式入口，未新增 launcher/scripts，未访问 `/data2`，
未启动 small/pressure/full-scale 实验。

## 操作

1. 读取目标说明和当前仓库状态，确认当前分支为 `refactor/stage1-route-audit`，工作区初始干净。
2. 复核 P11a run artifact schema、P10 canonical sample/supervision boundary、entrypoint
   migration plan、roadmap、结构索引和实验日志总览。
3. 新增 `docs/refactor/stage1_sample_manifest_physical_schema.md`，冻结：
   - `stage1_sample_manifest_v1` 最小物理字段；
   - `sample_key` 规则；
   - `stage1_split_summary_v1` split summary schema；
   - `run_dir/inputs/` 中 snapshot 和 reference 两种 manifest 保存方式；
   - Visual labels 与 TimeFuse feature/oracle source 到 canonical manifest 的映射策略；
   - feature vector、oracle/error、prediction cache path 不进入 `SampleManifest` 的边界。
4. 更新 `docs/refactor/stage1_canonical_run_artifact_schema.md`，把 P11b schema version 与
   `inputs/` 连接写回 P11a 文档。
5. 更新 `docs/refactor/stage1_canonical_sample_supervision_boundary.md`，把 P10 阶段的语义字段
   对齐到 P11b 物理字段，明确 `seq_len/pred_len/lineage` 和 split summary。
6. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`、`docs/refactor/stage1_refactor_roadmap.md`
   和 `docs/refactor/stage1_target_architecture.md`，记录 P11b 已完成状态和 P11c 后续连接。
7. 更新 `WORKSPACE_STRUCTURE.md`，登记新增长期文档。
8. 新增本中文实验日志，并更新 `experiment_logs/README.md` 总览追踪表。
9. 运行用户指定的四个 smoke 与 compileall 验收：
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router`

## 结果

- 新增 P11b 文档，明确 `SampleManifest` 最小物理字段为 `sample_key`、`split`、
  `config_name`、`dataset_name`、`item_id`、`channel_id`、`window_index`、`seq_len`、`pred_len`
  和 `lineage`。
- 冻结 `sample_manifest_schema_version=stage1_sample_manifest_v1` 和
  `split_summary_schema_version=stage1_split_summary_v1`。
- 明确 `inputs/split_summary.json` 至少记录 split strategy、split names、count、唯一/重复
  sample_key 数、split overlap check、ordered sample_keys policy、source manifest reference
  和 created_at。
- 明确 `run_dir/inputs/` 可保存小规模 snapshot 或 full-scale reference，两者都必须能恢复
  ordered sample_keys；`run_metadata.json` 只记录 manifest 摘要和引用，不承载完整大表。
- 明确 Visual labels 与 TimeFuse feature/oracle 只映射 sample identity、split、config/dataset/window
  字段和轻量 lineage；17 维 feature、oracle/error、prediction cache path 和 SQLite index path
  不进入 `SampleManifest`。
- 指定四个 smoke 与 compileall 均通过。
- 本步骤未修改三个正式入口，未新增 launcher/scripts，未访问 `/data2`，未启动实验。

## 结论

P11b 的文档冻结已建立 Stage 1 canonical `SampleManifest` 物理 schema 与 split summary
schema。后续 Runtime artifact writer/helper 可以围绕 `run_dir/inputs/sample_manifest.*`、
`sample_manifest_ref.json` 和 `split_summary.json` 设计，而不让 Provider 知道 `run_dir`。

## 下一步方案

1. 小步提交，提交信息使用 `docs: define stage1 sample manifest physical schema`。
2. 推送到远程 `refactor/stage1-route-audit` 分支。
