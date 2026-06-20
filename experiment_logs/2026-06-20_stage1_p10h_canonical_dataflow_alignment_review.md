# Stage 1 P10h canonical dataflow alignment review

日志日期：2026-06-20 11:04:26 CST

## 目的

更新 Stage 1 entrypoint migration plan，把迁移叙述从旧的“Visual Router 和
TimeFuse-style fusor 分别拆解入口职责”调整为“统一 canonical dataflow + branch-specific
implementations”视角。

## 背景

P10d/P10e/P10f/P10g 已经完成 canonical `SampleManifest` / `SupervisionBatch` 边界、
最小协议 smoke、Visual labels adapter smoke 和 TimeFuse feature/oracle adapter smoke。后续入口迁移
需要先统一文档口径，避免继续把 labels CSV、feature CSV、oracle reader 或 prediction reader
的历史分工作为长期接口边界。

## 操作

1. 重写 `docs/refactor/stage1_entrypoint_migration_plan.md`，新增 canonical dataflow 总览：
   `SampleManifest + SplitStrategy -> ExpertProvider / prediction backend -> SupervisionProvider
   -> FeatureProvider -> RouterHead -> EvaluationInputAdapter / Evaluator -> Runtime / artifact writer`。
2. 在 entrypoint migration plan 中明确两条路线应共用的层：
   `SampleManifest`、`SplitStrategy` 语义、prediction SQLite backend / `ExpertBatch`、
   `SupervisionBatch` / `SupervisionProvider` contract、`EvaluationInputAdapter` / Evaluator
   metrics 和 run artifact contract 方向。
3. 在 entrypoint migration plan 中明确 branch-specific 层：
   Visual Router 的 Quito history window / pseudo image / ViT feature provider、
   TimeFuse 的 17 维 feature cache / feature-only scaler、`VisualMLPRouterHead`、
   `TimeFuseLinearSoftmaxHead`、Visual `fusion_huber_kl` / classification objective 和
   TimeFuse SmoothL1 weighted fusion objective。
4. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P10h 小步，记录当前代码状态和下一阶段路线。
5. 更新 `docs/refactor/stage1_target_architecture.md`，把 P10h 作为 canonical dataflow
   对齐结论写入目标架构。
6. 更新 `docs/refactor/stage1_canonical_sample_supervision_boundary.md`，补充 P10h 与后续 schema
   冻结前置条件。
7. 更新 `WORKSPACE_STRUCTURE.md`，同步 P10h 文档索引和工作区更新时间。
8. 使用 `quito` conda 环境运行指定四个 smoke 和 compileall 验收。

## 结果

- 文档已明确当前完成状态：Visual evaluation/training `ExpertBatch` bypass、Visual labels
  adapter smoke、TimeFuse protocol chain smoke、TimeFuse sample/supervision adapter smoke 和
  shared prediction SQLite backend smoke 均已完成。
- 文档已明确正式入口尚未整体迁移到 canonical dataflow。
- 文档已明确下一阶段路线：审计真实 full-scale Visual labels schema 与 TimeFuse feature/oracle
  schema，冻结 `SampleManifest` 物理存储格式与版本号，冻结 run artifact schema，准备 small /
  pressure / full-scale scripts，并将 legacy `96_48_S` full-scale 结果作为 reference baseline。
- 本步未修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 或
  `launch_timefuse_fusor_full_scale.py`。
- 本步未新增 provider/head/runtime 代码，未修改 `time_router/protocols/types.py`，未访问
  `/data2`，未启动 pressure/full-scale，未改正式 CSV / summary / metadata / status /
  checkpoint schema。
- 验收结果：
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router` 通过。

## 结论

Stage 1 entrypoint migration plan 已对齐到 canonical dataflow 视角。后续迁移应以
`SampleManifest` 主索引和 shared prediction/supervision/evaluation/runtime artifact contract
为主线，同时保留 Visual 与 TimeFuse 的 feature、head 和 objective 分支差异。

## 下一步方案

1. 检查 diff，确认只包含文档、结构索引和实验日志。
2. 小步提交并推送 `refactor/stage1-route-audit`。
3. 后续进入真实 full-scale Visual labels schema 与 TimeFuse feature/oracle schema 审计。
