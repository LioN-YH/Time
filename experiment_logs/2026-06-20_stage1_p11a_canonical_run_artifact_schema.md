# Stage 1 P11a canonical run artifact schema

日志日期：2026-06-20 11:58:29 CST

## 目的

冻结 Stage 1 canonical run artifact schema，明确新版 canonical pipeline 运行后
`run_dir` 的最小推荐结构、目录职责、Runtime 与 Provider 边界、最小版本策略和 legacy
`96_48_S` full-scale policy。

## 背景

P10h 已经把 Stage 1 entrypoint migration plan 对齐到
`SampleManifest + SplitStrategy -> ExpertProvider / prediction backend -> SupervisionProvider
-> FeatureProvider -> RouterHead -> EvaluationInputAdapter / Evaluator -> Runtime / artifact writer`
的 canonical dataflow。下一步需要先冻结 run artifact schema，避免后续 Runtime、launcher、
provider 或 evaluator 接入时继续混用 legacy 输出目录语义。

本轮只做文档冻结，不修改正式入口，不新增 launcher/scripts，不启动实验，不访问 `/data2`，
不改变当前 legacy entrypoint 的实际输出 schema。

## 操作

1. 新增 `docs/refactor/stage1_canonical_run_artifact_schema.md`。
2. 在 P11a 文档中定义 future canonical `run_dir` 推荐结构：
   `run_metadata.json`、`run_status.json`、`inputs/`、`indexes/`、`predictions/`、
   `evaluation/`、`checkpoints/` 和 `logs/`。
3. 明确 `run_metadata.json` 是静态描述文件，记录 git commit/branch、config、branch、
   protocol/version、输入引用、环境摘要和创建时间；`run_status.json` 是动态状态文件，记录
   pending/running/completed/failed/interrupted/resumed、current_stage、failure_reason、
   checkpoint_pointer 和 updated_at。
4. 明确 `inputs/`、`indexes/`、`predictions/`、`evaluation/`、`checkpoints/`、`logs/`
   的职责边界，其中 `evaluation/` 承载 summary、comparison、selected counts、diagnostic
   metrics 和 `evaluation_report.md`，`predictions/` 承载逐样本 prediction/fusion/router rows。
5. 明确 Runtime 与 Provider 边界：`run_dir` 属于 Runtime，Provider 不持有、不解析、不硬编码
   `run_dir`，也不写 status、metadata、checkpoint 或 logs。
6. 明确最小 versioning strategy：`run_artifact_schema_version`、`protocol_version`、
   `sample_manifest_schema_version`、`supervision_schema_version`、`prediction_backend_version`
   和 `evaluation_schema_version`；不设计复杂 registry，不把 `ExpertBatch` / `RouterOutput` /
   `EvaluationInput` 等内存协议对象等同于磁盘 schema。
7. 明确 legacy `96_48_S` full-scale 输出只作为 sanity reference，不作为 canonical artifact
   schema 的兼容目标；后续重跑只比较指标量级和运行经验差异，不要求逐文件 schema 对齐。
8. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`，把 run artifact schema 从“尚未冻结”
   改为 P11a 已冻结，并把后续路线连接到 `SampleManifest` 物理存储和最小 Runtime artifact
   writer 边界。
9. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P11a 小步、验收命令和 P11b/P11c
   后续连接。
10. 更新 `docs/refactor/stage1_target_architecture.md`，补充 P11a schema 文档、推荐 `run_dir`
    结构和 legacy policy。
11. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P11a 文档和结构更新时间。
12. 使用 `quito` conda 环境运行指定四个 smoke 和 compileall 验收。

## 结果

- 已新增 P11a schema 文档，并明确：
  - `run_dir` 属于 Runtime，不属于 Provider。
  - Bash 属于 `exp_scripts` 操作层，不进入 `time_router`。
  - `time_router` 不知道 Bash，也不硬编码 `/data2`。
  - cache 是 implementation，不是 interface。
  - `evaluation/` 与 `predictions/` 的边界不重叠，不新增 `summaries/`。
  - Visual-specific 与 TimeFuse-style-specific artifact 只进入各自 branch-specific 区域，不成为另一分支必需 schema。
- 已同步 roadmap、entrypoint migration plan、target architecture 和结构索引。
- 本步未修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 或
  `launch_timefuse_fusor_full_scale.py`。
- 本步未新增 provider/head/runtime 代码，未新增 Bash/scripts，未访问 `/data2`，未启动
  small/pressure/full-scale，未改正式 CSV / summary / metadata / status / checkpoint schema，
  未改 loss、optimizer、scaler 或 checkpoint/resume。
- 验收结果：
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py` 通过。
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router` 通过。

## 结论

Stage 1 future canonical run artifact schema 已完成文档冻结。后续 Runtime、launcher 或
artifact writer 接入应以 P11a 文档为边界：Runtime 拥有 `run_dir` 和写出职责，Provider /
Head / Evaluator 保持路径无关，legacy `96_48_S` full-scale 结果只作为 sanity reference。

## 下一步方案

1. 检查 diff，确认只包含文档、结构索引和实验日志。
2. 小步提交并推送 `refactor/stage1-route-audit`。
3. 后续 P11b 可冻结 `SampleManifest` 物理存储格式、schema version 和 `inputs/` 中的 split
   summary 写入方式；P11c 可设计最小 Runtime artifact writer 或 helper。
