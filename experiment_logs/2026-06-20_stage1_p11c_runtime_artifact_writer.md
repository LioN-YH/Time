# Stage 1 P11c minimal Runtime artifact writer/helper

日志日期：2026-06-20 12:56:52 CST

## 目的

在 P11a canonical run artifact schema 和 P11b canonical `SampleManifest` physical schema
基础上，新增一个最小 Runtime artifact writer/helper，使 Runtime 能在临时 `run_dir` 中写出
P11a/P11b 定义的最小 canonical artifact。

本轮目标只覆盖低风险 helper、smoke、文档和日志；不迁移正式入口，不新增 launcher/scripts，
不访问 `/data2`，不启动 small/pressure/full-scale。

## 背景

P11a 已冻结 future canonical `run_dir` 结构：

- `run_metadata.json`
- `run_status.json`
- `inputs/`
- `indexes/`
- `predictions/`
- `evaluation/`
- `checkpoints/`
- `logs/`

P11b 已冻结：

- `stage1_sample_manifest_v1`
- `stage1_split_summary_v1`
- `inputs/sample_manifest_ref.json` 或 snapshot 保存方式
- `inputs/split_summary.json`

P11c 需要把上述文档 contract 变成最小可用写出 helper，同时继续保持 Provider / Head /
Evaluator 不知道 `run_dir` 的边界。

## 操作

1. 新增 `time_router/runtime/__init__.py`，导出 P11c Runtime artifact writer public API。
2. 新增 `time_router/runtime/artifact_writer.py`，实现：
   - `create_run_dir(output_root, run_name=None)`
   - `write_json_atomic(path, payload)`
   - `write_run_metadata(run_dir, metadata)`
   - `write_run_status(run_dir, status)`
   - `write_sample_manifest_ref(run_dir, manifest_ref)`
   - `write_split_summary(run_dir, split_summary)`
   - `write_evaluation_summary(run_dir, summary)`
   - `write_prediction_rows_csv(run_dir, rows)`
3. 新增 `tests/smoke/stage1_runtime_artifact_writer_smoke.py`，使用 `tempfile` 创建本地临时
   `output_root/run_dir`，写出最小 canonical artifact，并用 `ProviderWithoutRunDir` mock 验证
   Provider 只接收 `sample_keys`，不接收 `run_dir`。
4. 新增 `docs/refactor/stage1_runtime_artifact_writer.md`，说明 helper API、写出结构、
   Runtime / Provider / Head / Evaluator 边界、明确不做范围和 smoke 验收命令。
5. 更新以下文档，把 P11c 状态从“可设计”同步为“已新增最小 helper，正式入口尚未迁移”：
   - `docs/refactor/stage1_canonical_run_artifact_schema.md`
   - `docs/refactor/stage1_sample_manifest_physical_schema.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
6. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 runtime package、P11c 文档和 smoke。
7. 运行指定 smoke 与 compileall 验收。

## 结果

新增 helper 可写出以下最小结构：

```text
run_dir/
├── run_metadata.json
├── run_status.json
├── inputs/
│   ├── sample_manifest_ref.json
│   └── split_summary.json
├── indexes/
├── predictions/
│   └── prediction_rows.csv
├── evaluation/
│   └── evaluation_summary.json
├── checkpoints/
└── logs/
```

新增 smoke 检查通过：

- canonical 子目录存在；
- JSON 文件可读；
- schema version 字段存在；
- split summary count 正确；
- `predictions/` 与 `evaluation/` 分离；
- Provider mock 不接收 `run_dir`；
- 不访问 `/data2`；
- 不启动训练；
- 不修改正式入口。

实际运行并通过的命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

所有命令均返回成功。

## 结论

P11c 最小 Runtime artifact writer/helper 已完成。它只属于 Runtime artifact 写出层，复用
现有 JSON 原子写入工具，按 P11a/P11b 最小字段要求写出 canonical artifact，不引入复杂
Runtime 类层级、registry 或 migration framework。

本轮未修改以下正式入口：

- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`

本轮未访问 `/data2`，未启动训练或实验，未改变 legacy CSV / summary / metadata / status /
checkpoint schema。

## 下一步方案

1. 提交并推送 `runtime: add minimal stage1 artifact writer` 到
   `refactor/stage1-route-audit`。
2. 后续可在 P11d 或下一小步准备 small canonical entrypoint 的薄接入方案，但仍需保持
   Provider / Head / Evaluator 不知道 `run_dir`。
3. 正式入口迁移前仍需审计真实 full-scale Visual labels schema 与 TimeFuse feature/oracle
   schema，并只在小规模 canonical 链路通过后再考虑 pressure/full-scale。
