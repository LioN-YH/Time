# Stage 1 P12 canonical small entrypoint thin slice

日志日期：2026-06-20 13:48:45 CST

## 目的

新增 Stage 1 P12 small canonical entrypoint thin slice，把 P11d smoke 中已经验证的
`SampleManifest -> ExpertBatch -> FeatureBatch -> RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer`
canonical dataflow 包装为一个最小可执行 Python entrypoint。

## 背景

P11a 已冻结 canonical run artifact schema，P11b 已冻结 SampleManifest physical schema，
P11c 已新增 minimal Runtime artifact writer，P11d 已在 tiny fixture 上证明 canonical
dataflow 可以串到 tempfile canonical `run_dir`。P12 需要验证同一条 dataflow 能否通过
显式 CLI 接收 `output_root/run_name` 并写出 canonical run artifact。

本阶段只做 small/tiny entrypoint，不迁移正式 Visual Router / TimeFuse-style fusor
entrypoint，不新增 Bash launcher，不访问 `/data2`，不启动 pressure/full-scale。

## 操作

1. 新增 `scripts/run_stage1_canonical_small.py`。
   - CLI 参数包括 `--output-root`、`--run-name`、`--config-name`、`--branch-name`、
     `--feature-source` 和默认开启的 `--strict`。
   - 脚本内构造 3 行 tiny `SampleManifest`，从 manifest 取得 ordered sample_keys。
   - 使用内存 `TinyExpertProvider` 返回 `ExpertBatch`。
   - 默认在 tempfile 内写 tiny feature CSV，也支持通过 `--feature-source` 传入显式 CSV。
   - 使用 `TimeFuseFeatureCacheProvider`、`TimeFuseLinearSoftmaxHead` 和
     `EvaluationInputAdapter` 生成内存 summary 与 per-sample rows。
   - 只在 Runtime artifact writer 阶段创建 `run_dir` 并写出 canonical artifact。
2. 新增 `tests/smoke/stage1_canonical_small_entrypoint_smoke.py`。
   - 使用 `tempfile` 和 subprocess 调用新增 entrypoint。
   - 检查返回码、stdout `run_dir`、canonical 子目录、JSON/CSV 可读、prediction rows
     保持 manifest sample_key 顺序、未引用 `/data2`、未启动正式训练入口。
3. 新增 `docs/refactor/stage1_canonical_small_entrypoint.md`。
   - 明确 P12 是 small canonical entrypoint thin slice，不是正式入口迁移。
   - 明确 `scripts/` 只做薄 CLI，不承载 provider 内部逻辑。
   - 明确 `run_dir` 只传给 Runtime writer，不传给 Provider / Head / Evaluator。
4. 更新 `docs/refactor/stage1_refactor_roadmap.md`。
   - 将 P12 从后续计划更新为当前完成范围。
   - 增加 P12 验收命令和 P12b/P13 后续连接。
5. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`。
   - 记录 P12 small canonical Python entrypoint 已完成。
   - 将后续路线调整为 P12b 小规模输入映射审计、P13 正式入口迁移审计。
6. 更新 `WORKSPACE_STRUCTURE.md`。
   - 增加根级 `scripts/` 目录说明。
   - 登记 `scripts/run_stage1_canonical_small.py`、P12 文档和 P12 smoke。
7. 初步运行新增 smoke 和 entrypoint help：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python scripts/run_stage1_canonical_small.py --help
```

## 结果

新增 smoke 初跑已通过：

- subprocess 返回码为 0。
- stdout 包含 `run_dir`。
- canonical 子目录存在。
- `run_metadata.json`、`run_status.json`、`inputs/sample_manifest_ref.json`、
  `inputs/split_summary.json`、`evaluation/evaluation_summary.json` 可读。
- `predictions/prediction_rows.csv` 可读。
- prediction rows 保持 manifest sample_key 顺序。
- smoke stdout/stderr 未出现 `/data2`。
- 未启动正式 Visual Router / TimeFuse-style fusor 训练入口。

entrypoint help 可正常打印 CLI 参数说明。

随后运行 P12 指定的完整验收命令，全部通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

验收输出确认：

- P12 small canonical entrypoint smoke 全部通过。
- P11d canonical protocol run smoke 全部通过。
- P11c Runtime artifact writer smoke 全部通过。
- P10g/P10f/P10e/P10b 既有 smoke 全部通过。
- `compileall` 覆盖 `time_router`、`scripts`、`tests/smoke` 和
  `visual_router_experiments/stage1_vali_test_router`，无语法错误。

## 结论

P12 small canonical entrypoint thin slice 已完成最小实现：canonical pipeline 可以通过显式
Python entrypoint 运行，`output_root/run_name` 由 CLI 接收，`run_dir` 由 Runtime writer
显式创建并写出 canonical artifact。Provider / Head / Evaluator 仍只处理内存协议对象或
显式输入，不知道 `run_dir`。

## 下一步方案

1. 提交并推送到 `refactor/stage1-route-audit`。
2. 后续 P12b 可审计更真实的小规模 canonical input 映射；P13 再设计正式入口迁移审计或
   adapter 插入策略。正式入口、Bash launcher、legacy output schema 和 full-scale 资源调度
   仍保持不变。
