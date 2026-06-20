# Stage 1 P11d canonical protocol run smoke

日志日期：2026-06-20 13:22:22 CST

## 目的

新增一个最小 canonical protocol run smoke，把已有 Stage 1 protocol/adapters 与 P11c Runtime artifact writer 串起来，验证 tiny fixture 可以完整跑出 canonical `run_dir`。

## 背景

P11a 已冻结 canonical run artifact schema，P11b 已冻结 `SampleManifest` physical schema，P11c 已新增最小 Runtime artifact writer。P11d 的目标是在不迁移正式入口、不新增 launcher/scripts、不访问 `/data2`、不启动训练或 full-scale 的前提下，验证以下 dataflow 可以组合：

```text
SampleManifest + ordered sample_keys
  -> ExpertProvider / ExpertBatch
  -> FeatureProvider / FeatureBatch
  -> RouterHead / RouterOutput
  -> EvaluationInputAdapter / Evaluator
  -> Runtime artifact writer
  -> canonical run_dir
```

## 操作

1. 新增 `tests/smoke/stage1_canonical_protocol_run_smoke.py`。
   - 使用 3 行 tiny `SampleManifestRow`，包含 1 个 `vali` 和 2 个 `test` sample。
   - 从 `SampleManifest.sample_keys()` 获取 ordered sample_keys。
   - 使用测试内 `TinyExpertProvider` mock 构造 `ExpertBatch`，该 provider 只接收 sample_keys，不接收、不推导 `run_dir`。
   - 使用测试内临时 TimeFuse feature CSV 和 `TimeFuseFeatureCacheProvider` 构造 `FeatureBatch`；CSV 行顺序故意不同于 manifest，用于验证 provider 按调用方 sample_keys 保序。
   - 使用 `TimeFuseLinearSoftmaxHead` 生成 `RouterOutput`。
   - 使用 `EvaluationInputAdapter` 生成内存 summary 和 per-sample rows。
   - 使用 `time_router.runtime.artifact_writer` 在 `tempfile` 下创建 canonical `run_dir` 并写出 `run_metadata.json`、`run_status.json`、`inputs/sample_manifest_ref.json`、`inputs/split_summary.json`、`evaluation/evaluation_summary.json` 和 `predictions/prediction_rows.csv`。
2. 新增 `docs/refactor/stage1_canonical_protocol_run_smoke.md`，记录 P11d 目标、与 P11a/P11b/P11c 的连接、实现口径、Runtime/Provider 边界、写出结构、明确不做范围和验收命令。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，将 P11d 标记为已完成，并新增 P11d 章节和 P12 small canonical entrypoint thin slice 连接说明。
4. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`，记录 P11d 已串通 tiny canonical dataflow，但正式 legacy entrypoint 尚未迁移。
5. 更新 `docs/refactor/stage1_runtime_artifact_writer.md`，补充 P11d 对 P11c writer 的后续连接和边界验证。
6. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P11d 文档和 smoke。

## 结果

已运行新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
```

结果：通过。检查项包括：

- `SampleManifest` 按 row order 输出 ordered sample_keys；
- ExpertProvider / FeatureProvider 只接收 ordered sample_keys，不接收 `run_dir`；
- RouterHead / EvaluationInputAdapter 只处理内存协议对象，不访问文件系统；
- ordered sample_keys 从 `SampleManifest` 贯通到 `predictions/prediction_rows.csv`；
- Runtime artifact writer 是唯一写 `run_dir` 的组件；
- canonical `run_dir` artifact 可读，`predictions/` 与 `evaluation/` 分层正确；
- 临时 fixture 和 run_dir 均不在 `/data2` 下。

随后继续运行完整 P11d 验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

结果：全部通过，完成时间为 2026-06-20 13:24:24 CST。`compileall` 覆盖
`time_router`、`tests/smoke` 和 `visual_router_experiments/stage1_vali_test_router`，新增
`stage1_canonical_protocol_run_smoke.py` 可正常编译。

## 结论

P11d tiny canonical protocol run smoke 的核心代码和文档已完成，已证明 P11a/P11b/P11c 契约可以在 tiny fixture 上组合为 canonical `run_dir`。本次没有修改正式 Visual Router / TimeFuse-style fusor entrypoint，没有新增 launcher/scripts，没有访问 `/data2`，没有启动训练，也没有修改 legacy CSV / summary / metadata / status / checkpoint schema。

## 下一步方案

1. 检查 git diff，只保留 P11d smoke、文档、结构索引和实验日志相关改动。
2. 小步提交并 push 到远程 `refactor/stage1-route-audit` 分支。
3. 后续 P12 再考虑 small canonical entrypoint thin slice，仍需保持 Provider / Head / Evaluator 不知道 `run_dir`。
