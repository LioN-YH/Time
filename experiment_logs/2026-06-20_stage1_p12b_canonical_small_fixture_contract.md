# Stage 1 P12b canonical small fixture input contract

日志日期：2026-06-20 14:14:18 CST

## 目的

整理 P12 small canonical entrypoint 的 fixture input contract，让入口继续默认使用内联 tiny
fixture，同时可选读取显式 tiny manifest、feature 和 expert fixture 文件。

## 背景

P12 已完成 `scripts/run_stage1_canonical_small.py`，能够通过 `--output-root` 和 `--run-name`
运行 tiny canonical dataflow 并写出 canonical `run_dir`。但 tiny SampleManifest、
TinyExpertProvider 和 feature fixture 仍主要内联在脚本中，不利于后续 P13 审计真实
Visual/TimeFuse 小规模输入映射。

本轮只做 small/tiny input contract，不迁移正式入口，不访问 `/data2`，不启动训练、
pressure 或 full-scale。

## 操作

1. 更新 `scripts/run_stage1_canonical_small.py`：
   - 新增 `--sample-manifest` 和 `--expert-fixture` 参数，并复用既有 `--feature-source`。
   - 新增 tiny SampleManifest CSV/JSONL 读取 helper，manifest 文件行顺序作为 ordered sample_keys。
   - 新增 `JsonExpertFixtureProvider`，从 JSON 小数组读取 `model_columns`、`y_true` 和 `y_pred`，并按 manifest sample_keys 组装 `ExpertBatch`。
   - 保留未传参数时的 P12 内联 tiny manifest、内存 expert fixture 和临时 feature CSV。
   - 在 `run_metadata.json inputs` 中记录 `sample_manifest`、`feature_source` 和 `expert_fixture` 的来源摘要。
2. 新增 `tests/fixtures/stage1_canonical_small/`：
   - `sample_manifest.csv`
   - `features.csv`
   - `expert_predictions.json`
3. 新增 `tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py`：
   - 分别运行默认内联 fixture 与显式 fixture。
   - 对比两次运行的 `predictions/prediction_rows.csv` 业务输出一致。
   - 验证显式 fixture 的 prediction rows 保持 manifest 行顺序。
   - 验证 metadata inputs 记录显式 manifest、feature 和 expert fixture 来源。
4. 新增 `docs/refactor/stage1_canonical_small_fixture_contract.md`，并更新：
   - `docs/refactor/stage1_canonical_small_entrypoint.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `WORKSPACE_STRUCTURE.md`
5. 使用 conda 环境 `quito` 运行完整 P12b 验收 smoke 与 compileall。

## 结果

已完成的验证：

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

上述 smoke 与 compileall 均通过。新增 fixture smoke 证明显式 fixture 与默认内联 fixture 的
`prediction_rows.csv` 业务输出一致，显式 fixture 的 sample_key 顺序来自 manifest，且
`run_metadata.json inputs` 已记录 `sample_manifest`、`feature_source` 和
`expert_fixture` 来源摘要。

## 结论

P12b small fixture input contract 已落地。当前实现仍保持 `scripts/` 是薄 CLI，Provider /
Head / Evaluator 不接收 `run_dir`；只由 Runtime artifact writer 写出 canonical `run_dir`。
本轮未修改正式 Visual Router / TimeFuse-style fusor 入口，未访问 `/data2`，未启动训练或
full-scale。

## 下一步方案

1. 提交并推送到 `refactor/stage1-route-audit`。
2. 后续 P13 基于该 small fixture contract 审计真实 Visual labels、TimeFuse feature/oracle
   和 expert prediction cache 的小规模输入映射；在 P13 前不声称正式入口已迁移。
