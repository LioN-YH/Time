# Stage 1 P12 Small Canonical Entrypoint Thin Slice

更新日期：2026-06-20

## 1. 目标

P12 新增 `scripts/run_stage1_canonical_small.py`，把 P11d tiny canonical dataflow 包装为
一个显式 Python entrypoint。它验证 canonical pipeline 能通过 CLI 接收
`output_root/run_name`、显式创建 `run_dir`，并调用已有 protocol、provider、head、
evaluator 和 Runtime artifact writer 写出 canonical run artifact。

本阶段只做 small/tiny entrypoint thin slice，不迁移正式 Visual Router 或 TimeFuse-style
fusor 入口，不新增 Bash launcher，不访问 `/data2`，不启动 pressure/full-scale。

## 2. Entrypoint 边界

`scripts/run_stage1_canonical_small.py` 是薄 CLI：

- CLI 只解析 `--output-root`、`--run-name`、`--config-name`、`--branch-name`、
  `--feature-source` 和 `--strict`。
- Runtime 层从 tiny `SampleManifest` 取得 ordered sample_keys。
- `TinyExpertProvider` 只接收 ordered sample_keys，并返回 `ExpertBatch`。
- `TimeFuseFeatureCacheProvider` 只读取显式 tiny feature CSV，并返回 `FeatureBatch`。
- `TimeFuseLinearSoftmaxHead` 只消费 `FeatureBatch + model_columns`，返回 `RouterOutput`。
- `EvaluationInputAdapter` 只消费 `ExpertBatch + RouterOutput`，返回内存 summary 和 rows。
- 只有 Runtime artifact writer 接收 `run_dir` 并写出磁盘 artifact。

Provider / Head / Evaluator 不接收、不保存、不推导 `run_dir`。`scripts/` 只做薄 Python
entrypoint，不承载 provider 内部逻辑、训练 loop、checkpoint/resume、GPU 策略或 Bash
launcher 职责。

## 3. 写出 artifact

entrypoint 在 `output_root/run_name/` 下创建 canonical 子目录：

```text
inputs/
indexes/
predictions/
evaluation/
checkpoints/
logs/
```

当前写出：

- `run_metadata.json`
- `run_status.json`
- `inputs/sample_manifest_ref.json`
- `inputs/split_summary.json`
- `evaluation/evaluation_summary.json`
- `predictions/prediction_rows.csv`

`prediction_rows.csv` 保持 `SampleManifest` 原始 `sample_key` 顺序，`predictions/` 与
`evaluation/` 继续分层：逐样本 rows 不写入 `evaluation/`，聚合 summary 不写入
`predictions/`。

## 4. 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不声称正式入口已迁移。

## 5. 验收

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
```

既有回归 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 6. 后续连接

P12 只证明 canonical dataflow 可以通过一个显式 Python entrypoint 跑出 small canonical
`run_dir`。后续 P12b/P13 才能考虑更真实的小规模输入、正式 schema 映射或入口迁移审计；
在那之前仍应保持 Provider / Head / Evaluator 不知道 `run_dir`，并保持正式入口与 legacy
输出 schema 不变。
