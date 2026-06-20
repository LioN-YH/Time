# Stage 1 P12 Small Canonical Entrypoint Thin Slice

更新日期：2026-06-20

## 1. 目标

P12 新增 `scripts/run_stage1_canonical_small.py`，把 P11d tiny canonical dataflow 包装为
一个显式 Python entrypoint。它验证 canonical pipeline 能通过 CLI 接收
`output_root/run_name`、显式创建 `run_dir`，并调用已有 protocol、provider、head、
evaluator 和 Runtime artifact writer 写出 canonical run artifact。

P12b 在该入口上补充 small fixture input contract：entrypoint 继续默认使用内联 tiny
fixture，同时可选读取显式 tiny SampleManifest、feature CSV 和 expert prediction JSON。
本阶段只做 small/tiny entrypoint 与 fixture contract，不迁移正式 Visual Router 或
TimeFuse-style fusor 入口，不新增 Bash launcher，不访问 `/data2`，不启动 pressure/full-scale。

## 2. Entrypoint 边界

`scripts/run_stage1_canonical_small.py` 是薄 CLI：

- CLI 只解析 `--output-root`、`--run-name`、`--config-name`、`--branch-name`、
  `--sample-manifest`、`--expert-fixture`、`--feature-source` 和 `--strict`。
- Runtime 层从 tiny `SampleManifest` 取得 ordered sample_keys。
- 未传 `--sample-manifest` 时使用内联 tiny manifest；显式 manifest fixture 行顺序是 ordered
  sample_keys 来源。
- 未传 `--expert-fixture` 时使用内联 tiny expert fixture；显式 expert JSON 必须按 manifest
  sample_keys 组装 `ExpertBatch`。
- `TimeFuseFeatureCacheProvider` 只读取显式或临时 tiny feature CSV，并按 manifest sample_keys
  返回 `FeatureBatch`；feature CSV 行顺序可以不同于 manifest。
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

P12b 起，`run_metadata.json` 的 `inputs` 还记录 `sample_manifest`、`feature_source`
和 `expert_fixture` 的来源摘要。内联或临时生成输入记录为 `inline_fixture`；显式 fixture
文件记录为 `file` 与路径。`sample_manifest_ref_artifact` 指向 Runtime 写出的
`inputs/sample_manifest_ref.json`。

## 4. P12b Fixture Contract

P12b 的显式 fixture 位于：

```text
tests/fixtures/stage1_canonical_small/
├── sample_manifest.csv
├── features.csv
└── expert_predictions.json
```

`sample_manifest.csv` 使用 P11b `stage1_sample_manifest_v1` 的最小字段：
`sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、`channel_id`、
`window_index`、`seq_len` 和 `pred_len`。`features.csv` 至少包含 `sample_key`、
`trend_strength`、`seasonality_strength`、`recent_volatility`。`expert_predictions.json`
显式保存 `model_columns`、`y_true` 和 `y_pred` 小数组。

完整 contract 见 `docs/refactor/stage1_canonical_small_fixture_contract.md`。P12b 是
small fixture input contract hardening，不是正式入口迁移；fixture 后续用于 P13 审计真实
Visual/TimeFuse 小规模输入映射。

## 5. 明确不做

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

## 6. 验收

新增/更新 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
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

## 7. 后续连接

P12 证明 canonical dataflow 可以通过一个显式 Python entrypoint 跑出 small canonical
`run_dir`。P12b 固定 tiny fixture input contract。后续 P13 才能审计真实 Visual labels、
TimeFuse feature/oracle 和 expert prediction cache 的小规模映射，或设计正式入口迁移审计；
在那之前仍应保持 Provider / Head / Evaluator 不知道 `run_dir`，并保持正式入口与 legacy
输出 schema 不变。
