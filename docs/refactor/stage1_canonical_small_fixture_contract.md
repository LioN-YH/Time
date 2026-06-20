# Stage 1 P12b Small Canonical Fixture Input Contract

更新日期：2026-06-20

## 1. 目标

P12b 在 P12 small canonical entrypoint thin slice 基础上，固定 small/tiny fixture 的输入
表达方式。`scripts/run_stage1_canonical_small.py` 继续默认使用内联 tiny fixture，同时允许
调用方显式传入 tiny SampleManifest、feature CSV 和 expert prediction JSON。

本阶段只是 small fixture input contract hardening，不是正式 Visual Router / TimeFuse-style
fusor 入口迁移，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 2. Fixture 文件

仓库内固定 smoke fixture 位于：

```text
tests/fixtures/stage1_canonical_small/
├── sample_manifest.csv
├── features.csv
└── expert_predictions.json
```

entrypoint 可选参数：

```bash
--sample-manifest tests/fixtures/stage1_canonical_small/sample_manifest.csv
--feature-source tests/fixtures/stage1_canonical_small/features.csv
--expert-fixture tests/fixtures/stage1_canonical_small/expert_predictions.json
```

未传 `--sample-manifest` / `--expert-fixture` 时，entrypoint 继续使用 P12 内联 tiny
manifest 和内存 expert fixture；未传 `--feature-source` 时，entrypoint 继续在 tempfile
中生成 tiny feature CSV。默认 smoke 因此保持不变。

## 3. SampleManifest Contract

`sample_manifest.csv` 使用 P11b `stage1_sample_manifest_v1` 的最小物理字段：

- `sample_key`
- `split`
- `config_name`
- `dataset_name`
- `item_id`
- `channel_id`
- `window_index`
- `seq_len`
- `pred_len`

P12b fixture 不新增正式 manifest schema，只在 tiny 文件中落地 P11b 已冻结的最小字段。
文件行顺序是 canonical ordered sample_keys 来源；`prediction_rows.csv` 必须保持这个顺序。
entrypoint 也支持同字段 JSONL，但本仓库 smoke 固定使用 CSV，避免过度设计。

## 4. Feature Contract

`features.csv` 至少包含：

- `sample_key`
- `trend_strength`
- `seasonality_strength`
- `recent_volatility`

feature CSV 行顺序可以不同于 manifest。`TimeFuseFeatureCacheProvider` 必须按 manifest
ordered sample_keys 返回 `FeatureBatch`，不能把 feature CSV 行顺序当作 evaluation 顺序。

## 5. Expert Fixture Contract

`expert_predictions.json` 使用小数组 JSON：

```text
{
  "model_columns": [...],
  "samples": [
    {"sample_key": "...", "y_true": [[...]], "y_pred": [[[...]], ...]}
  ]
}
```

约束：

- `model_columns` 是 `y_pred` 第一维的专家顺序。
- 每个 sample 必须包含 `sample_key`、`y_true` 和 `y_pred`。
- `y_pred.shape[0]` 必须等于 `len(model_columns)`。
- `y_pred.shape[1:]` 必须与 `y_true.shape` 一致。
- provider 按 manifest ordered sample_keys 组装 `ExpertBatch`。

本阶段故意不引入 parquet、SQLite、packed npy 或正式 `PredictionCacheExpertProvider` 接入。

## 6. Artifact 与 Metadata

显式 fixture 与默认内联 fixture 的业务输出应保持一致。P12b smoke 对比两次运行的
`predictions/prediction_rows.csv`，并检查显式 fixture 的 sample_key 顺序仍来自 manifest。

`run_metadata.json` 的 `inputs` 记录以下来源摘要：

- `sample_manifest`
- `feature_source`
- `expert_fixture`

内联或临时生成输入记录为 `inline_fixture`；显式文件记录为 `file` 和路径。`Provider` /
`Head` / `Evaluator` 仍不接收 `run_dir`，只有 Runtime artifact writer 写出 canonical
`run_dir`。`sample_manifest_ref_artifact` 另外记录 Runtime 写出的
`inputs/sample_manifest_ref.json`，避免把输入来源摘要和 run artifact 路径混在同一字段。

## 7. 与 P13 的连接

P12b 只固定 small fixture 如何表达、读取和保序。后续 P13 可基于同一 contract 审计真实
Visual labels、TimeFuse feature/oracle 和 expert prediction cache 的小规模映射关系，包括：

- 真实字段如何映射到 `stage1_sample_manifest_v1`；
- Visual 与 TimeFuse 的 split/source lineage 是否能共用 ordered sample_keys；
- feature source 与 expert fixture 是否能按 manifest 保序 join；
- 哪些逻辑属于 provider 内部，哪些仍应留在 thin entrypoint 或 Runtime。

P13a 已新增 `docs/refactor/stage1_real_small_input_mapping_audit.md`，冻结真实 Visual /
TimeFuse 小规模输入到 P12b fixture contract 的 mapping 边界。结论是：

- Visual labels 与 TimeFuse feature/oracle source 都只向 `SampleManifest` 映射样本身份、
  split、顺序和轻量 lineage。
- oracle label / oracle value / per-model error 属于 `SupervisionProvider`，不进入
  `SampleManifest` 或 deployable `FeatureProvider`。
- TimeFuse 17 维 feature 与 Visual history / pseudo image / ViT feature 属于 branch-specific
  `FeatureProvider`，不进入 `SampleManifest`。
- `expert_predictions.json` 仍只是 tiny fixture 格式；正式路径应继续走 prediction backend /
  `ExpertProvider` / `ExpertBatch`。

P13a 之后仍不得声称正式入口已经迁移。P13b 已新增
`docs/refactor/stage1_real_derived_small_fixture.md`、
`tests/fixtures/stage1_real_derived_small/` 和
`tests/smoke/stage1_real_derived_small_fixture_smoke.py`，从 P10f/P10g smoke 的 ETTh1 /
ETTm2 / weather 小样本身份派生 real-derived / schema-style fixture，并用 P12b entrypoint
验证 manifest 保序、feature/expert join、canonical `run_dir` 写出、metadata inputs 来源摘要和
evaluation sample_count。P13b 仍只做字段派生和保序验证，不新增 full-scale 数据链路；
`expert_predictions.json` 继续只是 small fixture，不是正式 prediction backend。

## 8. 验收

新增 fixture smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
```

回归 smoke：

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
