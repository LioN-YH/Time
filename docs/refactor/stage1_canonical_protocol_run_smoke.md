# Stage 1 P11d Canonical Protocol Run Smoke

创建日期：2026-06-20

## 1. 目标

P11d 新增一个最小 canonical protocol run smoke，把 P11a/P11b/P11c 已冻结或实现的契约串成
tiny fixture 上的完整 dataflow：

```text
SampleManifest + ordered sample_keys
  -> ExpertProvider / ExpertBatch
  -> FeatureProvider / FeatureBatch
  -> RouterHead / RouterOutput
  -> EvaluationInputAdapter / Evaluator
  -> Runtime artifact writer
  -> canonical run_dir
```

本阶段仍然只是 smoke，不迁移正式 Visual Router / TimeFuse-style fusor 入口，不新增
launcher/scripts，不访问 `/data2`，不启动训练或 full-scale。

新增文件：

- `tests/smoke/stage1_canonical_protocol_run_smoke.py`
- `docs/refactor/stage1_canonical_protocol_run_smoke.md`

## 2. 与 P11a/P11b/P11c 的连接

- P11a：smoke 写出的 `run_dir` 使用 canonical 子目录：
  `inputs/`、`indexes/`、`predictions/`、`evaluation/`、`checkpoints/` 和 `logs/`。
- P11b：smoke 以 `SampleManifest` 构造 1 个 `vali` 与 2 个 `test` sample，并写出
  `inputs/sample_manifest_ref.json` 与 `inputs/split_summary.json`。
- P11c：smoke 只通过 `time_router.runtime.artifact_writer` 创建和写出 `run_dir`。

## 3. Smoke 实现口径

`tests/smoke/stage1_canonical_protocol_run_smoke.py` 使用仓库内已有 protocol/adapters：

- `SampleManifest` / `SampleManifestRow`
- `ExpertBatch`
- `TimeFuseFeatureCacheProvider`
- `FeatureBatch`
- `TimeFuseLinearSoftmaxHead`
- `RouterOutput`
- `EvaluationInputAdapter`
- `time_router.runtime.artifact_writer`

ExpertProvider 采用测试内 `TinyExpertProvider` mock，直接从内存 fixture 构造
`ExpertBatch`，避免读取 full-scale prediction cache。FeatureProvider 读取测试内临时 feature
CSV；CSV 行顺序刻意不同于 manifest，用于验证 provider 必须按 Runtime 显式传入的
ordered sample_keys 输出 `FeatureBatch`。

## 4. Runtime / Provider 边界

P11d 固定以下边界：

- `run_dir` 只由 Runtime artifact writer 创建和写出。
- ExpertProvider 只接收 ordered sample_keys，并返回内存 `ExpertBatch`。
- FeatureProvider 只接收 ordered sample_keys 和显式 feature source，并返回内存
  `FeatureBatch`。
- RouterHead 只消费 `FeatureBatch` 和 `model_columns`，返回内存 `RouterOutput`。
- EvaluationInputAdapter 只消费 `ExpertBatch + RouterOutput`，返回内存 summary 和 rows。
- Provider / Head / Evaluator 不接收、不解析、不创建 `run_dir`。

smoke 在 Head/Evaluator 阶段 patch 常见文件 IO API，若组件尝试读取 cache、写 checkpoint 或
写 run artifact 会立即失败。

## 5. 写出的 canonical run_dir

P11d smoke 在 `tempfile` 下创建临时 `run_dir`，最小写出：

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

检查重点：

- `sample_key` 顺序从 `SampleManifest` 贯通到 `ExpertBatch`、`FeatureBatch`、
  `RouterOutput`、`EvaluationInput` 和 `prediction_rows.csv`。
- `predictions/` 只保存 per-sample rows。
- `evaluation/` 只保存聚合 summary。
- schema version、sample count、row count、split count 与 manifest 一致。
- 临时 run_dir 不在 `/data2` 下。

## 6. 明确不做

P11d 不做以下事项：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 small/pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不设计复杂 Runtime class、registry 或 migration framework。
- 不声称正式入口已迁移。

## 7. 验收

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
```

P11d 完成时还应运行 P11c/P10 系列既有 smoke 与 compileall，确认新增 canonical protocol run
smoke 没有破坏 runtime writer、sample/supervision adapter、SQLite backend 或 Stage 1 代码导入。
