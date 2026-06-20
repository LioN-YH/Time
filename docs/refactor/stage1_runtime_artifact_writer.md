# Stage 1 P11c Minimal Runtime Artifact Writer

创建日期：2026-06-20

## 1. 目标

本文记录 Stage 1 P11c 最小 Runtime artifact writer/helper 的实现边界。P11c 只把
P11a/P11b 已冻结的 future canonical artifact contract 变成可在临时 `run_dir` 中写出的
低风险 helper 和 smoke，不迁移正式入口，不新增 launcher/scripts，不访问 `/data2`，不启动
small/pressure/full-scale。

新增代码：

- `time_router/runtime/__init__.py`
- `time_router/runtime/artifact_writer.py`
- `tests/smoke/stage1_runtime_artifact_writer_smoke.py`

## 2. Helper API

`time_router.runtime` 暴露的最小 API：

| API | 职责 |
| --- | --- |
| `create_run_dir(output_root, run_name=None)` | 在 Runtime 显式传入的 `output_root` 下创建 canonical `run_dir` 和标准子目录 |
| `write_json_atomic(path, payload)` | 对现有 JSON 原子写入工具的 Runtime 层薄封装 |
| `write_run_metadata(run_dir, metadata)` | 写 `run_metadata.json` |
| `write_run_status(run_dir, status)` | 写 `run_status.json` |
| `write_sample_manifest_ref(run_dir, manifest_ref)` | 写 `inputs/sample_manifest_ref.json` |
| `write_split_summary(run_dir, split_summary)` | 写 `inputs/split_summary.json` |
| `write_evaluation_summary(run_dir, summary)` | 写 `evaluation/evaluation_summary.json` |
| `write_prediction_rows_csv(run_dir, rows)` | 写 `predictions/prediction_rows.csv` |

helper 只做以下事情：

- 创建 P11a 定义的 canonical 子目录：`inputs/`、`indexes/`、`predictions/`、
  `evaluation/`、`checkpoints/`、`logs/`。
- 检查最小必需字段存在。
- 复用 `time_router.io.json_utils.atomic_write_json(...)` 写 JSON。
- 用标准库 `csv.DictWriter` 写最小 per-sample rows。

## 3. 最小写出结构

P11c smoke 写出的最小结构：

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

`run_metadata.json` 至少要求：

- `run_artifact_schema_version`
- `protocol_version`
- `sample_manifest_schema_version`
- `evaluation_schema_version`
- `config_name`
- `branch_name`
- `created_at`
- `inputs`

`run_status.json` 至少要求：

- `status`
- `current_stage`
- `updated_at`
- `failure_reason`
- `checkpoint_pointer`

`sample_manifest_ref.json`、`split_summary.json` 字段沿用 P11b 最小 contract。
`evaluation_summary.json` 至少包含 `evaluation_schema_version`、`sample_count` 和 `metrics`。
`prediction_rows.csv` 至少包含 `sample_key`、`selected_model`、`y_true`、`y_pred` 和 `split`。

## 4. Runtime / Provider 边界

P11c 固定以下边界：

- `run_dir` 属于 Runtime，不属于 Provider。
- Provider 不创建、不解析、不硬编码 `run_dir`。
- Provider 只接收 Runtime 解析后的显式 `sample_keys`、split 或 backend handle，并返回内存 batch。
- Head 不写文件路径，也不决定 checkpoint 路径。
- Evaluator 只生成内存 summary/rows；文件写出由 Runtime artifact writer 完成。
- Bash 属于 `exp_scripts` 操作层，不进入 `time_router`。
- `time_router` 不知道 Bash，也不硬编码 `/data2`。

`tests/smoke/stage1_runtime_artifact_writer_smoke.py` 使用 `ProviderWithoutRunDir` mock 验证：

- mock provider 只接收 `sample_keys`；
- provider 不接收 `run_dir`；
- prediction rows 由 Runtime 写入 `predictions/`；
- evaluation summary 由 Runtime 写入 `evaluation/`。

## 5. 明确不做

P11c 不做以下事项：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不修改 legacy entrypoint 实际输出。
- 不新增 launcher/scripts。
- 不访问 `/data2`。
- 不启动 small/pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不设计复杂 Runtime framework、registry 或 migration framework。
- 不声称正式入口已迁移。

## 6. 验收

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
```

本 smoke 只使用 `tempfile` 本地临时目录，检查：

- canonical 目录结构存在；
- JSON 文件可读；
- schema version 字段存在；
- split summary count 正确；
- `predictions/` 与 `evaluation/` 分离；
- Provider mock 不接收 `run_dir`；
- 不访问 `/data2`；
- 不启动训练；
- 不改正式入口。

P11c 还应继续运行 P11b 后的 golden smoke 与 compileall，确认新 helper 没有破坏已有
protocol、adapter、SQLite backend 和 Stage 1 代码导入。
