# Stage 1 P10f Visual Labels Sample/Supervision Adapter

更新日期：2026-06-20

## 1. 目标

P10f 新增一个最小 Visual labels CSV / DataFrame 到 canonical `SampleManifest` 与
`SupervisionBatch` 的 smoke adapter，用于验证历史 Visual Router labels 表可以被拆解为
sample manifest、split 与 supervision 三类职责。

本步只新增 smoke adapter，不修改 `train_visual_router_online_streaming.py`，不接正式训练入口，
不改变正式 labels CSV、summary、metadata、status 或 checkpoint schema。

## 2. 历史 labels CSV 职责拆分

Visual Router 历史 labels CSV 过去同时承担：

- sample manifest：`sample_key`、config、dataset、item/channel/window；
- split：vali/test 或其它划分字段；
- oracle supervision：oracle top-1、per-model error 或可由 per-model error 推导的信息；
- metadata / lineage：shard、来源、schema 版本等。

P10d/P10e 后的新架构将这些职责拆开：

- `SampleManifest` 只保存样本身份、split、顺序和轻量 lineage。
- `SupervisionBatch` 只保存 `sample_keys + model_columns + metric` 下的
  `oracle_model`、`oracle_value` 和 `[sample, expert]` per-model error 矩阵。
- oracle / error 只进入 supervision，不进入 deployable `FeatureProvider`。

## 3. P10f Adapter API

新增文件：

- `time_router/data/visual_labels_adapter.py`

导出函数：

- `visual_labels_to_sample_manifest(labels, allowed_splits=..., lineage_columns=...)`
- `visual_labels_to_supervision_batch(labels, sample_keys=..., model_columns=..., metric=...)`

`labels` 可以是小型 `pd.DataFrame` 或 CSV 路径。该 adapter 只服务 P10f smoke 和后续 schema
对齐讨论，不读取 `/data2`，不创建 run_dir，不扫描 full-scale artifact。

## 4. Smoke Fixture 字段

P10f fixture 使用以下最小 canonical-compatible 字段：

| 字段 | 用途 |
| --- | --- |
| `sample_key` | canonical join key |
| `split` | manifest split |
| `config_name` | config lineage |
| `dataset_name` | dataset lineage |
| `item_id` | item / series id |
| `channel_id` | channel id |
| `window_index` | item/channel 内窗口序号 |
| `seq_len` / `pred_len` | 可选窗口长度 lineage |
| `manifest_shard` | 可进入 `SampleManifestRow.extra` 的轻量 lineage |
| `{model_name}_{metric}_error` | smoke 中的 per-model error 列 |

当前真实历史 labels CSV 字段名未在 P10f 中强行冻结；正式入口接入前，需要单独审计真实
labels schema，并决定是否增加字段映射层。P10f 不为了猜字段名修改正式入口。

## 5. 校验范围

新增 smoke：

- `tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py`

覆盖内容：

- 临时构造 4 行 vali/test labels fixture。
- 包含五个专家的 `mae` per-model error 列。
- 构造 `SampleManifest` 并校验 `sample_key` 唯一、split 保序和 `split_counts()`。
- 分别对 vali/test 构造 `SupervisionBatch`。
- 校验 `oracle_model` 是每行 error 最小的专家名。
- 校验 `oracle_value` 是每行最小 error。
- 校验 `per_model_errors.shape == [sample, expert]`。
- 覆盖缺失专家列、重复 `sample_key` 和未知 split 的清晰报错。
- 额外写入 tempfile CSV，验证 CSV 和 DataFrame 入口一致。

## 6. 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接正式 Visual Router 或 TimeFuse-style fusor 入口。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

## 7. 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 8. 后续

P10f 之后建议进入 TimeFuse feature/oracle 到 `SampleManifest` / `SupervisionBatch` 的 smoke adapter，
验证 TimeFuse-style fusor 当前 feature CSV、oracle SQLite/parquet 与 prediction SQLite 分工也能
拆解到同一 canonical sample/supervision 边界。
