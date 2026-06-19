# PredictionBatchReader 设计说明

创建日期：2026-06-19

## 1. 目的

`PredictionBatchReader` 是 Stage 1 P1 重构抽出的共享 prediction cache batch reader。它负责从 `merged_cache/manifest.csv` 或小规模 fixture root 中读取同一批 `sample_key` 的五专家预测 `y_pred` 和共享真实值 `y_true`，供后续 Visual Router 与 TimeFuse-style fusor 迁移复用。

本阶段只新增共享 reader 并接入 golden smoke，不迁移正式训练入口，不改变 cache schema、sample_key、专家顺序、模型结构、loss 或正式输出目录。

## 2. 代码位置

- `time_router/io/prediction_cache_reader.py`
- `time_router/io/__init__.py`
- `time_router/__init__.py`

底层数组读取仍复用既有公共模块：

- `visual_router_experiments/common/prediction_array_io.py`
- `visual_router_experiments/common/prediction_cache_schema.py`

## 3. 输入接口

`PredictionBatchReader` 初始化参数：

- `manifest_path`：直接指向 prediction cache 的 `manifest.csv`。
- `fixture_root`：指向包含 `manifest.csv` 的 fixture 或 `merged_cache` 目录。
- `model_columns`：专家动作空间顺序，默认固定为 `["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]`。
- `chunk_rows`：扫描 manifest 的 chunk 大小。

`load()` 参数：

- `sample_keys`：可选 sample_key 列表；传入时输出严格按该列表排序。
- `verify_metrics`：是否复算每条专家记录的 MAE/MSE，并与 manifest 对齐。

若不传 `sample_keys`，reader 会按 manifest 中首次出现顺序推断 sample_key 顺序。该模式适合 golden fixture 或小规模 smoke；正式 full-scale 训练应显式传入当前 shard 或当前 batch 的 sample_key，避免构造全量 batch。

## 4. 输出契约

`PredictionBatch` 字段：

- `sample_keys`：与数组第一维对齐的 sample_key 顺序。
- `y_pred`：形状为 `[num_samples, num_experts, pred_len, channels]`。
- `y_true`：形状为 `[num_samples, pred_len, channels]`。
- `metadata`：包含 `manifest_path`、`model_columns`、命中的 `manifest_rows`、每个 sample 的 manifest 原始专家行顺序、以及每个 `sample_key/model_name` 的 row index。

## 5. 关键约束

- sample_key 顺序只来自显式输入或 manifest 首次出现顺序，不允许被排序函数意外改变。
- 专家动作空间顺序只来自 `model_columns`，不使用 manifest 原始专家行顺序。
- 每个 `sample_key + model_name` 必须唯一，且每个 sample 必须覆盖五专家。
- `packed_npy_v1` 必须使用 `y_true_row_index` 和 `y_pred_row_index` 读取。
- 同一 sample_key 下五专家的 y_true 必须内容一致；路径或 row index 不同时会读取并比对内容。
- `verify_metrics=True` 时会从数组复算 MAE/MSE，验证 row index 与 manifest 行一致。
- 底层 packed 数组按路径分组读取，同一个 batch 内同一路径只打开一次。

## 6. 后续迁移方式

P1 只完成 reader 抽取和 smoke 接入。后续 P6 迁移正式入口时，建议按以下顺序接入：

1. 在小规模 fixture 或单 shard smoke 中用 `PredictionBatchReader` 替换入口内的本地 prediction 组装逻辑。
2. 对迁移前后输出做逐样本 `sample_key/y_pred/y_true` shape 和指标 comparison。
3. 保留现有 streaming/shard-aware 调度方式，只把“当前 batch sample_keys -> prediction tensors”的逻辑替换成共享 reader。
4. 每次入口迁移前后运行 `tests/smoke/stage1_golden_smoke.py`。

