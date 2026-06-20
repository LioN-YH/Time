# Stage 1 P10g TimeFuse Sample/Supervision Adapter

更新日期：2026-06-20

## 1. 目标

P10g 新增一个最小 TimeFuse feature/oracle 到 canonical `SampleManifest` 与
`SupervisionBatch` 的 smoke adapter，用于验证 TimeFuse-style fusor 历史 feature source 与
oracle/supervision source 可以被拆解为 sample manifest、split 和 supervision 三类职责。

本步只新增 smoke adapter，不修改 `train_timefuse_fusor_streaming.py`，
不修改 `train_visual_router_online_streaming.py`，不接正式训练入口，不改变正式 feature CSV、
oracle SQLite/parquet、summary、metadata、status 或 checkpoint schema。

## 2. 历史 TimeFuse 输入职责拆分

TimeFuse-style fusor 历史输入中：

- feature CSV 过去同时承担 sample source 和 17 维 feature source；
- oracle SQLite/parquet 过去承担 oracle / per-model error supervision；
- prediction SQLite / packed cache 承担专家预测和共享真实值；
- reader 当前把 feature、oracle、prediction 和 training batch 组装放在同一条入口链路中。

P10d/P10e 后的新架构将这些职责拆开：

- `SampleManifest` 只保存样本身份、split、顺序和轻量 lineage。
- 17 维 TimeFuse feature 值属于未来 `FeatureProvider`，不进入 `SampleManifestRow.extra`。
- `SupervisionBatch` 只保存 `sample_keys + model_columns + metric` 下的
  `oracle_model`、`oracle_value` 和 `[sample, expert]` per-model error 矩阵。
- oracle / error 只进入 supervision，不进入 deployable `FeatureProvider`。

## 3. P10g Adapter API

新增文件：

- `time_router/data/timefuse_supervision_adapter.py`

导出函数：

- `timefuse_features_to_sample_manifest(features, allowed_splits=..., lineage_columns=...)`
- `timefuse_oracle_to_supervision_batch(oracle, sample_keys=..., model_columns=..., metric=...)`

`features` / `oracle` 可以是小型 `pd.DataFrame` 或 CSV 路径。该 adapter 只服务 P10g smoke
和后续 schema 对齐讨论，不读取 `/data2`，不创建 run_dir，不扫描 full-scale artifact。

## 4. Smoke Fixture 字段

P10g feature fixture 使用以下最小 canonical-compatible 字段：

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
| `feature_shard` | 可进入 `SampleManifestRow.extra` 的轻量 lineage |
| `feature_schema_version` | 可进入 `SampleManifestRow.extra` 的轻量 feature schema lineage |
| `timefuse_feature_00` ... `timefuse_feature_16` | 源表中的 17 维 feature 值；不得进入 manifest extra |

P10g oracle fixture 使用以下监督字段：

| 字段 | 用途 |
| --- | --- |
| `sample_key` | 与 manifest / prediction / feature 对齐的 join key |
| `{model_name}_{metric}_error` | smoke 中的 per-model error 列 |

当前真实历史 feature/oracle 字段名未在 P10g 中强行冻结；正式入口接入前，需要单独审计真实
feature CSV 和 oracle SQLite/parquet schema，并决定是否增加字段映射层。

## 5. 校验范围

新增 smoke：

- `tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py`

覆盖内容：

- 临时构造 4 行 vali/test feature fixture，包含 17 维 TimeFuse feature 列。
- 临时构造对应 4 行 oracle/supervision fixture，包含五个专家的 `mae` error 列。
- 由 feature fixture 构造 `SampleManifest`。
- 校验 `sample_key` 唯一、split 保序和 `split_counts()`。
- 确认 17 维 feature 值不会进入 `SampleManifestRow.extra`，只保留 feature lineage。
- 分别对 vali/test 构造 `SupervisionBatch`。
- 校验 `oracle_model` 是每行 error 最小的专家名。
- 校验 `oracle_value` 是每行最小 error。
- 校验 `per_model_errors.shape == [sample, expert]`。
- 覆盖 CSV 入口、缺失 oracle 专家列、feature duplicate sample_key、oracle 缺失 sample_key
  和未知 split 的清晰报错。

## 6. 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接正式 Visual Router 或 TimeFuse-style fusor 入口。
- 不改 `TimeFuseFeatureCacheProvider`。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

## 7. 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 8. 后续

P10g 后，Visual labels 和 TimeFuse feature/oracle 两条历史输入路径都已有 canonical
sample/supervision adapter smoke。后续若继续推进，应先审计真实 full-scale schema，再决定
`SampleManifest` 物理存储格式、`SupervisionProvider` 缺失策略和正式入口分批接入口径。
