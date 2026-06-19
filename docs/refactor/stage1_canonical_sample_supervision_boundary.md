# Stage 1 P10d/P10e Canonical SampleManifest 与 Supervision Boundary

设计日期：2026-06-20
更新日期：2026-06-20

## 1. 目标

本文定义 Visual Router 与 TimeFuse-style fusor baseline 后续可共用的 canonical
`SampleManifest` / `SplitStrategy` / `SupervisionProvider` 边界。

P10d 只做架构设计和文档冻结，不修改正式入口，不新增 provider 代码，不改训练行为。
P10e 在该边界上新增最小 lightweight dataclass/helper 与纯内存 smoke，用于锁定
`SampleManifest` / `SupervisionBatch` 的 public API 雏形；仍不接 Visual Router、
TimeFuse-style fusor 或任何正式训练入口。
用户已接受必要时重跑 Stage 1 实验，因此后续新 schema 应优先服务长期可扩展边界，
不再把完全兼容历史 labels CSV、feature CSV、oracle SQLite/parquet 或 runtime artifact schema
作为最高优先级。

## 2. 核心设计结论

Stage 1 后续主索引应从“某个历史输出文件”提升为显式 `SampleManifest`。

长期数据流为：

```text
SampleManifest + SplitStrategy
  -> ordered sample_keys
  -> ExpertProvider / prediction backend
  -> FeatureProvider
  -> SupervisionProvider
  -> RouterHead / training loop / Evaluator
```

其中：

- `SampleManifest` 是样本身份、顺序和 split 的 canonical source。
- `ExpertProvider / ExpertBatch` 只提供专家预测 `y_pred` 与共享 `y_true`。
- `SupervisionProvider` 只提供训练监督、诊断、baseline 或 upper-bound 需要的 oracle / error 信息。
- `FeatureProvider` 只提供可部署特征，不读取 oracle、expert error 或未来 `y`。
- Visual Router 与 TimeFuse-style fusor 必须消费同一套 `sample_key` 顺序。

## 3. Canonical SampleManifest 字段

P10e 已在 `time_router.protocols` 中新增最小 `SampleManifestRow` / `SampleManifest`
协议类型；P11/P12 若冻结物理存储 schema，至少应保留以下语义字段：

| 字段 | 必需性 | 含义 |
| --- | --- | --- |
| `sample_key` | 必需 | 稳定样本主键；后续 prediction、feature、supervision 均以它 join |
| `split` | 必需 | 由 `SplitStrategy` 写入或 materialize 的 train/vali/test/heldout 等划分 |
| `config_name` | 必需 | 例如 `96_48_S`，用于 config 级分组和输出 lineage |
| `dataset_name` | 必需 | Quito dataset 或等价数据集名称 |
| `item_id` | 必需 | 样本所在 item / series 标识 |
| `channel_id` | 必需 | 单变量或多变量 channel 标识；当前 Stage 1 S 口径通常为单 channel |
| `window_index` | 必需 | 同一 item/channel 下的窗口序号 |
| `seq_len` | 可选但推荐 | 历史窗口长度；同 manifest 支持多 config 时应写入 |
| `pred_len` | 可选但推荐 | 预测长度；同 manifest 支持多 config 时应写入 |
| `manifest_shard` | 可选 | 大规模 materialize 时的 shard lineage |
| `source_manifest_path` | 可选 | 兼容期记录生成来源 |
| `extra` | 可选 | branch-specific 或审计 metadata；不得放入 oracle label、专家误差或未来信息 |

字段约束：

- `sample_key` 必须唯一；若未来支持同一 `sample_key` 多 metric supervision，应在
  supervision 层用 `metric` 区分，而不是复制 manifest 行。
- `split` 不应由 labels CSV、feature CSV、oracle reader 和 prediction reader 各自推导。
- `SampleManifest` 可以 materialize 为 CSV/Parquet/SQLite，但 interface 语义不能绑定某一种存储。
- `extra` 只能保存不可训练泄漏的 lineage 或 branch-specific metadata。

P10e 最小 helper 能力：

- `SampleManifest.rows` 使用 `tuple[SampleManifestRow, ...]` 保持调用方原始顺序。
- `SampleManifest.validate_unique_sample_keys()` 校验 `sample_key` 唯一。
- `SampleManifest.sample_keys(split=None)` 按 rows 原始顺序返回 ordered sample keys。
- `SampleManifest.split_counts()` 返回 split 样本数统计。
- 该 helper 不读取 labels CSV、feature CSV、prediction cache、oracle backend 或正式输出目录。

## 4. SplitStrategy 边界

`SplitStrategy` 长期负责生成或校验 `SampleManifest.split`，并向 Visual Router 与 TimeFuse
提供一致的样本顺序。

它应承担：

- vali/test、train/vali/test、held-out TSF cell、cross-cell 或其它 split 策略；
- split 覆盖率、互斥性和顺序稳定性校验；
- 按 split 输出 ordered `sample_keys`；
- 将 split lineage 写入 `SampleManifest` 或 manifest metadata。

它不应承担：

- 读取 prediction arrays；
- 读取 oracle label 或 expert error；
- 生成 Visual pseudo image / ViT embedding；
- 生成 TimeFuse 17 维 feature；
- 写 checkpoint、status、CSV summary 或 launcher artifact。

## 5. SupervisionProvider / OracleLabelProvider 最小 Contract

建议将训练监督边界命名为 `SupervisionProvider`；若短期只覆盖 oracle 标签，可以实现
`OracleLabelProvider` 作为其中一种实现。

最小 batch contract：

| 字段 | 含义 |
| --- | --- |
| `sample_keys` | 与调用方输入顺序一致 |
| `oracle_model` | 每个 sample 的 oracle top-1 专家名；可为空，取决于 metric/任务 |
| `oracle_value` | 每个 sample 的 oracle metric 值，例如最小 MAE/MSE |
| `per_model_errors` 或 `model_error_matrix` | `[sample, expert]` 的 per-model error，用于 soft oracle、diagnostic 或 upper-bound |
| `model_columns` | 与 `ExpertBatch.model_columns` 对齐 |
| `metric` | `mae`、`mse` 或后续定义的 supervision metric |
| `extra` | supervision source、version、missing report、lineage 等轻量 metadata |

关键约束：

- `SupervisionProvider` 必须按 `sample_keys + model_columns + metric` 保序返回。
- `per_model_errors` 可以由 prediction `y_pred/y_true` 复算，也可以从 oracle backend 读取；
  但接口层必须表达清楚来源，避免把 oracle backend 误当作 prediction backend。
- oracle label 只用于训练监督、诊断、baseline 或 upper-bound。
- oracle label 不进入 deployable `FeatureProvider`，不作为 Visual/TimeFuse test-time 动态特征。

P10e 已在 `time_router.protocols` 中新增最小 `SupervisionBatch` 协议类型：

- `sample_keys` / `model_columns` 使用 tuple 保序。
- `metric` 显式记录 supervision metric。
- `oracle_model`、`oracle_value`、`per_model_errors` 仍使用 `Any`，不在协议层绑定
  numpy/torch/pandas。
- `validate_shapes()` 只校验 `per_model_errors` 的 `[sample, expert]` 维度与
  `sample_keys/model_columns` 对齐，以及 `oracle_model/oracle_value` 第一维与
  `sample_keys` 对齐。
- 该类型不实现 `SupervisionProvider`，不读取 oracle SQLite/parquet，不写正式 CSV/status/checkpoint。

## 6. Prediction 与 Supervision 的区别

`ExpertProvider / ExpertBatch` 与 `SupervisionProvider` 必须长期分层：

| 边界 | 提供内容 | 允许用途 | 禁止用途 |
| --- | --- | --- | --- |
| `ExpertProvider` | `sample_keys`、`model_columns`、`y_pred`、`y_true`、row index lineage | router/fusor loss、evaluation、fusion metrics | 推导 split、读取 feature、提供 oracle label、写 run_dir artifact |
| `SupervisionProvider` | `oracle_model`、`oracle_value`、`per_model_errors`、`metric` | classification label、soft oracle、diagnostic、baseline、upper-bound | 进入 deployable feature、替代 prediction array reader、写正式 CSV/status/checkpoint |
| `FeatureProvider` | Visual 或 TimeFuse 可部署特征 | router/fusor input | 读取 oracle、expert error、未来 `y` 或 test-time 不可用字段 |

因此，Visual Router 的 `fusion_huber_kl` 可以在训练期使用 supervision，但不能把
`per_model_errors` 混入 Visual feature；TimeFuse-style fusor 可以用 oracle 做诊断或 baseline，
但 17 维 feature cache 仍只能来自历史窗口 `x`。

## 7. 当前两条路线的历史差异

当前实现差异是历史实现路径，不是长期接口边界。

Visual Router 当前：

- labels CSV 同时承担 sample manifest、split、oracle supervision 和 metadata。
- prediction SQLite 只负责专家预测。
- `fusion_huber_kl` 的 expert error 当前在入口中从 prediction batch 复算。

TimeFuse-style fusor 当前：

- feature CSV 同时承担 sample source 和 feature source。
- oracle SQLite / parquet 承担 supervision。
- prediction SQLite 承担专家预测。
- reader 当前把 feature、oracle、prediction 和 training batch 组装混在一起。

长期应收敛为：

- `SampleManifest` 提供 sample source、split 和顺序。
- `FeatureProvider` 各自提供 Visual / TimeFuse 特征。
- `ExpertProvider` 提供专家预测和共享真实值。
- `SupervisionProvider` 提供 oracle / error supervision。

## 8. 允许重跑 Stage 1 后的新 canonical 路线

后续若重跑 Stage 1，建议按以下顺序冻结新底座：

1. 生成或冻结 canonical `SampleManifest`，先确认 `sample_key`、split、config、dataset、
   item/channel/window 和 seq/pred length lineage。
2. 由 `SampleManifest` 驱动 prediction backend，构建 `(sample_key, model_name)` prediction records。
3. 由 `SampleManifest` 驱动 supervision backend，生成或读取 `metric` 维度下的 oracle/error labels。
4. 由 `SampleManifest` 驱动 feature provider：Visual 在线 `x -> pseudo image -> ViT`，TimeFuse
   `sample_key -> feature tensor` 或在线 feature computation。
5. Visual Router 与 TimeFuse-style fusor 都以同一套 ordered `sample_keys` 为主索引训练和评估。
6. P11/P12 冻结新的 run artifact schema；旧 labels/feature/oracle/prediction artifact schema
   只作为迁移来源和复现材料，不再强制成为新 contract。

## 9. P10e smoke 验收

新增 `tests/smoke/stage1_sample_supervision_protocol_smoke.py`，覆盖：

- 构造 4 个 `SampleManifestRow`，包含 `vali/test` split。
- 校验 `sample_key` 唯一、split 过滤保序和 `split_counts()`。
- 分别构造 vali/test 两个 `SupervisionBatch`。
- 使用 5 个 `model_columns` 和小型 numpy `per_model_errors` 校验 shape、metric 与 oracle 输出。
- 故意构造重复 `sample_key`、专家维 shape mismatch 和 oracle shape mismatch，确认错误信息清晰。

P10e 验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
```

## 10. 本阶段明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增正式 provider 代码。
- 不实现 `SupervisionProvider` / `OracleLabelProvider` 正式读取逻辑。
- 不修改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。

## 11. 后续

P10e 后优先进入 P11/P12 schema 冻结设计或小规模 manifest/supervision fixture 设计。
真正实现前应先决定：

- `SampleManifest` 的物理存储格式和版本号；
- `SplitStrategy` 如何 materialize / validate split；
- `SupervisionProvider` 的 metric 维度、缺失策略和 per-model error 来源；
- Visual Router / TimeFuse-style fusor 现有历史 artifact 如何迁移到新 canonical schema。
