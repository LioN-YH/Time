# Stage 1 P11b Canonical SampleManifest Physical Schema

创建日期：2026-06-20

## 1. 目标

本文冻结 Stage 1 canonical `SampleManifest` 的物理存储 schema、schema version、
`inputs/split_summary.json` schema，以及 canonical `run_dir/inputs/` 中保存 manifest
snapshot 或 reference 的方式。

本阶段只做文档冻结，不迁移正式入口，不新增 launcher/scripts，不访问 `/data2`，不启动
small/pressure/full-scale，不改变 legacy CSV / summary / metadata / status / checkpoint schema。

## 2. 核心原则

- `SampleManifest` 是样本身份、split 和顺序的 canonical source。
- `SplitStrategy` 负责生成或校验 split，并输出 ordered `sample_keys`。
- split 不再由 Visual labels CSV、TimeFuse feature CSV、oracle reader 或 prediction reader
  各自推导。
- `SampleManifest` 只保存样本身份、split 和轻量 lineage。
- 17 维 TimeFuse feature 不进入 `SampleManifest`。
- oracle label、oracle value 和 per-model error 不进入 `SampleManifest`。
- prediction cache、SQLite index 和 packed npy path 是 implementation，不进入
  `SampleManifest` interface。
- `run_dir` 属于 Runtime；Provider 不知道 `run_dir`，也不决定 manifest snapshot/reference
  策略。
- 不为了兼容 legacy CSV 形状污染 canonical schema。
- 旧版 `96_48_S` full-scale 结果只作为 sanity reference。

## 3. Schema Version

P11b 冻结两个最小版本号：

| schema | version |
| --- | --- |
| `SampleManifest` physical schema | `stage1_sample_manifest_v1` |
| split summary schema | `stage1_split_summary_v1` |

升级条件：

- 必需字段增加、删除、重命名或语义变化；
- `sample_key` 生成规则变化；
- `split` 语义变化，例如默认互斥策略或 held-out 语义改变；
- ordered `sample_keys` 语义变化，例如排序来源不再是 manifest 行顺序；
- `lineage` 字段语义变化，例如开始承载非轻量来源信息；
- snapshot/reference artifact 的恢复语义发生变化。

不因以下变化升级 schema：

- 新增可选 lineage 子字段且不改变现有字段语义；
- 存储介质从 CSV 换成 Parquet，只要字段和顺序语义不变；
- Runtime metadata 增加额外摘要字段；
- 内存 dataclass helper 增加不改变磁盘 contract 的便利方法。

P11b 不设计复杂 registry 或 migration framework；版本字段和本文档足够支撑 Stage 1
小步迁移。

## 4. SampleManifest 物理字段

`stage1_sample_manifest_v1` 的最小物理字段如下：

| 字段 | 必需性 | 类型建议 | 说明 |
| --- | --- | --- | --- |
| `sample_key` | 必需 | string | canonical join key，必须稳定、唯一，可跨 prediction / supervision / feature / evaluation join |
| `split` | 必需 | string | 来自 `SplitStrategy` 或经 `SplitStrategy` 校验，例如 `train`、`vali`、`test`、`heldout` |
| `config_name` | 必需 | string | 例如 `96_48_S` |
| `dataset_name` | 必需 | string | Quito dataset 或等价数据集名称 |
| `item_id` | 必需 | string | item / series 标识 |
| `channel_id` | 必需 | string | 单变量或多变量 channel 标识 |
| `window_index` | 必需 | integer | 同一 `item_id + channel_id` 下的窗口顺序 |
| `seq_len` | 必需 | integer | 历史窗口长度 |
| `pred_len` | 必需 | integer | 预测长度 |
| `lineage` | 必需 | object/string | 轻量来源信息；推荐 JSON object，CSV 介质可保存为 JSON string |

字段约束：

- `sample_key` 在单个 manifest 内必须唯一。
- manifest 行顺序是默认 ordered `sample_keys` 来源；Runtime 和 Provider 不应隐式重排。
- `split` 必须由 `SplitStrategy` 生成或校验，不能由 downstream reader 自行猜测。
- `window_index` 表达同一 `item_id + channel_id` 下的窗口顺序，不表达全局行号。
- `seq_len` / `pred_len` 写入 manifest，避免同一 manifest 支持多 config 时只能从外部配置反推。
- `lineage` 只能保存轻量来源信息，例如 `source_kind`、`source_schema_version`、
  `source_row_id`、`source_shard`、`created_by`、`created_at`。
- `lineage` 不得保存大数组、TimeFuse 17 维 feature、oracle label/value、per-model error、
  prediction cache path、SQLite index path、checkpoint path 或 run_dir。

推荐物理介质：

- 小规模 smoke、canonical rerun 或人工审计可保存 CSV / JSONL / Parquet snapshot；
- full-scale 推荐 Parquet 或可分片表格式，Runtime 可以只在 `inputs/` 保存 reference；
- SQLite 可以作为 Runtime/index implementation，但不是 `SampleManifest` interface 的唯一物理形态。

## 5. sample_key 规则

`sample_key` 是 Stage 1 canonical join key：

- Visual Router 和 TimeFuse-style fusor 共用同一生成规则；
- prediction、supervision、feature 和 evaluation 都必须以它 join；
- `sample_key` 不依赖文件路径、`run_dir`、cache path、Bash 环境、GPU 分配或 launcher 名称；
- `sample_key` 生成必须由样本身份字段和稳定 config/window 信息决定；
- 同一 manifest 中 `sample_key` 必须唯一，重复时应在 manifest 校验阶段失败。

当前 Stage 1 物理实现可以继续沿用 legacy stable key，只要它能由
`config_name + dataset_name + item_id + channel_id + window_index + seq_len + pred_len`
等稳定字段重建或校验。该 legacy stable key 是当前 Stage 1 物理实现，不声明为永远不可变的
全局规则；若后续改变生成规则，必须升级 `sample_manifest_schema_version` 并在
`split_summary.json` 与 `run_metadata.json` 中记录。

## 6. Split Summary Schema

Runtime 应在 canonical `run_dir/inputs/split_summary.json` 写出 split summary。它是
Runtime/input artifact，不是 Provider interface。

`stage1_split_summary_v1` 最小字段：

| 字段 | 必需性 | 说明 |
| --- | --- | --- |
| `split_summary_schema_version` | 必需 | 固定为 `stage1_split_summary_v1` |
| `split_strategy_name` | 必需 | 生成或校验 split 的策略名 |
| `config_name` | 必需 | 例如 `96_48_S` |
| `split_names` | 必需 | 有序 split 名称列表 |
| `sample_count_by_split` | 必需 | 每个 split 的样本数 |
| `unique_sample_key_count` | 必需 | manifest 中唯一 sample_key 数 |
| `duplicate_sample_key_count` | 必需 | 重复 sample_key 数；canonical run 应为 0 |
| `split_overlap_check` | 必需 | split 间互斥检查结果和违规数量 |
| `ordered_sample_keys_policy` | 必需 | ordered sample_keys 的来源，例如 `manifest_row_order` |
| `source_manifest_reference` | 必需 | 指向 snapshot 或 reference artifact 的轻量引用 |
| `created_at` | 必需 | Runtime 写出时间 |

`split_overlap_check` 推荐结构：

```json
{
  "default_policy": "mutually_exclusive",
  "allowed_overlap": false,
  "overlap_sample_key_count": 0,
  "overlap_examples": []
}
```

约束：

- split 间样本默认互斥，除非未来协议显式允许并升级或扩展 split 语义。
- ordered `sample_keys` 是 `ExpertProvider`、`SupervisionProvider`、`FeatureProvider` 和
  Evaluator 的共同顺序来源。
- `split_summary.json` 不承载完整 sample manifest 大表。
- `source_manifest_reference` 只描述 manifest snapshot/reference 的位置、schema、checksum
  和 row count 摘要，不把 prediction/cache/index 路径混入 manifest interface。

## 7. run_dir/inputs/ Manifest 保存方式

canonical `run_dir/inputs/` 允许两种 manifest 保存方式。

### 7.1 snapshot

小规模 smoke、debug run 或 canonical rerun 可保存完整快照：

```text
run_dir/inputs/sample_manifest.csv
run_dir/inputs/sample_manifest.metadata.json
run_dir/inputs/split_summary.json
```

snapshot 要求：

- 快照文件包含 `stage1_sample_manifest_v1` 的所有必需字段；
- metadata 至少记录 `sample_manifest_schema_version`、row count、checksum、created_at 和
  ordered sample_keys policy；
- snapshot 必须能恢复完整 ordered `sample_keys`。

### 7.2 reference

full-scale run 可保存 manifest reference：

```text
run_dir/inputs/sample_manifest_ref.json
run_dir/inputs/split_summary.json
```

`sample_manifest_ref.json` 最小字段：

| 字段 | 必需性 | 说明 |
| --- | --- | --- |
| `sample_manifest_schema_version` | 必需 | 固定为 `stage1_sample_manifest_v1` |
| `reference_type` | 必需 | 例如 `path`、`uri` 或未来受控类型 |
| `path` | 必需 | manifest artifact 路径或 URI；由 Runtime 记录 |
| `checksum` | 必需 | manifest 内容校验摘要 |
| `checksum_algorithm` | 必需 | 例如 `sha256` |
| `row_count` | 必需 | manifest 行数 |
| `ordered_sample_keys_policy` | 必需 | 例如 `manifest_row_order` |
| `created_at` | 必需 | reference 写出时间 |

reference 要求：

- reference 必须能恢复 ordered `sample_keys`；
- `run_metadata.json` 只记录 manifest 摘要和引用，不承载完整大表；
- Provider 不决定 snapshot/reference 策略，也不解析 `run_dir/inputs/`；
- Runtime 可以把 reference resolve 成 provider 可消费的 manifest handle，但 provider 看到的是
  显式 sample_keys、split 或 backend handle。

## 8. Visual / TimeFuse 映射策略

### 8.1 Visual labels -> SampleManifest

Visual labels source 迁移到 canonical manifest 时，只抽取：

- `sample_key`；
- `split`；
- `config_name`；
- `dataset_name`；
- `item_id`；
- `channel_id`；
- `window_index`；
- `seq_len`；
- `pred_len`；
- 轻量 `lineage`，例如 source row id、source shard、source schema version。

Visual labels 中的 oracle model、oracle value、expert error、loss target、diagnostic metric
不进入 `SampleManifest`。这些信息进入 `SupervisionProvider`、evaluation diagnostics 或
legacy reference artifact。

### 8.2 TimeFuse feature/oracle -> SampleManifest

TimeFuse feature/oracle source 迁移到 canonical manifest 时，只抽取：

- `sample_key`；
- `split`；
- `config_name`；
- `dataset_name`；
- `item_id`；
- `channel_id`；
- `window_index`；
- `seq_len`；
- `pred_len`；
- 轻量 `lineage`，例如 feature shard、feature schema version、source row id。

17 维 TimeFuse feature 值属于 `FeatureProvider`，不进入 `SampleManifest.lineage` 或额外字段。
oracle label、oracle value 和 per-model error 属于 `SupervisionProvider`，不进入
`SampleManifest`。

## 9. 与 P11a run artifact schema 的关系

P11a 冻结的是 `run_dir` 目录职责；P11b 冻结的是 `inputs/` 中 manifest 与 split summary 的
物理 contract。

`run_metadata.json` 应只保存 manifest 摘要，例如：

```json
{
  "sample_manifest_schema_version": "stage1_sample_manifest_v1",
  "sample_manifest": {
    "storage_mode": "reference",
    "row_count": 123,
    "checksum_algorithm": "sha256",
    "checksum": "..."
  }
}
```

完整大表放在 `inputs/sample_manifest.*` 或由 `inputs/sample_manifest_ref.json` 引用。
`split_summary.json` 与 manifest snapshot/reference 一起构成本次 run 的 sample/split 输入证据。

## 10. P11b 明确不做

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
- 不把 feature、oracle/error、prediction cache path 塞进 `SampleManifest`。
- 不声称正式入口已迁移。

## 11. 后续连接

- P11c 可设计最小 Runtime artifact writer/helper，用于写出 `run_metadata.json`、
  `run_status.json`、`inputs/sample_manifest_ref.json` 和 `inputs/split_summary.json`，但必须继续保持
  Provider / Head / Evaluator 不知道 `run_dir`。
- 正式入口迁移前仍需审计真实 full-scale Visual labels schema 与 TimeFuse feature/oracle
  schema，并把字段映射到本文冻结的 canonical physical schema。
- canonical pipeline 后续需要重跑，legacy `96_48_S` full-scale 输出只作为 sanity reference。
