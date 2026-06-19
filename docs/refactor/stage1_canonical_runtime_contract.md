# Stage 1 Canonical Runtime Contract

创建日期：2026-06-19

## 1. 目标

本文定义 P5a 阶段的 Stage 1 canonical runtime contract，用于后续 Visual Router 主线和 TimeFuse-style fusor baseline 支线逐步迁移到统一运行目录、状态文件、元数据、checkpoint、日志和评估输出契约。

本次只写文档，不修改训练脚本，不迁移入口，不实现 config system、checkpoint index helper 或 logging framework，不接入 `/data2`，不移动或删除历史代码。

## 2. 适用范围

该 contract 只适用于未来新 canonical runtime 产物，不回写历史正式输出目录，也不要求历史 pilot、legacy、OOM 残留目录或非 streaming 入口补齐字段。

适用对象：

- Visual Router canonical entrypoint：`train_visual_router_online_streaming.py` 的未来 runtime 化版本。
- TimeFuse-style fusor baseline canonical entrypoint：`train_timefuse_fusor_streaming.py` 的未来 runtime 化版本。
- 未来围绕两条 canonical entrypoint 的 launcher、eval-only、train-only、calibration/report 子流程。

不适用对象：

- LogisticRegression hard-label router。
- offline ViT embedding cache builder/launcher。
- 旧 OOM lookup 路线和失败目录。
- `pilot/` 固定规模脚本。
- 非 streaming / 早期 online full-scale 入口。
- prediction cache builder 的 cache resume schema。

## 3. `run_dir` 最小结构

新 canonical run 至少采用以下结构：

```text
run_dir/
├── status.json
├── metadata.json
├── checkpoints/
│   ├── latest_checkpoint_index.json
│   └── ...
├── logs/
│   ├── main.log
│   └── ...
├── evaluation/
│   ├── summary.*
│   ├── comparison.*
│   └── ...
└── predictions/  # 或 prediction_outputs/
    └── ...
```

目录职责：

| 路径 | 必需性 | 职责 |
| --- | --- | --- |
| `status.json` | 必需 | 当前运行状态、阶段、进度、latest checkpoint、错误和可恢复信息 |
| `metadata.json` | 必需 | 运行静态说明、entrypoint、config、输入输出、schema、资源和 lineage |
| `checkpoints/` | 训练型入口必需 | epoch checkpoint、latest checkpoint 和 latest index；纯 eval-only 可为空但 metadata 必须说明 checkpoint 来源 |
| `logs/` | 必需 | 主日志、launcher 日志、lane 日志或后台接手信息；兼容期可同时保留根级 `main.log` |
| `evaluation/` | 评估型入口必需 | summary、comparison、calibration、selected counts、diagnostic report 等聚合结果 |
| `predictions/` 或 `prediction_outputs/` | 评估型入口必需 | per-sample prediction rows、router weights、soft fusion rows 等大表输出 |

`predictions/` 是推荐目录名；`prediction_outputs/` 可作为兼容别名，但同一新 run 内应只选一种主目录，并在 `metadata.json.outputs` 中明确。

## 4. `status.json` 最小字段

新 runtime 的 `status.json` 至少包含：

| 字段 | 类型 | 必需性 | 说明 |
| --- | --- | --- | --- |
| `status` | string | 必需 | `running`、`completed`、`failed`、`stopped` 等顶层状态 |
| `phase` | string | 必需 | 当前阶段，例如 `init`、`preflight`、`index`、`scaler`、`train`、`checkpoint_saved`、`eval`、`done` |
| `updated_at` | string | 必需 | 更新时间；新 runtime 推荐 UTC ISO 字符串，兼容展示可另写本地时间字段 |
| `run_dir` | string | 必需 | 当前 run 根目录绝对路径或 repo/root 相对可解析路径 |
| `entrypoint` | string | 必需 | 入口脚本或 runtime entrypoint 名称 |
| `config_name` | string | 必需 | 当前 Stage 1 config，例如 `96_48_S` |
| `progress` | object | 必需 | 当前进度对象；可包含 epoch、shard、batch、sample、split、phase-specific counters |
| `latest_checkpoint_path` | string/null | 必需 | 当前 latest checkpoint；无 checkpoint 时为 null |
| `error` | object/null | 必需 | 失败时记录 error type、message、traceback path 或简短 traceback；非失败时为 null |

推荐 `progress` 子字段：

| 子字段 | 说明 |
| --- | --- |
| `completed_epochs` | 已完成 epoch 数；新 runtime 推荐统一复数命名 |
| `current_epoch` | 当前 epoch，从 1 开始或在 metadata 中明确约定 |
| `completed_shards` | 已完成 shard 数 |
| `total_shards` | 总 shard 数 |
| `current_shard` | 当前 shard id |
| `train_samples` | 已处理 train/vali 样本数 |
| `eval_samples` | 已处理 eval/test 样本数 |
| `batches` | 当前阶段 batch 计数 |
| `metrics_preview` | 可选轻量指标预览；正式指标仍以 `evaluation/` 输出为准 |

`status.json` 是运行时 mutable 文件，只表达当前状态；完整 lineage、输入输出和 schema 不应只存在于 status 中。

## 5. `metadata.json` 最小字段

新 runtime 的 `metadata.json` 至少包含：

| 字段 | 类型 | 必需性 | 说明 |
| --- | --- | --- | --- |
| `stage` | string | 必需 | 固定为 Stage 1 相关标识，例如 `stage1_vali_test_router` |
| `entrypoint` | string | 必需 | 入口脚本或 runtime entrypoint 名称 |
| `config_name` | string | 必需 | 当前 config |
| `args` | object | 必需 | CLI 参数或 runtime 参数快照；Path 应序列化为字符串 |
| `inputs` | object | 必需 | sample manifest、prediction cache、oracle labels、TSF enrichment、feature cache、checkpoint 等输入 |
| `outputs` | object | 必需 | checkpoint、prediction、evaluation、logs、status、metadata 等输出路径 |
| `model_columns` | list[string] | 必需 | 固定五专家顺序 |
| `array_storage` | string | 必需 | prediction array 存储口径，例如 `packed_npy_v1` |
| `feature_schema` | object | 必需 | feature provider、feature columns、shape、dtype、是否在线生成、是否落盘 |
| `split_strategy` | object | 必需 | vali/train/test split 使用方式、训练 split、评估 split、是否 split 下推 |
| `created_at_utc` | string | 必需 | 创建时间，UTC ISO 字符串 |

推荐补充字段：

- `runtime_contract_version`：例如 `stage1_runtime_contract_v1`。
- `branch`：`visual_router` 或 `timefuse_fusor`。
- `resources`：device、CUDA_VISIBLE_DEVICES、DataParallel、num_workers、batch_size。
- `checkpoint`：checkpoint naming、latest index path、resume checkpoint、checkpoint payload version。
- `evaluation_schema`：summary、comparison、prediction rows 的 schema version 或字段说明。
- `git`：commit、branch、dirty 状态；P5a 只定义字段，不实现自动采集。

`metadata.json` 是运行静态说明，不应频繁覆盖进度；动态进度写入 `status.json`。

## 6. 共享字段与 Branch-Specific Extra

### 6.1 两条分支共享字段

Visual Router 与 TimeFuse-style fusor 都必须共享：

- `stage`
- `entrypoint`
- `config_name`
- `args`
- `inputs`
- `outputs`
- `model_columns`
- `array_storage`
- `feature_schema`
- `split_strategy`
- `created_at_utc`
- `status`
- `phase`
- `updated_at`
- `run_dir`
- `progress`
- `latest_checkpoint_path`
- `error`

共享字段的语义必须一致：例如 `model_columns` 永远是五专家动作空间顺序，`array_storage` 永远描述 prediction cache array 存储口径，`split_strategy` 永远描述训练和评估 split 选择。

### 6.2 Visual Router branch-specific extra

Visual Router 可在 `metadata.json.extra.visual_router` 或 `metadata.json.branch_specific` 中记录：

- `pseudo_image_preset`
- `pseudo_image_views`
- `vit_model_name`
- `vit_normalization_preset`
- `embedding_dim`
- `embedding_dtype`
- `online_embedding`
- `router_mode`
- `hidden_dim`
- `dropout`
- `huber_beta`
- `kl_tau`
- `lambda_kl`
- `scaler_policy`
- `train_only`
- `eval_only`

这些字段不应进入 TimeFuse fusor 的共享必需字段；TimeFuse 可以忽略它们。

### 6.3 TimeFuse-style fusor branch-specific extra

TimeFuse-style fusor 可在 `metadata.json.extra.timefuse_fusor` 或 `metadata.json.branch_specific` 中记录：

- `feature_columns`
- `feature_schema_name`
- `feature_cache_shards`
- `feature_only_scaler`
- `scaler_state`
- `fusor_head`
- `loss_name`
- `smooth_l1_beta`
- `prediction_num_workers`
- `prefetch_batches`
- `shard_local_index`
- `split_pushdown`
- `data_parallel`

这些字段不应进入 Visual Router 的共享必需字段；Visual Router 可以忽略它们。

## 7. P4 Helper 接入方式

P4a/P4b/P4c helper 可以作为新 runtime 的底层实现基础，但不能反向要求历史 schema 兼容。

建议接入方式：

- `atomic_write_json(...)`：写入 `status.json`、`metadata.json` 和未来 `latest_checkpoint_index.json`。
- `write_status_json(...)`：可用于最小 status 原子写，但新 runtime 需要在上层构造完整 payload；不应把 P4a 的最小 payload 当作最终 schema。
- `resolve_status_path(...)` / `resolve_metadata_path(...)`：计算 `run_dir/status.json` 和 `run_dir/metadata.json`。
- `build_run_metadata(...)`：可作为 `metadata.json` 基础构造器，但上层必须补齐 `config_name`、`model_columns`、`array_storage`、`feature_schema` 和 `split_strategy`。

明确边界：

- P4 helper 不负责创建 run name。
- P4 helper 不选择输出根或接入 `/data2`。
- P4 helper 不采集 git、CUDA、进程或训练参数。
- P4 helper 不保存 torch checkpoint。
- P4 helper 不定义 checkpoint payload。
- P4 helper 不实现 launcher、stop、resume 或 logging framework。

## 8. Checkpoint Index 最小概念

新 runtime 应保留 `checkpoints/latest_checkpoint_index.json` 的概念，但 P5a 不实现 helper。

未来最小字段建议：

| 字段 | 说明 |
| --- | --- |
| `checkpoint_index_version` | index schema version |
| `entrypoint` | 创建 checkpoint 的 canonical entrypoint |
| `branch` | `visual_router` 或 `timefuse_fusor` |
| `config_name` | 当前 config |
| `completed_epochs` | 统一使用已完成 epoch 数 |
| `checkpoint_path` | 本次 epoch checkpoint 路径 |
| `latest_checkpoint_path` | latest checkpoint 路径 |
| `updated_at` | 更新时间 |
| `checkpoint_payload_version` | torch checkpoint payload version |

边界：

- checkpoint index 只记录 latest 指针和最小恢复线索。
- 不负责 `torch.save`。
- 不负责 best checkpoint 选择。
- 不负责 resume signature 校验。
- 不替代 launcher 的 `resume.sh`、`pid.txt`、`pgid.txt`、`main.log`。
- 不回写历史 Visual Router 的 `completed_epochs` 或 TimeFuse 的 `completed_epoch` 字段差异。

## 9. 旧 Schema 舍弃边界

新 runtime 不再强兼容以下历史 schema：

- 非 streaming Visual Router 的 `visual_router_metadata.json` / `visual_router_online_metadata.json` 中只服务小规模评估摘要的字段。
- LogisticRegression hard-label router 的 legacy/deprecated 输出字段。
- offline ViT embedding cache 的 embedding manifest / `.npy` cache lineage 字段。
- 旧 OOM 目录中停留在 `running/training` 且没有 checkpoint 的 status。
- pilot launcher 的固定 120/1k/dry-run 字段。
- prediction cache builder 的 `--resume`、`resume_mode`、`checkpoint_selection` 作为 router/fusor checkpoint schema。
- TimeFuse full-scale launcher 私有的 shell script / pid / pgid 字段作为新 runtime 必需字段；未来 launcher 可以保留这些字段，但 contract 不把它们放入最小必需集。

历史目录仍可保留引用和复现价值；舍弃强兼容只表示新 runtime 不为它们新增 adapter 或 helper。

## 10. 验收与迁移门禁

P5a 只定义 contract，不迁移代码。后续任何入口迁移前至少需要：

1. 先在小规模 smoke/dry-run 新 run_dir 中写出该 contract。
2. 对 `status.json`、`metadata.json`、checkpoint index、evaluation output 做字段级检查。
3. 运行 golden smoke、oracle/TSF smoke、P4 helper smoke 和 compileall。
4. 确认没有改变 prediction cache、oracle/TSF、evaluation helper、模型结构、loss 或正式输出目录。
5. full-scale 只能在小规模契约验证后启动，且必须使用新的 run_dir，不能覆盖旧正式结果。

## 11. 本次明确不做

- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不为了兼容历史输出新增 helper。
- 不改变 `PredictionBatchReader` / `OracleTsfReader` / evaluation / IO helper 行为。
- 不改模型结构、loss 或正式输出目录。
