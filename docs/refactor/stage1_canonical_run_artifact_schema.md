# Stage 1 Canonical Run Artifact Schema

创建日期：2026-06-20

## 1. 目标

本文冻结 P11a 阶段的 Stage 1 canonical run artifact schema，回答新版
canonical pipeline 跑起来后 `run_dir` 中应该放什么、怎么分层、由谁负责写出，以及
哪些组件不应该知道这些路径。

本阶段只做文档冻结，不修改正式入口，不新增 launcher/scripts，不启动实验，不访问
`/data2`，不改变当前 legacy entrypoint 的实际输出 schema。

## 2. 设计原则

- `run_dir` 属于 Runtime，不属于 Provider。
- Provider 不持有、不解析、不硬编码 `run_dir`。
- Bash 属于 `exp_scripts` 操作层，不进入 `time_router`。
- `time_router` 不知道 Bash，也不硬编码 `/data2`。
- cache 是 implementation，不是 interface。
- `ExpertProvider` / `ExpertBatch` 是长期专家输出边界。
- metadata 偏静态，描述本次 run 是什么。
- status 偏动态，描述本次 run 进行到哪里、是否成功、失败、中断或恢复。
- `evaluation/` 承载 summary、comparison、selected counts、diagnostic metrics 等聚合评估输出。
- `predictions/` 承载 per-sample prediction rows / fusion rows / router output rows。
- 不再新增 `summaries/`，避免与 `evaluation/` 边界重叠。
- canonical `run_dir` 结构保持最小、清晰、可扩展，避免过深目录树。

## 3. 推荐 `run_dir` 结构

```text
run_dir/
├── run_metadata.json
├── run_status.json
├── inputs/
├── indexes/
├── predictions/
├── evaluation/
├── checkpoints/
└── logs/
```

与 P5a `stage1_canonical_runtime_contract.md` 的关系：

- P5a 的 `metadata.json` / `status.json` 是早期最小 runtime contract 名称。
- P11a 推荐新 canonical runtime 使用 `run_metadata.json` / `run_status.json`，让静态与动态文件从文件名上更明确。
- 后续若存在兼容期，可由 Runtime 同步写 alias 或在 migration 文档中声明映射；Provider、Head、Evaluator 不关心文件名。
- 本文定义 future canonical artifact schema，不反向要求 legacy entrypoint 改名或补齐字段。

## 4. 顶层文件职责

### 4.1 `run_metadata.json`

`run_metadata.json` 是静态描述文件，记录本次 run 是什么、使用哪些输入、遵循哪些协议和 schema。
它可以在 run 初始化后写入，若需要追加最终输出摘要，应由 Runtime 明确写入并保持字段语义稳定。

最小字段建议：

| 字段 | 说明 |
| --- | --- |
| `run_artifact_schema_version` | 本文定义的磁盘产物 schema 版本，例如 `stage1_run_artifact_v1` |
| `protocol_version` | 本次运行遵循的 canonical protocol 版本 |
| `sample_manifest_schema_version` | `SampleManifest` 物理或快照 schema 版本 |
| `supervision_schema_version` | `SupervisionBatch` 来源或磁盘监督 schema 版本 |
| `prediction_backend_version` | prediction backend / cache / index 实现版本 |
| `evaluation_schema_version` | `evaluation/` 聚合输出 schema 版本 |
| `git` | commit、branch、dirty 状态；采集由 Runtime 负责 |
| `config_name` | 例如 `96_48_S` |
| `branch_name` | `visual_router`、`timefuse_fusor` 或后续分支名 |
| `created_at` | run 创建时间 |
| `inputs` | 输入引用摘要；详细 spec 放入 `inputs/` |
| `environment` | Python、conda、torch、CUDA、设备等环境摘要 |
| `runtime` | entrypoint、launcher、资源策略、run_dir、output_root 等 Runtime 信息 |

边界：

- metadata 不记录 batch 级进度。
- metadata 不替代 `inputs/` 中的 resolved config 或 protocol spec。
- metadata 可以引用 checkpoint/evaluation/prediction 输出，但不承载大表。

### 4.2 `run_status.json`

`run_status.json` 是动态状态文件，由 Runtime 在运行中原子更新。

最小字段建议：

| 字段 | 说明 |
| --- | --- |
| `status` | `pending`、`running`、`completed`、`failed`、`interrupted`、`resumed` |
| `current_stage` | `init`、`preflight`、`indexing`、`scaler_fit`、`training`、`evaluation`、`finalizing` 等 |
| `updated_at` | 最近更新时间 |
| `failure_reason` | 失败或中断原因；成功时为 null |
| `checkpoint_pointer` | latest/best/resume checkpoint 指针摘要 |
| `progress` | epoch、split、shard、batch、sample 等轻量进度 |
| `resume` | 是否从 checkpoint 或 partial artifact 恢复、恢复来源和恢复时间 |

状态语义：

- `pending`：run_dir 已创建但主体任务尚未开始。
- `running`：主体任务正在执行。
- `completed`：所有必需 artifact 写出且校验通过。
- `failed`：运行失败且需要人工处理。
- `interrupted`：运行被停止，但可能保留可恢复状态。
- `resumed`：本次 run 曾经从已有 checkpoint/artifact 恢复；可作为历史事件或状态子字段记录。

边界：

- status 可以频繁覆盖。
- status 不保存静态 lineage 的唯一副本。
- status 不替代 launcher 的 PID/PGID 文件；进程接手信息放 `logs/`。

## 5. 子目录职责

### 5.1 `inputs/`

`inputs/` 保存本次 run 的输入协议、解析后配置和输入引用，便于复核“这次 run 是怎么构造的”。

建议内容：

- protocol spec。
- resolved config。
- `SampleManifest` 引用或快照。
- supervision source/schema。
- split summary。
- feature provider config reference。
- expert provider / prediction backend config reference。
- evaluator provider config reference。

边界：

- `inputs/` 可以保存轻量快照或引用，不要求复制大规模 cache。
- `inputs/` 不保存 Runtime 运行进度。
- `SampleManifest` 的 ordered sample_keys 应可从这里引用或重建。

### 5.2 `indexes/`

`indexes/` 保存 Runtime 为本次 run 准备的索引类 artifact。

建议内容：

- prediction SQLite index。
- prediction backend metadata。
- future supervision index metadata。
- shard-local 或 split-specific index metadata。
- 其他 runtime/cache/debug index artifact。

边界：

- index 是 Runtime/cache implementation，不是 Provider interface。
- Provider 可以消费 Runtime 传入的 backend handle 或 config，但不自行决定 index 落盘路径。
- index metadata 应足够说明 sample_key 覆盖、专家完整性、array storage 和构建时间。

### 5.3 `predictions/`

`predictions/` 保存逐样本输出和融合行级结果。

建议内容：

- per-sample prediction rows。
- hard top-1 rows。
- raw soft fusion rows。
- router output rows，例如 logits/weights。
- sample-level diagnostic rows。

边界：

- `predictions/` 负责 sample-level 大表。
- `evaluation/` 负责聚合指标和报告。
- hard/raw-soft 逐样本结果可以写在 `predictions/`；它们的汇总 MAE/MSE、selected counts、entropy 均值等写在 `evaluation/`。

### 5.4 `evaluation/`

`evaluation/` 保存聚合评估输出和可读报告。

建议内容：

- evaluation summary。
- comparison。
- selected counts。
- entropy / max weight 等 diagnostic metrics。
- branch-specific diagnostic metrics。
- `evaluation_report.md`。

边界：

- `evaluation/` 不保存完整逐样本 prediction 大表，避免与 `predictions/` 重叠。
- `summaries/` 不作为新 canonical 目录；summary 统一归入 `evaluation/`。
- Evaluator 负责生成内存评估对象或结构化结果；文件写出由 Runtime / artifact writer 负责。

### 5.5 `checkpoints/`

`checkpoints/` 保存训练状态、模型权重和 checkpoint metadata。

建议内容：

- latest checkpoint pointer。
- epoch checkpoint。
- best checkpoint。
- checkpoint metadata。
- checkpoint index，例如 `latest_checkpoint_index.json` 或后续 canonical 名称。

边界：

- checkpoint 保存策略属于 Runtime/training 层。
- RouterHead 不自行决定 checkpoint 文件名或路径。
- eval-only run 可以没有新 checkpoint，但 metadata/status 必须说明 checkpoint 来源。

### 5.6 `logs/`

`logs/` 保存可复现命令、运行日志、环境 dump 和接手信息。

建议内容：

- `command.sh`。
- stdout/stderr。
- launcher handoff info。
- environment dump。
- notes。
- PID/PGID、stop/resume 命令或对这些文件的引用。

边界：

- Bash launcher 只在操作层生成命令和后台管理信息。
- `time_router` 不解析 Bash 脚本，不假设 `/data2`，不通过 `logs/` 反推运行配置。

## 6. 最小 Versioning Strategy

P11a 只定义最小版本字段，不设计复杂 registry。

必需版本字段：

| 字段 | 所属位置 | 说明 |
| --- | --- | --- |
| `run_artifact_schema_version` | `run_metadata.json` | 磁盘 run artifact schema 版本 |
| `protocol_version` | `run_metadata.json` / `inputs/protocol_spec.*` | canonical pipeline protocol 版本 |
| `sample_manifest_schema_version` | `run_metadata.json` / `inputs/` | SampleManifest 物理或快照 schema 版本 |
| `supervision_schema_version` | `run_metadata.json` / `inputs/` | supervision source/schema 版本 |
| `prediction_backend_version` | `run_metadata.json` / `indexes/` | prediction backend 或 cache/index 版本 |
| `evaluation_schema_version` | `run_metadata.json` / `evaluation/` | evaluation summary/comparison/diagnostic 输出版本 |

约束：

- 版本字段描述磁盘 artifact 与运行协议，不把 runtime 文件格式和 protocol dataclass 强绑定死。
- `ExpertBatch`、`RouterOutput`、`EvaluationInput` 是内存协议对象，不等同于磁盘 schema。
- dataclass 字段调整不自动等于 run artifact schema 升级；只有磁盘产物字段、目录职责或跨组件 contract 变化时才升级 artifact schema。
- 暂不引入 registry、migration framework 或全局 schema catalog；文档和 metadata 字段足以支持 Stage 1 小步迁移。

## 7. Runtime 与 Provider 边界

Runtime 负责：

- 创建和选择 `run_dir`。
- 解析 CLI/config、output_root、launcher 传入路径和资源策略。
- 写 `run_metadata.json`、`run_status.json`。
- 准备 `inputs/`、`indexes/`、`checkpoints/`、`logs/`。
- 将 Evaluator 结果写入 `evaluation/`。
- 将逐样本 rows、fusion rows、router output rows 写入 `predictions/`。
- 处理 checkpoint/resume、失败、中断和接手信息。

Provider 负责：

- 按显式 sample_keys、split 或 backend handle 返回内存 batch。
- 保持 `sample_keys` 与 `model_columns` 保序。
- 返回必要 lineage metadata 给 Runtime 或 Evaluator 使用。

Provider 不负责：

- 创建、解析或硬编码 `run_dir`。
- 写 status、metadata、checkpoint、logs。
- 决定 `/data2`、repo 内输出根或 Bash launcher 结构。
- 把 cache 路径暴露成 interface 本身。

## 8. Branch-Specific Artifact Policy

### 8.1 共用层

两条分支共用：

- `SampleManifest`。
- `SplitStrategy` semantics。
- prediction backend / `ExpertBatch`。
- `SupervisionBatch` / `SupervisionProvider` contract。
- `EvaluationInputAdapter` / Evaluator metrics。
- Runtime artifact contract。

### 8.2 Visual-specific

Visual Router 可在 metadata、inputs 或 evaluation diagnostics 中记录：

- Quito history window。
- pseudo image / ViT feature artifacts。
- `VisualMLPRouterHead`。
- `fusion_huber_kl` / classification objective。
- online visual router diagnostics。

约束：

- pseudo image tensor 和 ViT embedding 默认是 batch runtime 中间态，不作为长期 canonical cache。
- Visual-specific artifact 不进入 TimeFuse-style 必需 schema。

### 8.3 TimeFuse-style-specific

TimeFuse-style fusor 可在 metadata、inputs、indexes 或 evaluation diagnostics 中记录：

- 17 维 feature cache。
- feature-only scaler。
- `TimeFuseLinearSoftmaxHead`。
- SmoothL1 weighted fusion objective。
- hard/raw-soft evaluation diagnostics。

约束：

- 17 维 feature cache 是当前 implementation，不是全局 FeatureProvider interface。
- scaler 属于 training/runtime 行为，不属于 ExpertProvider。
- TimeFuse-specific artifact 不进入 Visual Router 必需 schema。

## 9. Legacy `96_48_S` Full-Scale Policy

旧版 `96_48_S` full-scale 输出只作为 sanity reference。

明确策略：

- 不作为 canonical artifact schema 的兼容目标。
- canonical 结果写入新的 `output_root / run_dir`。
- 后续重跑 `96_48_S` 时，只记录指标量级和运行经验差异。
- 不要求逐文件 schema 对齐。
- 旧版结果不驱动新代码接口设计。
- 当前 legacy entrypoint 的 CSV、summary、metadata、status、checkpoint schema 本阶段不修改。

因此，P11a 可以定义 future canonical artifact schema，但不得要求现有
`train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 或
`launch_timefuse_fusor_full_scale.py` 立即改写输出。

## 10. P11a 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不修改当前 legacy entrypoint 的实际输出。
- 不新增 provider/head/runtime 代码。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 small/pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。
- 不引入过重 runtime 抽象、类层级或 registry。
- 不声称正式入口已迁移。
- 不声称整个 Time framework 已完成。

## 11. 后续连接

- P11b 可在本文基础上冻结 `SampleManifest` 物理存储格式、schema version 和 split summary
  写入 `inputs/` 的方式。
- P11c 可设计最小 Runtime artifact writer 或 helper，但必须先保持 provider/head/evaluator
  不知道 `run_dir` 的边界。
- 后续 scripts/launcher 接入应只把 `run_dir` 显式传给 Runtime，不把 Bash 语义下沉到
  `time_router`。
