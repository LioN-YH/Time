# Stage 1 P4d Run Artifacts 边界复核与接入规划

创建日期：2026-06-19

## 1. 目标

本文件复核 P4a/P4b/P4c 后 `time_router/io` 中 run artifacts 相关工具的架构边界，并给出后续接入正式 Visual Router / TimeFuse fusor 入口前的 integration plan。

本次只做文档化 review、public API 边界说明和后续规划；不迁移正式训练入口，不接入 `/data2` 或 full-scale 输出目录，不改变既有 `status.json` / `metadata.json` schema。

## 2. 当前模块职责

### 2.1 `prediction_cache_reader.py`

职责：

- 作为 Stage 1 prediction cache batch reader，读取 `merged_cache/manifest.csv` 或 fixture root 中的五专家 `y_pred` 与共享 `y_true`。
- 支持 `packed_npy_v1` 与 legacy `per_sample_npy`。
- 固定专家动作空间顺序，校验共享 `y_true`、row index 和 manifest MAE/MSE。

边界：

- 属于数据读取层，不属于 run artifacts writer。
- 不负责 `status.json` / `metadata.json` 写入。
- 不应与 status、metadata、path helper 混合。
- 正式 full-scale 场景仍应由训练入口或上层 streaming/shard-aware reader 控制 batch、sample_key 和资源策略。

### 2.2 `json_utils.py`

职责：

- 提供 `atomic_write_json(...)`，在目标同目录临时写入后用 `os.replace` 原子替换。
- 提供 `build_status_payload(...)` 和 `write_status_json(...)`，构造和写入最小 status payload。

边界：

- 只负责原子 JSON 写入和最小 status payload。
- 不负责路径选择。
- 不负责训练状态采集。
- 不负责 launcher / monitor / resume 语义。
- 不实现 logging framework。
- 不改变既有正式 `status.json` schema。

### 2.3 `path_resolver.py`

职责：

- 提供 `find_repo_root(...)`，从指定起点向上查找 repo root marker。
- 提供 `resolve_under_root(...)`，做 root 内安全路径拼接并拒绝路径逃逸。
- 提供 `resolve_status_path(...)` 和 `resolve_metadata_path(...)`，计算 run dir 下固定文件名路径。

边界：

- 只负责 repo root 查找、root 内安全路径拼接、status/metadata path 计算。
- 不创建正式输出目录。
- 不选择实验 run name。
- 不实现 config system。
- 不访问 `/data2`。
- 不判断 full-scale 输出目录是否有效。

### 2.4 `run_metadata.py`

职责：

- 提供 `build_run_metadata(...)`，构造最小 metadata-like payload。
- 提供 `write_run_metadata(...)`，把 payload 写入调用方显式传入的 path。
- 对 `Path` / `os.PathLike` 做 JSON-safe 字符串转换。

边界：

- 不自动调用 git。
- 不读取命令行。
- 不读取训练配置。
- 不自动解析输出目录。
- 不替换既有正式 `metadata.json` schema。
- 不承担 launcher lineage、checkpoint lineage 或恢复语义的最终定义。

### 2.5 `time_router/io/__init__.py`

职责：

- 作为 `time_router.io` 的 public API 聚合入口。
- 后续正式入口迁移时，应优先从 `time_router.io` 导入稳定 public API，而不是直接依赖模块内部下划线 helper。

边界：

- `__init__.py` 只表达导出面，不应隐藏有副作用的初始化逻辑。
- 不在导入时读取配置、访问文件系统或创建输出目录。

## 3. Public API 与 private helper

当前建议稳定 public API：

- `PredictionBatchReader`
- `PredictionBatch`
- `DEFAULT_MODEL_COLUMNS`
- `atomic_write_json`
- `build_status_payload`
- `write_status_json`
- `find_repo_root`
- `resolve_under_root`
- `resolve_status_path`
- `resolve_metadata_path`
- `build_run_metadata`
- `write_run_metadata`

当前应视为 private helper 的符号：

- `path_resolver.py` 中的 `_normalize_start_path`
- `run_metadata.py` 中的 `_require_dict_or_none`
- `run_metadata.py` 中的 `_to_json_safe`
- `prediction_cache_reader.py` 内部所有下划线 helper

后续正式入口迁移时，训练脚本不应依赖 private helper。若确实需要 private helper 的能力，应先把能力重新设计为 public API，并单独补文档、smoke 和兼容性说明。

## 4. 层级边界判断

### 4.1 低风险 IO helper

以下 helper 当前属于低风险 IO helper：

- `atomic_write_json(...)`
- `build_status_payload(...)`
- `write_status_json(...)`
- `find_repo_root(...)`
- `resolve_under_root(...)`
- `resolve_status_path(...)`
- `resolve_metadata_path(...)`
- `build_run_metadata(...)`
- `write_run_metadata(...)`

低风险的前提是：调用方显式传入 path、payload、stage、inputs、outputs 等信息，helper 不自行读取训练配置、不判断正式 run 语义、不接管 launcher/resume。

### 4.2 正式训练入口 / launcher / resume 层

以下职责仍属于正式训练入口、launcher、monitor 或 resume 层，不应下沉到当前 P4 helper：

- run_dir 命名和创建。
- `/data2` 或其他 full-scale 输出根选择。
- 是否允许覆盖已有目录。
- checkpoint 保存、latest checkpoint 指针和 checkpoint index。
- resume checkpoint 选择、epoch 续接、train-only/eval-only 分支。
- `status.json` 中 launcher / monitor / resume 依赖字段的完整语义。
- `metadata.json` 中正式 lineage、命令行、配置快照和 git 信息采集。
- stdout/stderr 文件、主日志、lane 日志和 TensorBoard 等 logging framework。
- 训练配置系统和默认值解析。

## 5. 后续 integration plan

### 5.1 什么时候可以接入正式入口

只有满足以下条件后，才建议把 P4a/P4b/P4c helper 接入正式 Visual Router / TimeFuse fusor 入口：

1. 已完成 Visual Router 和 TimeFuse fusor 现有 `status.json` / `metadata.json` schema audit。
2. 已列出 launcher、monitor、resume、handoff 文档和人工监控命令实际读取的字段。
3. 已选定一个小规模 smoke 或 dry-run 入口作为第一接入点，而不是直接改 full-scale 正式目录。
4. 接入前后同一 fixture 的 `status.json` / `metadata.json` 字段语义可对比，新增字段只允许向后兼容。
5. golden smoke、oracle/TSF smoke、P4 helper smoke 和 compileall 均通过。
6. git diff 中能明确证明没有改模型结构、loss、prediction/evaluation schema 或正式输出目录含义。

在当前状态下，P4 helper 还不应直接接入正式 full-scale Visual Router / TimeFuse fusor 入口。

### 5.2 接入前需要比较的字段

接入前至少比较这些 `status.json` 字段：

- `status`
- `phase`
- `message`
- `started_at` / `started_at_utc`
- `updated_at` / `updated_at_utc`
- `completed_at` / `completed_at_utc`
- `run_dir` / `output_dir`
- `pid` / `pgid`
- `current_shard`
- `completed_shards`
- `failed_shards`
- `total_shards`
- `resume_checkpoint`
- `checkpoint_path`
- `completed_epochs`
- `train_samples`
- `vali_samples`
- `test_samples`
- `metrics` 或 summary-like 字段
- `error` / `traceback`
- `command` / `resume_command` / `stop_command` / `monitor_command`

接入前至少比较这些 `metadata.json` 字段：

- `stage` / `run_name`
- `entrypoint`
- `command`
- `config`
- `args`
- `inputs`
- `outputs`
- `git` / `git_ref` / `commit`
- `created_at` / `created_at_utc`
- `sample_manifest`
- `prediction_cache`
- `oracle_labels`
- `tsf_enrichment`
- `feature_cache`
- `checkpoint`
- `model_columns`
- `split`
- `array_storage`
- `notes`

字段名称可能随历史入口不同而不同；audit 应以现有真实输出目录为准，不以 P4 helper 的最小字段反向要求旧目录改名。

### 5.3 launcher / monitor / resume 不能随便改的字段

以下字段通常被 launcher、monitor、resume 或人工 handoff 依赖，不能在未审计前改名、删除或改变含义：

- `status`：必须能区分 `running`、`completed`、`failed`、`stopped` 等终态/非终态。
- `phase`：用于判断当前处于 preflight、index、scaler、train、eval、merge、validation 等阶段。
- `pid` / `pgid`：用于停止和健康检查后台任务。
- `current_shard`、`completed_shards`、`failed_shards`、`total_shards`：用于 shard 进度和失败恢复。
- `run_dir` / `output_dir`：用于定位主日志、checkpoint 和产物。
- `checkpoint_path`、`resume_checkpoint`、`completed_epochs`：用于续训和 eval-only。
- `command`、`resume_command`、`monitor_command`、`stop_command`：用于 handoff 后接手。
- `error` / `traceback`：用于失败复盘。

P4 helper 接入时应保持旧字段继续存在。若需要新增更规范字段，应并行写入，并在后续版本中用明确迁移文档逐步收束。

### 5.4 checkpoint index 是否作为 P4e 单独做

checkpoint index 应作为 P4e 单独做，不应并入 P4d 或当前最小 metadata helper。

原因：

- checkpoint index 会影响 resume、latest checkpoint 指针、best checkpoint 选择和 eval-only 行为。
- 它属于训练入口 / resume 层的状态机边界，风险高于纯 JSON/path helper。
- 需要单独审计 Visual Router 和 TimeFuse fusor 当前 checkpoint 命名、保存频率、best/latest 选择口径和失败恢复策略。

P4e 可以先做 checkpoint index boundary review；只有确认现有口径后，才考虑 minimal helper。

### 5.5 config system 是否现在做

config system 不建议现在实现，应等 P5/P6 前再做。

原因：

- 当前 P4a/P4b/P4c 的价值是低风险 IO 基础设施，过早引入 config system 会把路径、默认值、CLI 优先级和正式入口行为耦合到一起。
- P5 `FeatureProvider` 设计会明确 Visual 和 TimeFuse 两条路线的输入、特征和运行时参数；config 边界应在这些接口稳定后再收束。
- P6 正式入口迁移时才有足够上下文判断哪些参数应进入共享 config，哪些应保留为入口显式 CLI。

短期可以先写 config integration plan 或 schema audit，不实现配置框架。

### 5.6 如何保证旧输出目录含义不变

P4 helper 接入正式入口时，应采用以下策略：

1. 先只在新 smoke/dry-run 输出目录接入，不回写历史正式目录。
2. 旧字段保持原名和原语义；新增字段只做向后兼容扩展。
3. 接入前后对同一小规模 run 的 status/metadata 做字段级 diff。
4. launcher、monitor、resume 命令仍能使用旧字段完成启动、监控、停止和恢复。
5. 正式 `/data2` 输出目录只在完成 schema audit 后接入；首次接入应使用新 run_dir，不复用或覆盖旧完成目录。
6. 文档中明确记录新旧输出目录的引用口径，避免把 helper 接入前后的目录混读为同一 schema。

## 6. 后续 P4 小步候选

建议候选：

- P4e：checkpoint index boundary review or minimal helper。
- P4f：existing status/metadata schema audit for Visual Router and TimeFuse fusor outputs。
- P4g：logging/path/config integration plan before P6。
- 若 P4 基础设施已足够，可转入 P5 FeatureProvider interface design。

建议优先顺序：

1. P4f：先审计现有真实 status/metadata schema，降低后续误改 launcher/resume 的风险。
2. P4e：再处理 checkpoint index 边界，因为它直接影响 resume 和 eval-only。
3. P4g：在 P5/P6 前补齐 logging/path/config integration plan。
4. P5：当 IO artifacts 边界稳定后，进入 FeatureProvider interface design。

## 7. 本次明确不做

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何现有正式训练脚本。
- 不替换现有 launcher / monitor / resume 行为。
- 不接入 `/data2` 或任何 full-scale 输出目录。
- 不改变既有正式输出目录含义。
- 不改变既有 `status.json` / `metadata.json` schema。
- 不实现 checkpoint index。
- 不实现 config system。
- 不实现 logging framework。
- 不自动调用 git。
- 不自动读取命令行或训练配置。
- 不创建正式输出目录。
- 不改 evaluation helper。
- 不实现 comparison / calibration / report schema。
- 不读取 oracle/TSF。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。
- 不移动或重命名 `time_router/io` 文件。
- 不改变现有 public API 和 helper 行为。

## 8. 验证

完整 P4d 门禁包括：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

本次门禁重点不是证明正式入口已迁移，而是证明文档化边界复核没有破坏现有 reader、evaluation 和 P4 helper 的基础行为。
