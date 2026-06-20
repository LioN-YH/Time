# Stage 1 重构路线图

设计日期：2026-06-19

## 1. 执行原则

Stage 1 后续重构必须小步提交、先锁定行为再抽象共享模块。每一步提交都应保持 Visual Router 与 TimeFuse-style fusor 当前正式口径可回退，并且不得在同一提交里同时抽 reader、改 metrics、换训练入口和清理历史代码。

`tests/smoke/stage1_golden_smoke.py` 是所有迁移步骤的前后门禁：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

每次抽 reader、metrics、SQLite、output schema 之前必须先跑一次，确认基线绿色；迁移后再跑一次，确认 sample_key 顺序、五专家顺序、`y_pred/y_true` shape、packed row index、hard top-1 和 raw soft fusion 指标不漂移。若失败，停止后续迁移，先定位差异并写实验日志。

## 2. Commit 顺序

### P0：architecture docs only

目标：只提交架构设计和迁移路线图，不实现任何新 package，不新增空目录，不做 import 改动。

范围：

- 新增 `docs/refactor/stage1_target_architecture.md`。
- 新增 `docs/refactor/stage1_refactor_roadmap.md`。
- 同步维护 `WORKSPACE_STRUCTURE.md` 与中文实验日志。
- 运行 golden smoke 和 `git status`。

验收：

- `tests/smoke/stage1_golden_smoke.py` 通过。
- `git diff --name-only` 只包含文档、实验日志和结构说明。
- 正式代码零改动。

### P1：extract prediction batch reader

目标：抽出统一 prediction batch reader，使 Visual Router 和 TimeFuse-style fusor 能从同一批 sample key 读取五专家 `y_pred` 与共享 `y_true`。

当前状态（2026-06-19）：已完成本阶段的共享 reader 抽取和 golden smoke 接入；尚未迁移正式 Visual Router / TimeFuse fusor 训练入口，入口迁移仍保留到 P6。

范围：

- 读取 `per_sample_npy` 与 `packed_npy_v1`。
- 固定专家顺序 `["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]`。
- 保留 `y_true_row_index`、`y_pred_row_index` 和 manifest 指标复算校验。
- 先接入 smoke 或小规模 fixture，再接入现有入口。

门禁：

- 迁移前后运行 `tests/smoke/stage1_golden_smoke.py`。
- 对比迁移前后的 sample_key 顺序、shape、hard top-1 和 raw soft fusion。
- 不允许回退到每 sample 重复打开 packed 文件或全量 Python lookup。

本次完成范围：

- 新增 `time_router/io/prediction_cache_reader.py`，提供 `PredictionBatchReader` 与 `PredictionBatch`。
- `PredictionBatchReader` 支持 `fixture_root` 或 `manifest_path` 输入、显式 sample_key 顺序或 manifest 首次出现顺序、固定五专家 `model_columns`、共享 y_true 校验、row index 元数据和 manifest MAE/MSE 复算。
- 将 packed npy 按路径分组读取 helper 下沉到 `visual_router_experiments/common/prediction_array_io.py`，避免正式 reader 默认每 sample 重复打开 packed 文件。
- `tests/smoke/stage1_golden_smoke.py` 已改为使用共享 reader 组装 `y_pred/y_true`，并保留原有 golden 检查。

### P2：extract oracle/TSF reader

目标：统一 oracle labels 与 TSF enrichment 的批量读取，明确其监督、上限和诊断用途。

当前状态（2026-06-19）：已完成本阶段共享 oracle/TSF reader 的最小抽取和小规模 smoke；尚未迁移正式 Visual Router / TimeFuse fusor 训练入口，入口迁移仍保留到 P6。

范围：

- 按 `sample_key + metric` 读取 oracle。
- 按 `sample_key` 读取 TSF 元信息。
- 提供 required/diagnostic_only 缺失策略。
- 保证 oracle/TSF 不混入可部署 FeatureProvider 的 test-time 动态特征。

门禁：

- 迁移前后运行 golden smoke，确认 prediction/fusion 契约未被间接改动。
- 对小规模 oracle/TSF fixture 做 join 覆盖率和 sample_key 集合校验。
- 记录任何缺失策略变化，不用默认填充值掩盖数据问题。

本次完成范围：

- 新增 `time_router/data/oracle_tsf_reader.py`，提供 `OracleTsfReader` 与 `OracleTsfBatch`。
- `OracleTsfReader` 支持 `fixture_root`、显式 oracle/TSF 路径、CSV chunk 过滤、Parquet `pyarrow.dataset` 过滤、`missing_policy=error/report`、显式 sample_key 保序和一对一 join 校验。
- 新增 `tests/smoke/stage1_oracle_tsf_smoke.py`，复用 4 sample dry-run fixture 检查 oracle label、TSF metadata、join 保序、缺失报告和冲突 TSF sample_key 报错。
- 新增 `docs/refactor/oracle_tsf_reader.md`，明确 oracle/TSF 仅用于监督、上限、baseline、分层汇总或诊断，不得作为可部署 FeatureProvider 的 test-time 动态特征；full-scale 后续仍应采用 SQLite / shard-local / batch query。

### P2.5：reader hardening only

目标：只补强 `OracleTsfReader` 的小规模 smoke、文档和日志边界，不迁移正式 Visual Router / TimeFuse fusor 入口，不改变 prediction/fusion 契约。

当前状态（2026-06-19）：已完成 hardening 范围界定；本阶段只允许修改 `docs/refactor/oracle_tsf_reader.md`、`docs/refactor/stage1_refactor_roadmap.md`、`tests/smoke/stage1_oracle_tsf_smoke.py`、中文实验日志和必要结构索引。

本次补强范围：

- smoke 明确覆盖 `allow_full_scan` 默认禁止无 sample_key 全扫描。
- smoke 明确覆盖 `missing_policy=error` 对缺失 sample_key 报错。
- 文档明确禁止正式入口使用 `allow_full_scan=True`。
- 文档明确 full-scale 正式训练入口后续必须走 SQLite / shard-local / batch query 或等价批查询方案。
- oracle/TSF 仍只作为监督、上限、baseline、分层汇总或诊断使用，不进入可部署 `FeatureProvider` 或 test-time 动态调权特征。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式入口；入口迁移仍保留到 P6。
- 不改 `PredictionBatchReader` 的输出 shape、专家顺序、sample_key 顺序。
- 不改 fusion metrics、模型结构或正式输出目录。

### P3：extract metrics/fusion

目标：抽出统一 metrics、fusion 和报告基础，减少 `fusion_utils.py`、Visual Router、calibration 与 TimeFuse fusor 之间的重复实现。

范围：

- hard top-1、raw soft fusion、temperature/top-k fusion。
- MAE、MSE、oracle regret、selected counts、entropy、max weight。
- summary、comparison 和 per-sample prediction schema。

门禁：

- 迁移前后运行 golden smoke。
- 对已有 1k 或 pressure 输出做 schema golden comparison。
- 所有指标必须从同一批 `y_pred/y_true/weights` 复算，不能混用不同 reader 的中间结果。

### P3a：minimal fusion/metrics helpers only

目标：只抽取最小共享 fusion/metrics helper，让 golden smoke 通过共享模块复算既有 hard top-1、raw soft fusion、MAE 和 MSE，不迁移正式 Visual Router / TimeFuse fusor 入口。

当前状态（2026-06-19）：已完成 P3a 最小抽取并接入 `tests/smoke/stage1_golden_smoke.py`；完整 P3 的 calibration、summary、comparison、per-sample schema 收束和正式入口接入仍保留到后续小步。

本次完成范围：

- 新增 `time_router/evaluation/metrics.py` 和 `time_router/evaluation/__init__.py`。
- helper 保持纯 numpy，不引入 torch/sklearn 训练依赖。
- 函数输入显式使用 `y_pred`、`y_true`、`weights`、`model_columns`。
- `hard_top1_fusion(...)` 返回 `selected_indices`、`selected_models`、`fused_pred`、MAE 和 MSE。
- `raw_soft_fusion(...)` 保持当前 golden smoke 的线性加权口径，不做 temperature/top-k/calibration。
- `tests/smoke/stage1_golden_smoke.py` 已调用共享 helper，同时保留 golden 数值、sample_key 顺序、五专家顺序、row index 和 shape 断言。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不实现 calibration，不改变 output schema。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。

### P3b：router weight diagnostics helpers only

目标：在 P3a 最小 fusion/metrics helper 基础上，只抽取 router/fusor weights 的纯 numpy 诊断 helper，覆盖 selected counts、weight entropy 和 max weight，不迁移正式入口。

当前状态（2026-06-19）：已完成 P3b 诊断 helper 抽取并接入 `tests/smoke/stage1_golden_smoke.py` 的最小断言；完整 P3 的 calibration、summary、comparison、per-sample schema 收束和正式入口接入仍保留到后续小步。

本次完成范围：

- 在 `time_router/evaluation/metrics.py` 新增 `compute_selected_counts(...)`、`compute_weight_entropy(...)` 和 `compute_max_weight(...)`。
- helper 保持纯 numpy，不引入 torch/sklearn 训练依赖。
- helper 只接受调用方显式传入的 `selected_indices`、`weights` 和 `model_columns`，不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录。
- 输入校验覆盖 `weights` 二维 `[sample, expert]`、`selected_indices` 一维 `[sample]`、`model_columns` 非空且不重复、`selected_indices` 不越界，以及权重有限非负。
- `tests/smoke/stage1_golden_smoke.py` 基于 `GOLDEN_WEIGHTS` 的 hard top-1 下标复算 selected counts，并校验 entropy/max_weight shape 与 max_weight 数值。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不实现 temperature/top-k/calibration。
- 不改变 summary/comparison/prediction output schema。
- 不新增正式训练 CLI。
- 不读取 oracle/TSF，不实现 oracle regret。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。
- 不把 oracle/TSF 作为 test-time 动态特征。

### P3c：minimal evaluation summary builder only

目标：在 P3a/P3b 的 fusion/metrics/diagnostics helper 基础上，只抽取最小 evaluation summary builder，把 hard top-1、raw soft fusion、selected counts、weight entropy 和 max weight 组合为稳定 summary dict。

当前状态（2026-06-19）：已完成 P3c 最小 summary helper 抽取并接入 `tests/smoke/stage1_golden_smoke.py` 的 deterministic summary 断言；完整 P3 的 calibration、comparison、per-sample schema 收束和正式入口接入仍保留到后续小步。

本次完成范围：

- 新增 `time_router/evaluation/summary.py`，提供 `build_fusion_summary(...)`。
- helper 保持纯 numpy / Python 标准库，不引入 torch/sklearn/pandas 训练依赖。
- helper 只接受调用方显式传入的 `FusionMetricsResult`、`weights` 和 `model_columns`，不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录。
- summary dict 固定包含 `hard_mae`、`hard_mse`、`raw_soft_mae`、`raw_soft_mse`、`selected_counts`、`mean_entropy`、`mean_max_weight`、`num_samples`、`num_experts` 和 `model_columns`。
- `tests/smoke/stage1_golden_smoke.py` 基于 `GOLDEN_WEIGHTS`、`hard_result` 和 `raw_soft_result` 断言 summary 的 golden MAE/MSE、selected counts、样本数、专家数、专家顺序、mean max weight 和 mean entropy。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不实现 temperature/top-k/calibration。
- 不改变正式 summary / comparison / prediction output schema。
- 不新增正式训练 CLI。
- 不读取 oracle/TSF，不实现 oracle regret。
- 不接入 `OracleTsfReader` 或 full-scale 输出目录。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。

### P3d：minimal per-sample evaluation rows builder only

目标：在 P3a/P3b/P3c 的 evaluation helper 基础上，只抽取最小逐样本 evaluation rows builder，把当前 batch 的 sample_key、hard top-1 选择、hard/raw-soft 逐样本指标和 router weight diagnostics 组合为稳定 rows。

当前状态（2026-06-19）：已完成 P3d 最小 rows helper 抽取并接入 `tests/smoke/stage1_golden_smoke.py` 的 deterministic rows 断言；完整 P3 的 calibration、comparison、正式 per-sample prediction schema 收束和正式入口接入仍保留到后续小步。

本次完成范围：

- 新增 `time_router/evaluation/prediction_rows.py`，提供 `build_per_sample_fusion_rows(...)`。
- helper 保持纯 numpy / Python 标准库，不引入 torch/sklearn/pandas 训练依赖。
- helper 只接受调用方显式传入的 `sample_keys`、`FusionMetricsResult`、`y_true`、`weights` 和 `model_columns`，不读取 manifest、prediction cache、oracle/TSF 或正式训练输出目录。
- 每个 row 固定包含 `sample_key`、`selected_model`、`selected_index`、`hard_mae`、`hard_mse`、`raw_soft_mae`、`raw_soft_mse`、`max_weight` 和 `weight_entropy`。
- 逐样本 MAE/MSE 只在每个 sample 内按 pred_len/channel 维度聚合，不改变现有全局 MAE/MSE helper 的口径。
- `tests/smoke/stage1_golden_smoke.py` 基于 4 sample packed fixture 断言 rows 数量、sample_key 顺序、selected_model 顺序、selected_index、字段集合、逐样本 MAE/MSE、max_weight 和 weight_entropy。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不实现 temperature/top-k/calibration。
- 不改变正式 summary / comparison / prediction output schema。
- 不新增正式训练 CLI。
- 不读取 oracle/TSF，不实现 oracle regret。
- 不接入 `OracleTsfReader` 或 full-scale 输出目录。
- 不写 CSV / JSON / Parquet 到正式输出目录。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。

### P3e：evaluation package boundary review and consolidation plan only

目标：对已完成 P3a/P3b/P3c/P3d 的 `time_router/evaluation` package 边界做一次文档化复核，明确模块职责、public API、private helper 和后续 consolidation 判断。

当前状态（2026-06-19）：已完成 P3e 文档化 review；本阶段只新增设计文档，不改变 helper 行为、不移动/重命名 evaluation 文件、不实现 comparison/calibration、不迁移正式入口。

本次完成范围：

- 新增 `docs/refactor/evaluation_package_boundary.md`。
- 明确 `metrics.py` 当前承载基础 MAE/MSE、fusion helper 和 router weight diagnostics；暂不拆成 `fusion.py` / `diagnostics.py`。
- 明确 `summary.py` 只负责 `build_fusion_summary(...)` 的内存 summary dict，不代表正式 summary output schema。
- 明确 `prediction_rows.py` 只负责内存中的 per-sample fusion rows，不写正式 CSV/JSON/Parquet，不代表正式 prediction output schema。
- 明确 `time_router/evaluation/__init__.py` 是稳定 public API 聚合入口；后续正式入口迁移应优先从 `time_router.evaluation` 导入，而不是依赖深层私有 helper。
- 明确当前不为“看起来整齐”而移动文件；若未来整理，应优先保持 `__init__.py` public API 不变，并运行 golden smoke、oracle/TSF smoke 和 compileall。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不实现 comparison。
- 不实现 temperature/top-k/calibration。
- 不改变正式 summary / comparison / prediction output schema。
- 不新增正式训练 CLI。
- 不读取 oracle/TSF，不实现 oracle regret。
- 不接入 `OracleTsfReader` 或 full-scale 输出目录。
- 不写 CSV / JSON / Parquet 到正式输出目录。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。
- 不为整理而移动或重命名 evaluation 文件。
- 不改变现有 public API 和 helper 行为。

### P4：extract logging/path/config

目标：收束日志、路径解析、状态落盘和配置默认值，减少脚本内硬编码。

范围：

- workspace root、输出根、manifest、prediction cache、oracle、TSF、feature cache、checkpoint 的集中解析。
- atomic JSON writer、`status.json`、metadata lineage、checkpoint index 和中文摘要。
- CLI 显式参数优先于集中默认值。

门禁：

- 迁移前后运行 golden smoke。
- 对现有 launcher 的 `status.json`、metadata 和恢复命令做兼容检查。
- 不改变既有正式输出目录含义，不破坏后台任务监控脚本。

### P4a：minimal atomic JSON and run status writer only

目标：先抽出 P4 中最低风险的 JSON 原子写入和最小 status writer，统一后续 `status.json` / metadata-like JSON 的安全写入能力，但不迁移正式训练入口、不接入 full-scale 任务。

当前状态（2026-06-19）：已完成 P4a 最小基础设施抽取和小规模 smoke；完整 P4 的 logging、path resolver、config system、launcher/monitor/resume 兼容迁移仍保留到后续小步。

本次完成范围：

- 新增 `time_router/io/json_utils.py`，提供纯标准库 `atomic_write_json(...)`、`build_status_payload(...)` 和 `write_status_json(...)`。
- `atomic_write_json(...)` 自动创建 parent directory，在目标同目录写临时文件，完成 `flush + fsync` 后通过 `os.replace` 原子替换目标文件。
- JSON 写入默认 UTF-8、`ensure_ascii=False`，适合中文 status / metadata 文本。
- `write_status_json(...)` 只写调用方显式传入的 path；payload 至少包含 `status`，可选 `phase`、`message`，`extra` 必须是 dict 或 None。
- 新增 `tests/smoke/stage1_json_utils_smoke.py`，只在 `tempfile.TemporaryDirectory` 下验证 status 写入、中文 message、第二次覆盖、nested parent directory 自动创建和 `extra` 类型检查。
- 新增 `docs/refactor/json_utils.md`，记录 P4a helper 边界和完整门禁。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何现有正式训练脚本。
- 不接入 full-scale 输出目录。
- 不改变既有 `status.json` schema。
- 不改现有 launcher / monitor / resume 行为。
- 不实现 path resolver、config system 或 logging framework。
- 不实现 comparison / calibration / report schema。
- 不读取 oracle/TSF。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。

### P4b：minimal path resolver helper only

目标：在 P4a atomic JSON / status writer 之后，抽出 P4 中最低风险的路径解析基础能力，用于 repo root、root 内安全拼接和 status/metadata path 计算；不迁移正式入口、不接入 full-scale 输出目录。

当前状态（2026-06-19）：已完成 P4b 最小 path resolver helper 抽取和 repo-level / tempfile smoke；完整 P4 的 config system、logging framework、checkpoint index、launcher/monitor/resume 替换仍保留到后续小步。

本次完成范围：

- 新增 `time_router/io/path_resolver.py`，提供 `find_repo_root(...)`、`resolve_under_root(...)`、`resolve_status_path(...)` 和 `resolve_metadata_path(...)`。
- `find_repo_root(...)` 从调用方起点或当前 helper 文件位置向上查找 `.git`、`WORKSPACE_STRUCTURE.md` 或 pyproject-like marker，找不到时明确报错。
- `resolve_under_root(...)` 返回 root 内部 resolved path，拒绝通过 `..` 或绝对路径逃逸 root；`must_exist=True` 时路径不存在会明确报错。
- `resolve_status_path(...)` / `resolve_metadata_path(...)` 只返回 `run_dir / "status.json"` 和 `run_dir / "metadata.json"`，不创建目录、不写文件、不假设正式输出目录。
- 新增 `tests/smoke/stage1_path_resolver_smoke.py`，覆盖 repo root 查找、`WORKSPACE_STRUCTURE.md` 定位、tempfile 正常路径、逃逸 root 报错、`must_exist=True` 不存在报错和 status/metadata helper 不创建文件。
- 新增 `docs/refactor/path_resolver.md`，记录 P4b helper 边界和完整门禁。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何现有正式训练脚本。
- 不替换现有 launcher / monitor / resume 行为。
- 不接入 `/data2` 或任何 full-scale 输出目录。
- 不改变既有正式输出目录含义。
- 不改变既有 `status.json` / `metadata.json` schema。
- 不实现 config system。
- 不实现 logging framework。
- 不实现 checkpoint index。
- 不创建正式输出目录。
- 不写 JSON 文件。
- 不改 evaluation helper。
- 不实现 comparison / calibration / report schema。
- 不读取 oracle/TSF。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。

### P4c：minimal run metadata payload builder only

目标：在 P4a atomic JSON/status writer 和 P4b path resolver 之后，抽出 P4 中最低风险的 run metadata payload builder，用于构造 metadata-like JSON payload；不迁移正式入口、不替换既有 launcher metadata schema、不接入 full-scale 输出目录。

当前状态（2026-06-19）：已完成 P4c 最小 run metadata payload builder 抽取和 tempfile smoke；完整 P4 的 config system、logging framework、checkpoint index、launcher/monitor/resume 替换仍保留到后续小步。

本次完成范围：

- 新增 `time_router/io/run_metadata.py`，提供 `build_run_metadata(...)` 和 `write_run_metadata(...)`。
- `build_run_metadata(...)` 校验 `stage` 为非空字符串，`inputs`、`outputs`、`extra` 为 dict 或 None，payload 至少包含 `stage`、`created_at_utc`、`inputs`、`outputs`。
- `created_at_utc` 使用 timezone-aware UTC ISO 字符串。
- `Path` / `os.PathLike` 会在内存中转换为字符串，便于 JSON 写入；不检查路径是否存在。
- `write_run_metadata(...)` 只写调用方显式传入的 path，内部调用 `atomic_write_json(...)`。
- 新增 `tests/smoke/stage1_run_metadata_smoke.py`，只在 `tempfile.TemporaryDirectory` 下验证 payload 构造、字段校验、UTC 时间、Path 转字符串和 writer JSON 可读。
- 新增 `docs/refactor/run_metadata.md`，记录 P4c helper 边界和完整门禁。

明确不做：

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何现有正式训练脚本。
- 不替换现有 launcher / monitor / resume 行为。
- 不接入 `/data2` 或任何 full-scale 输出目录。
- 不改变既有正式输出目录含义。
- 不改变既有 `metadata.json` schema。
- 不实现 config system。
- 不实现 logging framework。
- 不实现 checkpoint index。
- 不自动调用 git。
- 不自动读取命令行或训练配置。
- 不创建正式输出目录。
- 不改 evaluation helper。
- 不实现 comparison / calibration / report schema。
- 不读取 oracle/TSF。
- 不改 `PredictionBatchReader` / `OracleTsfReader`。
- 不改模型结构、loss 或正式输出目录。

### P4d：run artifacts boundary review and integration plan only

目标：在 P4a/P4b/P4c 之后，对 `time_router/io` 当前 run artifacts 相关工具做一次文档化架构边界复核，明确 public API、private helper、低风险 IO helper 与正式训练入口 / launcher / resume 层的职责边界，并给出后续接入规划。

当前状态（2026-06-19）：已完成 P4d 文档化 review；本阶段只新增边界复核和 integration plan，不迁移正式 Visual Router / TimeFuse fusor 入口，不接入 `/data2` 或 full-scale 输出目录，不改变既有 `status.json` / `metadata.json` schema。

本次完成范围：

- 新增 `docs/refactor/run_artifacts_boundary.md`，复核 `prediction_cache_reader.py`、`json_utils.py`、`path_resolver.py`、`run_metadata.py` 和 `time_router/io/__init__.py` 的职责边界。
- 明确 `PredictionBatchReader` / `PredictionBatch` / `DEFAULT_MODEL_COLUMNS` 仍属于 prediction cache 数据读取 public API，不属于 run artifacts writer。
- 明确 `atomic_write_json`、`build_status_payload`、`write_status_json`、`find_repo_root`、`resolve_under_root`、`resolve_status_path`、`resolve_metadata_path`、`build_run_metadata` 和 `write_run_metadata` 是当前低风险 IO public API。
- 明确 `_normalize_start_path`、`_require_dict_or_none`、`_to_json_safe` 以及 `prediction_cache_reader.py` 内部下划线 helper 属于 private helper，正式入口不应依赖。
- 回答后续正式入口接入条件、接入前需比较的 status/metadata 字段、launcher/monitor/resume 不能随便改的字段、checkpoint index 是否拆为 P4e、config system 是否推迟到 P5/P6 前，以及如何保证旧输出目录含义不变。
- 在 `time_router/io/__init__.py` 补充包级边界注释；不改导出列表、不改函数签名、不改函数行为。

明确不做：

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

### P4e：checkpoint index boundary review and integration plan only

目标：在 P4d 确认 checkpoint index 应单独处理之后，对 Visual Router / TimeFuse-style fusor 当前 checkpoint、best/latest model、resume、launcher、monitor、`status.json` 和 `metadata.json` 约定做文档化边界复核，并规划后续是否抽取 checkpoint index helper。

当前状态（2026-06-19）：已完成 P4e 文档化 review；本阶段只新增 checkpoint index 边界复核和接入规划，不实现 checkpoint index helper，不修改正式训练入口，不替换 launcher / monitor / resume 行为。

本次完成范围：

- 新增 `docs/refactor/checkpoint_index_boundary.md`，审查 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py`、`launch_timefuse_fusor_full_scale.py`、`launch_full_scale_prediction_cache.py` 和 `build_prediction_cache_from_manifest.py` 的 checkpoint/resume/status/metadata 约定。
- 明确非 streaming `train_visual_router.py` / `train_visual_router_online.py` 当前只有 metadata 与评估输出，没有通用 checkpoint/resume 语义，不应为统一 P4e 而补历史 checkpoint index。
- 明确 streaming Visual Router 当前使用 `checkpoints/router_{config}_epoch_000N.pt`、`latest_{config}.pt` 和 `latest_checkpoint_index.json`，字段为 `completed_epochs`。
- 明确 TimeFuse-style fusor 当前使用 `checkpoints/timefuse_fusor_epoch_000N.pt`、`latest_timefuse_fusor.pt` 和 `latest_checkpoint_index.json`，字段为 `completed_epoch`。
- 明确 TimeFuse full-scale launcher 依赖 `command.sh`、`command_resume.sh`、`launcher.sh`、`stop.sh`、`resume.sh`、`pid.txt`、`pgid.txt`、`main.log`、`status.json` 和 `metadata.json`；checkpoint index helper 不能替代 launcher 进程管理和 resume policy。
- 明确 prediction cache builder/launcher 的 `--resume` 与 `checkpoint_selection` 是 cache 构建和专家 checkpoint 选择口径，不是 router/fusor checkpoint index。
- 建议未来 checkpoint index helper 应属于 training/runtime 层，IO 层只复用原子 JSON；本次不放入 `time_router/io`，不改任何 helper 行为。

明确不做：

- 不实现 checkpoint index。
- 不修改任何正式训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不替换 launcher / monitor / resume 行为。
- 不改变 `status.json` / `metadata.json` / checkpoint 文件 schema。
- 不接入 `/data2` 或 full-scale 输出目录。
- 不创建输出目录。
- 不自动调用 git。
- 不实现 config system。
- 不实现 logging framework。
- 不改 `time_router/io` helper 行为。
- 不改 `PredictionBatchReader` / `OracleTsfReader` / evaluation helper。
- 不改模型结构、loss 或正式输出目录。

### P4f：paused config system; architecture pivot after P4

目标：P4d/P4e 完成后，不继续沿“兼容历史输出”的路径实现 config system，而是先做架构转向决策，明确 Stage 1 新主干只保留哪些 canonical entrypoint、哪些历史入口降级为 archive/deprecated/reference-only，以及新 runtime 最小契约是什么。

当前状态（2026-06-19）：已完成 P4 后 architecture pivot review；本阶段只新增决策文档，不改训练代码、不迁移入口、不实现 config system、不实现 checkpoint index、不接入 `/data2`、不移动或删除历史代码。

本次完成范围：

- 新增 `docs/refactor/stage1_architecture_pivot_after_p4.md`。
- 明确 Visual Router canonical entrypoint 为 `train_visual_router_online_streaming.py`，路线固定为 `x -> pseudo image -> frozen ViT -> router`。
- 明确 TimeFuse-style fusor baseline canonical entrypoint 为 `train_timefuse_fusor_streaming.py` 和 `launch_timefuse_fusor_full_scale.py`，路线固定为 `sample_key -> 17维 feature cache -> Linear-softmax fusor`。
- 明确 LogisticRegression fusor、offline ViT embedding cache、旧 OOM lookup、pilot-only 脚本和非 streaming full-scale 入口不再作为正式主干，不再为其新增兼容 helper。
- 明确继承 `sample_key`、固定五专家顺序、`packed_npy_v1` row index、oracle/TSF join 口径和 evaluation 复算口径；舍弃旧 pilot/status/metadata/checkpoint 的强兼容目标。
- 定义新 canonical runtime 最小契约：`run_dir`、`status.json`、`metadata.json`、`checkpoints/`、`predictions/` 或 evaluation outputs、`logs/` 或 `main.log`。
- 明确 P4 helper 只作为底层 JSON/path/metadata 能力接入，evaluation helper 负责统一复算，`PredictionBatchReader` 和 `OracleTsfReader` 作为契约来源但 full-scale 仍需 streaming/shard-aware 读取。
- 决定暂停 P4f config system，转向 P5 canonical entrypoint design / FeatureProvider interface design。

明确不做：

- 不改任何训练脚本。
- 不迁移入口。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不为了兼容历史输出新增 helper。
- 不改变 `PredictionBatchReader` / `OracleTsfReader` / evaluation / IO helper 行为。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

### P5：introduce canonical provider interface

目标：在 P4 后 architecture pivot 和 P5a canonical runtime contract 已确认的基础上，引入显式 provider interface 边界，把一次实验拆成 `ExperimentProtocol -> SplitStrategy -> ExpertProvider -> FeatureProvider -> RouterHead -> Evaluator`。P5 之前不再先实现共享 config system，也不把当前 frozen ViT 或 17 维 feature cache 写死为接口本身。

范围：

- `ExperimentProtocol`：描述一次实验如何绑定 split、expert、feature、head、evaluator 和 runtime contract，不等于某个脚本。
- `SplitStrategy`：当前默认 vali train/test eval，未来兼容 cell holdout 和 cross-cell generalization，并明确 split 下推边界。
- `ExpertProvider`：当前默认由 `PredictionBatchReader + packed_npy_v1` 实现，未来兼容 online expert prediction 和 router-expert joint training。
- `VisualFeatureProvider`：`x -> pseudo image -> frozen ViT embedding`，在线 batch 生成，不落盘 full-scale embedding。
- `TimeFuseFeatureProvider`：`sample_key -> 17维 feature cache`，支持 feature-only scaler 和 shard-aware 读取。
- `RouterHead`：把 feature tensor 或 structured feature 转成专家 logits/weights，不读取 prediction cache，不写 evaluation output。
- `Evaluator`：接收 sample_key、`y_pred`、`y_true`、weights/logits 和 `model_columns`，使用 `time_router.evaluation` public API 输出 summary、rows、comparison 和 calibration-ready object。
- 同步设计 canonical runtime 的 `run_dir/status/metadata/checkpoint/evaluation/logs` 契约，再决定哪些参数进入共享 config。

门禁：

- 迁移前后运行 golden smoke。
- Visual 小规模 smoke 需确认 online embedding 行为与既有入口一致。
- TimeFuse 小规模/pressure smoke 需确认 feature shape 为 17 维、split 下推和 scaler 不加载 prediction arrays。

### P5a：canonical runtime contract only

目标：在 P4 后 architecture pivot 之后，先定义 Stage 1 新 canonical runtime contract，明确新 `run_dir` 结构、`status.json` / `metadata.json` 最小字段、Visual Router 与 TimeFuse-style fusor 的共享字段和 branch-specific extra、P4 helper 接入边界、checkpoint index 最小概念和旧 schema 舍弃边界。

当前状态（2026-06-19）：已完成 P5a 文档化 contract；本阶段只新增设计文档，不改训练代码、不迁移入口、不实现 config system、不实现 checkpoint index、不实现 logging framework、不接入 `/data2`、不移动或删除历史代码。

本次完成范围：

- 新增 `docs/refactor/stage1_canonical_runtime_contract.md`。
- 定义新 canonical run_dir 最小结构：`status.json`、`metadata.json`、`checkpoints/`、`logs/`、`evaluation/`、`predictions/` 或 `prediction_outputs/`。
- 定义新 `status.json` 最小字段：`status`、`phase`、`updated_at`、`run_dir`、`entrypoint`、`config_name`、`progress`、`latest_checkpoint_path`、`error`。
- 定义新 `metadata.json` 最小字段：`stage`、`entrypoint`、`config_name`、`args`、`inputs`、`outputs`、`model_columns`、`array_storage`、`feature_schema`、`split_strategy`、`created_at_utc`。
- 明确 Visual Router 与 TimeFuse-style fusor 的共享字段语义一致，视觉伪图像/ViT/router loss 参数和 TimeFuse feature cache/fusor reader 参数分别进入 branch-specific extra。
- 明确 P4a/P4b/P4c helper 只作为原子 JSON、路径解析和 metadata-like payload 的底层能力，不反向兼容所有历史 schema。
- 明确新 runtime 中 checkpoint index 的最小概念，但不实现 helper。
- 明确舍弃旧非 streaming metadata、LogisticRegression、offline embedding cache、旧 OOM、pilot launcher、prediction cache builder resume 等历史 schema 的强兼容目标。

明确不做：

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

### P5b：canonical provider interface design only

目标：在 P5a runtime contract 之后，只做 Stage 1 canonical provider interface 设计，明确 `ExperimentProtocol`、`SplitStrategy`、`ExpertProvider`、`FeatureProvider`、`RouterHead` 和 `Evaluator` 的共享 contract、分支扩展点和历史路线舍弃边界。

当前状态（2026-06-19）：已完成 P5b 文档化 interface 设计；本阶段只新增设计文档，不改训练代码、不迁移入口、不实现 Python interface / abstract class、不实现 config system、run_dir helper、checkpoint index 或 logging framework、不接入 `/data2`、不移动或删除历史代码。

本次完成范围：

- 新增 `docs/refactor/stage1_provider_interface.md`。
- 明确 `ExperimentProtocol` 描述一次实验的 protocol，不等于某个脚本；只绑定 split、expert、feature、head、evaluator 和 runtime contract，不直接读写 full-scale 输出目录。
- 明确 `SplitStrategy` 当前默认 vali train/test eval，未来兼容 cell holdout 和 cross-cell generalization，并定义其向 `ExpertProvider`、`FeatureProvider` 和 `Evaluator` 下推的边界。
- 明确 `ExpertProvider` 当前可由 `PredictionBatchReader + packed_npy_v1` 实现，但 interface 只要求输出 `sample_key`、`model_columns`、`y_pred`、`y_true` 和 row index metadata，不假设专家一定来自离线 cache。
- 明确 `FeatureProvider` 同时覆盖 Visual Router 的 pseudo image / encoder feature 和 TimeFuse-style fusor 的 feature cache / online feature computation，允许 frozen encoder、finetuned encoder、joint-trained encoder、offline cache、online computation 和 branch-specific schema。
- 明确 oracle/TSF 只可用于监督、诊断、baseline 或分层分析，不进入可部署 Visual Router test-time 动态调权特征。
- 明确 `RouterHead` 只消费 feature 并输出 logits/weights，不读取 prediction cache、不写 evaluation output。
- 明确 `Evaluator` 只消费显式 `sample_key/y_pred/y_true/weights/logits/model_columns`，使用 `time_router.evaluation` public API 输出 summary、per-sample rows、comparison 和 calibration-ready object，不依赖 legacy output schema。
- 明确 provider interface 不决定 `run_dir` 位置，不硬编码 `/data2` 或 repo 内路径。
- 明确 Visual Router 主线、TimeFuse-style fusor baseline 的共享 contract 与 branch-specific implementation 边界，并把 LogisticRegression hard-label router、offline ViT embedding full-scale cache、旧 OOM lookup、pilot-only 和非 streaming full-scale 入口标为 deprecated/reference-only，不再适配新 interface。

明确不做：

- 不实现 Python interface / abstract class。
- 不新增 `time_router/protocols` 或 runtime 代码。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 config system。
- 不实现 run_dir helper。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不为了兼容历史输出新增 helper。
- 不改变 `PredictionBatchReader` / `OracleTsfReader` / evaluation / IO helper 行为。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

### P5c：minimal protocol types skeleton only

目标：基于 P5b provider interface design，新增最小 protocol dataclass 类型骨架。类型只作为 lightweight contract container，不实现训练逻辑、不做文件 IO、不绑定 numpy/torch/pandas/sklearn。

当前状态（2026-06-19）：已完成 P5c 最小类型骨架；本阶段只新增 `time_router.protocols` dataclass、纯内存 smoke、文档和日志，不迁移正式入口、不实现 provider/runtime/config/checkpoint/logging。

本次完成范围：

- 新增 `time_router/protocols/types.py`，定义 `SplitSpec`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput` 和 `ExperimentProtocolSpec`。
- 新增 `time_router/protocols/__init__.py`，从 `time_router.protocols` 导出 P5c public API。
- 所有 array/tensor-like 字段统一使用 `Any`，不访问 `.shape`，不做数值或 shape 校验。
- `sample_keys`、`model_columns`、`train_splits`、`eval_splits` 使用 tuple 保存调用方顺序。
- `extra`、`branch_specific` 和 `feature_schema` 使用 `field(default_factory=dict)`，避免跨实例共享默认 dict。
- `RouterOutput` 和 `EvaluationInput` 同时保留可选 `logits` 与 `weights`；P5c 不强制至少一个存在。
- `ExperimentProtocolSpec` 的 `split_strategy`、`expert_provider`、`feature_provider`、`router_head` 和 `evaluator` 字段只保存 spec、引用或配置描述，不实例化真实 provider。
- 新增 `tests/smoke/stage1_protocol_types_smoke.py`，纯内存验证 public API 导入、全部 dataclass 构造、tuple 保序、default_factory 独立性、logits/weights 可选组合和 object/list 字段原样保存。
- 新增 `docs/refactor/protocol_types.md`，记录 P5c 类型边界、统一约束、smoke 覆盖和明确不做范围。

明确不做：

- 不实现 Python abstract base class。
- 不实现 FeatureProvider / ExpertProvider 读取逻辑。
- 不实现 ExperimentProtocol 执行逻辑。
- 不实现 runtime / run_dir helper。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / io helper。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

### P5d：provider adapter boundary review only

目标：基于 P5b provider interface design 和 P5c protocol types skeleton，审查现有代码中哪些模块/函数未来可以适配为 canonical provider/head/evaluator adapter，并明确哪些旧路线不应继续适配。

当前状态（2026-06-19）：已完成 P5d 文档化边界审查；本阶段只新增 adapter boundary review 文档并更新路线图/目标架构/结构索引/实验日志，不实现 provider adapter、不修改任何代码行为。

本次完成范围：

- 新增 `docs/refactor/provider_adapter_boundary.md`。
- 明确 ExpertProvider 第一批候选为基于 `PredictionBatchReader` 的 `PredictionCacheExpertProvider`；`prediction_array_io` grouped loading 是底层数组能力，应间接复用，不作为 provider 本体。
- 明确 `packed_npy_v1` 是 full-scale 推荐读取边界，`per_sample_npy` 只保留 legacy/smoke 兼容；全量 Python lookup、每 sample 重复打开 packed 文件、legacy CSV 反推 prediction 和硬编码 `/data2` 的旧路线不应适配。
- 明确 Visual pseudo image / ViT feature 路径可作为中期 `VisualOnlineVitFeatureProvider` 候选，但因牵涉 encoder、Quito 历史窗口、dtype、GPU/DataParallel 和 future finetune/joint 训练，不适合作为第一批最小 adapter。
- 明确 TimeFuse 17 维 feature cache reader 可作为第二批 `TimeFuseFeatureCacheProvider` 候选，但必须拆出 feature-only provider，不能顺手读取 oracle/prediction arrays 或写入 runtime artifacts。
- 明确 future online TimeFuse feature computation 是扩展点，offline ViT embedding cache 仅 reference-only / debug-only，不作为 full-scale canonical adapter。
- 明确 Visual Router MLP 与 TimeFuse Linear-softmax 可以成为 RouterHead adapter；loss、optimizer、epoch loop、scaler fit、checkpoint/resume、DataParallel、prediction/oracle 读取和 CSV 写出不属于 RouterHead。
- 明确 `time_router.evaluation` public API 是 Evaluator adapter 的基础能力；legacy CSV schema 不应反向污染 Evaluator。
- 明确 provider/head/evaluator adapter 不决定 `run_dir`、不写 `status.json` / `metadata.json`、不创建 checkpoint index、不硬编码 `/data2`，由未来 launcher/runtime 显式编排。
- 给出架构判断：先做 entrypoint migration plan，再实现最小 `PredictionCacheExpertProvider`；TimeFuse feature cache provider 可后续接入，Visual online ViT provider 不作为第一批最小实现。

明确不做：

- 不实现 provider adapter。
- 不新增 ExpertProvider / FeatureProvider 读取代码。
- 不修改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / io helper。
- 不修改 protocol types。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 runtime / run_dir helper。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

### P5e：canonical entrypoint migration plan only

目标：基于 P5a runtime contract、P5b provider interface、P5c protocol types 和 P5d adapter boundary review，设计 Visual Router 与 TimeFuse-style fusor 两条 canonical entrypoint 的迁移路线；本阶段只写文档，不改训练代码、不实现 adapter、不迁移入口。

当前状态（2026-06-19）：已完成 P5e 文档化迁移计划；本阶段只新增 entrypoint migration plan 并更新路线图/目标架构/结构索引/实验日志，不创建 run_dir、不接入 `/data2`、不修改任何训练入口。

本次完成范围：

- 新增 `docs/refactor/stage1_entrypoint_migration_plan.md`。
- 明确 `train_visual_router_online_streaming.py` 当前把 runtime orchestration、ExpertProvider、Visual FeatureProvider、Visual RouterHead、Evaluator 和 run artifacts 写出混在同一脚本；未来旧入口保留 CLI/runtime/checkpoint/status/metadata 调度，prediction cache 读取、online ViT feature、MLP head 和 evaluator 复算逐步下沉。
- 明确 `train_timefuse_fusor_streaming.py` 当前把 shard 准备、TimeFuse feature streaming、prediction/oracle SQLite index、linear-softmax head、scaler、训练/eval、checkpoint 和报告写出混在同一脚本；未来按 ExpertProvider、FeatureProvider、RouterHead、Evaluator 和 runtime/report 分层迁移。
- 明确 `launch_timefuse_fusor_full_scale.py` 是 launcher/preflight/后台进程管理层，不是 provider 或训练 runtime 本体；full-scale `run_dir` 未来由 launcher/runtime 显式传入。
- 第一批代码迁移顺序建议为：先 `PredictionCacheExpertProvider`，再 evaluator adapter，再 `TimeFuseFeatureCacheProvider`，再 TimeFuse linear-softmax head，最后 Visual online ViT feature provider 和 Visual head。
- 明确新 adapter 先由 smoke 使用，不直接替换 full-scale streaming 入口；正式入口后续小步接入，允许重跑实验，不强兼容旧输出 schema。
- 明确 migration plan 不创建 `run_dir`，provider/entrypoint plan 不硬编码 `/data2`。

明确不做：

- 不实现 provider adapter。
- 不新增 ExpertProvider / FeatureProvider 读取代码。
- 不修改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / io helper。
- 不修改 protocol types。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 runtime / run_dir helper。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

### P5f：launcher architecture design only

目标：基于 P5a-P5e 已定义的 runtime contract、provider interface、protocol types、adapter boundary 和 entrypoint migration plan，只设计 Stage 1 及未来 Stage 的 launcher architecture，明确未来 `exp_scripts/*.sh -> scripts/*.py -> time_router runtime/protocol/provider/head/evaluator` 的分层职责。

当前状态（2026-06-19）：已完成 P5f 文档化 launcher architecture；本阶段只新增 launcher 架构设计并更新路线图/目标架构/入口迁移计划/结构索引/实验日志，不新增 Bash、不新增 Python entrypoint、不实现 config system、不修改任何训练脚本。

本次完成范围：

- 新增 `docs/refactor/launcher_architecture.md`。
- 明确 `exp_scripts/` 是 Bash launcher 层，负责选择 config、绑定 GPU/conda/env、设置日志与 `nohup`/后台策略、显式传入 `/data2/.../run_dir` 或 `output_root`、保存可复现实验命令，但不包含核心训练逻辑。
- 明确 `scripts/` 是极薄 Python entrypoint 层，只解析 config/CLI、构造 `ExperimentProtocolSpec` 或等价 runtime spec、调用 future runtime，不实现 provider 读取细节或模型训练主体逻辑。
- 明确 `time_router/` 是 runtime/protocol/provider/features/models/evaluation/io helper 实现层，不知道 Bash 存在，不硬编码 `exp_scripts` 路径，也不决定 full-scale `run_dir` 是否在 `/data2`。
- 明确 `configs/` 保存 Stage/config/branch 参数、Visual Router 与 TimeFuse-style fusor branch-specific config、future finetune ViT / joint training / online expert / online TimeFuse feature 扩展点；full-scale 路径由 launcher 显式传入，不默认写死到 repo 内。
- 明确 `run_dir` 与 `/data2` 边界：full-scale 通常在 `/data2/syh/Time/...`，repo 只保存代码、配置、文档、小 fixture 和 smoke；provider 不决定 `run_dir`。
- 明确 `train_visual_router_online_streaming.py` 与 `train_timefuse_fusor_streaming.py` 短期仍是 canonical-current，`launch_timefuse_fusor_full_scale.py` 短期仍是 full-scale launcher/preflight/后台进程管理层；未来逐步过渡到 `scripts/ + exp_scripts/`。
- 推荐未来目录形态：`configs/stage1/{visual_router,timefuse_fusor}/`、`exp_scripts/stage1/{visual_router,timefuse_fusor}/`、`scripts/stage1/`、`time_router/{runtime,providers,features,models,evaluation,protocols,io}/` 和 `archive/`。
- 给出 P5f 后小步建议：先做 `PredictionCacheExpertProvider` smoke-only，再做 evaluator adapter，再补最小 config skeleton，再做 `scripts/` thin entrypoint skeleton，最后新增 `exp_scripts/` Bash launcher。

明确不做：

- 不新增 Bash 脚本。
- 不新增 Python entrypoint。
- 不实现 config system。
- 不实现 runtime / run_dir helper。
- 不实现 provider adapter。
- 不修改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / io / protocols。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

### P6a：minimal PredictionCacheExpertProvider smoke-only

目标：基于 P1 `PredictionBatchReader`、P5c protocol types、P5d adapter boundary 和 P5e/P5f migration/launcher 设计，新增最小 `PredictionCacheExpertProvider`，只把 reader 输出包装为 `ExpertBatch`，先供 smoke 使用，不接正式训练入口。

当前状态（2026-06-19）：已完成 P6a 最小 adapter 与 smoke；本阶段新增 `time_router/experts/`，但不修改 Visual Router / TimeFuse fusor 正式入口，不实现 runtime、config、launcher 或 run_dir helper。

本次完成范围：

- 新增 `time_router/experts/__init__.py`，导出 `PredictionCacheExpertProvider`。
- 新增 `time_router/experts/prediction_cache.py`，内部复用 `time_router.io.PredictionBatchReader`。
- `PredictionCacheExpertProvider.load_batch(sample_keys, verify_metrics=True)` 要求调用方显式传入非空、不重复 sample_keys，不默认扫描全量 manifest。
- `load_batch(...)` 输出 `time_router.protocols.ExpertBatch`，包含 tuple 化 `sample_keys`、tuple 化 `model_columns`、`y_pred`、`y_true`、`row_index_metadata` 和轻量 `extra`。
- 保持 sample_key 顺序、固定五专家顺序、共享 `y_true` 校验、packed row index lineage 和 reader 的 `verify_metrics` 校验能力。
- `extra` 记录 `provider_name`、`array_storage`、`manifest_path`、当前 batch manifest 行数、原始 manifest 专家顺序、`verify_metrics` 和 `chunk_rows` 等轻量 reader metadata，不塞入 manifest DataFrame。
- 新增 `tests/smoke/stage1_prediction_cache_expert_provider_smoke.py`，使用 golden fixture 显式传入 4 个 sample_key，验证 `ExpertBatch` contract、row index metadata、extra metadata，并用 `time_router.evaluation` public API 复算 hard top-1 / raw soft fusion golden 指标。
- 新增 `docs/refactor/prediction_cache_expert_provider.md`，记录 API、reader 关系、metadata、边界和后续接入顺序。

明确不做：

- 不修改 `PredictionBatchReader` 行为。
- 不移动 `prediction_array_io`。
- 不读取 oracle/TSF。
- 不生成 router feature。
- 不计算 loss。
- 不做正式 evaluation；smoke 里的 evaluation 只用于验收 provider 输出未漂移。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 `status.json` / `metadata.json`。
- 不实现 config system。
- 不实现 runtime / launcher。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不修改 Visual Router / TimeFuse fusor 正式入口。
- 不改模型结构、loss 或正式输出目录。

### P6a.5：expert system boundary review only

目标：在 P6a `PredictionCacheExpertProvider` 之后、P6b evaluation adapter 之前，冻结专家系统边界，明确 `ExpertProvider / ExpertBatch` 是 Time framework 长期专家系统契约，而 `PredictionCacheExpertProvider` 只是当前 Stage 1 canonical experiment 的 prediction-cache adapter implementation。

当前状态（2026-06-19）：已完成文档化边界审计；本阶段只新增 `docs/refactor/expert_system_boundary_review.md` 并更新相关文档索引，不修改 provider/reader 行为，不实现 evaluator adapter、runtime、config、launcher 或正式入口迁移。

本次完成范围：

- 明确 `ExpertProvider` 是专家系统边界，不是 prediction cache 边界。
- 明确 `ExpertBatch` 是下游 Router / Fusor / Evaluator 的统一专家输出载体。
- 明确 cache 是 implementation，不是 interface。
- 明确固定五专家顺序属于 Stage 1 canonical experiment 契约，不上升为 Time framework 全局专家系统契约。
- 明确当前 P6a provider 可以保留固定五专家顺序校验，因为它服务的是 Stage 1 canonical experiment。
- 明确未来 `ExpertProvider` 可以来自 prediction cache、statistical baselines、online expert models、external expert systems、dynamic expert pools 和 TimeFuse-style fusor branch 所需专家输出。
- 明确 Visual Router 主线和 TimeFuse-style fusor 支线后续都应依赖 `ExpertBatch` / protocol types，而不是直接绑定 packed prediction cache。
- 明确 P6b evaluation adapter 后续应消费 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights，不重新读取 prediction cache。
- 明确 `ExpertProvider` 不承担 feature generation、oracle/TSF supervision、loss、evaluation、runtime artifact、run_dir 或 Bash launcher 等职责。

明确不做：

- 不改 `PredictionBatchReader` 行为。
- 不改 `PredictionCacheExpertProvider` smoke 语义。
- 不移动 `prediction_array_io`。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 `status.json` / `metadata.json`。
- 不实现 config system。
- 不实现 runtime / launcher。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不修改 Visual Router / TimeFuse fusor 正式入口。
- 不改模型结构、loss 或正式输出目录。
- 不新增正式 provider abstraction 代码。

### P6b：minimal EvaluationInput adapter from ExpertBatch + RouterOutput/weights

目标：在 P6a `PredictionCacheExpertProvider` 和 P6a.5 Expert System Boundary Review 之后，新增最小 `EvaluationInputAdapter`。该 adapter 只把 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 包装为 `EvaluationInput`，再调用 `time_router.evaluation` public API 复算 summary 与 per-sample rows。

当前状态（2026-06-20）：已完成 smoke-only adapter；新增 `time_router/evaluation/evaluation_input_adapter.py`、`tests/smoke/stage1_evaluation_input_adapter_smoke.py` 和 `docs/refactor/evaluation_input_adapter.md`。此前的 `FusionEvaluator` adapter 继续作为兼容层保留。本阶段不接 Visual Router / TimeFuse fusor 正式训练入口，不新增 runtime、launcher、config、run_dir 或文件写出。

P6c consolidation（2026-06-20）：已收束 evaluation adapter 命名和职责。`EvaluationInputAdapter` 是 canonical adapter；`FusionEvaluator` 只作为 legacy/compat wrapper，内部委托 `EvaluationInputAdapter`，不再保留独立 metrics/summary/rows 复算逻辑。

本次完成范围：

- 新增 `EvaluationInputAdapter` 和 `EvaluationInputAdapterResult`，并从 `time_router.evaluation` public API 导出。
- `EvaluationInputAdapter.build_evaluation_input(...)` 检查 `sample_keys` 与 `model_columns` 在 `ExpertBatch` / `RouterOutput` 两侧完全一致。
- adapter 原样复用 `ExpertBatch.y_pred`、`ExpertBatch.y_true` 和 `RouterOutput.weights` 或显式 fusion weights，不重新读取 prediction cache。
- canonical adapter 内部只在 `EvaluationInputAdapter.evaluate_input(...)` 调用 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 和 `build_per_sample_fusion_rows`。
- 输出保持为纯内存 `summary`、`per_sample_rows`、`hard_result`、`raw_soft_result`、`evaluation_input` 和轻量 `extra`。
- smoke 使用 golden fixture，经 `PredictionCacheExpertProvider` 显式加载 golden sample_keys 得到 `ExpertBatch`，再用 golden weights 构造 `RouterOutput`，并覆盖显式 fusion weights 输入路径。
- `FusionEvaluator` 兼容 smoke 只检查旧路径不漂移，并确认 diagnostics 中 `canonical_adapter_name=EvaluationInputAdapter`。
- smoke 检查 sample_keys 保序、固定五专家顺序、summary golden 数值、per-sample rows 字段集合和逐样本 hard/raw-soft MAE/MSE、max_weight、weight_entropy。
- smoke 在 adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，证明 adapter 不重新读取 prediction cache、oracle/TSF 或其他文件。
- smoke 检查 `experiment_logs/run_outputs/` 一层目录集合不变，证明 adapter 不创建正式输出目录。

明确不做：

- 不修改 `PredictionBatchReader` 行为。
- 不修改 `PredictionCacheExpertProvider` 行为。
- 不重新读取 manifest。
- 不重新读取 packed npy。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 `status.json` / `metadata.json`。
- 不写 CSV / JSON / Parquet。
- 不实现 runtime / launcher。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不实现 config system。
- 不迁移 Visual Router / TimeFuse fusor 正式入口。
- 不新增 calibration / temperature / top-k。
- 不新增 oracle regret。
- 不读取 oracle/TSF。
- 不改模型结构、loss 或正式输出目录。

### P7a：minimal TimeFuseFeatureCacheProvider smoke-only

目标：基于 P5c protocol types、P5d adapter boundary 和 P5e/P5f migration/launcher 设计，新增最小 `TimeFuseFeatureCacheProvider`，只把显式 feature CSV 中的小规模 TimeFuse 17 维 feature batch 包装为 `FeatureBatch`，先供 smoke 使用，不接正式 TimeFuse fusor / Visual Router 入口。

当前状态（2026-06-20）：已完成 P7a 最小 adapter 与 smoke；本阶段新增 `time_router/features/`，但不修改正式训练入口，不实现 runtime、config、launcher、run_dir helper 或 scaler。

本次完成范围：

- 新增 `time_router/features/__init__.py`，导出 `TimeFuseFeatureCacheProvider`。
- 新增 `time_router/features/timefuse_cache.py`，只读取调用方显式传入的 feature CSV。
- `TimeFuseFeatureCacheProvider.load_batch(sample_keys)` 要求调用方显式传入非空、不重复 sample_keys。
- 输出 `FeatureBatch.sample_keys` 保持调用方顺序，`features` 当前为 `numpy.float32` array。
- `feature_schema` 记录 `feature_schema_name`、`feature_columns`、`feature_dim` 和 `source`。
- `extra` 只记录 provider name、sample_key 列、feature CSV 路径、可用 feature 行数和 dtype。
- 新增 `tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py`，使用测试内临时 feature CSV，并阻断除该 CSV 之外的文件读取与 `np.load`。

明确不做：

- 不读取 prediction cache。
- 不读取 oracle/TSF、`y_true` 或 expert error。
- 不做 scaler fit；scaler 属于 training/runtime。
- 不创建 `run_dir`。
- 不写 status/metadata/CSV/JSON/Parquet。
- 不访问 `/data2`。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不迁移正式 TimeFuse fusor / Visual Router 入口。

### P7b：minimal TimeFuseLinearSoftmaxHead smoke-only

目标：新增最小 `TimeFuseLinearSoftmaxHead`，只把 `FeatureBatch.features` 转成 `RouterOutput(logits, weights)`，先供 smoke 使用，不接正式 TimeFuse fusor / Visual Router 入口。

当前状态（2026-06-20）：已完成 P7b 最小 adapter 与 smoke；本阶段新增 `time_router/models/`，但不修改正式训练入口，不实现训练、loss、optimizer、checkpoint、runtime 或 launcher。

本次完成范围：

- 新增 `time_router/models/__init__.py`，导出 `TimeFuseLinearSoftmaxHead`。
- 新增 `time_router/models/timefuse_linear.py`，使用纯 numpy 固定线性层和 stable softmax。
- `TimeFuseLinearSoftmaxHead.predict(feature_batch, model_columns)` 输出 `RouterOutput`。
- `sample_keys` 保持 `FeatureBatch.sample_keys` 顺序。
- `logits` / `weights` 的专家维度与 `model_columns` 对齐。
- `weights` 沿专家维度 softmax，逐样本和为 1。
- 新增 `tests/smoke/stage1_timefuse_linear_head_smoke.py`，使用固定小矩阵和固定权重验证 deterministic 输出，并阻断文件 IO、`np.load`、`np.save/np.savez`。

明确不做：

- 不训练。
- 不计算 loss。
- 不创建 optimizer。
- 不保存 checkpoint。
- 不读取 prediction cache、oracle/TSF 或 feature CSV。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 status/metadata/CSV/JSON/Parquet。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不迁移正式 TimeFuse fusor / Visual Router 入口。

### P7c：TimeFuse protocol chain smoke-only

目标：新增一个 smoke-only TimeFuse protocol chain，把已完成 adapter 串起来验证协议对象可组合：

```text
PredictionCacheExpertProvider -> ExpertBatch
TimeFuseFeatureCacheProvider -> FeatureBatch
TimeFuseLinearSoftmaxHead -> RouterOutput
EvaluationInputAdapter -> summary / per-sample rows
```

当前状态（2026-06-20）：已完成 P7c 链路 smoke 与文档；本阶段只新增 `tests/smoke/stage1_timefuse_protocol_chain_smoke.py` 和 `docs/refactor/timefuse_protocol_chain_smoke.md`，不修改正式训练入口。

本次完成范围：

- 使用 golden prediction fixture 构造 `ExpertBatch`。
- 使用测试内临时 TimeFuse feature CSV 构造 `FeatureBatch`。
- `FeatureBatch.sample_keys` 必须与 `ExpertBatch.sample_keys` 对齐。
- 使用固定 `TimeFuseLinearSoftmaxHead` 权重生成 `RouterOutput`。
- 使用 `EvaluationInputAdapter` 复算内存 summary 和 per-sample rows。
- 锁定 `sample_keys`、`model_columns`、features、weights、summary 和 rows 的 deterministic 输出。
- head/evaluator 阶段阻断 `open`、`Path.open`、`np.load`、`np.save` 和 `np.savez`，并检查 `run_outputs` 一层目录集合不变。

明确不做：

- 不训练。
- 不计算 loss。
- 不创建 optimizer。
- 不保存 checkpoint。
- 不访问 `/data2`。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不创建 `run_dir`。
- 不写 status/metadata/CSV/JSON/Parquet。
- 不迁移正式 TimeFuse fusor / Visual Router 入口。

### P8a：TimeFuse entrypoint adapter insertion audit only

目标：在 P7c TimeFuse protocol chain smoke 之后，审计正式入口 `train_timefuse_fusor_streaming.py` 的最小 `EvaluationInputAdapter` 接入点；本阶段只做文档化接入计划，不改正式训练入口行为。

当前状态（2026-06-20）：已完成 P8a 文档化审计；新增 `docs/refactor/timefuse_entrypoint_adapter_insertion_audit.md`，明确最小接入点是 `evaluate_streaming(...)` 中 torch fusor 输出 `weights_np` 后的 batch evaluation 阶段。

本次完成范围：

- 明确 `evaluate_streaming(...)` 已同时持有 `sample_keys`、`MODEL_COLUMNS`、`batch.y_pred`、`batch.y_true` 和 `weights_np`，最适合先构造 `EvaluationInput` 或等价 adapter 输入。
- 明确可复用 `time_router.evaluation` public API：`EvaluationInputAdapter.evaluate_input(...)`、hard top-1、raw soft fusion、summary、per-sample rows 和 weight diagnostics helper。
- 明确 CSV 写出、`summary.md`、checkpoint/status/metadata、scaler fit、optimizer/loss/epoch loop、reader/index 准备和正式 oracle regret 字段必须暂留正式入口或 runtime/report 层。
- 明确 P7a `TimeFuseFeatureCacheProvider` 只是 smoke-only 小规模 CSV adapter，不能直接替换 full-scale streaming reader。
- 明确 P7b `TimeFuseLinearSoftmaxHead` 只是 numpy smoke head，不能直接替换 torch 训练 head。
- 给出 P8b 最小代码迁移建议：优先只在 evaluation 阶段旁路使用 `EvaluationInputAdapter` 复算 batch metrics，不改变正式输出 schema。

明确不做：

- 不修改 `train_timefuse_fusor_streaming.py` 行为。
- 不迁移正式入口。
- 不改 CSV / summary / checkpoint / status / metadata schema。
- 不改 reader、scaler、optimizer、loss、epoch loop。
- 不访问 `/data2`。
- 不新增 Bash 或 `scripts/`。
- 不改 Visual Router 入口。

### P8d：TimeFuse baseline parity review docs only

目标：在 P8c evaluation adapter pressure verification 之后，补充 TimeFuse-style fusor baseline parity review，明确当前 baseline 保留了哪些 TimeFuse-style 核心、主动改造了哪些部分，以及后续论文和实验汇总中可声称/不可声称的边界。

当前状态（2026-06-20）：已完成文档化审计；新增 `docs/refactor/timefuse_baseline_parity_review.md`，本阶段只更新文档、结构索引和中文实验日志，不修改训练代码，不访问 `/data2`，不启动新实验。

本次完成范围：

- 明确当前 baseline 保留 `meta feature -> linear logits`、softmax expert weights、sample-level adaptive fusion、weighted prediction fusion 和 `SmoothL1Loss` 训练口径。
- 明确当前 baseline 有意从原版 TimeFuse 多变量设置改为单变量 sample-level 融合，特征集合改为当前 17 维 TimeFuse-derived feature，专家集合改为 QuitoBench 五专家，prediction 来源改为 Stage 1 packed prediction cache / streaming reader，evaluation 接入 `ExpertBatch` / `EvaluationInputAdapter` 口径。
- 明确不能声称完全复现原版 TimeFuse、不能直接和原论文数值一一复现比较、不能把当前 baseline 写成未改造的 TimeFuse。
- 明确可声称 `TimeFuse-style fusor baseline`、`TimeFuse-inspired sample-level adaptive expert fusion baseline` 和 `adapted TimeFuse-style baseline for single-variable QuitoBench expert routing`。
- 对照 `train_timefuse_fusor_streaming.py` 记录已保持的 torch `Linear -> softmax`、`StandardScaler`、`SmoothL1Loss beta`、weighted fusion loss 和 streaming train/eval 细节。
- 记录后续如需更强 parity，应再审原版特征定义、训练 split/loss/normalization、多变量处理方式和当前 17 维 feature 与原版 feature 的对应关系。

明确不做：

- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `TimeFuseFusor`、reader、scaler、loss 或 evaluation adapter。
- 不新增 smoke。
- 不新增 Bash 或 `scripts/`。
- 不访问 `/data2`。
- 不启动 pressure 或 full-scale。
- 不改 Visual Router 入口。

### P9a：Visual Router entrypoint adapter insertion audit only

目标：在 P8d TimeFuse baseline parity review 之后，回到 Visual Router 主线，审计正式入口 `train_visual_router_online_streaming.py` 的最小 adapter 接入点；本阶段只做文档化接入计划，不改正式 Visual Router 训练入口行为。

当前状态（2026-06-20）：已完成 P9a 文档化审计；新增 `docs/refactor/visual_router_entrypoint_adapter_insertion_audit.md`，明确 Visual Router 第一批接入点应比 TimeFuse 更保守。

本次完成范围：

- 梳理 `train_visual_router_online_streaming.py` 当前承担的 sample/manifest/prediction 读取、Quito history window、online pseudo image、ViT forward、router head/loss/optimizer、evaluation rows/summary、checkpoint/status/metadata/run_dir 职责。
- 判断 `PredictionCacheExpertProvider / ExpertBatch` 应优先规划为专家输出 contract，但 P9b 不应直接替换正式入口的 SQLite prediction index 与 batch query。
- 判断 `EvaluationInputAdapter` 可作为第一批旁路 evaluation 校验点，在 test batch 内用当前 sample_key、五专家预测、共享 `y_true` 和 router weights 复算 hard/raw-soft/diagnostics。
- 明确 Visual FeatureProvider / ViT provider 不作为第一批最小 adapter，因为它绑定 Quito 数据读取、pseudo image、Hugging Face ViT、GPU dtype、DataParallel、latency、scaler 和 metadata。
- 明确 P9b 最小代码迁移建议：只在 evaluation 阶段旁路使用 `EvaluationInputAdapter` 或临时 `ExpertBatch` 做对齐校验，不改变正式输出 schema。
- 对比 TimeFuse P8a-P8c，说明 Visual Router feature/provider/head 更重，入口迁移门禁必须更保守。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不迁移正式入口。
- 不改 Visual Router feature、ViT、router head、loss 或 evaluation output schema。
- 不新增 provider/head 代码。
- 不新增 smoke。
- 不新增 Bash 或 `scripts/`。
- 不访问 `/data2`。
- 不启动 pressure 或 full-scale。
- 不改 TimeFuse 正式入口。

### P9b/P9c：Visual Router evaluation adapter bypass and pressure verification

目标：在 P9a 审计之后，只在 Visual Router 正式入口的 test evaluation batch 内加入默认关闭的 `--verify-evaluation-adapter` 旁路校验，并用小规模正式入口 pressure run 证明开启该 flag 不改变正式输出 artifact。

当前状态（2026-06-20）：已完成 P9b 代码接入、P9c pressure 验证和 P9d ExpertBatch bridge。P9b 详见 `docs/refactor/visual_router_evaluation_adapter_bypass.md`；P9c 详见 `docs/refactor/visual_router_evaluation_adapter_pressure_verification.md`；P9d 详见 `docs/refactor/visual_router_expert_batch_evaluation_bridge.md`。

本次完成范围：

- P9b 在 `train_visual_router_online_streaming.py` 中新增默认关闭的 `--verify-evaluation-adapter`，只在 test evaluation batch 内用当前 batch 的 `sample_key`、`MODEL_COLUMNS`、router weights、`y_pred` 和 `y_true` 构造 `EvaluationInput`，通过 `EvaluationInputAdapter` 旁路复算 hard/raw-soft rows。
- P9b helper 逐样本校验 `selected_model`、`selected_index`、hard MAE/MSE、raw soft MAE/MSE、`max_weight` 和 `weight_entropy`；`--skip-soft-fusion` 与 verify 同开会直接报错。
- P9c 使用仓库内 `2026-06-14_stage1_full_scale_dry_run_v2` 小规模输入、CPU、`--local-files-only`、每 split 2 个样本和 1 epoch，分别运行关闭/开启 verify 的正式入口。
- P9c 比较 `visual_router_predictions.csv`、`visual_router_soft_fusion_predictions.csv`、summary、soft summary、selected counts、comparison 和 `visual_router_streaming_summary.md` 核心表格；除 run_dir 路径和生成时间这类预期差异外，归一化后正式口径一致。
- P9c 确认开启 verify 后不新增 adapter artifact，`visual_router_metadata.json`、`visual_router_online_metadata.json`、`status.json` 和 checkpoint index top-level schema 不漂移。
- P9d 新增 `build_visual_router_expert_batch_for_evaluation(...)`，只包装当前 batch 已经读取出的 legacy SQLite `y_pred/y_true` arrays 和 `MODEL_COLUMNS`，不读取 manifest，不读取 prediction cache，不创建 run_dir。
- P9d 将 `verify_evaluation_adapter_bypass_batch(...)` 的 adapter 调用从直接 `EvaluationInput` 改为 `EvaluationInputAdapter().evaluate(expert_batch=..., fusion_weights=...)`；权重仍从正式 `pred_df` 的 `weight_<model_name>` 列恢复，后续逐样本比较字段保持不变。
- P9d smoke 明确断言 `ExpertBatch.sample_keys`、`ExpertBatch.model_columns` 保序，`y_pred/y_true` 原样进入 adapter，且 failure message 仍包含 config/split/batch/sample/field/old value/adapter value/output_dir。

明确不做：

- 不继续迁移正式入口。
- 不把 P9d 解释为 `PredictionCacheExpertProvider` 正式接入。
- 不替换正式 prediction_index / SQLite index。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不迁移 `PredictionBatchReader` 到正式入口。
- 不替换 `predict_stream_batch(...)`。
- 不替换 `add_soft_fusion_metrics(...)`。
- 不改 `EvaluationInputAdapter`。
- 不新增 VisualFeatureProvider 或 ViT provider。
- 不改 `VisualMLPRouter`、router head、training loop、`fusion_huber_kl` loss、optimizer、checkpoint/resume 语义。
- 不改正式 CSV / summary / metadata / status schema。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 full-scale。
- 不改 TimeFuse 正式入口。

### P9e：Visual Router PredictionCacheExpertProvider full-scale gap audit only

目标：在 P9d ExpertBatch evaluation bridge 之后，只审计
`PredictionCacheExpertProvider` / `PredictionBatchReader` 与 Visual Router 正式入口
`SQLitePredictionIndex`、`build_lightweight_prediction_index(...)`、
`load_prediction_tensors_from_lightweight_index(...)` 的能力差距，并给出后续迁移路线。

当前状态（2026-06-20）：已完成 P9e 文档化审计；新增
`docs/refactor/visual_router_prediction_cache_provider_gap_audit.md`，本阶段不修改正式入口，
不替换 SQLite index，不新增 full-scale provider 实现。

本次完成范围：

- 梳理 `PredictionCacheExpertProvider` 当前已具备的显式 sample_keys batch 输入、固定五专家
  `model_columns`、`y_pred/y_true`、`row_index_metadata`、`verify_metrics`、
  `packed_npy_v1` / `per_sample_npy` 和 `ExpertBatch` 输出能力。
- 梳理 Visual Router 正式 SQLite path 仍承担的 `required_prediction_sample_keys(...)`、
  大 manifest chunk scan、SQLite 子集索引、batch query、packed row index 读取、
  `fusion_huber_kl` 训练 loss 所需 `expert_errors`、eval raw soft fusion lookup、
  `prediction_manifest_index.sqlite` runtime artifact 和 index metadata。
- 明确 provider 应只负责 `load_batch(sample_keys) -> ExpertBatch`、sample/model 保序、
  读取 `y_pred/y_true` 和保留 row index lineage，不应创建 run_dir、写 status/metadata/CSV、
  推导 split、决定 SQLite path、管理 checkpoint/resume、执行 full-scale preflight 或绑定 `/data2`。
- 比较三种迁移方案：A 保留 Visual SQLitePredictionIndex 并在 batch 后包装 ExpertBatch；
  B 给 provider 增加可选 prepared index / batch query 后端；C 直接用 `PredictionBatchReader`
  替换 Visual SQLite path。结论是 A 风险最低，B 可作为 Stage 1.5 候选，C 默认不建议。
- 明确 P9f/P10 建议：先做 training loss ExpertBatch bypass check，再抽 shared prediction index
  prepare helper，再整理 launcher/run script 层，真正 provider 替换推迟到 Stage 1.5 / Stage 2。

明确不做：

- 不修改 `train_visual_router_online_streaming.py` 行为。
- 不替换 `SQLitePredictionIndex`。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不迁移 `PredictionBatchReader` 到正式入口。
- 不改 `PredictionBatchReader`、`PredictionCacheExpertProvider` 或 `EvaluationInputAdapter`。
- 不改 VisualFeatureProvider / ViT provider / router head / training loop / `fusion_huber_kl` loss。
- 不改 checkpoint / status / metadata / CSV schema。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改 TimeFuse 正式入口。

### P9f：Visual Router training loss ExpertBatch bypass check

目标：在 P9e gap audit 之后，只为 Visual Router `fusion_huber_kl` training batch
增加默认关闭的 `--verify-training-expert-batch` 旁路校验，验证 legacy SQLite path
返回的 `y_pred/y_true/expert_errors` 可由 `ExpertBatch.y_pred/y_true` 显式表达。

当前状态（2026-06-20）：已完成 P9f 代码、smoke 和文档；新增
`docs/refactor/visual_router_training_expert_batch_bypass.md` 与
`tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py`。本阶段不改变默认
训练行为，不替换 SQLite index，不接 `PredictionCacheExpertProvider` 到正式入口。

本次完成范围：

- 在 `train_visual_router_online_streaming.py` 新增默认关闭 flag
  `--verify-training-expert-batch`。
- 新增 `verify_training_expert_errors_from_expert_batch(...)`，只包装当前 training batch
  已读取出的 legacy `y_pred/y_true`，并从 `ExpertBatch.y_pred/y_true` 复算 MAE/MSE
  `expert_errors`。
- flag 只在 `router_mode == "fusion_huber_kl"` 的 training batch 内生效；
  `classification` 同开时直接报错。
- 校验失败信息包含 `phase=training`、`router_mode=fusion_huber_kl`、metric、
  batch index、sample_key、model_name、expert_index、legacy/recomputed value 和 output_dir。
- 新 smoke 使用纯内存 numpy arrays 覆盖 MAE/MSE 一致性与故意 mismatch 的定位信息，
  不启动 ViT、不访问 `/data2`、不运行正式入口。

明确不做：

- 不改变默认训练行为。
- 不替换正式 `expert_errors`、loss、optimizer、scheduler、scaler 或 checkpoint/resume。
- 不改变 CSV / summary / metadata / status schema。
- 不替换 `SQLitePredictionIndex`。
- 不迁移 `PredictionBatchReader` 或 `PredictionCacheExpertProvider` 到正式入口。
- 不改 `PredictionBatchReader`、`PredictionCacheExpertProvider` 或 `EvaluationInputAdapter`。
- 不改 VisualFeatureProvider / ViT provider / router head。
- 不新增 Bash/scripts。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改 TimeFuse 正式入口。

P9f 之后，下一步应进入 shared prediction SQLite backend / index prepare consolidation，
而不是直接抽 VisualFeatureProvider / ViT provider，也不是直接用
`PredictionBatchReader` 替换 Visual Router 正式 SQLite path。

### P10a：shared prediction SQLite backend audit

目标：在 P9f 完成 Visual Router training/evaluation ExpertBatch 旁路对齐之后，只做
shared prediction SQLite backend / index prepare 文档化审计，明确 Visual Router 与
TimeFuse-style fusor 现有 SQLite path 的共同 backend 能力和不可共享边界。

当前状态（2026-06-20）：已完成文档化审计，详见
`docs/refactor/shared_prediction_sqlite_backend_audit.md`。本阶段不改 Visual Router /
TimeFuse 正式入口，不抽 helper 代码，不修改 provider/reader/adapter，不新增 Bash/scripts，
不访问 `/data2`，不启动 pressure/full-scale。

审计结论：

- Visual Router 当前 SQLite path 包含 `required_prediction_sample_keys(...)`、
  `build_lightweight_prediction_index(...)`、`SQLitePredictionIndex.fetch_records(...)`、
  `load_prediction_tensors_from_lightweight_index(...)`、run_dir 下
  `prediction_manifest_index.sqlite`、`fusion_huber_kl` expert_errors 和 eval soft lookup。
- TimeFuse-style fusor 当前 path 包含 feature-only scaler、feature shard split 下推、
  shard-local oracle/prediction SQLite index、batch 查询和 packed npy grouped mmap 读取。
- 可共享的是 prediction SQLite backend implementation：manifest chunk scan、target
  sample_keys、SQLite 子集索引、batch fetch records、packed row index lineage、
  grouped mmap loading、index metadata 和 atomic replace / cleanup。
- 不应共享到 backend 的是 Visual Quito/pseudo image/ViT、TimeFuse 17 维 feature/scaler、
  training loss/optimizer/checkpoint、status/metadata/CSV、launcher/Bash、`/data2` 路径绑定。
- `ExpertProvider` 边界仍是 `load_batch(sample_keys) -> ExpertBatch`；provider 可消费
  prepared backend，但不创建 run_dir、不写 status/metadata、不知道 launcher 或 `/data2`。
- prediction backend 可进入 `ExpertProvider`；oracle 只用于监督、诊断、baseline 和
  upper-bound，不进入 deployable `FeatureProvider`。

后续建议：

- P10b 已先抽 shared index prepare smoke helper。
- P10c 已先做 prediction array IO boundary consolidation，消除 `time_router.io`
  对 `visual_router_experiments.common` 的反向依赖。
- launcher / run scripts 边界整理继续后移；P10d 已改为先冻结 canonical
  SampleManifest / Split / Supervision 边界。
- provider prepared backend 推迟到 Stage 1.5 / Stage 2。

### P10b：minimal shared prediction SQLite backend smoke helper

目标：在 P10a 审计边界内，只新增最小 shared prediction SQLite backend helper 和
小规模 fixture smoke，验证 manifest chunk scan、target sample_keys 子集索引、batch
fetch records、metadata、missing report 和 atomic replace / cleanup。不接 Visual Router /
TimeFuse 正式入口。

当前状态（2026-06-20）：已完成 `time_router/io/prediction_sqlite_backend.py`、
`tests/smoke/stage1_prediction_sqlite_backend_smoke.py` 和
`docs/refactor/prediction_sqlite_backend.md`。新增 helper 只由 smoke 使用，不替换
`SQLitePredictionIndex`、`PredictionBatchReader`、`PredictionCacheExpertProvider` 或
`EvaluationInputAdapter`。

本次完成范围：

- `build_prediction_sqlite_backend(...)` 接收 `manifest_path`、`target_sample_keys`、
  `index_db_path`、`model_columns` 和 `chunk_read_rows`。
- SQLite `prediction_index` 使用 `(sample_key, model_name)` 主键，记录 `y_pred/y_true`
  path、MAE/MSE、`array_storage` 和 packed row index。
- 构建过程分块读取 manifest，只写目标 sample/model records，不加载全量 manifest 到内存。
- 构建使用同目录临时 SQLite 文件，成功后 `os.replace(...)` 原子替换；失败时删除临时文件。
- `PreparedPredictionSQLiteBackend.fetch_records(sample_keys)` 返回当前 batch records，调用方
  可用 `records_to_ordered_rows(...)` 按 `sample_keys + model_columns` 恢复顺序。
- metadata 记录 target keys、expected/actual records、chunk rows、model columns、
  manifest/index path、created_at 和 missing sample/model report。
- smoke 使用临时 packed fixture 构造 4 sample × 5 model，验证 fetch 保序、row index
  lineage、grouped packed loading、shape、缺失报错和 `allow_missing=True` missing report。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接正式入口，不新增 Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

### P10c：prediction array IO boundary consolidation

目标：在 P10b shared prediction SQLite backend helper 之后，只整理 prediction array
IO 的 package 边界，把通用 `packed_npy_v1` / `per_sample_npy` 数组读取能力上收到
`time_router.io`，消除 `time_router.io` 对 `visual_router_experiments.common` 的反向依赖。

当前状态（2026-06-20）：已完成 `time_router/io/prediction_array_io.py`，并把旧
`visual_router_experiments/common/prediction_array_io.py` 改为 compatibility wrapper。
本阶段未修改 Visual Router / TimeFuse 正式入口行为，未替换现有 SQLite index，
未改 provider/head/runtime、loss、optimizer、scaler、checkpoint/resume 或正式输出 schema。

本次完成范围：

- `time_router/io/prediction_array_io.py` canonical 提供 `PACKED_NPY_STORAGE`、
  `PER_SAMPLE_NPY_STORAGE`、`resolve_cache_array_path(...)`、
  `load_prediction_array(...)` 和 `load_prediction_arrays_grouped(...)`。
- `time_router/io/__init__.py` 导出上述 public API。
- `time_router/io/prediction_sqlite_backend.py` 和
  `time_router/io/prediction_cache_reader.py` 改为依赖
  `time_router.io.prediction_array_io`。
- `tests/smoke/stage1_prediction_sqlite_backend_smoke.py` 改为从 `time_router.io`
  导入 grouped array loading helper。
- `visual_router_experiments/common/prediction_array_io.py` 保留旧 import 路径，只
  re-export canonical implementation，避免旧脚本断裂。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接正式入口，不替换 Visual Router / TimeFuse 现有 SQLite index。
- 不改 `PredictionCacheExpertProvider` 或 `EvaluationInputAdapter` 行为。
- 不新增 provider/head/runtime 代码。
- 不新增 Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

### P10d：canonical SampleManifest and supervision boundary design

目标：在用户接受必要时重跑 Stage 1 的前提下，只用文档冻结 Visual Router 与
TimeFuse-style fusor 可共用的 canonical `SampleManifest` / `SplitStrategy` /
`SupervisionProvider` 边界。此步不实现代码，不修改正式入口，不把历史 labels CSV、
feature CSV、oracle SQLite/parquet 或 runtime artifact schema 继续作为最高兼容目标。

当前状态（2026-06-20）：已新增
`docs/refactor/stage1_canonical_sample_supervision_boundary.md`，并同步更新 target
architecture、entrypoint migration plan 和结构索引。正式入口、provider、reader、adapter、
launcher、loss 和输出 schema 均未修改。

设计结论：

- `SampleManifest` 是 Stage 1 的 canonical sample source，至少包含 `sample_key`、
  `split`、`config_name`、`dataset_name`、`item_id`、`channel_id`、`window_index`，
  并推荐包含 `seq_len` / `pred_len` 和轻量 lineage/extra。
- `SplitStrategy` 长期负责生成或校验 split，并按 split 输出 ordered `sample_keys`；
  split 不再由 labels CSV、feature CSV、oracle reader 和 prediction reader 各自推导。
- Visual Router 与 TimeFuse-style fusor 必须消费同一套 `sample_key` 顺序。
- `ExpertProvider / ExpertBatch` 只提供 `y_pred/y_true`、`model_columns` 和 row index
  lineage。
- `SupervisionProvider` 或 `OracleLabelProvider` 提供 `oracle_model`、`oracle_value`、
  `per_model_errors` / `model_error_matrix`、`model_columns`、`metric` 和 `extra`。
- oracle / per-model error 只用于训练监督、诊断、baseline 或 upper-bound，不进入
  deployable `FeatureProvider`。
- Visual Router 当前 labels CSV 和 TimeFuse 当前 feature/oracle/prediction 文件分工是历史实现差异，
  不是长期接口边界。

允许重跑后的 canonical 路线：

1. 先生成或冻结 `SampleManifest`。
2. 再由 `SampleManifest` 驱动 prediction backend、supervision backend 和 feature provider。
3. Visual Router / TimeFuse 都以 `SampleManifest` 为主索引训练和评估。
4. 新 run artifact schema 可在 P11/P12 冻结；旧 schema 不再强行作为最高兼容目标。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- P10d 本步不新增正式 provider 代码、不新增 `SampleManifest` 代码、不修改
  `time_router/protocols/types.py`；后续 P10e 已另起小步新增最小协议骨架。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。

### P10e：canonical SampleManifest / SupervisionBatch protocol smoke

目标：在 P10d 文档冻结的边界上，只新增最小 canonical `SampleManifest` /
`SupervisionBatch` lightweight dataclass/helper 与小规模 smoke，用于锁定
sample identity、split、ordered sample_keys 和 supervision shape 对齐的 public API 雏形。

当前状态（2026-06-20）：已在 `time_router/protocols/types.py` 新增
`SampleManifestRow`、`SampleManifest` 和 `SupervisionBatch`，并从
`time_router.protocols` 导出；新增
`tests/smoke/stage1_sample_supervision_protocol_smoke.py`。本步未修改 Visual Router /
TimeFuse-style fusor 正式入口，未接 provider，未访问 `/data2`，未改变正式输出 schema。

本次完成范围：

- `SampleManifestRow` 保存 `sample_key`、`split`、`config_name`、`dataset_name`、
  `item_id`、`channel_id`、`window_index`、可选 `seq_len/pred_len` 和 `extra`。
- `SampleManifest` 使用 rows 原始顺序返回 ordered sample keys，支持按 split 过滤、
  `split_counts()` 和 `validate_unique_sample_keys()`。
- `SupervisionBatch` 保存 `sample_keys`、`model_columns`、`metric`、`oracle_model`、
  `oracle_value`、`per_model_errors` 和 `extra`。
- `SupervisionBatch.validate_shapes()` 只校验 `per_model_errors` 的
  `[sample, expert]` 维度、`oracle_model/oracle_value` 第一维与显式顺序字段对齐；
  array-like 字段仍为 `Any`，不绑定 numpy/torch 强类型。
- smoke 构造 4 行 vali/test manifest、两个 split 的 supervision batch、5 个专家列，
  覆盖保序、split 统计、metric、shape、重复 sample_key 和 shape mismatch 报错。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接 Visual Router / TimeFuse 正式训练。
- 不新增 Visual/TimeFuse provider 实现。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

P10e 验收：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

### P10f：Visual labels to SampleManifest / SupervisionBatch smoke adapter

目标：在 P10e protocol skeleton 之上，只新增 Visual labels CSV / DataFrame 到
canonical `SampleManifest` 与 `SupervisionBatch` 的最小 adapter，验证历史 Visual labels
表可以被拆解为 sample manifest、split 和 supervision 三类职责。

当前状态（2026-06-20）：已新增 `time_router/data/visual_labels_adapter.py`、
`tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py` 和
`docs/refactor/visual_labels_sample_supervision_adapter.md`，并从 `time_router.data`
导出 adapter 函数。本步未修改 Visual Router / TimeFuse-style fusor 正式入口，未访问
`/data2`，未改变正式输出 schema。

本次完成范围：

- `visual_labels_to_sample_manifest(...)` 支持小型 `pd.DataFrame` 或 CSV 路径输入，
  构造 `SampleManifest` 并保持 labels 原始行顺序。
- manifest 字段覆盖 `sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、
  `channel_id`、`window_index` 和可选 `seq_len/pred_len`。
- `SampleManifestRow.extra` 只保存白名单 lineage，例如 `manifest_shard`，不保存
  oracle/error 字段。
- `visual_labels_to_supervision_batch(...)` 按显式 `sample_keys + model_columns + metric`
  保序输出 `SupervisionBatch`。
- smoke fixture 使用 `{model_name}_{metric}_error` 作为最小 per-model error 列约定，
  推导 `oracle_model`、`oracle_value` 和 `[sample, expert]` 误差矩阵。
- smoke 覆盖 4 行 vali/test labels、五专家列、CSV/DataFrame 双入口、缺失专家列、
  重复 `sample_key` 和未知 split 报错。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接 Visual Router / TimeFuse 正式训练。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

P10f 验收：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

后续建议：做 TimeFuse feature/oracle 到 `SampleManifest` / `SupervisionBatch` 的 smoke adapter；
正式入口接入前，仍需对齐真实 Visual labels schema 字段映射。

### P10g：TimeFuse feature/oracle to SampleManifest / SupervisionBatch smoke adapter

目标：在 P10f 之后，只新增 TimeFuse feature/oracle DataFrame / CSV 到 canonical
`SampleManifest` 与 `SupervisionBatch` 的最小 adapter，验证 TimeFuse-style fusor 当前
feature source 与 oracle/supervision source 可以拆解到同一 sample/split/supervision 边界。

当前状态（2026-06-20）：已新增 `time_router/data/timefuse_supervision_adapter.py`、
`tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py` 和
`docs/refactor/timefuse_sample_supervision_adapter.md`，并从 `time_router.data` 导出 adapter
函数。本步未修改 Visual Router / TimeFuse-style fusor 正式入口，未访问 `/data2`，未改变
正式输出 schema。

本次完成范围：

- `timefuse_features_to_sample_manifest(...)` 支持小型 `pd.DataFrame` 或 CSV 路径输入，
  构造 `SampleManifest` 并保持 feature source 原始行顺序。
- manifest 字段覆盖 `sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、
  `channel_id`、`window_index` 和可选 `seq_len/pred_len`。
- `SampleManifestRow.extra` 只保存 `feature_shard`、`feature_schema_version` 等轻量 lineage，
  不保存 17 维 TimeFuse feature 值。
- `timefuse_oracle_to_supervision_batch(...)` 按显式 `sample_keys + model_columns + metric`
  保序输出 `SupervisionBatch`。
- smoke fixture 使用 `{model_name}_{metric}_error` 作为最小 per-model error 列约定，
  推导 `oracle_model`、`oracle_value` 和 `[sample, expert]` 误差矩阵。
- smoke 覆盖 4 行 vali/test feature、17 维 feature 列、对应 4 行 oracle、五专家列、
  CSV/DataFrame 双入口、缺失 oracle 专家列、feature duplicate sample_key、oracle 缺失
  sample_key 和未知 split 报错。

明确不做：

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不接 Visual Router / TimeFuse 正式训练。
- 不改 `TimeFuseFeatureCacheProvider`。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

P10g 验收：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

后续建议：Visual labels 与 TimeFuse feature/oracle 均已有 canonical sample/supervision
adapter smoke；正式入口接入前，先对齐真实 full-scale schema 字段映射，再冻结
SampleManifest / SupervisionProvider 物理存储和缺失策略。

### P6：migrate visual router and TimeFuse fusor entrypoints

目标：让两个正式入口逐步消费共享 provider chain、metrics/report 和 runtime helper，但保留各自 head、loss 与实验变量。

范围：

- Visual Router 入口迁移到 `SplitStrategy + ExpertProvider + VisualFeatureProvider + Visual RouterHead + Evaluator`。
- TimeFuse fusor 入口迁移到 `SplitStrategy + ExpertProvider + TimeFuseFeatureProvider + Linear-softmax RouterHead + Evaluator`。
- 保留 eval-only、train-only、checkpoint/resume、DataParallel 和现有 full-scale 路径兼容。

门禁：

- 每个入口迁移前后运行 golden smoke。
- 对小规模已知输出做逐样本 predictions/summary comparison。
- full-scale 只在小规模与 schema 对齐后启动；不能用重构首跑覆盖旧可引用结果。

## 3. Archive 触发条件

旧代码只能在以下条件全部满足后进入 `archive/`：

- 已有 package 化替代实现。
- golden smoke 在迁移前后均通过。
- 小规模或压力测试证明输出 schema 和关键指标等价。
- `WORKSPACE_STRUCTURE.md` 和实验日志记录归档原因、替代路径和保留口径。

优先归档候选：

- offline embedding cache；
- logistic regression fusor；
- old OOM routes；
- pilot-only scripts。

## 4. 本路线图明确不做

- P0 不实现新 package、不新增空目录、不做 import 改动。
- 任一迁移步骤都不同时改变实验协议、模型结构和正式结果引用口径。
- 不把 oracle/TSF 当作可部署 test-time 动态调权特征。
- 不把历史 OOM、legacy 或 pilot 结果重新包装为正式结果。
