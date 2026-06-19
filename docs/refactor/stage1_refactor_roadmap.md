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

### P5：introduce FeatureProvider interface

目标：引入显式 `FeatureProvider` 边界，把共享 reader 与路线特征生成解耦。

范围：

- `VisualFeatureProvider`：`x -> pseudo image -> frozen ViT embedding`，在线 batch 生成，不落盘 full-scale embedding。
- `TimeFuseFeatureProvider`：`sample_key -> 17维 feature cache`，支持 feature-only scaler 和 shard-aware 读取。
- 定义统一输出：`sample_keys`、`features`、dtype/device metadata 和可选诊断信息。

门禁：

- 迁移前后运行 golden smoke。
- Visual 小规模 smoke 需确认 online embedding 行为与既有入口一致。
- TimeFuse 小规模/pressure smoke 需确认 feature shape 为 17 维、split 下推和 scaler 不加载 prediction arrays。

### P6：migrate visual router and TimeFuse fusor entrypoints

目标：让两个正式入口逐步消费共享 reader、FeatureProvider、metrics/report 和 logging/path/config，但保留各自 head、loss 与实验变量。

范围：

- Visual Router 入口迁移到共享 batch reader + `VisualFeatureProvider` + Visual MLP head。
- TimeFuse fusor 入口迁移到共享 batch reader + `TimeFuseFeatureProvider` + Linear-softmax head。
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
