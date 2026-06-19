# OracleTsfReader 设计说明

创建日期：2026-06-19

## 1. 目的

`OracleTsfReader` 是 Stage 1 P2 抽出的共享 oracle / TSF 只读 reader。它负责按 `sample_key` 批量读取 window-level oracle label 与 TSF enrichment / TSF-cell metadata，并在小规模 smoke 中验证 sample_key 覆盖、输入顺序保持、冲突重复检测、缺失报告和 oracle/TSF join。

本阶段不迁移正式训练入口，不改变 oracle/TSF 生成逻辑，不改变 sample_key、prediction cache schema、模型结构、loss 或正式输出目录。

## 2. 代码位置

- `time_router/data/oracle_tsf_reader.py`
- `time_router/data/__init__.py`
- `tests/smoke/stage1_oracle_tsf_smoke.py`

默认 smoke fixture 复用：

- `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv`
- `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest_with_tsf_cell.csv`

## 3. 接口

`OracleTsfReader(...)` 初始化参数：

- `oracle_path`：指向 `window_oracle_labels*.csv` 或 `window_oracle_labels.parquet`。
- `tsf_path`：指向 `sample_tsf_enrichment.parquet`、`manifest_with_tsf_cell.csv` 或带 TSF 字段的 oracle CSV。
- `fixture_root`：指向包含上述文件的 fixture / `merged_cache` 目录。
- `missing_policy`：`error` 或 `report`。`error` 在缺失 sample_key 时抛错；`report` 保留输入顺序并输出 missing report。
- `allow_full_scan`：默认 `False`。未显式传入 sample_key 时禁止全扫描；只允许小规模 fixture / smoke 使用 `True` 做读取契约验证，正式训练、评估、baseline、calibration 或 full-scale 入口不得设置为 `True`。

读取方法：

- `load_oracle(sample_keys, metric="mae")`
- `load_tsf(sample_keys)`
- `load_joined(sample_keys, metric="mae")`

输出 `OracleTsfBatch`：

- `sample_keys`：输出顺序，显式输入时与输入完全一致。
- `frame`：oracle、TSF 或 joined DataFrame。
- `missing_report`：记录 missing、duplicate 和 extra 情况。
- `metadata`：记录输入路径、metric、字段来源、join lineage 和用途约束。

## 4. 约束

- Oracle label 只可作为训练监督、上限或诊断信息；不得进入可部署 `FeatureProvider`，也不得用于 test-time 动态调权。
- TSF enrichment 只用于统计 baseline、分层汇总或诊断；不得和未来信息泄漏混用。
- Reader 只负责读取、校验和 join，不承担训练策略。
- 正式入口必须显式传入当前 batch / shard 的 sample_key，不得依赖 `allow_full_scan=True` 做全表扫描。
- 显式 sample_key 场景必须保持输入顺序。
- Oracle 在 `metric` 过滤后必须保证每个 sample_key 唯一。
- TSF 源若来自 `manifest_with_tsf_cell.csv`，允许五专家行携带完全一致的 TSF 元信息并折叠为单行；如果同一 sample_key 的 TSF 字段冲突，则明确报错。
- 缺失值不能用默认填充值掩盖；`missing_policy=report` 会显式输出 missing report。

## 5. Full-Scale 读取说明

当前 P2 reader 已支持 CSV chunk 过滤和 Parquet `pyarrow.dataset` 过滤，适合显式传入当前 batch / shard 的 sample_key。未显式 sample_key 的全扫描模式只允许小规模 fixture 或 smoke；正式入口禁止使用 `allow_full_scan=True` 作为临时捷径。

正式 full-scale 后续若要替换训练入口，必须采用 SQLite / shard-local / batch query 或等价方案，按当前 batch / shard 查询 oracle/TSF，避免一次性把全部 oracle/TSF 读入 Python 内存。本次提交只建立共享读取契约和小规模 smoke，不迁移正式 Visual Router / TimeFuse fusor 入口，不改变 prediction/fusion 契约、模型结构、fusion metrics 或正式输出目录。

## 6. P2 Reader Hardening 边界

本次 P2 reader hardening 只补强小规模 smoke、文档和日志边界：

- smoke 显式覆盖 `allow_full_scan` 默认禁止无 sample_key 全扫描。
- smoke 显式覆盖 `missing_policy=error` 对缺失 sample_key 报错。
- oracle/TSF 只作为监督、上限、baseline、分层汇总或诊断使用。
- oracle/TSF 不进入可部署 `FeatureProvider`，也不作为 test-time 动态调权特征。
- 不迁移 Visual Router / TimeFuse fusor 正式入口。
- 不改变 `PredictionBatchReader` 输出 shape、专家顺序、sample_key 顺序。
- 不改变 fusion metrics、模型结构或正式输出目录。
