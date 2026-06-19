# Stage 1 共享 OracleTsfReader 抽取

日志日期：2026-06-19 17:27:26 CST

## 目的

抽取 Stage 1 共享 oracle / TSF reader，用于按 `sample_key` 批量读取 window-level oracle labels 与 TSF enrichment / TSF-cell metadata，并建立小规模只读 smoke。

## 背景

`refactor/stage1-route-audit` 分支已经完成 Stage 1 路线审计、目标架构、重构路线图、golden fixture 和 `PredictionBatchReader`。本步骤对应 roadmap P2，只新增共享 reader 和 smoke，不迁移正式 Visual Router / TimeFuse fusor 训练入口。

## 操作

1. 新增 `time_router/data/__init__.py` 和 `time_router/data/oracle_tsf_reader.py`。
2. 新增 `docs/refactor/oracle_tsf_reader.md`，记录接口、用途边界和 full-scale 后续读取要求。
3. 新增 `tests/smoke/stage1_oracle_tsf_smoke.py`，复用 `2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/` 的 4 sample fixture。
4. 更新 `docs/refactor/stage1_refactor_roadmap.md`，记录 P2 当前完成范围。
5. 更新 `WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`。

## 结果

`OracleTsfReader` 当前支持：

- `load_oracle(sample_keys, metric="mae")`
- `load_tsf(sample_keys)`
- `load_joined(sample_keys, metric="mae")`
- 显式 `sample_key` 保序；
- CSV chunk 过滤和 Parquet `pyarrow.dataset` 过滤；
- `missing_policy=error/report`；
- 冲突 duplicate sample_key 显式报错；
- oracle/TSF 一对一 join lineage 与 metadata。

尚未迁移任何正式训练入口，未改变 oracle/TSF 生成逻辑、sample_key、prediction cache schema、模型结构、loss 或正式输出目录。

## 验证

已运行以下门禁：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
python -m compileall time_router tests/smoke
```

验证结果：

- `stage1_golden_smoke.py` 通过：五专家顺序、4 个 sample_key 顺序、`y_pred=(4,5,48,1)`、`y_true=(4,48,1)`、hard top-1 MAE/MSE 和 raw soft fusion MAE/MSE 均与 golden 一致。
- `stage1_oracle_tsf_smoke.py` 通过：oracle 显式 sample_key 保序、预期 oracle label、TSF metadata、oracle/TSF join、`missing_policy=report` 缺失报告和冲突 TSF sample_key 报错均通过。
- `compileall time_router tests/smoke` 通过：新增 `time_router/data` 和 oracle/TSF smoke 均可编译。

## 结论

P2 的共享 reader 契约和小规模 smoke 已建立。oracle label 仍只作为监督、上限或诊断信息，TSF enrichment 只作为 baseline、分层汇总或诊断信息，二者不得进入可部署 FeatureProvider 或 test-time 动态调权。

## 下一步方案

1. 提交并推送 `refactor: add shared oracle tsf reader`。
2. 后续 full-scale 正式入口迁移保留到 P6；读取策略应继续使用 SQLite / shard-local / batch query，避免全量 oracle/TSF 入内存。
