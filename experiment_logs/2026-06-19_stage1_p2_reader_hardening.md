# Stage 1 P2 OracleTsfReader reader hardening

日志日期：2026-06-19 17:48:00 CST

## 目的

补强 Stage 1 P2 `OracleTsfReader` 的小规模 smoke、文档和日志边界，确保 oracle/TSF 只作为监督、上限、baseline、分层汇总或诊断使用，不进入可部署 `FeatureProvider` 或 test-time 动态调权特征。

## 背景

前一轮已经抽出共享 `OracleTsfReader` 并完成最小 smoke。本次工作只做 P2 reader hardening，不迁移 Visual Router / TimeFuse fusor 正式入口，不改变 prediction/fusion 契约，不改变 `PredictionBatchReader` 输出 shape、专家顺序、sample_key 顺序，也不修改 fusion metrics、模型结构或正式输出目录。

## 操作

1. 审查 `docs/refactor/oracle_tsf_reader.md`、`docs/refactor/stage1_refactor_roadmap.md` 和 `tests/smoke/stage1_oracle_tsf_smoke.py` 的当前状态。
2. 在 `tests/smoke/stage1_oracle_tsf_smoke.py` 中新增两个负向 smoke 断言：
   - `allow_full_scan` 默认 `False` 时，`load_oracle(None, metric="mae")` 必须报错，防止无 sample_key 全扫描。
   - `missing_policy="error"` 时，缺失 sample_key 的 `load_joined(...)` 必须报错且错误信息包含缺失 key。
3. 在 `docs/refactor/oracle_tsf_reader.md` 中明确：
   - 正式训练、评估、baseline、calibration 或 full-scale 入口不得设置 `allow_full_scan=True`。
   - full-scale 后续正式入口必须采用 SQLite / shard-local / batch query 或等价批查询方案。
   - 本次不迁移正式入口，不改变 prediction/fusion 契约、模型结构、fusion metrics 或正式输出目录。
4. 在 `docs/refactor/stage1_refactor_roadmap.md` 中新增 P2.5 reader hardening only 状态，记录本次范围和明确不做事项。
5. 使用 Quito 环境运行两条验收 smoke。

## 结果

验收命令均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
```

关键结果：

- golden smoke 通过，五专家顺序仍为 `['DLinear', 'PatchTST', 'CrossFormer', 'ES', 'NaiveForecaster']`，`y_pred` shape 仍为 `(4, 5, 48, 1)`，`y_true` shape 仍为 `(4, 48, 1)`。
- golden smoke hard top-1 MAE 为 `0.416048437`，raw soft fusion MAE 为 `0.410296679`，未发现 prediction/fusion 契约漂移。
- oracle/TSF smoke 通过，新增覆盖 `allow_full_scan` 默认禁止无 sample_key 全扫描。
- oracle/TSF smoke 通过，新增覆盖 `missing_policy=error` 对缺失 sample_key 明确报错。
- 原有 oracle label、TSF metadata、join 保序、`missing_policy=report` 缺失报告和冲突 TSF sample_key 报错检查继续通过。

## 结论

本次只完成 P2 reader hardening：补强小规模 smoke 和文档边界，未迁移任何正式训练入口，未改变 prediction/fusion 契约，未改变模型结构、fusion metrics 或正式输出目录。`OracleTsfReader` 仍只用于读取、校验和 join，oracle/TSF 仍限定为监督、上限、baseline、分层汇总或诊断信息。

## 下一步方案

后续如要把 oracle/TSF 接入 full-scale 正式入口，应另起 P6 入口迁移步骤，并先设计 SQLite / shard-local / batch query 读取方案；迁移前后继续运行 `tests/smoke/stage1_golden_smoke.py` 和相关 oracle/TSF smoke，确认 sample_key 顺序、五专家顺序、shape 和 fusion 指标不漂移。
