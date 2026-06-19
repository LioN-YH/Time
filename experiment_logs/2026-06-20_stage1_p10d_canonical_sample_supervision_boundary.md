# Stage 1 P10d Canonical SampleManifest 与 Supervision Boundary

日志日期：2026-06-20 04:12:19 CST

## 1. 目的

设计 Visual Router 与 TimeFuse-style fusor baseline 可共用的 canonical
`SampleManifest` / `SplitStrategy` / `SupervisionProvider` 边界。

本步骤只做架构设计和文档，不修改正式训练入口，不新增正式 provider，不改变训练行为或正式输出 schema。

## 2. 背景

P10a/P10b/P10c 已完成 shared prediction SQLite backend 审计、最小 smoke helper 和
prediction array IO 边界收敛。当前方向调整为：必要时可以重跑 Stage 1 实验，因此后续应优先构建干净、长期可扩展的系统底座，不再以完全兼容历史 labels CSV、feature CSV、oracle SQLite/parquet 或旧 runtime artifact schema 为最高优先级。

需要补齐的上层边界是：

- 样本清单和 split 的 canonical source；
- Visual Router 与 TimeFuse-style fusor 共用的 ordered `sample_keys`；
- Expert prediction 与 oracle supervision 的分层；
- oracle / per-model error 不进入 deployable FeatureProvider 的约束。

## 3. 操作

新增文档：

- `docs/refactor/stage1_canonical_sample_supervision_boundary.md`

同步更新：

- `docs/refactor/stage1_target_architecture.md`
- `docs/refactor/stage1_refactor_roadmap.md`
- `docs/refactor/stage1_entrypoint_migration_plan.md`
- `WORKSPACE_STRUCTURE.md`
- `experiment_logs/README.md`

本步骤未修改以下文件：

- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`
- `time_router/protocols/types.py`
- `PredictionBatchReader`、`PredictionCacheExpertProvider`、`EvaluationInputAdapter` 相关实现

本步骤未新增 Bash/scripts，未访问 `/data2`，未启动 pressure/full-scale。

## 4. 结果

设计文档固定以下结论：

- `SampleManifest` 是 Stage 1 后续 canonical sample source，推荐字段包括
  `sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、`channel_id`、
  `window_index`，并可选记录 `seq_len` / `pred_len`、manifest shard、source lineage 和 `extra`。
- `SplitStrategy` 长期负责生成或校验 split，并按 split 输出 ordered `sample_keys`。
- split 不应继续散落在 labels CSV、feature CSV、oracle reader 和 prediction reader 中。
- Visual Router 与 TimeFuse-style fusor 都应消费同一套 `sample_key` 顺序。
- `ExpertProvider / ExpertBatch` 只提供 `y_pred`、`y_true`、`model_columns` 和 row index lineage。
- `SupervisionProvider` 或 `OracleLabelProvider` 提供 `oracle_model`、`oracle_value`、
  `per_model_errors` / `model_error_matrix`、`model_columns`、`metric` 和 `extra`。
- oracle / per-model error 只用于训练监督、诊断、baseline 或 upper-bound，不进入 deployable
  Visual / TimeFuse `FeatureProvider`。
- Visual Router 当前 labels CSV 同时承担 manifest/split/oracle/metadata，以及 TimeFuse 当前
  feature CSV + oracle SQLite/parquet + prediction SQLite 的分工，均视为历史实现差异，不作为长期接口边界。

## 5. 验证

已在 `quito` conda 环境运行以下命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router visual_router_experiments/common/prediction_array_io.py
```

结果：全部通过。

关键验证结果：

- P10b SQLite backend smoke 通过，完整 fixture 可构建 SQLite 子集索引、fetch records，并按 row index 读回 packed arrays；缺失 sample/model 默认报错，`allow_missing=True` 时写入 missing report。
- PredictionCacheExpertProvider smoke 通过，`ExpertBatch` 保序且 shape 为 `y_pred=(4, 5, 48, 1)`、`y_true=(4, 48, 1)`，并复算 `hard_mae=0.416048437`、`raw_soft_mae=0.410296679`。
- Visual Router training ExpertBatch bypass smoke 通过，MAE/MSE expert_errors 可由 `ExpertBatch.y_pred/y_true` 复算且 mismatch 报错上下文完整。
- TimeFuse protocol chain smoke 通过，head/evaluator 阶段不调用文件 IO、`np.load` 或 `np.save`，输出 `hard_mae=1.093573928`、`raw_soft_mae=0.556751269`。
- compileall 通过。

## 6. 结论

P10d 已把 Stage 1 后续重跑所需的 canonical sample / split / supervision 上层边界冻结到文档层。
后续 P11/P12 可以在该边界上设计新的 run artifact schema；旧 schema 仅作为迁移来源和复现材料，不再强行作为最高兼容目标。

## 7. 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续进入 P11/P12 schema 冻结设计，或设计小规模 `SampleManifest` / `SupervisionProvider` fixture。
