# Stage 1 P8a TimeFuse 正式入口 Adapter 插入点审计

日志日期：2026-06-20 01:20:02 CST

## 目的

在 P7c TimeFuse protocol chain smoke 之后，审计正式入口 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` 的最小 `EvaluationInputAdapter` 接入点，并形成 P8b 最小代码迁移建议。

本次只做文档化接入计划，不修改正式训练入口行为。

## 背景

P6/P7 已完成以下 smoke-only adapter：

- `PredictionCacheExpertProvider -> ExpertBatch`
- `EvaluationInputAdapter -> summary / per-sample rows`
- `TimeFuseFeatureCacheProvider -> FeatureBatch`
- `TimeFuseLinearSoftmaxHead -> RouterOutput`
- P7c protocol chain smoke 串联上述对象并通过 deterministic 验证。

但这些 adapter 尚未接入正式 TimeFuse full-scale streaming 入口。正式入口仍负责 feature/prediction/oracle streaming reader、scaler、torch fusor 训练、evaluation、CSV/summary/checkpoint/status/metadata 写出。

## 操作

1. 读取用户目标文件，确认本步边界为 P8a 文档化审计：不改正式入口、不访问 `/data2`、不新增 Bash/scripts、不改 Visual Router 入口。
2. 只读检查当前分支为 `refactor/stage1-route-audit`。
3. 只读检查 `train_timefuse_fusor_streaming.py`，重点定位：
   - `fit_scaler_streaming(...)`
   - `train_streaming(...)`
   - `evaluate_streaming(...)`
   - `save_checkpoint(...)` / `load_checkpoint(...)`
   - `write_markdown_summary(...)`
   - `metadata.json` / `status.json` 写出逻辑
4. 只读检查既有 P6/P7 文档：
   - `docs/refactor/evaluation_input_adapter.md`
   - `docs/refactor/timefuse_feature_cache_provider.md`
   - `docs/refactor/timefuse_linear_head.md`
   - `docs/refactor/timefuse_protocol_chain_smoke.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
5. 新增 `docs/refactor/timefuse_entrypoint_adapter_insertion_audit.md`。
6. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P8a 状态和边界。
7. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`，补充 P8a 审计结论。
8. 更新 `WORKSPACE_STRUCTURE.md`，登记新增审计文档和更新时间。
9. 使用 `quito` 环境运行 P7c protocol chain smoke。
10. 使用 `quito` 环境运行 `compileall`。

## 结果

本次审计结论如下：

- 最小 `EvaluationInputAdapter` 接入点应放在 `evaluate_streaming(...)` 中每个 test batch 完成 torch fusor 前向、得到 `weights_np` 之后。
- 该位置已经同时持有 `batch.sample_keys`、`MODEL_COLUMNS`、`batch.y_pred`、`batch.y_true` 和 `weights_np`，可以构造 batch 级 `EvaluationInput` 或等价 adapter 输入。
- 可复用 `time_router.evaluation` 中的 `EvaluationInputAdapter.evaluate_input(...)`、hard top-1、raw soft fusion、summary、per-sample rows 和 weight diagnostics helper。
- CSV 写出、`summary.md`、checkpoint/status/metadata、scaler fit、optimizer/loss/epoch loop、reader/index 准备、oracle regret 和正式字段命名必须暂留正式入口或后续 runtime/report 层。
- P7a `TimeFuseFeatureCacheProvider` 当前只是 smoke-only 小规模 CSV adapter，不能直接替换 full-scale streaming reader。
- P7b `TimeFuseLinearSoftmaxHead` 当前只是 numpy smoke head，不能直接替换正式 torch 训练 head。
- P8b 最小迁移建议是仅在 evaluation 阶段旁路复算 batch metrics，不改变正式输出 schema。

## 结论

P8a 文档化审计已完成。正式入口行为未修改，未访问 `/data2`，未新增 Bash/scripts，未改 Visual Router 入口。

当前最小、安全的 P8b 方向是：在 `evaluate_streaming(...)` 中先用 `EvaluationInputAdapter` 做内存一致性校验，继续由现有正式代码写 CSV、summary、metadata、status 和 checkpoint。

验收结果：

- `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py` 通过，输出确认链路 deterministic，`hard_mae=1.093573928`、`raw_soft_mae=0.556751269`。
- `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke` 通过。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续 P8b 如修改正式入口，应先只在小规模 pressure 输出中比较迁移前后的 CSV 字段、行数、sample_key 顺序、hard/raw-soft MAE/MSE 和 selected counts，比较通过前不能启动 full-scale。
