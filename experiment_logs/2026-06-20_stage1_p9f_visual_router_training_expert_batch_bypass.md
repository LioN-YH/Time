# Stage 1 P9f Visual Router Training ExpertBatch Bypass

日志日期：2026-06-20 03:17:00 CST

## 目的

在不改变 Visual Router 正式训练行为的前提下，为 `fusion_huber_kl` training batch
新增默认关闭的 ExpertBatch 旁路校验，验证 legacy SQLite path 已读取出的
`y_pred/y_true/expert_errors` 能由 `ExpertBatch.y_pred/y_true` 显式表达。

## 背景

P9a 已完成 Visual Router 正式入口 adapter 插入点审计；P9b/P9c 已完成 evaluation
adapter bypass 与小规模 pressure 验证；P9d 已将 evaluation bypass 输入边界收敛到
`ExpertBatch + fusion_weights`；P9e 已审计 `PredictionCacheExpertProvider` 与 Visual
Router 正式 SQLite path 的 full-scale 能力差距。

P9e 结论是短期不替换 `SQLitePredictionIndex`，不把
`PredictionCacheExpertProvider` 或 `PredictionBatchReader` 直接接入正式入口，而是先做
batch 后 `ExpertBatch` 旁路校验。本次 P9f 只覆盖 training loss supervision 旁路，不改变
loss、optimizer、checkpoint 或任何正式输出 schema。

## 操作

1. 修改 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`：
   - 新增默认关闭 flag `--verify-training-expert-batch`。
   - 新增 `verify_training_expert_errors_from_expert_batch(...)`。
   - helper 只包装当前 training batch 已读取出的 legacy `y_pred/y_true`，构造
     `ExpertBatch` 后显式复算 MAE/MSE `expert_errors`。
   - 在 `fusion_huber_kl` 分支读取完 `y_pred/y_true/expert_errors` 后、正式 loss 计算前，
     仅当 flag 开启时执行旁路校验。
   - `classification` 模式同开该 flag 时直接报错，避免误认为 classification 已完成
     prediction supervision 校验。
   - `output_dir`、`current_epoch` 和 `training_batch_index` 只作为失败信息定位上下文，
     不写入正式 artifact schema。

2. 新增 `tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py`：
   - 使用测试内小型 numpy arrays 构造 2 个样本、5 个专家、3 步单通道的
     `y_pred/y_true`。
   - 覆盖 MAE 与 MSE `expert_errors` 复算一致性。
   - 构造故意 mismatch，检查失败信息包含 `phase=training`、
     `router_mode=fusion_huber_kl`、metric、batch index、sample_key、model_name、
     expert_index、legacy/recomputed value 和 output_dir。
   - smoke 不启动 ViT、不访问 `/data2`、不运行正式入口。

3. 新增和更新文档：
   - 新增 `docs/refactor/visual_router_training_expert_batch_bypass.md`。
   - 更新 `docs/refactor/stage1_refactor_roadmap.md`，加入 P9f 小节。
   - 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`，记录 P9f 状态和后续路线。
   - 更新 `docs/refactor/visual_router_prediction_cache_provider_gap_audit.md`，把 P9f 从建议项更新为已完成项。
   - 更新 `WORKSPACE_STRUCTURE.md`，登记 P9f 文档、入口 flag 和 smoke。

4. 运行验收命令：
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`

## 结果

- P9f 新增 flag 默认关闭，默认训练行为不变。
- `verify_training_expert_errors_from_expert_batch(...)` 可从 `ExpertBatch.y_pred/y_true`
  复算 MAE/MSE `expert_errors`，并与 legacy `expert_errors` 比较。
- 校验失败信息包含 training phase、router mode、metric、batch/sample/model/expert/value
  和 output_dir 定位上下文。
- 未替换 `SQLitePredictionIndex`。
- 未接 `PredictionCacheExpertProvider` 到正式入口。
- 未迁移或修改 `PredictionBatchReader`。
- 未修改 `PredictionCacheExpertProvider`、`EvaluationInputAdapter`、VisualFeatureProvider、
  ViT provider、router head、正式 loss、optimizer、scheduler、scaler、checkpoint/resume、
  CSV、summary、metadata 或 status schema。
- 未新增 Bash/scripts，未访问 `/data2`，未启动 pressure/full-scale，未改 TimeFuse 正式入口。

验证结果：

- P9f training ExpertBatch bypass smoke 通过。
- P9d evaluation adapter bypass smoke 通过。
- PredictionCacheExpertProvider smoke 通过。
- EvaluationInputAdapter smoke 通过。
- compileall 通过。

## 结论

P9f 已证明 Visual Router `fusion_huber_kl` 的训练专家监督 `expert_errors` 可以从
`ExpertBatch.y_pred/y_true` 显式复算表达。该能力目前仅作为默认关闭的 training batch
旁路校验存在，不改变正式训练路径，也不代表 `PredictionCacheExpertProvider` 已正式接入
Visual Router full-scale。

P9c 仍是最近一次正式入口 verify off/on artifact 无漂移证据；P9f 没有重复 pressure，
因为本次新增逻辑默认关闭，且只在内存中校验当前 batch 已读取数组。

## 下一步方案

下一步应进入 shared prediction SQLite backend / index prepare consolidation：

1. 先整理 `build_lightweight_prediction_index(...)` 的 shared index prepare 边界。
2. 保持 runtime 负责 required sample_keys、SQLite artifact、metadata、status 和
   checkpoint/resume lifecycle。
3. 后续 provider 只接收 prepared backend 或显式 batch query 能力。
4. 不直接抽 VisualFeatureProvider / ViT provider。
5. 不直接用 `PredictionBatchReader` 替换 Visual Router 正式 SQLite path。
