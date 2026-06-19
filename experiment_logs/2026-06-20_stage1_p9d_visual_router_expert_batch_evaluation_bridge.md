# Stage 1 P9d Visual Router ExpertBatch Evaluation Bridge

日志日期：2026-06-20 02:56:28 CST

## 目的

在不替换 Visual Router 正式 prediction 读取、不迁移正式入口的前提下，把
`--verify-evaluation-adapter` 旁路校验中的 adapter 输入从直接 `EvaluationInput`
进一步收敛到 `ExpertBatch + fusion_weights`。

## 背景

P9b 已在 `train_visual_router_online_streaming.py` 中新增默认关闭的
`--verify-evaluation-adapter`，只在 test evaluation batch 内用当前 batch arrays
构造 evaluation 输入并旁路复算 hard/raw-soft 指标。P9c 已用小规模正式入口
off/on pressure 验证开启该 flag 不改变正式 artifact。

本轮 P9d 的目标不是正式接入 `PredictionCacheExpertProvider`，而是让 Visual
Router evaluation 旁路先接受 canonical `ExpertBatch` 边界，同时继续使用当前
legacy SQLite batch arrays 作为 `y_pred/y_true` 来源。

## 操作

1. 阅读用户目标文件 `/home/shiyuhong/.codex-tianyu/attachments/f46024e1-e87f-4ba4-bc29-6c755f90b2af/pasted-text-1.txt`，确认 P9d 范围和禁止项。
2. 审查 `time_router.protocols.ExpertBatch` 与 `time_router.evaluation.EvaluationInputAdapter`，确认 adapter 已有 `evaluate(expert_batch=..., fusion_weights=...)` public API。
3. 修改 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`：
   - import 从 `EvaluationInput` 改为 `ExpertBatch`。
   - 新增 `build_visual_router_expert_batch_for_evaluation(...)`，只包装当前 batch 已读取出的 `sample_keys`、`MODEL_COLUMNS`、`y_pred/y_true` 和轻量 lineage。
   - `verify_evaluation_adapter_bypass_batch(...)` 仍从 `pred_df` 的 `weight_<model_name>` 列恢复权重，但改为构造 `ExpertBatch` 后调用 `EvaluationInputAdapter().evaluate(expert_batch=..., fusion_weights=...)`。
   - 保留 selected_model、selected_index、hard MAE/MSE、raw soft MAE/MSE、max_weight、weight_entropy 和 mismatch 错误信息比较逻辑。
4. 更新 `tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py`：
   - 保留原有正常通过和故意 mismatch 测试。
   - 新增对 `ExpertBatch.sample_keys`、`ExpertBatch.model_columns` 保序的断言。
   - 通过 patch `EvaluationInputAdapter.evaluate` 捕获 helper 入参，确认不再直接传入 `EvaluationInput`。
   - 断言 float32 `y_pred/y_true` 原样进入 adapter，`fusion_weights` 与测试权重一致。
5. 新增 `docs/refactor/visual_router_expert_batch_evaluation_bridge.md`。
6. 更新 `docs/refactor/visual_router_evaluation_adapter_bypass.md`、`docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`，说明 P9d 不是 `PredictionCacheExpertProvider` 正式接入，也不替换 Visual SQLite index / prediction reader / 正式输出 schema。

## 结果

已运行并通过以下验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```

关键验证结果：

- Visual Router bypass smoke 确认 helper 经 `ExpertBatch + fusion_weights` 调用 adapter，样本/专家顺序和数组输入保持不变。
- EvaluationInput adapter smoke 继续确认 adapter 阶段不重新读取 prediction cache 或 oracle/TSF，golden hard/raw-soft 指标不漂移。
- PredictionCacheExpertProvider smoke 继续确认 provider 只读 golden fixture、`ExpertBatch` 保序和 hard/raw-soft golden 指标不漂移。
- `compileall` 覆盖 `time_router`、`tests/smoke` 和 Visual Router streaming 入口，通过。

## 结论

P9d 已完成：Visual Router evaluation 旁路输入已从直接 `EvaluationInput` 收敛到
canonical `ExpertBatch + fusion_weights`。该变化只影响默认关闭的旁路校验路径，
默认正式行为、正式 prediction_index / SQLite index、`predict_stream_batch(...)`、
`add_soft_fusion_metrics(...)`、正式 CSV/summary/metadata/status/checkpoint schema、
Visual FeatureProvider、ViT、router head、training loop 和 `fusion_huber_kl` loss 均未迁移或替换。

P9c off/on pressure 仍作为最近一次正式入口无漂移证据；本轮未重复运行 full-scale
或新的正式入口 pressure。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续如进入 P9e，应先审计 `PredictionCacheExpertProvider` 与 Visual SQLite index
   在 full-scale shard、row index、batch query、内存和 resume 语义上的能力差距。
3. 在完成上述审计前，不应把 smoke-only `PredictionCacheExpertProvider` 直接替换进
   Visual Router full-scale 正式入口。
