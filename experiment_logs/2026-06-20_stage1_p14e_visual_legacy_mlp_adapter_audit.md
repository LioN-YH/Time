# Stage 1 P14e Visual Legacy MLP Adapter Audit

日志日期：2026-06-20 20:40:00 CST

## 目的

审计 Stage 1 Visual eval-only 阶段 legacy `VisualMLPRouter` 如何在后续被薄包装为
canonical `RouterOutput`，明确 `FeatureBatch -> legacy MLP -> RouterOutput` 的最小边界。

## 背景

P14d 已完成 Visual mock protocol eval smoke，证明 tiny fixture 上可以从
`FeatureBatch + ExpertBatch` 经 smoke-only mock RouterHead 输出 `RouterOutput`，再由
`EvaluationInputAdapter` 生成 summary / rows。

正式 Visual Router 仍使用 legacy `VisualMLPRouter`、`StandardScaler`、checkpoint、
device/dtype 和 ViT `DataParallel` 等入口逻辑。P14e 只做文档审计，不新增正式 adapter
代码，不修改正式入口，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 操作

1. 阅读目标说明文件，确认 P14e 验收范围是纯文档审计、同步路线文档、运行既有 smoke、
   写中文实验日志、小步提交并 push。
2. 审计 `visual_router_experiments/stage1_vali_test_router/train_visual_router.py` 中
   `VisualMLPRouter` 定义、`forward(...)`、`train_router_for_config(...)` 和
   `predict_router_for_config(...)`。
3. 审计 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
   中 `StandardScaler` state 保存/恢复、checkpoint loading/signature、device/dtype、ViT
   `DataParallel`、`predict_stream_batch(...)`、P9d/P9f `ExpertBatch` 旁路和 evaluation
   adapter 校验逻辑。
4. 阅读 P9b/P9d/P9f、P14a/P14b/P14c/P14d 相关文档，确认 Visual evaluation/training
   `ExpertBatch` bypass、Visual `FeatureBatch`、mock head 和 canonical eval-only 链路的既有结论。
5. 新增 `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md`。
6. 更新 `docs/refactor/stage1_visual_eval_canonical_bypass_plan.md`、
   `docs/refactor/stage1_visual_mock_protocol_eval_smoke.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md`、
   `docs/refactor/stage1_refactor_roadmap.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

新增审计文档明确以下结论：

- legacy `VisualMLPRouter` eval-only 输入是 scaler transform 后的 ViT pooled embedding，
  对应 future adapter 的 head-ready `FeatureBatch.features`。
- future eval-only legacy MLP adapter 目标边界为
  `FeatureBatch(sample_keys, features) + model_columns + runtime-loaded legacy MLP checkpoint/scaler/device context -> RouterOutput(sample_keys, model_columns, logits, weights, extra)`。
- legacy MLP 输出 logits，weights 由 `softmax(logits, dim=1)` 得到；adapter 只应检查
  sample_key 保序、model_columns、shape、有限值和 softmax row sum。
- `model_columns` 应由调用方显式传入，并与 `ExpertBatch.model_columns` 完全一致；不得从
  CSV、checkpoint path 或 prediction cache 反推专家顺序。
- scaler fit 和 scaler checkpoint state 属于 training/runtime state；eval-only transform
  可由 Runtime 或 adapter 前的显式 pre-head transform step 完成，adapter 不自己 fit scaler。
- checkpoint loading、signature 校验、resume、optimizer state、device/dtype 和
  DataParallel 由 Runtime/entrypoint 管理；adapter 可以消费已加载并已放到目标 device 的
  legacy MLP，但不决定全局资源策略。
- 本轮未修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py`、
  `launch_timefuse_fusor_full_scale.py`，未新增正式 adapter/provider/head 代码，未访问
  `/data2`，未启动训练或 full-scale。

P14e 指定验收命令均已通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

验证结果显示六个 smoke 全部通过，`compileall` 对 `time_router`、`scripts`、`tests/smoke`
和 `visual_router_experiments/stage1_vali_test_router` 完成编译检查。

## 结论

P14e 的最小边界已经文档化：future legacy MLP eval-only adapter 应只负责
`FeatureBatch -> RouterOutput` 的 head 适配，不拥有 prediction cache、oracle/error、
checkpoint/resume、scaler fit、device/dtype/DataParallel、evaluation artifact 或 training loop。

## 下一步方案

1. 提交并 push 到 `refactor/stage1-route-audit`。
2. 后续 P14f 可做 smoke-only Visual legacy MLP adapter，使用 tiny `FeatureBatch` 和小型
   torch MLP / loaded state_dict fixture 输出 `RouterOutput`，继续不接正式入口。
