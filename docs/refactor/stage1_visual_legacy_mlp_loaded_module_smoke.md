# Stage 1 P16h Visual Legacy MLP Loaded-Module Smoke

记录日期：2026-06-21

## 1. 目标

P16h 只验证一个边界：

```text
legacy VisualMLPRouter 已加载 torch module
+ head-ready float32 FeatureBatch
+ explicit model_columns
-> P16a LoadedTorchMLPRouterHeadAdapter
-> RouterOutput
-> EvaluationInputAdapter summary / rows
```

本步不实现 checkpoint loader，不读取真实 checkpoint，不处理真实 scaler，不接 ViT，不迁移
正式 Visual Router 入口，也不声称正式 Visual Router 已迁移完成。

## 2. 与 P16a / P16g 的关系

- P16a 已提供正式最小 adapter：`LoadedTorchMLPRouterHeadAdapter` 只包装 Runtime 已经加载好的
  `torch.nn.Module`，并消费 head-ready `FeatureBatch`。
- P16g 是文档审计：确认 legacy `VisualMLPRouter` 的 constructor、forward、
  checkpoint payload、`router_state_dict`、scaler 和 DataParallel key 边界。
- P16h 是 in-memory loaded-module smoke：实例化 legacy `VisualMLPRouter`，用内存 fake
  `state_dict` strict load，再交给 P16a adapter。

真实 checkpoint loader、checkpoint payload discovery、`map_location` 策略、strict policy 的
错误上报、scaler loading / transform 和 Runtime device policy 仍属于后续 Runtime / entrypoint
步骤。

## 3. Smoke 覆盖

新增：

```text
tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py
```

该 smoke 执行以下检查：

- import legacy `VisualMLPRouter` 定义，只实例化 class，不调用正式训练入口。
- 使用 P13b `sample_manifest.csv` 的 ordered sample_keys。
- 使用 P16c `VisualPrecomputedFeatureProvider` 读取仓库内 head-ready float32 fixture。
- 使用 P13b `expert_predictions.json` 构造 `ExpertBatch`。
- 构造 in-memory fake normal state_dict。
- 构造 DataParallel 风格 `module.` 前缀 state_dict。
- 在测试内清洗 `module.` 前缀，并验证两种 state_dict 均可 strict load。
- 直接验证 legacy module forward 输出二维 logits tensor，shape 为
  `[num_samples, num_experts]`。
- 将已加载 module 交给 `LoadedTorchMLPRouterHeadAdapter`。
- 验证 `RouterOutput`、sample_key 保序、model_columns 与 `ExpertBatch` 对齐、weights finite
  且 softmax row sum 约等于 1。
- 验证 `EvaluationInputAdapter` 可生成 hard/raw-soft MAE/MSE 和 per-sample rows。
- patch `torch.load`，若 smoke 核心路径读取 checkpoint 则立即失败。
- 检查不创建 `experiment_logs/run_outputs/` 新目录。

## 4. 明确不做

- 不读取 `/data2`。
- 不调用 `torch.load`。
- 不读取真实 checkpoint。
- 不启动 ViT。
- 不处理真实 scaler。
- 不调用 `train_visual_router_online_streaming.py`。
- 不修改 `scripts/run_stage1_visual_small.py`。
- 不新增正式 checkpoint loader。
- 不把 checkpoint path、scaler path 或 run_dir 加进 adapter interface。
- 不修改正式训练/evaluation 入口。
- 不启动训练、pressure 或 full-scale。

## 5. 验证命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_protocol_smoke.py
```
