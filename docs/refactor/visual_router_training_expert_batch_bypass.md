# Visual Router Training ExpertBatch Bypass

日志日期：2026-06-20 03:14:10 CST

## 1. 目标

P9f 在 Visual Router `fusion_huber_kl` training loss 阶段增加默认关闭的
`--verify-training-expert-batch` 旁路校验。

该校验只证明：当前 legacy SQLite path 已读取出的 `y_pred` / `y_true` 可以包装为
`ExpertBatch`，并由 `ExpertBatch.y_pred` / `ExpertBatch.y_true` 显式复算出与旧路径
一致的 `expert_errors`。

本阶段不替换正式训练 loss，不替换 SQLite prediction index，不接
`PredictionCacheExpertProvider` 到正式入口。

## 2. 实现范围

- 在 `train_visual_router_online_streaming.py` 新增默认关闭 flag
  `--verify-training-expert-batch`。
- 新增 `verify_training_expert_errors_from_expert_batch(...)`。
- helper 构造 `ExpertBatch(sample_keys, model_columns, y_pred, y_true)`。
- helper 对 `error_metric == "mae"` 显式复算 per-sample per-expert MAE。
- helper 对 `error_metric == "mse"` 显式复算 per-sample per-expert MSE。
- helper 将复算结果与 legacy `expert_errors` 逐元素比较。
- helper 只在 `fusion_huber_kl` training batch 内由 flag 触发。
- 若 `router_mode == "classification"` 且开启该 flag，入口直接报错。
- `output_dir`、`epoch` 和 `training_batch_index` 只用于失败信息定位，不写任何文件。

失败信息包含 `phase=training`、`router_mode=fusion_huber_kl`、`metric`、
`batch_index` / `training_batch_index`、`sample_key`、`model_name`、`expert_index`、
`old_value` / `legacy_value`、`expert_batch_value` / `recomputed_value` 和
`output_dir`。

## 3. 明确不变

- 不改变默认训练行为；flag 默认关闭。
- 不改变 `train_on_stream_batch(...)` 的正式 loss 输入变量使用方式。
- 不返回替代 loss，不参与反传。
- 不改变 `fusion_huber_kl` loss、optimizer、scheduler、scaler、checkpoint/resume。
- 不改变 CSV、summary、metadata、status schema。
- 不替换 `SQLitePredictionIndex`。
- 不迁移 `PredictionBatchReader` 到正式入口。
- 不修改 `PredictionBatchReader`、`PredictionCacheExpertProvider` 或
  `EvaluationInputAdapter`。
- 不修改 VisualFeatureProvider / ViT provider / router head。
- 不新增 Bash/scripts。
- 不访问 `/data2`，不启动 full-scale。
- 不改 TimeFuse 正式入口。

## 4. Smoke 覆盖

新增 `tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py`。

该 smoke 只使用测试内小型 numpy arrays：

- 构造 2 个样本、5 个专家、3 步单通道的 `y_pred/y_true`。
- 分别覆盖 MAE 与 MSE 的 `expert_errors` 复算。
- 验证复算结果与测试内 legacy `expert_errors` 一致。
- 构造故意 mismatch，确认失败信息包含 sample_key、model_name、metric、old/recomputed
  value 和 output_dir 等定位上下文。
- 不启动 ViT，不访问 `/data2`，不运行正式入口。

## 5. 与 P9c 的关系

P9f 只影响默认关闭的 training batch 内存校验 helper，没有改变默认训练路径。
P9c 仍是最近一次正式入口 verify off/on artifact 无漂移证据：当时已在小规模正式入口
pressure run 中证明开启 `--verify-evaluation-adapter` 不改变目标 CSV、summary、
metadata/status/checkpoint schema，也不新增 adapter artifact。

P9f 没有重复 P9c pressure，因为本次新增校验默认关闭，且 smoke 已覆盖 helper 的 MAE/MSE
一致性和失败定位信息。

## 6. 后续建议

P9f 完成后，下一步应进入 shared prediction SQLite backend / index prepare
consolidation：

- 先整理 `build_lightweight_prediction_index(...)` 的 shared index prepare 边界。
- 保持 runtime 负责 required sample_keys、SQLite artifact、metadata 和 resume lifecycle。
- Provider 后续只接收 prepared backend 或显式 batch query 能力。
- 不应直接抽 VisualFeatureProvider / ViT provider，也不应直接用
  `PredictionBatchReader` 替换 Visual Router 正式 SQLite path。

## 7. 验证结果

本阶段目标验收命令均已通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```
