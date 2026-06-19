# Visual Router ExpertBatch Evaluation Bridge

日志日期：2026-06-20 02:54:04 CST

## 1. 目标

P9d 在 P9b/P9c 已完成的 `--verify-evaluation-adapter` 旁路基础上，
把 Visual Router evaluation 旁路校验的 adapter 输入从直接构造
`EvaluationInput` 收敛为 `ExpertBatch + fusion_weights`。

本阶段只验证 Visual Router 正式入口可以在 evaluation 旁路中接受
canonical `ExpertBatch` 边界。正式 prediction 来源仍是当前入口的 legacy
SQLite batch arrays。

## 2. 实现位置

代码变更集中在
`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`：

- 新增 `build_visual_router_expert_batch_for_evaluation(...)`。
- `verify_evaluation_adapter_bypass_batch(...)` 仍从 `pred_df` 的
  `weight_<model_name>` 列恢复 `fusion_weights`。
- helper 仍通过当前 batch 的 `prediction_lookup` 或 smoke 显式传入的
  `y_pred/y_true` 获得数组。
- helper 不再直接构造 `EvaluationInput`，而是构造 `ExpertBatch` 后调用
  `EvaluationInputAdapter().evaluate(expert_batch=..., fusion_weights=...)`。
- 后续 `selected_model`、`selected_index`、hard MAE/MSE、raw soft MAE/MSE、
  `max_weight` 和 `weight_entropy` 比较逻辑保持不变。

## 3. 边界

- P9d 只改默认关闭的 `--verify-evaluation-adapter` 旁路路径。
- P9d 不是 `PredictionCacheExpertProvider` 正式接入。
- P9d 不替换 Visual Router 正式 `prediction_index` 或 SQLite batch query。
- P9d 不迁移 `PredictionBatchReader` 到正式入口。
- P9d 不改变 `predict_stream_batch(...)`。
- P9d 不改变 `add_soft_fusion_metrics(...)`。
- P9d 不改变正式 CSV、summary、metadata、status 或 checkpoint schema。
- P9d 不新增 VisualFeatureProvider，不抽 ViT provider，不改
  `VisualMLPRouter`、router head、training loop 或 `fusion_huber_kl` loss。
- P9d 不新增 Bash/scripts，不访问 `/data2`，不启动 full-scale。

## 4. ExpertBatch 来源

当前 `ExpertBatch` 只包装已经在 Visual Router evaluation batch 中存在的
legacy arrays：

- `sample_keys` 来自 `pred_df["sample_key"]`，保留当前 batch 顺序。
- `model_columns` 使用 `MODEL_COLUMNS`，保留五专家顺序。
- `y_pred/y_true` 来自当前 batch 的 `prediction_lookup` 或 smoke 显式数组。
- `extra` 只记录轻量 lineage：source、batch_index、output_dir 和
  `expert_batch_source=visual_router_legacy_sqlite_batch_arrays`。

该 helper 不读取 manifest，不读取 prediction cache，不创建 run_dir，不写
status/metadata/CSV，不计算 loss/evaluation，也不假装当前来源已经是
`PredictionCacheExpertProvider`。

## 5. 测试

更新 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
```

该测试继续使用内存 numpy arrays / DataFrame，不启动 ViT，不访问 `/data2`，
不运行正式入口。新增断言覆盖：

- `ExpertBatch.sample_keys` 保序。
- `ExpertBatch.model_columns` 保序。
- `y_pred/y_true` 对 float32 输入原样进入 adapter。
- helper 实际调用 `EvaluationInputAdapter.evaluate(...)`，且不传入
  `evaluation_input`。
- 故意 mismatch 的失败信息仍包含 config、split、batch、sample、field、
  old value、adapter value 和 output_dir。

## 6. Pressure 口径

P9d 没有改变默认路径，也没有改变正式输出链路。最近一次正式入口无漂移证据仍
沿用 P9c：

`docs/refactor/visual_router_evaluation_adapter_pressure_verification.md`

P9c 已在小规模正式入口中比较关闭/开启 `--verify-evaluation-adapter` 后的正式
CSV、summary、comparison、selected counts、streaming summary、metadata/status
和 checkpoint schema，确认除 run_dir 路径和生成时间外无正式口径漂移。

## 7. 后续

P9d 通过后，后续可另起 P9e 审计 prediction cache provider 与 Visual SQLite
index 的 full-scale 能力差距。该后续审计应继续保持只读或 smoke-first，不应把
`PredictionCacheExpertProvider` 直接接到 Visual Router full-scale 正式入口。
