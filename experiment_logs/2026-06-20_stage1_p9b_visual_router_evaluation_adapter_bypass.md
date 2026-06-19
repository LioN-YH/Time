# Stage 1 P9b Visual Router Evaluation Adapter Bypass

日志日期：2026-06-20 02:28:08 CST

## 目的

为 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
新增默认关闭的 `--verify-evaluation-adapter`，只在 test evaluation batch 内用
`EvaluationInputAdapter` 做旁路一致性校验，验证当前 Visual Router 正式 evaluation
字段与 canonical adapter rows 对齐。

## 背景

P9a 已完成 Visual Router 正式入口 adapter 插入点审计，结论是第一步不迁移
Visual FeatureProvider、ViT provider、router head、training loop 或 `fusion_huber_kl`
loss，只在 evaluation batch 做旁路校验。P9b 需要保证默认行为不变，并且不修改
正式 CSV、summary、metadata、status 或 checkpoint schema。

## 操作

1. 在 `train_visual_router_online_streaming.py` 中新增默认关闭参数
   `--verify-evaluation-adapter`。
2. 新增 `verify_evaluation_adapter_bypass_batch(...)` helper：
   - 使用当前 batch 的 `sample_key` 顺序和 `MODEL_COLUMNS`；
   - 从 `weight_<model_name>` 列恢复权重；
   - 使用当前 batch 的 prediction lookup 读取 `y_pred/y_true`，或由 smoke
     显式传入内存数组；
   - 构造 `EvaluationInput` 并调用 `EvaluationInputAdapter.evaluate_input(...)`；
   - 逐样本比较 sample_key、selected_model、selected_index、hard MAE/MSE、
     raw soft MAE/MSE、max_weight 和 weight_entropy。
3. 在 test evaluation batch 内保持原顺序：
   `pred_df = predict_stream_batch(...)`、`soft_df = add_soft_fusion_metrics(...)`、
   旁路校验、再 append soft CSV。
4. 增加 `--verify-evaluation-adapter` 与 `--skip-soft-fusion` 同时开启时报错的保护，
   避免缺少 raw soft fusion 对齐证据。
5. 新增 `tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py`，使用
   小型 numpy arrays / DataFrame 调用 helper，不启动 ViT、不访问 `/data2`、不运行正式入口。
6. 新增 `docs/refactor/visual_router_evaluation_adapter_bypass.md`，记录 P9b 的旁路边界、
   校验字段、失败信息和后续 P9c 方向。
7. 更新 `WORKSPACE_STRUCTURE.md`，登记新增文档、入口 flag 和 smoke。

## 结果

验证命令均使用 conda 环境 `quito`：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
```

结果通过，输出确认 adapter rows 与正式 `soft_df` 的 selected/hard/raw-soft/权重诊断字段一致，
并确认故意 mismatch 时错误信息包含 config、split、batch、sample、字段、旧值、adapter 值和 output_dir。

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
  visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
  tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
```

结果通过，无语法错误。

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
```

结果通过，确认 canonical `EvaluationInputAdapter` golden smoke 口径未漂移。

## 结论

P9b 已完成最小 Visual Router evaluation adapter 旁路校验。默认关闭时正式入口行为不变；
开启 flag 时只增加 test evaluation batch 内的内存一致性校验，不写 adapter rows，不修改正式
artifact schema，不新增 provider/head 代码，不访问 `/data2`，不启动 pressure/full-scale。

## 下一步方案

后续若继续推进，应先提交并推送当前 P9b 变更。P9c 可在不扩大正式入口行为面的前提下，
考虑更小粒度的 `ExpertBatch` 对齐或 prediction batch adapter 对齐；不要直接迁移 Visual
FeatureProvider、ViT provider、router head 或 training loop。
