# Visual Router Evaluation Adapter Bypass

日志日期：2026-06-20 02:28:08 CST

## 1. 目标

P9b 在 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
中新增默认关闭的 `--verify-evaluation-adapter`。它只在 test evaluation batch 内
构造 `EvaluationInput`，调用 `time_router.evaluation.EvaluationInputAdapter`
做内存旁路复算，并逐样本校验当前正式路径生成的字段。

## 2. 边界

- P9b 只是旁路校验，不替换正式 evaluation。
- adapter rows 只作为内存校验对象，不写入正式 CSV。
- 不修改正式 CSV、summary、metadata、status 或 checkpoint schema。
- Visual Router feature、ViT、router head、training loop 和 `fusion_huber_kl`
  loss 仍留在正式入口内。
- 不迁移 Visual FeatureProvider，不抽 ViT provider，不新增 provider/head 代码。

## 3. 校验位置

旁路校验位于 test streaming batch 的正式 evaluation 路径中：

1. `pred_df = predict_stream_batch(...)`
2. `soft_df = add_soft_fusion_metrics(pred_df, soft_lookup)`
3. `verify_evaluation_adapter_bypass_batch(...)`
4. `append_csv(..., soft_df)`

因此正式 append 和 summary 逻辑仍使用既有 `pred_df` / `soft_df`，adapter 结果不
参与正式输出。

## 4. 校验字段

helper 从当前 batch 的 `sample_key` 顺序、`MODEL_COLUMNS` 和
`weight_<model_name>` 列恢复权重，并使用当前 batch prediction lookup 读取
`y_pred/y_true`。随后用 `EvaluationInputAdapter` 复算并比较：

- `sample_key` 顺序
- `selected_model`
- `selected_index`，由正式 `selected_model` 映射到 `MODEL_COLUMNS` 下标
- `hard_top1_mae_from_array` vs adapter `hard_mae`
- `hard_top1_mse_from_array` vs adapter `hard_mse`
- `soft_fusion_mae` vs adapter `raw_soft_mae`
- `soft_fusion_mse` vs adapter `raw_soft_mse`
- `max_weight`
- `weight_entropy`

如果 `--verify-evaluation-adapter` 与 `--skip-soft-fusion` 同时开启，入口直接报错。
本轮目标需要 raw soft fusion 对齐证据，跳过 soft fusion 会让校验证据不足。

## 5. 失败信息

校验失败时，错误信息包含：

- `config_name`
- `split`
- `batch_index`
- `row_offset`
- `sample_key`
- mismatch 字段名
- 正式路径旧值
- adapter 复算值
- `output_dir`

这保证 dry-run 或 eval-only 失败时能直接定位到具体 batch 和样本。

## 6. 测试

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
```

该测试不启动 ViT，不访问 `/data2`，不运行正式入口。它用小型 numpy arrays 和
DataFrame 构造与 Visual Router test batch 等价的 `pred_df` / `soft_df`，调用
新增 helper 验证 adapter rows 与正式字段一致，并覆盖一次故意 mismatch 的失败信息。

## 7. 后续

P9b 通过后，下一步才考虑 P9c 的 `ExpertBatch` 对齐，或更小粒度 prediction batch
adapter 对齐。P9b 本身不承担正式 evaluation 迁移。
