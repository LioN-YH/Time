# Stage 1 P8a TimeFuse Entrypoint Adapter Insertion Audit

创建日期：2026-06-20

## 1. 目标

本文记录 P7c TimeFuse protocol chain smoke 之后，对正式入口 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` 的最小 adapter 接入点审计。

本步只做文档化接入计划，不修改正式训练入口行为，不迁移入口，不改变 CSV、summary、checkpoint、status 或 metadata schema。

## 2. 当前正式入口边界

`train_timefuse_fusor_streaming.py` 目前仍是 TimeFuse-style fusor 的 canonical-current 入口。它在一个脚本内同时承担：

- CLI 参数、默认路径、输出目录和 GPU 约束。
- feature shard 发现、split subset、shard-local oracle/prediction SQLite index 准备。
- `StandardScaler` 在 vali feature streaming 上 `partial_fit`。
- `TimeFuseFusor` 的 torch 训练、`SmoothL1Loss`、optimizer、epoch loop、DataParallel。
- test split streaming evaluation。
- `timefuse_fusor_predictions.csv`、`sample_predictions.csv`、三个 summary CSV、`summary.md`、`metadata.json`、`status.json` 和 checkpoint 写出。

P8a 的结论是：最小 adapter 接入点应先选 evaluation 阶段，而不是 reader、scaler、training loop 或 report writer。

## 3. 最适合先接 EvaluationInputAdapter 的位置

最小接入点是 `evaluate_streaming(...)` 中每个 test batch 已经完成 torch 前向之后：

```text
batch.sample_keys
batch.y_pred
batch.y_true
weights_np = fusor(scaler.transform(batch.features))
MODEL_COLUMNS
```

这一段已经同时持有 `EvaluationInputAdapter` 需要的全部纯内存输入：

- `sample_keys`：来自 `TimeFuseFusorBatch.sample_keys`。
- `model_columns`：当前固定为 `MODEL_COLUMNS`。
- `y_pred`：来自 streaming reader 的五专家 prediction tensor。
- `y_true`：来自 streaming reader 的共享真实值 tensor。
- `weights`：来自正式 torch fusor 的 softmax 输出。

因此 P8b 可以在 `evaluate_streaming(...)` 内构造最小 in-memory adapter 输入，对当前 batch 复算 hard top-1、raw soft fusion、weight diagnostics 和 per-sample rows。这个接入点不需要改 reader、scaler、训练循环、checkpoint 或 launcher。

## 4. 可以复用 `time_router.evaluation` 的逻辑

P8b 可优先复用以下逻辑：

- `EvaluationInputAdapter.evaluate_input(...)` 作为 canonical adapter 调用点。
- `hard_top1_fusion(...)` 复算 hard top-1 选择和 hard metrics。
- `raw_soft_fusion(...)` 复算 raw soft fusion metrics。
- `build_fusion_summary(...)` 复算内存 summary dict。
- `build_per_sample_fusion_rows(...)` 复算内存 per-sample rows。
- `compute_weight_entropy(...)`、`compute_max_weight(...)` 和 selected counts 相关 helper 的既有口径。

复用范围应限制为内存 metrics 复算和一致性校验。P8b 不应让 `EvaluationInputAdapter` 决定正式文件名、CSV 字段顺序、summary markdown 格式或 run artifact 位置。

## 5. 必须暂留在正式入口的逻辑

以下逻辑短期必须继续留在 `train_timefuse_fusor_streaming.py` 或其 runtime/report 层，不能在 P8b 下沉到 `EvaluationInputAdapter`：

- CSV 写出：`timefuse_fusor_predictions.csv`、`sample_predictions.csv`、`timefuse_fusor_summary.csv`、`timefuse_fusor_raw_soft_fusion_summary.csv`、`timefuse_fusor_selected_model_counts.csv`。
- `summary.md` 写出和 `frame_to_markdown(...)` 展示口径。
- `checkpoint`、`status.json`、`metadata.json`、`latest_checkpoint_index.json` 和 `main.log`。
- `StandardScaler` 的 vali streaming `partial_fit`、checkpoint 保存/恢复和 test transform 调用顺序。
- `optimizer`、`SmoothL1Loss`、epoch loop、DataParallel、train-only/eval-only/resume 分支。
- feature shard subset、shard-local SQLite index、oracle label 和 prediction manifest 准备。
- oracle regret、oracle label correct、metadata 字段展开、正式 CSV 中的历史字段命名。

这些逻辑混合了正式 output schema、runtime 状态、训练副作用和 full-scale 接手能力，不属于 smoke-only `EvaluationInputAdapter` 的职责。

## 6. P7a/P7b 不能直接替换正式入口

P7a `TimeFuseFeatureCacheProvider` 当前只是 smoke-only 小规模 CSV adapter：

- 它读取调用方显式传入的小 CSV，并通过 `load_batch(sample_keys)` 返回一个 `FeatureBatch`。
- 它不实现 full-scale shard streaming、split 下推、chunked reader、shard-local index 复用或 scaler fit 数据流。
- 它不能直接替换 `Stage1TimeFuseFusorStreamingReader` 或 `fit_scaler_streaming(...)`。

P7b `TimeFuseLinearSoftmaxHead` 当前只是 numpy smoke head：

- 它用固定 numpy weight/bias 做 deterministic `features -> logits -> softmax`。
- 它不包含 torch `nn.Module`、梯度、`state_dict`、DataParallel、optimizer 或 checkpoint 兼容。
- 它不能直接替换正式 `TimeFuseFusor` torch 训练 head。

因此 P8b 的最小迁移只能先利用 `EvaluationInputAdapter` 复算 evaluation metrics；feature provider 和 head adapter 的正式替换必须晚于 reader/training/runtime 边界设计。

## 7. P8b 最小代码迁移建议

P8b 建议只做 evaluation 阶段旁路复算，不改变正式输出 schema：

1. 在 `evaluate_streaming(...)` 中，拿到 `weights_np` 后构造 batch 级 `EvaluationInput` 或等价 `ExpertBatch + RouterOutput`。
2. 调用 `EvaluationInputAdapter.evaluate_input(...)` 复算 batch hard/raw-soft summary 和 per-sample rows。
3. 用复算结果与现有手写 batch 逻辑做严格一致性断言或 debug-only 统计，例如 hard selected index、hard MAE/MSE、raw soft MAE/MSE、max weight 和 entropy。
4. 继续由现有代码写正式 CSV 和 summary CSV；adapter rows 只作为内存校验，不直接写文件。
5. 先在小 shard/pressure 或 smoke 参数下开启一致性校验，再决定是否默认开启。

P8b 不应：

- 改 `fieldnames` 或 CSV 字段顺序。
- 改 `summarize_hard_predictions(...)`、`summarize_soft_fusion(...)`、`summarize_selected_model_counts(...)` 的正式写出口径。
- 改 `summary.md`、`metadata.json`、`status.json` 或 checkpoint payload。
- 改 reader、scaler、optimizer、loss、epoch loop 或 launcher。
- 访问 `/data2` 做新的 full-scale 验证。

## 8. P8b 验收建议

最小验收应继续使用 smoke 和 compileall：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

若 P8b 修改正式入口，应额外使用小规模 explicit `--output-dir` pressure run 比较迁移前后的 CSV schema、行数、sample_key 顺序、hard/raw-soft MAE/MSE 和 selected counts；比较通过前不能启动 full-scale。

## 9. P8a 明确不做

- 不修改 `train_timefuse_fusor_streaming.py`。
- 不迁移正式入口。
- 不改 CSV、summary、checkpoint、status 或 metadata schema。
- 不改 reader、scaler、optimizer、loss 或 epoch loop。
- 不访问 `/data2`。
- 不新增 Bash 或 scripts。
- 不改 Visual Router 入口。
