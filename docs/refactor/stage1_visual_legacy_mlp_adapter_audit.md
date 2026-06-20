# Stage 1 P14e Visual Legacy MLP Adapter Audit

创建日期：2026-06-20

## 1. 目标

P14e 只审计 Visual eval-only 阶段 legacy `VisualMLPRouter` 如何被薄包装为
canonical `RouterOutput`。本阶段不新增正式 adapter 代码，不修改正式入口，不访问
`/data2`，不启动训练、pressure 或 full-scale。

P14d 已证明 tiny fixture 上可以完成：

```text
FeatureBatch + ExpertBatch
  -> smoke-only mock RouterHead
  -> RouterOutput
  -> EvaluationInputAdapter
  -> summary / rows
```

P14e 的问题是：正式 Visual Router 当前仍由 legacy `VisualMLPRouter`、`StandardScaler`、
checkpoint、device/dtype 和 DataParallel 等入口逻辑共同驱动。后续如果做 eval-only
legacy MLP adapter，应把边界收窄到 `FeatureBatch -> RouterOutput`，不要顺手迁移正式入口。

## 2. 当前 legacy MLP eval-only 输入

当前正式 streaming 入口的 eval-only 路径仍在
`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
内完成。核心链路是：

```text
labels / sample metadata
  -> windows_from_labels(...)
  -> iter_online_embedding_batches(...)
     -> history window x
     -> make_pseudo_images(...)
     -> frozen ViT forward
     -> pool_vit_outputs(...)
     -> embeddings: np.ndarray float32 [batch, embedding_dim]
  -> scaler.transform(embeddings).astype(np.float32)
  -> VisualMLPRouter(torch.from_numpy(x_scaled).to(device))
```

`VisualMLPRouter` 定义在
`visual_router_experiments/stage1_vali_test_router/train_visual_router.py`，其 `forward`
只接收一个 torch tensor `features`，返回未归一化 logits。类注释已说明输入是
“ViT embedding，经 vali-fitted StandardScaler 标准化”。

因此 future eval-only adapter 的最小输入应是：

```text
FeatureBatch(sample_keys, features)
+ model_columns
+ runtime-loaded legacy MLP checkpoint/scaler/device context
-> RouterOutput(sample_keys, model_columns, logits, weights, extra)
```

其中 `FeatureBatch.features` 对应 legacy eval-only 的 MLP 输入矩阵。严格说，它应是
**已经完成 eval-only transform 的 head-ready features**，也就是 `scaler.transform(...)`
后的 `float32 [sample, embedding_dim]`。如果调用方选择把 raw ViT embedding 放进
`FeatureBatch.features`，则必须在 adapter 前显式增加 pre-head transform step；不应让
adapter 自己寻找或 fit scaler。

## 3. logits / weights 到 RouterOutput

当前 legacy eval-only 在 `predict_stream_batch(...)` 中执行：

```text
logits = router(torch.from_numpy(x_scaled).to(device=device))
weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
selected_indices = weights.argmax(axis=1)
```

`VisualMLPRouter.forward(...)` 输出 logits，softmax 后得到专家融合权重。future thin
adapter 可把二者包装为：

- `RouterOutput.sample_keys`：等于 `FeatureBatch.sample_keys`，不得重排。
- `RouterOutput.model_columns`：来自调用方显式传入的 `model_columns`。
- `RouterOutput.logits`：legacy MLP 的未归一化输出，shape `[sample, model]`。
- `RouterOutput.weights`：对 logits 按 expert 维度 softmax 后的 `float32` 数组，shape
  `[sample, model]`，每行应近似归一化。
- `RouterOutput.extra`：只记录轻量 head lineage，例如 `head_source=legacy_visual_mlp`、
  `adapter_scope=eval_only`、`checkpoint_loaded_by=runtime`、`scaler_applied_by=runtime_or_pre_head`。

adapter 应显式检查：

- `len(sample_keys) == features.shape[0] == logits.shape[0] == weights.shape[0]`；
- `len(model_columns) == logits.shape[1] == weights.shape[1]`；
- `features/logits/weights` 均为有限值；
- `weights` 非负且 row sum 接近 1；
- 输出 `sample_keys` 与输入完全一致。

## 4. model_columns 对齐

当前 legacy 入口使用 `MODEL_COLUMNS` 作为统一专家顺序：

- checkpoint signature 保存 `model_columns: list(MODEL_COLUMNS)`；
- router 初始化使用 `output_dim=len(MODEL_COLUMNS)`；
- loss、专家预测读取、`weight_<model_name>` 列、selected_model 映射都使用同一个顺序；
- P9d/P9f 的 `ExpertBatch` 旁路也显式使用 `MODEL_COLUMNS`。

future eval-only adapter 不应从 CSV 列名、checkpoint 文件名或 prediction cache 反推专家顺序。
它应接收显式 `model_columns`，并要求该顺序与 `ExpertBatch.model_columns` 完全一致。
如果 `RouterOutput.model_columns != ExpertBatch.model_columns`，应由 adapter 或
`EvaluationInputAdapter` 明确报错，而不是按名称静默重排。

当前 Stage 1 canonical experiment 可继续使用固定五专家顺序，但该顺序属于本实验配置，
不应上升为全局专家系统契约。

## 5. scaler 边界

`StandardScaler` 在 legacy Visual Router 中有两类行为：

- training / runtime state：vali embedding 上 `partial_fit` 或旧 offline 路径 `fit_transform`，
  以及 checkpoint 中保存 `scaler_state`。
- eval-only transform：resume 后从 checkpoint 重建 scaler，对 test embedding 执行
  `scaler.transform(...).astype(np.float32)`。

P14e 结论：

- scaler fit 属于 training/runtime state，不属于 adapter。
- scaler checkpoint 序列化和恢复属于 Runtime/entrypoint，不属于 adapter。
- eval-only transform 可以由 Runtime 完成，也可以由 adapter 前的显式 pre-head transform step
  完成。
- thin adapter 最稳妥的输入是 head-ready `FeatureBatch.features`，即已经 transform 后的
  `float32` 特征。
- 如果后续为了 smoke 便利让 adapter 接收一个已加载 scaler 对 raw embedding 做 transform，
  也必须把它标注为 pre-head transform responsibility，而不是让 adapter 自己 fit scaler
  或寻找 checkpoint。

本轮不改变现有 scaler、checkpoint 或 resume schema。

## 6. checkpoint / device / dtype / DataParallel 边界

当前 streaming 入口中：

- `load_checkpoint(...)`、`assert_checkpoint_matches(...)`、`scaler_from_state(...)`、
  `router.load_state_dict(...)` 和 `optimizer.load_state_dict(...)` 都属于入口层。
- `resolve_device(...)` 与 `resolve_dtype(...)` 在入口层决定运行设备和 encoder dtype。
- `build_vit_model(...)` / `load_vit_model_with_retry(...)` 构建 frozen ViT，处理
  Hugging Face cache、本地/远端读取、retry、`fp16` 和 `DataParallel`。
- `--vit-data-parallel` 只包裹 frozen ViT encoder；router/scaler/checkpoint 仍保持单进程语义。
- legacy `VisualMLPRouter` 自身只 `.to(device)` 并执行 forward。

P14e 结论：

- checkpoint loading、checkpoint signature 校验、resume 决策和 optimizer state 迁移属于
  Runtime/entrypoint。
- adapter 可以消费已加载并已放到目标 device 的 legacy MLP，但不负责查找 checkpoint path。
- dtype/device/DataParallel 由 Runtime/entrypoint 管理。adapter 不决定全局 device，也不包装
  DataParallel。
- adapter 可以在 forward 前把 `FeatureBatch.features` 转为 torch tensor 并移动到 runtime
  指定 device；这只是消费 runtime context，不是资源策略决策。
- ViT encoder dtype 与 DataParallel 只影响 feature 生成阶段，属于 Visual FeatureProvider /
  encoder factory / Runtime 边界，不属于 legacy MLP adapter。

## 7. adapter 应该做什么

future eval-only legacy MLP adapter 可以做：

- 接收已经准备好的 `FeatureBatch`。
- 接收显式 `model_columns`。
- 接收 Runtime 已加载的 legacy `VisualMLPRouter` 或 smoke-only 小型 MLP。
- 在 `torch.inference_mode()` 下调用 head。
- 对 logits 做 softmax，输出 `RouterOutput`。
- 检查 sample_keys、feature shape、logits/weights shape、model_columns 和有限值。
- 在 `extra` 中记录轻量 head lineage。

## 8. adapter 不应该做什么

future eval-only legacy MLP adapter 不得：

- 读取 prediction cache。
- 读取或消费 `ExpertBatch.y_pred/y_true`。
- 读取 oracle/error、labels CSV 或 sample supervision。
- 读取 `run_dir`、status、metadata 或 checkpoint path。
- 自己加载 checkpoint、决定 resume 或写 checkpoint。
- 自己 fit scaler。
- 自己决定全局 device/dtype/DataParallel。
- 写 evaluation CSV、summary、prediction rows 或 canonical run artifacts。
- 改 loss、optimizer、scheduler、training loop 或 `fusion_huber_kl` 监督路径。

## 9. 后续小步建议

推荐 P14f：

- 做 Visual legacy MLP adapter smoke。
- 使用 tiny `FeatureBatch` 和小型 torch MLP / loaded `state_dict` fixture。
- 输入直接使用 head-ready float32 features，不访问真实 ViT、不访问 `/data2`。
- 输出 `RouterOutput`，验证 sample_key 保序、model_columns 对齐、logits/weights shape、
  softmax row sum、有限值和 `EvaluationInputAdapter` 可消费。
- smoke-only adapter 可放在测试内或明确标注为 smoke-only，避免被误认为正式入口已迁移。

推荐 P15a：

- branch-specific small entrypoint decision。
- 基于 P13d/P13e/P14a/P14b/P14c/P14d/P14e/P14f 结果，决定是否新增 Visual-specific /
  TimeFuse-specific small entrypoint，而不是把 branch-specific feature/head 逻辑塞回
  generic small canonical CLI。

## 10. P14e 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增正式 Visual RouterHead adapter 代码。
- 不新增正式 VisualFeatureProvider。
- 不抽真实 ViT provider。
- 不接 legacy `VisualMLPRouter` 到 canonical pipeline。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 11. P14e 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```
