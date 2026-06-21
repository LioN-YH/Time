# Stage 1 P16f Visual feature chain protocol skeleton

日志日期：2026-06-21 13:40:02 CST

## 目的

新增 Visual feature chain 的最小 protocol skeleton，确认后续真实视觉特征链路可以按可替换插槽逐步接入，同时不提前绑定真实 RevIN、pseudo image、resize、ViT、pooling、scaler、cache、checkpoint 或正式入口。

## 背景

P16a 已完成 `LoadedTorchMLPRouterHeadAdapter`，P16b 已审计真实 Visual feature provider 边界，P16c/P16d 分别完成 precomputed feature provider 和 loaded feature scaler 的 smoke，P16e 已把 Visual feature architecture variant 边界文档化。当前仍不应接真实 ViT、不访问 `/data2`、不迁移正式 Visual Router 入口。本步需要把 raw window 到最终 `FeatureBatch` 的可扩展插槽用轻量协议表达出来，供后续真实组件替换。

## 操作

1. 新增 `time_router/features/visual_chain.py`，定义 `RawWindowBatch`、`PreImageBatch`、`VisualInputBatch`、`VisualEmbeddingBatch`、`RawWindowProvider`、`PreImageTransform`、`PseudoImageTransformer`、`ResizePolicy`、`VisualEncoderProvider`、`PoolingStrategy`、`FeatureTransform` 和 `VisualFeatureChainSpec`。
2. 更新 `time_router/features/__init__.py`，同步导出 P16f protocol skeleton 类型。
3. 新增 `tests/smoke/stage1_visual_feature_chain_protocol_smoke.py`，用 smoke-local dummy components 串联 `RawWindowProvider -> PreImageTransform -> PseudoImageTransformer -> ResizePolicy -> VisualEncoderProvider -> PoolingStrategy -> FeatureTransform -> FeatureBatch -> LoadedTorchMLPRouterHeadAdapter -> EvaluationInputAdapter`。
4. 新增 `docs/refactor/stage1_visual_feature_chain_protocol.md`，说明 P16f 只是 protocol skeleton，不是真实 Visual provider 实现。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增协议模块、文档和 smoke。
6. 运行新增 compileall、新增 smoke、P16d/P16c/P16a 回归 smoke，并执行 diff 边界检查。

## 结果

新增 protocol skeleton 只表达输入输出 contract：

- 每层 batch 都带 ordered `sample_keys`、payload 和轻量 metadata。
- `VisualFeatureChainSpec` 记录 raw window、pre-image、pseudo image、resize、encoder、pooling 和 optional feature transform 组件组合。
- 最终输出仍是 canonical `FeatureBatch`。
- 模块未导入 torch、transformers 或 sklearn，也未绑定 cache path、checkpoint path 或 run_dir。

新增 smoke 验证结果：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_chain.py tests/smoke/stage1_visual_feature_chain_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_protocol_smoke.py
```

结果通过。新增 smoke 输出确认 raw/pre-image/pseudo-image/resize/encoder/pooling/transform 每层保序且 shape 合理，最终 `FeatureBatch.features` 为 `float32`，schema 记录 `raw_window / pre_image / pseudo_image / resize / encoder / pooling / transform` lineage，替换一个 dummy pre-image component 后仍输出合法 `FeatureBatch`，且最终可被 P16a adapter 和 `EvaluationInputAdapter` 消费。

指定回归 smoke 均通过：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
```

`git diff --check` 通过。`git diff -- scripts/run_stage1_visual_small.py time_router/features/visual_precomputed.py time_router/features/visual_scaler.py time_router/models/visual_mlp_adapter.py` 无输出，说明未修改正式 Visual small entrypoint，也未修改 P16a/P16c/P16d 代码。新增 smoke 的边界检查确认未访问 `/data2`、未读取 checkpoint、未启动 ViT、未创建 canonical run_dir。

## 结论

P16f 已完成 Visual feature chain 的最小协议骨架和 dummy-chain smoke。当前实现只为后续架构扩展提供接口形状和组合验证，不把原版 Visual Router、CSV/cache、ViT、scaler 或 checkpoint 设计成强制路径。

## 下一步方案

后续可以在该协议骨架上继续小步推进：先做真实 Visual chain 的 fake encoder / online ViT provider audit，再独立审计 legacy checkpoint/signature 和正式 Visual entrypoint migration plan。任何真实 ViT、RevIN、resize、pooling 或 scaler 接入都应继续保持 ordered `sample_keys`、轻量 lineage 和最终 canonical `FeatureBatch` contract。
