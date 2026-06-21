# Stage 1 P16f Visual Feature Chain Protocol Skeleton

## 目的

P16f 新增 Visual feature chain 的最小 protocol skeleton，用来确认 Stage 1 canonical 重构后续可以逐步替换视觉特征链路中的关键插槽。它不是正式 Visual provider 实现，也不代表真实 ViT、RevIN、resize、pseudo image 或 pooling 已经迁移。

## 当前边界

新增 `time_router/features/visual_chain.py` 只定义轻量 `Protocol`、`dataclass` 和 type alias：

- `RawWindowBatch`
- `PreImageBatch`
- `VisualInputBatch`
- `VisualEmbeddingBatch`
- `RawWindowProvider`
- `PreImageTransform`
- `PseudoImageTransformer`
- `ResizePolicy`
- `VisualEncoderProvider`
- `PoolingStrategy`
- `FeatureTransform`
- `VisualFeatureChainSpec`

这些类型只表达输入输出 contract。每一层都必须保留 ordered `sample_keys`，每一层 metadata 只保存轻量 lineage，不保存 cache path、checkpoint path、run_dir 或正式训练状态。最终 router/fusor 仍只消费 canonical `FeatureBatch`。

## 可替换插槽

P16f 把视觉特征链拆成以下可替换位置：

- `RawWindowProvider`：未来可以从 runtime 已规划的 sample batch 中取 history window。
- `PreImageTransform`：未来可以接 RevIN、normalization 或 identity。
- `PseudoImageTransformer`：未来可以接真实 pseudo image / imageization 逻辑。
- `ResizePolicy`：未来可以接 resize、image processor input policy 或 identity。
- `VisualEncoderProvider`：未来可以接 frozen ViT、其他 encoder 或测试 encoder。
- `PoolingStrategy`：未来可以接 `cls`、`mean_patch`、`mean_pool` 等 pooling 口径。
- `FeatureTransform`：未来可以接 `LoadedFeatureScaler`、identity 或其他 normalizer。

当前 skeleton 不把原版 Visual Router 写死为唯一方案，也不强制正式 pipeline 落盘 CSV、embedding cache 或 pseudo image cache。cache、checkpoint、device、batching 和 artifact 写出都应留在 Runtime / entrypoint 层。

## Smoke 覆盖

新增 `tests/smoke/stage1_visual_feature_chain_protocol_smoke.py` 使用 smoke-local dummy components 串联：

```text
P13b ordered sample_keys
-> DummyRawWindowProvider
-> IdentityPreImageTransform
-> DummyPseudoImageTransformer
-> IdentityResizePolicy
-> DummyVisualEncoderProvider
-> DummyPoolingStrategy
-> IdentityFeatureTransform
-> FeatureBatch
-> LoadedTorchMLPRouterHeadAdapter
-> EvaluationInputAdapter
```

smoke 只验证 protocol chain 可组合：

- 每层 `sample_keys` 保序；
- 每层 batch shape 合理；
- 最终 `FeatureBatch.features` 为 `float32`；
- `feature_schema.chain_lineage` 记录 `raw_window / pre_image / pseudo_image / resize / encoder / pooling / transform`；
- 替换一个 dummy `PreImageTransform` 后仍能输出合法 `FeatureBatch`；
- P16a `LoadedTorchMLPRouterHeadAdapter` 和 `EvaluationInputAdapter` 可消费最终 `FeatureBatch`。

## 明确不做

P16f 不实现以下内容：

- 真实 RevIN；
- 真实 pseudo image；
- 真实 resize；
- 真实 ViT / HuggingFace encoder；
- 真实 pooling 策略；
- scaler fit 或 checkpoint state discovery；
- 正式 provider；
- 正式 Visual Router entrypoint migration；
- Bash launcher、训练、pressure 或 full-scale；
- `/data2` 访问；
- canonical run_dir 写出。

后续真实实现可以逐步替换 dummy component，只要继续保持 ordered `sample_keys`、轻量 lineage 和最终 canonical `FeatureBatch` contract。
