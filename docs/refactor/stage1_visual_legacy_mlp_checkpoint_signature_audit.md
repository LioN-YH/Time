# Stage 1 P16g Visual Legacy MLP Checkpoint Signature Audit

审计日期：2026-06-21

## 1. 目标

P16g 只做 legacy `VisualMLPRouter` constructor、forward、checkpoint payload、
`state_dict` key、DataParallel、scaler 和 device/runtime loading 边界审计。目标是为后续
Runtime 加载真实 `VisualMLPRouter`，再交给 P16a `LoadedTorchMLPRouterHeadAdapter` 做准备。

本步不新增 checkpoint loader 代码，不修改 `time_router/models/visual_mlp_adapter.py`，
不读取真实 checkpoint，不访问 `/data2`，不修改正式 Visual Router 入口，也不声称正式
Visual Router 已迁移完成。

## 2. 代码来源

legacy `VisualMLPRouter` 定义在：

```text
visual_router_experiments/stage1_vali_test_router/train_visual_router.py
```

streaming 正式入口通过 import 复用该类：

```text
visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```

相关证据：

- `train_visual_router.py` 定义 `class VisualMLPRouter(nn.Module)`。
- `train_visual_router_online_streaming.py` import `VisualMLPRouter`，并在 runtime 中实例化、
  load `router_state_dict`、保存 checkpoint。

## 3. Constructor Signature

`VisualMLPRouter.__init__` 参数为：

```text
input_dim: int
hidden_dim: int
output_dim: int
dropout: float
```

内部结构是固定 `nn.Sequential`：

```text
Linear(input_dim, hidden_dim)
GELU()
Dropout(dropout)
Linear(hidden_dim, output_dim)
```

因此 legacy checkpoint 的 router 权重 key 预期来自 `network` 子模块，例如：

```text
network.0.weight
network.0.bias
network.3.weight
network.3.bias
```

本次未读取真实 checkpoint；上述 key 是从源码结构推导的预期，而不是来自真实文件扫描。

## 4. Forward Signature And Output

`forward` 签名为：

```text
forward(features: torch.Tensor) -> torch.Tensor
```

forward 直接返回：

```text
self.network(features)
```

legacy 调用侧在推理中执行：

```text
logits = router(torch.from_numpy(x_test_scaled).to(device=device))
weights = torch.softmax(logits, dim=1)
```

因此 `VisualMLPRouter` 自身输出未归一化 logits。正常输入为二维
`[num_samples, input_dim]` tensor，正常输出为二维 `[num_samples, output_dim]` logits
tensor。P16a `LoadedTorchMLPRouterHeadAdapter` 的“已加载 torch module + head-ready
FeatureBatch -> RouterOutput”边界与该 forward 形式匹配，但 adapter 仍只负责调用已加载
module，不负责构造 legacy model 或读取 checkpoint。

## 5. 输入 Feature Dim

非 streaming 入口：

- 先按 manifest 或 online lookup 得到 ViT embedding matrix。
- 使用 `StandardScaler().fit_transform(x_vali)` 产生 `x_vali_scaled`。
- `input_dim=int(x_vali_scaled.shape[1])`。

streaming 入口：

- vali split 运行 `iter_online_embedding_batches(...)`，在线构造 pseudo image、冻结 ViT 前向、
  pooling 得到 embedding。
- `StandardScaler.partial_fit(embeddings)` 在 vali embedding 流上估计 scaler。
- 新训练时 `input_dim=int(scaler.n_features_in_)`。
- resume 时从 checkpoint 的 `scaler_state` 重建 scaler，并用
  `embedding_dim = int(scaler.n_features_in_)`。

所以 `input_dim` 取决于 head-ready Visual feature 的维度，当前 legacy 正式 streaming
路径中实际来自 ViT pooled embedding 维度，并经 scaler transform 后送入 MLP。`pooling`
策略、ViT encoder hidden size、future pooling dim 或 alternative visual encoder 都会影响该维度。

scaler 不改变 feature_dim，只保存和校验同一维度上的 mean/scale。scaler state loading /
transform 应继续放在 P16d `LoadedFeatureScaler` 或 Runtime 侧，不进入 RouterHead adapter。

## 6. 输出维度和 model_columns

`output_dim` 在训练和 streaming 入口均使用：

```text
len(MODEL_COLUMNS)
```

`MODEL_COLUMNS` 从 `fusion_utils.py` 导入，当前 Stage 1 五专家顺序为：

```text
DLinear
PatchTST
CrossFormer
ES
NaiveForecaster
```

logits 第二维必须与该顺序一一对应。后续 Runtime 加载真实 legacy MLP 时，应显式传入
`model_columns` 给 P16a adapter，并校验 `logits.shape[1] == len(model_columns)`；不应从
checkpoint path、prediction cache 或 CSV 反推专家顺序。

## 7. Checkpoint Payload 格式

streaming 入口保存 checkpoint 的函数是 `save_checkpoint(...)`。保存格式不是裸
`state_dict`，也不是常见的 `model_state_dict` key，而是一个 dict payload，核心字段包括：

```text
checkpoint_version: "stage1_streaming_router_checkpoint_v1"
router_state_dict: router.state_dict()
optimizer_state_dict: optimizer.state_dict()
scaler_state: scaler_to_state(scaler)
completed_epochs: int
epoch_summaries: list[dict]
scaler_batches: int
scaler_samples: int
saved_at: str
```

此外 payload 还展开保存严格 resume signature，包括：

```text
config_name
model_columns
router_mode
metric
hidden_dim
dropout
lr
weight_decay
huber_beta
kl_tau
lambda_kl
embedding_metadata
stream_shard_index
stream_shard_count
labels_path
prediction_manifest_path
config_path
```

checkpoint 文件命名：

```text
checkpoints/router_{config}_epoch_000N.pt
checkpoints/latest_{config}.pt
checkpoints/latest_checkpoint_index.json
```

loading 当前由 `load_checkpoint(path)` 完成，使用：

```text
torch.load(path, map_location="cpu", weights_only=False)
```

若 PyTorch 版本不支持 `weights_only` 参数，则 fallback 为：

```text
torch.load(path, map_location="cpu")
```

`assert_checkpoint_matches(...)` 会检查 resume signature。随后：

```text
router.load_state_dict(resume_checkpoint["router_state_dict"])
optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
move_optimizer_state_to_device(optimizer, device)
```

因此后续 loader smoke 应按 `router_state_dict` key 读取，而不是假设 checkpoint 是裸
state_dict 或 `model_state_dict`。

## 8. DataParallel 和 module. 前缀

当前 streaming 入口的 `--vit-data-parallel` 只包裹冻结 ViT encoder：

```text
model = torch.nn.DataParallel(model)
```

router 本身未被 `DataParallel` 包裹，`save_checkpoint(...)` 保存的是
`router.state_dict()`。因此源码层面预期 `router_state_dict` 不带 `module.` 前缀。

但是后续 Runtime loader 仍应把 DataParallel key 作为独立 smoke 覆盖项处理：

- 如果历史 checkpoint 来自手工包裹的 router 或未来多卡训练，可能出现 `module.` 前缀。
- 前缀剥离或兼容策略属于 Runtime/checkpoint loader，不属于 P16a
  `LoadedTorchMLPRouterHeadAdapter`。
- 本次未读取真实 checkpoint，不能声称所有历史 checkpoint 都不存在 `module.` 前缀。

## 9. Scaler Boundary

streaming checkpoint 内嵌 `scaler_state`，由 `scaler_to_state(...)` 保存：

```text
mean_
scale_
var_
n_features_in_
n_samples_seen_
feature_names_in_  # 可选
```

resume 时 `scaler_from_state(...)` 重建 `StandardScaler`，后续 batch 用：

```text
x_scaled = scaler.transform(embeddings).astype(np.float32)
```

边界结论：

- scaler state loading 可以在 Runtime 中解析 legacy checkpoint，也可以转成 P16d
  `LoadedFeatureScaler` 能消费的显式 state。
- scaler transform 是 pre-head `FeatureBatch -> head-ready FeatureBatch`，不进入 P16a
  RouterHead adapter。
- adapter 不应接收 `scaler_path`、`checkpoint_path`、`run_dir` 或自动 fit scaler。

## 10. Device / Runtime Loading Boundary

checkpoint loading、module construction、state_dict strictness、device selection 和 optimizer
resume 都属于 Runtime / entrypoint，不属于 P16a adapter。

建议后续 Runtime 侧显式处理：

- `map_location`：先 CPU load，再将构造好的 router module `.to(device)`。
- constructor signature：从 checkpoint signature 或调用方 config 解析 `input_dim`、
  `hidden_dim`、`output_dim=len(model_columns)`、`dropout`。
- `strict=True` 默认加载，独立记录 missing / unexpected keys；任何兼容 `module.` 前缀或
  key rename 的策略都必须有 smoke 覆盖。
- device：adapter 只接收已加载 module 和显式 device；它不负责 auto device policy 或
  DataParallel。
- optimizer/scaler/epoch：训练 resume 可加载；eval-only loaded-module smoke 不需要加载
  optimizer state。

## 11. 后续最小实现建议

P16h 可做 legacy `VisualMLPRouter` loaded-module smoke：

1. 使用仓库内 in-memory fake `state_dict` 或 tiny checkpoint fixture。
2. 实例化 legacy `VisualMLPRouter(input_dim, hidden_dim, output_dim, dropout)`。
3. 加载 `router_state_dict`，构造 head-ready `FeatureBatch`。
4. 交给 P16a `LoadedTorchMLPRouterHeadAdapter`。
5. 验证 logits / weights shape、sample_key 保序、model_columns 保序、strict loading、
   missing/unexpected keys 和可选 `module.` 前缀兼容策略。

P16h 明确不做：

- 不读取 `/data2` 真实 checkpoint。
- 不接真实 ViT。
- 不启动训练、pressure 或 full-scale。
- 不迁移正式 Visual Router 入口。
- 不把 checkpoint path、scaler path 或 run_dir 设计进 P16a adapter interface。

## 12. P16a Adapter 对接结论

P16a `LoadedTorchMLPRouterHeadAdapter` 的长期边界保持不变：

```text
已加载 torch.nn.Module + head-ready float32 FeatureBatch + explicit model_columns
-> RouterOutput(logits, weights)
```

legacy `VisualMLPRouter` constructor、checkpoint payload、scaler state、DataParallel key、
`map_location`、strict loading、optimizer resume 和 runtime device policy 均在 adapter 外部处理。
这能避免把 legacy checkpoint/run_dir 细节泄漏进 canonical RouterHead adapter interface。
