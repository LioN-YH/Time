# Stage 1 P16b Real Visual Feature Provider Boundary Audit

创建日期：2026-06-21

## 1. 目标

P16b 审计并冻结 real Visual feature provider 的边界。P16a 已经把正式
Visual MLP RouterHead adapter 收束为：

```text
已加载 torch.nn.Module + head-ready FeatureBatch -> RouterOutput
```

因此 P16b 只回答 head-ready `FeatureBatch` 应如何由真实 Visual 特征链路准备出来。
本步只做设计文档和迁移拆分，不新增 provider 代码，不接真实 ViT，不读取 checkpoint，
不访问 `/data2`，不迁移正式 Visual Router 训练入口。

当前 legacy streaming 入口中的真实链路仍是：

```text
labels / sample metadata
  -> windows_from_labels(...)
  -> Quito history window x
  -> make_pseudo_images(...)
  -> frozen ViT forward
  -> pool_vit_outputs(...)
  -> StandardScaler.partial_fit / transform
  -> VisualMLPRouter
```

P16b 的核心结论是：real Visual feature chain 不应一次性塞进单个巨型 provider。
它应被拆成 raw history、pseudo image、visual encoder、可选 scaler/normalizer 和组合型
`VisualFeatureProvider` 几层。长期 canonical interface 仍只暴露
`FeatureBatch(sample_keys, features, feature_schema, extra)`。

## 2. 输入边界

real Visual feature provider 的调用方应显式传入或通过 Runtime/config 注入以下输入：

- `SampleManifest` 或已由 `SplitStrategy` 选出的 ordered `sample_keys`。
- 每个 `sample_key` 对应的样本身份元信息，例如 `config_name`、`split`、`dataset_name`、
  `item_id`、`channel_id`、`window_index`、`seq_len` 和可选 `pred_len`。
- 可读取历史窗口 `x` 的 raw input source 或专门的 `HistoryWindowProvider`。
- pseudo image 构造参数，例如 `variant`、`norm_mode`、`pixel_mode`、`clip`、`image_size`、
  `normalization_preset`、`period_selection` 和候选周期。
- Runtime 已选择的 visual encoder 配置或已初始化 encoder，例如 frozen ViT name、pooling、
  dtype、device 和 batch size。
- 如果需要 head-ready 标准化，则显式传入已加载的 scaler / normalizer state 或 pre-head
  transform 组件。

输入边界必须排除：

- provider 不从 `run_dir` 推断输入。
- provider 不硬编码 `/data2`、workspace 绝对输出根或 Bash launcher 路径。
- provider 不启动训练、不决定 resume、不读 router checkpoint。
- provider 不读取 oracle、expert error、prediction cache 或 future `y` 作为特征。
- provider 不知道 Bash、tmux、nohup、PID、PGID 或 launcher 接手语义。

换言之，`SampleManifest` / ordered sample_keys 是主索引，history window 是可部署输入，
pseudo image / ViT / scaler 是显式配置或组件，`run_dir` 和 checkpoint 只属于 Runtime。

## 3. 输出边界

real Visual feature provider 的目标输出是 canonical `FeatureBatch`：

- `sample_keys`：与调用方传入的 manifest ordered sample_keys 完全一致，不排序、不去重后重排。
- `features`：二维 array，shape 为 `[num_samples, feature_dim]`。
- `features.dtype`：建议最终为 `numpy.float32`，以匹配 P16a head adapter 的输入约束。
- `feature_schema`：记录 encoder/provider lineage，例如 history source、pseudo image 口径、
  encoder name、pooling、feature_dim、dtype、scaler/normalizer 状态标识。
- `extra`：只记录轻量 metadata，例如 provider name、batch source、版本、是否 runtime-only。

`FeatureBatch.features` 不应包含：

- oracle label、oracle value、per-model error 或 regret；
- expert prediction arrays、`y_true`、prediction cache path；
- checkpoint path、`run_dir`、status path、metadata path；
- pseudo image tensor 大对象路径或长期 cache shard 格式承诺。

P16a `LoadedTorchMLPRouterHeadAdapter` 期望输入已经是 head-ready `float32 FeatureBatch`。
因此如果 legacy Visual MLP 需要 `StandardScaler.transform(...)` 后的输入，这一步必须在
provider 前、provider 内部的显式 pre-head transform 层，或 Runtime 组装阶段完成。RouterHead
adapter 不得偷偷处理 scaler。

## 4. 候选组件拆分

P16b 建议把真实 Visual feature chain 拆成以下候选组件。本节只定义设计语言，不要求本步实现。

| 组件 | 职责 | 推荐归属 |
| --- | --- | --- |
| `HistoryWindowProvider` / `VisualRawInputProvider` | 按 ordered sample_keys 和 manifest metadata 读取或构造过去 history window `x` | 可进入 `time_router/features` 的 Visual-specific raw input 层；具体 Quito dataset 解析可能先留在 legacy/entrypoint |
| `PseudoImageTransformer` | 把 time-series window 转成 pseudo image / visual input tensor | 可作为 `time_router/features` 或 `visual_router_experiments/common` 复用 helper；参数由 Runtime/config 传入 |
| `VisualEncoderProvider` / `FrozenViTFeatureProvider` | 批量执行 frozen ViT 或等价 visual encoder，输出 pooled embedding | encoder 初始化和资源策略偏 Runtime；forward 组件可作为 provider 内部依赖注入 |
| `FeatureScaler` / `FeatureNormalizer` | 把 raw ViT embedding 转成 head-ready feature | 应单独设计边界；fit 和 state loading 不属于 test-time provider |
| `VisualFeatureProvider` | 组合上述组件，按 ordered sample_keys 输出 `FeatureBatch` | 可作为长期 canonical provider；不拥有 run_dir、checkpoint、Bash 或正式 artifact writer |

拆分原则：

- `time_router/features` 可以承载 provider contract、纯特征转换和轻量 schema。
- Runtime / entrypoint 负责 CLI/config、device、batch size、num workers、encoder/scaler state
  loading、checkpoint/resume 和 artifact 写出。
- legacy `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
  暂时仍保留正式路径；迁移前不得把它的全部状态机一次性搬进 provider。

## 5. Scaler 边界

scaler 是 P16b 最需要单独冻结的边界：

- scaler fit 不属于 evaluation/test-time provider。
- scaler state loading 属于 Runtime / entrypoint。
- scaler transform 可以作为 head-ready feature preparation 的一部分，但必须显式设计。
- 不允许 `LoadedTorchMLPRouterHeadAdapter` 内部偷偷做 scaler。
- 不允许 `FeatureProvider` 在没有显式 scaler state 的情况下 silently fit。
- 如果未来需要 scaler adapter，应单独设计和 smoke，例如：

```text
raw ViT FeatureBatch
  -> LoadedFeatureScaler / FeatureNormalizer
  -> head-ready float32 FeatureBatch
  -> LoadedTorchMLPRouterHeadAdapter
```

legacy streaming 入口当前在 vali embedding 上 `StandardScaler.partial_fit`，test 阶段只
`transform`。这个训练/runtime state 可以保留在正式入口，直到 scaler adapter 和 Runtime
checkpoint 接入有单独 smoke 证明边界清楚。

## 6. ViT / Device / Batching 边界

真实 Visual feature path 牵涉 GPU、dtype、Hugging Face cache 和批处理策略，不应由 provider
隐式决定：

- ViT model loading 属于 Runtime / entrypoint，或属于专门 `VisualEncoderProvider` 的初始化阶段。
- provider 不硬编码 GPU id，不调用 Bash，不知道 tmux/nohup/PID。
- `device`、`batch_size`、`num_workers`、dtype、DataParallel、local/remote model loading policy
  应由 Runtime/entrypoint/config 显式传入。
- provider 不硬编码 `/data2`，不把 output root 当输入源。
- 网络抖动 retry、`local_files_only`、Hugging Face cache 策略属于 Runtime/encoder factory。
- full-scale 可启用 cache 或 online encoder，但 cache 是实现选择，不是 provider interface。

P14a 已经指出 `build_vit_model(...)`、`load_vit_model_with_retry(...)` 和
`iter_online_embedding_batches(...)` 当前混合了 encoder 初始化、history 读取、pseudo image、
forward、pooling、latency 和 runtime metadata。P16b 不拆代码，只把未来职责边界固定下来。

## 7. Feature Cache 边界

feature cache 可以作为实现，不应成为长期接口：

- canonical interface 仍是 `FeatureBatch`。
- 不应把 cache path、cache shard、SQLite、NPY、Parquet 或 row-group 格式写成
  `FeatureProvider` 的长期接口。
- small smoke 可以读取 tiny fixture 或 precomputed embedding；full-scale 未来可以读 cache
  或在线编码，但二者都必须输出同一 `FeatureBatch`。
- cache lineage 可进入 `feature_schema` 或 `extra` 的轻量字段，例如 `storage_policy`、
  `cache_version` 或 `runtime_only`，但不应要求 RouterHead adapter 理解 cache 格式。
- cache materialization、cache validation、cache cleanup 和大规模输出目录仍归 Runtime /
  launcher / artifact writer。

对于 Stage 1 full-scale 主线，当前项目规范仍固定为 `x -> pseudo image -> frozen ViT -> router`：
伪图像 tensor 和 ViT embedding 优先只在 batch 运行时生成，不落盘、不作为长期缓存。若后续
为了速度引入 cache，应先证明不会改变 `FeatureBatch` contract。

## 8. 与 P16a Adapter 的关系

P16a adapter 的输入是 head-ready `FeatureBatch`。P16b 不修改 P16a adapter，不把 feature
provider 与 router head 耦合。

未来正式 Visual route 可以串为：

```text
SampleManifest / ordered sample_keys
  -> real VisualFeatureProvider / FeatureBatch
  -> LoadedTorchMLPRouterHeadAdapter / RouterOutput
  -> EvaluationInputAdapter / Evaluator
  -> Runtime artifact writer
```

其中：

- checkpoint loading、legacy `VisualMLPRouter` import、state_dict key 适配、DataParallel
  key 处理属于 Runtime / entrypoint 或单独 checkpoint audit。
- scaler state loading 和 transform 边界应在 adapter 前显式处理。
- P16a adapter 不读取 checkpoint、不处理 scaler、不启动 ViT、不访问 `/data2`。
- P16b 也不声明正式 Visual Router 已迁移完成。

## 9. 后续推荐拆分

后续建议保持 small-first，不锁死编号：

1. real Visual feature provider minimal smoke plan：
   先用 tiny fixture、fake encoder 或 precomputed embedding 验证 `FeatureBatch` 边界，
   不接真实 ViT 或正式入口。
2. scaler boundary design/smoke：
   单独验证 loaded scaler transform -> `float32` head-ready `FeatureBatch`，并证明不会
   silent fit。
3. legacy `VisualMLPRouter` checkpoint/signature audit：
   单独审计正式 legacy MLP 的 import、constructor signature、state_dict、DataParallel key
   和 device 处理。
4. online ViT provider audit/smoke：
   单独处理 pseudo image + frozen ViT + batching + dtype/device，不读 router checkpoint。
5. Visual full-scale entrypoint migration plan：
   在 feature/head/runtime 都清楚后，再设计正式 streaming entrypoint 的分步迁移。

## 10. P16b 明确不做

- 不新增 real `VisualFeatureProvider` 代码。
- 不新增 ViT provider 代码。
- 不修改 `time_router/features/visual_mock.py`。
- 不修改 `time_router/models/visual_mlp_adapter.py`。
- 不修改 `scripts/run_stage1_visual_small.py`。
- 不修改正式训练 / evaluation 入口。
- 不读取真实 checkpoint。
- 不启动 ViT embedding。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不新增 Bash launcher。
- 不把 Bash 引入 `time_router`。
- 不把 `run_dir` 传入 provider。
- 不把 cache 设计成 interface。
- 不为兼容旧版 `96_48_S` full-scale 输出 schema 写适配逻辑。
- 不声称正式 Visual Router 已迁移完成。

## 11. P16b 验收

P16b 是 docs-only 审计。轻量验收命令：

```bash
git diff --name-only
rg -n "FeatureBatch|scaler|ViT|pseudo|run_dir|cache|P16a|LoadedTorchMLPRouterHeadAdapter" docs/refactor/stage1_real_visual_feature_provider_audit.md
```

由于本步不改 runtime 代码、不新增 provider、不迁移入口，不需要运行 Python smoke。实验日志应记录
docs-only 口径和未改变 runtime 行为。
