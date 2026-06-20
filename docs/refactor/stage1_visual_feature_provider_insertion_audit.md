# Stage 1 P14a Visual FeatureProvider Insertion Audit

创建日期：2026-06-20

## 1. 目标

P14a 只审计 Visual Router 正式入口中可迁移为 Visual `FeatureProvider` 的最小边界。
本阶段不抽 provider，不改正式入口，不新增代码，不访问 `/data2`，不启动训练、pressure
或 full-scale。

审计对象是：

- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- Visual labels / sample metadata path
- SQLite prediction index / prediction arrays path
- history window construction
- pseudo image construction
- frozen ViT forward path
- router input feature shape
- Visual MLP router head 使用方式
- evaluation/training `ExpertBatch` bypass 既有结论
- checkpoint/resume、device/dtype、DataParallel、latency 相关 runtime 逻辑

核心结论：Visual `FeatureProvider` 的最小未来输出应是 `FeatureBatch`。它负责按
ordered `sample_keys` 提供可部署视觉特征；它不拥有 prediction cache、oracle/error、
loss、checkpoint、run_dir、status、metadata 或正式 CSV/summary 写出。

## 2. 当前正式入口中的视觉特征链路

当前 streaming Visual Router 的主线仍集中在
`train_visual_router_online_streaming.py` 中：

```text
labels CSV
  -> filter_stream_shard / limit_samples_per_split
  -> windows_from_labels
  -> iter_online_embedding_batches
     -> load_datasets
     -> item_dataset.data[channel_id, window_index:window_index + seq_len]
     -> make_pseudo_images
     -> frozen ViT forward
     -> pool_vit_outputs
  -> StandardScaler.partial_fit / transform
  -> VisualMLPRouter
  -> prediction/evaluation/runtime outputs
```

当前关键函数边界如下：

| 位置 | 当前职责 | P14a 判断 |
| --- | --- | --- |
| `load_labels(...)`、`filter_stream_shard(...)`、`limit_samples_per_split(...)` | 从 labels CSV 读取 split、sample metadata、oracle/error 字段，并做 shard / dry-run 限制 | 长期应由 `SampleManifest + SplitStrategy` 与 `SupervisionProvider` 分层；Visual FeatureProvider 只消费 ordered sample metadata，不读取 oracle/error |
| `windows_from_labels(...)` | 从 labels 行提取唯一 `sample_key + config/split/dataset/item/channel/window` 窗口清单 | 可作为 provider 输入准备的候选边界，但不应把 labels CSV 本身作为 provider 的长期 source |
| `iter_online_embedding_batches(...)` | 读取 Quito history window、构造 pseudo image、执行 ViT forward、返回 batch manifest、embedding 和 latency | 是 Visual FeatureProvider 的主要插入点；其中 resource policy / latency 是否放入 provider 需要继续拆分 |
| `make_pseudo_images(...)` | 把 history window `x` 转为视觉 encoder 输入 tensor | 属于 Visual feature extraction 逻辑，可进入 provider 或 provider 内部 helper |
| `build_vit_model(...)` / `load_vit_model_with_retry(...)` | 加载 frozen ViT、处理 dtype、DataParallel、本地/远端 Hugging Face cache 与 retry | 更偏 Runtime / encoder factory；不应由 pure FeatureProvider 私自决定全局 device、cache 策略或重试策略 |
| `StandardScaler.partial_fit/transform` | 对 ViT embedding 做训练期标准化 | 属于 training/runtime state；不属于 pure FeatureProvider 输出本身 |
| `train_on_stream_batch(...)` / `predict_stream_batch(...)` | 消费 scaled embedding，计算 VisualMLPRouter logits/weights、loss 或 prediction rows | 属于 RouterHead / training / evaluation，不属于 FeatureProvider |
| `SQLitePredictionIndex` 与 prediction arrays 读取 | 为 loss、raw soft fusion、ExpertBatch bypass 提供专家预测和共享 `y_true` | 属于 ExpertProvider / backend prepare / branch-specific loss，不属于 Visual FeatureProvider |
| checkpoint/status/metadata/CSV/latency 文件写出 | 管理长任务运行产物和可恢复性 | 属于 Runtime / artifact writer，不属于 provider |

## 3. Visual FeatureProvider 未来应输出什么

未来 Visual `FeatureProvider` 应输出 canonical `FeatureBatch`：

- `sample_keys`：保持调用方 manifest ordered `sample_keys`，不排序、不去重后重排。
- `features`：router/head 消费的视觉特征或其轻量表示。当前最小口径是 ViT pooled
  embedding，shape 类似 `[batch, embedding_dim]`，dtype 可统一落到 `float32`。
- `feature_schema`：记录 visual feature schema name、feature dim、history window 来源、
  pseudo-image variant、encoder name、pooling、normalization、image size、period selection
  等足以复核输入口径的轻量 schema。
- `extra`：只放轻量 lineage，例如 batch split、dataset/item/channel/window 范围、provider
  version、feature storage policy。不得放 `run_dir`、checkpoint、status、正式输出路径或
  大型数组路径。

推荐 schema 语义：

```text
feature_schema = {
  "name": "visual_online_vit_pooled_embedding_v1",
  "feature_dim": 768,
  "history_source": "quito_window_x",
  "pseudo_image": {
    "variant": "...",
    "norm_mode": "...",
    "pixel_mode": "...",
    "image_size": 224,
    "normalization_preset": "..."
  },
  "encoder": {
    "name": "google/vit-base-patch16-224",
    "pooling": "cls",
    "frozen": true
  },
  "storage": "batch_runtime_only_not_saved"
}
```

`FeatureBatch.features` 不应包含 oracle label、oracle value、per-model error、未来 `y`、
专家预测或 prediction cache 派生误差。

## 4. 哪些属于 Visual FeatureProvider

P14a 建议把以下逻辑作为未来 Visual `FeatureProvider` 或其内部 helper 的候选：

1. 根据 ordered `sample_keys` 和 sample metadata 定位 history window：
   - `config_name`
   - `split`
   - `dataset_name`
   - `item_id`
   - `channel_id`
   - `window_index`
   - `seq_len`
2. 从 Quito dataset 只读取历史窗口 `x`，保持“不访问未来 `y`”的输入边界。
3. 构造 pseudo image 或等价 router input：
   - `variant`
   - `norm_mode`
   - `pixel_mode`
   - `clip`
   - `image_size`
   - `normalization_preset`
   - `period_selection`
   - `period_candidate_values`
4. 执行 frozen ViT feature extraction，如果后续决定把 encoder forward 放入 provider。
5. 对 batch 输出视觉特征，保持 `sample_keys` 与输入顺序一致。

其中第 4 点需要谨慎：ViT forward 牵涉 GPU、dtype、DataParallel、Hugging Face cache、网络
retry、latency 和未来 finetune/joint training。若实现时把 encoder forward 放进 provider，
provider 也应由 Runtime 显式注入 device、dtype、encoder 实例或 encoder factory，而不是自己
决定全局运行策略。

## 5. 哪些不属于 Visual FeatureProvider

以下逻辑必须排除在 Visual `FeatureProvider` 之外：

- oracle label、oracle value、per-model error、oracle upper bound。
- prediction cache、SQLite prediction index、packed npy/per-sample npy path。
- `ExpertBatch` 构造、专家误差复算、`fusion_huber_kl` soft oracle。
- loss、optimizer、backprop、scheduler、class weight、KL/huber objective。
- `StandardScaler` 的 fit 状态、optimizer state、router checkpoint/resume。
- `run_dir`、output root、metadata、status、logs、checkpoint index。
- full evaluation summary、comparison、prediction rows、CSV/Markdown 写出。
- Bash launcher、`exp_scripts`、`/data2` 路径策略。
- Visual RouterHead 权重计算本身；除非短期 legacy coupling 只为了迁移期间适配。

`predict_stream_batch(...)` 当前会把 `oracle_model`、`oracle_value`、`regret_to_oracle` 写入
prediction rows。这属于 legacy evaluation/report 输出，不得反向进入 future provider。

## 6. Device / Dtype / Runtime 边界

Visual feature path 的特殊风险是它不像 TimeFuse CSV provider 那样纯 I/O：

- ViT forward 需要 device 与 dtype 决策。
- `fp16` 在 CPU 上不成立，当前 `resolve_dtype(...)` 会按 device 调整。
- `--vit-data-parallel` 会包裹 frozen ViT，但 router/scaler/checkpoint 仍保持单进程语义。
- `ViTModel.from_pretrained(...)` 可能触发 Hugging Face cache 或远端下载；当前入口已有
  `local_files_only` 与临时错误 retry。
- latency 统计当前在 `iter_online_embedding_batches(...)` 中产生，并追加到
  `online_embedding_latency_summary.csv`。
- checkpoint/resume 的 signature 当前包含 embedding metadata、dtype 参数、路径和 shard
  信息，防止不同 feature 口径误接。

因此 P14a 建议后续拆分为：

- Runtime 显式解析 device、dtype、DataParallel、Hugging Face cache、本地/远端策略和 retry。
- Runtime 或 encoder factory 构建 frozen encoder，并把 encoder 句柄注入 provider。
- Provider 只按输入 batch 生成 `FeatureBatch`，可以返回轻量 latency/lineage，但不写文件。
- Artifact writer / Runtime 决定是否把 latency 汇总写入 run artifact。
- Checkpoint/resume 继续由 Runtime 管理；provider 不保存 checkpoint，也不更新
  `latest_checkpoint_path`。

本阶段只做审计，不抽实现。

## 7. 与 ExpertBatch 的关系

Visual `FeatureProvider` 不读取 prediction cache。专家预测仍通过 `ExpertProvider` /
`ExpertBatch` 进入下游：

```text
SampleManifest ordered sample_keys
  -> Visual FeatureProvider -> FeatureBatch(sample_keys, features)
  -> ExpertProvider / backend -> ExpertBatch(sample_keys, model_columns, y_pred, y_true)
  -> RouterHead / training / evaluation
```

`FeatureBatch` 与 `ExpertBatch` 的唯一连接点应是 ordered `sample_keys` 对齐：

- 两者都应保持调用方传入的 sample order。
- 任何 join / 对齐失败都应在 protocol chain 或 runtime 层显式报错。
- `FeatureBatch.extra` 不应保存 prediction cache path。
- `ExpertBatch.extra` 不应保存 pseudo image tensor、ViT embedding 或 feature values。

当前 P9d/P9f 的 Visual evaluation/training `ExpertBatch` bypass 已证明可以把 legacy
SQLite batch arrays 包装为 `ExpertBatch` 做旁路校验，但它不代表 Visual FeatureProvider 已抽取，
也不代表 `PredictionCacheExpertProvider` 已接入正式入口。

## 8. 最小插入边界建议

若后续 P14b/P14c 开始实现，建议先做 smoke-only，不迁移正式入口：

1. **P14b：Visual FeatureProvider minimal mock/fixture smoke**
   - 已完成，见 `docs/refactor/stage1_visual_feature_provider_mock_smoke.md`。
   - 使用 P13b real-derived manifest 的 4 个 ordered sample_keys。
   - 使用 `tests/fixtures/stage1_visual_feature_mock/history_windows.json` 作为 tiny in-memory
     history window source。
   - 不加载真实 Hugging Face ViT，用 deterministic encoder stub 输出 `[sample, 8]`。
   - 验证 `FeatureBatch.sample_keys` 保序、`features` shape、schema/extra、且 provider
     阶段不读取 oracle / prediction / run_dir。
2. **P14c：Visual eval-only canonical bypass plan**
   - 只规划或 smoke 验证 eval-only 如何把 legacy Visual batch arrays 包装为
     `ExpertBatch`，再与 Visual `FeatureBatch` 和 Visual head/evaluator 对齐。
   - 不改正式 CSV/summary/metadata/status/checkpoint schema。
   - 不替换 `SQLitePredictionIndex`，不接 full-scale prepared backend。

P14b/P14c 的共同约束：

- 先 small smoke，再考虑 branch-specific small entrypoint。
- 不直接迁移 `train_visual_router_online_streaming.py`。
- 不抽正式 ViT provider。
- 不访问 `/data2`。
- 不启动 full-scale。

## 9. P14a 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增 VisualFeatureProvider 代码。
- 不抽 ViT provider。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual RouterHead adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 10. P14a 验收

P14a 是纯文档审计。验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

验收口径：

- 新增本审计文档。
- 明确 Visual `FeatureProvider` 的输入、输出和不做范围。
- 明确 history window / pseudo image / ViT feature 的边界。
- 明确 device/dtype/runtime 与 provider 的关系。
- 明确 `FeatureBatch` 与 `ExpertBatch` 只通过 ordered `sample_keys` 对齐。
- roadmap / migration plan 同步 P14a 状态和 P14b/P14c 连接。
