# Stage 1 P9a Visual Router Entrypoint Adapter Insertion Audit

创建日期：2026-06-20

## 1. 目标

本文记录 P8d TimeFuse baseline parity review 之后，对正式入口 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 的最小 adapter 接入点审计。

本步只做文档化接入计划，不修改正式 Visual Router 训练入口行为，不迁移入口，不改变 Visual Router feature、ViT、router head、loss、evaluation output schema、checkpoint、status 或 metadata。

## 2. 当前正式入口承担的职责

`train_visual_router_online_streaming.py` 当前仍是 Visual Router full-scale streaming 主线。它在一个脚本内同时承担以下职责。

### 2.1 sample / manifest / prediction 读取

- 解析 `labels_path`、`prediction_manifest_path`、`config_path` 和 streaming shard 参数。
- 读取 oracle labels CSV，并按 `stream_shard_index / stream_shard_count` 与 `max_samples_per_split` 限制本次样本。
- 从 labels 构造唯一 `windows_df`，并校验 `sample_key` 与 `config/split/dataset/item/channel/window` 元信息一致。
- `required_prediction_sample_keys(...)` 根据训练模式推导需要索引的 vali/test sample_key。
- `SQLitePredictionIndex` 与 `build_lightweight_prediction_index(...)` 为所需 sample_key 建立 SQLite 磁盘索引。
- `load_prediction_tensors_from_lightweight_index(...)` 在 batch 级查询五专家 `y_pred`、共享 `y_true` 和专家误差。
- eval 阶段通过 `prediction_index.fetch_records(...)` 为 raw soft fusion 读取当前 batch prediction record。

### 2.2 Quito 历史窗口与在线伪图像特征

- `load_data_config(...)` 读取 Quito evaluate config，并从 Quito dataset 读取历史窗口 `x`。
- `windows_from_labels(...)`、`build_required_index(...)`、`load_datasets(...)` 和 `select_user_data(...)` 共同定位历史窗口。
- `iter_online_embedding_batches(...)` 负责按 split/dataset/item/channel/window 流式读取历史窗口。
- `make_pseudo_images(...)` 在线构造 pseudo image tensor，使用 `variant`、`norm_mode`、`pixel_mode`、`clip`、`image_size` 和 `period_selection` 等视觉参数。
- `ViTModel` 前向、pooling、dtype、device、`local_files_only`、`vit_data_parallel` 和 latency 统计都在入口内完成。
- ViT embedding 与 pseudo image tensor 只作为 batch runtime 对象存在，不落盘 `.npy` 或长期 tensor cache。

### 2.3 ViT / router head / loss / optimizer

- `load_vit_model_with_retry(...)` 构建冻结 ViT，并对 429、503、timeout 等临时错误做有限重试。
- `StandardScaler.partial_fit` 只遍历 vali embedding；test 只 transform。
- `VisualMLPRouter` 在入口中实例化，并由 `AdamW` 优化。
- `train_on_stream_batch(...)` 支持 `classification` 与 `fusion_huber_kl` 两种训练目标。
- `fusion_huber_kl` 在 batch 内读取五专家 prediction，计算 weighted fusion `SmoothL1Loss` 与 KL soft oracle 辅助损失。
- `predict_stream_batch(...)` 对 test embedding 做 router 前向、softmax、hard top-1、weight entropy 和 max weight 计算。

### 2.4 evaluation / summary / rows

- `predict_stream_batch(...)` 生成正式 hard top-1 `visual_router_predictions.csv` 行。
- `add_soft_fusion_metrics(...)` 在正式 rows 上追加 raw soft fusion 指标。
- `summarize_csv_outputs(...)` 从正式 CSV 复算 `visual_router_summary.csv`、`visual_router_soft_fusion_summary.csv`、`visual_router_selected_model_counts.csv` 和 `visual_router_comparison.csv`。
- `write_summary_md(...)` 写出 `visual_router_streaming_summary.md`。
- 当前 evaluation 输出 schema 仍包含 Visual Router 历史字段，例如 `router_name`、`selected_value`、`oracle_model`、`oracle_value`、`regret_to_oracle`、`oracle_label_correct`、`weight_*`、`weight_entropy`、`normalized_weight_entropy` 和 `max_weight`。

### 2.5 checkpoint / status / metadata / run_dir

- CLI 负责 output root / explicit output dir 选择，默认生成 streaming run 目录。
- `write_status(...)` 记录 init、checkpoint loaded、scaler fit、training、checkpoint saved、train-only done 和 done。
- `save_checkpoint(...)` 写出 router、optimizer、scaler、resume signature、epoch summaries 和 latest checkpoint index。
- `cleanup_output_files(...)` 在 fresh/resume/train-only 场景下清理或保留正式输出。
- `visual_router_metadata.json` 与 `visual_router_online_metadata.json` 记录输入路径、模型参数、embedding metadata、config metadata、checkpoint、run flags 和 input exclusion。

## 3. 第一批最小接入点判断

P9a 的结论是：Visual Router 入口应比 P8 TimeFuse 入口更保守。第一批只建议做 evaluation 阶段旁路校验和 `ExpertBatch` 对齐校验，不建议迁移 feature extraction、ViT provider、router head 或 training loop。

### 3.1 是否优先接 PredictionCacheExpertProvider / ExpertBatch

应优先规划，但不应直接替换正式训练入口的 SQLite prediction index。

原因如下：

- `PredictionCacheExpertProvider` 与 `ExpertBatch` 已能表达五专家 `y_pred`、共享 `y_true`、`model_columns` 和 row index lineage，是 Visual Router 与 TimeFuse 共享的最小专家输出 contract。
- Visual Router 当前正式入口的 prediction 读取还承担 full-scale SQLite 子集建库、batch query、packed row index 单行读取和 `fusion_huber_kl` expert error 计算；这些 runtime 细节不能在 P9b 一步替换。
- 最小接入方式应是对当前 batch 已经通过 SQLite 读取出的 `y_pred/y_true` 构造临时 `ExpertBatch` 或等价 `EvaluationInput`，用于对齐校验。
- 若后续真正替换 prediction 读取，应先保证 `PredictionCacheExpertProvider` 具备 full-scale manifest 子集、SQLite/shard-aware batch query 或等价能力，再比较 sample_key 顺序、五专家顺序、shape、row index、hard/raw-soft 指标和 loss 监督误差。

### 3.2 是否优先接 EvaluationInputAdapter 做旁路 evaluation 校验

应优先接，并且仅作为旁路校验。

最小接入点在 test evaluation batch 内，当前入口已经同时持有：

```text
batch_manifest_df["sample_key"]
MODEL_COLUMNS
y_pred / y_true  或 prediction_index.fetch_records(...) 可读出的当前 batch record
weights = softmax(router(scaler.transform(embeddings)))
```

P9b 可在不改变正式 CSV schema 的前提下，用 `EvaluationInputAdapter.evaluate_input(...)` 旁路复算 hard top-1、raw soft fusion、summary/rows 和 weight diagnostics，再与当前 `predict_stream_batch(...) + add_soft_fusion_metrics(...)` 的结果逐样本对齐。

需要注意的是，adapter rows 只应作为内存校验对象，不能直接写成正式 `visual_router_predictions.csv`，也不能替代 `summarize_csv_outputs(...)` 的历史 output schema。

### 3.3 暂不接 Visual FeatureProvider / ViT provider

不应在第一批接入 Visual FeatureProvider / ViT provider。

原因如下：

- Visual feature path 绑定 Quito dataset 历史窗口读取、pseudo image tensor 构造、ViT/Hugging Face 模型加载、GPU dtype、DataParallel、latency、period candidate、normalization 和 pooling。
- 该路径既影响训练特征，又影响 scaler fit、resume signature、metadata、latency CSV 和 full-scale GPU 资源使用。
- 任何行为微小变化都会影响 router logits、checkpoint、evaluation 输出和后续 full-scale 可比性。
- 相比 TimeFuse 17 维 CSV feature，Visual provider/head 更重，迁移门禁必须单独设计 online embedding 小规模 smoke、GPU/CPU dtype 对照、pseudo image pixel checksum 或等价证据。

## 4. 必须暂留正式入口的逻辑

P9b 之前，以下逻辑必须继续留在 `train_visual_router_online_streaming.py` 或其 runtime/report 层：

- Quito history window 读取、dataset 选择、item/channel/window 定位和 `sample_key` 元信息校验。
- pseudo image / tensor 构造、period candidate、normalization、pixel mapping 和 clipping。
- ViT model 加载、retry、forward、pooling、dtype、device、DataParallel 和 GPU 资源逻辑。
- `StandardScaler.partial_fit`、router optimizer、loss、epoch loop、resume、checkpoint 和 latest checkpoint index。
- `fusion_huber_kl` 的 weighted fusion loss、KL soft oracle auxiliary loss 和 expert error 计算。
- `visual_router_predictions.csv`、soft fusion predictions、summary CSV、comparison CSV、selected counts、Markdown summary、metadata 和 status schema。
- `online_embedding_manifest.csv`、`online_embedding_latency_summary.csv` 和 runtime metadata 中的 embedding/pseudo image 不落盘声明。

这些逻辑共同定义当前 Visual Router 正式行为与 full-scale 可复现口径，不属于 smoke-only adapter 的职责。

## 5. P9b 最小代码迁移建议

P9b 建议只做 evaluation 阶段旁路校验，不改变正式输出 schema。

建议顺序：

1. 在 test prediction batch 中，保留现有 `predict_stream_batch(...)` 生成正式 hard rows。
2. 复用当前 batch 的 sample_key、`MODEL_COLUMNS`、router softmax weights，以及当前 batch 的 `y_pred/y_true` 构造 `EvaluationInput` 或临时 `ExpertBatch + RouterOutput`。
3. 调用 `EvaluationInputAdapter.evaluate_input(...)` 复算 hard top-1、raw soft fusion、per-sample rows、max weight 和 entropy。
4. 与现有输出逐样本比较 `selected_model/selected_index`、hard metric、raw soft metric、`max_weight`、`weight_entropy`；失败信息包含 config、split、sample_key、batch index 和输出目录上下文。
5. 该校验应由显式 flag 控制，默认保持正式 full-scale 行为不变；通过小规模 smoke 后再考虑在 pressure run 中开启。

P9b 不应：

- 改 `visual_router_predictions.csv`、`visual_router_soft_fusion_predictions.csv` 或 summary/comparison 字段。
- 改 `predict_stream_batch(...)` 的正式 row schema。
- 改 `add_soft_fusion_metrics(...)`、`summarize_csv_outputs(...)` 或 `write_summary_md(...)` 的写出口径。
- 迁移 Quito history window、pseudo image、ViT、scaler、optimizer、loss、epoch loop 或 checkpoint。
- 新增 full-scale run、pressure run、Bash/scripts 或访问 `/data2`。

## 6. 与 TimeFuse P8a-P8c 的经验对比

TimeFuse P8a-P8c 可以先在 `evaluate_streaming(...)` 中接 `EvaluationInputAdapter`，因为它的特征是 17 维结构化 CSV，正式 torch head 是单层 linear-softmax，evaluation batch 已直接持有 `batch.y_pred`、`batch.y_true` 和 `weights_np`。

Visual Router 更保守，原因是：

- feature path 是 `x -> pseudo image -> frozen ViT -> embedding`，不是轻量 CSV lookup。
- ViT provider 涉及 GPU、dtype、Hugging Face cache、DataParallel 和 latency 统计。
- router 训练使用 `fusion_huber_kl` 时 prediction tensor 同时进入训练 loss，不只是 evaluation。
- metadata 与 runtime 输出显式承诺 embedding/pseudo image 只在 batch 内存在，不落盘。
- 入口历史输出 schema 已承载 Visual Router 特有字段和 comparison 口径，不能用通用 adapter rows 直接替换。

因此，Visual Router 第一批接入点应只落在 batch evaluation consistency check，而不是 provider/head/training loop 迁移。

## 7. P9a 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不迁移正式入口。
- 不改 Visual Router feature、ViT、router head、loss 或 evaluation output schema。
- 不新增 provider/head 代码。
- 不新增 smoke。
- 不新增 Bash 或 scripts。
- 不访问 `/data2`。
- 不启动 pressure 或 full-scale。
- 不改 TimeFuse 正式入口。
