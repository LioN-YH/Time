# Stage 1 Provider Adapter Boundary Review

创建日期：2026-06-19

## 1. 目标

本文记录 P5d 阶段对现有 Stage 1 代码的 provider/head/evaluator adapter 边界审查。审查基于 P5b provider interface design 和 P5c protocol types skeleton，目标是判断哪些现有模块未来适合包装为 canonical adapter，哪些历史路线不应继续适配。

本阶段只写文档，不实现 provider adapter，不新增 ExpertProvider / FeatureProvider 读取代码，不修改 protocol types，不修改训练脚本，不迁移 Visual Router / TimeFuse-style fusor 入口。

## 2. 总体判断

未来 adapter 应围绕 P5c 的轻量对象组织：

```text
SplitSpec
  -> ExpertBatch
  -> FeatureBatch
  -> RouterOutput
  -> EvaluationInput
```

adapter 的职责是把现有实现包装成这些 contract object，而不是把现有训练脚本整体改名为 provider。provider adapter 不决定 `run_dir`，不写 `status.json` / `metadata.json`，不直接绑定 `/data2`，也不替代 launcher/runtime。

最小实现顺序建议：

1. 先写 entrypoint migration plan，明确 runtime 如何把 split、provider specs、head specs、evaluator specs 和输出目录编排起来。
2. 第一批代码实现优先选择 `PredictionCacheExpertProvider`，基于已有 `PredictionBatchReader` 输出 `ExpertBatch`。
3. 第二批可选择 `TimeFuseFeatureCacheProvider`，包装 17 维 feature cache 的 shard-aware 读取，输出 `FeatureBatch`。
4. Visual online ViT feature provider 应晚于上述两步，因为它牵涉 GPU encoder、Quito 历史窗口读取、batch latency、dtype、DataParallel 和 future finetune/joint 训练边界，不能作为第一个最小 adapter。

## 3. ExpertProvider Adapter 候选

### 3.1 `PredictionBatchReader`

候选等级：第一批最小 adapter 候选。

当前位置：

- `time_router/io/prediction_cache_reader.py`

适配方式：

- 新 adapter 可命名为 `PredictionCacheExpertProvider`。
- 输入应来自 runtime 显式传入的 `manifest_path`、`sample_keys`、`model_columns` 和 `verify_metrics` 等 spec。
- 输出应转换为 P5c `ExpertBatch`：
  - `sample_keys=tuple(batch.sample_keys)`
  - `model_columns=tuple(batch.metadata["model_columns"])` 或 runtime 明确传入顺序
  - `y_pred=batch.y_pred`
  - `y_true=batch.y_true`
  - `row_index_metadata=batch.metadata`
  - `extra` 保存 `array_storage`、reader version 或 manifest lineage 等轻量 metadata

保留边界：

- `PredictionBatchReader` 已支持 `packed_npy_v1` 与 legacy `per_sample_npy`，并按 `model_columns` 重排专家维度。
- 该 reader 会校验共享 `y_true`、manifest 指标和非有限值；这些是 ExpertProvider adapter 可以复用的读取层安全检查。
- full-scale 场景必须由 runtime 或 SplitStrategy 显式传入当前 batch/shard 的 sample_key，不能让 adapter 在 23M sample 上默认全量读取。

不应改变：

- 不修改 `PredictionBatchReader` 行为。
- 不把 adapter 写进 `time_router/io`；provider adapter 更适合未来 `time_router/providers/`、`time_router/features/` 或 `time_router/training/` 边界中单独承载。
- 不让 ExpertProvider 读取 oracle/TSF、router feature 或 loss。

### 3.2 `prediction_array_io` grouped loading

候选等级：底层数组读取能力，适合被 ExpertProvider 间接复用，不适合作为 provider 本体。

当前位置：

- `visual_router_experiments/common/prediction_array_io.py`

适配判断：

- `load_prediction_arrays_grouped(...)` 对 `packed_npy_v1` 按路径分组复用 mmap，是正式 full-scale 读取必须保留的性能边界。
- `load_prediction_array(...)` 仍保留 `per_sample_npy` 兼容能力，但不能把 per-sample 小文件路线重新设为正式默认。
- 未来应逐步把该 array IO 能力下沉到 `time_router/io` 或等价共享位置，但 P5d 不移动文件。

不应适配为 provider 的原因：

- 它只知道 manifest record 与数组路径，不知道 sample batch、专家顺序、共享 `y_true` 或 protocol metadata。
- 它不应决定 `model_columns`，不应写 `ExpertBatch`，也不应处理 split。

### 3.3 不应适配的旧 Expert 路线

以下路线不应包装为 canonical ExpertProvider：

- 全量 Python manifest lookup：旧 OOM 路线把数千万记录留在 Python 内存中，不符合 full-scale streaming/shard-aware 约束。
- 每 sample 重复 `np.load` packed shard：会把 `packed_npy_v1` 的少文件优势抵消，不应成为正式 reader 默认策略。
- pilot-only per-sample `.npy` 读取路线：可保留 smoke/历史复现价值，但不作为 full-scale 默认 provider。
- 直接读取 legacy CSV 输出再反推 `y_pred/y_true` 的路线：会让旧 output schema 反向污染 ExpertProvider。
- 依赖 `/data2` 固定路径的 reader：provider spec 可以引用 path，但 provider interface 不硬编码输出根。

## 4. FeatureProvider Adapter 候选

### 4.1 Visual pseudo image / ViT feature 路径

候选等级：中期 adapter 候选，不建议作为第一批最小实现。

当前相关位置：

- `visual_router_experiments/common/pseudo_imageization.py`
- `visual_router_experiments/common/vit_embedding_utils.py`
- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`

可分离部分：

- `normalize_window(...)`
- `imageize_3view(...)`
- `imageize_top3fold(...)`
- `encoder_normalize(...)`
- `make_pseudo_images(...)`
- `pool_vit_outputs(...)`
- `iter_online_embedding_batches(...)` 中“历史窗口 -> pseudo image -> ViT embedding”的核心流程

未来 adapter 形态：

- 可命名为 `VisualOnlineVitFeatureProvider`。
- 输入应由 runtime 传入 Quito 历史窗口读取句柄、sample batch、imageization spec、encoder spec、dtype/device spec。
- 输出 P5c `FeatureBatch`：
  - `sample_keys` 与 ExpertBatch 保序对齐；
  - `features` 为当前 batch embedding；
  - `feature_schema` 记录 encoder name、pooling、normalization preset、image size、period selection、embedding dim、online/batch-runtime-only。

必须保留的边界：

- full-scale 主线不长期保存 pseudo image tensor 或 ViT embedding `.npy`。
- adapter 不应写 `online_embedding_manifest.csv`、latency CSV、status 或 summary；这些属于 evaluator/runtime/reporting。
- adapter 不应读取 `y_true`、expert error 或 oracle label 作为可部署 test-time feature。
- future finetune ViT / joint ViT 训练应把 encoder 参数、optimizer、checkpoint 和 train/eval mode 放在 training/runtime 层，FeatureProvider 只暴露 feature 前向和 schema。

不建议第一批实现的原因：

- 当前路径牵涉 Hugging Face 模型加载、网络重试、本地 cache、dtype、CUDA/DataParallel、Quito 历史窗口 streaming 和 latency 记录。
- 如果先抽 Visual provider，容易把 runtime、logging、checkpoint 和 provider 混在一个 adapter 里。

### 4.2 TimeFuse 17 维 feature cache reader

候选等级：第二批最小 adapter 候选。

当前相关位置：

- `visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py`
- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`

可分离部分：

- `infer_feature_columns(...)`
- `collect_feature_sample_keys(...)`
- `Stage1TimeFuseFusorStreamingReader._iter_feature_frames(...)` 中按 feature CSV batch streaming 和 split 下推的部分
- `fit_scaler_streaming(...)` 中 feature-only scaler 的数据访问口径

未来 adapter 形态：

- 可命名为 `TimeFuseFeatureCacheProvider`。
- 输入应由 runtime 传入 feature shard path、feature columns、split filter、batch size 和 scaler/normalizer state。
- 输出 P5c `FeatureBatch`：
  - `sample_keys` 与 feature CSV 当前 batch 顺序一致；
  - `features` 为 `[B, 17]` 或未来 schema 定义的 feature tensor；
  - `feature_schema` 记录 `feature_schema_name=timefuse_single_variable_meta_v1`、feature columns、feature dim、source、split_filter、scaler state 引用。

必须拆开的内容：

- `Stage1TimeFuseFusorStreamingReader` 当前同时补齐 feature、oracle label、prediction tensors 和 expert errors；未来 FeatureProvider 只负责 feature，不应顺手读取 prediction arrays 或 oracle labels。
- scaler fit 是训练 split 上的 preprocessing/training runtime 行为，不应塞进纯 FeatureProvider 的 `__iter__` 副作用里；adapter 可暴露 transform 所需 state，fit 由训练流程调用。
- shard-local SQLite oracle/prediction index 属于 ExpertProvider / supervision / runtime 准备步骤，不属于 FeatureProvider。

### 4.3 Future online TimeFuse feature computation

候选等级：未来扩展点。

接口点：

- 当前 17 维 feature cache builder 证明 feature 可以从历史窗口 `x` 派生；未来可把“sample_key -> 历史窗口 -> TimeFuse-derived features”改为 online computation。
- canonical FeatureProvider contract 应允许 `feature_source=online_timefuse_computation`，而不是固定 `feature_cache.csv`。

约束：

- 只使用历史窗口和稳定 metadata，不读取未来 `y`、expert prediction、oracle top-1 或 test-time expert error。
- online computation 的 cache、latency、checkpoint 仍由 runtime/reporting 记录，不由 provider 决定 `run_dir`。

### 4.4 Offline ViT embedding cache

候选等级：reference-only，不作为 full-scale canonical adapter。

判断：

- 历史离线 ViT embedding cache 可保留小规模 debug、复现和 offline/online 对照价值。
- full-scale 主线已经固定为 batch 运行时在线生成伪图像与 ViT embedding，不长期落盘 embedding `.npy`。
- 不应为了兼容旧 cache 设计 `OfflineVitEmbeddingFeatureProvider` 作为第一批正式 provider；若未来需要 debug adapter，必须标注 `reference_only` 或 `debug_only`。

## 5. RouterHead Adapter 候选

### 5.1 Visual Router Head

候选等级：中期 adapter 候选。

当前相关位置：

- `train_visual_router_online_streaming.py`

可分离部分：

- MLP router module 本体；
- `features -> logits -> softmax weights` 的 inference 路径；
- router metadata：input dim、model columns、router mode、hidden dim、dropout、loss mode。

不属于 RouterHead 的内容：

- `CrossEntropyLoss`、`SmoothL1Loss`、KL soft oracle loss 和 loss 权重；
- scaler fit、optimizer、epoch loop、checkpoint/resume；
- prediction cache 读取、expert error 计算；
- status/metadata/log/CSV 写出；
- eval-only 输出 schema 和 comparison。

边界判断：

- RouterHead adapter 可以输出 P5c `RouterOutput(logits=..., weights=...)`。
- loss 与训练 loop 属于 training branch-specific 逻辑；Evaluator 仍只从显式 weights/logits 和 `ExpertBatch` 复算指标。

### 5.2 TimeFuse Linear-Softmax Fusor

候选等级：第一批或第二批 head adapter 候选，但应晚于 provider 读取边界稳定。

当前相关位置：

- `TimeFuseFusor` in `stage1_timefuse_fusor_streaming_reader.py`
- `train_timefuse_fusor_streaming.py`

可分离部分：

- `nn.Linear(input_dim, expert_count)`；
- `softmax(logits)`；
- `features -> weights`；
- `broadcast_weights(weights, y_pred)` 可作为 training/evaluation utility，而不是 head 本体。

不属于 RouterHead 的内容：

- `SmoothL1Loss(beta=0.01)`；
- weighted fusion loss 训练步骤；
- DataParallel 包裹策略；
- optimizer、checkpoint、scaler state；
- shard preparation、SQLite index、CSV/summary 写出。

边界判断：

- `TimeFuseLinearSoftmaxHead` 是最小 head adapter 的自然形态。
- 它不应读取 feature cache，也不应读取 prediction cache；它只消费 `FeatureBatch.features` 并输出 `RouterOutput`。

## 6. Evaluator Adapter 候选

### 6.1 `time_router.evaluation` public API

候选等级：Evaluator adapter 的基础能力。

当前位置：

- `time_router/evaluation/__init__.py`
- `time_router/evaluation/metrics.py`
- `time_router/evaluation/summary.py`
- `time_router/evaluation/prediction_rows.py`

适配方式：

- 未来可新增 `FusionEvaluator` 或等价 adapter。
- 输入为 P5c `EvaluationInput`：
  - `sample_keys`
  - `model_columns`
  - `y_pred`
  - `y_true`
  - `weights` 或可由 `logits` 显式转换得到的 weights
- 输出包括 in-memory summary dict、per-sample rows 和 calibration-ready object；文件写出由 runtime/report layer 完成。

当前可复用能力：

- `hard_top1_fusion(...)`
- `raw_soft_fusion(...)`
- `compute_mae(...)`
- `compute_mse(...)`
- `compute_selected_counts(...)`
- `compute_weight_entropy(...)`
- `compute_max_weight(...)`
- `build_fusion_summary(...)`
- `build_per_sample_fusion_rows(...)`

保留边界：

- `summary.py` 不是正式 summary output schema。
- `prediction_rows.py` 不写 CSV/JSON/Parquet。
- `metrics.py` 不读取 manifest、oracle/TSF 或训练输出目录。
- Evaluator 不应读取 legacy metadata 来猜专家顺序，必须使用显式 `model_columns`。

### 6.2 Legacy CSV schema

不应反向污染 Evaluator。

判断：

- `visual_router_predictions.csv`、`timefuse_fusor_predictions.csv`、summary CSV、selected counts CSV 和 comparison CSV 是现有入口的报告产物，不是 canonical Evaluator 的输入契约。
- 未来 Evaluator adapter 应先生成内存对象，再由 runtime/reporting 层按新的 schema 或兼容 schema写出。
- 如果为历史输出提供读取器，应标注为 report parser 或 legacy comparison helper，不应伪装成 Evaluator。

## 7. Runtime 边界

provider/head/evaluator adapter 与 runtime 的关系是“被编排”，不是“决定运行”。

明确边界：

- adapter 不决定 `run_dir`。
- adapter 不写 `status.json`。
- adapter 不写 `metadata.json`。
- adapter 不创建 checkpoint index。
- adapter 不启动后台进程、不管理 PID/PGID、不写 launcher 脚本。
- adapter 不硬编码 `/data2`；路径可由 spec 显式传入。
- adapter 不负责 full-scale 输出目录命名、覆盖策略或完成态判断。

未来 runtime 应负责：

- 解析 CLI/config，生成 `ExperimentProtocolSpec`。
- 显式选择 `run_dir`。
- 创建 status/metadata/checkpoint/log/evaluation/prediction output。
- 编排 split、provider、head、training loop 和 evaluator。
- 记录 provider specs、feature schema、expert source、head schema 和 evaluator schema。

## 8. 第一批实现建议

结论：先做 entrypoint migration plan，再实现最小 `PredictionCacheExpertProvider`。

原因：

- `PredictionBatchReader` 已经是共享 package 内稳定 public API，且有 golden smoke 覆盖。
- 它的输出天然对应 P5c `ExpertBatch`，adapter 只需做轻量包装和 metadata 显式化。
- ExpertProvider 是 Visual 与 TimeFuse 两条路线共同依赖的最小主干；先抽它可以减少后续 FeatureProvider 与 Evaluator 接入的重复。
- TimeFuse feature cache provider 也适合早期实现，但它容易牵连 scaler、oracle/prediction SQLite 和 streaming reader 拆分，应在 migration plan 中先划清“feature-only provider”和“训练 reader”的区别。
- Visual online ViT provider 不宜第一批实现，因为它的运行时复杂度最高，容易越界到 runtime、logging、checkpoint 或 GPU 资源策略。

建议下一步小步：

1. P5e：entrypoint migration plan only，设计 Visual / TimeFuse 两个正式入口如何逐步消费 P5c protocol objects 和 adapter specs。
2. P5f：minimal `PredictionCacheExpertProvider` skeleton + smoke，只包装小规模 fixture，不接 full-scale，不改训练入口。
3. P5g：TimeFuse feature cache provider boundary/skeleton，先做到 feature-only `FeatureBatch`，不读取 oracle/prediction arrays。

## 9. 明确不做

- 不实现 provider adapter。
- 不新增 ExpertProvider / FeatureProvider 读取代码。
- 不修改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / io helper。
- 不修改 protocol types。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 runtime / run_dir helper。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。
