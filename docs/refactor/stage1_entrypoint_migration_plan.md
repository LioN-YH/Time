# Stage 1 Canonical Entrypoint Migration Plan

创建日期：2026-06-19

## 1. 目标

本文记录 P5e 阶段的 Stage 1 canonical entrypoint 迁移路线。目标是在 P5a runtime contract、P5b provider interface、P5c protocol types 和 P5d provider adapter boundary review 之后，规划 Visual Router 与 TimeFuse-style fusor 两条 canonical entrypoint 如何逐步迁移到统一编排边界。

本阶段只写文档，不改训练代码，不实现 adapter，不迁移入口，不创建 `run_dir`，不接入 `/data2`。

当前两条 canonical entrypoint：

- Visual Router：`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- TimeFuse-style fusor：`visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`

TimeFuse full-scale 后台编排入口：

- `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`

## 2. Visual Router Entrypoint 边界拆分

当前 `train_visual_router_online_streaming.py` 是可运行的 full-scale streaming 主线，但它把 runtime orchestration、ExpertProvider、FeatureProvider、RouterHead、Evaluator 和 run artifacts 写出放在同一个脚本里。迁移时应先保留现入口行为，再逐步把共享能力下沉到 canonical adapter/runtime。

### 2.1 Runtime Orchestration

当前属于 runtime orchestration 的逻辑：

- CLI 参数解析、默认路径、输出目录生成和 `output_dir.mkdir(...)`。
- 随机种子、device/dtype 解析、Hugging Face ViT 加载重试、DataParallel 选择。
- `labels_df` 读取、stream shard 过滤、每 split 样本数限制、config 分组、vali/test 分组。
- checkpoint/resume 签名、checkpoint 加载校验、optimizer/scaler 恢复、epoch 续接。
- scaler fit、训练 epoch loop、eval-only / train-only 分支、status 更新节奏。
- 输出文件清理、metadata 写出、Markdown summary 写出、checkpoint latest 指针维护。

未来应保留在 canonical entrypoint/runtime 的旧逻辑：

- CLI 或 config 解析出的 protocol spec。
- run_dir 显式传入与状态初始化。
- seed/device/resource 选择。
- checkpoint/resume policy。
- epoch/batch 编排和 train-only/eval-only 分支。
- status/metadata/checkpoint/log/evaluation/prediction 输出调度。

未来应下沉的旧逻辑：

- 预测 cache SQLite 索引与 batch query 下沉到 `PredictionCacheExpertProvider` 或其 full-scale runtime 准备步骤。
- 在线伪图像和 ViT embedding 前向下沉到 `VisualOnlineVitFeatureProvider`。
- MLP logits/weights 前向包装为 Visual `RouterHead`。
- hard/raw-soft 指标、selected counts、per-sample rows 和 summary 复算下沉到 `Evaluator`。

### 2.2 ExpertProvider

当前属于 ExpertProvider 的逻辑：

- `required_prediction_sample_keys(...)` 根据 vali/test 和训练模式收集需要的 sample_key。
- `SQLitePredictionIndex`、`build_lightweight_prediction_index(...)`、`load_prediction_lookup_for_sample_keys(...)`。
- `load_prediction_tensors_from_lightweight_index(...)` 对当前 batch 读取五专家 `y_pred`、共享 `y_true` 和 array row index。
- `train_on_stream_batch(...)` 中为 `fusion_huber_kl` 读取当前 batch prediction record 并计算可训练 weighted fusion loss 所需专家预测。
- eval 阶段为 raw soft fusion 读取 `soft_lookup`。

迁移原则：

- 第一批 adapter 应用已有 `PredictionBatchReader` 先实现 `PredictionCacheExpertProvider`，输出 P5c `ExpertBatch`。
- full-scale 场景必须由 runtime/SplitStrategy 显式传入当前 batch 或 shard 的 sample_key，不允许 provider 默认扫描全量 manifest。
- 保留 `packed_npy_v1` row index、固定五专家顺序和共享 `y_true` 校验。

### 2.3 FeatureProvider

当前属于 Visual `FeatureProvider` 的逻辑：

- `windows_from_labels(...)` 从 label rows 构造 Quito 历史窗口索引。
- `load_data_config(...)`、`resolve_period_candidates(...)` 和 Quito dataset 历史窗口读取。
- `iter_online_embedding_batches(...)` 中的历史窗口读取、normalization、pseudo image 生成、ViT 前向、pooling 和 embedding batch 输出。
- scaler fit 读取 vali embedding 流，test 只 transform。
- pseudo image tensor 与 ViT embedding 仅 batch runtime 存在，不落盘 `.npy`。

迁移原则：

- Visual provider 不作为第一批最小 adapter，因为它牵涉 GPU encoder、Quito 数据读取、dtype、Hugging Face cache、DataParallel、latency 和未来 finetune/joint training。
- 未来 `VisualOnlineVitFeatureProvider` 只输出与 ExpertBatch 保序的 `FeatureBatch`，不写 latency CSV、status、metadata 或 summary。
- 继续禁止把未来 `y`、专家误差、oracle label 作为可部署 Visual Router 的 test-time 动态特征。

### 2.4 RouterHead

当前属于 RouterHead 的逻辑：

- `VisualMLPRouter(input_dim, hidden_dim, output_dim, dropout)`。
- scaler transform 后的 `features -> logits -> softmax weights`。
- `predict_stream_batch(...)` 中的 inference 和 hard top-1 选择。

不属于 RouterHead 的逻辑：

- `CrossEntropyLoss`、`SmoothL1Loss`、KL auxiliary loss、optimizer 和 epoch loop。
- prediction cache 读取、expert error 计算、status/metadata/checkpoint/CSV 写出。

迁移原则：

- RouterHead adapter 只输出 P5c `RouterOutput(logits, weights)`，专家维度必须与 `model_columns` 对齐。
- loss 与训练步骤保留在 branch-specific training/runtime 层。

### 2.5 Evaluator

当前属于 Evaluator 的逻辑：

- `predict_stream_batch(...)` 生成 per-sample hard rows 的一部分。
- `add_soft_fusion_metrics(...)` 生成 raw soft fusion per-sample 指标。
- `summarize_csv_outputs(...)`、`write_summary_md(...)`、`summarize_hard_predictions(...)`、`summarize_soft_fusion(...)`、`summarize_selected_model_counts(...)`、`compare_with_baselines(...)`。

迁移原则：

- 先实现 in-memory `FusionEvaluator` 或等价 evaluator adapter，复用 `time_router.evaluation` public API。
- 文件写出仍由 runtime/report 层负责；Evaluator 不决定 `run_dir`，也不依赖历史 CSV schema 反推专家顺序。
- golden smoke 先覆盖 evaluator adapter 的 summary/per-sample rows，再考虑正式入口接入。

## 3. TimeFuse-Style Fusor Entrypoint 边界拆分

当前 `train_timefuse_fusor_streaming.py` 是 1-2 shard 压力测试和可恢复训练/eval 入口；full-scale 正式编排由 `launch_timefuse_fusor_full_scale.py` 生成后台脚本并调用它。迁移时训练入口和 launcher 的边界必须分开处理。

### 3.1 Runtime Orchestration

当前属于 runtime orchestration 的逻辑：

- CLI 参数解析、默认 full-scale 路径、输出目录创建、GPU 约束、资源快照。
- feature shard 发现、split-limited subset 创建、shard-local oracle/prediction SQLite index 准备。
- scaler fit、DataParallel 包裹、train/eval-only 分支、epoch loop、status 更新。
- checkpoint 保存/加载、latest checkpoint index、metadata、summary.md、sample predictions 和 CSV 输出。

未来应保留在 canonical entrypoint/runtime 的旧逻辑：

- full-scale run_dir 显式传入。
- shard 准备和恢复策略。
- scaler fit 调用顺序。
- optimizer/loss/epoch 编排。
- checkpoint/status/metadata/evaluation/prediction 文件写出调度。

未来应下沉的旧逻辑：

- prediction tensors 读取下沉到 `PredictionCacheExpertProvider`。
- feature cache streaming 下沉到 `TimeFuseFeatureCacheProvider`。
- `TimeFuseFusor` 的 `nn.Linear -> softmax` 包装为 `TimeFuseLinearSoftmaxHead`。
- hard/raw-soft 指标、weight diagnostics 和 rows/summary 下沉到 `Evaluator`。

### 3.2 ExpertProvider

当前属于 ExpertProvider 的逻辑：

- `PredictionSQLiteIndex`、`build_prediction_sqlite_index(...)`。
- `discover_prediction_shard_manifests(...)`。
- `load_prediction_tensors_from_index(...)` 和 `_load_array_grouped(...)`。
- `Stage1TimeFuseFusorStreamingReader` 中把当前 feature batch 的 sample_key 转成五专家 `y_pred/y_true`。

迁移原则：

- 第一批仍以 `PredictionCacheExpertProvider` 为共同 adapter，Visual 与 TimeFuse 共享同一个专家预测 contract。
- `Stage1TimeFuseFusorStreamingReader` 当前把 feature、oracle、prediction、expert error 合在一起；未来 ExpertProvider 只负责 `ExpertBatch`，不读取 feature，不决定 scaler。

### 3.3 FeatureProvider

当前属于 TimeFuse `FeatureProvider` 的逻辑：

- `infer_feature_columns(...)`。
- `collect_feature_sample_keys(...)`。
- `Stage1TimeFuseFusorStreamingReader._iter_feature_frames(...)` 的 feature CSV batch streaming 和 split 下推。
- `fit_scaler_streaming(...)` 中对 vali feature 流的 `partial_fit` 数据访问。

迁移原则：

- `TimeFuseFeatureCacheProvider` 适合作为第二批 adapter，晚于 `PredictionCacheExpertProvider` 和 evaluator adapter。
- FeatureProvider 只输出 `[B, 17]` 或 schema 定义的 feature tensor、sample_keys、feature_schema 和必要 provider state。
- scaler fit 是 training/runtime 行为；provider 可以提供 transform 所需数据流，但不应在纯 provider `__iter__` 中隐式完成训练副作用。
- FeatureProvider 不读取 expert prediction、oracle top-1 或未来 `y` 作为可部署动态调权特征。

### 3.4 RouterHead

当前属于 RouterHead 的逻辑：

- `TimeFuseFusor(input_dim, output_dim)`。
- `fusor(batch_x)` 生成五专家 softmax weights。
- `broadcast_weights(...)` 只是 weighted fusion 训练/eval utility，不是 head 本体。

不属于 RouterHead 的逻辑：

- `SmoothL1Loss(beta=0.01)`、optimizer、DataParallel 策略、scaler、checkpoint、SQLite index、CSV/summary 写出。

迁移原则：

- `TimeFuseLinearSoftmaxHead` 是低风险 head adapter，但应在 provider/evaluator contract 稳定后再抽。
- 该 head 只消费 FeatureBatch.features，输出 RouterOutput，不读取 prediction cache 或 feature cache。

### 3.5 Evaluator

当前属于 Evaluator 的逻辑：

- `evaluate_streaming(...)` 中 hard top-1、raw soft fusion、per-sample MAE/MSE、weight entropy、max weight、selected counts、summary CSV 和 sample predictions。
- `array_metrics(...)`、`compute_weight_stats(...)`。
- `write_markdown_summary(...)` 的指标展示部分。

迁移原则：

- Evaluator adapter 应从 `EvaluationInput(sample_keys, model_columns, y_pred, y_true, weights)` 复算结果。
- oracle labels 可作为 regret、上限、baseline 或诊断输入，但不应污染 deployable feature provider。
- 文件名兼容由 runtime/report 层处理；Evaluator 本体只返回结构化 summary、rows 和 calibration-ready object。

## 4. Full-Scale Launcher 与 Runtime Contract 边界

`launch_timefuse_fusor_full_scale.py` 当前是后台编排入口，不是训练 runtime 本体。

属于 launcher 的职责：

- preflight：检查 64 个 feature shard、320 个 prediction manifest、oracle labels、merged cache 状态、磁盘、已有进程和 GPU 策略。
- 生成 `command.sh`、`command_resume.sh`、`launcher.sh`、`stop.sh`、`resume.sh`。
- 后台启动、记录 PID/PGID、launcher log、monitor command、stop/resume command。
- 写 launcher 级 `metadata.json` 和 `status.json`，说明接手方式。

不应属于 launcher 的职责：

- 实现 provider adapter。
- 训练 fusor 或保存 torch checkpoint。
- 决定 `ExpertProvider` / `FeatureProvider` / `RouterHead` 的内部行为。
- 把 `/data2` 写入 provider interface 或 canonical entrypoint plan。

未来边界：

- launcher 可以继续负责 full-scale run_dir 选择和后台进程管理。
- canonical runtime contract 要求的 `run_dir/status/metadata/checkpoints/logs/evaluation/predictions` 由 runtime 和训练入口共同满足。
- provider/entrypoint plan 不硬编码 `/data2`；实际 full-scale run_dir 由 launcher/runtime 显式传入并写进 metadata。

## 5. 第一批代码迁移顺序

建议顺序：

1. **PredictionCacheExpertProvider**：最低风险，基于已有 `PredictionBatchReader` 和 golden packed fixture，可用现有 `stage1_golden_smoke.py` 锁定 sample_key 顺序、五专家顺序、`y_pred/y_true` shape、共享 `y_true` 和 row index。
2. **Evaluator adapter**：风险低于 FeatureProvider，可复用 P3a-P3d 的 `time_router.evaluation` public API；先在 smoke 中对 `ExpertBatch + RouterOutput` 复算 summary/rows。
3. **TimeFuseFeatureCacheProvider**：风险中等，feature 为 17 维结构化 CSV，适合用小 shard pressure smoke 锁定 split 下推、feature dim 和 scaler fit 不加载 prediction arrays。
4. **TimeFuseLinearSoftmaxHead**：在 feature/expert/evaluator contract 稳定后抽出，保持 `nn.Linear -> softmax` 行为不变。
5. **VisualOnlineVitFeatureProvider**：风险最高，推迟到最后；迁移前需要单独 visual 小规模 smoke 锁定 pseudo image、ViT normalization、pooling、dtype、device 和 online batch 行为。
6. **Visual Router head adapter**：可与 Visual provider 分离，但正式接入应等 Visual feature path smoke 稳定后进行。

是否先实现 `PredictionCacheExpertProvider`：是。它是两条 canonical entrypoint 的共享最小依赖，也是最容易用 golden smoke 证明不漂移的 adapter。

是否先做 evaluator adapter：应作为第二步。它不触碰 feature 生成和训练 loop，能先统一 hard/raw-soft/diagnostics 复算口径。

是否推迟 FeatureProvider：是。TimeFuse FeatureProvider 可第二批接入，Visual FeatureProvider 必须更晚接入。

最低风险、最容易 golden smoke 锁住的一步：`PredictionCacheExpertProvider`，因为已有 reader、fixture、固定专家顺序和 row index 断言。

## 6. 新旧入口过渡策略

过渡原则：

- 不破坏当前 streaming 入口；旧入口继续作为可运行正式主线，直到新 canonical path 完成 smoke、pressure 和 full-scale 复验。
- 新 adapter 先由 smoke 使用，不直接替换 full-scale 训练入口。
- 正式入口后续小步接入：先只替换 ExpertProvider 读取，再替换 Evaluator 复算，再替换 TimeFuse FeatureProvider，最后处理 Visual online ViT FeatureProvider。
- 每一步迁移前后运行 golden smoke；涉及 TimeFuse feature 时增加小 shard/pressure smoke；涉及 Visual feature 时增加 online embedding 小规模 smoke。
- 允许重跑实验，不强兼容旧输出 schema。新 runtime 可以采用 P5a `metadata.json/status.json/evaluation/predictions` 契约，不为 legacy CSV/metadata schema 增加反向 adapter。
- 历史输出目录保留复现价值，但不作为新 contract 必须兼容的 schema 来源。

## 7. `run_dir` 与 `/data2` 边界

本迁移计划不创建 `run_dir`，也不启动任何 full-scale 任务。

未来约束：

- full-scale `run_dir` 由 launcher/runtime 显式传入。
- provider adapter 不命名、不创建、不硬编码输出根。
- canonical entrypoint 可以接收 `--run-dir` 或等价 runtime 参数，但不应在 provider interface 中写死 `/data2`。
- P4 path/json/metadata helper 只能作为底层能力使用，不反向决定 full-scale 输出目录。
- 文档、adapter 和 provider spec 中只记录“实际路径由 runtime/launcher 传入”，具体 `/data2/syh/Time/...` 只出现在实际运行 metadata、日志或 launcher 参数中。

## 8. P5e 明确不做

- 不实现 provider adapter。
- 不新增 ExpertProvider / FeatureProvider 读取代码。
- 不修改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / IO helper。
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

## 9. 后续门禁

本计划完成后的验证只证明文档和现有 smoke 仍可运行。后续任何代码迁移至少需要：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

其中 `PredictionCacheExpertProvider` 迁移前后必须重点比较 golden fixture 的 sample_key 顺序、专家顺序、shape、row index、hard top-1、raw soft fusion 和 summary/rows 数值。
