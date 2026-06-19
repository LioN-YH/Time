# Stage 1 Expert System Boundary Review

创建日期：2026-06-19

## 1. 目标

本文记录 P6a `PredictionCacheExpertProvider` 之后、P6b FusionEvaluator adapter 之前的专家系统边界审计。结论只冻结架构边界和文档口径，不实现新的训练入口，不迁移正式入口，不新增 runtime、launcher、config 或 run_dir。

核心结论：

```text
ExpertProvider / ExpertBatch 是 Time framework 的长期专家系统边界。
PredictionCacheExpertProvider 只是当前 Stage 1 canonical experiment 的 prediction-cache adapter implementation。
```

固定五专家顺序是当前 Stage 1 canonical experiment 的契约，不是 Time framework 长期必须绑定的全局专家系统契约。

## 2. 三层契约

### 2.1 Time framework long-term expert-system contract

长期专家系统契约由 `ExpertProvider` 与 `ExpertBatch` 表达。

- `ExpertProvider` 负责提供专家输出、共享 `y_true`、专家列顺序和必要 lineage。
- `ExpertBatch` 是下游 Router、Fusor、Evaluator 的统一专家输出载体。
- contract 不绑定固定专家数量。
- contract 不绑定 QuitoBench。
- contract 不绑定 prediction cache。
- contract 不绑定 `packed_npy_v1`。
- contract 不要求专家一定来自离线数组，也不要求专家一定是当前五个模型。

长期允许的专家输出来源包括：

- prediction cache；
- statistical baselines；
- online expert models；
- external expert systems；
- dynamic expert pools；
- TimeFuse-style fusor branch 所需专家输出。

因此，cache 是实现，不是 interface。`packed_npy_v1`、manifest row index 和 `PredictionBatchReader` 都属于当前实现路径的能力，不应上升为所有 `ExpertProvider` 必须实现的全局接口。

### 2.2 Stage 1 canonical experiment contract

Stage 1 canonical experiment 当前仍绑定更具体的实验契约：

- 固定五专家顺序：`DLinear`、`PatchTST`、`CrossFormer`、`ES`、`NaiveForecaster`；
- golden fixture；
- packed prediction cache；
- `sample_key` 保序；
- 共享 `y_true`；
- `y_pred_row_index` / `y_true_row_index` lineage；
- `verify_metrics` 对 manifest MAE/MSE 的复算校验。

这些约束用于保证 Stage 1 当前 Visual Router 与 TimeFuse-style fusor 能在同一批专家输出上公平比较。它们可以继续作为 Stage 1 canonical experiment 的强约束，但不能反向定义 Time framework 的长期专家系统边界。

### 2.3 PredictionCacheExpertProvider implementation constraint

`PredictionCacheExpertProvider` 是 Stage 1 canonical experiment 的 adapter implementation。

该实现可以继续保留固定五专家顺序校验，因为它服务的是当前 Stage 1 canonical experiment，而不是所有未来专家系统。

实现约束：

- 复用 `PredictionBatchReader`。
- 调用方必须显式传入 `sample_keys`，并保持 batch loading 保序。
- 不扫描全量 manifest 作为默认行为。
- 不做 evaluation / loss。
- 不生成 Visual Router 或 TimeFuse feature。
- 不读取 oracle/TSF supervision。
- 不创建 runtime artifact、`run_dir`、`status.json` 或 `metadata.json`。
- 不承担 Bash launcher 或 config system 职责。

## 3. ExpertBatch 下游边界

`ExpertBatch` 是下游统一输入，而不是 prediction cache 的泄漏对象。

下游应只依赖：

- `sample_keys`
- `model_columns`
- `y_pred`
- `y_true`
- `row_index_metadata`
- `extra` 中的轻量 lineage

Visual Router 主线和 TimeFuse-style fusor 支线后续都应依赖 `ExpertBatch` / protocol types，而不是直接绑定 packed prediction cache。packed cache 的路径、manifest schema、array storage 和 row index 细节应留在 `PredictionCacheExpertProvider` 与 `PredictionBatchReader` 内部，除非 evaluator 或 runtime metadata 明确需要记录 lineage。

## 4. P6b FusionEvaluator Adapter 边界

P6b FusionEvaluator adapter 后续应消费显式 protocol 对象：

```text
ExpertBatch + RouterOutput
  -> EvaluationInput
  -> Evaluator / time_router.evaluation public API
```

P6b 不应重新读取 prediction cache，也不应绕过 `ExpertBatch` 直接从 manifest 或 packed npy 组装 `y_pred/y_true`。如果需要 row index、array storage 或 manifest path 作为诊断信息，应从 `ExpertBatch.row_index_metadata` 或 `ExpertBatch.extra` 获取轻量 lineage。

该边界确保：

- ExpertProvider 只负责专家输出；
- RouterHead / Fusor 只负责 logits 或 weights；
- Evaluator 只从显式输入复算指标；
- prediction cache adapter 不变成隐式 evaluator 或 runtime。

## 5. ExpertProvider 不承担的职责

`ExpertProvider` 不应承担以下职责：

- feature generation；
- Visual pseudo image 或 ViT embedding 生成；
- TimeFuse feature cache 读取或 scaler fit；
- oracle / TSF supervision 读取；
- loss 计算；
- evaluation summary、comparison 或 prediction rows 写出；
- runtime artifact 管理；
- `run_dir` 命名或创建；
- `status.json` / `metadata.json` 写出；
- checkpoint、resume 或 latest index 管理；
- Bash launcher、GPU 资源绑定或后台进程管理；
- config system 默认值解析。

这些职责分别属于 `FeatureProvider`、training/runtime、Evaluator、launcher 或 config 层。

## 6. 当前保留判断

P6a provider 当前可以继续：

- 使用 `DEFAULT_MODEL_COLUMNS` 的固定五专家顺序；
- 拒绝空 `sample_keys` 和重复 `sample_key`；
- 复用 `PredictionBatchReader` 的共享 `y_true` 校验；
- 保留 `verify_metrics`；
- 在 smoke 中复算 golden hard top-1 / raw soft 指标来证明 adapter 未改变 prediction 输出。

这些都是 Stage 1 canonical experiment adapter 的正确实现约束，不代表未来所有 `ExpertProvider` 都必须是五专家、packed cache 或 QuitoBench manifest。

## 7. 本次明确不做

- 不修改 `PredictionBatchReader` 行为。
- 不修改 `PredictionCacheExpertProvider` smoke 语义。
- 不移动 `prediction_array_io`。
- 不访问 `/data2`。
- 不创建 `run_dir`。
- 不写 `status.json` / `metadata.json`。
- 不实现 config system。
- 不实现 runtime / launcher。
- 不新增 Bash 或 `scripts/` entrypoint。
- 不修改 Visual Router / TimeFuse fusor 正式入口。
- 不改模型结构、loss 或正式输出目录。
- 不新增正式 provider abstraction 代码。

## 8. 验收命令

本文档冻结后，仍以现有 smoke 证明文档更新没有破坏 P1-P6a 低风险 contract：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```
