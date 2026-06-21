# Stage 1 P16c Visual Precomputed FeatureProvider

## 1. 目标

P16c 新增最小 precomputed/head-ready Visual FeatureProvider：

```text
precomputed head-ready visual embedding CSV
-> VisualPrecomputedFeatureProvider
-> FeatureBatch
-> LoadedTorchMLPRouterHeadAdapter
-> RouterOutput
-> EvaluationInputAdapter
```

本步骤只验证 Visual feature provider 的输出边界，不接真实 ViT、不构造 pseudo image、不处理 scaler、不读取 checkpoint，也不迁移正式 Visual Router 训练入口。

## 2. 与 P16a / P16b 的关系

P16a 已证明 `LoadedTorchMLPRouterHeadAdapter` 可以消费 head-ready `float32 FeatureBatch.features`，并输出 canonical `RouterOutput`。

P16b 已冻结真实 Visual feature chain 边界：

```text
HistoryWindowProvider / VisualRawInputProvider
-> PseudoImageTransformer
-> VisualEncoderProvider / FrozenViTFeatureProvider
-> optional FeatureScaler / FeatureNormalizer
-> VisualFeatureProvider / FeatureBatch
```

P16c 不实现上述完整链路，而是先落地最窄的 head-ready provider：读取仓库内 tiny precomputed embedding fixture，并输出 canonical `FeatureBatch`。这一步让 `time_router/features` 先拥有正式化的 Visual feature provider 边界，但不声称真实 ViT provider 已完成。

## 3. 新增实现

- `time_router/features/visual_precomputed.py`
  - 新增 `VisualPrecomputedFeatureProvider`。
  - 输入 `feature_source_path`、可选 `feature_columns`、可选 `feature_schema_name/source_name/provider_name`。
  - 自动识别 `feature_` 前缀列，校验 sample_key 非空且唯一、feature column 非空、feature 值可转 float 且有限。
  - `load_batch(sample_keys)` 按请求顺序输出，不按 CSV 文件顺序输出。
  - 缺失 sample_key、重复 requested sample_key、坏 fixture 都 fail-fast。
- `time_router/features/__init__.py`
  - 导出 `VisualPrecomputedFeatureProvider`。
- `tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv`
  - 使用 P13b real-derived small manifest 的 4 个 sample_key，覆盖 test split 两个 sample_key。
  - 行顺序故意不同于 manifest 顺序。
  - 只包含 `sample_key` 与 `feature_0 ... feature_7`。
- `tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py`
  - 串联 `SampleManifest ordered sample_keys -> VisualPrecomputedFeatureProvider / FeatureBatch -> LoadedTorchMLPRouterHeadAdapter / RouterOutput -> EvaluationInputAdapter / summary + rows`。

## 4. FeatureBatch Metadata

P16c 输出的 `FeatureBatch.feature_schema` 固定记录：

```python
{
    "provider_name": "VisualPrecomputedFeatureProvider",
    "feature_schema_name": "visual_precomputed_head_ready_v1",
    "feature_dim": 8,
    "feature_columns": ("feature_0", ..., "feature_7"),
    "head_ready": True,
    "loads_real_vit": False,
    "handles_scaler": False,
    "precomputed": True,
    "dtype": "float32",
}
```

`FeatureBatch.extra` 只保存轻量来源 metadata，例如 provider name、fixture source、sample_key column 和可用行数；不保存 `run_dir`、checkpoint、prediction cache、oracle 或 expert error。

## 5. 明确不负责

P16c 不负责：

- history window 读取；
- pseudo image 构造；
- ViT embedding；
- scaler fit / transform；
- checkpoint loading；
- prediction backend；
- oracle / expert error；
- run_dir；
- training loop；
- Bash launcher；
- 正式 Visual Router 入口迁移。

## 6. Smoke 验收

P16c smoke 覆盖：

- provider 输出 `FeatureBatch`；
- sample_key 按 P13b manifest 顺序输出；
- fixture 文件行顺序打乱不影响输出；
- features shape 为 `[4, 8]`；
- features dtype 为 `np.float32`；
- schema 记录 `precomputed=True`、`head_ready=True`、`loads_real_vit=False`、`handles_scaler=False`；
- missing sample_key 抛错；
- duplicate sample_key fixture 抛错；
- non-finite feature fixture 抛错；
- provider 不持有 `run_dir`；
- provider 源码不出现 `/data2`、`torch.load`、`ViTModel`、`AutoImageProcessor`、`VisualMLPRouter` 或正式训练入口；
- patch `torch.load` 后，P16a adapter/evaluator 链路不读取 checkpoint；
- P16a `LoadedTorchMLPRouterHeadAdapter` 可消费该 `FeatureBatch` 并输出 `RouterOutput`；
- `EvaluationInputAdapter` 可消费 `RouterOutput` 与 `ExpertBatch`，生成 summary 和 per-sample rows；
- per-sample rows 保持 sample_key 顺序；
- smoke 不创建 canonical `run_dir` 或 `experiment_logs/run_outputs/` 新目录。

验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_precomputed.py tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
```

回归 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
```

## 7. 后续

后续应单独推进：

1. scaler boundary smoke：loaded scaler transform -> head-ready `float32 FeatureBatch`，并禁止 silent fit；
2. fake encoder provider：验证 Visual encoder provider 边界而不加载真实 ViT；
3. online ViT provider audit/smoke：单独处理 pseudo image、frozen ViT、batching、device/dtype/resource policy；
4. legacy checkpoint/signature audit：处理 `VisualMLPRouter` import、constructor、state_dict 和 DataParallel key；
5. 正式 Visual entrypoint migration：在 Runtime 中加载 checkpoint/scaler/encoder，再把 head-ready `FeatureBatch` 和已加载 module 交给 provider/head/evaluator 链路。
