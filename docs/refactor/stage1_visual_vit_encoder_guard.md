# Stage 1 P19b Visual ViT Encoder Guard

## 目标

P19b 在 P19a `VisualFeatureChainRunner` 之后新增 guarded real ViT encoder provider
dry-run contract：

```text
VisualInputBatch
-> guarded ViT model / processor path policy
-> lazy transformers import
-> VisualVitEncoderProvider
-> VisualEmbeddingBatch
-> PoolingStrategy
-> FeatureBatch
```

本步只迁移 `VisualEncoderProvider` 的真实 ViT adapter 边界。默认 smoke 仍保持
no-transformers，不迁移训练入口，不跑 full-scale，不自动下载 HuggingFace 模型，不自动搜索
`/data2`。

## 新增实现

- `time_router/runtime/visual_vit_guard.py`
  - 新增 `VisualVitModelPathPolicy`。
  - 新增 `authorize_visual_vit_model_paths(...)`。
  - 默认只允许 `tests/fixtures` 或 `/tmp` tiny/local dry-run path。
  - 非 fixture/tmp path 必须显式 `allow_real_vit=True`。
  - `/data2` path 必须额外显式 `allow_external_vit_path=True`。
  - helper 不检查文件存在、不读取文件、不导入 transformers。

- `time_router/features/visual_vit_encoder.py`
  - 新增 `VisualVitEncoderProvider`，实现 P16f `VisualEncoderProvider.encode(batch)`。
  - `encode(...)` 输出 `VisualEmbeddingBatch`，embedding 为三维 `float32` 且 sample_key 保序。
  - provider metadata 固定记录 `encoder_provider=VisualVitEncoderProvider`、
    `loads_real_vit`、`model_path_policy`、`processor_path_policy`、`allow_real_vit`、
    `allow_external_vit_path`、`lazy_transformers_import=true`、`training_started=false`
    和 `formal_training_migration=false`。
  - 新增 `build_visual_vit_encoder_provider(...)`，只在显式构造且未注入 fake class 时，
    在函数体内 lazy import `AutoImageProcessor` / `ViTModel`。
  - 默认 `local_files_only=True`，避免隐式联网下载。

- `time_router/features/__init__.py` / `time_router/runtime/__init__.py`
  - 暴露 P19b provider 和 guard bridge。
  - package import 阶段不导入 transformers。

## Smoke 覆盖

`tests/smoke/stage1_visual_vit_encoder_guard_smoke.py` 覆盖：

- 默认 import boundary：`import time_router.features/runtime` 成功，且 `sys.modules`
  中没有 `transformers`。
- guard policy：fixture/tmp path 默认通过；仓库内非 fixture/tmp path 未授权 fail-fast；
  `/data2` path 未开 external flag fail-fast；双授权只通过 guard，不读取文件、不导入
  transformers。
- fake local ViT injection：通过注入 fake processor/model 构造 `VisualVitEncoderProvider`，
  不依赖真实 transformers；输出 `VisualEmbeddingBatch` 并验证保序、三维 shape、`float32`、
  finite 和 metadata。
- chain integration：将注入式 provider 接入 `VisualFeatureChainRunner`，输出
  `FeatureBatch`，再接 small `LoadedTorchMLPRouterHeadAdapter` 和
  `EvaluationInputAdapter`。
- manual real ViT dry-run 默认跳过：只有设置
  `STAGE1_VISUAL_REAL_VIT_MODEL_PATH` 和 `STAGE1_VISUAL_REAL_VIT_ALLOW_REAL=1` 时才运行；
  `STAGE1_VISUAL_REAL_VIT_PROCESSOR_PATH` 可选；`/data2` 或其他外部路径需要
  `STAGE1_VISUAL_REAL_VIT_ALLOW_EXTERNAL=1`。

## 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不迁移训练入口。
- 不启动 full-scale。
- 不新增 Bash launcher。
- 不自动搜索 `/data2`。
- 不从 checkpoint 或 run_dir 推断 ViT 路径。
- 不默认下载 HuggingFace 模型，不默认联网。
- 不默认导入 transformers。
- 不要求与 legacy Visual Router 数值对齐。
- 不删除 P17 precomputed feature path。
- 不把 fake encoder 放回 `time_router.features` public API。

## 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_vit_encoder.py time_router/runtime/visual_vit_guard.py tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
```

## 后续

P19b 之后可以继续迁移真实 raw window provider、pseudo image transform 或把真实 ViT provider
接到显式 small/manual entrypoint。训练入口、full-scale launcher 和正式 `/data2` 资源调度仍需另起小步。
