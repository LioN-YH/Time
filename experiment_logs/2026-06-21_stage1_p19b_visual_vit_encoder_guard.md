# Stage 1 P19b guarded VisualVitEncoderProvider

日志日期：2026-06-21 21:47:12 CST

## 目的

新增 guarded real ViT encoder provider dry-run contract，只迁移 P16f
`VisualEncoderProvider` 的真实 ViT adapter 边界，使它可以接入 P19a
`VisualFeatureChainRunner` 并输出 canonical `VisualEmbeddingBatch` / `FeatureBatch`。

## 背景

P17a-P17d 已完成 Visual canonical eval entrypoint、real checkpoint guard、external
feature/scaler guard 和 manual real-artifact dry-run contract。P18a/P18b 已完成
`time_router` public API cleanup 和 smoke-only scaffold 迁出。P19a 已新增
`VisualFeatureChainRunner` dry-run skeleton，但只使用 smoke-local fake no-transformers
encoder，没有真实 ViT provider。

本步要求默认 smoke 保持 no-transformers；只有用户显式授权并提供真实 model / processor
路径时，才允许通过 lazy import 构造 real ViT encoder provider。不迁移训练入口，不跑
full-scale，不默认下载 HuggingFace 模型，不默认访问 `/data2`。

## 操作

1. 新增 `time_router/runtime/visual_vit_guard.py`。
   - 实现 `VisualVitModelPathPolicy`。
   - 实现 `authorize_visual_vit_model_paths(...)`。
   - 默认只允许 `tests/fixtures` 或 `/tmp` tiny/local dry-run path。
   - 非 fixture/tmp path 必须显式 `allow_real_vit=True`。
   - `/data2` path 必须额外显式 `allow_external_vit_path=True`。
   - helper 不检查文件存在、不读取文件、不导入 transformers。
2. 新增 `time_router/features/visual_vit_encoder.py`。
   - 实现 `VisualVitEncoderProvider.encode(batch)`，输出三维 `float32`
     `VisualEmbeddingBatch` 并保持 sample_key 顺序。
   - provider metadata 记录 `encoder_provider=VisualVitEncoderProvider`、
     `loads_real_vit`、`model_path_policy`、`processor_path_policy`、`allow_real_vit`、
     `allow_external_vit_path`、`lazy_transformers_import=true`、`training_started=false`
     和 `formal_training_migration=false`。
   - 新增 `build_visual_vit_encoder_provider(...)`，真实 `AutoImageProcessor` /
     `ViTModel` 只在函数体内 lazy import，默认 `local_files_only=True`。
3. 更新 public API。
   - `time_router/features/__init__.py` 暴露 `VisualVitEncoderProvider` 和
     `build_visual_vit_encoder_provider`。
   - `time_router/runtime/__init__.py` 暴露 `VisualVitModelPathPolicy`、
     `authorize_visual_vit_model_paths` 和 fixture/tmp 分类 helper。
4. 新增 `tests/smoke/stage1_visual_vit_encoder_guard_smoke.py`。
   - 覆盖默认 import boundary。
   - 覆盖 fixture/tmp、未授权 real path 和 `/data2` 双授权 guard policy。
   - 用注入式 fake processor/model 构造 `VisualVitEncoderProvider`，不依赖真实
     transformers。
   - 验证 `VisualEmbeddingBatch` 保序、三维 shape、`float32`、finite 和 metadata。
   - 接入 P19a `VisualFeatureChainRunner`，输出 `FeatureBatch`，再接 small
     `LoadedTorchMLPRouterHeadAdapter` 和 `EvaluationInputAdapter`。
   - manual real ViT dry-run 默认跳过；只有
     `STAGE1_VISUAL_REAL_VIT_MODEL_PATH` 和 `STAGE1_VISUAL_REAL_VIT_ALLOW_REAL=1`
     显式设置时才运行。
5. 更新文档和结构索引。
   - 新增 `docs/refactor/stage1_visual_vit_encoder_guard.md`。
   - 更新 P19a skeleton 文档、public API cleanup audit、Stage 1 roadmap。
   - 更新 `WORKSPACE_STRUCTURE.md` 登记新增 provider、guard、smoke 和文档。

## 结果

已通过新增 P19b smoke、P19a/P17/P18b 回归 smoke、compileall、`git diff --check`
和边界审计：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_boundary_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_vit_encoder.py time_router/runtime/visual_vit_guard.py time_router/features/__init__.py time_router/runtime/__init__.py tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
git diff --check
git diff --name-only -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
git diff --name-only -- '*.sh' 'exp_scripts/*.sh'
```

新增 smoke 输出确认：

- 默认 import `time_router.features/runtime` 未导入 transformers，未触碰正式 streaming 训练入口。
- ViT model/processor path guard 覆盖 fixture/tmp、未授权 real path 和 `/data2` 双授权。
- 注入式 `VisualVitEncoderProvider` 输出保序三维 `float32` finite `VisualEmbeddingBatch`。
- 注入式 provider 可接 `VisualFeatureChainRunner`、small MLP adapter 和 `EvaluationInputAdapter`。
- manual real ViT dry-run 因环境变量未设置按预期跳过。
- P19a `VisualFeatureChainRunner` smoke 仍通过。
- P17a/P17b/P17c/P17d Visual eval canonical 相关 smoke 仍通过。
- P18a/P18b public API / scaffold cleanup smoke 仍通过。
- `git diff --check` 无输出。
- `train_visual_router_online_streaming.py` 无 diff。
- 没有新增或修改 Bash launcher。
- 新增生产 provider 只在 `build_visual_vit_encoder_provider(...)` 函数体内包含
  `from transformers import AutoImageProcessor, ViTModel`，默认 package import 不导入
  transformers。

## 结论

P19b guarded real ViT encoder provider 的核心实现已完成，默认路径仍保持 no-transformers。
真实 ViT model / processor path 只能由调用方显式传入，并通过 Runtime guard 授权；provider
本身不接收 run_dir、不搜索模型路径、不训练、不启动 full-scale。

## 下一步方案

1. 小步提交并 push 到 `origin/refactor/stage1-route-audit`。
2. 后续如需连接真实本地 ViT artifact，应通过 manual small/dry-run entrypoint 显式传入 model /
   processor path，并保留 `allow_real_vit` / `allow_external_vit_path` 双授权口径。
