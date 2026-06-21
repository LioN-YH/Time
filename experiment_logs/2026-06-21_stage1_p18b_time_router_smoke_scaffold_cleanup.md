# Stage 1 P18b time_router smoke scaffold cleanup

日志日期：2026-06-21 19:13:13 CST

## 目的

把 P18a 已识别为 smoke-only scaffold 的 `VisualMockFeatureProvider` 和
`DeterministicVisualEncoderStub` 从 `time_router.features` package public API 边界迁出，
改由 tests helper 局部承接，为后续 P19 real `VisualFeatureChain` / ViT provider 迁移清理
正式 feature package 边界。

## 背景

P18a 已完成 `time_router` public API cleanup audit，并将当前模块分为长期 canonical core、
Stage 1 migration bridge 和 smoke-only scaffold。P18a 已先将 visual mock/stub 从
`time_router.features.__all__` 移出，但 `time_router.features` package 入口仍保留兼容属性。
本步目标是在不迁移 ViT、不迁移训练入口、不删除 smoke 能力的前提下，把 smoke-only visual
scaffold 的实现归属迁到 `tests/helpers`。

## 操作

1. 新增 `tests/helpers/__init__.py`，建立 tests helper package。
2. 新增 `tests/helpers/visual_smoke_providers.py`，承接
   `DeterministicVisualEncoderStub` 和 `VisualMockFeatureProvider` 的实现，并保留原有
   8 维 deterministic history statistics embedding、`FeatureBatch` 输出和不访问
   `/data2`/checkpoint/run_dir/ViT 的边界注释。
3. 修改 `time_router/features/__init__.py`，移除
   `DeterministicVisualEncoderStub` / `VisualMockFeatureProvider` 的 package 入口导入；
   `__all__` 继续只保留 `VisualPrecomputedFeatureProvider`、`LoadedFeatureScaler`、
   `TimeFuseFeatureCacheProvider` 和 visual chain protocol skeleton 等 core/bridge 名称。
4. 将 `time_router/features/visual_mock.py` 改为 P18c 前兼容 wrapper，只 re-export
   `tests.helpers.visual_smoke_providers` 中的两个 smoke-only 类，不再作为 feature package
   入口 public API。
5. 更新依赖 mock/stub 的 smoke 和 `scripts/run_stage1_visual_small.py`：
   - `tests/smoke/stage1_visual_feature_provider_mock_smoke.py`
   - `tests/smoke/stage1_visual_mock_protocol_eval_smoke.py`
   - `tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py`
   - `tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py`
   - `tests/smoke/stage1_visual_small_entrypoint_smoke.py`
   - `scripts/run_stage1_visual_small.py`
6. 新增 `tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py`，
   覆盖 core feature import、`features.__all__` 不泄漏、package 入口属性不保留、
   tests helper 可用、旧子模块兼容层指向 helper、import 阶段不导入 `transformers`。
7. 更新 `tests/smoke/stage1_time_router_public_api_boundary_smoke.py`，把 P18a 兼容属性预期
   改为 P18b 后 package 入口不保留 smoke-only 属性。
8. 更新 `docs/refactor/stage1_time_router_public_api_cleanup_audit.md`，补充 P18b 迁移结果。
9. 新增 `docs/refactor/stage1_time_router_smoke_scaffold_cleanup.md`，记录本步边界和后续 P18c。
10. 更新 `WORKSPACE_STRUCTURE.md`，记录新增 helper、P18b 文档和 cleanup smoke，并修正
    `time_router.features` / `visual_mock.py` 的当前口径。

## 结果

已使用 `/home/shiyuhong/application/miniconda3/envs/quito/bin/python` 完成以下验证：

- `tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py`：通过。
- `tests/smoke/stage1_time_router_public_api_boundary_smoke.py`：通过。
- 依赖 mock/stub 的旧 smoke：
  - `stage1_visual_feature_provider_mock_smoke.py`：通过。
  - `stage1_visual_mock_protocol_eval_smoke.py`：通过。
  - `stage1_visual_legacy_mlp_adapter_smoke.py`：通过。
  - `stage1_visual_mlp_routerhead_adapter_smoke.py`：通过。
  - `stage1_visual_small_entrypoint_smoke.py`：通过。
- P16j/P16k：
  - `stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py`：通过。
  - `stage1_visual_small_loaded_path_artifact_parity_smoke.py`：通过。
- P17a/P17b/P17c/P17d：
  - `stage1_visual_eval_canonical_thin_slice_smoke.py`：通过。
  - `stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py`：通过。
  - `stage1_visual_eval_canonical_external_feature_guard_smoke.py`：通过。
  - `stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py`：通过，真实 artifact
    环境变量未设置部分按预期跳过，synthetic contract 通过。
- P16c/P16d/P16i：
  - `stage1_visual_precomputed_feature_provider_smoke.py`：通过。
  - `stage1_visual_feature_scaler_smoke.py`：通过。
  - `stage1_visual_legacy_mlp_checkpoint_payload_smoke.py`：通过。
- `python -m compileall` 覆盖本次新增/修改的 `time_router/features`、`tests/helpers`、
  P18b/P18a smoke、相关旧 smoke 和 `scripts/run_stage1_visual_small.py`：通过。
- `git diff --check`：通过。
- `rg` 确认没有 smoke/script 再从 `time_router.features` 导入
  `VisualMockFeatureProvider` / `DeterministicVisualEncoderStub`。
- `git diff -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
  为空，确认未修改正式 streaming 训练入口。

## 结论

P18b 小步清理完成：visual smoke-only mock/stub 的实现归属已从 `time_router.features`
public package 边界迁到 `tests.helpers.visual_smoke_providers`；长期 core
`VisualPrecomputedFeatureProvider` 和 `LoadedFeatureScaler` 保持 public；P17 runtime migration
bridge 未迁移；Visual small 默认 mock path 行为保持不变；没有启动 ViT、训练或 full-scale，
没有访问 `/data2`，没有新增 Bash launcher。

## 下一步方案

1. 提交并 push 本次 P18b 小步变更到 `origin/refactor/stage1-route-audit`。
2. P18c 可在确认无旧子模块直接依赖后，删除 `time_router.features.visual_mock` 兼容 wrapper。
3. 后续 P19 再单独迁移 real `VisualFeatureChain` / ViT provider，不与 smoke scaffold cleanup 混做。

