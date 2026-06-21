# Stage 1 P18a time_router public API cleanup audit

日志日期：2026-06-21 18:53:12 CST

## 目的

审计 `time_router` 当前 public API、P16/P17 迁移桥和 smoke-only scaffold 边界，
新增 public API boundary smoke，并在不破坏 P17a/P17b/P17c/P17d、P16j/P16k smoke
的前提下做轻量整理。

## 背景

P17 已完成 Visual canonical eval 第一阶段：显式 `SampleManifest`、precomputed
Visual feature、可选 loaded scaler、guarded checkpoint payload、legacy
`VisualMLPRouter` loaded module、`LoadedTorchMLPRouterHeadAdapter`、`ExpertBatch`、
`EvaluationInputAdapter` 和 Runtime writer 已能串联 canonical `run_dir`。

当前仍未迁移 ViT provider、训练入口和 full-scale 正式运行；`time_router` 下也保留了
P14/P15/P16 smoke-only mock、protocol skeleton 和 P17 guard bridge。本步需要先分类
并收紧 public export 边界，避免把测试 scaffold 固化为长期 API。

## 操作

1. 读取并审计 `time_router/__init__.py`、`time_router/protocols/__init__.py`、
   `time_router/runtime/__init__.py`、`time_router/features/__init__.py`、
   `time_router/models/__init__.py`、`time_router/evaluation/__init__.py` 和
   `time_router/experts/__init__.py`。
2. 读取 P17 canonical eval 入口 `scripts/run_stage1_visual_eval_canonical.py`，
   确认其依赖的 public API 是 protocols、features、models、evaluation 和 runtime
   中的 canonical core / migration bridge。
3. 新增 `docs/refactor/stage1_time_router_public_api_cleanup_audit.md`，按 A/B/C 三类记录：
   A 为长期 canonical core，B 为 Stage 1 migration bridge，C 为 smoke-only scaffold。
4. 轻量调整 `time_router/features/__init__.py`：保留 `VisualMockFeatureProvider` 和
   `DeterministicVisualEncoderStub` 兼容属性，但从 `__all__` 移除，避免继续作为 public
   star-import API 推广。
5. 新增 `tests/smoke/stage1_time_router_public_api_boundary_smoke.py`，验证 canonical core、
   P17 bridge import 边界、smoke-only mock 不在 `features.__all__`，并确认 import 阶段不加载
   `transformers`。
6. 同步更新 `docs/refactor/stage1_refactor_roadmap.md`、`WORKSPACE_STRUCTURE.md` 和
   `experiment_logs/README.md`。

## 结果

新增或修改的关键文件：

- `docs/refactor/stage1_time_router_public_api_cleanup_audit.md`
- `tests/smoke/stage1_time_router_public_api_boundary_smoke.py`
- `time_router/features/__init__.py`
- `docs/refactor/stage1_refactor_roadmap.md`
- `WORKSPACE_STRUCTURE.md`
- `experiment_logs/README.md`

P18a 当前分类结论：

- 长期 core：protocol dataclass、Runtime artifact writer、`EvaluationInputAdapter`、
  `LoadedTorchMLPRouterHeadAdapter`、`VisualPrecomputedFeatureProvider`、
  `LoadedFeatureScaler`、`PredictionCacheExpertProvider` 等。
- migration bridge：legacy Visual MLP checkpoint payload helper、P17 checkpoint guard、
  P17 feature/scaler guard、P16f Visual feature chain protocol skeleton。
- smoke-only scaffold：`VisualMockFeatureProvider`、`DeterministicVisualEncoderStub`、
  smoke 内 dummy chain/mock head/fake state_dict helper、entrypoint-local JSON fixture provider。

已在 `2026-06-21 18:58:46 CST` 前完成验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall \
  time_router/features/__init__.py \
  time_router/runtime/__init__.py \
  tests/smoke/stage1_time_router_public_api_boundary_smoke.py \
  scripts/run_stage1_visual_eval_canonical.py \
  docs/refactor/stage1_time_router_public_api_cleanup_audit.md \
  experiment_logs/2026-06-21_stage1_p18a_time_router_public_api_cleanup_audit.md
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_boundary_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py
git diff --check
git diff -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```

结果：

- `compileall` 通过。
- 新增 P18a public API boundary smoke 通过，确认 P17 canonical eval entrypoint import 后未加载
  `transformers`。
- P17a/P17b/P17c/P17d smoke 全部通过。
- P16j/P16k smoke 全部通过。
- P16a/P16c/P16d/P16i 依赖链 smoke 全部通过。
- `git diff --check` 通过。
- `git diff -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
  为空，确认未修改正式 streaming 训练入口。

## 结论

P18a 使用最小代码改动明确了 public API 边界：canonical/P17 依赖仍可 public import，
smoke-only Visual mock 不再属于 `time_router.features.__all__`，但旧 smoke 兼容属性仍保留。
本步未删除核心能力，未迁移 ViT provider，未迁移训练入口，未修改
`train_visual_router_online_streaming.py`，也未改 artifact schema。

## 下一步方案

1. P18b 可把 Visual mock scaffold 迁到 `tests/helpers` 或 legacy smoke helper。
2. P18c 可在 Runtime checkpoint/artifact policy 成型后合并 P17 guard helper。
3. ViT provider 和训练入口迁移应另起步骤，继续保持 P17 eval-only dry-run 能力不变。
