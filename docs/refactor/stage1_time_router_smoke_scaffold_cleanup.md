# Stage 1 P18b time_router smoke scaffold cleanup

## 目标

本步在 P18a public API cleanup audit 之后，做一次小规模、可回滚的真实清理：
把 visual smoke-only scaffold 从 `time_router.features` package public 边界迁出，
改由 `tests.helpers.visual_smoke_providers` 局部承接。

本步不迁移真实 ViT provider、不迁移训练入口、不修改
`train_visual_router_online_streaming.py`，也不访问 `/data2` 或启动 full-scale。

## 迁移内容

- 新增 `tests/helpers/__init__.py`。
- 新增 `tests/helpers/visual_smoke_providers.py`，承接：
  - `VisualMockFeatureProvider`
  - `DeterministicVisualEncoderStub`
- `time_router.features.__init__` 不再导入上述 smoke-only scaffold。
- `time_router.features.visual_mock` 暂时保留为 P18c 前的 compatibility wrapper，
  只 re-export tests helper，不进入 `time_router.features.__all__`。
- 依赖 mock/stub 的 smoke 与 `scripts/run_stage1_visual_small.py` 显式从
  `tests.helpers.visual_smoke_providers` 导入，保持 Visual small 默认 mock path 行为不变。

## 保留边界

- `VisualPrecomputedFeatureProvider` 保持在 `time_router.features` public API 中。
- `LoadedFeatureScaler` 保持在 `time_router.features` public API 中。
- P16f `VisualFeatureChainSpec` 等 protocol skeleton 暂时保留，等待后续真实 ViT provider 迁移。
- P17 runtime migration bridge 继续保留：
  - `visual_mlp_checkpoint.py`
  - `visual_eval_checkpoint_guard.py`
  - `visual_eval_feature_guard.py`

## 新增 smoke

`tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py` 覆盖：

- `from time_router.features import VisualPrecomputedFeatureProvider, LoadedFeatureScaler` 成功。
- `time_router.features.__all__` 不包含 `VisualMockFeatureProvider` /
  `DeterministicVisualEncoderStub`。
- `time_router.features` package 入口不再保留这两个 smoke-only 属性。
- `tests.helpers.visual_smoke_providers` 可导入并实际使用 mock/stub。
- import 阶段不导入 `transformers`，不访问 `/data2`，不启动 ViT/训练/full-scale。

## 后续

P18c 可在确认没有旧子模块直接依赖后，删除 `time_router.features.visual_mock`
compatibility wrapper。真实 `VisualFeatureChain` / ViT provider 迁移仍应另起步骤处理。
