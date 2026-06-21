# Stage 1 P18a time_router public API cleanup audit

## 目标

本审计在 P17 Visual canonical evaluation 入口迁移第一阶段之后，梳理
`time_router` 当前 public API、P16/P17 迁移桥和 smoke-only scaffold 的边界。
本步只做标注、文档和轻量 `__all__` 收紧，不迁移 ViT provider、不迁移训练入口、
不删除 P17 能力、不改 artifact schema，也不访问 `/data2`。

## 当前结论

`time_router` 已形成一条可用于 P17 canonical eval 的长期核心链路：

```text
SampleManifest
-> VisualPrecomputedFeatureProvider
-> optional LoadedFeatureScaler
-> Runtime checkpoint payload helpers / guards
-> LoadedTorchMLPRouterHeadAdapter
-> ExpertBatch
-> EvaluationInputAdapter / Evaluator helpers
-> Runtime artifact writer
```

当前仍同时保留 P14/P15 smoke-only Visual mock、P16f visual chain protocol skeleton、
P16i/P17 guard helper 等过渡对象。P18a 的清理口径是先把这些对象按用途分类，
把明显的 smoke-only mock 从 package `__all__` 中移出。P18b 已进一步把
`VisualMockFeatureProvider` / `DeterministicVisualEncoderStub` 的实现迁到
`tests.helpers.visual_smoke_providers`，并移除 `time_router.features` package 入口属性；
旧 `time_router.features.visual_mock` 子模块仅保留 P18c 前的 compatibility alias。

## 分类表

| 范围 | 分类 | 当前 public 名称 / 文件 | P18a 处理 |
| --- | --- | --- | --- |
| `time_router.protocols` | A. 长期 canonical core | `SampleManifest`、`SampleManifestRow`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput`、`SupervisionBatch`、`SplitSpec`、`ExperimentProtocolSpec` | 保持 public `__all__`。这些是 canonical dataflow 的轻量 contract object。 |
| `time_router.runtime.artifact_writer` | A. 长期 canonical core | `create_run_dir`、`write_run_metadata`、`write_run_status`、`write_sample_manifest_ref`、`write_split_summary`、`write_evaluation_summary`、`write_prediction_rows_csv` | 保持 public `__all__`。Runtime writer 是 canonical run_dir 写出核心。 |
| `time_router.evaluation` | A. 长期 canonical core | `EvaluationInputAdapter`、`EvaluationInputAdapterResult`、metrics、summary、prediction rows helper | 保持 public `__all__`。`FusionEvaluator` 是兼容包装，短期仍保留。 |
| `time_router.models.visual_mlp_adapter` | A. 长期 canonical core | `LoadedTorchMLPRouterHeadAdapter` | 保持 public `__all__`。它只消费已加载 torch module 和 head-ready `FeatureBatch`，不读取 checkpoint。 |
| `time_router.features.visual_precomputed` | A. 长期 canonical core / debug provider | `VisualPrecomputedFeatureProvider` | 保持 public `__all__`。当前服务 P17 eval precomputed feature，后续 ViT provider 迁移后仍可作为 fixture/debug/ablation provider。 |
| `time_router.features.visual_scaler` | A. 长期 canonical core / transform | `LoadedFeatureScaler` | 保持 public `__all__`。它表达 loaded scaler transform，不执行 fit。 |
| `time_router.features.timefuse_cache` | A. 长期 canonical core / branch provider | `TimeFuseFeatureCacheProvider` | 保持 public `__all__`。当前仍偏 smoke/small，但接口方向与 canonical `FeatureBatch` 一致。 |
| `time_router.experts.prediction_cache` | A. 长期 canonical core / adapter | `PredictionCacheExpertProvider` | 保持 public `__all__`。目前是 prediction-cache adapter，后续可继续向正式 provider 收束。 |
| `time_router.runtime.visual_mlp_checkpoint` | B. migration bridge | `load_checkpoint_payload`、`extract_router_state_dict`、`strip_dataparallel_prefix`、`load_router_state_dict` | 保持 public `__all__`，并在 runtime 注释中标为 Stage 1 bridge。未来可并入正式 checkpoint/artifact policy。 |
| `time_router.runtime.visual_eval_checkpoint_guard` | B. migration bridge | `CheckpointPathPolicy`、`authorize_visual_eval_checkpoint_path`、`is_data2_path`、`is_fixture_or_tempfile_checkpoint` | 保持 public `__all__`。服务 P17b/P17d guarded real-checkpoint dry-run。 |
| `time_router.runtime.visual_eval_feature_guard` | B. migration bridge | `VisualEvalPathPolicy`、`authorize_visual_eval_feature_path`、`authorize_visual_eval_scaler_path`、`is_fixture_or_tempfile_visual_eval_artifact` | 保持 public `__all__`。服务 P17c/P17d external feature/scaler dry-run。 |
| `time_router.features.visual_chain` | B. migration bridge / protocol skeleton | `RawWindowBatch`、`PreImageBatch`、`VisualInputBatch`、`VisualEmbeddingBatch`、`RawWindowProvider`、`VisualEncoderProvider`、`PoolingStrategy`、`FeatureTransform`、`VisualFeatureChainSpec` 等 | 保持 public `__all__`。它是 future ViT provider 的协议骨架，真实 provider 迁移后再判断是否拆分。 |
| `tests.helpers.visual_smoke_providers` | C. smoke-only scaffold | `VisualMockFeatureProvider`、`DeterministicVisualEncoderStub` | P18b 后作为 visual smoke / Visual small rehearsal 的唯一实现归属；旧 smoke 与 small rehearsal 显式从 tests helper import。 |
| `time_router.features.visual_mock` | C. smoke-only compatibility wrapper | `VisualMockFeatureProvider`、`DeterministicVisualEncoderStub` | P18b 后只 re-export tests helper，兼容直接导入旧子模块的临时代码；不由 `time_router.features` package 入口导入，不进入 `__all__`，P18c 可删除。 |
| `scripts/run_stage1_visual_eval_canonical.py` 内部 helper | B/C 混合 | `JsonExpertProvider`、`save/load` 相关 CLI helper、`import_legacy_visual_mlp_router` | 不导出到 `time_router`。`JsonExpertProvider` 仅是 P17 eval fixture JSON bridge，不应进入 package public API。 |
| `tests/smoke` 中 dummy/mock 类 | C. smoke-only scaffold | dummy chain components、smoke-only mock RouterHead、小型 fake state_dict helper | 不迁入 `time_router.__all__`。后续如复用需求变强，应迁到 `tests/helpers`，而不是 package public API。 |

## P17 canonical eval 依赖的 public API

`scripts/run_stage1_visual_eval_canonical.py` 依赖的 package public imports 当前为：

- `time_router.protocols`: `SampleManifest`、`SampleManifestRow`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`。
- `time_router.features`: `VisualPrecomputedFeatureProvider`、`LoadedFeatureScaler`。
- `time_router.models`: `LoadedTorchMLPRouterHeadAdapter`。
- `time_router.evaluation`: `EvaluationInputAdapter`、`EvaluationInputAdapterResult`。
- `time_router.runtime`: Runtime writer、checkpoint payload helper、checkpoint/feature/scaler path guard。

这些 import 都由 `tests/smoke/stage1_time_router_public_api_boundary_smoke.py` 覆盖。
该 smoke 只做 import boundary 检查，不读取 fixture、不访问 `/data2`、不导入
`transformers`，也不调用 CLI main。

## P18b/P18c 后续建议

1. P18b 已将 `VisualMockFeatureProvider` 和 `DeterministicVisualEncoderStub` 迁到
   `tests/helpers/visual_smoke_providers.py`，并把旧 smoke / Visual small 默认路径改为显式
   测试 helper import。P18c 可在确认无旧子模块依赖后删除 `time_router.features.visual_mock`
   compatibility wrapper。
2. 将 `JsonExpertProvider` 这类 entrypoint-local fixture JSON bridge 明确保留在 CLI 或
   测试 helper 内，不进入 `time_router.experts`，除非后续要把 JSON fixture provider
   定义成正式 debug provider。
3. `visual_mlp_checkpoint.py`、`visual_eval_checkpoint_guard.py` 和
   `visual_eval_feature_guard.py` 可在 Runtime config/checkpoint/artifact policy 成型后合并；
   在 real checkpoint / real feature dry-run 仍频繁变化前，暂不合并，降低 diff 风险。
4. `visual_chain.py` 应保留到 ViT provider 迁移后再判断拆分方式。真实实现出现前，
   该文件只表达协议骨架，不应承诺具体 RevIN、pseudo image、resize、HF processor、
   CLS/mean pooling 或 scaler fit 行为。
5. `time_router.__init__.py` 当前不做 star export 聚合，保持根包轻量。后续如果新增根级
   public API，应先写 boundary smoke，再扩展根级 `__all__`。

## 本步验证口径

P18a 新增 boundary smoke 的验收点：

- import `time_router` 和 `protocols/runtime/features/models/evaluation/experts` 成功。
- canonical core 和 P17 bridge public imports 均存在于对应 `__all__`。
- `VisualMockFeatureProvider`、`DeterministicVisualEncoderStub` 不在
  `time_router.features.__all__`，也不作为 `time_router.features` package 入口属性保留。
- 上述 smoke-only 类由 `tests.helpers.visual_smoke_providers` 承接；旧子模块兼容层只保留到
  P18c。
- import P17 canonical eval entrypoint 文件后，未加载 `transformers`。
- smoke 不读取 `/data2`，不启动 ViT、训练或 full-scale。
