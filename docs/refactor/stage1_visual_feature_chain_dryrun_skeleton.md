# Stage 1 P19a VisualFeatureChain Dry-run Skeleton

## 目标

P19a 在 P16f protocol skeleton 之后新增真实 Visual feature chain 的轻量编排骨架：

```text
SampleManifest ordered sample_keys
-> explicit raw window fixture
-> RawWindowProvider
-> PreImageTransform
-> PseudoImageTransformer
-> ResizePolicy
-> VisualEncoderProvider
-> PoolingStrategy
-> optional FeatureTransform
-> canonical FeatureBatch
```

本步只落实组件编排、保序检查和 metadata lineage，不加载真实 ViT，不导入
`transformers`，不迁移训练入口，不自动搜索 raw window / feature / checkpoint，也不访问
`/data2`。

## 新增实现

- `time_router/features/visual_chain_runner.py`
  - 新增 `VisualFeatureChainRunner`。
  - 新增 `VisualFeatureChainResult`。
  - 复用 P16f `VisualFeatureChainSpec` 和各协议组件，不把 checkpoint、run_dir、HF processor
    或真实 ViT state 写入接口。
  - 每一层执行后立即校验 `sample_keys` 与输入 manifest order 完全一致。
  - 最终把 `chain_runner`、`raw_window_source`、`pseudo_image`、`resize_policy`、
    `encoder`、`pooling_strategy`、`feature_transform` 和 `chain_lineage` 写入
    `FeatureBatch.feature_schema`。

- `time_router/features/__init__.py`
  - 将 `VisualFeatureChainRunner`、`VisualFeatureChainResult` 和 `CHAIN_LINEAGE` 暴露为
    Stage 1 migration bridge public API。
  - 仍不暴露任何 fake encoder 或 mock provider。

## Smoke Fixture

- `tests/fixtures/stage1_visual_feature_chain_dryrun/raw_windows.json`
  - 明确保存 4 个 P13b sample_key 的 raw window。
  - 每个 window 长度为 6，数值仅用于 deterministic shape/lineage smoke。
  - fixture 不来自 `/data2`，不代表正式 Visual Router 数值。

## Smoke 覆盖

`tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py` 覆盖：

- 读取 P13b `sample_manifest.csv` 的 ordered sample_keys。
- 读取显式 raw window JSON fixture。
- 使用 smoke-local fake no-transformers visual encoder。
- 经 `VisualFeatureChainRunner` 输出 canonical `FeatureBatch`。
- 验证 sample_key 保序、features 为二维 `float32`、全部有限。
- 验证 `feature_schema` / `extra.chain_metadata.stage_metadata` 包含完整 lineage。
- 将 FeatureBatch 接入内存小型 `LoadedTorchMLPRouterHeadAdapter` 和
  `EvaluationInputAdapter`，证明可接 P17 canonical eval 后半段。
- 确认新增生产源码未导入 `transformers`、未包含 ViT/HF processor token、未访问 `/data2`。
- 确认未修改 `train_visual_router_online_streaming.py`。

## 明确不做

- 不加载真实 ViT。
- 不导入 `transformers`。
- 不迁移 `train_visual_router_online_streaming.py`。
- 不运行 full-scale，不新增 launcher。
- 不访问 `/data2`。
- 不读取 checkpoint，不自动搜索 feature/raw window。
- 不把 fake encoder 放入 `time_router.features` public core。
- 不要求与 legacy Visual Router 数值对齐。
- 不删除 P17 precomputed feature path。

## 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_chain_runner.py tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
```

## 后续

P19b 已在 P19a skeleton 后新增 guarded `VisualVitEncoderProvider`，见
`docs/refactor/stage1_visual_vit_encoder_guard.md`。该 provider 只迁移
`VisualEncoderProvider` 真实 ViT adapter 边界，默认 import 仍不导入 transformers，真实
model / processor path 由 Runtime guard 显式授权。

后续可以继续做真实 raw window provider、pseudo image transform 或把 guarded ViT provider 接到
显式 small/manual entrypoint。每一步仍应保持 provider/transform/encoder 只消费显式输入，
训练入口、full-scale launcher、真实资源加载和路径授权留在 Runtime / entrypoint 层。
