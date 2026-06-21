# Stage 1 P19a VisualFeatureChain Dry-run Skeleton

日志日期：2026-06-21 20:44:32 CST

## 目的

新增真实 Visual feature chain 的 dry-run 编排骨架，把 P16f 已定义的
RawWindowProvider、PreImageTransform、PseudoImageTransformer、ResizePolicy、
VisualEncoderProvider、PoolingStrategy 和可选 FeatureTransform 串联为 canonical
FeatureBatch，为后续真实 ViT provider 迁移预留结构。

## 背景

P17a-P17d 已完成 Visual canonical eval entrypoint 的 evaluation-only thin slice、
real checkpoint guard、external feature/scaler guard 和 manual real-artifact dry-run contract。
P18a/P18b 已清理 `time_router` public API，把 smoke-only Visual mock/stub 迁到
`tests.helpers`。当前 `time_router.features` 保留 VisualPrecomputedFeatureProvider、
LoadedFeatureScaler 和 P16f visual_chain protocol skeleton，但还没有真实 Visual feature
chain 的编排器。

本步要求不加载真实 ViT、不导入 transformers、不训练、不跑 full-scale、不修改
`train_visual_router_online_streaming.py`，只新增可用 fake encoder dry-run 的 runner skeleton。

## 操作

1. 新增 `time_router/features/visual_chain_runner.py`。
   - 实现 `VisualFeatureChainRunner`、`VisualFeatureChainResult` 和 `CHAIN_LINEAGE`。
   - Runner 串联 P16f `VisualFeatureChainSpec` 中的协议组件。
   - 每一层执行后校验 `sample_keys` 与输入 ordered sample_keys 完全一致。
   - 最终把 `chain_runner`、`raw_window_source`、`pseudo_image`、`resize_policy`、
     `encoder`、`pooling_strategy`、`feature_transform` 和 stage metadata 写入
     `FeatureBatch.feature_schema` / `FeatureBatch.extra`。
2. 更新 `time_router/features/__init__.py`。
   - 暴露 `VisualFeatureChainRunner`、`VisualFeatureChainResult` 和 `CHAIN_LINEAGE`。
   - 未暴露 fake encoder 或 smoke-only provider。
3. 新增 `tests/fixtures/stage1_visual_feature_chain_dryrun/`。
   - `raw_windows.json` 保存 4 个 P13b sample_key 的显式 raw window。
   - README 说明 fixture 只用于 smoke，不来自 `/data2`，不代表正式 ViT embedding。
4. 新增 `tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py`。
   - 从 P13b manifest 读取 ordered sample_keys。
   - 从 JSON fixture 读取 raw window。
   - 使用测试内 fake no-transformers encoder、pseudo image、resize、pooling 和 identity transform。
   - 通过 `VisualFeatureChainRunner` 输出 canonical `FeatureBatch`。
   - 验证 sample_key 保序、features 为二维 `float32`、全部 finite、schema/metadata lineage 完整。
   - 额外接入内存小型 `LoadedTorchMLPRouterHeadAdapter` 和 `EvaluationInputAdapter`，验证可接 canonical eval 后半段。
5. 新增和更新文档。
   - 新增 `docs/refactor/stage1_visual_feature_chain_dryrun_skeleton.md`。
   - 更新 `docs/refactor/stage1_time_router_public_api_cleanup_audit.md`，把 runner 归类为 Stage 1 migration bridge。
   - 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P19a 完成内容和明确不做范围。
   - 更新 `WORKSPACE_STRUCTURE.md`，登记新增 runner、fixture、smoke 和文档。

## 结果

新增 P19a smoke、P17a/P17b/P17c/P17d 回归 smoke、P18b scaffold cleanup smoke、
compileall、`git diff --check` 和边界审计均已通过。

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_chain_runner.py time_router/features/__init__.py tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
git diff --check
git diff --name-only -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
git diff -U0 -- time_router/features tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py | rg -n "(^\\+.*(import transformers|from transformers|ViTModel|AutoImageProcessor|/data2|train_visual_router_online_streaming.py))" || true
```

输出确认：

- P13b manifest、显式 raw window fixture 和内存 ExpertBatch 可读取。
- `VisualFeatureChainRunner` 输出 canonical FeatureBatch。
- sample_key 保序、`float32`、二维、finite 和 lineage 均成立。
- FeatureBatch 可接 `LoadedTorchMLPRouterHeadAdapter + EvaluationInputAdapter`。
- P17a/P17b/P17c/P17d 和 P18b smoke 均通过，P17d manual real-artifact 环境变量缺失场景按预期 skip。
- `train_visual_router_online_streaming.py` 无 diff。
- 新增生产源码没有 `import transformers`、`from transformers`、`ViTModel`、`AutoImageProcessor`
  或 `/data2` token。

## 结论

P19a 已完成真实 Visual feature chain dry-run skeleton。当前实现只负责协议组件编排、
保序检查和 lineage 合并；fake encoder 仍留在 smoke 内部，没有进入 `time_router.features`
public core。P17 precomputed feature path 和正式 streaming 训练入口未修改，全部验收命令通过。

## 下一步方案

1. 小步提交并 push 到 `origin/refactor/stage1-route-audit`。
2. 后续可以按小步迁移真实 raw window provider、pseudo image transform 或 frozen ViT provider，但真实资源加载和路径授权仍应留在 Runtime / entrypoint 层。
