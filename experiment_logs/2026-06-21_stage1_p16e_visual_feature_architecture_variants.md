# Stage 1 P16e Visual feature architecture variant boundary 文档收束

日志日期：2026-06-21 11:55:46 CST

## 目的

在 P16a-P16d 已经分别完成 Visual MLP RouterHead adapter、real Visual feature provider boundary
audit、precomputed/head-ready fixture provider 和 loaded FeatureScaler transform 后，新增一份文档明确：
这些工作不是要把原版 Visual Router 的 CSV/cache/scaler 固化为唯一方案，而是为后续不同 Visual
feature architecture 提供可替换插槽。

## 背景

当前 Stage 1 canonical 重构仍未接真实 ViT，未迁移正式 Visual Router 入口，也未访问 `/data2`。
P16c 的 `VisualPrecomputedFeatureProvider` 容易被误读为正式 Visual Router 必须读 CSV 或落盘
embedding/cache；P16d 的 `LoadedFeatureScaler` 也可能被误读为正式路径必须使用 scaler。为避免后续架构探索被
过早锁死，需要文档化固定契约与可替换组件边界。

## 操作

1. 新增 `docs/refactor/stage1_visual_feature_architecture_variants.md`。
2. 文档中明确长期固定契约是：
   `SampleManifest / ordered sample_keys -> FeatureProvider / FeatureTransform -> FeatureBatch -> RouterHeadAdapter -> EvaluationInputAdapter / Evaluator -> Runtime artifact writer`。
3. 文档中明确长期不固定的是 RevIN、resize、pseudo image 口径、encoder choice、ViT pooling、CLS vs mean patch、scaler/normalizer、precompute embedding 和 cache。
4. 文档中把 Visual feature chain 拆成 `RawWindowProvider`、optional pre-image transform、`PseudoImageTransformer`、`ResizePolicy` / image input policy、`VisualEncoderProvider`、`PoolingStrategy`、optional `FeatureTransform` 和 `FeatureBatch` 等 architecture variant 插槽。
5. 文档中单独说明 P16c `VisualPrecomputedFeatureProvider` 只是 precomputed/head-ready fixture provider，可用于 smoke/debug/ablation，不代表正式 Visual Router 必须读 CSV 或落盘 embedding/cache。
6. 文档中单独说明 P16d `LoadedFeatureScaler` 只是 `FeatureBatch -> FeatureBatch` transform 的一个实现，不代表正式 Visual Router 必须使用 scaler，也不允许把 scaler 塞进 RouterHead adapter 或 provider silent fit。
7. 同步更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md`、`WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`。

## 结果

本步只新增和修改文档、实验日志与结构索引，没有新增 provider/transform/model 代码，没有修改
P16a/P16c/P16d 实现，没有修改 `scripts/run_stage1_visual_small.py` 或正式训练/evaluation 入口。

轻量验收命令：

```bash
git diff --name-only
rg -n "RevIN|resize|CLS|mean_patch|cache|FeatureBatch|FeatureTransform|VisualPrecomputedFeatureProvider|LoadedFeatureScaler" docs/refactor/stage1_visual_feature_architecture_variants.md
```

## 结论

P16e 明确了 Stage 1 Visual feature 的长期稳定边界是 canonical pipeline 契约，而不是某个具体视觉特征实现。
后续无论采用 RevIN before imaging、resize/no resize、CLS pooling、mean_patch pooling、different visual
encoder、no scaler 或 different normalizer，都应通过替换 feature-chain 组件接入现有 canonical pipeline，
而不是重写 evaluator、runtime 或 expert provider。

## 下一步方案

后续可以在该边界下继续做 fake encoder / online ViT provider audit 或 smoke、legacy checkpoint/signature
审计，以及正式 Visual entrypoint migration plan。正式迁移前仍应保持小步验证，不把 cache、scaler 或 CSV
读取路径上升为长期接口。
