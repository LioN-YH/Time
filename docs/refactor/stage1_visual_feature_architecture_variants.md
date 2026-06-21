# Stage 1 Visual Feature Architecture Variant Boundary

记录日期：2026-06-21

## 1. 目的

本文收束 P16a-P16d 后的 Visual feature architecture variant 边界。当前已经完成的
`LoadedTorchMLPRouterHeadAdapter`、`VisualPrecomputedFeatureProvider` 和
`LoadedFeatureScaler` 不是要把原版 Visual Router 的 CSV、cache、scaler 或 ViT 使用方式固化为
唯一方案，而是先把长期 canonical pipeline 的插槽边界暴露出来，方便后续接入不同 Visual
feature architecture。

本步只做文档收束，不新增 provider、transform、model 代码，不修改正式入口，不访问
`/data2`，不启动 ViT、训练、pressure 或 full-scale。

## 2. 长期固定契约

Stage 1 长期固定的是数据流契约，而不是某一种视觉特征实现：

```text
SampleManifest / ordered sample_keys
  -> FeatureProvider / FeatureTransform
  -> FeatureBatch
  -> RouterHeadAdapter
  -> EvaluationInputAdapter / Evaluator
  -> Runtime artifact writer
```

这些契约固定的原因是它们定义了样本身份、保序 join、特征 batch、router 输出、评估输入和运行产物的
交接方式。后续替换视觉特征架构时，应优先保持这些交接对象稳定，避免重写 evaluator、runtime、
ExpertProvider 或 artifact writer。

长期不固定的是 Visual feature chain 内部选择，包括但不限于：

- 是否使用 RevIN。
- 是否 resize。
- pseudo image 口径。
- encoder choice。
- ViT pooling 方式。
- CLS vs mean patch。
- 是否使用 scaler / normalizer。
- 是否 precompute embedding。
- 是否使用 cache。

换言之，`FeatureBatch` 是稳定接口；生成 `FeatureBatch` 的视觉架构可以变化。

## 3. 可替换的 Visual Feature Chain 插槽

后续真实 Visual feature chain 不应被写死进一个巨型 provider。建议把可替换位置拆成以下组件：

| 插槽 | 职责 | 可替换示例 | 边界 |
| --- | --- | --- | --- |
| `RawWindowProvider` | 按 ordered sample_keys 提供历史窗口 `x` 或其轻量引用 | Quito window reader、fixture reader、future streaming reader | 只提供历史输入，不读取未来 `y`、oracle、expert error 或 prediction cache |
| Optional pre-image transform | 在成图前处理 raw window | RevIN、normalization、identity | 显式注入 state 或策略，不在 provider 内 silent fit |
| `PseudoImageTransformer` | 将时间序列窗口转换为视觉输入 | 当前 pseudo image、不同 channel/fold 口径、identity debug path | 只定义 tensor/image 语义，不负责 encoder checkpoint 或 runtime resource |
| `ResizePolicy` / image input policy | 定义视觉 encoder 输入尺寸和 preprocessing | resize、no resize、crop/pad、encoder-specific processor | 属于 architecture variant 或 encoder factory 配置，不是 evaluator contract |
| `VisualEncoderProvider` | 运行或包装视觉 encoder | frozen ViT、different visual encoder、future finetuned encoder | device/dtype/DataParallel/HF cache/retry 由 Runtime 或 encoder factory 显式管理 |
| `PoolingStrategy` | 从 encoder 输出得到 feature vector | CLS、mean_patch、mean_pool | pooling choice 必须写入 `feature_schema`，不能由 head adapter 猜测 |
| Optional `FeatureTransform` | 将 raw/pre-head feature 转为 head-ready feature | `LoadedFeatureScaler`、identity、future normalizer | state 由 Runtime/entrypoint 显式注入；transform 输出仍是 `FeatureBatch` |
| `FeatureBatch` | canonical feature 交接对象 | head-ready float32 `[sample, feature_dim]` | `sample_keys` 必须保序，schema/extra 记录轻量 lineage |

这些组件都是 architecture variant 插槽。一个实现可以把多个插槽组合在同一个小型 wrapper 中，但不应把
RevIN、pseudo image、resize、encoder、pooling、scaler、cache path 和 run artifact 写出全部揉进一个
无法替换的 provider。

## 4. P16c / P16d 的定位

### 4.1 P16c VisualPrecomputedFeatureProvider

`VisualPrecomputedFeatureProvider` 是 precomputed/head-ready fixture provider：

- 可用于 smoke、debug、ablation 和跳过 ViT 的快速验证。
- 输入是显式传入的 head-ready feature fixture，输出是 canonical `FeatureBatch`。
- 它不代表正式 Visual Router 必须读 CSV。
- 它不代表正式路径必须落盘 embedding 或 cache。
- 它不代表 precomputed CSV/cache path 是长期 `FeatureProvider` contract。

正式 online provider 仍可以直接走：

```text
raw window -> pseudo image -> visual encoder -> pooling -> FeatureBatch
```

中间不要求落盘伪图像 tensor、ViT embedding 或 CSV。

### 4.2 P16d LoadedFeatureScaler

`LoadedFeatureScaler` 是 `FeatureBatch -> FeatureBatch` transform 的一个实现：

- 它只表达 raw/pre-head feature 到 head-ready feature 的显式 transform。
- 它不代表正式 Visual Router 必须使用 scaler。
- 它不代表 scaler 应塞进 `RouterHeadAdapter`。
- 它不代表 provider 可以根据 batch silent fit。
- scaler state 应由 Runtime/entrypoint 显式注入，并在 `feature_schema` / runtime metadata 中记录来源。

后续如果某个 architecture variant 不需要 scaler，应接入 identity `FeatureTransform` 或跳过该插槽；如果使用
不同 normalizer，也应作为 `FeatureTransform` 替换，而不是改 evaluator 或 expert provider。

## 5. 与并行架构探索的关系

如果另一个分支探索出更合理的视觉特征架构，例如：

- RevIN before imaging。
- resize / no resize。
- CLS pooling。
- mean_patch pooling。
- different visual encoder。
- no scaler / different normalizer。

后续应通过替换 feature-chain 组件接入当前 canonical pipeline：

```text
SampleManifest / ordered sample_keys
  -> 新的 RawWindowProvider / pre-image transform / PseudoImageTransformer / ResizePolicy
  -> 新的 VisualEncoderProvider / PoolingStrategy / FeatureTransform
  -> FeatureBatch
  -> 既有 RouterHeadAdapter
  -> 既有 EvaluationInputAdapter / Evaluator
  -> 既有 Runtime artifact writer
```

不应因为更换 RevIN、resize、pooling、encoder 或 normalizer，就重写 evaluator/runtime/expert provider。
这也是 P16a-P16d 的主要价值：先固定 head-ready `FeatureBatch`、RouterHead adapter 和 evaluator
交接面，再允许视觉特征架构在前段自由替换。

## 6. Cache / Precomputed 边界

cache 是 implementation，不是 interface。

因此：

- precomputed CSV/cache path 不是长期 `FeatureProvider` contract。
- `VisualPrecomputedFeatureProvider` 只是 fixture/debug/ablation 路径，不是正式 online 路径要求。
- 正式 online provider 可以在 batch runtime 中直接生成 feature，不要求中间落盘。
- full-scale 若为了吞吐或恢复能力使用 cache，也必须让 cache reader 输出同一 `FeatureBatch` contract。
- cache lineage 可以记录在 `feature_schema` / `extra` / Runtime metadata 中，但 evaluator、RouterHead adapter
  和 ExpertProvider 不应依赖 cache 物理路径。

## 7. 明确不做范围

本步不做以下事项：

- 不新增 provider / transform / model 代码。
- 不修改 P16a / P16c / P16d 代码。
- 不修改 `scripts/run_stage1_visual_small.py`。
- 不修改正式训练或 evaluation 入口。
- 不访问 `/data2`。
- 不启动 ViT、训练、pressure 或 full-scale。
- 不新增 Bash launcher。
- 不声称正式 Visual Router 已迁移完成。

## 8. 验收口径

本步验收是文档级别：

```bash
git diff --name-only
rg -n "RevIN|resize|CLS|mean_patch|cache|FeatureBatch|FeatureTransform|VisualPrecomputedFeatureProvider|LoadedFeatureScaler" docs/refactor/stage1_visual_feature_architecture_variants.md
```

通过标准：

- 新增文档清楚说明 architecture variant 插槽。
- 文档明确 CSV/cache/scaler 不是长期强制方案。
- 文档明确支持后续 RevIN、resize、pooling、encoder、normalizer 等替换。
- roadmap、entrypoint migration plan、`WORKSPACE_STRUCTURE.md` 和实验日志索引已同步更新。
