# Stage 1 P8d TimeFuse Baseline Parity Review

创建日期：2026-06-20

## 1. 目标

本文在 P8c evaluation adapter pressure verification 之后，审计当前 Stage 1 TimeFuse-style fusor baseline 与原版 TimeFuse 思路之间的 parity 边界。

本步只做文档审计，不修改 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`，不修改 `TimeFuseFusor`、reader、scaler、loss 或 evaluation adapter，不新增 smoke、Bash 或实验运行。

## 2. 当前 baseline 保留的 TimeFuse-style 核心

当前 baseline 保留的是 TimeFuse 中“基于元特征动态生成专家融合权重”的核心结构，而不是原论文完整实验协议。

已保留的核心如下：

- `meta feature -> linear logits`：`TimeFuseFusor` 使用 `torch.nn.Linear(input_dim, output_dim)` 将 TimeFuse-derived meta feature 映射到专家 logits。
- `softmax expert weights`：`TimeFuseFusor.forward(...)` 对 logits 沿专家维度执行 `torch.softmax(logits, dim=-1)`，得到每个样本的五专家权重。
- `sample-level adaptive fusion`：每个 sample 的 feature 独立前向，产生 sample-level 权重；不是全局固定专家权重。
- `weighted prediction fusion`：训练和 evaluation 均按当前样本权重对五专家 `y_pred` 加权求和，得到 fused prediction。
- `SmoothL1Loss` 训练口径：streaming 训练入口使用 `nn.SmoothL1Loss(beta=args.huber_beta)`，默认 `--huber-beta 0.01`，监督目标是 fused prediction 与共享 `y_true`。

这些细节在当前正式入口和共享 fusor 工具中均可定位：

- `visual_router_experiments/stage1_vali_test_router/fusion_utils.py` 中 `TimeFuseFusor` 定义 `nn.Linear -> torch.softmax`。
- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` 中 `train_streaming(...)` 使用 `SmoothL1Loss`，并通过 `broadcast_weights(weights, batch_pred) * batch_pred` 做 weighted fusion loss。
- 同一入口的 `evaluate_streaming(...)` 使用当前 batch 权重输出 hard top-1 与 raw soft weighted fusion 指标。

## 3. 当前 baseline 有意改变的部分

当前实现是面向 Time 工作区 Stage 1 / QuitoBench 五专家 prediction cache 的适配版本，以下差异是有意设计，不应被描述为缺口修复之外的“无改动复现”。

| 维度 | 原版 TimeFuse 思路 | 当前 Stage 1 baseline |
| --- | --- | --- |
| 数据粒度 | 多变量时间序列场景下的自适应融合 | 单变量 sample-level window 融合 |
| 元特征 | 原版 TimeFuse 特征定义 | 当前 17 维 `timefuse_single_variable_meta_v1` / TimeFuse-derived feature |
| 专家集合 | 原版论文实验中的专家/模型集合 | QuitoBench 当前五专家：`DLinear`、`PatchTST`、`CrossFormer`、`ES`、`NaiveForecaster` |
| prediction 来源 | 原版流程中的专家预测 | Stage 1 packed prediction cache / streaming reader |
| 评估对象 | 原版论文评估协议 | Time 工作区 `ExpertBatch` / `EvaluationInputAdapter` 对齐的 hard top-1 与 raw soft fusion 口径 |
| 训练/评估运行方式 | 原版实现运行协议 | shard-aware streaming train/eval，支持 feature-only scaler、packed npy grouped loading 和正式 runtime 状态落盘 |

这些改变的目的，是让 TimeFuse-style 动态加权思想接入当前 Stage 1 的 canonical prediction cache、五专家动作空间和 evaluation contract，作为 Visual Router 的非视觉自适应融合 baseline。

## 4. 不能声称的内容

后续论文、README、实验日志和汇总表中不应使用以下表述：

- 不能声称当前 baseline 完全复现原版 TimeFuse。
- 不能声称当前数值可与原版 TimeFuse 论文数值一一复现比较。
- 不能把当前 baseline 说成未改造的 TimeFuse。
- 不能省略“single-variable / QuitoBench / Stage 1 prediction cache / adapted”这类边界后直接写成原版 TimeFuse 结果。

如果需要引用，应明确这是受 TimeFuse 启发、在当前单变量 QuitoBench 五专家 routing/fusion 设置中改造后的 baseline。

## 5. 可以声称的内容

以下表述与当前实现边界一致：

- `TimeFuse-style fusor baseline`
- `TimeFuse-inspired sample-level adaptive expert fusion baseline`
- `adapted TimeFuse-style baseline for single-variable QuitoBench expert routing`
- `linear-softmax sample-level expert fusion baseline using TimeFuse-derived features`

推荐默认使用 `TimeFuse-style fusor baseline`。当需要更严谨地区分原版论文时，使用 `adapted TimeFuse-style baseline for single-variable QuitoBench expert routing`。

## 6. 对照正式入口已保持的实现细节

对照 `train_timefuse_fusor_streaming.py`，当前正式入口已保持以下 parity-relevant 细节：

| 细节 | 当前证据 | 结论 |
| --- | --- | --- |
| torch Linear -> softmax | `TimeFuseFusor` 来自 `fusion_utils.py`，入口用 `TimeFuseFusor(input_dim=len(feature_cols), output_dim=len(MODEL_COLUMNS))` | 已保持 linear-softmax 动态权重核心 |
| `StandardScaler` | `fit_scaler_streaming(...)` 只在 vali feature streaming 上 `partial_fit`，checkpoint 可保存/恢复 scaler state | 已保持训练特征标准化口径，且避免读取 prediction arrays 做 scaler |
| `SmoothL1Loss beta` | CLI 默认 `--huber-beta 0.01`，`train_streaming(...)` 使用 `nn.SmoothL1Loss(beta=float(args.huber_beta))` | 已保持 SmoothL1/Huber-style fusion loss 口径 |
| weighted fusion loss | `train_streaming(...)` 对 `weights` 广播后与 `batch_pred` 相乘求和，再对 `batch_true` 计算 loss | 已保持基于融合预测而非 hard label 的训练目标 |
| streaming train/eval | `iter_reader_batches(...)` 分别按 vali/test batch 读取 feature、五专家 `y_pred` 和共享 `y_true`；`evaluate_streaming(...)` 流式写出 summary 输入 | 已接入 Stage 1 full-scale streaming reader/evaluation |
| evaluation adapter 对齐 | P8b/P8c 已验证 `--verify-evaluation-adapter` 对 batch 旁路复算不改变正式 CSV 输出 | 当前 evaluation 口径已可与 `EvaluationInputAdapter` 对齐 |

## 7. 后续更强 parity 需要再审的内容

如果未来需要把当前 baseline 从 “TimeFuse-style / inspired” 提升为更强 parity claim，应另起审计并至少补齐以下证据：

- 原版 TimeFuse 特征定义、每个特征的数学含义和计算窗口。
- 原版训练 split、loss、normalization、batching、optimizer 和 early stopping 口径。
- 原版多变量处理方式，以及多变量到当前单变量 sample-level 的严格映射。
- 当前 17 维 TimeFuse-derived feature 与原版 feature 的逐项对应关系。
- 原版专家集合、专家预测生成方式和当前 QuitoBench 五专家 prediction cache 的可比性。
- 原版 evaluation metric、aggregation 粒度和当前 hard top-1 / raw soft fusion summary 的差异。

在上述审计完成前，当前结果只能作为改造后的 TimeFuse-style baseline 引用。

## 8. 本步明确不做

- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `TimeFuseFusor`、reader、scaler、loss 或 evaluation adapter。
- 不新增 smoke。
- 不新增 Bash 或 scripts。
- 不访问 `/data2`。
- 不启动 pressure 或 full-scale。
- 不改 Visual Router 入口。
