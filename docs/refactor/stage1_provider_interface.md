# Stage 1 Canonical Provider Interface Design

创建日期：2026-06-19

## 1. 目标

本文定义 P5b 阶段的 Stage 1 canonical provider interface 设计，用于把 P5a runtime contract 进一步拆成可扩展实验协议、split、专家预测、特征、router head 和评估六个边界。

本次只写设计文档，不实现 Python interface / abstract class，不新增 `time_router/protocols` 或 runtime 代码，不修改训练脚本，不迁移 Visual Router / TimeFuse-style fusor 入口，不实现 config system、run_dir helper、checkpoint index 或 logging framework，不接入 `/data2`，不移动或删除历史代码。

目标链路：

```text
ExperimentProtocol
  -> SplitStrategy
  -> ExpertProvider
  -> FeatureProvider
  -> RouterHead
  -> Evaluator
```

设计重点是未来可扩展 interface，而不是绑定当前 frozen ViT 或 17 维 TimeFuse feature cache 实现。当前 Stage 1 默认实现可以继续是 fixed config、fixed five experts、prediction cache、vali train-test eval，但共享 contract 不能把这些默认值写死成唯一形态。

## 2. 设计原则

1. **protocol 不等于脚本**：`ExperimentProtocol` 描述一次实验如何组合 split、expert、feature、head、evaluator 和 runtime contract，不代表某个现有训练脚本。
2. **provider 不决定 run_dir**：provider interface 不命名、不创建、不硬编码 full-scale `run_dir`；未来 launcher/runtime 显式传入输出目录，通常可以位于 `/data2/syh/Time/...`，但 interface 中不得硬编码 `/data2` 或 repo 内路径。
3. **shared contract 稳定，extra 可扩展**：共享字段必须跨 Visual Router 与 TimeFuse-style fusor 语义一致；分支实现可放 `branch_specific` / `extra`，但不能污染共享必需字段。
4. **cache 是一种 provider 实现，不是接口本身**：`packed_npy_v1` prediction cache、17 维 feature cache、离线 ViT embedding cache 都只能作为实现选项；interface 必须允许 online computation、finetune、joint training 和新 encoder/head/evaluator。
5. **oracle/TSF 使用边界清晰**：oracle label 和 TSF metadata 可用于监督、诊断、baseline 或分层分析；不能误作为可部署 Visual Router 在 test-time 动态调权的特征。
6. **评估从显式输入复算**：`Evaluator` 只消费 sample_key、专家预测、真实值、权重/logits 和专家列顺序，不读取 legacy 输出目录，也不依赖历史 CSV schema。

## 3. ExperimentProtocol

`ExperimentProtocol` 是一次实验的稳定描述层，负责绑定以下组件：

- `runtime_contract`：P5a 定义的 `run_dir/status/metadata/checkpoint/evaluation/logs` 契约版本。
- `split_strategy`：训练、验证、测试、held-out cell 或 cross-cell 的 split 选择。
- `expert_provider`：专家预测和 `y_true` 来源。
- `feature_provider`：router/fusor 特征来源。
- `router_head`：把 feature 转成专家 logits/weights 的 head。
- `evaluator`：统一指标、逐样本 rows、comparison 和 calibration-ready object。

最小共享字段建议：

| 字段 | 说明 |
| --- | --- |
| `protocol_name` | 实验协议名，例如 `stage1_vali_test_router_v1` |
| `protocol_version` | 协议版本，独立于脚本文件名 |
| `stage` | 例如 `stage1_vali_test_router` |
| `config_name` | 当前 TSF config，例如 `96_48_S` |
| `model_columns` | 专家动作空间顺序；当前默认五专家，但未来可扩展 |
| `runtime_contract_version` | P5a runtime contract 版本 |
| `split_strategy` | split provider 规格或引用 |
| `expert_provider` | expert provider 规格或引用 |
| `feature_provider` | feature provider 规格或引用 |
| `router_head` | head 规格或引用 |
| `evaluator` | evaluator 规格或引用 |
| `branch_specific` | Visual / TimeFuse / future branch extra |

边界：

- 不直接读写 full-scale 输出目录。
- 不保存 mutable 训练进度；进度属于 `status.json`。
- 不代替 CLI/config system；未来 config 只负责生成或选择 protocol。
- 不为历史 pilot/OOM/legacy output schema 增加 adapter。

## 4. SplitStrategy

`SplitStrategy` 负责把样本空间划分为训练、验证、测试或泛化评估集合，并把 split 约束下推到需要它的 provider 和 evaluator。

当前默认：

- router/fusor 训练使用 vali split。
- router/fusor 评估使用 test split。
- prediction cache 与 sample manifest 中已有 split 字段。

未来必须兼容：

- cell holdout：按 TSF cell 或其它稳定 cell 元信息留出。
- cross-cell generalization：训练 cell 集合和评估 cell 集合显式分离。
- branch-specific split：例如某些 feature provider 只能在训练 split 做 scaler fit，评估 split 只 transform。

下推边界：

| 接收方 | SplitStrategy 下推内容 | 不应下推内容 |
| --- | --- | --- |
| `ExpertProvider` | 需要读取哪些 split 的 sample_key、专家预测和 `y_true` | router feature 或特征 scaler 策略 |
| `FeatureProvider` | 需要为哪些 sample_key 生成/读取 feature；训练 split 可用于 scaler/normalizer fit | 专家预测、oracle top-1 或未来 `y` |
| `RouterHead` | 训练循环传入的 batch 已经按 split 组织 | split 文件路径、manifest scan 策略 |
| `Evaluator` | 当前评估 split 名称、样本集合、分层汇总维度 | 训练阶段如何采样或优化 |

`SplitStrategy` 不负责读取 prediction arrays，也不负责创建 feature cache。它只定义 split 语义、样本集合和下推规则。

## 5. ExpertProvider

`ExpertProvider` 负责提供专家动作空间的预测、共享真实值和行级 lineage。

当前默认实现可以由 `PredictionBatchReader + packed_npy_v1 prediction cache` 承担，但 interface 不假设专家一定来自离线 cache。未来应兼容：

- online expert prediction；
- router + expert joint training；
- 更大或不同顺序的专家池；
- 不同 array storage 或服务化 expert backend。

最小输出：

| 输出 | 形态 | 说明 |
| --- | --- | --- |
| `sample_keys` | list/string array | 与输入 batch 保序 |
| `model_columns` | list[string] | 专家动作空间顺序 |
| `y_pred` | tensor/array | `[sample, expert, pred_len, channel]` 或 protocol 明确的等价结构 |
| `y_true` | tensor/array | `[sample, pred_len, channel]`，同一 sample 对所有专家共享 |
| `row_index_metadata` | object/table | `y_pred_row_index`、`y_true_row_index`、shard/path/source 等 lineage |
| `expert_metadata` | object | storage、dtype、source、online/offline、provider version |

边界：

- 不负责 router/fusor feature。
- 不读取 oracle/TSF 来决定 test-time 权重。
- 不写 evaluation output。
- 不决定 checkpoint 或 run_dir。
- 可以暴露 cache lineage，但不能要求所有实现都有 `packed_npy_v1`。

## 6. FeatureProvider

`FeatureProvider` 负责把同一批 sample 转成 `RouterHead` 可消费的 feature tensor 或 structured feature。

当前两条默认实现：

1. Visual Router 默认：`window/sample -> pseudo image -> ViT feature`。
2. TimeFuse-style fusor 默认：`sample_key -> TimeFuse feature cache -> feature tensor`。

interface 必须允许：

- frozen encoder；
- finetuned encoder；
- router + encoder joint training；
- router + expert joint training；
- offline feature cache；
- online feature computation；
- TimeFuse feature cache；
- online TimeFuse feature computation；
- branch-specific feature schema。

最小输出：

| 输出 | 说明 |
| --- | --- |
| `sample_keys` | 与 ExpertProvider batch 保序，缺失或重排必须显式报错 |
| `features` | tensor、array 或 structured object；shape/dtype/device 在 metadata 中说明 |
| `feature_schema` | schema name、feature columns、embedding dim、normalization、online/offline、cache/source |
| `provider_state` | scaler/encoder/checkpoint/cache index 等可恢复或可诊断信息 |
| `diagnostics` | 可选 latency、missing feature、fallback count、device stats |

Visual branch 共享约束：

- 当前主线可以是 frozen ViT，但 interface 不把 frozen 作为唯一合法状态。
- pseudo image 与 ViT embedding 可在线生成，也可在未来小规模 debug 中离线缓存；full-scale 主线仍不长期保存伪图像 tensor 或 ViT embedding。
- finetune ViT 或 joint training 时，encoder checkpoint 属于 `FeatureProvider` / training runtime 的 branch-specific state，而不是共享 `ExpertProvider`。

TimeFuse branch 共享约束：

- 当前默认 17 维 feature cache 只是 `feature_schema_name=timefuse_single_variable_meta_v1` 的一种实现。
- scaler fit 可由训练 split 驱动，eval/test 只 transform。
- 未来可以替换为 online TimeFuse feature computation，但仍不能读取专家预测、oracle top-1 或未来 `y` 作为可部署动态调权特征。

oracle/TSF 边界：

- `OracleTsfReader` 输出可用于监督 label、oracle regret、baseline、分层 summary 和诊断。
- TSF cell 可用于 cell holdout / cross-cell split 定义和分层分析。
- oracle/TSF 不进入可部署 Visual Router test-time `FeatureProvider` 动态特征；如果某个 baseline 故意使用 TSF metadata，必须在 protocol 中标记为 diagnostic/baseline，而非 deployable visual router。

## 7. RouterHead

`RouterHead` 负责把 `FeatureProvider` 输出转成专家 logits/weights。

当前可支持：

- Visual Router MLP head；
- TimeFuse linear-softmax fusor；
- calibration 前的 raw soft weights；
- hard top-1 选择所需 logits/weights。

未来可替换为：

- 更深 MLP、attention head、mixture head；
- 校准感知 head；
- 多任务 head；
- joint encoder/head 或 expert-aware head。

最小输入输出：

| 项 | 说明 |
| --- | --- |
| input | `features` 或 structured feature；不直接读取 prediction cache |
| output | `logits` 和/或 `weights`，专家维度与 `model_columns` 对齐 |
| metadata | head name、version、input schema、normalization、loss/call mode |

边界：

- 不读取 prediction cache。
- 不写 evaluation output。
- 不创建 run_dir 或 checkpoint index。
- 可持有可训练参数，checkpoint 保存由 training/runtime 层负责。
- loss 可属于 training branch-specific 逻辑，但 evaluation 仍通过统一 `Evaluator` 从 weights/logits 复算。

## 8. Evaluator

`Evaluator` 负责从显式输入复算指标和报告对象。

最小输入：

- `sample_keys`
- `y_pred`
- `y_true`
- `weights` 或 `logits`
- `model_columns`
- 可选 `split_name`
- 可选 oracle/TSF diagnostic tables

应使用 `time_router.evaluation` public API 作为基础能力，统一输出：

| 输出 | 说明 |
| --- | --- |
| `summary` | hard top-1、raw soft、MAE/MSE、selected counts、weight diagnostics |
| `per_sample_rows` | sample_key、选择专家、逐样本 hard/raw-soft 指标、权重诊断 |
| `comparison` | 与 best single、oracle、baseline、calibration 策略的同表比较 |
| `calibration_ready` | 温度、top-k、校准或后处理可继续消费的结构化对象 |
| `diagnostics` | split/cell/source/provider 维度的诊断信息 |

边界：

- 不依赖 legacy output schema。
- 不读取训练入口私有 metadata 来推断专家顺序；专家顺序由显式 `model_columns` 输入提供。
- 不回写 prediction cache。
- 不决定 split，只消费 `SplitStrategy` 给出的评估集合和分层元信息。

## 9. Runtime / run_dir 关系

provider interface 与 P5a runtime contract 的关系是“被 runtime 编排”，不是“决定 runtime”。

- `ExperimentProtocol` 可以引用 runtime contract version，但不创建目录。
- `SplitStrategy`、`ExpertProvider`、`FeatureProvider`、`RouterHead`、`Evaluator` 都不硬编码输出根。
- full-scale `run_dir` 未来由 launcher/runtime 显式传入，通常在 `/data2/syh/Time/...`；本接口文档不把任何绝对路径写成默认值。
- `metadata.json` 应记录实际 provider specs、feature schema、expert source、split strategy、head name、evaluator schema 和 branch-specific extra。
- `status.json` 只记录运行状态，不承载 provider interface 的全部定义。

## 10. 共享主干与分支边界

共享 contract：

- `sample_key` 保序；
- `model_columns` 专家动作空间；
- `y_pred/y_true` 形态和共享 `y_true` 约束；
- split 语义和 split 下推规则；
- feature output 与 feature schema metadata；
- logits/weights 与专家维度对齐；
- evaluation 从显式数组和权重复算；
- provider 不决定 run_dir。

Visual Router branch-specific：

- pseudo image preset、view/fold 策略；
- ViT/encoder model、normalization、frozen/finetuned/joint mode；
- embedding dtype/device/online cache policy；
- MLP/router loss、KL/Huber 参数；
- 视觉特征诊断 latency。

TimeFuse-style fusor branch-specific：

- TimeFuse feature schema name 和 feature columns；
- feature cache shard/index/scaler；
- online feature computation backend；
- linear-softmax head、SmoothL1 参数；
- packed prediction reader worker/prefetch 策略。

Deprecated/reference-only，不再适配新 interface：

- LogisticRegression hard-label structure router；
- offline ViT embedding full-scale cache 路线；
- 旧 OOM full-scale lookup 路线；
- pilot-only 固定 120/1k/dry-run launcher；
- 非 streaming full-scale 入口；
- 为历史 status/metadata/checkpoint/output schema 反向新增 adapter。

## 11. 未来兼容性矩阵

| 未来能力 | 主要扩展点 | 当前接口要求 |
| --- | --- | --- |
| cell holdout | `SplitStrategy` | split 可下推到 expert/feature/evaluator |
| cross-cell generalization | `SplitStrategy` + evaluator diagnostics | train/eval cell 集合显式记录 |
| finetune ViT | `FeatureProvider` | encoder trainability 属于 branch-specific state |
| router + ViT joint training | `FeatureProvider` + `RouterHead` + training runtime | feature provider 可持有可训练 encoder |
| router + expert joint training | `ExpertProvider` + `RouterHead` + training runtime | expert prediction 不要求来自离线 cache |
| online expert prediction | `ExpertProvider` | source 可为 online backend |
| TimeFuse feature cache | `FeatureProvider` | feature cache 是实现，不是接口唯一形态 |
| online TimeFuse feature computation | `FeatureProvider` | feature schema 记录 online/source/latency |
| 新 router head | `RouterHead` | 只需保持 expert 维度对齐 |
| calibration / evaluator 扩展 | `Evaluator` | 使用显式 weights/logits/y_pred/y_true 生成结构化输出 |

## 12. 迁移门禁

P5b 只做 interface design，不迁移代码。后续若实现或迁移入口，至少需要：

1. 先为小规模 smoke 写出 protocol/provider metadata。
2. 用 `PredictionBatchReader` 或等价 streaming provider 证明 `sample_key/model_columns/y_pred/y_true` 契约不漂移。
3. 用 Visual 小规模 smoke 证明 online pseudo image / encoder 输出与现有入口一致。
4. 用 TimeFuse 小规模或 pressure smoke 证明 feature schema、split 下推、scaler 和 packed npy batch 读取不退化。
5. 使用 `time_router.evaluation` public API 复算 summary/rows/comparison。
6. 运行 golden smoke、oracle/TSF smoke、P4 helper smoke 和 compileall。
7. full-scale 必须新建 run_dir，不覆盖旧可引用结果。

## 13. 本次明确不做

- 不实现 Python interface / abstract class。
- 不新增 `time_router/protocols` 或 runtime 代码。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 config system。
- 不实现 run_dir helper。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不为了兼容历史输出新增 helper。
- 不改变 `PredictionBatchReader` / `OracleTsfReader` / evaluation / IO helper 行为。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。
