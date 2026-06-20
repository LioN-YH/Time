# Stage 1 P15a Branch-Specific Small Entrypoint Decision

创建日期：2026-06-21

## 1. 目标

P15a 在 P13d/P13e/P14a-P14f 之后，只做 branch-specific small entrypoint 的文档决策。
本阶段不新增 `scripts/run_stage1_timefuse_small.py`，不新增
`scripts/run_stage1_visual_small.py`，不修改 `scripts/run_stage1_canonical_small.py`，
不迁移正式 Visual Router 或 TimeFuse-style fusor 训练入口，不访问 `/data2`，不读取真实
checkpoint，不启动训练、pressure 或 full-scale。

当前 Stage 1 canonical dataflow 仍为：

```text
SampleManifest + SplitStrategy
  -> ordered sample_keys
  -> ExpertProvider / prediction backend
  -> SupervisionProvider
  -> FeatureProvider
  -> RouterHead
  -> EvaluationInputAdapter / Evaluator
  -> Runtime / artifact writer
```

P15a 的核心结论是：P14 可以收束；generic small CLI 必须继续保持 thin；后续需要分别新增
TimeFuse-specific 和 Visual-specific small canonical entrypoint，但都不在 P15a 实现。

## 2. P14 收束判断

P14 可以收束。

理由如下：

- P14a 已完成 Visual FeatureProvider 插入点审计，明确 Visual history window、pseudo image、
  frozen ViT、MLP head、SQLite prediction path、device/dtype/DataParallel、latency 与
  checkpoint/resume 的边界。
- P14b 已完成 `VisualMockFeatureProvider` smoke，证明 Visual 分支可以先以 mock provider
  输出 head-ready `FeatureBatch`，并保持 sample_key 保序和 provider 不读取 prediction /
  oracle / run_dir 的边界。
- P14c 已冻结 Visual eval-only canonical bypass plan，明确 future eval-only 链路从
  `FeatureBatch`、`ExpertBatch`、`RouterOutput` 到 `EvaluationInputAdapter` 的插入关系。
- P14d 已完成 Visual mock protocol eval smoke，证明
  `FeatureBatch + ExpertBatch -> smoke-only RouterHead -> RouterOutput -> EvaluationInputAdapter`
  可在内存中串通。
- P14e 已完成 Visual legacy MLP adapter audit，明确 legacy `VisualMLPRouter` eval-only
  thin adapter 的输入、输出、checkpoint/scaler/runtime 边界。
- P14f 已完成 smoke-only Visual legacy MLP adapter smoke，证明未来正式 adapter pattern
  可以消费 canonical `FeatureBatch` / `ExpertBatch`，输出 `RouterOutput`，并进入
  `EvaluationInputAdapter`。

P14f 只验证未来正式 Visual legacy MLP adapter 的插入点，不代表正式 `VisualMLPRouter`
已经迁移完成。P14f 的 `SmokeOnlyLegacyMLPAdapter` 定义在 smoke 内，不能直接提升为正式
adapter；正式 legacy Visual MLP adapter 仍需要单独设计、单独 smoke。

## 3. Generic Small CLI 必须保持 Thin

`scripts/run_stage1_canonical_small.py` 继续只服务通用 tiny fixture 和 canonical dataflow 的
最小验证。它的责任是：

- 接收 tiny 或 fixture-driven `SampleManifest`、feature fixture、expert fixture 和输出参数；
- 验证 manifest row order、feature/expert join、canonical protocol object 串联和 Runtime
  artifact writer 写出；
- 作为 branch-neutral 的最小回归入口，证明 canonical dataflow 的公共骨架可运行。

它不应承载：

- Visual legacy MLP、ViT embedding、pseudo image、`SQLitePredictionIndex` 或真实
  Visual feature/provider/head 组合逻辑；
- TimeFuse 17 维 feature cache、oracle parquet、shard-local SQLite、linear-softmax
  fusor 或 full-scale reader 组合逻辑；
- 任何 branch-specific provider/head 绑定、runtime resource policy、Bash launcher、
  `/data2` 路径策略或 cache prepare 策略。

如果把 Visual / TimeFuse 的 provider/head 细节塞回 generic small CLI，会重新制造入口级耦合，
违背 unified canonical dataflow + branch-specific implementations 的分层目标。generic small
CLI 应继续是公共协议骨架的 thin slice，而不是两条正式路线的兼容容器。

## 4. TimeFuse-Specific Small Entrypoint 决策

结论：需要新增 TimeFuse-specific small canonical entrypoint，但不在 P15a 实现。

推荐未来入口名称：

```text
scripts/run_stage1_timefuse_small.py
```

推荐作为 P15b 的 thin slice，原因如下：

- TimeFuse-style fusor 是正式 baseline 支线，不是临时 smoke 或历史附录。
- P13e 已证明 17 维 `TimeFuseFeatureCacheProvider` 可以从 small feature CSV 输出
  canonical `FeatureBatch`。
- 现有 `TimeFuseLinearSoftmaxHead` / protocol chain smoke 已验证 TimeFuse head 与
  canonical evaluator 的基本连接口径。
- TimeFuse 侧 feature/head/runtime 组合相对稳定，适合优先形成 branch-specific small
  entrypoint，作为 full-scale fusor / pressure 前的稳定验证入口。

未来 P15b 可串联：

```text
SampleManifest
  -> prediction backend / ExpertBatch
  -> TimeFuse 17-dim FeatureProvider
  -> TimeFuseLinearSoftmaxHead
  -> EvaluationInputAdapter / Evaluator
  -> Runtime artifact writer
```

P15b 的边界应保持：

- 使用已有 small fixture 或 real-derived small input；
- 写 canonical `run_dir`；
- 不访问 `/data2`；
- 不启动正式训练；
- 不把 feature cache path 设计成长期接口；
- 不把 Bash、launcher 或 full-scale path policy 引入 `time_router`。

## 5. Visual-Specific Small Entrypoint 决策

结论：需要新增 Visual-specific small canonical entrypoint，但应分阶段推进，不在 P15a 实现。

推荐未来入口名称：

```text
scripts/run_stage1_visual_small.py
```

Visual Router 是当前主实验。P14b/P14d/P14f 已证明 `VisualMockFeatureProvider`、mock protocol
eval、legacy MLP adapter pattern 的 canonical 插入点可行。但 Visual 真实部分更复杂，后续会
牵涉 legacy MLP、embedding、scaler、checkpoint、pseudo image、ViT provider 和 runtime
resource policy，因此应比 TimeFuse 更谨慎。

初期未来 P15c 可先串联：

```text
SampleManifest
  -> VisualMockFeatureProvider / future legacy embedding wrapper
  -> FeatureBatch
  -> prediction backend / ExpertBatch fixture
  -> smoke/legacy MLP adapter pattern
  -> RouterOutput
  -> EvaluationInputAdapter / Evaluator
  -> Runtime artifact writer
```

Visual-specific small entrypoint 不等于正式训练入口迁移。它只是正式迁移前的
branch-specific canonical rehearsal，用于验证 Visual branch 的 feature/head/evaluation/runtime
artifact 组合边界。P14f 的 `SmokeOnlyLegacyMLPAdapter` 不能直接提升为正式 adapter；正式 legacy
`VisualMLPRouter` adapter 应单独设计、单独 smoke，并显式处理 checkpoint/state_dict、
scaler、device/dtype/DataParallel 和 signature 边界。

## 6. 推荐后续 P15 拆分

推荐拆分如下：

1. **P15a：branch-specific small entrypoint decision（本步）**
   - 只做文档决策。
   - 明确 P14 已收束、generic small CLI 保持 thin、后续需要两条 branch-specific small
     entrypoint。
   - 不新增入口、不写 scripts、不迁移正式训练入口。

2. **P15b：TimeFuse-specific small canonical entrypoint thin slice**
   - 优先实现 `scripts/run_stage1_timefuse_small.py`。
   - 使用已有 small fixture 或 real-derived small input。
   - 串联 `SampleManifest -> ExpertBatch -> TimeFuse 17-dim FeatureProvider ->
     TimeFuseLinearSoftmaxHead -> EvaluationInputAdapter / Evaluator -> Runtime artifact writer`。
   - 写 canonical `run_dir`。
   - 不访问 `/data2`，不启动正式训练。

3. **P15c：Visual-specific small canonical entrypoint thin slice**
   - 后续实现 `scripts/run_stage1_visual_small.py`。
   - 初期先使用 `VisualMockFeatureProvider + smoke/legacy MLP adapter pattern`。
   - 写 canonical `run_dir`。
   - 不加载真实 checkpoint，不访问 `/data2`，不接真实 ViT，不迁移正式训练入口。

候选后续方向，编号不强制锁死：

- 正式 legacy `VisualMLPRouter` adapter design/smoke，单独验证 checkpoint/state_dict 边界。
- real Visual feature / embedding provider audit，决策 legacy embedding wrapper 与 future
  online ViT provider 边界。
- branch-specific run artifact parity smoke。
- entrypoint output schema check。

## 7. 明确不做

- 不新增 `scripts/run_stage1_timefuse_small.py`。
- 不新增 `scripts/run_stage1_visual_small.py`。
- 不修改 `scripts/run_stage1_canonical_small.py`。
- 不修改正式训练或 evaluation 入口。
- 不访问 `/data2`。
- 不读取真实 checkpoint。
- 不接真实 `VisualMLPRouter`。
- 不启动 ViT embedding。
- 不启动训练、pressure 或 full-scale。
- 不新增 provider/head/runtime core。
- 不把 Bash 引入 `time_router`。
- 不把 `run_dir` 传入 provider。
- 不把 cache 设计成 interface。
- 不为兼容旧版 `96_48_S` 输出 schema 写适配逻辑。

## 8. 验收

P15a 是 docs-only 决策步。验收重点不是跑训练或 smoke，而是确认文档和 diff 范围：

```bash
git diff --name-only
rg -n "P14 可以收束|generic small CLI|TimeFuse-specific|Visual-specific|P15b|P15c" docs/refactor/stage1_branch_specific_small_entrypoints.md
```

本步若不运行测试，应在中文实验日志中说明原因：本次只新增/更新文档，不新增代码、不修改脚本、
不改变 runtime 行为，因此不需要运行 Quito smoke 或 compileall。
