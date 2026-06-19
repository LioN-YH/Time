# Stage 1 Protocol Types Skeleton

创建日期：2026-06-19

## 1. 目标

本文记录 P5c 阶段新增的最小 protocol dataclass 类型骨架。该骨架基于 P5b provider interface design，只提供 lightweight contract container，用于后续在 `ExperimentProtocol -> SplitStrategy -> ExpertProvider -> FeatureProvider -> RouterHead -> Evaluator` 链路之间传递显式对象。

本次允许新增少量代码，但只限类型骨架、smoke、文档和日志；不实现训练逻辑、不实现 Python abstract base class、不做文件 IO、不绑定 numpy/torch/pandas/sklearn、不迁移 Visual Router 或 TimeFuse-style fusor 正式入口。

## 2. Public API

P5c 新增 `time_router/protocols/`：

| 路径 | 作用 |
| --- | --- |
| `time_router/protocols/types.py` | 定义最小 dataclass 类型骨架 |
| `time_router/protocols/__init__.py` | 从 `time_router.protocols` 导出稳定 public API |

公开导出的类型：

- `SplitSpec`
- `ExpertBatch`
- `FeatureBatch`
- `RouterOutput`
- `EvaluationInput`
- `ExperimentProtocolSpec`

正式入口后续应优先从 `time_router.protocols` 导入这些类型，而不是依赖深层文件路径。

## 3. 类型边界

### 3.1 `SplitSpec`

字段：

- `name: str`
- `train_splits: tuple[str, ...]`
- `eval_splits: tuple[str, ...]`
- `extra: dict[str, Any] = field(default_factory=dict)`

用途：保存 split strategy 的轻量规格。它不读取 manifest，不解析 sample_key，不创建 split 文件，也不决定 provider 如何扫描数据。

### 3.2 `ExpertBatch`

字段：

- `sample_keys: tuple[str, ...]`
- `model_columns: tuple[str, ...]`
- `y_pred: Any`
- `y_true: Any`
- `row_index_metadata: Any | None = None`
- `extra: dict[str, Any] = field(default_factory=dict)`

用途：保存 ExpertProvider 输出的专家预测、共享真实值和可选 row lineage。`y_pred` / `y_true` 统一使用 `Any`，P5c 不访问 `.shape`，不检查五专家完整性，不复算 MAE/MSE，不假设数据来自 `packed_npy_v1`。

### 3.3 `FeatureBatch`

字段：

- `sample_keys: tuple[str, ...]`
- `features: Any`
- `feature_schema: dict[str, Any] = field(default_factory=dict)`
- `extra: dict[str, Any] = field(default_factory=dict)`

用途：保存 FeatureProvider 输出的特征和 schema metadata。它不执行 pseudo image、ViT encoder、TimeFuse feature cache 读取、scaler fit 或 online computation。

### 3.4 `RouterOutput`

字段：

- `sample_keys: tuple[str, ...]`
- `model_columns: tuple[str, ...]`
- `logits: Any | None = None`
- `weights: Any | None = None`
- `extra: dict[str, Any] = field(default_factory=dict)`

用途：保存 RouterHead 输出。`logits` 与 `weights` 都保留为可选字段；P5c 不强制至少一个存在，避免把 contract container 变成业务 validator。后续训练或 evaluator 层可做语义校验。

### 3.5 `EvaluationInput`

字段：

- `sample_keys: tuple[str, ...]`
- `model_columns: tuple[str, ...]`
- `y_pred: Any`
- `y_true: Any`
- `logits: Any | None = None`
- `weights: Any | None = None`
- `extra: dict[str, Any] = field(default_factory=dict)`

用途：保存 Evaluator 复算指标所需的显式输入。它同时保留 logits 和 weights，兼容 future calibration、temperature scaling 与 raw logits analysis；P5c 不计算 fusion，不写 summary/rows 文件。

### 3.6 `ExperimentProtocolSpec`

字段：

- `protocol_name: str`
- `protocol_version: str`
- `stage: str`
- `config_name: str`
- `model_columns: tuple[str, ...]`
- `runtime_contract_version: str`
- `split_strategy: Any`
- `expert_provider: Any`
- `feature_provider: Any`
- `router_head: Any`
- `evaluator: Any`
- `branch_specific: dict[str, Any] = field(default_factory=dict)`
- `extra: dict[str, Any] = field(default_factory=dict)`

用途：保存一次实验 protocol 的轻量规格。`split_strategy`、`expert_provider`、`feature_provider`、`router_head` 和 `evaluator` 字段是 spec、引用或配置描述，不是真实 provider 对象。该类型不实例化 provider，不读取路径，不创建 `run_dir`。

## 4. 统一约束

1. 使用 `dataclass + typing`，不实现 abstract base class。
2. 不引入 numpy、torch、pandas、sklearn。
3. array/tensor-like 字段统一使用 `Any`。
4. 不访问 `.shape`，不做数值、shape、finite 或专家维度校验。
5. `sample_keys`、`model_columns`、`train_splits`、`eval_splits` 稳定使用 tuple。
6. `extra`、`branch_specific` 和 `feature_schema` 使用 `field(default_factory=dict)`，避免共享默认 dict。
7. 不自动解析 `Path` / `os.PathLike`，不访问文件系统。
8. 不包含 `run_dir`；`run_dir` 属于 runtime contract，不属于 provider 类型。
9. provider/head/evaluator 字段只保存 spec 或引用描述，不保存真实 provider 对象。
10. public API 从 `time_router.protocols` 导出。

## 5. Smoke 覆盖

新增 `tests/smoke/stage1_protocol_types_smoke.py`，覆盖：

- 从 `time_router.protocols` 导入 public API；
- 构造全部 6 个 dataclass；
- tuple 顺序保持；
- `extra` / `branch_specific` / `feature_schema` 的 default_factory 独立性；
- `RouterOutput` 和 `EvaluationInput` 支持 weights-only、logits-only、两者都有和都为空；
- array/tensor 字段可以保存普通 object/list，且不会访问 `.shape`；
- 不创建文件，不访问 `/data2`，不访问正式输出目录。

## 6. 明确不做

- 不实现 Python abstract base class。
- 不实现 FeatureProvider / ExpertProvider 读取逻辑。
- 不实现 ExperimentProtocol 执行逻辑。
- 不实现 runtime / run_dir helper。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不改 PredictionBatchReader / OracleTsfReader / evaluation / io helper。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

## 7. 验收命令

P5c 的验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

本阶段通过这些 smoke 后，只能说明 protocol 类型骨架和既有低风险 helper 没有被破坏；不代表 provider 执行链、正式 runtime 或 full-scale 入口已经迁移。
