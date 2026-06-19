# Stage 1 P5c protocol types skeleton

日志日期：2026-06-19 21:55:28 CST

## 目的

基于 P5b provider interface design，新增最小 protocol dataclass 类型骨架，为后续 `ExperimentProtocol -> SplitStrategy -> ExpertProvider -> FeatureProvider -> RouterHead -> Evaluator` 链路提供轻量 contract container。

## 背景

P5a 已定义 canonical runtime contract，P5b 已定义 provider interface design。本阶段只允许少量代码，用于把 P5b 的设计落成最小类型容器；不能实现训练逻辑、provider 读取、runtime helper、config system、checkpoint index、logging framework，也不能迁移 Visual Router / TimeFuse-style fusor 正式入口。

## 操作

1. 新增 `time_router/protocols/types.py`，定义 `SplitSpec`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput` 和 `ExperimentProtocolSpec`。
2. 新增 `time_router/protocols/__init__.py`，从 `time_router.protocols` 导出 public API。
3. 新增 `tests/smoke/stage1_protocol_types_smoke.py`，纯内存验证全部 dataclass 构造、tuple 保序、default_factory 独立性、logits/weights 可选组合和普通 object/list 字段原样保存。
4. 新增 `docs/refactor/protocol_types.md`，记录 P5c 类型边界、统一约束、smoke 覆盖和明确不做范围。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`，登记 P5c 状态与新增长期文件。

## 结果

- P5c dataclass 只使用 `dataclass`、`field(default_factory=dict)` 和 `typing.Any`，未引入 numpy、torch、pandas、sklearn。
- array/tensor-like 字段保持 `Any`，类型本身不访问 `.shape`，不做数值或 shape 校验。
- `sample_keys`、`model_columns`、`train_splits`、`eval_splits` 使用 tuple 保存调用方顺序。
- `RouterOutput` 与 `EvaluationInput` 保留可选 `logits` 和 `weights`，不强制至少一个存在。
- `ExperimentProtocolSpec` 不包含 `run_dir`，provider/head/evaluator 字段只作为 spec、引用或配置描述。
- smoke 不创建文件，不访问 `/data2`，不访问正式输出目录。
- 验收命令均已通过：
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke`

## 结论

P5c 只完成 protocol 类型骨架和文档化边界，未改变既有 reader、evaluation、IO helper 或训练入口行为。后续正式入口若接入 provider chain，应先从 `time_router.protocols` 导入这些轻量 contract，并在更高层实现业务语义校验。

## 下一步方案

1. 运行 P5c 指定验收命令，包括 golden/oracle/json/path/run_metadata/protocol smoke 和 `compileall`。
2. 小步提交并推送到远程 `refactor/stage1-route-audit` 分支。
3. 后续如进入 P6，应先设计小规模 provider metadata 写出与入口接入门禁，再迁移正式 streaming Visual Router / TimeFuse-style fusor 入口。
