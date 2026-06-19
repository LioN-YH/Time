"""
文件功能：
    Stage 1 P5c 最小 protocol dataclass 类型骨架。

关键约束：
    这些类型只作为 lightweight contract container，不承载训练逻辑、文件 IO
    或数值校验。array/tensor-like 字段统一使用 Any，避免在协议层绑定
    numpy、torch、pandas、sklearn 或具体 shape 语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitSpec:
    """
    功能：
        描述一次实验中训练与评估 split 的轻量规格。

    输入/输出：
        `train_splits` 和 `eval_splits` 使用 tuple 保持调用方传入顺序；
        `extra` 留给后续 split strategy 的非共享扩展字段。

    关键约束：
        不读取 manifest，不解析路径，不负责生成 sample_key 集合。
    """

    name: str
    train_splits: tuple[str, ...]
    eval_splits: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertBatch:
    """
    功能：
        承载 ExpertProvider 输出的专家预测、共享真实值和可选 lineage。

    输入/输出：
        `y_pred` / `y_true` 可以是任意 array/tensor-like object；协议层不访问
        `.shape`，也不检查专家维度或数值内容。

    关键约束：
        不假设预测来自 packed cache、在线 expert 或任何特定存储格式。
    """

    sample_keys: tuple[str, ...]
    model_columns: tuple[str, ...]
    y_pred: Any
    y_true: Any
    row_index_metadata: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureBatch:
    """
    功能：
        承载 FeatureProvider 输出的 router/fusor 特征和 schema 描述。

    输入/输出：
        `features` 可以是 tensor、array、list 或 structured object；
        `feature_schema` 只保存调用方提供的 schema metadata。

    关键约束：
        不读取 feature cache，不执行 scaler/encoder，不绑定 Visual 或 TimeFuse 分支。
    """

    sample_keys: tuple[str, ...]
    features: Any
    feature_schema: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterOutput:
    """
    功能：
        承载 RouterHead 输出的 logits 和/或 weights。

    输入/输出：
        `logits` 与 `weights` 都是可选字段；P5c 不强制至少一个存在，
        语义校验留给后续训练或 evaluator 层。

    关键约束：
        不检查 shape、归一化、finite 或专家列对齐。
    """

    sample_keys: tuple[str, ...]
    model_columns: tuple[str, ...]
    logits: Any | None = None
    weights: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationInput:
    """
    功能：
        承载 Evaluator 复算指标所需的最小显式输入。

    输入/输出：
        同时保留 `logits` 和 `weights`，兼容后续 calibration、temperature
        scaling 与 raw logits analysis。

    关键约束：
        不计算 fusion，不做数值或 shape 校验，不写 summary/rows 文件。
    """

    sample_keys: tuple[str, ...]
    model_columns: tuple[str, ...]
    y_pred: Any
    y_true: Any
    logits: Any | None = None
    weights: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentProtocolSpec:
    """
    功能：
        描述一次 Stage 1 protocol 如何组合 runtime contract、split、provider、
        router head 和 evaluator 的轻量规格。

    输入/输出：
        provider/head/evaluator 字段保存 spec、引用或配置描述，而不是真实
        provider 对象；`branch_specific` 用于 Visual、TimeFuse 或 future branch
        的非共享扩展。

    关键约束：
        不实例化 provider，不解析路径，不创建或保存 run_dir。
    """

    protocol_name: str
    protocol_version: str
    stage: str
    config_name: str
    model_columns: tuple[str, ...]
    runtime_contract_version: str
    split_strategy: Any
    expert_provider: Any
    feature_provider: Any
    router_head: Any
    evaluator: Any
    branch_specific: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
