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


def _axis_length(value: Any, axis: int, field_name: str) -> int:
    """
    函数功能：
        在不绑定 numpy/torch 类型的前提下读取 array-like 指定维度长度。

    输入/输出：
        优先使用 `.shape`，否则在 axis=0 时退回 `len(value)`；无法判断时抛出
        清晰错误，避免 supervision batch 在样本或专家维度上静默错位。
    """

    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) <= axis:
            raise ValueError(f"{field_name} 缺少 axis={axis} 维度，shape={shape}")
        return int(shape[axis])
    if axis == 0:
        try:
            return len(value)
        except TypeError as exc:
            raise ValueError(f"{field_name} 无法通过 len(...) 校验第一维") from exc
    try:
        first_row = value[0]
        return len(first_row)
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError(f"{field_name} 无法校验 axis={axis} 维度，请提供带 shape 的 array-like") from exc


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
class SampleManifestRow:
    """
    功能：
        描述 Stage 1 canonical SampleManifest 的单条样本身份与 lineage。

    输入/输出：
        `sample_key` 是跨 prediction、feature 和 supervision join 的稳定主键；
        `extra` 只保存非监督、非未来信息的轻量 lineage。

    关键约束：
        不读取历史 labels/feature/oracle 文件，不推导 split，不保存 oracle/error。
    """

    sample_key: str
    split: str
    config_name: str
    dataset_name: str
    item_id: int
    channel_id: int
    window_index: int
    seq_len: int | None = None
    pred_len: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleManifest:
    """
    功能：
        承载 Stage 1 canonical sample source、split 与原始顺序。

    输入/输出：
        `rows` 使用 tuple 固定调用方传入顺序；`sample_keys(...)` 按该顺序返回
        全量或单个 split 的 ordered sample_keys。

    关键约束：
        只做轻量身份和 split 校验，不 materialize CSV/Parquet/SQLite，不接正式入口。
    """

    rows: tuple[SampleManifestRow, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    def validate_unique_sample_keys(self) -> None:
        """函数功能：校验 manifest 内 sample_key 唯一，失败时给出重复 key。"""
        seen: set[str] = set()
        duplicates: list[str] = []
        for row in self.rows:
            if row.sample_key in seen:
                duplicates.append(row.sample_key)
            seen.add(row.sample_key)
        if duplicates:
            duplicate_text = ", ".join(duplicates)
            raise ValueError(f"SampleManifest sample_key 必须唯一，重复值：{duplicate_text}")

    def sample_keys(self, split: str | None = None) -> tuple[str, ...]:
        """
        函数功能：
            按 rows 原始顺序返回 ordered sample_keys，可选按 split 过滤。
        """
        if split is None:
            return tuple(row.sample_key for row in self.rows)
        return tuple(row.sample_key for row in self.rows if row.split == split)

    def split_counts(self) -> dict[str, int]:
        """函数功能：按 rows 原始扫描顺序统计 split 样本数。"""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.split] = counts.get(row.split, 0) + 1
        return counts


@dataclass
class SupervisionBatch:
    """
    功能：
        承载 SupervisionProvider 输出的 oracle/error 训练监督信息。

    输入/输出：
        `sample_keys` 和 `model_columns` 保持调用方顺序；array-like 字段使用 Any，
        允许 numpy、torch、list 或后续 structured object。

    关键约束：
        supervision 只用于训练监督、诊断、baseline 或 upper-bound，不进入
        deployable FeatureProvider；本类型不读取 prediction cache 或正式输出目录。
    """

    sample_keys: tuple[str, ...]
    model_columns: tuple[str, ...]
    metric: str
    oracle_model: Any
    oracle_value: Any
    per_model_errors: Any
    extra: dict[str, Any] = field(default_factory=dict)

    def validate_shapes(self) -> None:
        """
        函数功能：
            校验 supervision batch 的样本维和专家维与显式顺序字段对齐。

        输入/输出：
            无返回值；若 `oracle_model`、`oracle_value` 或 `per_model_errors`
            与 `sample_keys/model_columns` 维度不一致则抛出 ValueError。
        """
        expected_samples = len(self.sample_keys)
        expected_models = len(self.model_columns)

        per_model_sample_dim = _axis_length(self.per_model_errors, 0, "per_model_errors")
        per_model_expert_dim = _axis_length(self.per_model_errors, 1, "per_model_errors")
        if per_model_sample_dim != expected_samples:
            raise ValueError(
                "per_model_errors 第一维必须与 sample_keys 对齐："
                f"expected={expected_samples}, actual={per_model_sample_dim}"
            )
        if per_model_expert_dim != expected_models:
            raise ValueError(
                "per_model_errors 第二维必须与 model_columns 对齐："
                f"expected={expected_models}, actual={per_model_expert_dim}"
            )

        oracle_model_sample_dim = _axis_length(self.oracle_model, 0, "oracle_model")
        oracle_value_sample_dim = _axis_length(self.oracle_value, 0, "oracle_value")
        if oracle_model_sample_dim != expected_samples:
            raise ValueError(
                "oracle_model 第一维必须与 sample_keys 对齐："
                f"expected={expected_samples}, actual={oracle_model_sample_dim}"
            )
        if oracle_value_sample_dim != expected_samples:
            raise ValueError(
                "oracle_value 第一维必须与 sample_keys 对齐："
                f"expected={expected_samples}, actual={oracle_value_sample_dim}"
            )


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
