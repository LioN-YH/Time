"""
文件功能：
    定义 Stage 1 P16f Visual feature chain 的最小协议骨架。

设计边界：
    本模块只描述 raw window、pre-image transform、pseudo image、resize/input
    policy、visual encoder、pooling 和 feature transform 之间的输入输出契约。
    它不实现真实 RevIN、normalization、pseudo image、resize、ViT、pooling
    或 scaler fit，也不绑定 cache path、checkpoint、run_dir 或正式训练入口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, TypeAlias

from time_router.protocols import FeatureBatch

SampleKeys: TypeAlias = tuple[str, ...]
LineageMetadata: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class RawWindowBatch:
    """
    类功能：
        承载按 ordered sample_keys 取出的历史窗口 payload。

    关键约束：
        `sample_keys` 顺序必须由上游 manifest / batch planner 决定并逐层保留；
        `windows` 的具体 array/tensor 类型由实现决定，协议层不做数值处理。
    """

    sample_keys: SampleKeys
    windows: Any
    metadata: LineageMetadata = field(default_factory=dict)


@dataclass(frozen=True)
class PreImageBatch:
    """
    类功能：
        承载进入 pseudo image 前的可选预处理结果。

    关键约束：
        这里可以表达 future RevIN、normalization 或 identity 的输出，但不在
        skeleton 中实现任何真实算法。
    """

    sample_keys: SampleKeys
    values: Any
    metadata: LineageMetadata = field(default_factory=dict)


@dataclass(frozen=True)
class VisualInputBatch:
    """
    类功能：
        承载 pseudo image / resize policy 后准备交给视觉 encoder 的输入。

    关键约束：
        `images` 只是视觉输入 payload，占位未来 CHW/NHWC tensor、processor
        input 或其他结构；协议层不假设具体 layout。
    """

    sample_keys: SampleKeys
    images: Any
    metadata: LineageMetadata = field(default_factory=dict)


@dataclass(frozen=True)
class VisualEmbeddingBatch:
    """
    类功能：
        承载视觉 encoder 的原始 embedding/token 输出。

    关键约束：
        该 batch 仍不是 canonical FeatureBatch；CLS/mean_patch/mean_pool 等
        pooling strategy 后才进入 router/fusor feature contract。
    """

    sample_keys: SampleKeys
    embeddings: Any
    metadata: LineageMetadata = field(default_factory=dict)


class RawWindowProvider(Protocol):
    """协议功能：按显式 sample_keys 顺序提供 raw history window batch。"""

    def load_batch(self, sample_keys: Sequence[str]) -> RawWindowBatch:
        """函数功能：返回 `RawWindowBatch`，并保持输入 sample_keys 顺序。"""


class PreImageTransform(Protocol):
    """协议功能：对 raw window 执行可选 pre-image transform。"""

    def transform(self, batch: RawWindowBatch) -> PreImageBatch:
        """函数功能：返回 `PreImageBatch`，sample_keys 必须与输入完全一致。"""


class PseudoImageTransformer(Protocol):
    """协议功能：将 pre-image payload 转为 pseudo image payload。"""

    def transform(self, batch: PreImageBatch) -> VisualInputBatch:
        """函数功能：返回 `VisualInputBatch`，sample_keys 必须与输入完全一致。"""


class ResizePolicy(Protocol):
    """协议功能：表达 image input policy / resize policy 插槽。"""

    def apply(self, batch: VisualInputBatch) -> VisualInputBatch:
        """函数功能：返回 resize/input-policy 后的 `VisualInputBatch`。"""


class VisualEncoderProvider(Protocol):
    """协议功能：将 visual input 编码为 raw visual embeddings。"""

    def encode(self, batch: VisualInputBatch) -> VisualEmbeddingBatch:
        """函数功能：返回 `VisualEmbeddingBatch`，sample_keys 必须保序。"""


class PoolingStrategy(Protocol):
    """协议功能：将 raw visual embeddings 池化为 canonical FeatureBatch。"""

    def pool(self, batch: VisualEmbeddingBatch) -> FeatureBatch:
        """函数功能：返回 canonical `FeatureBatch`，sample_keys 必须保序。"""


class FeatureTransform(Protocol):
    """协议功能：对 canonical FeatureBatch 执行可选 feature transform。"""

    def transform(self, batch: FeatureBatch) -> FeatureBatch:
        """函数功能：返回新的或等价 `FeatureBatch`，sample_keys 必须保序。"""


@dataclass(frozen=True)
class VisualFeatureChainSpec:
    """
    类功能：
        记录 Visual feature chain 的可替换组件组合。

    输入：
        各字段均为协议组件实例；`feature_transform` 可为 None，表示 identity。

    输出：
        该 dataclass 不执行链路，只为 Runtime / smoke 保存组合说明。

    关键约束：
        skeleton 不把 cache path、checkpoint path 或具体 ViT/scaler state 设计进
        interface；真实资源发现和加载必须留在 Runtime / entrypoint。
    """

    raw_window_provider: RawWindowProvider
    pre_image_transform: PreImageTransform
    pseudo_image_transformer: PseudoImageTransformer
    resize_policy: ResizePolicy
    visual_encoder_provider: VisualEncoderProvider
    pooling_strategy: PoolingStrategy
    feature_transform: FeatureTransform | None = None
    chain_name: str = "stage1_visual_feature_chain_protocol_v1"
    metadata: LineageMetadata = field(default_factory=dict)
