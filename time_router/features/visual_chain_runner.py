"""
文件功能：
    提供 Stage 1 P19a Visual feature chain dry-run skeleton 的轻量编排器。

设计边界：
    Runner 只负责串联 P16f 已定义的 visual_chain 协议组件、校验
    sample_key 保序，并把各阶段 lineage 合并到最终 canonical FeatureBatch。
    本模块不加载真实 ViT、不导入 transformers、不读取 checkpoint 或 run_dir，
    也不决定 raw window / pseudo image / encoder 的具体实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from time_router.features.visual_chain import VisualFeatureChainSpec
from time_router.protocols import FeatureBatch


CHAIN_LINEAGE: tuple[str, ...] = (
    "raw_window",
    "pre_image",
    "pseudo_image",
    "resize_policy",
    "encoder",
    "pooling_strategy",
    "feature_transform",
)


@dataclass(frozen=True)
class VisualFeatureChainResult:
    """
    类功能：
        保存 dry-run chain 的最终 FeatureBatch 与中间阶段 lineage。

    输入/输出：
        `feature_batch` 是 canonical FeatureBatch；`stage_metadata` 只保存
        各协议组件主动返回的轻量 metadata，不包含 checkpoint/run_dir 等运行资源。
    """

    feature_batch: FeatureBatch
    stage_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualFeatureChainRunner:
    """
    类功能：
        串联 RawWindowProvider -> PreImageTransform -> PseudoImageTransformer
        -> ResizePolicy -> VisualEncoderProvider -> PoolingStrategy ->
        optional FeatureTransform。

    关键约束：
        Runner 的输入只有显式 sample_keys 和协议组件 spec。真实资源路径、ViT
        checkpoint、HF processor、训练 run_dir 等都必须留在 Runtime / entrypoint，
        不能进入本编排器接口。
    """

    spec: VisualFeatureChainSpec
    runner_name: str = "VisualFeatureChainRunner"

    def run(self, sample_keys: Sequence[str]) -> VisualFeatureChainResult:
        """
        函数功能：
            执行完整 Visual feature dry-run chain，并返回带 lineage 的 FeatureBatch。

        输入：
            `sample_keys` 必须由上游 SampleManifest 或 batch planner 显式给出。

        输出：
            返回 `VisualFeatureChainResult`，其中 FeatureBatch.sample_keys 与输入顺序一致。
        """
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        if not ordered_keys:
            raise ValueError("VisualFeatureChainRunner.run 需要非空 sample_keys")

        raw_batch = self.spec.raw_window_provider.load_batch(ordered_keys)
        self._assert_sample_keys("raw_window", raw_batch.sample_keys, ordered_keys)

        pre_image_batch = self.spec.pre_image_transform.transform(raw_batch)
        self._assert_sample_keys("pre_image", pre_image_batch.sample_keys, ordered_keys)

        visual_input_batch = self.spec.pseudo_image_transformer.transform(pre_image_batch)
        self._assert_sample_keys("pseudo_image", visual_input_batch.sample_keys, ordered_keys)

        resized_batch = self.spec.resize_policy.apply(visual_input_batch)
        self._assert_sample_keys("resize_policy", resized_batch.sample_keys, ordered_keys)

        embedding_batch = self.spec.visual_encoder_provider.encode(resized_batch)
        self._assert_sample_keys("encoder", embedding_batch.sample_keys, ordered_keys)

        pooled_batch = self.spec.pooling_strategy.pool(embedding_batch)
        self._assert_sample_keys("pooling_strategy", pooled_batch.sample_keys, ordered_keys)

        stage_metadata = {
            "raw_window": dict(raw_batch.metadata),
            "pre_image": dict(pre_image_batch.metadata),
            "pseudo_image": dict(visual_input_batch.metadata),
            "resize_policy": dict(resized_batch.metadata),
            "encoder": dict(embedding_batch.metadata),
            "pooling_strategy": dict(pooled_batch.extra.get("pooling_metadata", {})),
        }

        pooled_with_schema = self._with_runner_schema(
            batch=pooled_batch,
            ordered_keys=ordered_keys,
            stage_metadata=stage_metadata,
            has_feature_transform=self.spec.feature_transform is not None,
        )
        if self.spec.feature_transform is None:
            return VisualFeatureChainResult(feature_batch=pooled_with_schema, stage_metadata=stage_metadata)

        transformed_batch = self.spec.feature_transform.transform(pooled_with_schema)
        self._assert_sample_keys("feature_transform", transformed_batch.sample_keys, ordered_keys)
        transform_metadata = dict(transformed_batch.extra.get("feature_transform_metadata", {}))
        if not transform_metadata and "feature_transform" in transformed_batch.extra:
            transform_metadata = {"component": transformed_batch.extra["feature_transform"]}
        stage_metadata["feature_transform"] = transform_metadata
        final_batch = self._with_runner_schema(
            batch=transformed_batch,
            ordered_keys=ordered_keys,
            stage_metadata=stage_metadata,
            has_feature_transform=True,
        )
        return VisualFeatureChainResult(feature_batch=final_batch, stage_metadata=stage_metadata)

    def _with_runner_schema(
        self,
        *,
        batch: FeatureBatch,
        ordered_keys: tuple[str, ...],
        stage_metadata: Mapping[str, Mapping[str, Any]],
        has_feature_transform: bool,
    ) -> FeatureBatch:
        """
        函数功能：
            将 runner-level lineage 写入 FeatureBatch schema/extra。
        """
        schema = dict(batch.feature_schema)
        schema.update(
            {
                "chain_runner": self.runner_name,
                "chain_name": self.spec.chain_name,
                "chain_lineage": CHAIN_LINEAGE if has_feature_transform else CHAIN_LINEAGE[:-1],
                "raw_window_source": self._component_name(stage_metadata.get("raw_window", {})),
                "pseudo_image": self._component_name(stage_metadata.get("pseudo_image", {})),
                "resize_policy": self._component_name(stage_metadata.get("resize_policy", {})),
                "encoder": self._component_name(stage_metadata.get("encoder", {})),
                "pooling_strategy": self._component_name(stage_metadata.get("pooling_strategy", {})),
                "sample_key_order": "manifest_order",
            }
        )
        if has_feature_transform:
            schema["feature_transform"] = self._component_name(stage_metadata.get("feature_transform", {}))

        extra = dict(batch.extra)
        extra["chain_runner"] = self.runner_name
        extra["chain_metadata"] = {
            "spec_metadata": dict(self.spec.metadata),
            "stage_metadata": {stage: dict(metadata) for stage, metadata in stage_metadata.items()},
            "sample_count": len(ordered_keys),
        }
        return FeatureBatch(
            sample_keys=ordered_keys,
            features=batch.features,
            feature_schema=schema,
            extra=extra,
        )

    @staticmethod
    def _component_name(metadata: Mapping[str, Any]) -> str:
        """函数功能：从阶段 metadata 中取稳定组件名称，缺失时返回 unknown。"""
        for key in ("component", "source", "name"):
            value = metadata.get(key)
            if value:
                return str(value)
        return "unknown"

    @staticmethod
    def _assert_sample_keys(stage: str, actual: Sequence[str], expected: tuple[str, ...]) -> None:
        """函数功能：在每个 stage 后立即检查 sample_key 是否保序。"""
        actual_keys = tuple(str(sample_key) for sample_key in actual)
        if actual_keys != expected:
            raise ValueError(f"{stage} 未保持 sample_keys 顺序：actual={actual_keys!r} expected={expected!r}")
