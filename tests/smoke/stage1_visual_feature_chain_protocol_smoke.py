#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16f Visual feature chain protocol skeleton smoke。

输入：
    使用 P13b `sample_manifest.csv` 的 ordered sample_keys、smoke-local dummy
    Visual chain components，以及 P13b `expert_predictions.json` 构造的小型
    ExpertBatch。

输出：
    标准输出打印中文检查日志；若 Visual feature chain 的 sample_key 保序、
    batch shape、FeatureBatch dtype/schema lineage、dummy component 可替换性、
    P16a adapter 或 EvaluationInputAdapter 消费链路发生漂移，则抛出
    AssertionError。

关键约束：
    本 smoke 只验证 protocol chain 可组合。不实现真实 RevIN、pseudo image、
    resize、ViT、pooling 或 scaler fit；不读取 checkpoint，不访问 `/data2`，
    不创建 canonical run_dir，不修改正式 Visual entrypoint。
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import (  # noqa: E402
    PreImageBatch,
    RawWindowBatch,
    VisualEmbeddingBatch,
    VisualFeatureChainSpec,
    VisualInputBatch,
)
from time_router.models import LoadedTorchMLPRouterHeadAdapter  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
VISUAL_CHAIN_SOURCE_PATH = REPO_ROOT / "time_router" / "features" / "visual_chain.py"
VISUAL_SMALL_ENTRYPOINT_PATH = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

WINDOW_LENGTH = 6
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 4
IMAGE_WIDTH = 4
TOKEN_COUNT = 5
EMBED_DIM = 4
FEATURE_DIM = 4
ATOL = 1e-6
EXPECTED_LINEAGE = (
    "raw_window",
    "pre_image",
    "pseudo_image",
    "resize",
    "encoder",
    "pooling",
    "transform",
)
DISALLOWED_SOURCE_TOKENS = (
    "/data2",
    "torch.load",
    "ViTModel",
    "AutoImageProcessor",
    "import torch",
    "from torch",
    "import transformers",
    "from transformers",
    "import sklearn",
    "from sklearn",
    "checkpoint_path",
    "run_dir=",
)


class TinyLoadedMLP(torch.nn.Module):
    """
    类功能：
        P16f smoke 使用的内存小型 torch MLP。

    关键约束：
        该模型模拟 Runtime 已经持有已加载 router head；不读取 checkpoint，
        不代表 P16f 已经迁移正式 Visual Router 入口。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim + 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim + 2, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：将 head-ready dummy features 前向映射为专家 logits。"""
        return self.net(features)


@dataclass(frozen=True)
class DummyRawWindowProvider:
    """
    类功能：
        从内存 mapping 按显式 sample_keys 输出 RawWindowBatch。
    """

    windows_by_key: Mapping[str, np.ndarray]
    provider_name: str = "DummyRawWindowProvider"

    def load_batch(self, sample_keys: Sequence[str]) -> RawWindowBatch:
        """函数功能：按输入顺序堆叠 raw window，不排序、不访问文件。"""
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        windows = np.stack([self.windows_by_key[sample_key] for sample_key in ordered_keys], axis=0).astype(
            np.float32,
            copy=False,
        )
        return RawWindowBatch(
            sample_keys=ordered_keys,
            windows=windows,
            metadata={"stage": "raw_window", "component": self.provider_name, "shape": tuple(windows.shape)},
        )


class IdentityPreImageTransform:
    """类功能：测试用 identity pre-image transform，占位 future RevIN / normalization。"""

    transform_name = "IdentityPreImageTransform"

    def transform(self, batch: RawWindowBatch) -> PreImageBatch:
        """函数功能：复制 raw window payload 并保留 sample_keys 顺序。"""
        values = np.asarray(batch.windows, dtype=np.float32).copy()
        return PreImageBatch(
            sample_keys=batch.sample_keys,
            values=values,
            metadata={
                "stage": "pre_image",
                "component": self.transform_name,
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(values.shape),
            },
        )


class OffsetPreImageTransform(IdentityPreImageTransform):
    """类功能：替换性测试用 transform，确认 chain component 可替换。"""

    transform_name = "OffsetPreImageTransform"

    def __init__(self, *, offset: float) -> None:
        self.offset = float(offset)

    def transform(self, batch: RawWindowBatch) -> PreImageBatch:
        """函数功能：给 raw window 加固定 offset，模拟替换组件影响输出。"""
        values = (np.asarray(batch.windows, dtype=np.float32) + self.offset).astype(np.float32)
        return PreImageBatch(
            sample_keys=batch.sample_keys,
            values=values,
            metadata={
                "stage": "pre_image",
                "component": self.transform_name,
                "offset": self.offset,
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(values.shape),
            },
        )


class DummyPseudoImageTransformer:
    """
    类功能：
        将一维 dummy window 展开成固定 shape 的 pseudo image payload。
    """

    transformer_name = "DummyPseudoImageTransformer"

    def transform(self, batch: PreImageBatch) -> VisualInputBatch:
        """函数功能：构造 deterministic pseudo image tensor，仅用于 protocol smoke。"""
        values = np.asarray(batch.values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != WINDOW_LENGTH:
            raise AssertionError(f"pre-image values shape 异常：{values.shape}")
        base = np.mean(values, axis=1, keepdims=True).reshape(values.shape[0], 1, 1, 1)
        image_grid = np.linspace(0.0, 1.0, num=IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH, dtype=np.float32)
        images = (base + image_grid.reshape(1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
        return VisualInputBatch(
            sample_keys=batch.sample_keys,
            images=images,
            metadata={
                "stage": "pseudo_image",
                "component": self.transformer_name,
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(images.shape),
            },
        )


class IdentityResizePolicy:
    """类功能：测试用 identity image input policy，占位 future resize / processor policy。"""

    policy_name = "IdentityResizePolicy"

    def apply(self, batch: VisualInputBatch) -> VisualInputBatch:
        """函数功能：返回等 shape 的 VisualInputBatch，并记录 resize stage lineage。"""
        images = np.asarray(batch.images, dtype=np.float32).copy()
        return VisualInputBatch(
            sample_keys=batch.sample_keys,
            images=images,
            metadata={
                "stage": "resize",
                "component": self.policy_name,
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(images.shape),
            },
        )


class DummyVisualEncoderProvider:
    """
    类功能：
        将 dummy visual input 编码为固定 token embedding。
    """

    encoder_name = "DummyVisualEncoderProvider"

    def encode(self, batch: VisualInputBatch) -> VisualEmbeddingBatch:
        """函数功能：用 numpy 生成 deterministic embedding，不启动 ViT。"""
        images = np.asarray(batch.images, dtype=np.float32)
        flattened = images.reshape(images.shape[0], -1)
        sample_mean = np.mean(flattened, axis=1, keepdims=True)
        token_axis = np.arange(TOKEN_COUNT, dtype=np.float32).reshape(1, TOKEN_COUNT, 1)
        dim_axis = np.arange(EMBED_DIM, dtype=np.float32).reshape(1, 1, EMBED_DIM)
        embeddings = (sample_mean[:, None, :] + token_axis * 0.10 + dim_axis * 0.01).astype(np.float32)
        return VisualEmbeddingBatch(
            sample_keys=batch.sample_keys,
            embeddings=embeddings,
            metadata={
                "stage": "encoder",
                "component": self.encoder_name,
                "loads_real_vit": False,
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(embeddings.shape),
            },
        )


class DummyPoolingStrategy:
    """
    类功能：
        将 dummy token embedding 池化为 canonical FeatureBatch。
    """

    pooling_name = "DummyPoolingStrategy"

    def __init__(self, *, mode: str = "mean_patch") -> None:
        self.mode = str(mode)

    def pool(self, batch: VisualEmbeddingBatch) -> FeatureBatch:
        """函数功能：按 mode 做测试用池化，并输出 float32 FeatureBatch。"""
        embeddings = np.asarray(batch.embeddings, dtype=np.float32)
        if self.mode == "cls":
            features = embeddings[:, 0, :]
        elif self.mode == "mean_patch":
            features = np.mean(embeddings[:, 1:, :], axis=1)
        else:
            raise AssertionError(f"未知 dummy pooling mode：{self.mode}")
        features = features.astype(np.float32, copy=False)
        return FeatureBatch(
            sample_keys=batch.sample_keys,
            features=features,
            feature_schema={
                "stage": "pooling",
                "component": self.pooling_name,
                "pooling_mode": self.mode,
                "feature_dim": int(features.shape[1]),
                "feature_columns": tuple(f"visual_feature_{idx}" for idx in range(features.shape[1])),
                "input_stage": batch.metadata.get("stage"),
                "head_ready": False,
                "dtype": str(features.dtype),
            },
            extra={"pooling_metadata": dict(batch.metadata)},
        )


class IdentityFeatureTransform:
    """类功能：测试用 identity FeatureTransform，占位 LoadedFeatureScaler / normalizer。"""

    transform_name = "IdentityFeatureTransform"

    def transform(self, batch: FeatureBatch) -> FeatureBatch:
        """函数功能：复制 features 为 float32 并补充 transform lineage。"""
        features = np.asarray(batch.features, dtype=np.float32).copy()
        input_schema = dict(batch.feature_schema)
        lineage = tuple(input_schema.get("chain_lineage", ())) + ("transform",)
        return FeatureBatch(
            sample_keys=tuple(batch.sample_keys),
            features=features,
            feature_schema={
                **input_schema,
                "stage": "transform",
                "component": self.transform_name,
                "chain_lineage": lineage,
                "feature_dim": int(features.shape[1]),
                "head_ready": True,
                "dtype": str(features.dtype),
            },
            extra={**dict(batch.extra), "feature_transform": self.transform_name, "fit_performed": False},
        )


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入文件存在于仓库内且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P16f smoke 不应访问 /data2：{path}")


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """函数功能：从 P13b manifest 中读取 ordered sample_keys。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{sample_keys}")
    return sample_keys


def load_expert_batch_from_reference(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """函数功能：用 P13b expert JSON 数值参考构造最小 ExpertBatch。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_columns = tuple(str(model_name) for model_name in payload["model_columns"])
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise AssertionError("expert_predictions.json 缺少 samples list")
    sample_by_key: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        sample_key = str(sample["sample_key"])
        if sample_key in sample_by_key:
            raise AssertionError(f"expert_predictions.json 存在重复 sample_key：{sample_key}")
        sample_by_key[sample_key] = sample

    y_true_rows = []
    y_pred_rows = []
    for sample_key in ordered_sample_keys:
        sample = sample_by_key[str(sample_key)]
        y_true_rows.append(np.asarray(sample["y_true"], dtype=np.float32))
        y_pred_rows.append(np.asarray(sample["y_pred"], dtype=np.float32))

    return ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in ordered_sample_keys),
        model_columns=model_columns,
        y_pred=np.stack(y_pred_rows, axis=0).astype(np.float32),
        y_true=np.stack(y_true_rows, axis=0).astype(np.float32),
        row_index_metadata={"source": "tests/fixtures/stage1_real_derived_small/expert_predictions.json"},
        extra={"provider_name": "P16fInMemoryExpertBatchReference"},
    )


def build_dummy_windows(ordered_sample_keys: Sequence[str]) -> dict[str, np.ndarray]:
    """函数功能：为每个 sample_key 构造 deterministic raw window。"""
    windows_by_key: dict[str, np.ndarray] = {}
    for row_index, sample_key in enumerate(ordered_sample_keys):
        base = np.linspace(0.0, 1.0, num=WINDOW_LENGTH, dtype=np.float32)
        windows_by_key[str(sample_key)] = (base + row_index * 0.25).astype(np.float32)
    return windows_by_key


def build_chain_spec(
    *,
    windows_by_key: Mapping[str, np.ndarray],
    pre_image_transform: Any | None = None,
) -> VisualFeatureChainSpec:
    """函数功能：构造 smoke-local dummy VisualFeatureChainSpec。"""
    return VisualFeatureChainSpec(
        raw_window_provider=DummyRawWindowProvider(windows_by_key=windows_by_key),
        pre_image_transform=pre_image_transform or IdentityPreImageTransform(),
        pseudo_image_transformer=DummyPseudoImageTransformer(),
        resize_policy=IdentityResizePolicy(),
        visual_encoder_provider=DummyVisualEncoderProvider(),
        pooling_strategy=DummyPoolingStrategy(mode="mean_patch"),
        feature_transform=IdentityFeatureTransform(),
        metadata={"purpose": "P16f protocol skeleton smoke", "loads_real_vit": False},
    )


def append_chain_lineage(feature_batch: FeatureBatch, stages: Sequence[str]) -> FeatureBatch:
    """函数功能：在 pooling 后补充完整链路 lineage，供 transform 继续转递。"""
    schema = dict(feature_batch.feature_schema)
    schema["chain_lineage"] = tuple(str(stage) for stage in stages)
    return FeatureBatch(
        sample_keys=feature_batch.sample_keys,
        features=feature_batch.features,
        feature_schema=schema,
        extra=dict(feature_batch.extra),
    )


def execute_dummy_chain(
    *,
    spec: VisualFeatureChainSpec,
    sample_keys: Sequence[str],
) -> tuple[FeatureBatch, dict[str, Any]]:
    """
    函数功能：
        串联 dummy Visual feature chain，并返回最终 FeatureBatch 与中间 batch。
    """
    raw_batch = spec.raw_window_provider.load_batch(sample_keys)
    pre_image_batch = spec.pre_image_transform.transform(raw_batch)
    visual_input_batch = spec.pseudo_image_transformer.transform(pre_image_batch)
    resized_batch = spec.resize_policy.apply(visual_input_batch)
    embedding_batch = spec.visual_encoder_provider.encode(resized_batch)
    pooled_batch = spec.pooling_strategy.pool(embedding_batch)
    pooled_batch = append_chain_lineage(pooled_batch, EXPECTED_LINEAGE[:-1])
    feature_batch = spec.feature_transform.transform(pooled_batch) if spec.feature_transform else pooled_batch

    intermediates = {
        "raw_window": raw_batch,
        "pre_image": pre_image_batch,
        "pseudo_image": visual_input_batch,
        "resize": resized_batch,
        "encoder": embedding_batch,
        "pooling": pooled_batch,
        "transform": feature_batch,
    }
    return feature_batch, intermediates


def assert_source_boundaries() -> None:
    """函数功能：扫描 P16f 新增源码和正式入口，确认未越界接入真实实现。"""
    assert_repo_file(VISUAL_CHAIN_SOURCE_PATH)
    source = VISUAL_CHAIN_SOURCE_PATH.read_text(encoding="utf-8")
    for token in DISALLOWED_SOURCE_TOKENS:
        if token in source:
            raise AssertionError(f"P16f visual_chain 源码不应包含 {token!r}")

    assert_repo_file(VISUAL_SMALL_ENTRYPOINT_PATH)
    visual_small_source = VISUAL_SMALL_ENTRYPOINT_PATH.read_text(encoding="utf-8")
    if "VisualFeatureChainSpec" in visual_small_source:
        raise AssertionError("P16f 不应修改或接入 scripts/run_stage1_visual_small.py")


def assert_intermediate_batches(intermediates: Mapping[str, Any], ordered_sample_keys: Sequence[str]) -> None:
    """函数功能：验证每层 sample_keys 保序和 dummy batch shape。"""
    expected_keys = tuple(str(sample_key) for sample_key in ordered_sample_keys)
    expected_shapes = {
        "raw_window": (len(expected_keys), WINDOW_LENGTH),
        "pre_image": (len(expected_keys), WINDOW_LENGTH),
        "pseudo_image": (len(expected_keys), IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        "resize": (len(expected_keys), IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
        "encoder": (len(expected_keys), TOKEN_COUNT, EMBED_DIM),
        "pooling": (len(expected_keys), FEATURE_DIM),
        "transform": (len(expected_keys), FEATURE_DIM),
    }
    for stage, batch in intermediates.items():
        if tuple(batch.sample_keys) != expected_keys:
            raise AssertionError(f"{stage} 未保持 sample_keys 顺序：{batch.sample_keys}")
        payload = batch.features if isinstance(batch, FeatureBatch) else getattr(batch, "windows", None)
        payload = getattr(batch, "values", payload)
        payload = getattr(batch, "images", payload)
        payload = getattr(batch, "embeddings", payload)
        shape = tuple(np.asarray(payload).shape)
        if shape != expected_shapes[stage]:
            raise AssertionError(f"{stage} shape 漂移：actual={shape} expected={expected_shapes[stage]}")


def assert_feature_batch_contract(feature_batch: FeatureBatch, ordered_sample_keys: Sequence[str]) -> None:
    """函数功能：验证最终 canonical FeatureBatch 的 dtype、shape 和 lineage。"""
    if not isinstance(feature_batch, FeatureBatch):
        raise AssertionError(f"chain 未输出 FeatureBatch：actual={type(feature_batch)!r}")
    if feature_batch.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"FeatureBatch sample_keys 未保序：{feature_batch.sample_keys}")
    features = np.asarray(feature_batch.features)
    if features.shape != (len(ordered_sample_keys), FEATURE_DIM):
        raise AssertionError(f"FeatureBatch features shape 漂移：{features.shape}")
    if features.dtype != np.float32:
        raise AssertionError(f"FeatureBatch dtype 必须为 float32：actual={features.dtype}")
    if not np.all(np.isfinite(features)):
        raise AssertionError("FeatureBatch features 包含 NaN 或 Inf")
    schema = dict(feature_batch.feature_schema)
    if tuple(schema.get("chain_lineage", ())) != EXPECTED_LINEAGE:
        raise AssertionError(f"FeatureBatch schema 未记录完整 chain lineage：{schema}")
    if schema.get("head_ready") is not True:
        raise AssertionError(f"FeatureBatch transform 后应标记 head_ready=True：{schema}")
    if feature_batch.extra.get("fit_performed") is not False:
        raise AssertionError(f"IdentityFeatureTransform 不应执行 scaler fit：{feature_batch.extra}")


def assert_component_replacement_works(
    *,
    windows_by_key: Mapping[str, np.ndarray],
    ordered_sample_keys: Sequence[str],
    baseline_features: np.ndarray,
) -> None:
    """函数功能：替换一个 dummy component 后仍输出合法 FeatureBatch。"""
    replacement_spec = build_chain_spec(
        windows_by_key=windows_by_key,
        pre_image_transform=OffsetPreImageTransform(offset=0.5),
    )
    replacement_batch, replacement_intermediates = execute_dummy_chain(
        spec=replacement_spec,
        sample_keys=ordered_sample_keys,
    )
    assert_intermediate_batches(replacement_intermediates, ordered_sample_keys)
    assert_feature_batch_contract(replacement_batch, ordered_sample_keys)
    if np.allclose(np.asarray(replacement_batch.features), baseline_features, rtol=0.0, atol=ATOL):
        raise AssertionError("替换 pre-image dummy component 后 features 不应与 baseline 完全相同")


def build_loaded_tiny_mlp(*, input_dim: int, output_dim: int) -> TinyLoadedMLP:
    """函数功能：构造固定 seed 的已加载小型 MLP，不调用 torch.load。"""
    torch.manual_seed(20260621)
    model = TinyLoadedMLP(input_dim=input_dim, output_dim=output_dim)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, a=-0.08, b=0.08)
    model.eval()
    return model


def fail_torch_load(*args: object, **kwargs: object) -> object:
    """函数功能：若 smoke 核心路径调用 torch.load，则立即失败。"""
    raise AssertionError(f"P16f smoke 不应调用 torch.load 或读取 checkpoint：args={args} kwargs={kwargs}")


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def assert_router_and_evaluator(
    *,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> EvaluationInputAdapterResult:
    """函数功能：验证 P16a adapter 和 EvaluationInputAdapter 可消费最终 FeatureBatch。"""
    model = build_loaded_tiny_mlp(
        input_dim=int(np.asarray(feature_batch.features).shape[1]),
        output_dim=len(expert_batch.model_columns),
    )
    adapter = LoadedTorchMLPRouterHeadAdapter(model=model, device=torch.device("cpu"))
    evaluator = EvaluationInputAdapter()
    router_output: RouterOutput = adapter.predict(feature_batch, expert_batch.model_columns)
    result = evaluator.evaluate(expert_batch=expert_batch, router_output=router_output)
    if router_output.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"RouterOutput 未保持 FeatureBatch sample_keys：{router_output.sample_keys}")
    if result.evaluation_input.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"EvaluationInput sample_keys 未保序：{result.evaluation_input.sample_keys}")
    weights = np.asarray(router_output.weights)
    expected_shape = (len(feature_batch.sample_keys), len(expert_batch.model_columns))
    if weights.shape != expected_shape:
        raise AssertionError(f"RouterOutput weights shape 漂移：actual={weights.shape} expected={expected_shape}")
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(expected_shape[0]), rtol=0.0, atol=ATOL)
    return result


def run_smoke() -> None:
    """函数功能：执行 P16f Visual feature chain protocol smoke。"""
    print("开始 Stage 1 P16f Visual feature chain protocol smoke")
    before_outputs = snapshot_run_outputs()
    assert_source_boundaries()
    print("通过：visual_chain 源码未引入 /data2、checkpoint、ViT、transformers、sklearn 或 run_dir")

    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    expert_batch = load_expert_batch_from_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    windows_by_key = build_dummy_windows(ordered_sample_keys)
    spec = build_chain_spec(windows_by_key=windows_by_key)
    print("通过：已构造 P13b ordered sample_keys、内存 ExpertBatch 和 smoke-local dummy chain spec")

    feature_batch, intermediates = execute_dummy_chain(spec=spec, sample_keys=ordered_sample_keys)
    assert_intermediate_batches(intermediates, ordered_sample_keys)
    assert_feature_batch_contract(feature_batch, ordered_sample_keys)
    print("通过：raw/pre-image/pseudo-image/resize/encoder/pooling/transform 每层保序且 shape 合理")

    assert_component_replacement_works(
        windows_by_key=windows_by_key,
        ordered_sample_keys=ordered_sample_keys,
        baseline_features=np.asarray(feature_batch.features),
    )
    print("通过：替换 pre-image dummy component 后仍输出合法 FeatureBatch")

    with patch.object(torch, "load", side_effect=fail_torch_load):
        result = assert_router_and_evaluator(feature_batch=feature_batch, expert_batch=expert_batch)
    print(
        "通过：最终 FeatureBatch 可被 P16a adapter 和 EvaluationInputAdapter 消费，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P16f smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P16f Visual feature chain protocol smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
