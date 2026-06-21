#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P19a VisualFeatureChainRunner real-chain dry-run skeleton smoke。

输入：
    使用 P13b small manifest 的 ordered sample_keys、显式 raw window JSON fixture、
    smoke-local no-transformers fake encoder，以及 P13b expert_predictions.json。

输出：
    标准输出打印中文检查日志；若 VisualFeatureChainRunner 未输出 canonical
    FeatureBatch、sample_key 保序、dtype/shape/finite、feature_schema lineage、
    fake encoder 边界或 P17/P18 相关入口边界漂移，则抛出 AssertionError。

关键约束：
    本 smoke 不加载真实 ViT，不导入 transformers，不访问 `/data2`，不训练，
    不调用 full-scale，也不修改 `train_visual_router_online_streaming.py`。
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
    CHAIN_LINEAGE,
    PreImageBatch,
    RawWindowBatch,
    VisualEmbeddingBatch,
    VisualFeatureChainRunner,
    VisualFeatureChainSpec,
    VisualInputBatch,
)
from time_router.models import LoadedTorchMLPRouterHeadAdapter  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
P19A_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_chain_dryrun"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
RAW_WINDOWS_PATH = P19A_FIXTURE_ROOT / "raw_windows.json"
RUNNER_SOURCE_PATH = REPO_ROOT / "time_router" / "features" / "visual_chain_runner.py"
FEATURES_INIT_PATH = REPO_ROOT / "time_router" / "features" / "__init__.py"
STREAMING_ENTRYPOINT_PATH = (
    REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"
)

WINDOW_LENGTH = 6
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 4
IMAGE_WIDTH = 4
TOKEN_COUNT = 5
EMBED_DIM = 4
FEATURE_DIM = 4
ATOL = 1e-6


class TinyLoadedMLP(torch.nn.Module):
    """
    类功能：
        P19a smoke 使用的内存小型 torch MLP。

    关键约束：
        模型已在内存中构造，不读取 checkpoint，不代表真实 Visual Router 训练入口迁移。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim + 3),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim + 3, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：将 dry-run features 映射为专家 logits。"""
        return self.net(features)


@dataclass(frozen=True)
class JsonRawWindowProvider:
    """
    类功能：
        从显式 JSON fixture 按输入 sample_keys 顺序输出 RawWindowBatch。
    """

    windows_by_key: Mapping[str, np.ndarray]
    fixture_path: Path
    provider_name: str = "json_raw_window_fixture"

    def load_batch(self, sample_keys: Sequence[str]) -> RawWindowBatch:
        """函数功能：按 manifest 顺序堆叠 raw window，不排序、不访问外部数据。"""
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        windows = np.stack([self.windows_by_key[sample_key] for sample_key in ordered_keys], axis=0).astype(
            np.float32,
            copy=False,
        )
        return RawWindowBatch(
            sample_keys=ordered_keys,
            windows=windows,
            metadata={
                "stage": "raw_window",
                "component": self.provider_name,
                "source": str(self.fixture_path.relative_to(REPO_ROOT)),
                "shape": tuple(windows.shape),
            },
        )


class IdentityPreImageTransform:
    """类功能：测试用 identity pre-image transform，占位 future RevIN / normalization。"""

    def transform(self, batch: RawWindowBatch) -> PreImageBatch:
        """函数功能：复制 raw window payload 并保留 sample_keys 顺序。"""
        values = np.asarray(batch.windows, dtype=np.float32).copy()
        return PreImageBatch(
            sample_keys=batch.sample_keys,
            values=values,
            metadata={
                "stage": "pre_image",
                "component": "identity_pre_image",
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(values.shape),
            },
        )


class TinyPseudoImageTransform:
    """
    类功能：
        将一维 raw window 变换为 deterministic CHW pseudo image。
    """

    def transform(self, batch: PreImageBatch) -> VisualInputBatch:
        """函数功能：构造固定 shape 的 pseudo image tensor。"""
        values = np.asarray(batch.values, dtype=np.float32)
        if values.shape != (len(batch.sample_keys), WINDOW_LENGTH):
            raise AssertionError(f"pre-image shape 异常：{values.shape}")
        window_mean = np.mean(values, axis=1, keepdims=True).reshape(values.shape[0], 1, 1, 1)
        window_std = np.std(values, axis=1, keepdims=True).reshape(values.shape[0], 1, 1, 1)
        grid = np.linspace(0.0, 1.0, num=IMAGE_HEIGHT * IMAGE_WIDTH, dtype=np.float32).reshape(
            1,
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        channels = np.concatenate(
            [
                window_mean + grid,
                window_std + grid * 0.5,
                (window_mean + window_std) + grid * 0.25,
            ],
            axis=1,
        ).astype(np.float32)
        return VisualInputBatch(
            sample_keys=batch.sample_keys,
            images=channels,
            metadata={
                "stage": "pseudo_image",
                "component": "tiny_deterministic_pseudo_image",
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(channels.shape),
            },
        )


class IdentityResizePolicy:
    """类功能：测试用 resize/input policy，占位 future resize / processor policy。"""

    def apply(self, batch: VisualInputBatch) -> VisualInputBatch:
        """函数功能：返回等 shape 的 VisualInputBatch，并记录 resize lineage。"""
        images = np.asarray(batch.images, dtype=np.float32).copy()
        return VisualInputBatch(
            sample_keys=batch.sample_keys,
            images=images,
            metadata={
                "stage": "resize_policy",
                "component": "identity_resize_policy",
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(images.shape),
            },
        )


class FakeNoTransformersVisualEncoder:
    """
    类功能：
        P19a 专用 no-transformers fake encoder。

    关键约束：
        该类只在 smoke 内部存在，不进入 `time_router.features` public core。
    """

    def encode(self, batch: VisualInputBatch) -> VisualEmbeddingBatch:
        """函数功能：用 numpy deterministic 规则生成 token embedding。"""
        images = np.asarray(batch.images, dtype=np.float32)
        flattened = images.reshape(images.shape[0], -1)
        mean = np.mean(flattened, axis=1, keepdims=True)
        maximum = np.max(flattened, axis=1, keepdims=True)
        token_axis = np.arange(TOKEN_COUNT, dtype=np.float32).reshape(1, TOKEN_COUNT, 1)
        dim_axis = np.arange(EMBED_DIM, dtype=np.float32).reshape(1, 1, EMBED_DIM)
        embeddings = (mean[:, None, :] + maximum[:, None, :] * 0.05 + token_axis * 0.1 + dim_axis * 0.01).astype(
            np.float32
        )
        return VisualEmbeddingBatch(
            sample_keys=batch.sample_keys,
            embeddings=embeddings,
            metadata={
                "stage": "encoder",
                "component": "fake_no_transformers",
                "loads_real_vit": False,
                "input_stage": batch.metadata.get("stage"),
                "shape": tuple(embeddings.shape),
            },
        )


class MeanPatchPoolingStrategy:
    """类功能：将 fake token embedding 池化为 canonical FeatureBatch。"""

    def pool(self, batch: VisualEmbeddingBatch) -> FeatureBatch:
        """函数功能：对 patch tokens 做 mean pooling，并输出 float32 FeatureBatch。"""
        embeddings = np.asarray(batch.embeddings, dtype=np.float32)
        if embeddings.shape[1:] != (TOKEN_COUNT, EMBED_DIM):
            raise AssertionError(f"embedding shape 异常：{embeddings.shape}")
        features = np.mean(embeddings[:, 1:, :], axis=1).astype(np.float32, copy=False)
        return FeatureBatch(
            sample_keys=batch.sample_keys,
            features=features,
            feature_schema={
                "stage": "pooling_strategy",
                "component": "mean_patch_pooling",
                "feature_dim": int(features.shape[1]),
                "dtype": str(features.dtype),
            },
            extra={"pooling_metadata": {"component": "mean_patch_pooling", "input_stage": batch.metadata.get("stage")}},
        )


class IdentityFeatureTransform:
    """类功能：测试用 identity FeatureTransform，占位 loaded scaler transform。"""

    def transform(self, batch: FeatureBatch) -> FeatureBatch:
        """函数功能：复制 float32 features，并记录 transform 未执行 fit。"""
        features = np.asarray(batch.features, dtype=np.float32).copy()
        return FeatureBatch(
            sample_keys=batch.sample_keys,
            features=features,
            feature_schema={**dict(batch.feature_schema), "head_ready": True},
            extra={
                **dict(batch.extra),
                "feature_transform": "identity_feature_transform",
                "feature_transform_metadata": {"component": "identity_feature_transform", "fit_performed": False},
            },
        )


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入文件存在且位于当前仓库。"""
    if not path.is_file():
        raise AssertionError(f"文件缺失：{path}")
    if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise AssertionError(f"fixture 不在仓库内：{path}")


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """函数功能：从 P13b manifest 读取 ordered sample_keys。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if not sample_keys or len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"manifest sample_keys 异常：{sample_keys}")
    return sample_keys


def load_raw_windows(path: Path) -> dict[str, np.ndarray]:
    """函数功能：读取显式 raw window fixture，并做 shape/dtype 基础校验。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    windows_payload = payload.get("windows")
    if not isinstance(windows_payload, dict):
        raise AssertionError("raw window fixture 缺少 windows object")
    windows_by_key = {
        str(sample_key): np.asarray(values, dtype=np.float32) for sample_key, values in windows_payload.items()
    }
    for sample_key, values in windows_by_key.items():
        if values.shape != (WINDOW_LENGTH,) or not np.all(np.isfinite(values)):
            raise AssertionError(f"raw window fixture 异常：{sample_key} -> {values}")
    return windows_by_key


def load_expert_batch(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """函数功能：用 P13b expert JSON 构造内存 ExpertBatch。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_columns = tuple(str(model_name) for model_name in payload["model_columns"])
    samples = {str(sample["sample_key"]): sample for sample in payload["samples"]}
    y_true_rows = []
    y_pred_rows = []
    for sample_key in ordered_sample_keys:
        sample = samples[str(sample_key)]
        y_true_rows.append(np.asarray(sample["y_true"], dtype=np.float32))
        y_pred_rows.append(np.asarray(sample["y_pred"], dtype=np.float32))
    return ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in ordered_sample_keys),
        model_columns=model_columns,
        y_pred=np.stack(y_pred_rows, axis=0).astype(np.float32),
        y_true=np.stack(y_true_rows, axis=0).astype(np.float32),
        row_index_metadata={"source": str(path.relative_to(REPO_ROOT))},
        extra={"provider_name": "P19aInMemoryExpertReference"},
    )


def build_chain_spec(windows_by_key: Mapping[str, np.ndarray]) -> VisualFeatureChainSpec:
    """函数功能：构造 P19a dry-run chain spec。"""
    return VisualFeatureChainSpec(
        raw_window_provider=JsonRawWindowProvider(windows_by_key=windows_by_key, fixture_path=RAW_WINDOWS_PATH),
        pre_image_transform=IdentityPreImageTransform(),
        pseudo_image_transformer=TinyPseudoImageTransform(),
        resize_policy=IdentityResizePolicy(),
        visual_encoder_provider=FakeNoTransformersVisualEncoder(),
        pooling_strategy=MeanPatchPoolingStrategy(),
        feature_transform=IdentityFeatureTransform(),
        chain_name="stage1_p19a_visual_feature_chain_dryrun_skeleton",
        metadata={"purpose": "P19a real VisualFeatureChain dry-run skeleton", "loads_real_vit": False},
    )


def assert_feature_batch_contract(feature_batch: FeatureBatch, ordered_sample_keys: Sequence[str]) -> None:
    """函数功能：验证最终 canonical FeatureBatch 的 shape、dtype、finite 和 lineage。"""
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
    expected_schema = {
        "chain_runner": "VisualFeatureChainRunner",
        "raw_window_source": "json_raw_window_fixture",
        "pseudo_image": "tiny_deterministic_pseudo_image",
        "resize_policy": "identity_resize_policy",
        "encoder": "fake_no_transformers",
        "pooling_strategy": "mean_patch_pooling",
        "feature_transform": "identity_feature_transform",
        "head_ready": True,
    }
    for key, expected_value in expected_schema.items():
        if schema.get(key) != expected_value:
            raise AssertionError(f"FeatureBatch schema {key} 异常：actual={schema.get(key)!r} schema={schema}")
    if tuple(schema.get("chain_lineage", ())) != CHAIN_LINEAGE:
        raise AssertionError(f"FeatureBatch chain_lineage 异常：{schema}")
    stage_metadata = feature_batch.extra.get("chain_metadata", {}).get("stage_metadata", {})
    for stage in ("raw_window", "pre_image", "pseudo_image", "resize_policy", "encoder", "pooling_strategy"):
        if stage not in stage_metadata:
            raise AssertionError(f"FeatureBatch extra 缺少 stage metadata：{stage} -> {feature_batch.extra}")


def build_loaded_tiny_mlp(*, input_dim: int, output_dim: int) -> TinyLoadedMLP:
    """函数功能：构造固定 seed 的小型已加载 MLP，不读取 checkpoint。"""
    torch.manual_seed(20260621)
    model = TinyLoadedMLP(input_dim=input_dim, output_dim=output_dim)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, a=-0.07, b=0.07)
    model.eval()
    return model


def fail_torch_load(*args: object, **kwargs: object) -> object:
    """函数功能：若 P19a 核心路径读取 checkpoint，则立即失败。"""
    raise AssertionError(f"P19a smoke 不应调用 torch.load：args={args} kwargs={kwargs}")


def assert_router_and_evaluator(
    *,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> EvaluationInputAdapterResult:
    """函数功能：证明 dry-run FeatureBatch 可接 canonical eval 后半段。"""
    model = build_loaded_tiny_mlp(
        input_dim=int(np.asarray(feature_batch.features).shape[1]),
        output_dim=len(expert_batch.model_columns),
    )
    adapter = LoadedTorchMLPRouterHeadAdapter(model=model, device=torch.device("cpu"))
    router_output: RouterOutput = adapter.predict(feature_batch, expert_batch.model_columns)
    result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)
    if router_output.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"RouterOutput 未保持 sample_keys：{router_output.sample_keys}")
    if result.evaluation_input.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"EvaluationInput 未保持 sample_keys：{result.evaluation_input.sample_keys}")
    weights = np.asarray(router_output.weights)
    expected_shape = (len(feature_batch.sample_keys), len(expert_batch.model_columns))
    if weights.shape != expected_shape:
        raise AssertionError(f"RouterOutput weights shape 漂移：actual={weights.shape} expected={expected_shape}")
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(expected_shape[0]), rtol=0.0, atol=ATOL)
    return result


def assert_source_boundaries(before_streaming_bytes: bytes) -> None:
    """函数功能：确认新增生产源码未引入真实 ViT/训练入口边界。"""
    for path in (RUNNER_SOURCE_PATH, FEATURES_INIT_PATH):
        assert_repo_file(path)
        source = path.read_text(encoding="utf-8")
        for token in ("ViTModel", "AutoImageProcessor", "import transformers", "from transformers", "/data2"):
            if token in source:
                raise AssertionError(f"{path.relative_to(REPO_ROOT)} 不应包含禁止 token：{token}")
    if "transformers" in sys.modules:
        raise AssertionError("P19a smoke 不应导入 transformers")
    if STREAMING_ENTRYPOINT_PATH.read_bytes() != before_streaming_bytes:
        raise AssertionError("P19a smoke 不应修改 train_visual_router_online_streaming.py")


def run_smoke() -> None:
    """函数功能：执行 P19a VisualFeatureChainRunner dry-run skeleton smoke。"""
    print("开始 Stage 1 P19a VisualFeatureChainRunner dry-run skeleton smoke")
    assert_repo_file(STREAMING_ENTRYPOINT_PATH)
    before_streaming_bytes = STREAMING_ENTRYPOINT_PATH.read_bytes()

    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    windows_by_key = load_raw_windows(RAW_WINDOWS_PATH)
    if set(windows_by_key) != set(ordered_sample_keys):
        raise AssertionError("raw window fixture sample_keys 必须与 P13b manifest 完全一致")
    expert_batch = load_expert_batch(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    print("通过：已读取 P13b manifest、显式 raw window fixture 和内存 ExpertBatch")

    runner = VisualFeatureChainRunner(spec=build_chain_spec(windows_by_key))
    result = runner.run(ordered_sample_keys)
    assert_feature_batch_contract(result.feature_batch, ordered_sample_keys)
    print("通过：VisualFeatureChainRunner 输出 canonical FeatureBatch，保序、float32、二维、finite 和 lineage 均成立")

    with patch.object(torch, "load", side_effect=fail_torch_load):
        eval_result = assert_router_and_evaluator(feature_batch=result.feature_batch, expert_batch=expert_batch)
    print(
        "通过：FeatureBatch 可接 LoadedTorchMLPRouterHeadAdapter + EvaluationInputAdapter，"
        f"hard_mae={eval_result.summary['hard_mae']:.9f}，raw_soft_mae={eval_result.summary['raw_soft_mae']:.9f}"
    )

    assert_source_boundaries(before_streaming_bytes)
    print("完成：Stage 1 P19a VisualFeatureChainRunner dry-run skeleton smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
