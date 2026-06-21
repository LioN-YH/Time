#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P19b guarded real ViT encoder provider dry-run contract smoke。

输入：
    默认只使用内存 tiny pseudo image、注入式 fake processor/model 和显式
    fixture/tmp path；manual real ViT dry-run 仅在环境变量显式提供时运行。

输出：
    标准输出打印中文检查日志；若 import boundary、path guard、VisualVitEncoderProvider
    输出 contract、VisualFeatureChainRunner integration 或 manual skip 口径漂移，
    则抛出 AssertionError。

关键约束：
    默认 smoke 不导入 transformers，不访问 `/data2`，不下载 HuggingFace 模型，
    不训练，不启动 full-scale，不修改正式 streaming 训练入口。
"""

from __future__ import annotations

import json
import os
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


STREAMING_ENTRYPOINT_PATH = (
    REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"
)
TOKEN_COUNT = 5
EMBED_DIM = 4
FEATURE_DIM = 4
MODEL_COLUMNS = ("PatchTST", "DLinear", "CrossFormer")
ATOL = 1e-6


@dataclass(frozen=True)
class ConstantRawWindowProvider:
    """类功能：按输入 sample_keys 返回内存固定 raw window。"""

    def load_batch(self, sample_keys: Sequence[str]) -> Any:
        """函数功能：构造 P19b chain integration 使用的 RawWindowBatch。"""
        from time_router.features import RawWindowBatch

        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        windows = np.arange(len(ordered_keys) * 6, dtype=np.float32).reshape(len(ordered_keys), 6)
        return RawWindowBatch(
            sample_keys=ordered_keys,
            windows=windows,
            metadata={"stage": "raw_window", "component": "p19b_constant_raw_window"},
        )


class IdentityPreImageTransform:
    """类功能：测试用 identity pre-image transform。"""

    def transform(self, batch: Any) -> Any:
        """函数功能：保持 sample_keys 顺序并复制 values。"""
        from time_router.features import PreImageBatch

        return PreImageBatch(
            sample_keys=batch.sample_keys,
            values=np.asarray(batch.windows, dtype=np.float32).copy(),
            metadata={"stage": "pre_image", "component": "p19b_identity_pre_image"},
        )


class TinyPseudoImageTransform:
    """类功能：将 raw window 转换为 tiny CHW pseudo image。"""

    def transform(self, batch: Any) -> Any:
        """函数功能：构造固定 `[N, 3, 4, 4]` pseudo image。"""
        from time_router.features import VisualInputBatch

        values = np.asarray(batch.values, dtype=np.float32)
        mean = np.mean(values, axis=1, keepdims=True).reshape(values.shape[0], 1, 1, 1)
        grid = np.linspace(0.0, 1.0, num=16, dtype=np.float32).reshape(1, 1, 4, 4)
        images = np.concatenate([mean + grid, mean * 0.5 + grid, mean * 0.25 + grid], axis=1).astype(np.float32)
        return VisualInputBatch(
            sample_keys=batch.sample_keys,
            images=images,
            metadata={"stage": "pseudo_image", "component": "p19b_tiny_pseudo_image"},
        )


class IdentityResizePolicy:
    """类功能：测试用 resize/input policy。"""

    def apply(self, batch: Any) -> Any:
        """函数功能：返回复制后的 VisualInputBatch，不执行真实 resize。"""
        from time_router.features import VisualInputBatch

        images = np.asarray(batch.images, dtype=np.float32).copy()
        return VisualInputBatch(
            sample_keys=batch.sample_keys,
            images=images,
            metadata={"stage": "resize_policy", "component": "p19b_identity_resize_policy"},
        )


class MeanPatchPoolingStrategy:
    """类功能：将 ViT token embedding 池化为 canonical FeatureBatch。"""

    def pool(self, batch: Any) -> Any:
        """函数功能：对 patch tokens 做 mean pooling，并输出二维 float32 features。"""
        from time_router.protocols import FeatureBatch

        embeddings = np.asarray(batch.embeddings, dtype=np.float32)
        if embeddings.shape[1:] != (TOKEN_COUNT, EMBED_DIM):
            raise AssertionError(f"embedding shape 漂移：{embeddings.shape}")
        features = np.mean(embeddings[:, 1:, :], axis=1).astype(np.float32)
        return FeatureBatch(
            sample_keys=batch.sample_keys,
            features=features,
            feature_schema={
                "stage": "pooling_strategy",
                "component": "p19b_mean_patch_pooling",
                "feature_dim": int(features.shape[1]),
                "dtype": str(features.dtype),
            },
            extra={"pooling_metadata": {"component": "p19b_mean_patch_pooling"}},
        )


class InjectedProcessor:
    """类功能：替代 AutoImageProcessor 的注入式 fake processor。"""

    def __call__(self, images: Any, *, return_tensors: str) -> dict[str, torch.Tensor]:
        """函数功能：把 numpy image batch 转为 torch tensor。"""
        if return_tensors != "pt":
            raise AssertionError(f"fake processor 只接受 return_tensors='pt'：actual={return_tensors}")
        return {"pixel_values": torch.as_tensor(np.asarray(images, dtype=np.float32))}


class InjectedVitOutput:
    """类功能：模拟 HF ViT output，提供 last_hidden_state。"""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class InjectedVitModel(torch.nn.Module):
    """类功能：替代 ViTModel 的注入式 fake model。"""

    def forward(self, *, pixel_values: torch.Tensor) -> InjectedVitOutput:
        """函数功能：用 deterministic 规则生成三维 token embedding。"""
        flattened = pixel_values.reshape(pixel_values.shape[0], -1)
        mean = torch.mean(flattened, dim=1, keepdim=True)
        maximum = torch.max(flattened, dim=1, keepdim=True).values
        token_axis = torch.arange(TOKEN_COUNT, dtype=torch.float32).reshape(1, TOKEN_COUNT, 1)
        dim_axis = torch.arange(EMBED_DIM, dtype=torch.float32).reshape(1, 1, EMBED_DIM)
        hidden = mean[:, None, :] + maximum[:, None, :] * 0.05 + token_axis * 0.1 + dim_axis * 0.01
        return InjectedVitOutput(last_hidden_state=hidden)


class TinyLoadedMLP(torch.nn.Module):
    """类功能：用于验证 FeatureBatch 可接 small MLP adapter。"""

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：输出专家 logits。"""
        return self.net(features)


def assert_default_import_boundary(before_streaming_bytes: bytes) -> None:
    """函数功能：验证默认 import 不导入 transformers、不触碰训练入口。"""
    if "transformers" in sys.modules:
        raise AssertionError("smoke 进程启动时不应已导入 transformers")
    import time_router.features as features
    import time_router.runtime as runtime

    if "transformers" in sys.modules:
        raise AssertionError("import time_router.features/runtime 不应导入 transformers")
    if not hasattr(features, "VisualVitEncoderProvider"):
        raise AssertionError("time_router.features 缺少 VisualVitEncoderProvider public bridge")
    if not hasattr(runtime, "authorize_visual_vit_model_paths"):
        raise AssertionError("time_router.runtime 缺少 authorize_visual_vit_model_paths public bridge")
    source = (REPO_ROOT / "time_router" / "features" / "__init__.py").read_text(encoding="utf-8")
    for token in ("ViTModel", "AutoImageProcessor", "from transformers", "import transformers"):
        if token in source:
            raise AssertionError(f"features package 入口不应包含 eager transformers token：{token}")
    if STREAMING_ENTRYPOINT_PATH.read_bytes() != before_streaming_bytes:
        raise AssertionError("P19b smoke 不应修改 train_visual_router_online_streaming.py")


def assert_guard_policy() -> None:
    """函数功能：验证 fixture/tmp、未授权 real path 和 `/data2` 双授权策略。"""
    from time_router.runtime import authorize_visual_vit_model_paths

    fixture_policy = authorize_visual_vit_model_paths(
        model_path=REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_chain_dryrun" / "tiny-vit",
        processor_path=None,
        repo_root=REPO_ROOT,
        allow_real_vit=False,
        allow_external_vit_path=False,
    )
    if fixture_policy.model_path_policy != "default_fixture_or_tmp_vit_model":
        raise AssertionError(f"fixture/tmp model policy 异常：{fixture_policy}")
    if fixture_policy.processor_path_policy != "same_as_model_path":
        raise AssertionError(f"processor fallback policy 异常：{fixture_policy}")

    try:
        authorize_visual_vit_model_paths(
            model_path=REPO_ROOT / "not_fixture_real_vit",
            processor_path=None,
            repo_root=REPO_ROOT,
            allow_real_vit=False,
            allow_external_vit_path=False,
        )
    except ValueError as exc:
        if "--allow-real-vit" not in str(exc):
            raise AssertionError(f"未授权 real path 错误信息缺少 allow flag：{exc}") from exc
    else:
        raise AssertionError("非 fixture/tmp ViT path 未授权时必须 fail-fast")

    try:
        authorize_visual_vit_model_paths(
            model_path="/data2/syh/manual-vit",
            processor_path="/data2/syh/manual-processor",
            repo_root=REPO_ROOT,
            allow_real_vit=True,
            allow_external_vit_path=False,
        )
    except ValueError as exc:
        if "--allow-external-vit-path" not in str(exc):
            raise AssertionError(f"/data2 未授权错误信息缺少 external flag：{exc}") from exc
    else:
        raise AssertionError("/data2 path 未开启 allow_external_vit_path 时必须 fail-fast")

    with patch.object(Path, "exists", side_effect=AssertionError("guard 不应读取或检查 /data2 文件")):
        data2_policy = authorize_visual_vit_model_paths(
            model_path="/data2/syh/manual-vit",
            processor_path="/data2/syh/manual-processor",
            repo_root=REPO_ROOT,
            allow_real_vit=True,
            allow_external_vit_path=True,
        )
    if data2_policy.model_path_policy != "explicit_real_vit_model_external_data2_authorized":
        raise AssertionError(f"/data2 model policy 异常：{data2_policy}")
    if "transformers" in sys.modules:
        raise AssertionError("guard policy 不应导入 transformers")


def build_injected_provider() -> Any:
    """函数功能：构造注入式 VisualVitEncoderProvider，不依赖真实 transformers。"""
    from time_router.features import VisualVitEncoderProvider

    return VisualVitEncoderProvider(
        processor=InjectedProcessor(),
        model=InjectedVitModel(),
        policy={
            "loads_real_vit": True,
            "model_path_policy": "dependency_injected_fixture_or_tmp_model",
            "processor_path_policy": "dependency_injected_fixture_or_tmp_processor",
            "allow_real_vit": False,
            "allow_external_vit_path": False,
        },
        device="cpu",
        mocked_real_vit=True,
    )


def assert_provider_encode_contract(provider: Any) -> None:
    """函数功能：验证注入式 provider 输出 VisualEmbeddingBatch contract。"""
    from time_router.features import VisualInputBatch

    sample_keys = ("sample-b", "sample-a", "sample-c")
    images = np.arange(len(sample_keys) * 3 * 4 * 4, dtype=np.float32).reshape(len(sample_keys), 3, 4, 4) / 100.0
    batch = VisualInputBatch(sample_keys=sample_keys, images=images, metadata={"stage": "resize_policy"})
    embedding_batch = provider.encode(batch)
    if embedding_batch.sample_keys != sample_keys:
        raise AssertionError(f"VisualEmbeddingBatch sample_keys 未保序：{embedding_batch.sample_keys}")
    embeddings = np.asarray(embedding_batch.embeddings)
    if embeddings.shape != (len(sample_keys), TOKEN_COUNT, EMBED_DIM):
        raise AssertionError(f"VisualEmbeddingBatch shape 异常：{embeddings.shape}")
    if embeddings.dtype != np.float32:
        raise AssertionError(f"VisualEmbeddingBatch dtype 必须为 float32：actual={embeddings.dtype}")
    if not np.all(np.isfinite(embeddings)):
        raise AssertionError("VisualEmbeddingBatch embeddings 包含 NaN 或 Inf")
    metadata = dict(embedding_batch.metadata)
    expected_metadata = {
        "encoder_provider": "VisualVitEncoderProvider",
        "loads_real_vit": True,
        "lazy_transformers_import": True,
        "training_started": False,
        "formal_training_migration": False,
        "mocked_real_vit": True,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise AssertionError(f"encoder metadata {key} 异常：actual={metadata.get(key)!r} metadata={metadata}")


def assert_chain_and_eval(provider: Any) -> None:
    """函数功能：验证注入式 provider 可接 VisualFeatureChainRunner 和 eval adapter。"""
    from time_router.evaluation import EvaluationInputAdapter
    from time_router.features import VisualFeatureChainRunner, VisualFeatureChainSpec
    from time_router.models import LoadedTorchMLPRouterHeadAdapter
    from time_router.protocols import ExpertBatch

    sample_keys = ("sample-b", "sample-a", "sample-c")
    spec = VisualFeatureChainSpec(
        raw_window_provider=ConstantRawWindowProvider(),
        pre_image_transform=IdentityPreImageTransform(),
        pseudo_image_transformer=TinyPseudoImageTransform(),
        resize_policy=IdentityResizePolicy(),
        visual_encoder_provider=provider,
        pooling_strategy=MeanPatchPoolingStrategy(),
        feature_transform=None,
        chain_name="stage1_p19b_visual_vit_encoder_guard",
        metadata={"purpose": "P19b guarded ViT provider chain smoke"},
    )
    feature_batch = VisualFeatureChainRunner(spec=spec).run(sample_keys).feature_batch
    features = np.asarray(feature_batch.features)
    if feature_batch.sample_keys != sample_keys or features.shape != (len(sample_keys), FEATURE_DIM):
        raise AssertionError(f"FeatureBatch contract 异常：keys={feature_batch.sample_keys} shape={features.shape}")
    if feature_batch.feature_schema.get("encoder") != "VisualVitEncoderProvider":
        raise AssertionError(f"FeatureBatch schema 未记录真实 provider：{feature_batch.feature_schema}")
    encoder_metadata = feature_batch.extra.get("chain_metadata", {}).get("stage_metadata", {}).get("encoder", {})
    if encoder_metadata.get("loads_real_vit") is not True or encoder_metadata.get("mocked_real_vit") is not True:
        raise AssertionError(f"FeatureBatch encoder metadata 异常：{encoder_metadata}")

    torch.manual_seed(20260621)
    model = TinyLoadedMLP(input_dim=FEATURE_DIM, output_dim=len(MODEL_COLUMNS))
    adapter = LoadedTorchMLPRouterHeadAdapter(model=model, device=torch.device("cpu"))
    router_output = adapter.predict(feature_batch, MODEL_COLUMNS)
    y_pred = np.stack(
        [
            np.full((2,), 0.4, dtype=np.float32),
            np.full((2,), 0.6, dtype=np.float32),
            np.full((2,), 0.8, dtype=np.float32),
        ],
        axis=0,
    )
    expert_batch = ExpertBatch(
        sample_keys=sample_keys,
        model_columns=MODEL_COLUMNS,
        y_pred=np.stack([y_pred + idx * 0.01 for idx in range(len(sample_keys))], axis=0).astype(np.float32),
        y_true=np.stack([np.asarray([0.5, 0.7], dtype=np.float32) for _ in sample_keys], axis=0),
        row_index_metadata={"source": "p19b_in_memory"},
        extra={"provider_name": "P19bInMemoryExpertBatch"},
    )
    result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)
    if result.evaluation_input.sample_keys != sample_keys:
        raise AssertionError(f"EvaluationInput sample_keys 未保序：{result.evaluation_input.sample_keys}")
    weights = np.asarray(router_output.weights)
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(weights.shape[0]), rtol=0.0, atol=ATOL)


def maybe_run_manual_real_vit_dryrun() -> None:
    """函数功能：环境变量显式授权时运行真实 ViT dry-run，否则默认跳过。"""
    model_path = os.environ.get("STAGE1_VISUAL_REAL_VIT_MODEL_PATH")
    allow_real = os.environ.get("STAGE1_VISUAL_REAL_VIT_ALLOW_REAL") == "1"
    if not model_path or not allow_real:
        print("跳过：manual real ViT dry-run 未设置 STAGE1_VISUAL_REAL_VIT_MODEL_PATH 和 STAGE1_VISUAL_REAL_VIT_ALLOW_REAL=1")
        return

    from time_router.features import VisualInputBatch, build_visual_vit_encoder_provider

    processor_path = os.environ.get("STAGE1_VISUAL_REAL_VIT_PROCESSOR_PATH") or None
    allow_external = os.environ.get("STAGE1_VISUAL_REAL_VIT_ALLOW_EXTERNAL") == "1"
    provider = build_visual_vit_encoder_provider(
        model_path=model_path,
        processor_path=processor_path,
        repo_root=REPO_ROOT,
        allow_real_vit=True,
        allow_external_vit_path=allow_external,
        device="cpu",
        local_files_only=True,
    )
    batch = VisualInputBatch(
        sample_keys=("manual-real-vit-sample",),
        images=np.zeros((1, 3, 224, 224), dtype=np.float32),
        metadata={"stage": "manual_real_vit_input"},
    )
    embedding_batch = provider.encode(batch)
    embeddings = np.asarray(embedding_batch.embeddings)
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.dtype != np.float32:
        raise AssertionError(f"manual real ViT embedding contract 异常：shape={embeddings.shape} dtype={embeddings.dtype}")
    print(
        "通过：manual real ViT dry-run 输出 VisualEmbeddingBatch，"
        f"shape={embeddings.shape} metadata={json.dumps(embedding_batch.metadata, ensure_ascii=False, default=str)}"
    )


def assert_no_forbidden_side_effects(before_streaming_bytes: bytes) -> None:
    """函数功能：确认默认路径没有修改训练入口或导入 transformers。"""
    if "transformers" in sys.modules:
        raise AssertionError("默认 P19b smoke 不应导入 transformers")
    if STREAMING_ENTRYPOINT_PATH.read_bytes() != before_streaming_bytes:
        raise AssertionError("P19b smoke 不应修改 train_visual_router_online_streaming.py")


def run_smoke() -> None:
    """函数功能：执行 P19b guarded VisualVitEncoderProvider smoke。"""
    print("开始 Stage 1 P19b guarded VisualVitEncoderProvider smoke")
    before_streaming_bytes = STREAMING_ENTRYPOINT_PATH.read_bytes()

    assert_default_import_boundary(before_streaming_bytes)
    print("通过：默认 import time_router.features/runtime 未导入 transformers，且未触碰正式 streaming 训练入口")

    assert_guard_policy()
    print("通过：ViT model/processor path guard 覆盖 fixture/tmp、未授权 real path 和 /data2 双授权")

    provider = build_injected_provider()
    assert_provider_encode_contract(provider)
    print("通过：注入式 VisualVitEncoderProvider 输出保序三维 float32 finite VisualEmbeddingBatch")

    assert_chain_and_eval(provider)
    print("通过：注入式 VisualVitEncoderProvider 可接 VisualFeatureChainRunner、small MLP adapter 和 EvaluationInputAdapter")

    maybe_run_manual_real_vit_dryrun()
    assert_no_forbidden_side_effects(before_streaming_bytes)
    print("完成：Stage 1 P19b guarded VisualVitEncoderProvider smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
