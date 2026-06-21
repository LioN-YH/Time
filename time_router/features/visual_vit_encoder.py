"""
文件功能：
    Stage 1 P19b guarded real ViT VisualEncoderProvider adapter。

设计边界：
    本模块提供 `VisualVitEncoderProvider.encode(batch)`，把 P16f
    `VisualInputBatch` 编码为 `VisualEmbeddingBatch`。模块 import 阶段不导入
    transformers；`ViTModel` / `AutoImageProcessor` 只允许在显式构造函数内部
    lazy import。provider 不接收 run_dir，不搜索模型路径，不启动训练或 full-scale。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from time_router.features.visual_chain import VisualEmbeddingBatch, VisualInputBatch
from time_router.runtime.visual_vit_guard import VisualVitModelPathPolicy, authorize_visual_vit_model_paths


def _policy_metadata(policy: VisualVitModelPathPolicy | Mapping[str, Any] | None) -> dict[str, Any]:
    """
    函数功能：
        将 guard policy 转换为 provider metadata，确保 P19b 要求字段稳定存在。
    """
    if policy is None:
        return {
            "loads_real_vit": True,
            "model_path_policy": "dependency_injected_no_path_policy",
            "processor_path_policy": "dependency_injected_no_path_policy",
            "allow_real_vit": False,
            "allow_external_vit_path": False,
        }
    if isinstance(policy, VisualVitModelPathPolicy):
        return {
            "loads_real_vit": bool(policy.loads_real_vit),
            "model_path_policy": policy.model_path_policy,
            "processor_path_policy": policy.processor_path_policy,
            "allow_real_vit": bool(policy.allow_real_vit),
            "allow_external_vit_path": bool(policy.allow_external_vit_path),
        }
    return {
        "loads_real_vit": bool(policy.get("loads_real_vit", True)),
        "model_path_policy": str(policy.get("model_path_policy", "dependency_injected_no_path_policy")),
        "processor_path_policy": str(policy.get("processor_path_policy", "dependency_injected_no_path_policy")),
        "allow_real_vit": bool(policy.get("allow_real_vit", False)),
        "allow_external_vit_path": bool(policy.get("allow_external_vit_path", False)),
    }


@dataclass
class VisualVitEncoderProvider:
    """
    类功能：
        实现 P16f `VisualEncoderProvider` protocol 的 real ViT adapter。

    输入：
        processor: 已构造的 HF processor 或测试注入对象；
        model: 已构造的 ViT model 或测试注入对象；
        policy: runtime guard 返回的 path policy 或等价 metadata；
        device: model/input tensor 目标设备。

    输出：
        `encode(batch)` 返回三维 `float32` `VisualEmbeddingBatch`，保留输入
        sample_key 顺序，并在 metadata 中记录 guarded real ViT 边界。

    关键约束：
        构造器不接收 run_dir；provider 不训练、不下载、不搜索路径。真实
        transformers artifact 的加载由 `build_visual_vit_encoder_provider(...)`
        在显式授权后完成。
    """

    processor: Any
    model: Any
    policy: VisualVitModelPathPolicy | Mapping[str, Any] | None = None
    device: str = "cpu"
    provider_name: str = "VisualVitEncoderProvider"
    mocked_real_vit: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """函数功能：尽量把已注入模型切到 eval；测试 fake model 可无该方法。"""
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def encode(self, batch: VisualInputBatch) -> VisualEmbeddingBatch:
        """
        函数功能：
            使用已构造 processor/model 编码 `VisualInputBatch`。

        输入：
            batch.images 由上游 pseudo image / resize policy 显式提供。

        输出：
            `VisualEmbeddingBatch.embeddings` 为 `[sample, token, dim]` 的
            `numpy.float32` 数组，sample_keys 与输入完全一致。
        """
        ordered_keys = tuple(str(sample_key) for sample_key in batch.sample_keys)
        if not ordered_keys:
            raise ValueError("VisualVitEncoderProvider.encode 需要非空 sample_keys")

        processed = self.processor(batch.images, return_tensors="pt")
        model_inputs = self._move_model_inputs(processed)
        outputs = self._run_model(model_inputs)
        embeddings = self._extract_embeddings(outputs, expected_samples=len(ordered_keys))

        metadata = {
            "stage": "encoder",
            "component": self.provider_name,
            "encoder_provider": self.provider_name,
            "lazy_transformers_import": True,
            "training_started": False,
            "formal_training_migration": False,
            "input_stage": batch.metadata.get("stage"),
            "shape": tuple(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "mocked_real_vit": bool(self.mocked_real_vit),
        }
        metadata.update(_policy_metadata(self.policy))
        metadata.update(dict(self.extra_metadata))
        return VisualEmbeddingBatch(sample_keys=ordered_keys, embeddings=embeddings, metadata=metadata)

    def _move_model_inputs(self, processed: Any) -> Any:
        """
        函数功能：
            将 processor 输出中的 tensor-like 对象移动到目标 device。
        """
        if not isinstance(processed, Mapping):
            return processed
        moved: dict[str, Any] = {}
        for key, value in processed.items():
            to_fn = getattr(value, "to", None)
            moved[key] = to_fn(self.device) if callable(to_fn) else value
        return moved

    def _run_model(self, model_inputs: Any) -> Any:
        """
        函数功能：
            在 no_grad 下调用模型；如果环境没有 torch，则退化为普通调用。
        """
        try:
            import torch
        except ImportError:
            if isinstance(model_inputs, Mapping):
                return self.model(**model_inputs)
            return self.model(model_inputs)

        with torch.no_grad():
            if isinstance(model_inputs, Mapping):
                return self.model(**model_inputs)
            return self.model(model_inputs)

    @staticmethod
    def _extract_embeddings(outputs: Any, *, expected_samples: int) -> np.ndarray:
        """
        函数功能：
            从 HF output 或测试 fake output 中取 last_hidden_state 并转为 float32。
        """
        hidden_state = getattr(outputs, "last_hidden_state", None)
        if hidden_state is None and isinstance(outputs, Mapping):
            hidden_state = outputs.get("last_hidden_state")
        if hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden_state = outputs[0]
        if hidden_state is None:
            raise ValueError("VisualVitEncoderProvider model output 缺少 last_hidden_state")

        detach_fn = getattr(hidden_state, "detach", None)
        if callable(detach_fn):
            hidden_state = detach_fn()
        cpu_fn = getattr(hidden_state, "cpu", None)
        if callable(cpu_fn):
            hidden_state = cpu_fn()
        numpy_fn = getattr(hidden_state, "numpy", None)
        if callable(numpy_fn):
            embeddings = numpy_fn()
        else:
            embeddings = np.asarray(hidden_state)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 3:
            raise ValueError(f"VisualVitEncoderProvider embeddings 必须是三维：actual_shape={embeddings.shape}")
        if int(embeddings.shape[0]) != expected_samples:
            raise ValueError(
                "VisualVitEncoderProvider embeddings 第一维必须与 sample_keys 对齐："
                f"expected={expected_samples}, actual={embeddings.shape[0]}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("VisualVitEncoderProvider embeddings 包含 NaN 或 Inf")
        return embeddings.astype(np.float32, copy=False)


def build_visual_vit_encoder_provider(
    *,
    model_path: str | Path,
    processor_path: str | Path | None = None,
    repo_root: str | Path,
    allow_real_vit: bool,
    allow_external_vit_path: bool = False,
    device: str = "cpu",
    local_files_only: bool = True,
    model_cls: Any | None = None,
    processor_cls: Any | None = None,
    provider_name: str = "VisualVitEncoderProvider",
) -> VisualVitEncoderProvider:
    """
    函数功能：
        在显式 path guard 通过后构造 `VisualVitEncoderProvider`。

    关键约束：
        只有当调用方未注入 `model_cls` / `processor_cls` 时，本函数才在函数体内
        lazy import `transformers.AutoImageProcessor` 和 `transformers.ViTModel`。
        默认 `local_files_only=True`，避免隐式联网下载模型。
    """
    policy = authorize_visual_vit_model_paths(
        model_path=model_path,
        processor_path=processor_path,
        repo_root=repo_root,
        allow_real_vit=allow_real_vit,
        allow_external_vit_path=allow_external_vit_path,
    )

    if processor_cls is None or model_cls is None:
        from transformers import AutoImageProcessor, ViTModel

        processor_cls = processor_cls or AutoImageProcessor
        model_cls = model_cls or ViTModel

    processor_source = policy.processor_path or policy.model_path
    processor = processor_cls.from_pretrained(str(processor_source), local_files_only=local_files_only)
    model = model_cls.from_pretrained(str(policy.model_path), local_files_only=local_files_only)
    to_fn = getattr(model, "to", None)
    if callable(to_fn):
        model = to_fn(device)
    return VisualVitEncoderProvider(
        processor=processor,
        model=model,
        policy=policy,
        device=device,
        provider_name=provider_name,
        mocked_real_vit=model_cls.__module__.startswith("tests.") or processor_cls.__module__.startswith("tests."),
    )
