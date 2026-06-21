"""
文件功能：
    提供 Stage 1 P16a 正式 Visual MLP RouterHead adapter 边界。

设计边界：
    本 adapter 只包装 Runtime 已经加载完成的 torch.nn.Module，并把
    head-ready float32 FeatureBatch.features 前向映射为 RouterOutput。
    checkpoint loading、scaler transform、device/DataParallel 选择、ViT
    embedding、pseudo image 构造、run_dir 和训练循环都属于 Runtime 或
    entrypoint，不属于这里。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from time_router.protocols import FeatureBatch, RouterOutput


class LoadedTorchMLPRouterHeadAdapter:
    """
    类功能：
        将“已加载 torch module + head-ready FeatureBatch”适配为 RouterOutput。

    输入：
        model: Runtime 已实例化并加载好权重的 torch.nn.Module；
        device: adapter 前向使用的 torch device，默认 cpu；
        adapter_name: 写入 RouterOutput.extra 的轻量 lineage 名称。

    输出：
        `predict(feature_batch, model_columns)` 返回 RouterOutput，其中
        sample_keys 保持 FeatureBatch 原顺序，model_columns 保持调用方输入顺序。

    关键约束：
        adapter 不读取 checkpoint，不处理 scaler，不启动 ViT，不知道 run_dir。
        P16a 仅支持 `model(features)` 直接返回二维 logits Tensor；tuple/dict
        或额外中间结果兼容留给后续步骤。
    """

    default_adapter_name = "LoadedTorchMLPRouterHeadAdapter"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device | str = "cpu",
        adapter_name: str | None = None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model 必须是 torch.nn.Module：actual={type(model)!r}")
        self.model = model
        self.device = torch.device(device)
        self.adapter_name = adapter_name or self.default_adapter_name
        # Runtime 负责 checkpoint/DataParallel 等决策；adapter 只把已加载 module
        # 移到调用方指定设备并切换 eval，确保 smoke 前向不会更新训练状态。
        self.model.to(self.device)
        self.model.eval()

    def predict(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """
        函数功能：
            在 torch.inference_mode() 下执行已加载 MLP，并 softmax 为专家权重。

        输入：
            feature_batch: 已经完成 scaler/ViT/pre-head transform 的 float32
                二维 head-ready features；
            model_columns: 专家动作空间顺序，必须非空且无重复。

        输出：
            RouterOutput；logits/weights 均为 numpy.float32 二维矩阵。
        """
        columns = self._validate_model_columns(model_columns)
        features = self._validate_feature_batch(feature_batch)
        sample_keys = tuple(str(sample_key) for sample_key in feature_batch.sample_keys)

        feature_tensor = torch.from_numpy(features).to(device=self.device)
        with torch.inference_mode():
            logits_tensor = self.model(feature_tensor)
            self._validate_logits_tensor(logits_tensor=logits_tensor, num_samples=len(sample_keys), num_models=len(columns))
            weights_tensor = torch.softmax(logits_tensor, dim=1)

        logits = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        weights = weights_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return RouterOutput(
            sample_keys=sample_keys,
            model_columns=columns,
            logits=logits,
            weights=weights,
            extra={
                "adapter_name": self.adapter_name,
                "loaded_model_boundary": "runtime_supplied_torch_nn_module",
                "feature_contract": "head_ready_float32_features",
                "loads_checkpoint": False,
                "handles_scaler": False,
                "handles_vit": False,
            },
        )

    def __call__(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """函数功能：提供与 RouterHead 常见调用习惯一致的转发入口。"""
        return self.predict(feature_batch, model_columns)

    def _validate_model_columns(self, model_columns: Sequence[str]) -> tuple[str, ...]:
        """
        函数功能：
            校验专家列顺序，避免 adapter 静默接受空动作空间或重复专家。
        """
        columns = tuple(str(model_name) for model_name in model_columns)
        if not columns:
            raise ValueError("LoadedTorchMLPRouterHeadAdapter.predict 需要非空 model_columns")
        if len(columns) != len(set(columns)):
            raise ValueError(f"LoadedTorchMLPRouterHeadAdapter.predict 收到重复 model_columns：{columns}")
        return columns

    def _validate_feature_batch(self, feature_batch: FeatureBatch) -> np.ndarray:
        """
        函数功能：
            校验 FeatureBatch.features 已是二维 float32 head-ready 特征。

        关键约束：
            这里不做 dtype 修复或 scaler transform；非 float32 直接失败，避免
            Runtime 前置处理缺失时被 adapter 静默掩盖。
        """
        features = np.asarray(feature_batch.features)
        if features.dtype != np.float32:
            raise ValueError(f"FeatureBatch.features 必须是 head-ready float32：actual={features.dtype}")
        if features.ndim != 2:
            raise ValueError(f"FeatureBatch.features 必须是二维矩阵：actual_shape={features.shape}")
        if features.shape[0] != len(feature_batch.sample_keys):
            raise ValueError(
                "FeatureBatch.features 样本维度必须等于 sample_keys 数量："
                f"features={features.shape[0]} sample_keys={len(feature_batch.sample_keys)}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("FeatureBatch.features 包含 NaN 或 Inf")
        return features

    def _validate_logits_tensor(
        self,
        *,
        logits_tensor: object,
        num_samples: int,
        num_models: int,
    ) -> None:
        """
        函数功能：
            校验 P16a 支持的唯一模型输出：二维 torch logits Tensor。
        """
        if not isinstance(logits_tensor, torch.Tensor):
            raise TypeError(f"P16a 仅支持 model(features) 返回 torch.Tensor：actual={type(logits_tensor)!r}")
        if logits_tensor.ndim != 2:
            raise ValueError(f"Visual MLP logits 必须是二维矩阵：actual_shape={tuple(logits_tensor.shape)}")
        if tuple(logits_tensor.shape) != (num_samples, num_models):
            raise ValueError(
                "Visual MLP logits shape 必须等于 [num_samples, num_models]："
                f"actual={tuple(logits_tensor.shape)} expected={(num_samples, num_models)}"
            )
        if not torch.isfinite(logits_tensor).all().item():
            raise ValueError("Visual MLP logits 包含 NaN 或 Inf")
