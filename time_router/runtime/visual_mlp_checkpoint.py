"""
文件功能：
    Stage 1 P16i legacy VisualMLPRouter tiny checkpoint payload loader helper。

设计边界：
    本 helper 只处理 Runtime 已显式传入的 checkpoint path/payload 到
    `router_state_dict` strict load 的最小边界。它不做 run_dir discovery，不访问
    `/data2`，不构造 FeatureBatch，不启动 ViT，不执行 scaler transform，也不改变
    P16a `LoadedTorchMLPRouterHeadAdapter` 的接口。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def strip_dataparallel_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    函数功能：
        清理 DataParallel 常见的 `module.` key 前缀，并返回新的 state_dict。

    输入：
        state_dict: checkpoint payload 中已经定位到的 router 权重映射。

    输出：
        清理后的 key -> tensor 字典。

    关键约束：
        清理后如果 normal key 与 `module.` key 落到同一个目标 key，立即报错，避免
        strict load 前静默覆盖权重。
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"router_state_dict 必须是 mapping：actual={type(state_dict)!r}")

    cleaned: dict[str, torch.Tensor] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"router_state_dict key 必须是 str：actual={type(raw_key)!r}")
        normalized_key = raw_key.removeprefix("module.")
        if normalized_key in cleaned:
            raise ValueError(f"state_dict 清理 `module.` 前缀后出现重复 key：{normalized_key}")
        cleaned[normalized_key] = value
    if not cleaned:
        raise ValueError("router_state_dict 不能为空")
    return cleaned


def extract_router_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """
    函数功能：
        从 checkpoint payload 中提取并清理 `router_state_dict`。

    输入：
        payload: `torch.load` 得到的 checkpoint object，必须是 mapping，并显式包含
            `router_state_dict`。

    输出：
        已清理 `module.` 前缀的 router state_dict。

    关键约束：
        本函数只识别 `router_state_dict`，不做裸 state_dict 猜测、不查找 run_dir，也不
        处理 optimizer/scaler transform。
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload 必须是 mapping：actual={type(payload)!r}")
    if "router_state_dict" not in payload:
        raise KeyError("checkpoint payload 缺少 router_state_dict")
    return strip_dataparallel_prefix(payload["router_state_dict"])


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu") -> Mapping[str, Any]:
    """
    函数功能：
        从显式 checkpoint path 读取 payload。

    输入：
        path: 调用方显式传入的 checkpoint 文件路径；
        map_location: 透传给 `torch.load` 的设备映射，默认 cpu。

    输出：
        checkpoint payload mapping。

    关键约束：
        不做 run_dir discovery，不拼接默认 checkpoint 名，不访问外部固定路径；checkpoint
        path 是 Runtime/entrypoint implementation detail，不进入 RouterHead adapter。
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 文件不存在：{checkpoint_path}")
    # P16i 只需要 tensor state_dict 与基础 dict/list/标量 metadata，使用 weights_only
    # 避免 tiny checkpoint smoke 依赖任意 pickle object 反序列化能力。
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload 必须是 mapping：actual={type(payload)!r}")
    return payload


def load_router_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    """
    函数功能：
        将 router state_dict 加载到已构造的 legacy VisualMLPRouter module。

    输入：
        model: Runtime 已实例化的 legacy VisualMLPRouter 或同签名 module；
        state_dict: 已定位到的 router state_dict，可含 `module.` 前缀；
        strict: 透传给 `torch.nn.Module.load_state_dict`，默认严格加载。

    输出：
        PyTorch `_IncompatibleKeys` 结果；strict=True 时 missing/unexpected 会由 PyTorch
        抛错。

    关键约束：
        本 helper 不构造 model，不决定 hidden_dim/output_dim，不把 checkpoint path
        暴露给 adapter；它只负责 Runtime-side strict loading 边界。
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model 必须是 torch.nn.Module：actual={type(model)!r}")
    cleaned_state_dict = strip_dataparallel_prefix(state_dict)
    return model.load_state_dict(cleaned_state_dict, strict=strict)
