#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 online Visual Router 与历史离线 embedding pilot 共用的 ViT 输入和
    输出处理工具。

设计约束：
    - 本文件只负责运行内 tensor 构造、encoder dtype 解析和 ViT 输出池化；
    - 不保存 `.npy`，不生成 embedding manifest，不创建长期 cache；
    - 需要落盘 embedding 的旧 smoke / 对照流程应放在 stage1 `pilot/` 中。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from visual_router_experiments.common.pseudo_imageization import (
    encoder_normalize,
    imageize_3view,
    imageize_top3fold,
    make_default_period_candidates,
    normalize_window,
    parse_period_candidates,
    select_fft_periods,
)


EMBEDDING_VERSION = "visual_router_vit_embedding_v1"


def build_required_index(windows_df) -> Mapping[Tuple[str, str, int], Set[Tuple[int, int, str]]]:
    """函数功能：把待覆盖窗口整理为 split/dataset/item -> channel/window/sample_key 索引。"""
    required: Dict[Tuple[str, str, int], Set[Tuple[int, int, str]]] = {}
    for row in windows_df.itertuples(index=False):
        group_key = (str(row.split), str(row.dataset_name), int(row.item_id))
        required.setdefault(group_key, set()).add((int(row.channel_id), int(row.window_index), str(row.sample_key)))
    return required


def batch_required_pairs(required_for_item: Set[Tuple[int, int, str]], batch_size: int) -> List[List[Tuple[int, int, str]]]:
    """函数功能：按稳定顺序将 required window 切成小 batch。"""
    sorted_pairs = sorted(required_for_item, key=lambda item: (item[0], item[1], item[2]))
    return [sorted_pairs[start : start + batch_size] for start in range(0, len(sorted_pairs), batch_size)]


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """函数功能：解析 ViT 前向 dtype；CPU 路径保持 fp32，避免半精度算子兼容问题。"""
    if device.type == "cpu":
        return torch.float32
    if dtype_arg == "fp32":
        return torch.float32
    return torch.float16


def parse_period_candidate_arg(candidate_text: Optional[str]) -> Optional[List[int]]:
    """
    函数功能：
        解析命令行传入的逗号分隔候选周期。

    输入：
        candidate_text: 形如 `2,3,4,6,8,12,24,48,96` 的字符串；None 表示自动。

    输出：
        int 列表或 None。
    """
    if candidate_text is None or str(candidate_text).strip() == "":
        return None
    values: List[int] = []
    for part in str(candidate_text).split(","):
        text = part.strip()
        if not text:
            continue
        value = int(text)
        if value < 2:
            raise ValueError(f"period candidate 必须 >=2，实际为 {value}")
        values.append(value)
    if not values:
        raise ValueError("--period-candidates 解析后为空")
    return values


def make_pseudo_images(
    x_batch: torch.Tensor,
    *,
    variant: str,
    norm_mode: str,
    pixel_mode: str,
    clip: float,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    normalization_preset: str,
    period_selection: str,
    period_candidate_values: Optional[Sequence[int]],
) -> torch.Tensor:
    """
    函数功能：
        从历史窗口 batch 构造 ViT `pixel_values`。

    关键约束：
        只使用历史 x。`encoder_normalize()` 在伪图像 [0,1] 输出之后执行，
        这里不走 HF processor，避免 rescale / normalize 口径隐式变化。
    """
    x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=False)
    x_norm, _ = normalize_window(x_batch, norm_mode=norm_mode)
    series_length = int(x_norm.shape[1])
    period_candidates = None
    if period_selection == "fixed_candidates":
        period_candidates = parse_period_candidates(
            period_candidate_values,
            history_length=series_length,
            device=device,
        )
        if period_candidates is None:
            period_candidates = make_default_period_candidates(series_length, device=device)
    elif period_selection != "dynamic_fft_topk":
        raise ValueError(f"未知 period_selection={period_selection}")
    periods = select_fft_periods(x_norm, top_k=3, period_candidates=period_candidates)
    if variant == "variant_a_3view":
        images = imageize_3view(
            x_norm,
            image_size=image_size,
            periods=periods,
            period_candidates=period_candidates,
            pixel_mode=pixel_mode,
            clip=clip,
        )
    elif variant == "variant_b_top3fold":
        images = imageize_top3fold(
            x_norm,
            image_size=image_size,
            periods=periods,
            period_candidates=period_candidates,
            pixel_mode=pixel_mode,
            clip=clip,
        )
    else:
        raise ValueError(f"未知 variant={variant}")
    return encoder_normalize(images.to(dtype=dtype), preset=normalization_preset)


def pool_vit_outputs(outputs, pooling: str) -> torch.Tensor:
    """
    函数功能：
        将 ViT 输出聚合成单个视觉特征向量。

    说明：
        默认使用 last_hidden_state 的 CLS token，这是 Hugging Face ViT 分类头前最常见
        的图像级表示；mean_patch 可作为后续 ablation。
    """
    if pooling == "cls":
        return outputs.last_hidden_state[:, 0]
    if pooling == "mean_patch":
        return outputs.last_hidden_state[:, 1:].mean(dim=1)
    if pooling == "pooler":
        if outputs.pooler_output is None:
            raise ValueError("当前 encoder 输出没有 pooler_output，不能使用 --pooling pooler")
        return outputs.pooler_output
    raise ValueError(f"未知 pooling={pooling}")
