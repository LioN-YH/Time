#!/usr/bin/env python3
"""
文件功能：
    提供 Visual Router 跨阶段复用的在线伪图像化张量工具。

设计约束：
    - 主路径只使用 torch tensor 操作，便于在训练/测试时 GPU-first 在线生成视觉输入；
    - 图像化本体输出 [B, 3, H, W] 且数值范围为 [0, 1]；
    - 三个 channel 表示不同语义视图，不等价于自然图像 RGB；
    - encoder normalization 独立放在视觉 encoder 前调用，不混入图像化本体；
    - 第一版显式支持 HF ViT 的 [0.5, 0.5, 0.5] mean/std，以及兼容保留的
      torchvision/ImageNet mean/std。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-6
HF_VIT_MEAN = (0.5, 0.5, 0.5)
HF_VIT_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENCODER_NORMALIZATION_PRESETS = {
    "hf_vit_0_5": (HF_VIT_MEAN, HF_VIT_STD),
    "torchvision_imagenet": (IMAGENET_MEAN, IMAGENET_STD),
}


def _as_series_batch(x: torch.Tensor) -> torch.Tensor:
    """
    函数功能：
        将 Quito 历史窗口统一转换为 [B, L] 单变量序列批次。

    输入：
        x: 支持 [L]、[L, C]、[B, L]、[B, L, C]。Stage 1 当前正式口径是
            [B, L, 1] 的 S 配置；若后续传入多通道窗口，这里先按通道均值折成
            单个视觉样本，避免隐式展开改变 sample_key 粒度。

    输出：
        [B, L] torch tensor。
    """
    if not torch.is_tensor(x):
        raise TypeError("x 必须是 torch.Tensor")

    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        # [L, C] 且 C 很小时视作单样本多通道；否则视作 [B, L]。
        if x.shape[1] == 1:
            return x.transpose(0, 1)
        return x
    if x.ndim == 3:
        if x.shape[-1] == 1:
            return x[..., 0]
        return x.mean(dim=-1)
    raise ValueError(f"x 维度必须为 1/2/3，实际为 {tuple(x.shape)}")


def normalize_window(x: torch.Tensor, norm_mode: str = "revin_aux") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    函数功能：
        对 Quito train-based normalized 历史窗口做可选 per-window normalization。

    输入：
        x: Quito 已按 train statistics 标准化后的历史窗口。
        norm_mode:
            - quito: 不额外变换，只记录当前窗口统计量；
            - revin: 对每个窗口做 mean/std normalization；
            - revin_aux: 与 revin 相同，同时默认作为后续 pilot 记录 aux metadata 的口径。

    输出：
        (x_norm, metadata)。x_norm 保持为 [B, L, 1]，metadata 的统计量为 [B] tensor。
    """
    if norm_mode not in {"quito", "revin", "revin_aux"}:
        raise ValueError(f"未知 norm_mode={norm_mode}")

    series = _as_series_batch(x).to(dtype=torch.float32)
    mean = series.mean(dim=1)
    std = series.std(dim=1, unbiased=False).clamp_min(EPS)
    min_value = series.min(dim=1).values
    max_value = series.max(dim=1).values
    value_range = max_value - min_value

    if norm_mode == "quito":
        normalized = series
    else:
        # 这里只使用历史 x 的窗口内统计量，不访问未来 y，也不使用专家误差。
        normalized = (series - mean[:, None]) / std[:, None]

    metadata = {
        "norm_mean": mean,
        "norm_std": std,
        "norm_min": min_value,
        "norm_max": max_value,
        "norm_range": value_range,
    }
    return normalized.unsqueeze(-1), metadata


def select_fft_periods(x: torch.Tensor, top_k: int = 3) -> torch.Tensor:
    """
    函数功能：
        基于历史窗口的 FFT power 选择每个样本的 top-k 周期。

    输入：
        x: 历史窗口 tensor，支持 [B, L, 1] 或 [B, L]。
        top_k: 返回周期数量，Stage 1 默认 3。

    输出：
        int64 tensor，形状 [B, top_k]，每个周期都落在 [2, history_length]。

    关键约束：
        多个频率 bin 可能 round 到同一个周期；此时按确定性的从长到短 fallback
        补足，避免 top3fold 出现完全重复 channel。
    """
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")

    series = _as_series_batch(x).to(dtype=torch.float32)
    batch_size, history_length = series.shape
    if history_length < 2:
        raise ValueError(f"history_length 必须 >= 2，实际为 {history_length}")

    centered = series - series.mean(dim=1, keepdim=True)
    fft_values = torch.fft.rfft(centered, dim=1)
    power = torch.abs(fft_values) ** 2
    non_dc_power = power[:, 1:]
    freq_bins = torch.arange(1, non_dc_power.shape[1] + 1, device=series.device, dtype=torch.float32)
    fallback_periods = list(range(history_length, 1, -1))

    selected_rows = []
    for row_idx in range(batch_size):
        selected = []
        if non_dc_power.shape[1] > 0:
            order = torch.argsort(non_dc_power[row_idx], descending=True)
            for bin_pos in order.tolist():
                period = int(torch.round(torch.tensor(history_length, device=series.device) / freq_bins[bin_pos]).item())
                period = max(2, min(history_length, period))
                if period not in selected:
                    selected.append(period)
                if len(selected) == top_k:
                    break

        for period in fallback_periods:
            if len(selected) == top_k:
                break
            if period not in selected:
                selected.append(period)

        while len(selected) < top_k:
            # history_length=2 时唯一合法周期只有 2；此分支只为极短序列兜底。
            selected.append(2)
        selected_rows.append(torch.tensor(selected[:top_k], device=series.device, dtype=torch.long))

    return torch.stack(selected_rows, dim=0)


def to_vision_pixels(x: torch.Tensor, pixel_mode: str = "vision", clip: float = 5.0) -> torch.Tensor:
    """
    函数功能：
        将标准化后的张量映射到视觉 encoder 常用的 [0, 1] pixel 空间。

    输入：
        x: 任意形状 tensor。
        pixel_mode: 当前仅支持 vision，即先 clamp 到 [-clip, clip]，再线性映射。
        clip: 对称截断阈值。

    输出：
        与输入同形状的 [0, 1] tensor。
    """
    if pixel_mode != "vision":
        raise ValueError(f"未知 pixel_mode={pixel_mode}")
    if clip <= 0:
        raise ValueError("clip 必须为正数")

    return ((x.clamp(min=-clip, max=clip) + clip) / (2.0 * clip)).clamp(0.0, 1.0)


def _clip_ratio(x: torch.Tensor, clip: float) -> torch.Tensor:
    """函数功能：计算每个窗口中超过视觉截断阈值的比例，用于审计 metadata。"""
    series = _as_series_batch(x)
    return ((series < -clip) | (series > clip)).to(torch.float32).mean(dim=1)


def _line_raster(series: torch.Tensor, image_size: int, pixel_mode: str, clip: float) -> torch.Tensor:
    """
    函数功能：
        将一维历史窗口画成无 PIL/无 matplotlib 的 line raster 视图。

    实现说明：
        先把数值映射到 [0, 1]，再对每个输出列计算与曲线 y 坐标的距离，
        用窄高斯带生成连续线条，避免 scatter 画线导致断裂。
    """
    pixels = to_vision_pixels(series, pixel_mode=pixel_mode, clip=clip)
    y_coord = (1.0 - pixels) * float(image_size - 1)
    y_coord = F.interpolate(
        y_coord.unsqueeze(1),
        size=image_size,
        mode="linear",
        align_corners=True,
    ).squeeze(1)
    rows = torch.arange(image_size, device=series.device, dtype=series.dtype).view(1, image_size, 1)
    sigma = max(float(image_size) / 160.0, 1.0)
    return torch.exp(-0.5 * ((rows - y_coord[:, None, :]) / sigma) ** 2).clamp(0.0, 1.0)


def _fold_one_period(series: torch.Tensor, period: int, image_size: int, pixel_mode: str, clip: float) -> torch.Tensor:
    """
    函数功能：
        将单个样本按指定周期折叠为 [image_size, image_size] 视图。

    约束：
        period 可能不能整除 history_length；末尾 padding 使用最后一个历史值，避免
        人为引入 0 值边界。
    """
    history_length = int(series.shape[0])
    period = max(2, min(history_length, int(period)))
    cycle_count = (history_length + period - 1) // period
    pad_count = cycle_count * period - history_length
    if pad_count:
        series = torch.cat([series, series[-1:].repeat(pad_count)], dim=0)
    folded = series.reshape(cycle_count, period)
    folded = folded.view(1, 1, cycle_count, period)
    image = F.interpolate(folded, size=(image_size, image_size), mode="bilinear", align_corners=True)
    return to_vision_pixels(image.squeeze(0).squeeze(0), pixel_mode=pixel_mode, clip=clip)


def _period_fold_batch(
    series: torch.Tensor,
    periods: torch.Tensor,
    period_column: int,
    image_size: int,
    pixel_mode: str,
    clip: float,
) -> torch.Tensor:
    """函数功能：按每个样本自己的 FFT 周期生成 period fold 视图。"""
    images = []
    for row_idx in range(series.shape[0]):
        period = int(periods[row_idx, period_column].item())
        images.append(_fold_one_period(series[row_idx], period, image_size, pixel_mode, clip))
    return torch.stack(images, dim=0)


def _fft_power_view(series: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    函数功能：
        将 FFT log power 变为 [B, H, W] 频谱强度视图。

    说明：
        当前 pilot 是轻量 2D 伪图像，不构造 STFT；频谱强度沿宽度复制，用于让
        视觉 encoder 看到窗口的主频结构。
    """
    centered = series - series.mean(dim=1, keepdim=True)
    power = torch.log1p(torch.abs(torch.fft.rfft(centered, dim=1)[:, 1:]) ** 2)
    if power.shape[1] == 0:
        return torch.zeros((series.shape[0], image_size, image_size), device=series.device, dtype=series.dtype)

    min_value = power.min(dim=1, keepdim=True).values
    max_value = power.max(dim=1, keepdim=True).values
    norm_power = (power - min_value) / (max_value - min_value).clamp_min(EPS)
    freq_profile = F.interpolate(
        norm_power.unsqueeze(1),
        size=image_size,
        mode="linear",
        align_corners=True,
    ).squeeze(1)
    return freq_profile[:, :, None].expand(-1, image_size, image_size).contiguous()


def imageize_3view(
    x: torch.Tensor,
    image_size: int = 224,
    *,
    periods: Optional[torch.Tensor] = None,
    pixel_mode: str = "vision",
    clip: float = 5.0,
) -> torch.Tensor:
    """
    函数功能：
        构造 variant_a=3view 伪图像。

    channel 语义：
        0: line_raster；
        1: top1 FFT period fold；
        2: FFT power。

    输出：
        [B, 3, image_size, image_size]，范围 [0, 1]。
    """
    if image_size <= 1:
        raise ValueError("image_size 必须大于 1")

    series = _as_series_batch(x).to(dtype=torch.float32)
    if periods is None:
        periods = select_fft_periods(series, top_k=3)
    periods = periods.to(device=series.device)

    channels = [
        _line_raster(series, image_size=image_size, pixel_mode=pixel_mode, clip=clip),
        _period_fold_batch(series, periods, 0, image_size, pixel_mode, clip),
        _fft_power_view(series, image_size=image_size),
    ]
    return torch.stack(channels, dim=1).clamp(0.0, 1.0)


def imageize_top3fold(
    x: torch.Tensor,
    image_size: int = 224,
    *,
    periods: Optional[torch.Tensor] = None,
    pixel_mode: str = "vision",
    clip: float = 5.0,
) -> torch.Tensor:
    """
    函数功能：
        构造 variant_b=top3fold 伪图像。

    channel 语义：
        0/1/2 分别为 top1/top2/top3 FFT period fold。

    输出：
        [B, 3, image_size, image_size]，范围 [0, 1]。
    """
    if image_size <= 1:
        raise ValueError("image_size 必须大于 1")

    series = _as_series_batch(x).to(dtype=torch.float32)
    if periods is None:
        periods = select_fft_periods(series, top_k=3)
    periods = periods.to(device=series.device)

    channels = [
        _period_fold_batch(series, periods, column_idx, image_size, pixel_mode, clip)
        for column_idx in range(3)
    ]
    return torch.stack(channels, dim=1).clamp(0.0, 1.0)


def _validate_encoder_image_tensor(x: torch.Tensor, caller_name: str) -> None:
    """
    函数功能：
        校验 encoder normalization 的输入 tensor 形状。

    输入：
        x: 视觉 pixel tensor，必须为 [B, 3, H, W]。
        caller_name: 调用方名称，用于错误信息定位。
    """
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"{caller_name} 需要 [B, 3, H, W]，实际为 {tuple(x.shape)}")


def encoder_normalize(x: torch.Tensor, preset: str = "hf_vit_0_5") -> torch.Tensor:
    """
    函数功能：
        在进入冻结视觉 encoder 前，按指定预训练模型口径标准化伪图像。

    输入：
        x: [B, 3, H, W] 且范围应为 [0, 1] 的视觉 pixel tensor。
        preset:
            - hf_vit_0_5: Hugging Face `google/vit-base-patch16-224` direct
              forward 口径，执行 (x - 0.5) / 0.5；
            - torchvision_imagenet: torchvision/MAE/timm 常用 ImageNet
              mean/std 口径，作为旧 `imagenet_normalize()` 的兼容路径。

    输出：
        按对应 encoder mean/std 标准化后的 tensor。

    关键约束：
        该函数用于直接构造 `pixel_values` 的路径。如果后续走 HF processor，
        需要在调用 processor 时另行处理 `do_rescale`，不要在这里混入 processor
        的 0..255 uint8 路径。
    """
    _validate_encoder_image_tensor(x, "encoder_normalize")
    if preset not in ENCODER_NORMALIZATION_PRESETS:
        available = ", ".join(sorted(ENCODER_NORMALIZATION_PRESETS))
        raise ValueError(f"未知 encoder normalization preset={preset}；可选值：{available}")

    mean_values, std_values = ENCODER_NORMALIZATION_PRESETS[preset]
    mean = torch.tensor(mean_values, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def hf_vit_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    函数功能：
        按 Hugging Face `google/vit-base-patch16-224` 的 image processor
        mean/std 口径标准化 `[0, 1]` 伪图像。

    输入：
        x: [B, 3, H, W] 且范围应为 [0, 1] 的视觉 pixel tensor。

    输出：
        执行 `(x - 0.5) / 0.5` 后的 tensor，数值通常落在 [-1, 1]。
    """
    return encoder_normalize(x, preset="hf_vit_0_5")


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    函数功能：
        按 torchvision/ImageNet mean/std 标准化 `[0, 1]` 伪图像。

    输入：
        x: [B, 3, H, W] 且范围应为 [0, 1] 的视觉 pixel tensor。

    输出：
        按 ImageNet mean/std 标准化后的 tensor。

    说明：
        该函数保留为旧代码兼容入口，不适合作为 HF ViT 的默认路径；
        HF ViT 请使用 `hf_vit_normalize()` 或
        `encoder_normalize(..., preset="hf_vit_0_5")`。
    """
    return encoder_normalize(x, preset="torchvision_imagenet")


def audit_metadata(x_norm: torch.Tensor, norm_metadata: Dict[str, torch.Tensor], periods: torch.Tensor, clip: float) -> Dict[str, torch.Tensor]:
    """
    函数功能：
        汇总 pilot 审计所需的 per-window metadata tensor。

    输出字段：
        norm_mean、norm_std、norm_range、top3_periods、clip_ratio。
    """
    return {
        "norm_mean": norm_metadata["norm_mean"],
        "norm_std": norm_metadata["norm_std"],
        "norm_range": norm_metadata["norm_range"],
        "top3_periods": periods,
        "clip_ratio": _clip_ratio(x_norm, clip=clip),
    }
