#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 前端 layout registry。

设计约束：
    - 每个 layout 只使用历史窗口 x，不读取 future y、专家 prediction 或 oracle label；
    - 主图像化路径使用 torch tensor 操作，输出 ViT-compatible `[B, 3, H, W]`；
    - 输出 tensor 保持 `[0, 1]` pixel range，encoder normalization 由后续 ViT 前处理负责；
    - deferred layout 只登记 metadata，不在 Round2b smoke 默认生成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from visual_router_experiments.common.pseudo_imageization import (
    EPS,
    _as_series_batch,
    _fft_power_view,
    _fold_fixed_period_batch,
    _line_raster,
    _period_fold_batch,
    imageize_3view,
    imageize_top3fold,
    make_default_period_candidates,
    normalize_window,
    parse_period_candidates,
    select_fft_periods,
    to_vision_pixels,
)


DEFAULT_ROUND2_LAYOUTS = (
    "current_rgb_3view",
    "spatial_panel_3view",
    "line_only",
    "line_difference_band",
    "fft_absolute_energy",
    "top3fold_period_layout",
)
DEFERRED_ROUND2_LAYOUTS = ("period_soft_mixture", "independent_view_encoder")


@dataclass(frozen=True)
class LayoutSpec:
    """类功能：登记单个 Round2 layout 的静态协议信息。"""

    name: str
    status: str
    description: str
    uses_periods: bool
    uses_fft: bool
    uses_difference: bool
    implementation_note: str


@dataclass
class LayoutImageizationResult:
    """类功能：返回 layout tensor 和可写入 summary 的 side metadata。"""

    images: torch.Tensor
    metadata: Dict[str, object]


def _resize_grayscale_view(view: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """
    函数功能：
        将 `[B, H, W]` 单通道视图 resize 到指定空间 panel。

    输入/输出：
        输入 view 为 GPU/CPU tensor；输出仍在同一设备，形状为 `[B, height, width]`。
    """
    return F.interpolate(
        view.unsqueeze(1),
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=True,
    ).squeeze(1)


def _profile_to_image(profile: torch.Tensor, *, image_size: int, normalize: bool) -> torch.Tensor:
    """
    函数功能：
        将一维频谱或差分 profile 变成 `[B, H, W]` 竖向复制图。

    说明：
        `normalize=True` 时只做窗口内 min-max，用于可视化稳定；absolute/log energy
        的原始尺度会另写 metadata，避免把“保留了绝对能量尺度”误写成模型输入事实。
    """
    if profile.shape[1] == 0:
        return torch.zeros((profile.shape[0], image_size, image_size), device=profile.device, dtype=profile.dtype)
    values = profile
    if normalize:
        min_value = values.min(dim=1, keepdim=True).values
        max_value = values.max(dim=1, keepdim=True).values
        values = (values - min_value) / (max_value - min_value).clamp_min(EPS)
    resized = F.interpolate(values.unsqueeze(1), size=image_size, mode="linear", align_corners=True).squeeze(1)
    return resized[:, :, None].expand(-1, image_size, image_size).contiguous().clamp(0.0, 1.0)


def _fft_energy_tensors(series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    函数功能：
        计算 absolute FFT energy、log energy 和累计能量曲线。

    约束：
        只对 RevIN-normalized history x 做 rFFT，去掉 DC 分量；不访问未来信息。
    """
    centered = series - series.mean(dim=1, keepdim=True)
    abs_energy = torch.abs(torch.fft.rfft(centered, dim=1)[:, 1:]) ** 2
    log_energy = torch.log1p(abs_energy)
    cumulative = torch.cumsum(abs_energy, dim=1)
    cumulative = cumulative / cumulative[:, -1:].clamp_min(EPS) if cumulative.shape[1] else cumulative
    return abs_energy, log_energy, cumulative


def _period_score_summary(series: torch.Tensor, periods: torch.Tensor) -> Dict[str, object]:
    """
    函数功能：
        为 top-k 周期记录 score/energy 统计和 bucket counts。

    说明：
        summary 级统计允许 detach 到 CPU；这不属于主图像化 tensor 路径。
    """
    centered = series - series.mean(dim=1, keepdim=True)
    power = torch.abs(torch.fft.rfft(centered, dim=1)[:, 1:]) ** 2
    if power.shape[1] == 0:
        scores = torch.zeros_like(periods, dtype=torch.float32)
    else:
        freq_count = int(power.shape[1])
        bins = torch.round(float(series.shape[1]) / periods.to(dtype=torch.float32)).to(dtype=torch.long)
        bins = bins.clamp(min=1, max=freq_count)
        scores = torch.gather(power, dim=1, index=bins - 1)
    periods_cpu = periods.detach().cpu()
    scores_cpu = scores.detach().cpu()
    bucket_counts: Dict[str, int] = {}
    for value in periods_cpu[:, 0].tolist():
        bucket_counts[str(int(value))] = bucket_counts.get(str(int(value)), 0) + 1
    return {
        "topk_periods_first_rows": periods_cpu[:8].tolist(),
        "period_score_mean_by_rank": scores_cpu.mean(dim=0).tolist(),
        "period_score_max_by_rank": scores_cpu.max(dim=0).values.tolist(),
        "top1_period_bucket_counts": bucket_counts,
    }


def _padding_summary(history_length: int, periods: torch.Tensor) -> Dict[str, object]:
    """函数功能：统计 period fold 是否发生 padding 以及 padding 比例。"""
    period_values = periods.to(dtype=torch.long)
    cycle_counts = torch.div(history_length + period_values - 1, period_values, rounding_mode="floor")
    padded_lengths = cycle_counts * period_values
    pad_counts = padded_lengths - int(history_length)
    return {
        "padding_mask_available": True,
        "padding_mask_used_as_vit_input": False,
        "pad_count_max": int(pad_counts.max().detach().cpu().item()) if pad_counts.numel() else 0,
        "pad_count_mean": float(pad_counts.to(dtype=torch.float32).mean().detach().cpu().item()) if pad_counts.numel() else 0.0,
        "padded_sample_ratio": float((pad_counts > 0).to(dtype=torch.float32).mean().detach().cpu().item()) if pad_counts.numel() else 0.0,
    }


def _shared_metadata(*, layout_name: str, image_size: int, series: torch.Tensor, norm_mode: str, clip: float) -> Dict[str, object]:
    """函数功能：构造所有 layout 共享的 metadata 字段。"""
    return {
        "layout_name": layout_name,
        "input_source": "history_x_only",
        "norm_mode": norm_mode,
        "history_length": int(series.shape[1]),
        "output_shape_contract": "[B,3,H,W]",
        "output_value_range_contract": "[0,1]",
        "resize_interpolation_mode": "linear_for_1d_profile; bilinear_for_2d_panel_or_fold",
        "resize_antialias": False,
        "effective_mapping": f"L={int(series.shape[1])} interpolated/resized to H=W={int(image_size)}",
        "pixel_mode": "vision",
        "clip": float(clip),
        "cpu_fallback": False,
    }


def _layout_current_rgb_3view(context: Mapping[str, object]) -> LayoutImageizationResult:
    """函数功能：Round1 baseline，三个语义视图直接 packed 到 RGB channel。"""
    series = context["series"]
    periods = context["periods"]
    image_size = int(context["image_size"])
    clip = float(context["clip"])
    images = imageize_3view(series, image_size=image_size, periods=periods, pixel_mode="vision", clip=clip)
    metadata = _shared_metadata(layout_name="current_rgb_3view", image_size=image_size, series=series, norm_mode=str(context["norm_mode"]), clip=clip)
    metadata.update(
        {
            "channel_design": "channel-packed line_raster/top1_period_fold/fft_power",
            "padding": _padding_summary(int(series.shape[1]), periods[:, :1]),
            "period_layout": "top1 fold in channel 1",
            "explicit_cpu_gpu_transfer_in_main_path": "period bucket scalar sync inherited from period fold implementation",
        }
    )
    return LayoutImageizationResult(images=images, metadata=metadata)


def _layout_spatial_panel_3view(context: Mapping[str, object]) -> LayoutImageizationResult:
    """函数功能：把 line/fold/FFT 三个视图放到水平空间 panel，而不是 RGB channel。"""
    series = context["series"]
    periods = context["periods"]
    image_size = int(context["image_size"])
    clip = float(context["clip"])
    line = _line_raster(series, image_size=image_size, pixel_mode="vision", clip=clip)
    fold = _period_fold_batch(series, periods, 0, image_size, "vision", clip)
    fft = _fft_power_view(series, image_size=image_size)
    widths = [image_size // 3, image_size // 3, image_size - 2 * (image_size // 3)]
    canvas = torch.zeros((series.shape[0], image_size, image_size), device=series.device, dtype=series.dtype)
    start = 0
    for view, width in zip([line, fold, fft], widths):
        canvas[:, :, start : start + width] = _resize_grayscale_view(view, height=image_size, width=width)
        start += width
    if image_size >= 6:
        # 用单像素 panel 边界帮助人工 debug，仍保持 [0,1]。
        canvas[:, :, widths[0] - 1 : widths[0] + 1] = 1.0
        canvas[:, :, widths[0] + widths[1] - 1 : widths[0] + widths[1] + 1] = 1.0
    images = canvas.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0.0, 1.0)
    metadata = _shared_metadata(layout_name="spatial_panel_3view", image_size=image_size, series=series, norm_mode=str(context["norm_mode"]), clip=clip)
    metadata.update(
        {
            "channel_design": "grayscale spatial panel replicated to 3 ViT channels",
            "panel_design": f"horizontal panels widths={widths}: line_raster/top1_period_fold/fft_power",
            "effective_panel_resolution": {"line": [image_size, widths[0]], "fold": [image_size, widths[1]], "fft": [image_size, widths[2]]},
            "padding": _padding_summary(int(series.shape[1]), periods[:, :1]),
            "explicit_cpu_gpu_transfer_in_main_path": "period bucket scalar sync inherited from period fold implementation",
        }
    )
    return LayoutImageizationResult(images=images, metadata=metadata)


def _layout_line_only(context: Mapping[str, object]) -> LayoutImageizationResult:
    """函数功能：只编码 RevIN-normalized history x 的 shape line。"""
    series = context["series"]
    image_size = int(context["image_size"])
    clip = float(context["clip"])
    line = _line_raster(series, image_size=image_size, pixel_mode="vision", clip=clip)
    images = line.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0.0, 1.0)
    metadata = _shared_metadata(layout_name="line_only", image_size=image_size, series=series, norm_mode=str(context["norm_mode"]), clip=clip)
    metadata.update(
        {
            "channel_design": "single line raster replicated to 3 channels",
            "uses_period_fold": False,
            "uses_fft_energy": False,
            "padding": {"padding_mask_available": False, "padding_mask_used_as_vit_input": False},
            "explicit_cpu_gpu_transfer_in_main_path": "none",
        }
    )
    return LayoutImageizationResult(images=images, metadata=metadata)


def _layout_line_difference_band(context: Mapping[str, object]) -> LayoutImageizationResult:
    """函数功能：同时编码 value line、signed first-difference 和 abs first-difference band。"""
    series = context["series"]
    image_size = int(context["image_size"])
    clip = float(context["clip"])
    first_diff = torch.cat([torch.zeros_like(series[:, :1]), series[:, 1:] - series[:, :-1]], dim=1)
    abs_diff = first_diff.abs()
    value_line = _line_raster(series, image_size=image_size, pixel_mode="vision", clip=clip)
    signed_diff_line = _line_raster(first_diff, image_size=image_size, pixel_mode="vision", clip=clip)
    abs_diff_band = _profile_to_image(abs_diff, image_size=image_size, normalize=True)
    images = torch.stack([value_line, signed_diff_line, abs_diff_band], dim=1).clamp(0.0, 1.0)
    metadata = _shared_metadata(layout_name="line_difference_band", image_size=image_size, series=series, norm_mode=str(context["norm_mode"]), clip=clip)
    metadata.update(
        {
            "channel_design": "channel0=value_line; channel1=signed_first_difference_line; channel2=abs_first_difference_minmax_band",
            "difference_definition": "first_diff=x[t]-x[t-1], first element padded with 0; abs band uses abs(first_diff)",
            "uses_rolling_volatility": False,
            "padding": {"padding_mask_available": False, "padding_mask_used_as_vit_input": False},
            "explicit_cpu_gpu_transfer_in_main_path": "none",
        }
    )
    return LayoutImageizationResult(images=images, metadata=metadata)


def _layout_fft_absolute_energy(context: Mapping[str, object]) -> LayoutImageizationResult:
    """函数功能：构造 absolute/log FFT energy layout，模型输入为归一化 profile 图。"""
    series = context["series"]
    image_size = int(context["image_size"])
    clip = float(context["clip"])
    abs_energy, log_energy, cumulative = _fft_energy_tensors(series)
    log_profile = _profile_to_image(log_energy, image_size=image_size, normalize=True)
    abs_profile = _profile_to_image(abs_energy, image_size=image_size, normalize=True)
    cumulative_profile = _profile_to_image(cumulative, image_size=image_size, normalize=False)
    images = torch.stack([log_profile, abs_profile, cumulative_profile], dim=1).clamp(0.0, 1.0)
    metadata = _shared_metadata(layout_name="fft_absolute_energy", image_size=image_size, series=series, norm_mode=str(context["norm_mode"]), clip=clip)
    metadata.update(
        {
            "channel_design": "channel0=log1p(abs_fft_energy) minmax profile; channel1=abs_fft_energy minmax profile; channel2=cumulative_abs_energy_ratio",
            "fft_energy_definition": "abs_energy=abs(rfft(centered_x)[1:])**2; log_energy=log1p(abs_energy)",
            "absolute_energy_preserved_in_model_input": False,
            "absolute_energy_preserved_in_metadata": True,
            "model_input_normalization": "per-window minmax for channel0/channel1; cumulative ratio for channel2",
            "frequency_axis_bins": int(abs_energy.shape[1]),
            "abs_energy_mean": float(abs_energy.mean().detach().cpu().item()) if abs_energy.numel() else 0.0,
            "log_energy_mean": float(log_energy.mean().detach().cpu().item()) if log_energy.numel() else 0.0,
            "padding": {"padding_mask_available": False, "padding_mask_used_as_vit_input": False},
            "explicit_cpu_gpu_transfer_in_main_path": "none; summary scalar detach only",
        }
    )
    return LayoutImageizationResult(images=images, metadata=metadata)


def _layout_top3fold_period_layout(context: Mapping[str, object]) -> LayoutImageizationResult:
    """函数功能：通过 registry 复用已有 top3fold channel-packed period fold 实现。"""
    series = context["series"]
    periods = context["periods"]
    image_size = int(context["image_size"])
    clip = float(context["clip"])
    images = imageize_top3fold(series, image_size=image_size, periods=periods, pixel_mode="vision", clip=clip)
    metadata = _shared_metadata(layout_name="top3fold_period_layout", image_size=image_size, series=series, norm_mode=str(context["norm_mode"]), clip=clip)
    metadata.update(
        {
            "channel_design": "channel-packed top1/top2/top3 period fold",
            "period_layout": "current Round2b adapter reuses imageize_top3fold; not panelized",
            "padding": _padding_summary(int(series.shape[1]), periods),
            "period_summary": _period_score_summary(series, periods),
            "explicit_cpu_gpu_transfer_in_main_path": "period bucket scalar sync inherited from period fold implementation",
        }
    )
    return LayoutImageizationResult(images=images, metadata=metadata)


LAYOUT_SPECS: Dict[str, LayoutSpec] = {
    "current_rgb_3view": LayoutSpec(
        name="current_rgb_3view",
        status="implemented",
        description="Round1 baseline: line raster, top1 period fold and FFT power packed into RGB channels.",
        uses_periods=True,
        uses_fft=True,
        uses_difference=False,
        implementation_note="adapter around existing imageize_3view",
    ),
    "spatial_panel_3view": LayoutSpec(
        name="spatial_panel_3view",
        status="implemented",
        description="Same three semantic views as current_rgb_3view, arranged as spatial panels.",
        uses_periods=True,
        uses_fft=True,
        uses_difference=False,
        implementation_note="tensorized horizontal panel canvas, replicated to RGB-compatible channels",
    ),
    "line_only": LayoutSpec(
        name="line_only",
        status="implemented",
        description="Minimal shape-only line raster from RevIN-normalized history x.",
        uses_periods=False,
        uses_fft=False,
        uses_difference=False,
        implementation_note="tensorized line raster replicated to three channels",
    ),
    "line_difference_band": LayoutSpec(
        name="line_difference_band",
        status="implemented",
        description="Value line plus first-difference/local-change bands.",
        uses_periods=False,
        uses_fft=False,
        uses_difference=True,
        implementation_note="first diff and abs first diff computed with torch tensor ops",
    ),
    "fft_absolute_energy": LayoutSpec(
        name="fft_absolute_energy",
        status="implemented",
        description="Frequency-domain absolute/log energy profile layout.",
        uses_periods=False,
        uses_fft=True,
        uses_difference=False,
        implementation_note="rFFT absolute/log energy computed with torch; model input uses profile visualization",
    ),
    "top3fold_period_layout": LayoutSpec(
        name="top3fold_period_layout",
        status="implemented",
        description="Top1/top2/top3 selected periods folded into channels.",
        uses_periods=True,
        uses_fft=True,
        uses_difference=False,
        implementation_note="registry adapter around existing imageize_top3fold",
    ),
    "period_soft_mixture": LayoutSpec(
        name="period_soft_mixture",
        status="deferred",
        description="Deferred Round2d continuity/soft-period experiment.",
        uses_periods=True,
        uses_fft=True,
        uses_difference=False,
        implementation_note="stub only; intended to mix multiple period folds by FFT strengths",
    ),
    "independent_view_encoder": LayoutSpec(
        name="independent_view_encoder",
        status="deferred",
        description="Deferred architecture-level candidate with separate encoder pass per view.",
        uses_periods=True,
        uses_fft=True,
        uses_difference=True,
        implementation_note="stub only; changes encoder architecture and compute budget",
    ),
}

_IMPLEMENTATIONS: Dict[str, Callable[[Mapping[str, object]], LayoutImageizationResult]] = {
    "current_rgb_3view": _layout_current_rgb_3view,
    "spatial_panel_3view": _layout_spatial_panel_3view,
    "line_only": _layout_line_only,
    "line_difference_band": _layout_line_difference_band,
    "fft_absolute_energy": _layout_fft_absolute_energy,
    "top3fold_period_layout": _layout_top3fold_period_layout,
}


def list_layout_specs() -> Dict[str, Dict[str, object]]:
    """函数功能：返回 registry 中所有 layout 的可 JSON 序列化静态说明。"""
    return {name: spec.__dict__.copy() for name, spec in LAYOUT_SPECS.items()}


def imageize_round2_layout(
    x: torch.Tensor,
    *,
    layout_name: str,
    image_size: int = 224,
    norm_mode: str = "revin_aux",
    clip: float = 5.0,
    period_candidates: Optional[Union[Sequence[int], torch.Tensor]] = None,
    period_selection: str = "fixed_candidates",
) -> LayoutImageizationResult:
    """
    函数功能：
        通过统一 registry 生成一个 Round2 layout 的 ViT-compatible pseudo image tensor。

    输入：
        x: 历史窗口 batch，支持 `[B, L, C]` / `[B, L]`。
        layout_name: registry 名称；deferred layout 会显式报错。

    输出：
        `LayoutImageizationResult(images=[B,3,H,W], metadata=dict)`。
    """
    if layout_name not in LAYOUT_SPECS:
        raise KeyError(f"未知 layout_name={layout_name}；可选值={sorted(LAYOUT_SPECS)}")
    if LAYOUT_SPECS[layout_name].status != "implemented":
        raise NotImplementedError(f"layout={layout_name} 当前为 deferred：{LAYOUT_SPECS[layout_name].implementation_note}")
    if image_size <= 1:
        raise ValueError("image_size 必须大于 1")

    x_norm, norm_metadata = normalize_window(x, norm_mode=norm_mode)
    series = _as_series_batch(x_norm).to(dtype=torch.float32)
    parsed_candidates = None
    if period_selection == "fixed_candidates":
        parsed_candidates = parse_period_candidates(period_candidates, history_length=int(series.shape[1]), device=series.device)
        if parsed_candidates is None:
            parsed_candidates = make_default_period_candidates(int(series.shape[1]), device=series.device)
    elif period_selection == "dynamic_fft_topk":
        parsed_candidates = None
    else:
        raise ValueError(f"未知 period_selection={period_selection}")
    periods = select_fft_periods(series, top_k=3, period_candidates=parsed_candidates)

    context = {
        "series": series,
        "norm_metadata": norm_metadata,
        "periods": periods,
        "period_candidates": parsed_candidates,
        "image_size": int(image_size),
        "clip": float(clip),
        "norm_mode": norm_mode,
        "period_selection": period_selection,
    }
    result = _IMPLEMENTATIONS[layout_name](context)
    if result.images.ndim != 4 or result.images.shape[1] != 3:
        raise ValueError(f"{layout_name} 输出必须为 [B,3,H,W]，实际={tuple(result.images.shape)}")
    result.metadata.update(
        {
            "registry_status": LAYOUT_SPECS[layout_name].status,
            "registry_implementation_note": LAYOUT_SPECS[layout_name].implementation_note,
            "period_selection": period_selection,
            "period_candidates_count": int(parsed_candidates.numel()) if torch.is_tensor(parsed_candidates) else 0,
            "selected_periods_shape": list(periods.shape),
        }
    )
    return result
