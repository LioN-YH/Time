#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 `spatial_panel_3view` 的 ViT patch panel-wise pooling
    工具。

核心约束：
    - 只解释既有 spatial panel layout 在 224x224 图像和 ViT 16x16 patch grid
      上的 region mapping；
    - pooling 输入必须是 Hugging Face ViT `last_hidden_state`，不保存伪图像 tensor；
    - 默认忽略跨 panel 边界的 patch 列，避免白色 debug 边界和相邻 view 混入
      panel mean。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F


PANEL_POOLING_SCHEMA_VERSION = "visual_router_v2_round2_panel_pooling_v1"
DEFAULT_PANEL_NAMES = ("line_panel", "fold_panel", "fft_panel")


@dataclass(frozen=True)
class PanelPatchRegion:
    """类功能：描述一个 spatial panel 在 ViT patch grid 中使用的严格内部 patch。"""

    name: str
    semantic_view: str
    pixel_x_start: int
    pixel_x_end: int
    strict_patch_cols: tuple[int, ...]
    patch_indices: tuple[int, ...]

    def to_dict(self) -> Dict[str, object]:
        """函数功能：转成可写入 JSON/metadata 的 dict。"""
        return {
            "name": self.name,
            "semantic_view": self.semantic_view,
            "pixel_x_range_inclusive_exclusive": [int(self.pixel_x_start), int(self.pixel_x_end)],
            "strict_patch_cols": [int(value) for value in self.strict_patch_cols],
            "patch_indices": [int(value) for value in self.patch_indices],
            "patch_count": int(len(self.patch_indices)),
        }


def spatial_panel_widths(image_size: int) -> List[int]:
    """函数功能：复现 `spatial_panel_3view` 的三段水平 panel 宽度。"""
    width = int(image_size) // 3
    return [width, width, int(image_size) - 2 * width]


def _patch_index(row: int, col: int, grid_width: int) -> int:
    """函数功能：把 2D patch grid 坐标转成 ViT patch token 的 0-based patch index。"""
    return int(row) * int(grid_width) + int(col)


def build_spatial_panel_region_mapping(
    *,
    image_size: int = 224,
    patch_size: int = 16,
    ignore_boundary_patches: bool = True,
) -> Dict[str, object]:
    """
    函数功能：
        计算 `spatial_panel_3view` 从 image-space panel 到 ViT patch grid 的映射。

    输入：
        image_size: ViT 输入图像边长，Round2 默认 224。
        patch_size: ViT patch 边长，`google/vit-base-patch16-224` 默认 16。
        ignore_boundary_patches: 为 True 时只保留完全落入 panel 内部的 patch 列；
            横跨两个 panel 或白色 debug 边界的列进入 ignored/mixed。

    输出：
        可 JSON 序列化 mapping；`regions[*].patch_indices` 对应
        `last_hidden_state[:, 1:, :]` 的 0-based patch token index。
    """
    if int(image_size) % int(patch_size) != 0:
        raise ValueError(f"image_size={image_size} 必须能被 patch_size={patch_size} 整除")
    grid = int(image_size) // int(patch_size)
    widths = spatial_panel_widths(int(image_size))
    panel_ranges = []
    start = 0
    for width in widths:
        stop = start + int(width)
        panel_ranges.append((start, stop))
        start = stop

    # spatial_panel_3view 在两个 panel 交界处写入 2 像素白线；这些像素会进入相邻 patch。
    debug_boundary_pixel_ranges = []
    if int(image_size) >= 6:
        b1 = widths[0]
        b2 = widths[0] + widths[1]
        debug_boundary_pixel_ranges = [[b1 - 1, b1 + 1], [b2 - 1, b2 + 1]]

    regions: List[PanelPatchRegion] = []
    ignored_patch_cols: List[int] = []
    col_assignment: Dict[int, str] = {}
    for col in range(grid):
        patch_start = col * int(patch_size)
        patch_end = patch_start + int(patch_size)
        assigned = None
        for panel_idx, (panel_start, panel_end) in enumerate(panel_ranges):
            if ignore_boundary_patches:
                inside = patch_start >= panel_start and patch_end <= panel_end
            else:
                center = patch_start + int(patch_size) / 2.0
                inside = panel_start <= center < panel_end
            if inside:
                assigned = DEFAULT_PANEL_NAMES[panel_idx]
                break
        if assigned is None:
            ignored_patch_cols.append(col)
            col_assignment[col] = "ignored_boundary_or_mixed"
        else:
            col_assignment[col] = assigned

    for panel_idx, panel_name in enumerate(DEFAULT_PANEL_NAMES):
        cols = tuple(col for col, value in col_assignment.items() if value == panel_name)
        indices = tuple(_patch_index(row, col, grid) for row in range(grid) for col in cols)
        start_x, end_x = panel_ranges[panel_idx]
        semantic_view = ("line_raster", "top1_period_fold", "fft_power")[panel_idx]
        regions.append(
            PanelPatchRegion(
                name=panel_name,
                semantic_view=semantic_view,
                pixel_x_start=start_x,
                pixel_x_end=end_x,
                strict_patch_cols=cols,
                patch_indices=indices,
            )
        )

    used_indices = [idx for region in regions for idx in region.patch_indices]
    return {
        "schema_version": PANEL_POOLING_SCHEMA_VERSION,
        "layout_name": "spatial_panel_3view",
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "patch_grid": [grid, grid],
        "patch_count_excluding_cls": int(grid * grid),
        "panel_widths": widths,
        "panel_pixel_ranges_inclusive_exclusive": {
            DEFAULT_PANEL_NAMES[idx]: [int(start), int(end)] for idx, (start, end) in enumerate(panel_ranges)
        },
        "debug_boundary_pixel_ranges_inclusive_exclusive": debug_boundary_pixel_ranges,
        "ignore_boundary_patches": bool(ignore_boundary_patches),
        "ignored_patch_cols": [int(value) for value in ignored_patch_cols],
        "ignored_patch_count": int(len(ignored_patch_cols) * grid),
        "regions": [region.to_dict() for region in regions],
        "used_patch_indices": [int(value) for value in used_indices],
        "used_patch_count": int(len(used_indices)),
        "all_patch_indices": [int(value) for value in range(grid * grid)],
        "notes": [
            "ViT patch token index is 0-based after removing CLS token.",
            "Strict mapping drops patch columns that cross panel boundaries; this preserves view separation but panel means do not cover every patch.",
            "global_mean_patch remains available as the fallback and exact mean over all patch tokens.",
        ],
    }


def _panel_index_tensor(mapping: Mapping[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    """函数功能：把 mapping 中的 patch index 转成当前 device 上的 LongTensor。"""
    result: Dict[str, torch.Tensor] = {}
    for region in mapping["regions"]:  # type: ignore[index]
        region_map = dict(region)
        result[str(region_map["name"])] = torch.as_tensor(region_map["patch_indices"], dtype=torch.long, device=device)
    return result


def pool_spatial_panel_hidden_states(
    last_hidden_state: torch.Tensor,
    *,
    mapping: Mapping[str, object] | None = None,
    include_panel_variance: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    函数功能：
        从 ViT `last_hidden_state` 生成 global/panel-wise pooled embedding。

    输入：
        last_hidden_state: `[B, 1 + N, D]`，第 0 个 token 为 CLS。
        mapping: `build_spatial_panel_region_mapping` 输出；为空时使用 224/16 默认。
        include_panel_variance: 是否输出 panel 间 variance/disagreement 诊断特征。

    输出：
        包含 `global_mean_patch`、`panel_mean_concat`、`global_plus_panel_mean`、
        `panel_mean_stack` 以及可选 `panel_variance`。
    """
    if last_hidden_state.ndim != 3:
        raise ValueError(f"last_hidden_state 必须为 [B,1+N,D]，实际 shape={tuple(last_hidden_state.shape)}")
    patch_tokens = last_hidden_state[:, 1:, :]
    if mapping is None:
        mapping = build_spatial_panel_region_mapping()
    expected_patch_count = int(mapping["patch_count_excluding_cls"])
    if patch_tokens.shape[1] != expected_patch_count:
        raise ValueError(f"ViT patch token 数不匹配：expected={expected_patch_count} actual={patch_tokens.shape[1]}")
    panel_indices = _panel_index_tensor(mapping, patch_tokens.device)
    panel_means: List[torch.Tensor] = []
    for name in DEFAULT_PANEL_NAMES:
        idx = panel_indices[name]
        if idx.numel() == 0:
            raise ValueError(f"panel={name} 没有可用 patch index")
        panel_means.append(patch_tokens.index_select(dim=1, index=idx).mean(dim=1))
    panel_stack = torch.stack(panel_means, dim=1)
    result = {
        "global_mean_patch": patch_tokens.mean(dim=1),
        "panel_mean_stack": panel_stack,
        "panel_mean_concat": torch.cat(panel_means, dim=1),
        "global_plus_panel_mean": torch.cat([patch_tokens.mean(dim=1), *panel_means], dim=1),
    }
    if include_panel_variance:
        result["panel_variance"] = panel_stack.var(dim=1, unbiased=False)
    return result


def panel_difference_summary(panel_mean_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    函数功能：
        计算 panel means 之间的 cosine/L2 差异，用于 smoke 证明 panel 表征没有完全塌缩。
    """
    if panel_mean_stack.ndim != 3 or panel_mean_stack.shape[1] != 3:
        raise ValueError(f"panel_mean_stack 必须为 [B,3,D]，实际 shape={tuple(panel_mean_stack.shape)}")
    pairs = [(0, 1, "line_vs_fold"), (1, 2, "fold_vs_fft"), (0, 2, "line_vs_fft")]
    result: Dict[str, torch.Tensor] = {}
    for left, right, name in pairs:
        a = panel_mean_stack[:, left, :]
        b = panel_mean_stack[:, right, :]
        result[f"{name}_cosine"] = F.cosine_similarity(a, b, dim=1)
        result[f"{name}_l2"] = torch.linalg.vector_norm(a - b, ord=2, dim=1)
    return result

