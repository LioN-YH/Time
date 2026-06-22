#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 panel-aware gating / residual architecture probe。

设计边界：
    - `global_mean_patch` 仍是主视觉表示；
    - `panel_mean_stack` 只生成轻量 gate / residual，不直接作为高维 router 输入；
    - 输出默认保持 768 维，供后续 FiLM router 复用；
    - 本模块只定义 architecture，不读取数据、不训练、不保存 pseudo image tensor。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F


PANEL_GATING_SCHEMA_VERSION = "visual_router_v2_round2_panel_gating_v1"
PANEL_GATING_VARIANTS = ("film_panel_gated_mean_aux", "film_panel_lowrank_aux", "film_panel_attention_aux")


@dataclass(frozen=True)
class PanelGatingConfig:
    """类功能：记录 panel gating probe 的轻量结构超参。"""

    visual_dim: int = 768
    panel_count: int = 3
    gate_hidden_dim: int = 64
    lowrank_dim: int = 256
    init_alpha: float = 0.1


def validate_panel_inputs(global_mean_patch: torch.Tensor, panel_mean_stack: torch.Tensor) -> None:
    """
    函数功能：
        校验 global mean 与 panel stack 的基础 shape/finite 约束。

    输入：
        global_mean_patch: `[B,D]` 的主视觉表示。
        panel_mean_stack: `[B,3,D]` 的 line/fold/FFT panel mean。

    输出：
        无返回；异常表示输入不符合 probe contract。
    """
    if global_mean_patch.ndim != 2:
        raise ValueError(f"global_mean_patch 必须为 [B,D]，实际 shape={tuple(global_mean_patch.shape)}")
    if panel_mean_stack.ndim != 3:
        raise ValueError(f"panel_mean_stack 必须为 [B,3,D]，实际 shape={tuple(panel_mean_stack.shape)}")
    if panel_mean_stack.shape[0] != global_mean_patch.shape[0] or panel_mean_stack.shape[2] != global_mean_patch.shape[1]:
        raise ValueError(
            "global_mean_patch 与 panel_mean_stack 的 batch/dim 不一致："
            f"global={tuple(global_mean_patch.shape)} panel={tuple(panel_mean_stack.shape)}"
        )
    if panel_mean_stack.shape[1] != 3:
        raise ValueError(f"panel_mean_stack 当前只支持 3 个 panel，实际 panel_count={panel_mean_stack.shape[1]}")
    if not torch.isfinite(global_mean_patch).all() or not torch.isfinite(panel_mean_stack).all():
        raise ValueError("global_mean_patch 或 panel_mean_stack 中存在 NaN/Inf")


def panel_concat_to_stack(panel_mean_concat: torch.Tensor, *, panel_count: int = 3) -> torch.Tensor:
    """
    函数功能：
        将既有 35k cache 中的 panel concat `[B,2304]` 复原为 `[B,3,768]`。
    """
    if panel_mean_concat.ndim != 2:
        raise ValueError(f"panel_mean_concat 必须为 [B,3D]，实际 shape={tuple(panel_mean_concat.shape)}")
    if panel_mean_concat.shape[1] % int(panel_count) != 0:
        raise ValueError(f"panel_mean_concat dim={panel_mean_concat.shape[1]} 不能被 panel_count={panel_count} 整除")
    visual_dim = int(panel_mean_concat.shape[1]) // int(panel_count)
    return panel_mean_concat.reshape(panel_mean_concat.shape[0], int(panel_count), visual_dim)


def residual_norm_report(
    *,
    global_mean_patch: torch.Tensor,
    visual: torch.Tensor,
    panel_residual: torch.Tensor,
) -> Dict[str, float]:
    """
    函数功能：
        计算 residual/gated 表示的尺度诊断，检查是否相对 global baseline 爆炸。
    """
    global_norm = torch.linalg.vector_norm(global_mean_patch, ord=2, dim=1).clamp_min(1.0e-12)
    visual_norm = torch.linalg.vector_norm(visual, ord=2, dim=1)
    residual_norm = torch.linalg.vector_norm(panel_residual, ord=2, dim=1)
    delta_norm = torch.linalg.vector_norm(visual - global_mean_patch, ord=2, dim=1)
    return {
        "global_norm_mean": float(global_norm.mean().detach().cpu().item()),
        "visual_norm_mean": float(visual_norm.mean().detach().cpu().item()),
        "panel_residual_norm_mean": float(residual_norm.mean().detach().cpu().item()),
        "visual_delta_norm_mean": float(delta_norm.mean().detach().cpu().item()),
        "panel_residual_to_global_norm_ratio_mean": float((residual_norm / global_norm).mean().detach().cpu().item()),
        "visual_delta_to_global_norm_ratio_mean": float((delta_norm / global_norm).mean().detach().cpu().item()),
        "visual_delta_to_global_norm_ratio_max": float((delta_norm / global_norm).max().detach().cpu().item()),
    }


class PanelGatedResidual(nn.Module):
    """
    类功能：
        轻量 gated residual：panel stack 只生成 3 个 gate，最终输出保持 768 维。

    输入：
        global_mean_patch `[B,D]`，panel_mean_stack `[B,3,D]`。

    输出：
        dict，包含 `visual`、`gates`、`panel_residual`、`alpha`。
    """

    def __init__(self, config: PanelGatingConfig | None = None) -> None:
        super().__init__()
        self.config = config or PanelGatingConfig()
        flat_dim = int(self.config.panel_count * self.config.visual_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, int(self.config.gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(self.config.gate_hidden_dim), int(self.config.panel_count)),
        )
        self.alpha = nn.Parameter(torch.tensor(float(self.config.init_alpha), dtype=torch.float32))
        # 初始 gate 置于中性区间，避免未训练 smoke 中凭随机 gate 放大 panel residual。
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, global_mean_patch: torch.Tensor, panel_mean_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        validate_panel_inputs(global_mean_patch, panel_mean_stack)
        panel_delta = panel_mean_stack - global_mean_patch.unsqueeze(1)
        gate_logits = self.gate_mlp(panel_mean_stack.reshape(panel_mean_stack.shape[0], -1))
        gates = torch.sigmoid(gate_logits)
        # 用 gate sum 归一化，避免三个 panel residual 直接累加导致尺度超过 global mean。
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        panel_residual = torch.sum(weights.unsqueeze(-1) * panel_delta, dim=1)
        alpha = torch.clamp(self.alpha.to(device=global_mean_patch.device, dtype=global_mean_patch.dtype), min=-1.0, max=1.0)
        visual = global_mean_patch + alpha * panel_residual
        return {"visual": visual, "gates": gates, "panel_residual": panel_residual, "alpha": alpha}


class PanelLowRankAdapter(nn.Module):
    """
    类功能：
        低秩 panel adapter：2304 维 panel concat 经 256 维瓶颈生成 768 维 residual。
    """

    def __init__(self, config: PanelGatingConfig | None = None) -> None:
        super().__init__()
        self.config = config or PanelGatingConfig()
        flat_dim = int(self.config.panel_count * self.config.visual_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, int(self.config.lowrank_dim)),
            nn.GELU(),
            nn.Linear(int(self.config.lowrank_dim), int(self.config.visual_dim)),
        )
        self.alpha = nn.Parameter(torch.tensor(float(self.config.init_alpha), dtype=torch.float32))
        # 低秩 residual 初始为小扰动，重点验证结构与尺度，不让随机未训练 adapter 主导表示。
        nn.init.normal_(self.adapter[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, global_mean_patch: torch.Tensor, panel_mean_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        validate_panel_inputs(global_mean_patch, panel_mean_stack)
        panel_concat = panel_mean_stack.reshape(panel_mean_stack.shape[0], -1)
        panel_residual = self.adapter(panel_concat)
        alpha = torch.clamp(self.alpha.to(device=global_mean_patch.device, dtype=global_mean_patch.dtype), min=-1.0, max=1.0)
        visual = global_mean_patch + alpha * panel_residual
        return {"visual": visual, "panel_residual": panel_residual, "alpha": alpha}


class PanelAttentionResidual(nn.Module):
    """
    类功能：
        三 panel token 的极小 attention probe，用 softmax 权重汇总 panel residual。
    """

    def __init__(self, config: PanelGatingConfig | None = None) -> None:
        super().__init__()
        self.config = config or PanelGatingConfig()
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(int(self.config.visual_dim)),
            nn.Linear(int(self.config.visual_dim), int(self.config.gate_hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(self.config.gate_hidden_dim), 1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(self.config.init_alpha), dtype=torch.float32))
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward(self, global_mean_patch: torch.Tensor, panel_mean_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        validate_panel_inputs(global_mean_patch, panel_mean_stack)
        panel_delta = panel_mean_stack - global_mean_patch.unsqueeze(1)
        scores = self.score_mlp(panel_delta).squeeze(-1)
        attention_weights = F.softmax(scores, dim=1)
        panel_residual = torch.sum(attention_weights.unsqueeze(-1) * panel_delta, dim=1)
        alpha = torch.clamp(self.alpha.to(device=global_mean_patch.device, dtype=global_mean_patch.dtype), min=-1.0, max=1.0)
        visual = global_mean_patch + alpha * panel_residual
        return {
            "visual": visual,
            "attention_weights": attention_weights,
            "panel_residual": panel_residual,
            "alpha": alpha,
        }


def build_panel_gating_model(variant: str, config: PanelGatingConfig | None = None) -> nn.Module:
    """函数功能：根据候选名称构建对应 panel-aware residual/gating 模块。"""
    if variant == "film_panel_gated_mean_aux":
        return PanelGatedResidual(config)
    if variant == "film_panel_lowrank_aux":
        return PanelLowRankAdapter(config)
    if variant == "film_panel_attention_aux":
        return PanelAttentionResidual(config)
    raise ValueError(f"未知 panel gating variant={variant}")


def summarize_probe_output(
    *,
    variant: str,
    output: Mapping[str, torch.Tensor],
    global_mean_patch: torch.Tensor,
) -> Dict[str, object]:
    """函数功能：把单个候选的 smoke 输出转成可写入 JSON/Markdown 的标量摘要。"""
    visual = output["visual"]
    panel_residual = output["panel_residual"]
    summary: Dict[str, object] = {
        "variant": str(variant),
        "visual_shape": [int(value) for value in visual.shape],
        "visual_finite": bool(torch.isfinite(visual).all().detach().cpu().item()),
        "alpha": float(output["alpha"].detach().cpu().item()),
    }
    summary.update(residual_norm_report(global_mean_patch=global_mean_patch, visual=visual, panel_residual=panel_residual))
    if "gates" in output:
        gates = output["gates"]
        summary.update(
            {
                "gate_shape": [int(value) for value in gates.shape],
                "gate_min": float(gates.min().detach().cpu().item()),
                "gate_max": float(gates.max().detach().cpu().item()),
                "gate_mean": float(gates.mean().detach().cpu().item()),
                "gate_std": float(gates.std(unbiased=False).detach().cpu().item()),
                "gate_range_valid": bool(((gates >= 0.0) & (gates <= 1.0)).all().detach().cpu().item()),
            }
        )
    if "attention_weights" in output:
        weights = output["attention_weights"]
        summary.update(
            {
                "attention_shape": [int(value) for value in weights.shape],
                "attention_min": float(weights.min().detach().cpu().item()),
                "attention_max": float(weights.max().detach().cpu().item()),
                "attention_mean": float(weights.mean().detach().cpu().item()),
                "attention_row_sum_max_abs_error": float(torch.max(torch.abs(weights.sum(dim=1) - 1.0)).detach().cpu().item()),
            }
        )
    return summary
