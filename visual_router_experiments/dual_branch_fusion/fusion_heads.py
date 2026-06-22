#!/usr/bin/env python3
"""
文件功能：
    定义 65k PatchTST + fixed visual embedding 双分支实验的轻量融合头。

设计边界：
    这里只训练融合头，不包含 PatchTST 主干或视觉 encoder；PatchTST 与视觉特征
    均由外部 cache 提供并默认视为 frozen 表示。
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


FusionMode = Literal["feature_concat", "film", "residual_feature", "visual_residual", "pred_gate"]


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """函数功能：构造稳定的小型 MLP，供各融合变体复用。"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden_dim, output_dim),
    )


class DualBranchFusionHead(nn.Module):
    """
    类功能：
        在 frozen PatchTST 表示和 fixed visual embedding 上训练轻量预测头。

    输入：
        h_ts: PatchTST 时序表示，shape `[B, ts_dim]`。
        h_vis: 固定视觉表示，shape `[B, visual_dim]`。
        y_patchtst: PatchTST baseline 预测，shape `[B, ...]`。

    输出：
        y_fusion: 双分支预测，shape 与 y_patchtst/y_true 一致。
    """

    def __init__(
        self,
        *,
        mode: FusionMode,
        ts_dim: int,
        visual_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.output_dim = int(output_dim)
        self.ts_dim = int(ts_dim)
        self.visual_dim = int(visual_dim)

        if mode == "feature_concat":
            self.fusion = _make_mlp(ts_dim + visual_dim, hidden_dim, hidden_dim, dropout)
            self.prediction_head = nn.Linear(hidden_dim, output_dim)
        elif mode == "film":
            self.visual_to_gamma_beta = _make_mlp(visual_dim, hidden_dim, ts_dim * 2, dropout)
            self.prediction_head = _make_mlp(ts_dim, hidden_dim, output_dim, dropout)
        elif mode == "residual_feature":
            self.visual_residual = _make_mlp(visual_dim, hidden_dim, ts_dim, dropout)
            self.prediction_head = _make_mlp(ts_dim, hidden_dim, output_dim, dropout)
        elif mode == "visual_residual":
            self.visual_delta = _make_mlp(visual_dim, hidden_dim, output_dim, dropout)
        elif mode == "pred_gate":
            self.visual_prediction = _make_mlp(visual_dim, hidden_dim, output_dim, dropout)
            self.gate = _make_mlp(ts_dim + visual_dim, hidden_dim, output_dim, dropout)
        else:
            raise ValueError(f"不支持的 fusion mode：{mode}")

    def forward(self, h_ts: torch.Tensor, h_vis: torch.Tensor, y_patchtst: torch.Tensor) -> torch.Tensor:
        """函数功能：根据 mode 执行对应的轻量融合预测。"""
        batch_size = y_patchtst.shape[0]
        y_base_flat = y_patchtst.reshape(batch_size, self.output_dim)

        if self.mode == "feature_concat":
            hidden = self.fusion(torch.cat([h_ts, h_vis], dim=-1))
            return self.prediction_head(hidden).reshape_as(y_patchtst)

        if self.mode == "film":
            gamma_beta = self.visual_to_gamma_beta(h_vis)
            gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
            # 让 gamma 从接近 1 的区域开始，避免初始阶段过度破坏时序表示。
            h_fused = (1.0 + torch.tanh(gamma)) * h_ts + beta
            return self.prediction_head(h_fused).reshape_as(y_patchtst)

        if self.mode == "residual_feature":
            h_fused = h_ts + self.visual_residual(h_vis)
            return self.prediction_head(h_fused).reshape_as(y_patchtst)

        if self.mode == "visual_residual":
            delta_y = self.visual_delta(h_vis).reshape_as(y_patchtst)
            return y_patchtst + delta_y

        if self.mode == "pred_gate":
            y_vis = self.visual_prediction(h_vis)
            alpha = torch.sigmoid(self.gate(torch.cat([h_ts, h_vis], dim=-1)))
            y_fused = alpha * y_base_flat + (1.0 - alpha) * y_vis
            return y_fused.reshape_as(y_patchtst)

        raise RuntimeError(f"不可达 fusion mode：{self.mode}")


def build_fusion_head(
    *,
    mode: str,
    ts_dim: int,
    visual_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout: float,
) -> DualBranchFusionHead:
    """函数功能：对外提供字符串 mode 到融合头实例的统一构造入口。"""
    valid_modes = {"feature_concat", "film", "residual_feature", "visual_residual", "pred_gate"}
    if mode not in valid_modes:
        raise ValueError(f"fusion_mode 必须属于 {sorted(valid_modes)}，actual={mode}")
    return DualBranchFusionHead(
        mode=mode,  # type: ignore[arg-type]
        ts_dim=int(ts_dim),
        visual_dim=int(visual_dim),
        output_dim=int(output_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
