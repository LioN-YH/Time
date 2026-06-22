#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 探索分支的 PatchTST + fixed visual embedding 双分支预测实验包。

边界：
    本包只消费已有 PatchTST 预测/特征 cache 和固定视觉 embedding cache，不生成图像、
    不运行 ViT、不修改 Stage 1 canonical 入口。
"""

from __future__ import annotations

__all__ = []
