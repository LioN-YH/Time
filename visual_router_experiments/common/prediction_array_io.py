#!/usr/bin/env python3
"""
文件功能：
    Stage 1 prediction array IO 的旧路径兼容层。

设计约束：
    canonical implementation 已迁移到 `time_router.io.prediction_array_io`。
    旧实验脚本仍可从本模块导入同名常量和函数，但新代码应优先从
    `time_router.io` 或 `time_router.io.prediction_array_io` 导入，避免核心
    `time_router` 反向依赖实验目录。
"""

from __future__ import annotations

from time_router.io.prediction_array_io import (
    PACKED_NPY_STORAGE,
    PER_SAMPLE_NPY_STORAGE,
    load_prediction_array,
    load_prediction_arrays_grouped,
    resolve_cache_array_path,
)

__all__ = [
    "PACKED_NPY_STORAGE",
    "PER_SAMPLE_NPY_STORAGE",
    "load_prediction_array",
    "load_prediction_arrays_grouped",
    "resolve_cache_array_path",
]
