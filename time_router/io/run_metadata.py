"""
文件功能：
    提供 Stage 1 P4c 使用的最小 run metadata payload builder。

关键约束：
    本文件只构造 metadata-like JSON payload，并可写入调用方显式传入的
    path；不自动调用 git、不读取命令行或训练配置、不解析 full-scale 输出目录。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from time_router.io.json_utils import atomic_write_json


def _require_dict_or_none(name: str, value: Mapping[str, Any] | None) -> Dict[str, Any]:
    """
    函数功能：
        校验 metadata builder 的 dict-like 输入边界。

    输入：
        name: 字段名，用于错误信息。
        value: 待校验字段值。

    输出：
        value 为 None 时返回空 dict；value 为 dict 时返回浅拷贝。
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 dict 或 None")
    return dict(value)


def _to_json_safe(value: Any) -> Any:
    """
    函数功能：
        将 Path 等常见路径对象转换为 JSON-safe 值。

    关键约束：
        该函数只做内存中的轻量类型转换，不检查路径是否存在，不访问外部目录。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return value


def build_run_metadata(
    *,
    stage: str,
    entrypoint: Any = None,
    command: Any = None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    git_ref: Any = None,
    notes: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    函数功能：
        构造最小 run metadata payload，用于记录阶段、输入输出、命令和补充说明。

    输入：
        stage: 非空阶段名。
        entrypoint / command / git_ref / notes: 调用方显式传入的可选元信息。
        inputs / outputs: 调用方显式传入的输入输出路径或标识 dict。
        extra: 调用方显式传入的附加 dict，保留在 payload["extra"]。

    输出：
        至少包含 `stage`、`created_at_utc`、`inputs`、`outputs` 的 dict。

    关键约束：
        不自动调用 git、不自动读取当前命令行、不读取训练配置、不解析输出目录。
    """
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage 必须是非空字符串")

    payload: Dict[str, Any] = {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": _to_json_safe(_require_dict_or_none("inputs", inputs)),
        "outputs": _to_json_safe(_require_dict_or_none("outputs", outputs)),
    }
    if entrypoint is not None:
        payload["entrypoint"] = _to_json_safe(entrypoint)
    if command is not None:
        payload["command"] = _to_json_safe(command)
    if git_ref is not None:
        payload["git_ref"] = _to_json_safe(git_ref)
    if notes is not None:
        payload["notes"] = _to_json_safe(notes)

    extra_dict = _require_dict_or_none("extra", extra)
    if extra_dict:
        payload["extra"] = _to_json_safe(extra_dict)
    return payload


def write_run_metadata(path: str | os.PathLike[str], **kwargs: Any) -> None:
    """
    函数功能：
        构造 run metadata payload 并通过 atomic_write_json 写入显式 path。

    输入：
        path: 调用方显式传入的 metadata JSON 路径。
        **kwargs: 透传给 build_run_metadata 的字段。

    输出：
        无返回值；成功后 path 指向新的 metadata JSON。

    关键约束：
        本函数不自行选择输出目录；父目录创建行为仅来自 atomic_write_json。
    """
    payload = build_run_metadata(**kwargs)
    atomic_write_json(payload, path)
