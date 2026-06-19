"""
文件功能：
    提供 Stage 1 P4a 使用的最小 JSON 原子写入工具和 run status payload
    构造工具。

关键约束：
    本文件只依赖 Python 标准库，只消费调用方显式传入的 path 和 payload；
    不读取训练状态、不解析工作区路径、不绑定 Visual Router / TimeFuse fusor。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def atomic_write_json(
    data: Any,
    path: str | os.PathLike[str],
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    """
    函数功能：
        将 JSON payload 先写入目标同目录临时文件，经 flush + fsync 后用
        os.replace 原子替换目标路径。

    输入：
        data: 可被 json.dump 序列化的数据。
        path: 目标 JSON 文件路径；父目录不存在时自动创建。
        indent / ensure_ascii / sort_keys: 透传给 json.dump 的格式化参数。

    输出：
        无返回值；成功后目标文件包含完整 UTF-8 JSON。

    关键约束：
        临时文件必须位于目标同目录，避免跨文件系统 replace 破坏原子性。
    """
    target_path = Path(path)
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_path: Optional[Path] = None
    try:
        # NamedTemporaryFile 放在目标同目录，确保 os.replace 在同一文件系统内完成。
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                data,
                handle,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
        temp_path = None
    finally:
        # 序列化或 fsync 失败时只清理本次临时文件，不触碰已有目标文件。
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def build_status_payload(
    *,
    status: str,
    phase: str | None = None,
    message: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    函数功能：
        构造最小 status.json payload，固定包含 status，并按需合并 phase、
        message 和调用方显式提供的 extra 字段。

    输入：
        status: 状态字符串，必须由调用方显式传入。
        phase/message: 可选阶段和说明文本。
        extra: 可选附加字段；必须是 dict-like mapping 或 None。

    输出：
        可直接写入 JSON 的 dict。
    """
    if not isinstance(status, str) or not status:
        raise ValueError("status 必须是非空字符串")
    if extra is not None and not isinstance(extra, dict):
        raise TypeError("extra 必须是 dict 或 None")

    payload: Dict[str, Any] = {"status": status}
    if phase is not None:
        payload["phase"] = phase
    if message is not None:
        payload["message"] = message
    if extra:
        payload.update(extra)
    return payload


def write_status_json(
    path: str | os.PathLike[str],
    *,
    status: str,
    phase: str | None = None,
    message: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """
    函数功能：
        构造最小 status payload 并通过 atomic_write_json 安全写入指定路径。

    输入：
        path: 调用方显式传入的 status.json 路径。
        status/phase/message/extra: 传给 build_status_payload 的状态字段。

    输出：
        无返回值；成功后 path 指向新的 status JSON。
    """
    payload = build_status_payload(status=status, phase=phase, message=message, extra=extra)
    atomic_write_json(payload, path)
