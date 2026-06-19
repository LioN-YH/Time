"""
文件功能：
    提供 Stage 1 P4b 使用的最小路径解析 helper。

关键约束：
    本文件只做 repo root、root 内安全拼接、status/metadata 路径计算；
    不读取训练配置、不创建目录、不访问 full-scale 输出目录、不写任何文件。
"""

from __future__ import annotations

import os
from pathlib import Path


_REPO_ROOT_MARKERS = (
    ".git",
    "WORKSPACE_STRUCTURE.md",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
)


def _normalize_start_path(start_path: str | os.PathLike[str] | None) -> Path:
    """
    函数功能：
        把调用方传入的起点规整为可向上查找的目录路径。

    输入：
        start_path: 可选起点；None 时使用当前 helper 文件所在位置。

    输出：
        起点目录的绝对 resolved Path。
    """
    if start_path is None:
        candidate = Path(__file__)
    else:
        candidate = Path(start_path)
    resolved = candidate.resolve(strict=False)
    if resolved.is_file():
        return resolved.parent
    return resolved


def find_repo_root(start_path: str | os.PathLike[str] | None = None) -> Path:
    """
    函数功能：
        从 start_path 或当前文件位置向上查找仓库根目录。

    输入：
        start_path: 可选起点文件或目录；None 时从本文件位置开始。

    输出：
        命中 `.git`、`WORKSPACE_STRUCTURE.md` 或 pyproject-like marker 的目录。

    关键约束：
        该函数只检查本地路径 marker，不读取训练配置，不访问任何输出目录。
    """
    current = _normalize_start_path(start_path)
    for candidate in (current, *current.parents):
        for marker in _REPO_ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(
        f"无法从 {current} 向上找到仓库根；未命中 marker={list(_REPO_ROOT_MARKERS)}"
    )


def resolve_under_root(
    root: str | os.PathLike[str],
    *parts: str | os.PathLike[str],
    must_exist: bool = False,
) -> Path:
    """
    函数功能：
        将 root 与 parts 拼接为 resolved path，并拒绝通过 `..` 或绝对路径逃逸 root。

    输入：
        root: 允许访问的根目录。
        parts: 需要拼接到 root 下的路径片段。
        must_exist: True 时要求最终路径已经存在。

    输出：
        位于 root 内部或等于 root 的 resolved Path。

    关键约束：
        该函数不自动创建目录；只做路径解析和边界校验。
    """
    root_path = Path(root).resolve(strict=False)
    resolved_path = root_path.joinpath(*parts).resolve(strict=False)
    try:
        resolved_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"解析路径逃逸 root：root={root_path} path={resolved_path}") from exc
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"路径不存在：{resolved_path}")
    return resolved_path


def resolve_status_path(run_dir: str | os.PathLike[str]) -> Path:
    """
    函数功能：
        返回 run_dir 下的 `status.json` 路径。

    输入：
        run_dir: 调用方显式传入的运行目录。

    输出：
        `Path(run_dir) / "status.json"`。

    关键约束：
        只返回路径，不创建目录、不写文件、不假设 run_dir 属于正式输出目录。
    """
    return Path(run_dir) / "status.json"


def resolve_metadata_path(run_dir: str | os.PathLike[str]) -> Path:
    """
    函数功能：
        返回 run_dir 下的 `metadata.json` 路径。

    输入：
        run_dir: 调用方显式传入的运行目录。

    输出：
        `Path(run_dir) / "metadata.json"`。

    关键约束：
        只返回路径，不创建目录、不写文件、不假设 run_dir 属于正式输出目录。
    """
    return Path(run_dir) / "metadata.json"
