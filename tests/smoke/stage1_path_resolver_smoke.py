#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P4b path resolver 的最小 smoke。

输入：
    无命令行输入；除读取仓库根 marker 外，所有路径校验都发生在
    tempfile.TemporaryDirectory 下。

输出：
    标准输出打印中文检查日志；任一路径解析契约不一致时抛出异常。

关键约束：
    该脚本不访问 /data2，不访问 full-scale 输出目录，不创建正式输出目录，
    不写 JSON 文件。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.io import (  # noqa: E402
    find_repo_root,
    resolve_metadata_path,
    resolve_status_path,
    resolve_under_root,
)


def run_smoke() -> None:
    """函数功能：执行 P4b path resolver 的 repo-level 与 tempfile 检查。"""
    print("开始 Stage 1 P4b path resolver smoke")

    smoke_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(smoke_dir)
    expected_root = REPO_ROOT.resolve()
    if repo_root != expected_root:
        raise AssertionError(f"repo root 解析不一致：actual={repo_root} expected={expected_root}")
    workspace_doc = resolve_under_root(repo_root, "WORKSPACE_STRUCTURE.md", must_exist=True)
    if workspace_doc != expected_root / "WORKSPACE_STRUCTURE.md":
        raise AssertionError(f"WORKSPACE_STRUCTURE.md 定位不一致：{workspace_doc}")
    print(f"通过：从 tests/smoke 找到仓库根 {repo_root}")

    default_root = find_repo_root()
    if default_root != expected_root:
        raise AssertionError(f"默认起点 repo root 解析不一致：actual={default_root} expected={expected_root}")
    print("通过：默认起点可找到仓库根")

    with tempfile.TemporaryDirectory(prefix="stage1_path_resolver_smoke_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        existing_dir = tmp_root / "safe"
        existing_dir.mkdir()

        safe_path = resolve_under_root(tmp_root, "safe", "child.txt")
        if safe_path != (tmp_root / "safe" / "child.txt").resolve(strict=False):
            raise AssertionError(f"root 下正常路径解析不一致：{safe_path}")

        existing_path = resolve_under_root(tmp_root, "safe", must_exist=True)
        if existing_path != existing_dir.resolve(strict=False):
            raise AssertionError(f"must_exist=True 已存在路径解析不一致：{existing_path}")
        print("通过：tempfile root 下正常路径和 must_exist=True 已存在路径可解析")

        try:
            resolve_under_root(tmp_root, "..", "escape.txt")
        except ValueError:
            print("通过：`..` 逃逸 root 会明确失败")
        else:
            raise AssertionError("`..` 逃逸 root 未触发 ValueError")

        try:
            resolve_under_root(tmp_root, "missing.txt", must_exist=True)
        except FileNotFoundError:
            print("通过：must_exist=True 对不存在路径会明确失败")
        else:
            raise AssertionError("must_exist=True 对不存在路径未触发 FileNotFoundError")

        run_dir = tmp_root / "run_without_files"
        status_path = resolve_status_path(run_dir)
        metadata_path = resolve_metadata_path(run_dir)
        if status_path != run_dir / "status.json":
            raise AssertionError(f"status path 不一致：{status_path}")
        if metadata_path != run_dir / "metadata.json":
            raise AssertionError(f"metadata path 不一致：{metadata_path}")
        if run_dir.exists() or status_path.exists() or metadata_path.exists():
            raise AssertionError("status/metadata path helper 不应创建目录或文件")
        print("通过：status/metadata helper 只返回路径，不创建目录或文件")

    print("Stage 1 P4b path resolver smoke 通过")


if __name__ == "__main__":
    run_smoke()
