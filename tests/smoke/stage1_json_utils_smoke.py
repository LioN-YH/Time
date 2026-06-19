#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P4a JSON 原子写入与 status writer 的最小 smoke。

输入：
    无命令行输入；所有写入都发生在 tempfile.TemporaryDirectory 下。

输出：
    标准输出打印中文检查日志；任一原子写入或 payload 契约不一致时抛出异常。

关键约束：
    该脚本不读取训练状态、不写正式输出目录、不接入 Visual Router / TimeFuse fusor。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.io import atomic_write_json, build_status_payload, write_status_json  # noqa: E402


def read_json(path: Path) -> dict:
    """函数功能：按 UTF-8 读取 JSON 文件，供 smoke 断言使用。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_smoke() -> None:
    """函数功能：执行 P4a JSON writer 的临时目录 smoke 检查。"""
    print("开始 Stage 1 P4a JSON utils smoke")
    with tempfile.TemporaryDirectory(prefix="stage1_json_utils_smoke_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        status_path = tmp_root / "runs" / "status.json"

        write_status_json(
            status_path,
            status="running",
            phase="smoke",
            message="中文状态检查",
            extra={"step": 1},
        )
        if not status_path.exists():
            raise AssertionError(f"status.json 未写出：{status_path}")

        payload = read_json(status_path)
        expected_payload = {
            "status": "running",
            "phase": "smoke",
            "message": "中文状态检查",
            "step": 1,
        }
        if payload != expected_payload:
            raise AssertionError(f"status payload 不一致：actual={payload} expected={expected_payload}")
        raw_text = status_path.read_text(encoding="utf-8")
        if "中文状态检查" not in raw_text or "\\u4e2d" in raw_text:
            raise AssertionError("中文 message 被 ASCII 转义或未按 UTF-8 保留")
        print("通过：status.json 可读，中文 message 保持 UTF-8")

        write_status_json(
            status_path,
            status="completed",
            phase="done",
            message="第二次覆盖",
            extra={"step": 2, "ok": True},
        )
        overwritten = read_json(status_path)
        if overwritten.get("status") != "completed" or overwritten.get("step") != 2:
            raise AssertionError(f"第二次写入未正确覆盖旧内容：{overwritten}")
        if "running" in status_path.read_text(encoding="utf-8"):
            raise AssertionError("原子覆盖后仍残留旧 status 内容")
        print("通过：第二次写入原子覆盖旧 status 内容")

        nested_path = tmp_root / "nested" / "parent" / "metadata.json"
        atomic_write_json({"status": "ok", "message": "嵌套目录"}, nested_path)
        if read_json(nested_path) != {"status": "ok", "message": "嵌套目录"}:
            raise AssertionError("nested parent directory 自动创建或 JSON 写入失败")
        print("通过：nested parent directory 自动创建")

        built = build_status_payload(status="queued", extra={"rank": 3})
        if built != {"status": "queued", "rank": 3}:
            raise AssertionError(f"build_status_payload 输出不一致：{built}")
        try:
            build_status_payload(status="bad", extra=[("not", "dict")])  # type: ignore[arg-type]
        except TypeError:
            print("通过：extra 非 dict 时拒绝写入")
        else:
            raise AssertionError("extra 非 dict 未触发 TypeError")

    print("Stage 1 P4a JSON utils smoke 通过")


if __name__ == "__main__":
    run_smoke()
