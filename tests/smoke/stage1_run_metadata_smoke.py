#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P4c run metadata payload builder 的最小 smoke。

输入：
    无命令行输入；所有写入都发生在 tempfile.TemporaryDirectory 下。

输出：
    标准输出打印中文检查日志；任一 payload 契约不一致时抛出异常。

关键约束：
    该脚本不访问 /data2，不访问 full-scale 输出目录，不读取训练配置。
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.io import build_run_metadata, write_run_metadata  # noqa: E402


def assert_utc_timestamp(value: str) -> None:
    """函数功能：验证 created_at_utc 是可解析且带时区的 ISO 时间字符串。"""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AssertionError(f"created_at_utc 不是 timezone-aware ISO 字符串：{value}")


def read_json(path: Path) -> dict:
    """函数功能：按 UTF-8 读取 JSON 文件，供 smoke 断言使用。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_smoke() -> None:
    """函数功能：执行 P4c run metadata builder 的临时目录 smoke 检查。"""
    print("开始 Stage 1 P4c run metadata smoke")

    with tempfile.TemporaryDirectory(prefix="stage1_run_metadata_smoke_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_path = tmp_root / "inputs" / "manifest.csv"
        output_path = tmp_root / "outputs" / "metadata.json"

        payload = build_run_metadata(
            stage="p4c_smoke",
            entrypoint=Path("tests/smoke/stage1_run_metadata_smoke.py"),
            command=["python", "tests/smoke/stage1_run_metadata_smoke.py"],
            inputs={"manifest": input_path, "nested": {"root": tmp_root}},
            outputs={"metadata": output_path},
            git_ref="manual-smoke-ref",
            notes="中文 metadata smoke",
            extra={"constraints": ["tempfile only", Path("no_full_scale")]},
        )
        required_keys = {"stage", "created_at_utc", "inputs", "outputs"}
        if not required_keys.issubset(payload):
            raise AssertionError(f"metadata payload 缺少基础字段：payload_keys={sorted(payload)}")
        if payload["stage"] != "p4c_smoke":
            raise AssertionError(f"stage 不一致：{payload['stage']}")
        assert_utc_timestamp(payload["created_at_utc"])
        if payload["inputs"]["manifest"] != str(input_path):
            raise AssertionError("inputs 内 Path 未转换为字符串")
        if payload["inputs"]["nested"]["root"] != str(tmp_root):
            raise AssertionError("inputs nested Path 未转换为字符串")
        if payload["outputs"]["metadata"] != str(output_path):
            raise AssertionError("outputs 内 Path 未转换为字符串")
        if payload["entrypoint"] != "tests/smoke/stage1_run_metadata_smoke.py":
            raise AssertionError("entrypoint Path 未转换为字符串")
        if payload["extra"]["constraints"][1] != "no_full_scale":
            raise AssertionError("extra 内 Path 未转换为字符串")
        print("通过：metadata payload 基础字段、UTC 时间和 Path 转字符串")

        for bad_stage in ("", None):
            try:
                build_run_metadata(stage=bad_stage)  # type: ignore[arg-type]
            except ValueError:
                continue
            raise AssertionError(f"stage={bad_stage!r} 未触发 ValueError")
        print("通过：stage 必须是非空字符串")

        invalid_cases = [
            ("inputs", {"inputs": ["not", "dict"]}),
            ("outputs", {"outputs": ["not", "dict"]}),
            ("extra", {"extra": ["not", "dict"]}),
        ]
        for field_name, kwargs in invalid_cases:
            try:
                build_run_metadata(stage="bad", **kwargs)  # type: ignore[arg-type]
            except TypeError:
                continue
            raise AssertionError(f"{field_name} 非 dict 未触发 TypeError")
        print("通过：inputs/outputs/extra 非 dict 时会明确失败")

        metadata_path = tmp_root / "nested" / "metadata.json"
        write_run_metadata(
            metadata_path,
            stage="p4c_write_smoke",
            inputs={"fixture": tmp_root / "fixture"},
            outputs={"metadata": metadata_path},
            notes="写入 tempfile metadata",
        )
        written = read_json(metadata_path)
        if written["stage"] != "p4c_write_smoke":
            raise AssertionError(f"写入 metadata stage 不一致：{written}")
        assert_utc_timestamp(written["created_at_utc"])
        if written["inputs"]["fixture"] != str(tmp_root / "fixture"):
            raise AssertionError("写入 metadata 后 Path 未保持字符串")
        if not metadata_path.exists():
            raise AssertionError("write_run_metadata 未写出 tempfile metadata.json")
        print("通过：write_run_metadata 只在 tempfile 下写入 JSON 且可读")

    print("Stage 1 P4c run metadata smoke 通过")


if __name__ == "__main__":
    run_smoke()
