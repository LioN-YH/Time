#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P12b small canonical entrypoint 显式 fixture input contract smoke。

输入：
    仓库内 `tests/fixtures/stage1_canonical_small/` 的 tiny manifest、feature CSV
    和 expert JSON；测试内用 tempfile 分别运行默认内联 fixture 与显式 fixture。

输出：
    标准输出打印中文检查日志；若显式 fixture 输出与默认内联输出不一致、metadata
    input 摘要缺失、sample_key 顺序漂移或错误引用 `/data2` 则抛错。

关键约束：
    只验证 small/tiny input contract，不访问 `/data2`，不启动训练，不迁移正式入口。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_canonical_small.py"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_canonical_small"
EXPECTED_SAMPLE_KEYS = (
    "96_48_S__vali__TINY_DATA__item10__ch0__win0",
    "96_48_S__test__TINY_DATA__item10__ch0__win1",
    "96_48_S__test__TINY_DATA__item11__ch1__win2",
)


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 canonical run artifact 中的 JSON object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    """函数功能：读取 prediction_rows.csv 并保留 CSV 原始字符串字段。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_entrypoint(output_root: Path, run_name: str, extra_args: list[str]) -> Path:
    """函数功能：调用 small canonical entrypoint 并返回 run_dir。"""
    cmd = [
        sys.executable,
        str(ENTRYPOINT),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
        *extra_args,
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "entrypoint subprocess 返回码异常："
            f"run_name={run_name}, returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if "/data2" in completed.stdout or "/data2" in completed.stderr:
        raise AssertionError("P12b fixture smoke stdout/stderr 不应出现 /data2")
    run_dir = output_root / run_name
    if not run_dir.is_dir():
        raise AssertionError(f"entrypoint 未创建 run_dir：{run_dir}")
    return run_dir


def canonicalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    函数功能：
        比较显式 fixture 与内联 fixture 的 prediction rows 时保留业务字段。
    """
    ignored_fields: set[str] = set()
    return [{key: value for key, value in row.items() if key not in ignored_fields} for row in rows]


def run_smoke() -> None:
    """函数功能：执行 P12b 显式 small fixture input contract smoke。"""
    print("开始 Stage 1 P12b small canonical fixture input contract smoke")

    sample_manifest = FIXTURE_ROOT / "sample_manifest.csv"
    features = FIXTURE_ROOT / "features.csv"
    expert_fixture = FIXTURE_ROOT / "expert_predictions.json"
    for path in (sample_manifest, features, expert_fixture):
        if not path.is_file():
            raise AssertionError(f"fixture 文件缺失：{path}")
        if str(path.resolve()).startswith("/data2/"):
            raise AssertionError(f"fixture 文件不应位于 /data2：{path}")

    with tempfile.TemporaryDirectory(prefix="stage1_p12b_small_fixture_") as temp_dir:
        output_root = Path(temp_dir) / "run_outputs"
        inline_run = run_entrypoint(output_root, "inline_fixture", [])
        explicit_run = run_entrypoint(
            output_root,
            "explicit_fixture",
            [
                "--sample-manifest",
                str(sample_manifest),
                "--feature-source",
                str(features),
                "--expert-fixture",
                str(expert_fixture),
            ],
        )
        print("通过：默认内联 fixture 与显式 fixture entrypoint 均成功运行")

        inline_rows = load_prediction_rows(inline_run / "predictions" / "prediction_rows.csv")
        explicit_rows = load_prediction_rows(explicit_run / "predictions" / "prediction_rows.csv")
        if canonicalize_rows(explicit_rows) != canonicalize_rows(inline_rows):
            raise AssertionError("显式 fixture prediction_rows.csv 与默认内联 fixture 不一致")
        actual_keys = tuple(row["sample_key"] for row in explicit_rows)
        if actual_keys != EXPECTED_SAMPLE_KEYS:
            raise AssertionError(f"显式 fixture prediction rows 未保持 manifest 行顺序：{actual_keys}")
        print("通过：显式 fixture 输出与内联 fixture prediction rows 一致，且保持 manifest 顺序")

        manifest_ref = load_json(explicit_run / "inputs" / "sample_manifest_ref.json")
        metadata = load_json(explicit_run / "run_metadata.json")
        if manifest_ref["reference_type"] != "file" or manifest_ref["path"] != str(sample_manifest):
            raise AssertionError(f"sample_manifest_ref 未记录显式 manifest 文件：{manifest_ref}")
        inputs = metadata.get("inputs")
        if not isinstance(inputs, dict):
            raise AssertionError(f"run_metadata.inputs 不是 object：{metadata}")
        expected_references = {
            "sample_manifest": str(sample_manifest),
            "feature_source": str(features),
            "expert_fixture": str(expert_fixture),
        }
        for key, expected_path in expected_references.items():
            reference = inputs.get(key)
            if not isinstance(reference, dict):
                raise AssertionError(f"run_metadata.inputs.{key} 缺少引用摘要：{inputs}")
            if reference.get("reference_type") != "file" or reference.get("path") != expected_path:
                raise AssertionError(f"run_metadata.inputs.{key} 引用摘要异常：{reference}")
        if inputs.get("sample_manifest_ref_artifact") != "inputs/sample_manifest_ref.json":
            raise AssertionError(f"run_metadata.inputs 缺少 sample_manifest_ref_artifact：{inputs}")
        print("通过：run_metadata.inputs 记录 sample_manifest / feature_source / expert_fixture 来源摘要")

        split_summary = load_json(explicit_run / "inputs" / "split_summary.json")
        if split_summary["sample_count_by_split"] != {"vali": 1, "test": 2}:
            raise AssertionError(f"显式 manifest split count 异常：{split_summary}")
        if metadata["protocol_version"] != "stage1_canonical_small_entrypoint_v1":
            raise AssertionError(f"protocol_version 异常：{metadata}")
        print("通过：显式 fixture artifact schema 与 split summary 保持 P12 口径")

    print("完成：Stage 1 P12b fixture input contract smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
