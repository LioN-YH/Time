#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P13b real-derived small fixture smoke。

输入：
    仓库内 `tests/fixtures/stage1_real_derived_small/` 的 sample manifest、
    feature CSV 和 expert JSON；通过 subprocess 调用 P12b small canonical entrypoint。

输出：
    标准输出打印中文检查日志；若 canonical run_dir 未写出、sample_key 顺序未保持
    manifest 行顺序、metadata inputs 来源摘要缺失、evaluation sample_count 异常或
    stdout/stderr 出现 `/data2` 则抛错。

关键约束：
    只验证 real-derived / schema-style small fixture，不访问 `/data2`，不启动正式训练、
    pressure 或 full-scale，不迁移正式 Visual Router / TimeFuse 入口。
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
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 canonical run artifact 中的 JSON object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """函数功能：读取 CSV 并保留字段字符串，便于做顺序和计数断言。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_repo_fixture(path: Path) -> None:
    """函数功能：确认 fixture 存在且位于仓库内而非 `/data2`。"""
    if not path.is_file():
        raise AssertionError(f"fixture 文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P13b fixture 不应位于 /data2：{path}")


def run_smoke() -> None:
    """函数功能：执行 P13b real-derived small fixture smoke 验收。"""
    print("开始 Stage 1 P13b real-derived small fixture smoke")

    sample_manifest = FIXTURE_ROOT / "sample_manifest.csv"
    features = FIXTURE_ROOT / "features.csv"
    expert_fixture = FIXTURE_ROOT / "expert_predictions.json"
    for path in (sample_manifest, features, expert_fixture):
        assert_repo_fixture(path)

    manifest_rows = load_csv_rows(sample_manifest)
    if not manifest_rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    expected_keys = tuple(row["sample_key"] for row in manifest_rows)
    if len(expected_keys) != len(set(expected_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{expected_keys}")

    # P13b 只接受 P11b 最小 manifest 字段，避免把 supervision、feature 或 prediction 路径混入 manifest。
    expected_manifest_columns = (
        "sample_key",
        "split",
        "config_name",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "seq_len",
        "pred_len",
    )
    if tuple(manifest_rows[0].keys()) != expected_manifest_columns:
        raise AssertionError(f"sample_manifest.csv 字段不是 P11b 最小字段：{tuple(manifest_rows[0].keys())}")

    feature_rows = load_csv_rows(features)
    feature_keys = tuple(row["sample_key"] for row in feature_rows)
    if set(feature_keys) != set(expected_keys):
        raise AssertionError(f"features.csv sample_key 集合未与 manifest 对齐：{feature_keys}")
    if feature_keys == expected_keys:
        raise AssertionError("features.csv 行顺序应刻意不同于 manifest，用于验证按 sample_key join")

    expert_payload = load_json(expert_fixture)
    expert_keys = tuple(str(sample["sample_key"]) for sample in expert_payload["samples"])
    if set(expert_keys) != set(expected_keys):
        raise AssertionError(f"expert_predictions.json sample_key 集合未与 manifest 对齐：{expert_keys}")
    if expert_keys == expected_keys:
        raise AssertionError("expert_predictions.json sample 顺序应刻意不同于 manifest，用于验证按 sample_key join")
    print("通过：real-derived fixture 文件存在、manifest 字段最小、feature/expert 键集合对齐且行顺序打乱")

    with tempfile.TemporaryDirectory(prefix="stage1_p13b_real_derived_small_") as temp_dir:
        output_root = Path(temp_dir) / "run_outputs"
        run_name = "real_derived_fixture"
        cmd = [
            sys.executable,
            str(ENTRYPOINT),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--sample-manifest",
            str(sample_manifest),
            "--feature-source",
            str(features),
            "--expert-fixture",
            str(expert_fixture),
            "--strict",
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
                f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        if "/data2" in completed.stdout or "/data2" in completed.stderr:
            raise AssertionError("P13b smoke stdout/stderr 不应出现 /data2")

        run_dir = output_root / run_name
        if not run_dir.is_dir():
            raise AssertionError(f"entrypoint 未创建 canonical run_dir：{run_dir}")
        for relative_path in (
            "run_metadata.json",
            "run_status.json",
            "inputs/sample_manifest_ref.json",
            "inputs/split_summary.json",
            "evaluation/evaluation_summary.json",
            "predictions/prediction_rows.csv",
        ):
            if not (run_dir / relative_path).is_file():
                raise AssertionError(f"canonical run_dir 缺少 artifact：{relative_path}")
        print("通过：P12b small entrypoint 使用 P13b fixture 写出 canonical run_dir")

        prediction_rows = load_csv_rows(run_dir / "predictions" / "prediction_rows.csv")
        actual_keys = tuple(row["sample_key"] for row in prediction_rows)
        if actual_keys != expected_keys:
            raise AssertionError(f"prediction_rows.csv 未保持 manifest 行顺序：{actual_keys}")

        metadata = load_json(run_dir / "run_metadata.json")
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

        evaluation_summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")
        if evaluation_summary.get("sample_count") != len(manifest_rows):
            raise AssertionError(f"evaluation_summary sample_count 异常：{evaluation_summary}")

        split_summary = load_json(run_dir / "inputs" / "split_summary.json")
        if split_summary.get("sample_count_by_split") != {"vali": 2, "test": 2}:
            raise AssertionError(f"split_summary split count 异常：{split_summary}")
        print("通过：prediction rows 保持 manifest 顺序，metadata inputs 和 evaluation sample_count 正确")

    print("完成：Stage 1 P13b real-derived small fixture smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
