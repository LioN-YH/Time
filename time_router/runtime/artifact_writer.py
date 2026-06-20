"""
文件功能：
    提供 Stage 1 P11c 最小 canonical Runtime artifact writer。

关键约束：
    - 只由 Runtime 调用，负责创建 canonical run_dir 并写出 P11a/P11b
      定义的最小 artifact。
    - Provider、Head、Evaluator 不应接收或解析 run_dir；它们只返回内存对象
      或显式结构化 payload，由 Runtime 决定是否落盘。
    - 本文件不读取 /data2，不启动实验，不迁移 legacy entrypoint，也不实现
      checkpoint/resume 或复杂 Runtime framework。
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from time_router.io.json_utils import atomic_write_json


CANONICAL_RUN_SUBDIRS = ("inputs", "indexes", "predictions", "evaluation", "checkpoints", "logs")


def _coerce_mapping(payload: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """
    函数功能：
        将调用方传入的 mapping 复制为普通 dict，并对空 payload 给出清晰错误。

    输入：
        payload: 调用方显式构造的结构化写出内容。
        name: 用于错误信息的 artifact 名称。

    输出：
        可交给 JSON writer 的 dict 副本。
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} 必须是 mapping")
    result = dict(payload)
    if not result:
        raise ValueError(f"{name} 不能为空")
    return result


def _require_fields(payload: Mapping[str, Any], required_fields: Sequence[str], *, name: str) -> None:
    """
    函数功能：
        校验 canonical 最小字段存在，避免 smoke 写出看似成功但 schema contract 不完整。
    """
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{name} 缺少必需字段：{missing}")


def _ensure_run_dir(run_dir: str | os.PathLike[str]) -> Path:
    """
    函数功能：
        确认 run_dir 及 canonical 子目录存在。

    输入：
        run_dir: Runtime 显式传入的运行目录。

    输出：
        Path 类型 run_dir。
    """
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    for subdir in CANONICAL_RUN_SUBDIRS:
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path


def _timestamp_for_name() -> str:
    """函数功能：生成用于默认 run_name 的本地时间戳。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_run_name(run_name: str) -> str:
    """
    函数功能：
        将调用方传入的 run_name 限制为单层目录名，避免误把路径片段塞入 output_root。
    """
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name 必须是非空字符串")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip())
    normalized = normalized.strip("._")
    if not normalized:
        raise ValueError("run_name 清理后为空")
    return normalized


def create_run_dir(output_root: str | os.PathLike[str], run_name: str | None = None) -> Path:
    """
    函数功能：
        在 Runtime 指定的 output_root 下创建 canonical run_dir 和标准子目录。

    输入：
        output_root: Runtime/launcher 显式传入的输出根目录。
        run_name: 可选单层 run 目录名；为空时使用本地时间戳生成最小名称。

    输出：
        已创建完成的 run_dir Path。

    关键约束：
        本函数只操作调用方传入的本地路径，不读取配置、不推导 /data2、不检查训练状态。
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    name = _safe_run_name(run_name) if run_name is not None else f"stage1_run_{_timestamp_for_name()}"
    return _ensure_run_dir(root / name)


def write_json_atomic(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """
    函数功能：
        Runtime artifact writer 对现有 JSON 原子写入工具的薄封装。

    输入：
        path: 目标 JSON 路径。
        payload: 调用方显式构造的 mapping。

    输出：
        无返回值；成功后 path 包含 UTF-8 JSON。
    """
    atomic_write_json(_coerce_mapping(payload, name="json payload"), path)


def write_run_metadata(run_dir: str | os.PathLike[str], metadata: Mapping[str, Any]) -> Path:
    """函数功能：写出 canonical `run_metadata.json` 并返回文件路径。"""
    payload = _coerce_mapping(metadata, name="run_metadata")
    _require_fields(
        payload,
        (
            "run_artifact_schema_version",
            "protocol_version",
            "sample_manifest_schema_version",
            "evaluation_schema_version",
            "config_name",
            "branch_name",
            "created_at",
            "inputs",
        ),
        name="run_metadata",
    )
    path = _ensure_run_dir(run_dir) / "run_metadata.json"
    atomic_write_json(payload, path)
    return path


def write_run_status(run_dir: str | os.PathLike[str], status: Mapping[str, Any]) -> Path:
    """函数功能：写出 canonical `run_status.json` 并返回文件路径。"""
    payload = _coerce_mapping(status, name="run_status")
    _require_fields(
        payload,
        ("status", "current_stage", "updated_at", "failure_reason", "checkpoint_pointer"),
        name="run_status",
    )
    path = _ensure_run_dir(run_dir) / "run_status.json"
    atomic_write_json(payload, path)
    return path


def write_sample_manifest_ref(run_dir: str | os.PathLike[str], manifest_ref: Mapping[str, Any]) -> Path:
    """函数功能：写出 `inputs/sample_manifest_ref.json` 并返回文件路径。"""
    payload = _coerce_mapping(manifest_ref, name="sample_manifest_ref")
    _require_fields(
        payload,
        (
            "sample_manifest_schema_version",
            "reference_type",
            "path",
            "checksum",
            "checksum_algorithm",
            "row_count",
            "ordered_sample_keys_policy",
            "created_at",
        ),
        name="sample_manifest_ref",
    )
    path = _ensure_run_dir(run_dir) / "inputs" / "sample_manifest_ref.json"
    atomic_write_json(payload, path)
    return path


def write_split_summary(run_dir: str | os.PathLike[str], split_summary: Mapping[str, Any]) -> Path:
    """函数功能：写出 `inputs/split_summary.json` 并返回文件路径。"""
    payload = _coerce_mapping(split_summary, name="split_summary")
    _require_fields(
        payload,
        (
            "split_summary_schema_version",
            "split_strategy_name",
            "config_name",
            "split_names",
            "sample_count_by_split",
            "unique_sample_key_count",
            "duplicate_sample_key_count",
            "split_overlap_check",
            "ordered_sample_keys_policy",
            "source_manifest_reference",
            "created_at",
        ),
        name="split_summary",
    )
    path = _ensure_run_dir(run_dir) / "inputs" / "split_summary.json"
    atomic_write_json(payload, path)
    return path


def write_evaluation_summary(run_dir: str | os.PathLike[str], summary: Mapping[str, Any]) -> Path:
    """函数功能：写出 `evaluation/evaluation_summary.json` 并返回文件路径。"""
    payload = _coerce_mapping(summary, name="evaluation_summary")
    _require_fields(payload, ("evaluation_schema_version", "sample_count", "metrics"), name="evaluation_summary")
    path = _ensure_run_dir(run_dir) / "evaluation" / "evaluation_summary.json"
    atomic_write_json(payload, path)
    return path


def write_prediction_rows_csv(
    run_dir: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """
    函数功能：
        写出 `predictions/prediction_rows.csv`。

    输入：
        run_dir: Runtime 显式传入的运行目录。
        rows: 逐样本预测行，最小字段为 sample_key、selected_model、y_true、y_pred、split。
        fieldnames: 可选 CSV 字段顺序；为空时使用最小 canonical 字段顺序。

    输出：
        写出的 CSV 文件路径。
    """
    csv_fieldnames = tuple(fieldnames or ("sample_key", "selected_model", "y_true", "y_pred", "split"))
    required = {"sample_key", "selected_model", "y_true", "y_pred", "split"}
    missing_fields = required.difference(csv_fieldnames)
    if missing_fields:
        raise ValueError(f"prediction_rows fieldnames 缺少必需字段：{sorted(missing_fields)}")

    materialized_rows = [_coerce_mapping(row, name="prediction_row") for row in rows]
    for index, row in enumerate(materialized_rows):
        row_missing = required.difference(row)
        if row_missing:
            raise ValueError(f"prediction_rows 第 {index} 行缺少必需字段：{sorted(row_missing)}")

    path = _ensure_run_dir(run_dir) / "predictions" / "prediction_rows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized_rows)
    return path
