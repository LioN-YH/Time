#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P15d branch-specific small entrypoint artifact parity smoke。

输入：
    使用仓库内默认 small fixture，通过 subprocess 分别运行
    `scripts/run_stage1_timefuse_small.py` 和 `scripts/run_stage1_visual_small.py`。

输出：
    标准输出打印中文检查日志；若两个 branch-specific small entrypoint 写出的
    canonical run_dir 共同结构、共同 schema、ordered sample_keys 或边界发生漂移则抛错。

关键约束：
    本 smoke 只比较 small canonical artifact parity，不比较 TimeFuse/Visual 指标优劣；
    不访问 `/data2`，不启动正式训练、pressure 或 full-scale，不读取真实 checkpoint，
    不启动 ViT，也不修改 generic/TimeFuse/Visual small CLI。
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage1_timefuse_small import (  # noqa: E402
    DEFAULT_EXPERT_PREDICTIONS_JSON as TIMEFUSE_EXPERT_PREDICTIONS_JSON,
    DEFAULT_FEATURES_CSV,
    DEFAULT_SAMPLE_MANIFEST_CSV as TIMEFUSE_SAMPLE_MANIFEST_CSV,
)
from scripts.run_stage1_visual_small import (  # noqa: E402
    DEFAULT_EXPERT_PREDICTIONS_JSON as VISUAL_EXPERT_PREDICTIONS_JSON,
    DEFAULT_HISTORY_WINDOWS_JSON,
    DEFAULT_SAMPLE_MANIFEST_CSV as VISUAL_SAMPLE_MANIFEST_CSV,
)


GENERIC_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_canonical_small.py"
TIMEFUSE_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_timefuse_small.py"
VISUAL_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
CANONICAL_SUBDIRS = ("inputs", "indexes", "predictions", "evaluation", "checkpoints", "logs")
COMMON_METADATA_FIELDS = (
    "run_artifact_schema_version",
    "protocol_version",
    "sample_manifest_schema_version",
    "evaluation_schema_version",
    "config_name",
    "branch_name",
    "created_at",
    "inputs",
)
COMMON_INPUT_FIELDS = ("sample_manifest", "split_summary", "expert_predictions_json")
COMMON_STATUS_FIELDS = ("status", "current_stage", "updated_at", "failure_reason", "checkpoint_pointer")
COMMON_EVALUATION_FIELDS = (
    "evaluation_schema_version",
    "sample_count",
    "metrics",
    "selected_counts",
    "model_columns",
)
COMMON_METRIC_FIELDS = (
    "hard_mae",
    "hard_mse",
    "raw_soft_mae",
    "raw_soft_mse",
    "mean_entropy",
    "mean_max_weight",
)
PREDICTION_FIELDS = (
    "sample_key",
    "split",
    "selected_model",
    "selected_index",
    "y_true",
    "y_pred",
    "hard_mae",
    "hard_mse",
    "raw_soft_mae",
    "raw_soft_mse",
    "max_weight",
    "weight_entropy",
)
FORBIDDEN_RUNTIME_TOKENS = (
    "/data2",
    "train_visual_router_online_streaming",
    "train_timefuse_fusor_streaming",
    "torch.load",
    "ViTModel",
    "AutoImageProcessor",
)


def load_json(path: Path) -> dict[str, object]:
    """函数功能：读取 JSON artifact 并确认其为 object，便于后续 schema 断言。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    """函数功能：读取 prediction_rows.csv，并确认 CSV 至少有一行数据。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"{path} 缺少 CSV 表头")
        missing = [field for field in PREDICTION_FIELDS if field not in reader.fieldnames]
        if missing:
            raise AssertionError(f"{path} 缺少 prediction schema 字段：{missing}")
        rows = list(reader)
    if not rows:
        raise AssertionError(f"{path} 不应为空")
    return rows


def assert_has_fields(payload: Mapping[str, object], fields: Sequence[str], *, name: str) -> None:
    """函数功能：检查 payload 包含指定字段，错误信息带 artifact 名称。"""
    missing = [field for field in fields if field not in payload]
    if missing:
        raise AssertionError(f"{name} 缺少字段：{missing}")


def assert_finite_number(value: object, *, name: str) -> float:
    """函数功能：将 artifact 字段转成 float，并要求为有限数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name} 不能转为 float：{value!r}") from exc
    if not math.isfinite(number):
        raise AssertionError(f"{name} 必须为有限数值：{number!r}")
    return number


def assert_entrypoint_files_unchanged(before: Mapping[Path, bytes]) -> None:
    """函数功能：确认 smoke 运行不会修改 generic/branch-specific small CLI 文件。"""
    for path, content in before.items():
        if path.read_bytes() != content:
            raise AssertionError(f"small CLI 文件在 smoke 运行前后发生变化：{path}")


def run_entrypoint(cmd: list[str], *, run_name: str) -> subprocess.CompletedProcess[str]:
    """函数功能：运行指定 small entrypoint，并检查运行时禁止 token 未出现在 stdout/stderr。"""
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
            f"{run_name} 返回码异常：returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if "run_dir:" not in completed.stdout:
        raise AssertionError(f"{run_name} stdout 未包含 run_dir：{completed.stdout}")
    stdout_stderr = completed.stdout + completed.stderr
    for token in FORBIDDEN_RUNTIME_TOKENS:
        if token in stdout_stderr:
            raise AssertionError(f"{run_name} stdout/stderr 不应出现禁止 token：{token}")
    return completed


def collect_run_artifacts(run_dir: Path, *, expected_log_name: str) -> dict[str, object]:
    """函数功能：加载 canonical run_dir 的共同 artifact，并检查共同目录结构。"""
    if not run_dir.is_dir():
        raise AssertionError(f"run_dir 不存在：{run_dir}")
    for subdir in CANONICAL_SUBDIRS:
        if not (run_dir / subdir).is_dir():
            raise AssertionError(f"{run_dir} 缺少 canonical 子目录：{subdir}")
    required_files = (
        "run_metadata.json",
        "run_status.json",
        "inputs/sample_manifest_ref.json",
        "inputs/split_summary.json",
        "evaluation/evaluation_summary.json",
        "predictions/prediction_rows.csv",
    )
    for relative_path in required_files:
        if not (run_dir / relative_path).is_file():
            raise AssertionError(f"{run_dir} 缺少 canonical artifact：{relative_path}")
    if not (run_dir / "logs" / expected_log_name).is_file():
        raise AssertionError(f"{run_dir} 缺少最小日志文件：logs/{expected_log_name}")

    return {
        "metadata": load_json(run_dir / "run_metadata.json"),
        "status": load_json(run_dir / "run_status.json"),
        "manifest_ref": load_json(run_dir / "inputs" / "sample_manifest_ref.json"),
        "split_summary": load_json(run_dir / "inputs" / "split_summary.json"),
        "evaluation_summary": load_json(run_dir / "evaluation" / "evaluation_summary.json"),
        "prediction_rows": read_prediction_rows(run_dir / "predictions" / "prediction_rows.csv"),
    }


def assert_common_metadata_schema(artifacts: Mapping[str, object], *, branch_label: str) -> None:
    """函数功能：检查 run_metadata 和 run_status 的共同 canonical schema。"""
    metadata = artifacts["metadata"]
    status = artifacts["status"]
    if not isinstance(metadata, Mapping) or not isinstance(status, Mapping):
        raise AssertionError(f"{branch_label} metadata/status 必须是 mapping")
    assert_has_fields(metadata, COMMON_METADATA_FIELDS, name=f"{branch_label} run_metadata")
    inputs = metadata["inputs"]
    if not isinstance(inputs, Mapping):
        raise AssertionError(f"{branch_label} run_metadata.inputs 必须是 object")
    assert_has_fields(inputs, COMMON_INPUT_FIELDS, name=f"{branch_label} run_metadata.inputs")
    assert_has_fields(status, COMMON_STATUS_FIELDS, name=f"{branch_label} run_status")
    if status["status"] != "completed":
        raise AssertionError(f"{branch_label} run_status.status 应为 completed：{status}")
    if status["failure_reason"] is not None or status["checkpoint_pointer"] is not None:
        raise AssertionError(f"{branch_label} run_status failure/checkpoint 边界异常：{status}")


def assert_input_consistency(timefuse: Mapping[str, object], visual: Mapping[str, object]) -> None:
    """函数功能：比较两个 branch small run 的 manifest/split/config/sample_key 顺序一致性。"""
    timefuse_manifest = timefuse["manifest_ref"]
    visual_manifest = visual["manifest_ref"]
    timefuse_split = timefuse["split_summary"]
    visual_split = visual["split_summary"]
    timefuse_rows = timefuse["prediction_rows"]
    visual_rows = visual["prediction_rows"]
    timefuse_metadata = timefuse["metadata"]
    visual_metadata = visual["metadata"]
    if not all(isinstance(item, Mapping) for item in (timefuse_manifest, visual_manifest, timefuse_split, visual_split)):
        raise AssertionError("manifest_ref/split_summary 必须是 mapping")
    if timefuse_manifest["row_count"] != visual_manifest["row_count"]:
        raise AssertionError("两边 sample_manifest_ref.row_count 不一致")
    if timefuse_split["sample_count_by_split"] != visual_split["sample_count_by_split"]:
        raise AssertionError("两边 split_summary.sample_count_by_split 不一致")
    timefuse_sample_keys = [str(row["sample_key"]) for row in timefuse_rows]  # type: ignore[index]
    visual_sample_keys = [str(row["sample_key"]) for row in visual_rows]  # type: ignore[index]
    if timefuse_sample_keys != visual_sample_keys:
        raise AssertionError(f"两边 prediction_rows sample_key 顺序不一致：{timefuse_sample_keys} vs {visual_sample_keys}")
    timefuse_splits = [str(row["split"]) for row in timefuse_rows]  # type: ignore[index]
    visual_splits = [str(row["split"]) for row in visual_rows]  # type: ignore[index]
    if timefuse_splits != visual_splits:
        raise AssertionError(f"两边 prediction_rows split 列不一致：{timefuse_splits} vs {visual_splits}")
    if timefuse_metadata["config_name"] != visual_metadata["config_name"]:  # type: ignore[index]
        raise AssertionError("两边 config_name 不一致")


def assert_evaluation_summary_schema(artifacts: Mapping[str, object], *, branch_label: str) -> None:
    """函数功能：检查 evaluation summary 共同字段、model_columns、selected_counts 和有限指标。"""
    summary = artifacts["evaluation_summary"]
    if not isinstance(summary, Mapping):
        raise AssertionError(f"{branch_label} evaluation_summary 必须是 mapping")
    assert_has_fields(summary, COMMON_EVALUATION_FIELDS, name=f"{branch_label} evaluation_summary")
    metrics = summary["metrics"]
    model_columns = summary["model_columns"]
    selected_counts = summary["selected_counts"]
    if not isinstance(metrics, Mapping):
        raise AssertionError(f"{branch_label} metrics 必须是 object")
    if not isinstance(model_columns, list) or not model_columns:
        raise AssertionError(f"{branch_label} model_columns 必须是非空 list")
    if len(model_columns) != len(set(str(model) for model in model_columns)):
        raise AssertionError(f"{branch_label} model_columns 不应重复：{model_columns}")
    if not isinstance(selected_counts, Mapping):
        raise AssertionError(f"{branch_label} selected_counts 必须是 object")
    for metric_name in COMMON_METRIC_FIELDS:
        if metric_name not in metrics:
            raise AssertionError(f"{branch_label} metrics 缺少字段：{metric_name}")
        assert_finite_number(metrics[metric_name], name=f"{branch_label} metrics.{metric_name}")
    invalid_selected = [key for key in selected_counts if key not in model_columns]
    if invalid_selected:
        raise AssertionError(f"{branch_label} selected_counts 出现非法模型名：{invalid_selected}")


def assert_prediction_rows_schema(artifacts: Mapping[str, object], *, branch_label: str) -> None:
    """函数功能：检查 prediction rows 共同字段、合法 selected index 和有限逐样本指标。"""
    summary = artifacts["evaluation_summary"]
    rows = artifacts["prediction_rows"]
    if not isinstance(summary, Mapping) or not isinstance(rows, list):
        raise AssertionError(f"{branch_label} summary/rows 类型异常")
    model_columns = summary["model_columns"]
    if not isinstance(model_columns, list) or not model_columns:
        raise AssertionError(f"{branch_label} model_columns 类型异常")
    if len(rows) != int(summary["sample_count"]):
        raise AssertionError(f"{branch_label} prediction row 数与 sample_count 不一致")
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AssertionError(f"{branch_label} prediction row 必须是 mapping：index={row_index}")
        if row["selected_model"] not in model_columns:
            raise AssertionError(f"{branch_label} selected_model 不属于 model_columns：{row}")
        try:
            selected_index = int(str(row["selected_index"]))
        except ValueError as exc:
            raise AssertionError(f"{branch_label} selected_index 不是整数：{row}") from exc
        if not 0 <= selected_index < len(model_columns):
            raise AssertionError(f"{branch_label} selected_index 越界：{row}")
        for field in (
            "y_true",
            "y_pred",
            "hard_mae",
            "hard_mse",
            "raw_soft_mae",
            "raw_soft_mse",
            "max_weight",
            "weight_entropy",
        ):
            assert_finite_number(row[field], name=f"{branch_label} row[{row_index}].{field}")


def assert_cross_branch_evaluation_parity(timefuse: Mapping[str, object], visual: Mapping[str, object]) -> None:
    """函数功能：比较两边 evaluation summary 的共同规模和专家列，不比较数值优劣。"""
    timefuse_summary = timefuse["evaluation_summary"]
    visual_summary = visual["evaluation_summary"]
    timefuse_rows = timefuse["prediction_rows"]
    visual_rows = visual["prediction_rows"]
    if not isinstance(timefuse_summary, Mapping) or not isinstance(visual_summary, Mapping):
        raise AssertionError("evaluation summary 类型异常")
    if timefuse_summary["sample_count"] != visual_summary["sample_count"]:
        raise AssertionError("两边 evaluation sample_count 不一致")
    if timefuse_summary["model_columns"] != visual_summary["model_columns"]:
        raise AssertionError("两边 model_columns 不一致")
    if len(timefuse_rows) != len(visual_rows):  # type: ignore[arg-type]
        raise AssertionError("两边 prediction_rows 行数不一致")


def assert_branch_specific_metadata(timefuse: Mapping[str, object], visual: Mapping[str, object]) -> None:
    """函数功能：检查 TimeFuse/Visual 允许存在且必须存在的 branch-specific metadata。"""
    timefuse_metadata = timefuse["metadata"]
    visual_metadata = visual["metadata"]
    if not isinstance(timefuse_metadata, Mapping) or not isinstance(visual_metadata, Mapping):
        raise AssertionError("metadata 类型异常")
    if timefuse_metadata["branch_name"] != "timefuse_fusor_small":
        raise AssertionError(f"TimeFuse branch_name 异常：{timefuse_metadata}")
    timefuse_fusor = timefuse_metadata.get("timefuse_fusor")
    if not isinstance(timefuse_fusor, Mapping):
        raise AssertionError("TimeFuse run_metadata 缺少 timefuse_fusor object")
    if timefuse_fusor.get("training") != "not_started_p15b_small_rehearsal_only":
        raise AssertionError(f"TimeFuse training metadata 异常：{timefuse_fusor}")
    timefuse_inputs = timefuse_metadata["inputs"]
    if not isinstance(timefuse_inputs, Mapping):
        raise AssertionError("TimeFuse inputs 类型异常")
    features_csv = timefuse_inputs.get("features_csv")
    if not isinstance(features_csv, Mapping) or features_csv.get("feature_dim") != 17:
        raise AssertionError(f"TimeFuse features_csv.feature_dim 应为 17：{features_csv}")

    if visual_metadata["branch_name"] != "visual_router_small":
        raise AssertionError(f"Visual branch_name 异常：{visual_metadata}")
    visual_router = visual_metadata.get("visual_router")
    if not isinstance(visual_router, Mapping):
        raise AssertionError("Visual run_metadata 缺少 visual_router object")
    expected_visual_fields = {
        "training": "not_started_p15c_small_rehearsal_only",
        "formal_visual_router_migration": False,
        "loads_real_checkpoint": False,
        "loads_real_vit": False,
        "feature_provider": "VisualMockFeatureProvider",
    }
    for field, expected in expected_visual_fields.items():
        if visual_router.get(field) != expected:
            raise AssertionError(f"Visual metadata 字段异常：{field}={visual_router.get(field)!r}")


def run_smoke() -> None:
    """函数功能：执行 P15d 跨分支 small entrypoint artifact parity smoke。"""
    print("开始 Stage 1 P15d branch-specific small entrypoint artifact parity smoke")
    before = {
        GENERIC_ENTRYPOINT: GENERIC_ENTRYPOINT.read_bytes(),
        TIMEFUSE_ENTRYPOINT: TIMEFUSE_ENTRYPOINT.read_bytes(),
        VISUAL_ENTRYPOINT: VISUAL_ENTRYPOINT.read_bytes(),
    }
    with tempfile.TemporaryDirectory(prefix="stage1_p15d_branch_artifact_parity_") as temp_dir:
        output_dir = Path(temp_dir) / "run_outputs"
        if str(output_dir.resolve()).startswith("/data2/"):
            raise AssertionError("P15d smoke 不应使用 /data2 tempfile")
        run_entrypoint(
            [
                sys.executable,
                str(TIMEFUSE_ENTRYPOINT),
                "--sample-manifest-csv",
                str(TIMEFUSE_SAMPLE_MANIFEST_CSV),
                "--features-csv",
                str(DEFAULT_FEATURES_CSV),
                "--expert-predictions-json",
                str(TIMEFUSE_EXPERT_PREDICTIONS_JSON),
                "--output-dir",
                str(output_dir),
                "--split-name",
                "test",
                "--run-id",
                "p15d_timefuse_artifact_parity",
            ],
            run_name="TimeFuse small entrypoint",
        )
        run_entrypoint(
            [
                sys.executable,
                str(VISUAL_ENTRYPOINT),
                "--sample-manifest-csv",
                str(VISUAL_SAMPLE_MANIFEST_CSV),
                "--history-windows-json",
                str(DEFAULT_HISTORY_WINDOWS_JSON),
                "--expert-predictions-json",
                str(VISUAL_EXPERT_PREDICTIONS_JSON),
                "--output-dir",
                str(output_dir),
                "--split-name",
                "test",
                "--run-id",
                "p15d_visual_artifact_parity",
            ],
            run_name="Visual small entrypoint",
        )
        print("通过：两个 branch-specific small entrypoint subprocess 均完成，stdout/stderr 未出现禁止 token")

        timefuse = collect_run_artifacts(
            output_dir / "p15d_timefuse_artifact_parity",
            expected_log_name="timefuse_small_entrypoint.log",
        )
        visual = collect_run_artifacts(
            output_dir / "p15d_visual_artifact_parity",
            expected_log_name="visual_small_entrypoint.log",
        )
        print("通过：两边 canonical run_dir 共同结构、inputs/evaluation/predictions/logs 均存在")

        assert_common_metadata_schema(timefuse, branch_label="TimeFuse")
        assert_common_metadata_schema(visual, branch_label="Visual")
        assert_input_consistency(timefuse, visual)
        print("通过：共同 metadata/status schema、split/input consistency 和 sample_key 顺序一致")

        assert_evaluation_summary_schema(timefuse, branch_label="TimeFuse")
        assert_evaluation_summary_schema(visual, branch_label="Visual")
        assert_prediction_rows_schema(timefuse, branch_label="TimeFuse")
        assert_prediction_rows_schema(visual, branch_label="Visual")
        assert_cross_branch_evaluation_parity(timefuse, visual)
        print("通过：evaluation summary 与 prediction_rows 共同 schema 一致，指标字段均为有限值")

        assert_branch_specific_metadata(timefuse, visual)
        print("通过：TimeFuse/Visual branch-specific metadata 均符合 P15d 边界")

    assert_entrypoint_files_unchanged(before)
    print("完成：Stage 1 P15d branch-specific small entrypoint artifact parity smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
