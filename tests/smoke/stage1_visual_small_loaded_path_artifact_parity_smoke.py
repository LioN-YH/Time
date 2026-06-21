#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16k Visual small entrypoint loaded path artifact parity smoke。

输入：
    使用 tempfile output root，分别运行 `scripts/run_stage1_visual_small.py`
    默认 mock path 和显式 loaded legacy path。loaded path 使用 P16c precomputed
    visual feature fixture 与 tempfile tiny checkpoint payload，本 smoke 选择 no-scaler
    口径以专注 artifact parity。

输出：
    标准输出打印中文检查日志；若两条 Visual small path 的 canonical run_dir
    共同结构、metadata schema、evaluation summary schema、prediction rows schema
    或 sample_key 顺序发生分叉则抛错。

关键约束：
    本 smoke 不比较 metrics 数值优劣；不读取真实 checkpoint，不访问 `/data2`，
    不启动 ViT/transformers，不调用或迁移正式 streaming 训练入口，不修改 TimeFuse
    small entrypoint，也不把 checkpoint/scaler/run_dir 放进 P16a adapter。
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage1_visual_small import (  # noqa: E402
    DEFAULT_EXPERT_PREDICTIONS_JSON,
    DEFAULT_HISTORY_WINDOWS_JSON,
    DEFAULT_SAMPLE_MANIFEST_CSV,
)


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
STREAMING_ENTRYPOINT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"
TIMEFUSE_SMALL_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_timefuse_small.py"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
VISUAL_FEATURES_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_precomputed_small" / "visual_embeddings.csv"
CANONICAL_SUBDIRS = ("inputs", "indexes", "predictions", "evaluation", "checkpoints", "logs")
REQUIRED_ARTIFACTS = (
    "run_metadata.json",
    "run_status.json",
    "inputs/sample_manifest_ref.json",
    "inputs/split_summary.json",
    "evaluation/evaluation_summary.json",
    "predictions/prediction_rows.csv",
    "logs/visual_small_entrypoint.log",
)
COMMON_METADATA_FIELDS = (
    "run_artifact_schema_version",
    "protocol_version",
    "sample_manifest_schema_version",
    "evaluation_schema_version",
    "config_name",
    "branch_name",
    "created_at",
    "inputs",
    "visual_router",
)
COMMON_VISUAL_METADATA_FIELDS = (
    "feature_source",
    "feature_provider",
    "feature_schema",
    "scaler_enabled",
    "loaded_legacy_mlp",
    "checkpoint_payload_source",
    "training",
    "formal_visual_router_migration",
    "loads_real_checkpoint",
    "loads_real_vit",
    "feature_lineage",
    "head_lineage",
)
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
EXPECTED_MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
EXPECTED_FEATURE_DIM = 8
HIDDEN_DIM = 11
FORBIDDEN_RUNTIME_TOKENS = (
    "/data2",
    "ViTModel",
    "AutoImageProcessor",
    "train_visual_router_online_streaming",
)


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON artifact 并确认其为 object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    """函数功能：读取 prediction_rows.csv，并确认共同字段存在。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError(f"{path} 缺少 CSV 表头")
        missing = [field for field in PREDICTION_FIELDS if field not in reader.fieldnames]
        if missing:
            raise AssertionError(f"{path} 缺少 prediction rows 字段：{missing}")
        rows = list(reader)
    if not rows:
        raise AssertionError(f"{path} 不应为空")
    return rows


def assert_has_fields(payload: Mapping[str, Any], fields: Sequence[str], *, name: str) -> None:
    """函数功能：检查 artifact object 包含指定字段。"""
    missing = [field for field in fields if field not in payload]
    if missing:
        raise AssertionError(f"{name} 缺少字段：{missing}")


def assert_finite_number(value: object, *, name: str) -> float:
    """函数功能：将 artifact 字段转为 float，并要求为有限数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name} 不能转为 float：{value!r}") from exc
    if not math.isfinite(number):
        raise AssertionError(f"{name} 必须是有限数值：{number!r}")
    return number


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    函数功能：
        基于 legacy VisualMLPRouter 的参数形状生成 deterministic tiny state_dict。

    关键约束：
        这里只写 tempfile small payload；不读取真实 checkpoint。
    """
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        tensor = torch.linspace(-0.04 + index * 0.01, 0.04 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_tiny_checkpoint(path: Path) -> None:
    """函数功能：创建 loaded legacy path 使用的 tempfile tiny checkpoint payload。"""
    if str(path.resolve()).startswith("/data2/"):
        raise AssertionError(f"P16k smoke 不应向 /data2 写 checkpoint：{path}")
    module = importlib.import_module(LEGACY_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    model = router_cls(input_dim=EXPECTED_FEATURE_DIM, hidden_dim=HIDDEN_DIM, output_dim=len(EXPECTED_MODEL_COLUMNS), dropout=0.0)
    payload: Mapping[str, Any] = {
        "router_state_dict": build_fake_state_dict(model),
        "config": {
            "input_dim": EXPECTED_FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": len(EXPECTED_MODEL_COLUMNS),
            "dropout": 0.0,
            "payload_name": "p16k_tempfile_tiny_checkpoint_no_scaler",
        },
        "metadata": {
            "stage": "P16k",
            "source": "tempfile checkpoint payload artifact parity smoke",
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }
    torch.save(dict(payload), path)


def run_visual_entrypoint(cmd: list[str], *, run_name: str) -> subprocess.CompletedProcess[str]:
    """函数功能：运行 Visual small entrypoint，并检查 stdout/stderr 禁止 token。"""
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
            f"{run_name} 返回码异常：returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if "run_dir:" not in completed.stdout:
        raise AssertionError(f"{run_name} stdout 未包含 run_dir：{completed.stdout}")
    stdout_stderr = completed.stdout + completed.stderr
    for token in FORBIDDEN_RUNTIME_TOKENS:
        if token in stdout_stderr:
            raise AssertionError(f"{run_name} stdout/stderr 不应出现禁止 token：{token}")
    return completed


def collect_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """函数功能：加载 canonical run_dir 共同 artifact，并检查共同目录/文件存在。"""
    if not run_dir.is_dir():
        raise AssertionError(f"run_dir 不存在：{run_dir}")
    for subdir in CANONICAL_SUBDIRS:
        if not (run_dir / subdir).is_dir():
            raise AssertionError(f"{run_dir} 缺少 canonical 子目录：{subdir}")
    for relative_path in REQUIRED_ARTIFACTS:
        if not (run_dir / relative_path).is_file():
            raise AssertionError(f"{run_dir} 缺少共同 artifact：{relative_path}")
    return {
        "metadata": load_json(run_dir / "run_metadata.json"),
        "status": load_json(run_dir / "run_status.json"),
        "manifest_ref": load_json(run_dir / "inputs" / "sample_manifest_ref.json"),
        "split_summary": load_json(run_dir / "inputs" / "split_summary.json"),
        "evaluation_summary": load_json(run_dir / "evaluation" / "evaluation_summary.json"),
        "prediction_rows": read_prediction_rows(run_dir / "predictions" / "prediction_rows.csv"),
    }


def assert_metadata_schema(artifacts: Mapping[str, Any], *, label: str) -> None:
    """函数功能：检查 run_metadata/run_status 的共同 Visual small schema。"""
    metadata = artifacts["metadata"]
    status = artifacts["status"]
    if not isinstance(metadata, Mapping) or not isinstance(status, Mapping):
        raise AssertionError(f"{label} metadata/status 必须是 object")
    assert_has_fields(metadata, COMMON_METADATA_FIELDS, name=f"{label} run_metadata")
    assert_has_fields(status, COMMON_STATUS_FIELDS, name=f"{label} run_status")
    if metadata["branch_name"] != "visual_router_small":
        raise AssertionError(f"{label} branch_name 异常：{metadata}")
    if status["status"] != "completed":
        raise AssertionError(f"{label} run_status 未 completed：{status}")
    if status["failure_reason"] is not None or status["checkpoint_pointer"] is not None:
        raise AssertionError(f"{label} failure/checkpoint 边界异常：{status}")
    inputs = metadata["inputs"]
    if not isinstance(inputs, Mapping):
        raise AssertionError(f"{label} metadata.inputs 必须是 object")
    for field in ("sample_manifest", "split_summary", "expert_predictions_json"):
        if field not in inputs:
            raise AssertionError(f"{label} metadata.inputs 缺少字段：{field}")

    visual_router = metadata["visual_router"]
    if not isinstance(visual_router, Mapping):
        raise AssertionError(f"{label} metadata.visual_router 必须是 object")
    assert_has_fields(visual_router, COMMON_VISUAL_METADATA_FIELDS, name=f"{label} metadata.visual_router")
    fixed_expected = {
        "loads_real_checkpoint": False,
        "loads_real_vit": False,
        "formal_visual_router_migration": False,
    }
    for field, expected in fixed_expected.items():
        if visual_router[field] is not expected:
            raise AssertionError(f"{label} visual_router.{field} 异常：{visual_router[field]!r}")


def assert_evaluation_summary_schema(artifacts: Mapping[str, Any], *, label: str) -> None:
    """函数功能：检查 evaluation_summary 共同字段和有限指标。"""
    summary = artifacts["evaluation_summary"]
    if not isinstance(summary, Mapping):
        raise AssertionError(f"{label} evaluation_summary 必须是 object")
    assert_has_fields(summary, COMMON_EVALUATION_FIELDS, name=f"{label} evaluation_summary")
    metrics = summary["metrics"]
    model_columns = summary["model_columns"]
    selected_counts = summary["selected_counts"]
    if not isinstance(metrics, Mapping):
        raise AssertionError(f"{label} metrics 必须是 object")
    if not isinstance(model_columns, list) or not model_columns:
        raise AssertionError(f"{label} model_columns 必须是非空 list")
    if len(model_columns) != len(set(str(model) for model in model_columns)):
        raise AssertionError(f"{label} model_columns 不应重复：{model_columns}")
    if not isinstance(selected_counts, Mapping):
        raise AssertionError(f"{label} selected_counts 必须是 object")
    for metric_name in COMMON_METRIC_FIELDS:
        if metric_name not in metrics:
            raise AssertionError(f"{label} metrics 缺少字段：{metric_name}")
        assert_finite_number(metrics[metric_name], name=f"{label} metrics.{metric_name}")
    invalid_selected = [key for key in selected_counts if key not in model_columns]
    if invalid_selected:
        raise AssertionError(f"{label} selected_counts 出现非法模型名：{invalid_selected}")


def assert_prediction_rows_schema(artifacts: Mapping[str, Any], *, label: str) -> None:
    """函数功能：检查 prediction_rows 共同字段、模型选择合法性和有限指标。"""
    summary = artifacts["evaluation_summary"]
    rows = artifacts["prediction_rows"]
    if not isinstance(summary, Mapping) or not isinstance(rows, list):
        raise AssertionError(f"{label} summary/rows 类型异常")
    model_columns = summary["model_columns"]
    if not isinstance(model_columns, list) or not model_columns:
        raise AssertionError(f"{label} model_columns 类型异常")
    if len(rows) != int(summary["sample_count"]):
        raise AssertionError(f"{label} prediction_rows 行数与 sample_count 不一致")
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AssertionError(f"{label} prediction row 必须是 object：index={row_index}")
        if row["selected_model"] not in model_columns:
            raise AssertionError(f"{label} selected_model 不属于 model_columns：{row}")
        selected_index = int(str(row["selected_index"]))
        if not 0 <= selected_index < len(model_columns):
            raise AssertionError(f"{label} selected_index 越界：{row}")
        for field in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse", "max_weight", "weight_entropy"):
            assert_finite_number(row[field], name=f"{label} row[{row_index}].{field}")


def assert_cross_path_parity(default_path: Mapping[str, Any], loaded_path: Mapping[str, Any]) -> None:
    """函数功能：比较默认 mock path 与 loaded legacy path 的共同 artifact parity。"""
    default_rows = default_path["prediction_rows"]
    loaded_rows = loaded_path["prediction_rows"]
    default_summary = default_path["evaluation_summary"]
    loaded_summary = loaded_path["evaluation_summary"]
    default_manifest = default_path["manifest_ref"]
    loaded_manifest = loaded_path["manifest_ref"]
    default_split = default_path["split_summary"]
    loaded_split = loaded_path["split_summary"]
    if not all(isinstance(item, Mapping) for item in (default_summary, loaded_summary, default_manifest, loaded_manifest, default_split, loaded_split)):
        raise AssertionError("共同 JSON artifact 类型异常")
    if len(default_rows) != len(loaded_rows):  # type: ignore[arg-type]
        raise AssertionError("两条 path prediction_rows 行数不一致")
    if default_summary["sample_count"] != loaded_summary["sample_count"]:
        raise AssertionError("两条 path evaluation_summary.sample_count 不一致")
    if default_summary["model_columns"] != loaded_summary["model_columns"]:
        raise AssertionError("两条 path model_columns 不一致")
    if default_manifest["row_count"] != loaded_manifest["row_count"]:
        raise AssertionError("两条 path sample_manifest_ref.row_count 不一致")
    if default_split["sample_count_by_split"] != loaded_split["sample_count_by_split"]:
        raise AssertionError("两条 path split_summary.sample_count_by_split 不一致")

    default_sample_keys = [str(row["sample_key"]) for row in default_rows]  # type: ignore[index]
    loaded_sample_keys = [str(row["sample_key"]) for row in loaded_rows]  # type: ignore[index]
    if default_sample_keys != loaded_sample_keys:
        raise AssertionError(f"两条 path sample_key 顺序不一致：{default_sample_keys} vs {loaded_sample_keys}")
    default_splits = [str(row["split"]) for row in default_rows]  # type: ignore[index]
    loaded_splits = [str(row["split"]) for row in loaded_rows]  # type: ignore[index]
    if default_splits != loaded_splits:
        raise AssertionError(f"两条 path split 列不一致：{default_splits} vs {loaded_splits}")


def assert_path_specific_metadata(
    *,
    default_path: Mapping[str, Any],
    loaded_path: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    """函数功能：检查默认 path 与 loaded path 各自必须保留的 metadata 边界。"""
    default_metadata = default_path["metadata"]
    loaded_metadata = loaded_path["metadata"]
    if not isinstance(default_metadata, Mapping) or not isinstance(loaded_metadata, Mapping):
        raise AssertionError("metadata 类型异常")
    default_visual = default_metadata["visual_router"]
    loaded_visual = loaded_metadata["visual_router"]
    if not isinstance(default_visual, Mapping) or not isinstance(loaded_visual, Mapping):
        raise AssertionError("visual_router metadata 类型异常")

    expected_default = {
        "feature_source": "mock",
        "loaded_legacy_mlp": False,
        "checkpoint_payload_source": "none",
        "scaler_enabled": False,
    }
    for field, expected in expected_default.items():
        if default_visual.get(field) != expected:
            raise AssertionError(f"default path metadata {field} 异常：{default_visual}")

    expected_loaded = {
        "feature_source": "precomputed",
        "loaded_legacy_mlp": True,
        "checkpoint_payload_source": "explicit_small_fixture",
        "scaler_enabled": False,
        "p16i_helper_used": True,
        "p16a_adapter_used": True,
    }
    for field, expected in expected_loaded.items():
        if loaded_visual.get(field) != expected:
            raise AssertionError(f"loaded path metadata {field} 异常：{loaded_visual}")
    if loaded_visual.get("checkpoint_payload_loaded") not in (True, None):
        raise AssertionError(f"checkpoint_payload_loaded 若存在必须为 true：{loaded_visual}")
    head_lineage = loaded_visual["head_lineage"]
    if not isinstance(head_lineage, Mapping):
        raise AssertionError("loaded path head_lineage 必须是 object")
    if head_lineage.get("checkpoint_payload_source") != "explicit_small_fixture":
        raise AssertionError(f"loaded path head_lineage 未记录 explicit small fixture：{head_lineage}")
    if head_lineage.get("checkpoint_payload_path") != str(checkpoint_path):
        raise AssertionError(f"loaded path head_lineage checkpoint path 异常：{head_lineage}")
    feature_lineage = loaded_visual["feature_lineage"]
    if not isinstance(feature_lineage, Mapping):
        raise AssertionError("loaded path feature_lineage 必须是 object")
    if feature_lineage.get("feature_source") != "precomputed":
        raise AssertionError(f"loaded path feature_lineage 未记录 precomputed：{feature_lineage}")


def run_smoke() -> None:
    """函数功能：执行 P16k 两条 Visual small path artifact parity smoke。"""
    print("开始 Stage 1 P16k Visual small loaded path artifact parity smoke")
    if not VISUAL_FEATURES_CSV.is_file():
        raise AssertionError("P16k smoke 需要 P16c precomputed visual feature fixture")
    before = {
        ENTRYPOINT: ENTRYPOINT.read_bytes(),
        STREAMING_ENTRYPOINT: STREAMING_ENTRYPOINT.read_bytes(),
        TIMEFUSE_SMALL_ENTRYPOINT: TIMEFUSE_SMALL_ENTRYPOINT.read_bytes(),
    }

    with tempfile.TemporaryDirectory(prefix="stage1_p16k_visual_loaded_path_parity_") as temp_dir:
        temp_root = Path(temp_dir)
        output_dir = temp_root / "run_outputs"
        checkpoint_path = temp_root / "tiny_legacy_visual_mlp_payload.pt"
        save_tiny_checkpoint(checkpoint_path)

        run_visual_entrypoint(
            [
                sys.executable,
                str(ENTRYPOINT),
                "--sample-manifest-csv",
                str(DEFAULT_SAMPLE_MANIFEST_CSV),
                "--history-windows-json",
                str(DEFAULT_HISTORY_WINDOWS_JSON),
                "--expert-predictions-json",
                str(DEFAULT_EXPERT_PREDICTIONS_JSON),
                "--output-dir",
                str(output_dir),
                "--split-name",
                "test",
                "--run-id",
                "p16k_visual_default_mock_path",
            ],
            run_name="Visual default mock path",
        )
        run_visual_entrypoint(
            [
                sys.executable,
                str(ENTRYPOINT),
                "--sample-manifest-csv",
                str(DEFAULT_SAMPLE_MANIFEST_CSV),
                "--expert-predictions-json",
                str(DEFAULT_EXPERT_PREDICTIONS_JSON),
                "--output-dir",
                str(output_dir),
                "--split-name",
                "test",
                "--run-id",
                "p16k_visual_loaded_legacy_path",
                "--feature-source",
                "precomputed",
                "--visual-features-csv",
                str(VISUAL_FEATURES_CSV),
                "--use-loaded-legacy-mlp",
                "--router-checkpoint-payload",
                str(checkpoint_path),
            ],
            run_name="Visual loaded legacy path",
        )
        print("通过：默认 mock path 与 loaded legacy path subprocess 均 completed，stdout/stderr 未出现禁止 token")

        default_artifacts = collect_run_artifacts(output_dir / "p16k_visual_default_mock_path")
        loaded_artifacts = collect_run_artifacts(output_dir / "p16k_visual_loaded_legacy_path")
        print("通过：两条 path canonical run_dir 共同目录、inputs/evaluation/predictions/logs 均存在")

        for label, artifacts in (("default mock path", default_artifacts), ("loaded legacy path", loaded_artifacts)):
            assert_metadata_schema(artifacts, label=label)
            assert_evaluation_summary_schema(artifacts, label=label)
            assert_prediction_rows_schema(artifacts, label=label)
        assert_cross_path_parity(default_artifacts, loaded_artifacts)
        assert_path_specific_metadata(default_path=default_artifacts, loaded_path=loaded_artifacts, checkpoint_path=checkpoint_path)
        print("通过：metadata/evaluation/prediction rows schema 与 sample_key/split/model_columns parity 成立")

    for path, content in before.items():
        if path.read_bytes() != content:
            raise AssertionError(f"P16k smoke 不应修改入口文件：{path}")
    print("完成：Stage 1 P16k Visual small loaded path artifact parity smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
