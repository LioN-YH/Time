#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P11d canonical protocol run smoke。

输入：
    无命令行输入；测试内构造 tiny SampleManifest、临时 feature CSV 和 tempfile run_dir。

输出：
    标准输出打印中文检查日志；若 canonical dataflow 的 sample_key 保序、组件边界、
    run_dir 写出或 artifact schema 不符合预期则抛出 AssertionError。

关键约束：
    只验证 tiny fixture 上的 canonical dataflow，不迁移正式入口，不新增 launcher/scripts，
    不访问 /data2，不启动训练，不修改 legacy CSV/summary/metadata/status/checkpoint schema。
"""

from __future__ import annotations

import builtins
import csv
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import TimeFuseFeatureCacheProvider  # noqa: E402
from time_router.models import TimeFuseLinearSoftmaxHead  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput, SampleManifest, SampleManifestRow  # noqa: E402
from time_router.runtime import (  # noqa: E402
    CANONICAL_RUN_SUBDIRS,
    create_run_dir,
    write_evaluation_summary,
    write_prediction_rows_csv,
    write_run_metadata,
    write_run_status,
    write_sample_manifest_ref,
    write_split_summary,
)


CREATED_AT = "2026-06-20T13:20:00+08:00"
MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
FEATURE_COLUMNS = ("trend_strength", "seasonality_strength", "recent_volatility")
MANIFEST_ROWS = (
    SampleManifestRow(
        sample_key="96_48_S__vali__TINY_DATA__item10__ch0__win0",
        split="vali",
        config_name="96_48_S",
        dataset_name="TINY_DATA",
        item_id=10,
        channel_id=0,
        window_index=0,
        seq_len=96,
        pred_len=48,
        extra={"fixture_role": "validation"},
    ),
    SampleManifestRow(
        sample_key="96_48_S__test__TINY_DATA__item10__ch0__win1",
        split="test",
        config_name="96_48_S",
        dataset_name="TINY_DATA",
        item_id=10,
        channel_id=0,
        window_index=1,
        seq_len=96,
        pred_len=48,
        extra={"fixture_role": "evaluation"},
    ),
    SampleManifestRow(
        sample_key="96_48_S__test__TINY_DATA__item11__ch1__win2",
        split="test",
        config_name="96_48_S",
        dataset_name="TINY_DATA",
        item_id=11,
        channel_id=1,
        window_index=2,
        seq_len=96,
        pred_len=48,
        extra={"fixture_role": "evaluation"},
    ),
)
SAMPLE_KEYS = tuple(row.sample_key for row in MANIFEST_ROWS)
FEATURE_MATRIX = np.asarray(
    [
        [0.10, 0.20, 0.05],
        [0.35, 0.10, 0.20],
        [0.05, 0.40, 0.15],
    ],
    dtype=np.float64,
)
HEAD_WEIGHT = np.asarray(
    [
        [0.20, -0.10, 0.05],
        [-0.05, 0.15, 0.10],
        [0.10, 0.05, -0.20],
    ],
    dtype=np.float64,
)
HEAD_BIAS = np.asarray([0.02, -0.01, 0.00], dtype=np.float64)
CSV_SAMPLE_ORDER = (SAMPLE_KEYS[2], SAMPLE_KEYS[0], SAMPLE_KEYS[1])


class TinyExpertProvider:
    """
    类功能：
        smoke-only ExpertProvider mock，用内存 fixture 构造 ExpertBatch。

    输入：
        sample_keys: Runtime 从 SampleManifest 取得的显式样本顺序。

    输出：
        `load_batch` 返回 ExpertBatch，sample_keys/model_columns/y_pred/y_true 全部按输入保序。

    关键约束：
        不接收、不保存、不推导 run_dir；不读取 prediction cache 或 /data2。
    """

    provider_name = "TinyExpertProvider"

    def __init__(self) -> None:
        self.received_run_dir = False
        self._y_true_by_key = {
            SAMPLE_KEYS[0]: np.asarray([[1.0], [1.2]], dtype=np.float64),
            SAMPLE_KEYS[1]: np.asarray([[2.0], [1.8]], dtype=np.float64),
            SAMPLE_KEYS[2]: np.asarray([[0.5], [0.7]], dtype=np.float64),
        }
        self._y_pred_by_key = {
            SAMPLE_KEYS[0]: np.asarray([[[1.1], [1.3]], [[0.9], [1.1]], [[1.4], [1.5]]], dtype=np.float64),
            SAMPLE_KEYS[1]: np.asarray([[[2.2], [1.9]], [[1.7], [1.6]], [[2.1], [1.7]]], dtype=np.float64),
            SAMPLE_KEYS[2]: np.asarray([[[0.6], [0.8]], [[0.4], [0.6]], [[0.9], [1.0]]], dtype=np.float64),
        }

    def load_batch(self, sample_keys: Sequence[str]) -> ExpertBatch:
        """函数功能：按 Runtime 显式传入的 sample_key 顺序返回 tiny ExpertBatch。"""
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        if ordered_keys != SAMPLE_KEYS:
            raise AssertionError(f"TinyExpertProvider 必须收到 manifest row order：{ordered_keys}")
        y_true = np.stack([self._y_true_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        y_pred = np.stack([self._y_pred_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        return ExpertBatch(
            sample_keys=ordered_keys,
            model_columns=MODEL_COLUMNS,
            y_pred=y_pred,
            y_true=y_true,
            row_index_metadata={"source": "memory://p11d_tiny_expert_fixture"},
            extra={"provider_name": self.provider_name, "fixture": "p11d_tiny_expert_batch"},
        )


def write_feature_csv(feature_csv_path: Path) -> None:
    """
    函数功能：
        写入临时 feature CSV；CSV 行顺序刻意不同于 manifest，用于验证 FeatureProvider 保序。
    """
    matrix_by_key = {sample_key: FEATURE_MATRIX[index] for index, sample_key in enumerate(SAMPLE_KEYS)}
    with feature_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_key", *FEATURE_COLUMNS))
        writer.writeheader()
        for sample_key in CSV_SAMPLE_ORDER:
            row = {"sample_key": sample_key}
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = f"{matrix_by_key[sample_key][feature_index]:.8f}"
            writer.writerow(row)


@contextmanager
def allow_only_feature_csv_reads(feature_csv_path: Path) -> Iterator[None]:
    """函数功能：限制 FeatureProvider 阶段只能读取本 smoke 创建的临时 feature CSV。"""
    original_open = builtins.open
    original_path_open = Path.open
    allowed_path = feature_csv_path.resolve()

    def checked_open(file: object, *args: object, **kwargs: object) -> object:
        path = Path(file).resolve()
        if path != allowed_path:
            raise AssertionError(f"FeatureProvider 不应读取临时 feature CSV 之外的路径：{path}")
        return original_open(file, *args, **kwargs)

    def checked_path_open(path_self: Path, *args: object, **kwargs: object) -> object:
        path = path_self.resolve()
        if path != allowed_path:
            raise AssertionError(f"FeatureProvider 不应读取临时 feature CSV 之外的路径：{path}")
        return original_path_open(path_self, *args, **kwargs)

    with patch.object(builtins, "open", side_effect=checked_open), patch.object(Path, "open", checked_path_open):
        yield


@contextmanager
def forbid_non_runtime_file_io() -> Iterator[None]:
    """
    函数功能：
        在 Head/Evaluator 阶段阻断常见文件 IO，证明它们只处理内存协议对象。
    """
    with patch.object(builtins, "open", side_effect=fail_non_runtime_io), patch.object(
        Path, "open", side_effect=fail_non_runtime_io
    ), patch.object(np, "load", side_effect=fail_non_runtime_io), patch.object(
        np, "save", side_effect=fail_non_runtime_io
    ), patch.object(
        np, "savez", side_effect=fail_non_runtime_io
    ):
        yield


def fail_non_runtime_io(*args: object, **kwargs: object) -> object:
    """函数功能：Head/Evaluator 不应访问文件系统或写训练产物。"""
    raise AssertionError(f"Provider/Head/Evaluator 边界内不应访问文件系统：args={args} kwargs={kwargs}")


def load_json(path: Path) -> dict[str, object]:
    """函数功能：读取 Runtime writer 写出的 JSON object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def build_evaluation_summary_payload(result: EvaluationInputAdapterResult) -> dict[str, object]:
    """
    函数功能：
        将 Evaluator 内存 summary 包装为 P11c writer 要求的最小 evaluation summary schema。
    """
    summary = dict(result.summary)
    return {
        "evaluation_schema_version": "stage1_evaluation_summary_v1",
        "sample_count": int(summary["num_samples"]),
        "metrics": {
            "hard_mae": summary["hard_mae"],
            "hard_mse": summary["hard_mse"],
            "raw_soft_mae": summary["raw_soft_mae"],
            "raw_soft_mse": summary["raw_soft_mse"],
            "mean_entropy": summary["mean_entropy"],
            "mean_max_weight": summary["mean_max_weight"],
        },
        "selected_counts": summary["selected_counts"],
        "model_columns": summary["model_columns"],
    }


def build_prediction_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    manifest: SampleManifest,
    result: EvaluationInputAdapterResult,
) -> list[dict[str, object]]:
    """
    函数功能：
        将 Evaluator 逐样本 rows 补齐 split、y_true、y_pred 后交给 Runtime writer。
    """
    split_by_key = {row.sample_key: row.split for row in manifest.rows}
    y_true = np.asarray(result.evaluation_input.y_true)
    hard_pred = np.asarray(result.hard_result.fused_pred)
    prediction_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        sample_key = str(row["sample_key"])
        prediction_rows.append(
            {
                "sample_key": sample_key,
                "split": split_by_key[sample_key],
                "selected_model": row["selected_model"],
                "selected_index": row["selected_index"],
                "y_true": float(np.mean(y_true[row_index])),
                "y_pred": float(np.mean(hard_pred[row_index])),
                "hard_mae": row["hard_mae"],
                "hard_mse": row["hard_mse"],
                "raw_soft_mae": row["raw_soft_mae"],
                "raw_soft_mse": row["raw_soft_mse"],
                "max_weight": row["max_weight"],
                "weight_entropy": row["weight_entropy"],
            }
        )
    return prediction_rows


def assert_memory_dataflow(
    *,
    manifest: SampleManifest,
    expert_batch: ExpertBatch,
    feature_batch: FeatureBatch,
    router_output: RouterOutput,
    result: EvaluationInputAdapterResult,
) -> None:
    """函数功能：验证 SampleManifest 到 Evaluator 的 ordered sample_keys 贯通。"""
    ordered_keys = manifest.sample_keys()
    if ordered_keys != SAMPLE_KEYS:
        raise AssertionError(f"SampleManifest sample_keys 顺序异常：{ordered_keys}")
    if manifest.sample_keys("vali") != (SAMPLE_KEYS[0],) or manifest.sample_keys("test") != SAMPLE_KEYS[1:]:
        raise AssertionError("SampleManifest split 过滤顺序异常")
    if manifest.split_counts() != {"vali": 1, "test": 2}:
        raise AssertionError(f"SampleManifest split_counts 异常：{manifest.split_counts()}")
    if expert_batch.sample_keys != ordered_keys:
        raise AssertionError("ExpertBatch 未保持 manifest sample_key 顺序")
    if feature_batch.sample_keys != ordered_keys:
        raise AssertionError("FeatureBatch 未保持 manifest sample_key 顺序")
    if router_output.sample_keys != ordered_keys:
        raise AssertionError("RouterOutput 未保持 manifest sample_key 顺序")
    if result.evaluation_input.sample_keys != ordered_keys:
        raise AssertionError("EvaluationInput 未保持 manifest sample_key 顺序")
    if [row["sample_key"] for row in result.per_sample_rows] != list(ordered_keys):
        raise AssertionError("per-sample evaluation rows 未保持 manifest sample_key 顺序")
    if feature_batch.feature_schema.get("feature_columns") != FEATURE_COLUMNS:
        raise AssertionError(f"FeatureBatch schema 漂移：{feature_batch.feature_schema}")
    np.testing.assert_allclose(feature_batch.features, FEATURE_MATRIX.astype(np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(np.sum(router_output.weights, axis=1), np.ones(len(SAMPLE_KEYS)), rtol=0.0, atol=1e-9)
    if result.evaluation_input.y_pred is not expert_batch.y_pred or result.evaluation_input.y_true is not expert_batch.y_true:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch arrays")
    if result.evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")


def assert_run_artifacts(run_dir: Path, *, expected_prediction_rows: list[dict[str, object]]) -> None:
    """函数功能：读取 canonical run_dir，并验证 artifact 分层、schema 和 row order。"""
    for subdir in CANONICAL_RUN_SUBDIRS:
        if not (run_dir / subdir).is_dir():
            raise AssertionError(f"canonical 子目录缺失：{subdir}")

    metadata = load_json(run_dir / "run_metadata.json")
    status = load_json(run_dir / "run_status.json")
    manifest_ref = load_json(run_dir / "inputs" / "sample_manifest_ref.json")
    split_summary = load_json(run_dir / "inputs" / "split_summary.json")
    evaluation_summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")

    if metadata["run_artifact_schema_version"] != "stage1_run_artifact_v1":
        raise AssertionError("run_metadata schema version 异常")
    if metadata["protocol_version"] != "stage1_canonical_protocol_run_smoke_v1":
        raise AssertionError("run_metadata protocol_version 异常")
    if status["status"] != "completed" or status["current_stage"] != "canonical_protocol_smoke":
        raise AssertionError(f"run_status 异常：{status}")
    if manifest_ref["sample_manifest_schema_version"] != "stage1_sample_manifest_v1":
        raise AssertionError("sample_manifest_ref schema version 异常")
    if manifest_ref["row_count"] != len(SAMPLE_KEYS):
        raise AssertionError("sample_manifest_ref row_count 异常")
    if split_summary["split_summary_schema_version"] != "stage1_split_summary_v1":
        raise AssertionError("split_summary schema version 异常")
    if split_summary["sample_count_by_split"] != {"vali": 1, "test": 2}:
        raise AssertionError(f"split_summary count 异常：{split_summary['sample_count_by_split']}")
    if evaluation_summary["evaluation_schema_version"] != "stage1_evaluation_summary_v1":
        raise AssertionError("evaluation_summary schema version 异常")
    if evaluation_summary["sample_count"] != len(SAMPLE_KEYS):
        raise AssertionError("evaluation_summary sample_count 异常")

    prediction_csv = run_dir / "predictions" / "prediction_rows.csv"
    if not prediction_csv.is_file():
        raise AssertionError("prediction_rows.csv 未写入 predictions/")
    if (run_dir / "evaluation" / "prediction_rows.csv").exists():
        raise AssertionError("prediction rows 不应写入 evaluation/")
    if (run_dir / "predictions" / "evaluation_summary.json").exists():
        raise AssertionError("evaluation summary 不应写入 predictions/")
    with prediction_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["sample_key"] for row in rows] != list(SAMPLE_KEYS):
        raise AssertionError(f"prediction rows sample_key 顺序漂移：{rows}")
    if len(rows) != len(expected_prediction_rows):
        raise AssertionError(f"prediction rows 行数异常：{len(rows)}")
    if [row["split"] for row in rows] != ["vali", "test", "test"]:
        raise AssertionError(f"prediction rows split 异常：{rows}")
    if str(run_dir).startswith("/data2/"):
        raise AssertionError("P11d smoke 不应在 /data2 写 run_dir")


def run_smoke() -> None:
    """函数功能：执行 Stage 1 P11d tiny canonical protocol run smoke。"""
    print("开始 Stage 1 P11d canonical protocol run smoke")
    manifest = SampleManifest(rows=MANIFEST_ROWS, extra={"fixture": "p11d_tiny_manifest"})
    manifest.validate_unique_sample_keys()
    ordered_sample_keys = manifest.sample_keys()
    print("通过：SampleManifest 构造完成，并按 row order 输出 ordered sample_keys")

    with tempfile.TemporaryDirectory(prefix="stage1_p11d_canonical_protocol_") as temp_dir:
        temp_root = Path(temp_dir)
        feature_csv_path = temp_root / "p11d_timefuse_features.csv"
        write_feature_csv(feature_csv_path)
        if str(temp_root.resolve()).startswith("/data2/"):
            raise AssertionError("P11d smoke 不应在 /data2 构造临时 fixture")

        expert_provider = TinyExpertProvider()
        expert_batch = expert_provider.load_batch(ordered_sample_keys)
        with allow_only_feature_csv_reads(feature_csv_path):
            feature_provider = TimeFuseFeatureCacheProvider(
                feature_csv_path=feature_csv_path,
                feature_columns=FEATURE_COLUMNS,
                feature_schema_name="p11d_tiny_timefuse_feature_schema_v1",
            )
            feature_batch = feature_provider.load_batch(expert_batch.sample_keys)
        print("通过：ExpertProvider / FeatureProvider 只接收 ordered sample_keys，不接收 run_dir")

        head = TimeFuseLinearSoftmaxHead(weight=HEAD_WEIGHT, bias=HEAD_BIAS)
        adapter = EvaluationInputAdapter()
        with forbid_non_runtime_file_io():
            router_output = head.predict(feature_batch, expert_batch.model_columns)
            result = adapter.evaluate(expert_batch=expert_batch, router_output=router_output)
        print("通过：RouterHead / EvaluationInputAdapter 仅处理内存协议对象，不访问文件系统")

        assert_memory_dataflow(
            manifest=manifest,
            expert_batch=expert_batch,
            feature_batch=feature_batch,
            router_output=router_output,
            result=result,
        )
        print("通过：ordered sample_keys 从 SampleManifest 贯通到 predictions rows")

        run_dir = create_run_dir(temp_root / "run_outputs", run_name="p11d_canonical_protocol_smoke")
        manifest_ref = {
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "reference_type": "inline_fixture",
            "path": "memory://p11d_tiny_sample_manifest",
            "checksum": "not_applicable_tiny_inline_fixture",
            "checksum_algorithm": "none",
            "row_count": len(manifest.rows),
            "ordered_sample_keys_policy": "manifest_row_order",
            "created_at": CREATED_AT,
        }
        split_summary = {
            "split_summary_schema_version": "stage1_split_summary_v1",
            "split_strategy_name": "p11d_tiny_vali_test_fixture",
            "config_name": "96_48_S",
            "split_names": ["vali", "test"],
            "sample_count_by_split": manifest.split_counts(),
            "unique_sample_key_count": len(ordered_sample_keys),
            "duplicate_sample_key_count": 0,
            "split_overlap_check": {
                "default_policy": "mutually_exclusive",
                "allowed_overlap": False,
                "overlap_sample_key_count": 0,
                "overlap_examples": [],
            },
            "ordered_sample_keys_policy": "manifest_row_order",
            "source_manifest_reference": manifest_ref,
            "created_at": CREATED_AT,
        }
        metadata = {
            "run_artifact_schema_version": "stage1_run_artifact_v1",
            "protocol_version": "stage1_canonical_protocol_run_smoke_v1",
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "evaluation_schema_version": "stage1_evaluation_summary_v1",
            "config_name": "96_48_S",
            "branch_name": "p11d_canonical_protocol_run_smoke",
            "created_at": CREATED_AT,
            "inputs": {
                "sample_manifest": "inputs/sample_manifest_ref.json",
                "split_summary": "inputs/split_summary.json",
                "feature_fixture": str(feature_csv_path),
            },
        }
        status = {
            "status": "completed",
            "current_stage": "canonical_protocol_smoke",
            "updated_at": CREATED_AT,
            "failure_reason": None,
            "checkpoint_pointer": None,
        }
        evaluation_summary = build_evaluation_summary_payload(result)
        prediction_rows = build_prediction_rows(rows=result.per_sample_rows, manifest=manifest, result=result)

        write_run_metadata(run_dir, metadata)
        write_run_status(run_dir, status)
        write_sample_manifest_ref(run_dir, manifest_ref)
        write_split_summary(run_dir, split_summary)
        write_evaluation_summary(run_dir, evaluation_summary)
        write_prediction_rows_csv(
            run_dir,
            prediction_rows,
            fieldnames=(
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
            ),
        )
        print("通过：Runtime artifact writer 是唯一写 run_dir 的组件")

        if expert_provider.received_run_dir:
            raise AssertionError("ExpertProvider 不应接收 run_dir")
        assert_run_artifacts(run_dir, expected_prediction_rows=prediction_rows)
        print("通过：canonical run_dir artifact 可读，predictions/ 与 evaluation/ 分层正确")

    print("完成：Stage 1 P11d canonical protocol run smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
