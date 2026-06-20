#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P12 small canonical entrypoint thin slice。

输入：
    命令行显式传入 `--output-root` 和 `--run-name`；可选指定 config、branch、
    feature CSV 和 strict 校验开关。

输出：
    在 `output_root/run_name/` 下创建 canonical run_dir，并写出最小 run metadata、
    status、inputs、evaluation summary 和 prediction rows。

关键约束：
    本脚本只包装 tiny canonical dataflow，不迁移正式 Visual Router / TimeFuse
    entrypoint，不新增 Bash launcher，不访问 `/data2`，不启动训练或 full-scale。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import TimeFuseFeatureCacheProvider  # noqa: E402
from time_router.models import TimeFuseLinearSoftmaxHead  # noqa: E402
from time_router.protocols import ExpertBatch, SampleManifest, SampleManifestRow  # noqa: E402
from time_router.runtime import (  # noqa: E402
    create_run_dir,
    write_evaluation_summary,
    write_prediction_rows_csv,
    write_run_metadata,
    write_run_status,
    write_sample_manifest_ref,
    write_split_summary,
)


MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
FEATURE_COLUMNS = ("trend_strength", "seasonality_strength", "recent_volatility")
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


def build_tiny_manifest(*, config_name: str) -> SampleManifest:
    """
    函数功能：
        构造 P12 small entrypoint 使用的 tiny SampleManifest。

    输入：
        config_name: 写入每行 manifest 的配置名，默认与 Stage 1 小切片一致。

    输出：
        `SampleManifest`；rows 原始顺序即后续 Expert/Feature/Evaluator 顺序。
    """
    rows = (
        SampleManifestRow(
            sample_key=f"{config_name}__vali__TINY_DATA__item10__ch0__win0",
            split="vali",
            config_name=config_name,
            dataset_name="TINY_DATA",
            item_id=10,
            channel_id=0,
            window_index=0,
            seq_len=96,
            pred_len=48,
            extra={"fixture_role": "validation", "source": "p12_small_entrypoint"},
        ),
        SampleManifestRow(
            sample_key=f"{config_name}__test__TINY_DATA__item10__ch0__win1",
            split="test",
            config_name=config_name,
            dataset_name="TINY_DATA",
            item_id=10,
            channel_id=0,
            window_index=1,
            seq_len=96,
            pred_len=48,
            extra={"fixture_role": "evaluation", "source": "p12_small_entrypoint"},
        ),
        SampleManifestRow(
            sample_key=f"{config_name}__test__TINY_DATA__item11__ch1__win2",
            split="test",
            config_name=config_name,
            dataset_name="TINY_DATA",
            item_id=11,
            channel_id=1,
            window_index=2,
            seq_len=96,
            pred_len=48,
            extra={"fixture_role": "evaluation", "source": "p12_small_entrypoint"},
        ),
    )
    manifest = SampleManifest(rows=rows, extra={"fixture": "p12_small_canonical_manifest"})
    manifest.validate_unique_sample_keys()
    return manifest


class TinyExpertProvider:
    """
    类功能：
        small entrypoint 内存 ExpertProvider，用固定 fixture 构造 ExpertBatch。

    输入：
        `load_batch(sample_keys)` 接收 Runtime 从 SampleManifest 取出的显式顺序。

    输出：
        ExpertBatch；sample_keys/model_columns/y_pred/y_true 均按输入顺序对齐。

    关键约束：
        不接收 run_dir，不读取 prediction cache，不访问 `/data2`，不写运行产物。
    """

    provider_name = "TinyExpertProvider"

    def __init__(self, sample_keys: Sequence[str]) -> None:
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        self.received_run_dir = False
        self._y_true_by_key = {
            ordered_keys[0]: np.asarray([[1.0], [1.2]], dtype=np.float64),
            ordered_keys[1]: np.asarray([[2.0], [1.8]], dtype=np.float64),
            ordered_keys[2]: np.asarray([[0.5], [0.7]], dtype=np.float64),
        }
        self._y_pred_by_key = {
            ordered_keys[0]: np.asarray([[[1.1], [1.3]], [[0.9], [1.1]], [[1.4], [1.5]]], dtype=np.float64),
            ordered_keys[1]: np.asarray([[[2.2], [1.9]], [[1.7], [1.6]], [[2.1], [1.7]]], dtype=np.float64),
            ordered_keys[2]: np.asarray([[[0.6], [0.8]], [[0.4], [0.6]], [[0.9], [1.0]]], dtype=np.float64),
        }

    def load_batch(self, sample_keys: Sequence[str]) -> ExpertBatch:
        """函数功能：按调用方传入的 sample_key 顺序返回 tiny ExpertBatch。"""
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        y_true = np.stack([self._y_true_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        y_pred = np.stack([self._y_pred_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        return ExpertBatch(
            sample_keys=ordered_keys,
            model_columns=MODEL_COLUMNS,
            y_pred=y_pred,
            y_true=y_true,
            row_index_metadata={"source": "memory://p12_tiny_expert_fixture"},
            extra={"provider_name": self.provider_name, "fixture": "p12_tiny_expert_batch"},
        )


def write_tiny_feature_csv(feature_csv_path: Path, *, sample_keys: Sequence[str]) -> None:
    """
    函数功能：
        写出 tiny feature CSV；行顺序刻意与 manifest 不同，用于证明 provider 保序。
    """
    ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
    matrix_by_key = {sample_key: FEATURE_MATRIX[index] for index, sample_key in enumerate(ordered_keys)}
    csv_order = (ordered_keys[2], ordered_keys[0], ordered_keys[1])
    feature_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_key", *FEATURE_COLUMNS))
        writer.writeheader()
        for sample_key in csv_order:
            row = {"sample_key": sample_key}
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = f"{matrix_by_key[sample_key][feature_index]:.8f}"
            writer.writerow(row)


def build_evaluation_summary_payload(result: EvaluationInputAdapterResult) -> dict[str, object]:
    """
    函数功能：
        将 Evaluator 内存 summary 包装为 Runtime writer 要求的最小 schema。
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
        将 Evaluator 逐样本 rows 补齐 split、y_true 和 hard-fusion y_pred 后写出。
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


def now_iso() -> str:
    """函数功能：生成带 UTC 时区的轻量 artifact 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def assert_not_data2(path: Path, *, role: str) -> None:
    """函数功能：阻止 P12 small entrypoint 意外访问或写入 `/data2`。"""
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise ValueError(f"P12 small entrypoint 不应使用 /data2 {role}：{resolved}")


def run_entrypoint(args: argparse.Namespace) -> Path:
    """
    函数功能：
        执行 tiny canonical dataflow，并只在 Runtime writer 阶段写出 run_dir。

    输入：
        argparse 解析后的参数对象。

    输出：
        已完成写出的 canonical run_dir。
    """
    output_root = Path(args.output_root)
    assert_not_data2(output_root, role="output_root")
    if args.feature_source is not None:
        assert_not_data2(Path(args.feature_source), role="feature_source")

    manifest = build_tiny_manifest(config_name=str(args.config_name))
    ordered_sample_keys = manifest.sample_keys()
    expert_provider = TinyExpertProvider(ordered_sample_keys)
    expert_batch = expert_provider.load_batch(ordered_sample_keys)

    with tempfile.TemporaryDirectory(prefix="stage1_p12_canonical_small_") as temp_dir:
        temp_root = Path(temp_dir)
        assert_not_data2(temp_root, role="tempfile")
        feature_csv_path = Path(args.feature_source) if args.feature_source else temp_root / "p12_timefuse_features.csv"
        if args.feature_source is None:
            write_tiny_feature_csv(feature_csv_path, sample_keys=ordered_sample_keys)

        feature_provider = TimeFuseFeatureCacheProvider(
            feature_csv_path=feature_csv_path,
            feature_columns=FEATURE_COLUMNS,
            feature_schema_name="p12_tiny_timefuse_feature_schema_v1",
        )
        feature_batch = feature_provider.load_batch(expert_batch.sample_keys)
        head = TimeFuseLinearSoftmaxHead(weight=HEAD_WEIGHT, bias=HEAD_BIAS)
        router_output = head.predict(feature_batch, expert_batch.model_columns)
        result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)

        if args.strict:
            # strict 模式只校验内存协议边界，不把 run_dir 传给 provider/head/evaluator。
            if expert_batch.sample_keys != ordered_sample_keys:
                raise AssertionError("ExpertBatch 未保持 manifest sample_key 顺序")
            if feature_batch.sample_keys != ordered_sample_keys:
                raise AssertionError("FeatureBatch 未保持 manifest sample_key 顺序")
            if router_output.sample_keys != ordered_sample_keys:
                raise AssertionError("RouterOutput 未保持 manifest sample_key 顺序")
            if result.evaluation_input.sample_keys != ordered_sample_keys:
                raise AssertionError("EvaluationInput 未保持 manifest sample_key 顺序")
            if expert_provider.received_run_dir:
                raise AssertionError("ExpertProvider 不应接收 run_dir")

        created_at = now_iso()
        run_dir = create_run_dir(output_root, run_name=str(args.run_name))
        manifest_ref = {
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "reference_type": "inline_fixture",
            "path": "memory://p12_small_canonical_manifest",
            "checksum": "not_applicable_tiny_inline_fixture",
            "checksum_algorithm": "none",
            "row_count": len(manifest.rows),
            "ordered_sample_keys_policy": "manifest_row_order",
            "created_at": created_at,
        }
        split_summary = {
            "split_summary_schema_version": "stage1_split_summary_v1",
            "split_strategy_name": "p12_tiny_vali_test_fixture",
            "config_name": str(args.config_name),
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
            "created_at": created_at,
        }
        metadata = {
            "run_artifact_schema_version": "stage1_run_artifact_v1",
            "protocol_version": "stage1_canonical_small_entrypoint_v1",
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "evaluation_schema_version": "stage1_evaluation_summary_v1",
            "config_name": str(args.config_name),
            "branch_name": str(args.branch_name),
            "created_at": created_at,
            "inputs": {
                "sample_manifest": "inputs/sample_manifest_ref.json",
                "split_summary": "inputs/split_summary.json",
                "feature_source": str(feature_csv_path),
            },
        }
        status = {
            "status": "completed",
            "current_stage": "canonical_small_entrypoint",
            "updated_at": created_at,
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

    print(f"run_dir: {run_dir}")
    print(
        json.dumps(
            {
                "status": "completed",
                "sample_count": len(ordered_sample_keys),
                "splits": manifest.split_counts(),
                "hard_mae": evaluation_summary["metrics"]["hard_mae"],
                "raw_soft_mae": evaluation_summary["metrics"]["raw_soft_mae"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """函数功能：解析 P12 small canonical entrypoint 的最小 CLI 参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 P12 small canonical entrypoint thin slice.")
    parser.add_argument("--output-root", required=True, help="canonical run_dir 的父目录；P12 禁止使用 /data2。")
    parser.add_argument("--run-name", required=True, help="Runtime writer 在 output_root 下创建的单层 run 目录名。")
    parser.add_argument("--config-name", default="96_48_S", help="写入 manifest 和 metadata 的 config 名。")
    parser.add_argument("--branch-name", default="canonical_small_smoke", help="写入 run_metadata 的 branch 名。")
    parser.add_argument("--feature-source", default=None, help="可选 tiny feature CSV；默认在 tempfile 内生成。")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启，校验 sample_key 保序和 Provider/run_dir 边界。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """函数功能：CLI 主入口；异常交给调用方/subprocess 暴露非零返回码。"""
    args = parse_args(argv)
    run_entrypoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
