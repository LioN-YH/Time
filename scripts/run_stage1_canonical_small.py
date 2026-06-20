#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P12 small canonical entrypoint thin slice。

输入：
    命令行显式传入 `--output-root` 和 `--run-name`；可选指定 config、branch、
    tiny SampleManifest CSV/JSONL、expert JSON、feature CSV 和 strict 校验开关。

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
from typing import Any, Mapping, Sequence

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


def _coerce_optional_int(value: object, *, column: str, sample_key: str) -> int | None:
    """
    函数功能：
        将 fixture CSV/JSONL 中可选整数列转换为 int 或 None。

    关键约束：
        tiny fixture 必须显式给出可审计字段；空字符串只允许出现在可选 seq/pred length。
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_manifest fixture 字段不是整数：sample_key={sample_key}, column={column}") from exc


def _coerce_required_int(value: object, *, column: str, sample_key: str) -> int:
    """函数功能：读取 manifest fixture 必需整数列，失败时带上 sample_key 定位。"""
    if value is None or value == "":
        raise ValueError(f"sample_manifest fixture 缺少必需整数列：sample_key={sample_key}, column={column}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_manifest fixture 字段不是整数：sample_key={sample_key}, column={column}") from exc


def _row_to_manifest_row(row: Mapping[str, object]) -> SampleManifestRow:
    """
    函数功能：
        将 CSV/JSONL 的单行 fixture 转成 canonical SampleManifestRow。
    """
    required_columns = (
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
    missing = [column for column in required_columns if column not in row]
    if missing:
        raise ValueError(f"sample_manifest fixture 缺少必需列：{missing}")

    sample_key = str(row["sample_key"])
    if not sample_key:
        raise ValueError("sample_manifest fixture 存在空 sample_key")
    return SampleManifestRow(
        sample_key=sample_key,
        split=str(row["split"]),
        config_name=str(row["config_name"]),
        dataset_name=str(row["dataset_name"]),
        item_id=_coerce_required_int(row["item_id"], column="item_id", sample_key=sample_key),
        channel_id=_coerce_required_int(row["channel_id"], column="channel_id", sample_key=sample_key),
        window_index=_coerce_required_int(row["window_index"], column="window_index", sample_key=sample_key),
        seq_len=_coerce_optional_int(row["seq_len"], column="seq_len", sample_key=sample_key),
        pred_len=_coerce_optional_int(row["pred_len"], column="pred_len", sample_key=sample_key),
        extra={"fixture_role": "explicit_fixture", "source": "p12b_small_fixture_file"},
    )


def load_sample_manifest_fixture(path: Path) -> SampleManifest:
    """
    函数功能：
        读取 P12b tiny SampleManifest fixture，支持 CSV 与 JSONL。

    输入：
        path: 调用方显式传入的小规模 fixture 路径；不得指向 `/data2`。

    输出：
        `SampleManifest`，rows 顺序完全沿用文件行顺序，作为 ordered sample_keys 来源。
    """
    assert_not_data2(path, role="sample_manifest")
    suffix = path.suffix.lower()
    rows: list[SampleManifestRow] = []
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("sample_manifest CSV 缺少表头")
            rows = [_row_to_manifest_row(row) for row in reader]
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"sample_manifest JSONL 第 {line_number} 行不是 object")
                rows.append(_row_to_manifest_row(payload))
    else:
        raise ValueError(f"sample_manifest fixture 只支持 .csv 或 .jsonl：{path}")
    if not rows:
        raise ValueError("sample_manifest fixture 没有任何样本行")
    manifest = SampleManifest(rows=tuple(rows), extra={"fixture": "p12b_small_canonical_manifest_file", "path": str(path)})
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


class JsonExpertFixtureProvider:
    """
    类功能：
        从显式 tiny expert JSON fixture 读取 y_true/y_pred，并按 manifest sample_keys 保序组装。

    关键约束：
        只支持小数组 JSON；不读取 prediction cache，不引入 parquet/SQLite/packed npy，不接收 run_dir。
    """

    provider_name = "JsonExpertFixtureProvider"

    def __init__(self, expert_fixture_path: Path) -> None:
        assert_not_data2(expert_fixture_path, role="expert_fixture")
        self.expert_fixture_path = Path(expert_fixture_path)
        self.received_run_dir = False
        with self.expert_fixture_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("expert fixture JSON 必须是 object")
        model_columns = payload.get("model_columns")
        samples = payload.get("samples")
        if not isinstance(model_columns, list) or not model_columns:
            raise ValueError("expert fixture JSON 需要非空 model_columns list")
        if not isinstance(samples, list) or not samples:
            raise ValueError("expert fixture JSON 需要非空 samples list")
        self.model_columns = tuple(str(model_name) for model_name in model_columns)
        if len(self.model_columns) != len(set(self.model_columns)):
            raise ValueError("expert fixture JSON model_columns 不能重复")

        self._y_true_by_key: dict[str, np.ndarray] = {}
        self._y_pred_by_key: dict[str, np.ndarray] = {}
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise ValueError(f"expert fixture samples[{index}] 不是 object")
            sample_key = str(sample.get("sample_key", ""))
            if not sample_key:
                raise ValueError(f"expert fixture samples[{index}] 缺少 sample_key")
            if sample_key in self._y_true_by_key:
                raise ValueError(f"expert fixture 存在重复 sample_key：{sample_key}")
            if "y_true" not in sample or "y_pred" not in sample:
                raise ValueError(f"expert fixture sample 缺少 y_true/y_pred：{sample_key}")
            y_true = np.asarray(sample["y_true"], dtype=np.float64)
            y_pred = np.asarray(sample["y_pred"], dtype=np.float64)
            if y_pred.shape[0] != len(self.model_columns):
                raise ValueError(
                    "expert fixture y_pred 第一维必须与 model_columns 对齐："
                    f"sample_key={sample_key}, expected={len(self.model_columns)}, actual={y_pred.shape}"
                )
            if y_pred.shape[1:] != y_true.shape:
                raise ValueError(
                    "expert fixture y_pred 后续维度必须与 y_true 对齐："
                    f"sample_key={sample_key}, y_pred={y_pred.shape}, y_true={y_true.shape}"
                )
            self._y_true_by_key[sample_key] = y_true
            self._y_pred_by_key[sample_key] = y_pred

    def load_batch(self, sample_keys: Sequence[str]) -> ExpertBatch:
        """函数功能：按 manifest sample_key 顺序组装显式 fixture ExpertBatch。"""
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        missing_keys = [sample_key for sample_key in ordered_keys if sample_key not in self._y_true_by_key]
        if missing_keys:
            raise KeyError(f"expert fixture 缺少 manifest sample_key：{missing_keys}")
        y_true = np.stack([self._y_true_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        y_pred = np.stack([self._y_pred_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        return ExpertBatch(
            sample_keys=ordered_keys,
            model_columns=self.model_columns,
            y_pred=y_pred,
            y_true=y_true,
            row_index_metadata={"source": str(self.expert_fixture_path)},
            extra={"provider_name": self.provider_name, "fixture": "p12b_json_expert_fixture"},
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


def build_input_reference(*, path: Path | None, inline_reference: str) -> dict[str, Any]:
    """
    函数功能：
        为 run_metadata.inputs 记录 tiny fixture 来源摘要，避免只留下运行期临时路径。
    """
    if path is None:
        return {"reference_type": "inline_fixture", "path": inline_reference}
    return {"reference_type": "file", "path": str(path)}


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
    if args.sample_manifest is not None:
        assert_not_data2(Path(args.sample_manifest), role="sample_manifest")
    if args.expert_fixture is not None:
        assert_not_data2(Path(args.expert_fixture), role="expert_fixture")
    if args.feature_source is not None:
        assert_not_data2(Path(args.feature_source), role="feature_source")

    manifest = (
        load_sample_manifest_fixture(Path(args.sample_manifest))
        if args.sample_manifest is not None
        else build_tiny_manifest(config_name=str(args.config_name))
    )
    ordered_sample_keys = manifest.sample_keys()
    expert_provider = (
        JsonExpertFixtureProvider(Path(args.expert_fixture))
        if args.expert_fixture is not None
        else TinyExpertProvider(ordered_sample_keys)
    )
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
        sample_manifest_path = Path(args.sample_manifest) if args.sample_manifest is not None else None
        expert_fixture_path = Path(args.expert_fixture) if args.expert_fixture is not None else None
        feature_source_path = Path(args.feature_source) if args.feature_source is not None else None
        manifest_ref = {
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "reference_type": "file" if sample_manifest_path is not None else "inline_fixture",
            "path": str(sample_manifest_path) if sample_manifest_path is not None else "memory://p12_small_canonical_manifest",
            "checksum": "not_applicable_tiny_fixture",
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
                "sample_manifest": {
                    **build_input_reference(
                        path=sample_manifest_path,
                        inline_reference="memory://p12_small_canonical_manifest",
                    ),
                    "artifact_path": "inputs/sample_manifest_ref.json",
                },
                "sample_manifest_ref_artifact": "inputs/sample_manifest_ref.json",
                "split_summary": "inputs/split_summary.json",
                "expert_fixture": build_input_reference(
                    path=expert_fixture_path,
                    inline_reference="memory://p12_tiny_expert_fixture",
                ),
                "feature_source": build_input_reference(
                    path=feature_source_path,
                    inline_reference="generated_tempfile://p12_timefuse_features.csv",
                ),
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
    parser.add_argument("--sample-manifest", default=None, help="可选 tiny SampleManifest CSV/JSONL；默认使用内联 fixture。")
    parser.add_argument("--expert-fixture", default=None, help="可选 tiny expert JSON fixture；默认使用内联 fixture。")
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
