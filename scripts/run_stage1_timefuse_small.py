#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P15b TimeFuse-specific small canonical entrypoint thin slice。

输入：
    命令行显式传入 small SampleManifest CSV、17 维 TimeFuse feature CSV、
    small expert prediction JSON 和 output dir；默认使用仓库内 P13b/P13e fixture。

输出：
    在 `output_dir/run_id/` 下写出 canonical run_dir：run metadata/status、
    inputs 引用、evaluation summary 和 prediction rows。

关键约束：
    本脚本只做 TimeFuse-style small fixture / real-derived small input 编排；
    不访问 `/data2`，不启动训练、pressure 或 full-scale，不读取旧 full-scale artifact，
    不修改 generic small CLI，不接 Bash launcher，也不把 run_dir 传入 provider。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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


DEFAULT_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"
DEFAULT_SAMPLE_MANIFEST_CSV = DEFAULT_FIXTURE_ROOT / "stage1_real_derived_small" / "sample_manifest.csv"
DEFAULT_EXPERT_PREDICTIONS_JSON = DEFAULT_FIXTURE_ROOT / "stage1_real_derived_small" / "expert_predictions.json"
DEFAULT_FEATURES_CSV = DEFAULT_FIXTURE_ROOT / "stage1_timefuse_17dim_small" / "features_17d.csv"
DEFAULT_RUN_ID = "stage1_timefuse_small"
TIMEFUSE_FEATURE_COLUMNS = (
    "mean",
    "std",
    "min",
    "max",
    "skewness",
    "kurtosis",
    "autocorrelation_mean",
    "stationarity",
    "rate_of_change_mean",
    "rate_of_change_std",
    "autoreg_coef_mean",
    "residual_std_mean",
    "frequency_mean",
    "frequency_peak",
    "spectral_entropy",
    "spectral_skewness",
    "spectral_kurtosis",
)


def assert_not_data2(path: Path, *, role: str) -> None:
    """函数功能：阻止 P15b small entrypoint 读写 `/data2` 路径。"""
    resolved = str(path.resolve())
    if resolved == "/data2" or resolved.startswith("/data2/"):
        raise ValueError(f"P15b TimeFuse small entrypoint 不应使用 /data2 {role}：{resolved}")


def now_iso() -> str:
    """函数功能：生成 Runtime artifact 使用的 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """函数功能：计算 small input 文件 sha256，供 input reference 留痕。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_int(row: Mapping[str, str], column: str) -> int:
    """函数功能：读取 manifest 必需整数字段，失败时携带 sample_key 定位。"""
    sample_key = str(row.get("sample_key", ""))
    value = row.get(column, "")
    if value == "":
        raise ValueError(f"sample_manifest 缺少必需整数列：sample_key={sample_key}, column={column}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"sample_manifest 整数字段非法：sample_key={sample_key}, column={column}") from exc


def _optional_int(row: Mapping[str, str], column: str) -> int | None:
    """函数功能：读取 manifest 可选整数字段，空字符串转为 None。"""
    value = row.get(column, "")
    if value == "":
        return None
    return _required_int(row, column)


def load_sample_manifest_csv(path: Path, *, split_name: str) -> SampleManifest:
    """
    函数功能：
        读取 P13b real-derived small SampleManifest，并按 split 保留 ordered sample_keys。

    关键约束：
        split 过滤不重排文件行顺序；`split_name=all` 表示使用全部 small fixture。
    """
    assert_not_data2(path, role="sample_manifest_csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("sample_manifest CSV 缺少表头")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("sample_manifest CSV 没有样本行")

    rows: list[SampleManifestRow] = []
    for source_row in source_rows:
        row_split = str(source_row["split"])
        if split_name != "all" and row_split != split_name:
            continue
        sample_key = str(source_row["sample_key"])
        rows.append(
            SampleManifestRow(
                sample_key=sample_key,
                split=row_split,
                config_name=str(source_row["config_name"]),
                dataset_name=str(source_row["dataset_name"]),
                item_id=_required_int(source_row, "item_id"),
                channel_id=_required_int(source_row, "channel_id"),
                window_index=_required_int(source_row, "window_index"),
                seq_len=_optional_int(source_row, "seq_len"),
                pred_len=_optional_int(source_row, "pred_len"),
                extra={"source": "p13b_real_derived_small", "split_filter": split_name},
            )
        )
    if not rows:
        raise ValueError(f"sample_manifest CSV 在 split={split_name!r} 下没有样本")
    manifest = SampleManifest(
        rows=tuple(rows),
        extra={"source_path": str(path), "split_name": split_name, "source_row_count": len(source_rows)},
    )
    manifest.validate_unique_sample_keys()
    return manifest


class JsonExpertSmallProvider:
    """
    类功能：
        从 small expert prediction JSON 构造 TimeFuse small entrypoint 的 ExpertBatch。

    输入：
        expert_predictions_json: P13b small JSON，包含 model_columns 和 samples。

    输出：
        `load_batch(sample_keys)` 按调用方 ordered sample_keys 返回 ExpertBatch。

    关键约束：
        该 provider 只读取显式 small JSON，不读取 full-scale cache，不接收 run_dir。
    """

    provider_name = "JsonExpertSmallProvider"

    def __init__(self, expert_predictions_json: Path) -> None:
        assert_not_data2(expert_predictions_json, role="expert_predictions_json")
        self.expert_predictions_json = Path(expert_predictions_json)
        self.received_run_dir = False
        with self.expert_predictions_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("expert predictions JSON 必须是 object")
        model_columns = payload.get("model_columns")
        samples = payload.get("samples")
        if not isinstance(model_columns, list) or not model_columns:
            raise ValueError("expert predictions JSON 需要非空 model_columns")
        if not isinstance(samples, list) or not samples:
            raise ValueError("expert predictions JSON 需要非空 samples")
        self.model_columns = tuple(str(model_name) for model_name in model_columns)
        if len(self.model_columns) != len(set(self.model_columns)):
            raise ValueError(f"expert predictions JSON model_columns 重复：{self.model_columns}")

        self._y_true_by_key: dict[str, np.ndarray] = {}
        self._y_pred_by_key: dict[str, np.ndarray] = {}
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise ValueError(f"expert predictions samples[{index}] 不是 object")
            sample_key = str(sample.get("sample_key", ""))
            if not sample_key:
                raise ValueError(f"expert predictions samples[{index}] 缺少 sample_key")
            if sample_key in self._y_true_by_key:
                raise ValueError(f"expert predictions JSON 存在重复 sample_key：{sample_key}")
            y_true = np.asarray(sample.get("y_true"), dtype=np.float64)
            y_pred = np.asarray(sample.get("y_pred"), dtype=np.float64)
            if y_pred.shape[0] != len(self.model_columns):
                raise ValueError(f"y_pred 专家维与 model_columns 不一致：sample_key={sample_key}, shape={y_pred.shape}")
            if y_pred.shape[1:] != y_true.shape:
                raise ValueError(
                    "y_pred 后续维度必须与 y_true 一致："
                    f"sample_key={sample_key}, y_pred={y_pred.shape}, y_true={y_true.shape}"
                )
            self._y_true_by_key[sample_key] = y_true
            self._y_pred_by_key[sample_key] = y_pred

    def load_batch(self, sample_keys: Sequence[str]) -> ExpertBatch:
        """函数功能：按 manifest sample_key 顺序输出 ExpertBatch。"""
        ordered_keys = tuple(str(sample_key) for sample_key in sample_keys)
        missing = [sample_key for sample_key in ordered_keys if sample_key not in self._y_true_by_key]
        if missing:
            raise KeyError(f"expert predictions JSON 缺少 sample_key：{missing}")
        y_true = np.stack([self._y_true_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        y_pred = np.stack([self._y_pred_by_key[sample_key] for sample_key in ordered_keys], axis=0)
        return ExpertBatch(
            sample_keys=ordered_keys,
            model_columns=self.model_columns,
            y_pred=y_pred,
            y_true=y_true,
            row_index_metadata={"source": str(self.expert_predictions_json), "storage": "small_json_fixture"},
            extra={"provider_name": self.provider_name, "fixture": "p15b_timefuse_small_expert_json"},
        )


def build_head(feature_dim: int, num_experts: int) -> TimeFuseLinearSoftmaxHead:
    """
    函数功能：
        构造 deterministic TimeFuse linear-softmax head。

    关键约束：
        P15b 只做 fixed-weight small rehearsal，不训练、不保存 checkpoint。
    """
    if feature_dim != len(TIMEFUSE_FEATURE_COLUMNS):
        raise ValueError(f"P15b TimeFuse small head 期望 17 维 feature，actual={feature_dim}")
    weight = (np.arange(feature_dim * num_experts, dtype=np.float64).reshape(feature_dim, num_experts) - 17.0) / 250.0
    bias = np.linspace(0.03, -0.03, num_experts, dtype=np.float64)
    return TimeFuseLinearSoftmaxHead(weight=weight, bias=bias)


def build_evaluation_summary_payload(result: EvaluationInputAdapterResult) -> dict[str, object]:
    """函数功能：把 evaluator 内存 summary 包装成 Runtime writer 当前支持的最小 schema。"""
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
    """函数功能：为 Runtime writer 补齐 split、hard-fusion y_pred 和聚合 y_true。"""
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


def assert_strict_contract(
    *,
    manifest: SampleManifest,
    expert_batch: ExpertBatch,
    feature_batch: Any,
    router_output: Any,
    result: EvaluationInputAdapterResult,
    expert_provider: JsonExpertSmallProvider,
) -> None:
    """函数功能：集中校验 P15b small canonical chain 的保序、shape 和边界。"""
    ordered_sample_keys = manifest.sample_keys()
    if expert_batch.sample_keys != ordered_sample_keys:
        raise AssertionError("ExpertBatch 未保持 manifest sample_key 顺序")
    if feature_batch.sample_keys != ordered_sample_keys:
        raise AssertionError("FeatureBatch 未保持 manifest sample_key 顺序")
    if router_output.sample_keys != ordered_sample_keys:
        raise AssertionError("RouterOutput 未保持 manifest sample_key 顺序")
    if result.evaluation_input.sample_keys != ordered_sample_keys:
        raise AssertionError("EvaluationInput 未保持 manifest sample_key 顺序")
    if tuple(feature_batch.features.shape) != (len(ordered_sample_keys), len(TIMEFUSE_FEATURE_COLUMNS)):
        raise AssertionError(f"FeatureBatch shape 不是 17 维：{feature_batch.features.shape}")
    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError("RouterOutput model_columns 未与 ExpertBatch 对齐")
    if tuple(router_output.weights.shape) != (len(ordered_sample_keys), len(expert_batch.model_columns)):
        raise AssertionError(f"weights shape 异常：{router_output.weights.shape}")
    if not np.isfinite(router_output.weights).all() or not np.isfinite(router_output.logits).all():
        raise AssertionError("router logits/weights 必须全为有限值")
    np.testing.assert_allclose(np.sum(router_output.weights, axis=1), np.ones(len(ordered_sample_keys)), rtol=0.0, atol=1e-9)
    if result.evaluation_input.y_pred is not expert_batch.y_pred or result.evaluation_input.y_true is not expert_batch.y_true:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch arrays")
    if result.evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")
    if expert_provider.received_run_dir:
        raise AssertionError("ExpertProvider 不应接收 run_dir")


def write_runtime_artifacts(
    *,
    args: argparse.Namespace,
    manifest: SampleManifest,
    result: EvaluationInputAdapterResult,
    run_dir: Path,
    created_at: str,
) -> None:
    """函数功能：Runtime 层统一写出 canonical run_dir artifact。"""
    sample_manifest_csv = Path(args.sample_manifest_csv)
    features_csv = Path(args.features_csv)
    expert_predictions_json = Path(args.expert_predictions_json)
    manifest_ref = {
        "sample_manifest_schema_version": "stage1_sample_manifest_v1",
        "reference_type": "file",
        "path": str(sample_manifest_csv),
        "checksum": sha256_file(sample_manifest_csv),
        "checksum_algorithm": "sha256",
        "row_count": len(manifest.rows),
        "ordered_sample_keys_policy": "manifest_row_order_after_split_filter",
        "created_at": created_at,
    }
    split_summary = {
        "split_summary_schema_version": "stage1_split_summary_v1",
        "split_strategy_name": "p15b_timefuse_small_split_filter",
        "config_name": str(args.config_name),
        "split_names": list(manifest.split_counts().keys()),
        "sample_count_by_split": manifest.split_counts(),
        "unique_sample_key_count": len(manifest.sample_keys()),
        "duplicate_sample_key_count": 0,
        "split_overlap_check": {
            "default_policy": "single_split_or_all_fixture",
            "allowed_overlap": False,
            "overlap_sample_key_count": 0,
            "overlap_examples": [],
        },
        "ordered_sample_keys_policy": "manifest_row_order_after_split_filter",
        "source_manifest_reference": manifest_ref,
        "created_at": created_at,
    }
    metadata = {
        "run_artifact_schema_version": "stage1_run_artifact_v1",
        "protocol_version": "stage1_timefuse_small_entrypoint_v1",
        "sample_manifest_schema_version": "stage1_sample_manifest_v1",
        "evaluation_schema_version": "stage1_evaluation_summary_v1",
        "config_name": str(args.config_name),
        "branch_name": "timefuse_fusor_small",
        "created_at": created_at,
        "inputs": {
            "sample_manifest": {"reference_type": "file", "path": str(sample_manifest_csv), "artifact_path": "inputs/sample_manifest_ref.json"},
            "split_summary": "inputs/split_summary.json",
            "features_csv": {
                "reference_type": "file",
                "path": str(features_csv),
                "feature_dim": len(TIMEFUSE_FEATURE_COLUMNS),
                "checksum": sha256_file(features_csv),
                "checksum_algorithm": "sha256",
            },
            "expert_predictions_json": {
                "reference_type": "file",
                "path": str(expert_predictions_json),
                "checksum": sha256_file(expert_predictions_json),
                "checksum_algorithm": "sha256",
            },
        },
        "timefuse_fusor": {
            "feature_schema_name": "timefuse_single_variable_meta_v1",
            "feature_columns": list(TIMEFUSE_FEATURE_COLUMNS),
            "head": "TimeFuseLinearSoftmaxHead fixed deterministic smoke weights",
            "training": "not_started_p15b_small_rehearsal_only",
        },
    }
    status = {
        "status": "completed",
        "current_stage": "timefuse_small_entrypoint",
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
    log_path = run_dir / "logs" / "timefuse_small_entrypoint.log"
    log_path.write_text(
        "\n".join(
            [
                f"created_at={created_at}",
                "stage=P15b TimeFuse-specific small canonical entrypoint",
                f"sample_count={len(manifest.sample_keys())}",
                f"split_counts={manifest.split_counts()}",
                f"hard_mae={evaluation_summary['metrics']['hard_mae']}",
                f"raw_soft_mae={evaluation_summary['metrics']['raw_soft_mae']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_entrypoint(args: argparse.Namespace) -> Path:
    """函数功能：执行 P15b TimeFuse small canonical dataflow，并返回 run_dir。"""
    output_dir = Path(args.output_dir)
    sample_manifest_csv = Path(args.sample_manifest_csv)
    features_csv = Path(args.features_csv)
    expert_predictions_json = Path(args.expert_predictions_json)
    for role, path in (
        ("output_dir", output_dir),
        ("sample_manifest_csv", sample_manifest_csv),
        ("features_csv", features_csv),
        ("expert_predictions_json", expert_predictions_json),
    ):
        assert_not_data2(path, role=role)
    for role, path in (
        ("sample_manifest_csv", sample_manifest_csv),
        ("features_csv", features_csv),
        ("expert_predictions_json", expert_predictions_json),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} 不存在：{path}")

    manifest = load_sample_manifest_csv(sample_manifest_csv, split_name=str(args.split_name))
    ordered_sample_keys = manifest.sample_keys()
    expert_provider = JsonExpertSmallProvider(expert_predictions_json)
    expert_batch = expert_provider.load_batch(ordered_sample_keys)
    feature_provider = TimeFuseFeatureCacheProvider(
        feature_csv_path=features_csv,
        feature_columns=TIMEFUSE_FEATURE_COLUMNS,
        feature_schema_name="timefuse_single_variable_meta_v1",
    )
    feature_batch = feature_provider.load_batch(expert_batch.sample_keys)
    head = build_head(feature_dim=int(feature_batch.features.shape[1]), num_experts=len(expert_batch.model_columns))
    router_output = head.predict(feature_batch, expert_batch.model_columns)
    result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)

    if args.strict:
        assert_strict_contract(
            manifest=manifest,
            expert_batch=expert_batch,
            feature_batch=feature_batch,
            router_output=router_output,
            result=result,
            expert_provider=expert_provider,
        )

    created_at = now_iso()
    run_dir = create_run_dir(output_dir, run_name=str(args.run_id))
    write_runtime_artifacts(args=args, manifest=manifest, result=result, run_dir=run_dir, created_at=created_at)

    summary = build_evaluation_summary_payload(result)
    print(f"run_dir: {run_dir}")
    print(
        json.dumps(
            {
                "status": "completed",
                "sample_count": len(ordered_sample_keys),
                "split_counts": manifest.split_counts(),
                "feature_dim": int(feature_batch.features.shape[1]),
                "num_experts": len(expert_batch.model_columns),
                "hard_mae": summary["metrics"]["hard_mae"],
                "raw_soft_mae": summary["metrics"]["raw_soft_mae"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """函数功能：解析 P15b TimeFuse-specific small CLI 参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 P15b TimeFuse-specific small canonical entrypoint.")
    parser.add_argument("--sample-manifest-csv", default=str(DEFAULT_SAMPLE_MANIFEST_CSV), help="small SampleManifest CSV。")
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV), help="small 17 维 TimeFuse feature CSV。")
    parser.add_argument("--expert-predictions-json", default=str(DEFAULT_EXPERT_PREDICTIONS_JSON), help="small expert prediction JSON。")
    parser.add_argument("--output-dir", required=True, help="canonical run_dir 的父目录；P15b 禁止使用 /data2。")
    parser.add_argument("--split-name", default="test", help="使用的 split；默认 test，传 all 使用全部 small fixture。")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="output-dir 下创建的单层 run 目录名。")
    parser.add_argument("--config-name", default="96_48_S", help="写入 run metadata 的 config 名。")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启，校验 sample_key 保序、17 维 feature、weights 和 provider/run_dir 边界。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """函数功能：CLI 主入口。"""
    args = parse_args(argv)
    run_entrypoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
