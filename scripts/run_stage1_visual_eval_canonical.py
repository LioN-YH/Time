#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P17a Visual canonical evaluation entrypoint thin slice。

输入：
    调用方必须显式传入 SampleManifest CSV、expert prediction JSON、precomputed
    Visual feature CSV、router checkpoint payload、output dir 和 run id；可选传入
    loaded scaler state JSON。

输出：
    在 `output_dir/run_id/` 下创建 canonical run_dir，并写出 run metadata、
    status、inputs、evaluation summary、prediction rows 和最小日志。

关键约束：
    本脚本只做 Visual evaluation，不启动训练、不启动 ViT、不迁移
    `train_visual_router_online_streaming.py`，也不修改 P15c/P16j Visual small
    entrypoint 的默认行为。checkpoint/scaler path 只属于 Runtime/entrypoint，
    不进入 FeatureProvider 或 RouterHead adapter interface。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import LoadedFeatureScaler, VisualPrecomputedFeatureProvider  # noqa: E402
from time_router.models import LoadedTorchMLPRouterHeadAdapter  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput, SampleManifest, SampleManifestRow  # noqa: E402
from time_router.runtime import (  # noqa: E402
    create_run_dir,
    extract_router_state_dict,
    load_checkpoint_payload,
    load_router_state_dict,
    write_evaluation_summary,
    write_prediction_rows_csv,
    write_run_metadata,
    write_run_status,
    write_sample_manifest_ref,
    write_split_summary,
)


LEGACY_MODULE_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
ENTRYPOINT_NAME = "visual_eval_canonical"
PROTOCOL_VERSION = "stage1_visual_eval_canonical_entrypoint_v1"


def assert_not_data2(path: Path, *, role: str) -> None:
    """函数功能：阻止 P17a 默认 thin slice 读取或写入 `/data2`。"""
    resolved = str(path.resolve())
    if resolved == "/data2" or resolved.startswith("/data2/"):
        raise ValueError(f"P17a Visual eval canonical 默认禁止 /data2 {role}：{resolved}")


def assert_checkpoint_allowed(path: Path, *, allow_real_checkpoint: bool) -> None:
    """
    函数功能：
        执行 checkpoint guard：默认只允许 fixture/tempfile tiny payload。

    关键约束：
        即使显式开启 `--allow-real-checkpoint`，`/data2` checkpoint 在 P17a 仍禁止；
        后续真实 dry-run 必须单独设计更完整的授权和审计口径。
    """
    assert_not_data2(path, role="router_checkpoint_payload")
    if allow_real_checkpoint:
        return
    resolved = path.resolve()
    allowed_roots = (
        (REPO_ROOT / "tests" / "fixtures").resolve(),
        Path("/tmp").resolve(),
    )
    if not any(resolved == root or str(resolved).startswith(f"{root}/") for root in allowed_roots):
        raise ValueError(
            "--allow-real-checkpoint 未开启时，router checkpoint payload 只能位于 tests/fixtures 或 /tmp tempfile："
            f"{resolved}"
        )


def is_fixture_or_tempfile_checkpoint(path: Path) -> bool:
    """函数功能：判断 checkpoint path 是否属于 P17a 默认 tiny fixture/tempfile 范围。"""
    resolved = path.resolve()
    allowed_roots = (
        (REPO_ROOT / "tests" / "fixtures").resolve(),
        Path("/tmp").resolve(),
    )
    return any(resolved == root or str(resolved).startswith(f"{root}/") for root in allowed_roots)


def now_iso() -> str:
    """函数功能：生成 Runtime artifact 使用的 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """函数功能：计算输入文件 sha256，供 Runtime metadata 留痕。"""
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
        读取显式 SampleManifest CSV，并按 split 保留 ordered sample_keys。

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
        rows.append(
            SampleManifestRow(
                sample_key=str(source_row["sample_key"]),
                split=row_split,
                config_name=str(source_row["config_name"]),
                dataset_name=str(source_row["dataset_name"]),
                item_id=_required_int(source_row, "item_id"),
                channel_id=_required_int(source_row, "channel_id"),
                window_index=_required_int(source_row, "window_index"),
                seq_len=_optional_int(source_row, "seq_len"),
                pred_len=_optional_int(source_row, "pred_len"),
                extra={"source": "p17a_explicit_sample_manifest", "split_filter": split_name},
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


class JsonExpertProvider:
    """
    类功能：
        从显式 expert prediction JSON 构造 Visual eval 的 ExpertBatch。

    输入：
        expert_predictions_json: 包含 model_columns 和 samples 的小规模 JSON。

    输出：
        `load_batch(sample_keys)` 按调用方 ordered sample_keys 返回 ExpertBatch。

    关键约束：
        provider 只读取显式 JSON，不读取 full-scale cache，不接收 run_dir。
    """

    provider_name = "JsonExpertProvider"

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
            row_index_metadata={"source": str(self.expert_predictions_json), "storage": "json_fixture"},
            extra={"provider_name": self.provider_name, "fixture": "p17a_visual_eval_expert_json"},
        )


def import_legacy_visual_mlp_router() -> type[torch.nn.Module]:
    """
    函数功能：
        动态导入 legacy `VisualMLPRouter` 类型，只取 class，不启动训练入口。
    """
    module = importlib.import_module(LEGACY_MODULE_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    if not isinstance(router_cls, type) or not issubclass(router_cls, torch.nn.Module):
        raise TypeError(f"legacy VisualMLPRouter 必须是 torch.nn.Module class：actual={type(router_cls)!r}")
    return router_cls


def build_feature_batch(args: argparse.Namespace, ordered_sample_keys: Sequence[str]) -> tuple[FeatureBatch, Mapping[str, object]]:
    """
    函数功能：
        使用 precomputed fixture feature source，并可选执行 LoadedFeatureScaler。

    关键约束：
        P17a 只支持 precomputed feature CSV；真实 ViT provider 留到后续。
        scaler 只在显式传入 state JSON 时 transform，不做 silent fit。
    """
    if str(args.feature_source) != "precomputed":
        raise ValueError(f"P17a thin slice 只支持 precomputed feature_source：actual={args.feature_source}")
    visual_features_csv = Path(args.visual_features_csv)
    assert_not_data2(visual_features_csv, role="visual_features_csv")
    if not visual_features_csv.is_file():
        raise FileNotFoundError(f"visual_features_csv 不存在：{visual_features_csv}")

    provider = VisualPrecomputedFeatureProvider(feature_source_path=visual_features_csv)
    feature_batch = provider.load_batch(ordered_sample_keys)
    metadata: dict[str, object] = {
        "feature_source": "precomputed",
        "feature_provider": "VisualPrecomputedFeatureProvider",
        "visual_features_csv": {
            "reference_type": "file",
            "path": str(visual_features_csv),
            "checksum": sha256_file(visual_features_csv),
            "checksum_algorithm": "sha256",
        },
        "loads_real_vit": False,
    }

    if args.scaler_state_json:
        scaler_state_json = Path(args.scaler_state_json)
        assert_not_data2(scaler_state_json, role="scaler_state_json")
        if not scaler_state_json.is_file():
            raise FileNotFoundError(f"scaler_state_json 不存在：{scaler_state_json}")
        scaler = LoadedFeatureScaler.from_json(scaler_state_json)
        feature_batch = scaler.transform(feature_batch)
        metadata["scaler"] = {
            "scaler_enabled": True,
            "scaler_name": "LoadedFeatureScaler",
            "scaler_state_json": {
                "reference_type": "file",
                "path": str(scaler_state_json),
                "checksum": sha256_file(scaler_state_json),
                "checksum_algorithm": "sha256",
            },
            "fit_performed": False,
        }
    else:
        metadata["scaler"] = {
            "scaler_enabled": False,
            "fit_performed": False,
        }
    return feature_batch, metadata


def build_router_output(
    *,
    args: argparse.Namespace,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> tuple[RouterOutput, Mapping[str, object]]:
    """
    函数功能：
        在 Runtime/entrypoint 侧加载 checkpoint payload 和 legacy MLP，再交给 P16a adapter。

    关键约束：
        checkpoint path 不进入 adapter；默认不允许真实 checkpoint 或 `/data2` path。
    """
    checkpoint_payload_path = Path(args.router_checkpoint_payload)
    assert_checkpoint_allowed(checkpoint_payload_path, allow_real_checkpoint=bool(args.allow_real_checkpoint))
    payload = load_checkpoint_payload(checkpoint_payload_path, map_location="cpu")
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        config = {}

    input_dim = int(feature_batch.features.shape[1])
    output_dim = len(expert_batch.model_columns)
    hidden_dim = int(config.get("hidden_dim", max(4, input_dim + 1)))
    dropout = float(config.get("dropout", 0.0))
    configured_input_dim = int(config.get("input_dim", input_dim))
    configured_output_dim = int(config.get("output_dim", output_dim))
    if configured_input_dim != input_dim:
        raise ValueError(f"checkpoint config input_dim 与 FeatureBatch 不一致：checkpoint={configured_input_dim} feature={input_dim}")
    if configured_output_dim != output_dim:
        raise ValueError(f"checkpoint config output_dim 与 ExpertBatch 不一致：checkpoint={configured_output_dim} expert={output_dim}")

    model = import_legacy_visual_mlp_router()(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    router_state_dict = extract_router_state_dict(payload)
    load_router_state_dict(model, router_state_dict, strict=bool(args.strict_checkpoint_load))
    adapter = LoadedTorchMLPRouterHeadAdapter(
        model=model,
        device=torch.device("cpu"),
        adapter_name="P17aVisualEvalCanonicalLoadedLegacyMLPAdapter",
    )
    router_output = adapter.predict(feature_batch, expert_batch.model_columns)
    metadata = {
        "head": "LoadedTorchMLPRouterHeadAdapter over legacy VisualMLPRouter",
        "adapter_name": "LoadedTorchMLPRouterHeadAdapter",
        "loaded_legacy_mlp": True,
        "checkpoint_payload_source": "explicit_cli_path",
        "checkpoint_payload_path": str(checkpoint_payload_path),
        "checkpoint_payload_sha256": sha256_file(checkpoint_payload_path),
        "checkpoint_load_helper": "time_router.runtime.visual_mlp_checkpoint",
        "strict_checkpoint_load": bool(args.strict_checkpoint_load),
        "allow_real_checkpoint": bool(args.allow_real_checkpoint),
        "router_state_dict_key_count": len(router_state_dict),
        "legacy_module_import_path": LEGACY_MODULE_IMPORT_PATH,
        "legacy_router_class": "VisualMLPRouter",
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "loads_real_checkpoint": bool(args.allow_real_checkpoint) and not is_fixture_or_tempfile_checkpoint(checkpoint_payload_path),
    }
    return router_output, metadata


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


def assert_eval_contract(
    *,
    manifest: SampleManifest,
    expert_batch: ExpertBatch,
    feature_batch: FeatureBatch,
    router_output: RouterOutput,
    result: EvaluationInputAdapterResult,
    expert_provider: JsonExpertProvider,
) -> None:
    """函数功能：集中校验 P17a eval 链路的保序、shape 和边界。"""
    ordered_sample_keys = manifest.sample_keys()
    if expert_batch.sample_keys != ordered_sample_keys:
        raise AssertionError("ExpertBatch 未保持 manifest sample_key 顺序")
    if feature_batch.sample_keys != ordered_sample_keys:
        raise AssertionError("FeatureBatch 未保持 manifest sample_key 顺序")
    if router_output.sample_keys != ordered_sample_keys:
        raise AssertionError("RouterOutput 未保持 manifest sample_key 顺序")
    if result.evaluation_input.sample_keys != ordered_sample_keys:
        raise AssertionError("EvaluationInput 未保持 manifest sample_key 顺序")
    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError("RouterOutput model_columns 未与 ExpertBatch 对齐")
    if tuple(router_output.weights.shape) != (len(ordered_sample_keys), len(expert_batch.model_columns)):
        raise AssertionError(f"weights shape 异常：{router_output.weights.shape}")
    if not np.isfinite(router_output.weights).all() or not np.isfinite(router_output.logits).all():
        raise AssertionError("router logits/weights 必须全为有限值")
    np.testing.assert_allclose(np.sum(router_output.weights, axis=1), np.ones(len(ordered_sample_keys)), rtol=0.0, atol=1e-6)
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
    feature_batch: FeatureBatch,
    result: EvaluationInputAdapterResult,
    feature_metadata: Mapping[str, object],
    head_metadata: Mapping[str, object],
    run_dir: Path,
    created_at: str,
) -> None:
    """函数功能：Runtime 层统一写出 Visual eval canonical run_dir artifact。"""
    sample_manifest_csv = Path(args.sample_manifest_csv)
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
        "split_strategy_name": "p17a_visual_eval_split_filter",
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
    scaler_metadata = dict(feature_metadata["scaler"])
    metadata = {
        "run_artifact_schema_version": "stage1_run_artifact_v1",
        "protocol_version": PROTOCOL_VERSION,
        "sample_manifest_schema_version": "stage1_sample_manifest_v1",
        "evaluation_schema_version": "stage1_evaluation_summary_v1",
        "config_name": str(args.config_name),
        "branch_name": "visual_router_eval",
        "created_at": created_at,
        "inputs": {
            "sample_manifest": {"reference_type": "file", "path": str(sample_manifest_csv), "artifact_path": "inputs/sample_manifest_ref.json"},
            "split_summary": "inputs/split_summary.json",
            "expert_predictions_json": {
                "reference_type": "file",
                "path": str(expert_predictions_json),
                "checksum": sha256_file(expert_predictions_json),
                "checksum_algorithm": "sha256",
            },
            "visual_features_csv": dict(feature_metadata["visual_features_csv"]),
            "router_checkpoint_payload": {
                "reference_type": "file",
                "path": str(head_metadata["checkpoint_payload_path"]),
                "checksum": str(head_metadata["checkpoint_payload_sha256"]),
                "checksum_algorithm": "sha256",
            },
            "scaler_state_json": scaler_metadata.get("scaler_state_json"),
        },
        "visual_router": {
            "entrypoint": ENTRYPOINT_NAME,
            "feature_source": feature_metadata["feature_source"],
            "feature_provider": feature_metadata["feature_provider"],
            "feature_schema": dict(feature_batch.feature_schema),
            "loaded_legacy_mlp": True,
            "scaler_enabled": bool(scaler_metadata["scaler_enabled"]),
            "loads_real_checkpoint": bool(head_metadata["loads_real_checkpoint"]),
            "loads_real_vit": False,
            "training_started": False,
            "formal_training_migration": False,
            "head": head_metadata["head"],
            "adapter_name": head_metadata["adapter_name"],
            "checkpoint_payload_source": head_metadata["checkpoint_payload_source"],
            "scaler": scaler_metadata,
            "feature_lineage": dict(feature_metadata),
            "head_lineage": dict(head_metadata),
        },
    }
    status = {
        "status": "completed",
        "current_stage": ENTRYPOINT_NAME,
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
    log_path = run_dir / "logs" / "visual_eval_canonical.log"
    log_path.write_text(
        "\n".join(
            [
                f"created_at={created_at}",
                "stage=P17a Visual canonical evaluation entrypoint thin slice",
                f"sample_count={len(manifest.sample_keys())}",
                f"split_counts={manifest.split_counts()}",
                f"feature_shape={tuple(feature_batch.features.shape)}",
                f"hard_mae={evaluation_summary['metrics']['hard_mae']}",
                f"raw_soft_mae={evaluation_summary['metrics']['raw_soft_mae']}",
                "training_started=false",
                "loads_real_vit=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_entrypoint(args: argparse.Namespace) -> Path:
    """函数功能：执行 Visual eval canonical dataflow，并返回 run_dir。"""
    output_dir = Path(args.output_dir)
    for role, path in (
        ("output_dir", output_dir),
        ("sample_manifest_csv", Path(args.sample_manifest_csv)),
        ("expert_predictions_json", Path(args.expert_predictions_json)),
        ("visual_features_csv", Path(args.visual_features_csv)),
    ):
        assert_not_data2(path, role=role)
    for role, path in (
        ("sample_manifest_csv", Path(args.sample_manifest_csv)),
        ("expert_predictions_json", Path(args.expert_predictions_json)),
        ("visual_features_csv", Path(args.visual_features_csv)),
        ("router_checkpoint_payload", Path(args.router_checkpoint_payload)),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} 不存在：{path}")

    manifest = load_sample_manifest_csv(Path(args.sample_manifest_csv), split_name=str(args.split_name))
    ordered_sample_keys = manifest.sample_keys()
    feature_batch, feature_metadata = build_feature_batch(args, ordered_sample_keys)
    expert_provider = JsonExpertProvider(Path(args.expert_predictions_json))
    expert_batch = expert_provider.load_batch(feature_batch.sample_keys)
    router_output, head_metadata = build_router_output(args=args, feature_batch=feature_batch, expert_batch=expert_batch)
    result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)

    assert_eval_contract(
        manifest=manifest,
        expert_batch=expert_batch,
        feature_batch=feature_batch,
        router_output=router_output,
        result=result,
        expert_provider=expert_provider,
    )

    created_at = now_iso()
    run_dir = create_run_dir(output_dir, run_name=str(args.run_id))
    write_runtime_artifacts(
        args=args,
        manifest=manifest,
        feature_batch=feature_batch,
        result=result,
        feature_metadata=feature_metadata,
        head_metadata=head_metadata,
        run_dir=run_dir,
        created_at=created_at,
    )

    summary = build_evaluation_summary_payload(result)
    print(f"run_dir: {run_dir}")
    print(
        json.dumps(
            {
                "status": "completed",
                "entrypoint": ENTRYPOINT_NAME,
                "sample_count": len(ordered_sample_keys),
                "split_counts": manifest.split_counts(),
                "feature_source": str(args.feature_source),
                "loaded_legacy_mlp": True,
                "scaler_enabled": bool(dict(feature_metadata["scaler"])["scaler_enabled"]),
                "loads_real_checkpoint": bool(head_metadata["loads_real_checkpoint"]),
                "loads_real_vit": False,
                "training_started": False,
                "formal_training_migration": False,
                "hard_mae": summary["metrics"]["hard_mae"],
                "raw_soft_mae": summary["metrics"]["raw_soft_mae"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """函数功能：解析 Visual eval canonical CLI 参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 Visual canonical evaluation entrypoint thin slice.")
    parser.add_argument("--sample-manifest-csv", required=True, help="显式 SampleManifest CSV；P17a 禁止自动搜索。")
    parser.add_argument("--expert-predictions-json", required=True, help="显式 expert prediction JSON。")
    parser.add_argument("--visual-features-csv", required=True, help="显式 precomputed visual feature CSV fixture。")
    parser.add_argument("--router-checkpoint-payload", required=True, help="显式 router checkpoint payload；默认只允许 fixture/tempfile tiny payload。")
    parser.add_argument("--scaler-state-json", default=None, help="可选 loaded scaler state JSON；传入时执行 LoadedFeatureScaler。")
    parser.add_argument("--output-dir", required=True, help="canonical run_dir 的父目录；P17a 禁止使用 /data2。")
    parser.add_argument("--run-id", required=True, help="output-dir 下创建的单层 run 目录名。")
    parser.add_argument("--config-name", required=True, help="写入 run metadata 的 config 名。")
    parser.add_argument("--split-name", required=True, help="使用的 split；可传 all 使用全部 fixture。")
    parser.add_argument("--feature-source", default="precomputed", choices=("precomputed",), help="Visual feature source；P17a 只支持 precomputed。")
    parser.add_argument("--strict-checkpoint-load", action="store_true", help="对 checkpoint state_dict 执行 strict load。")
    parser.add_argument("--allow-real-checkpoint", action="store_true", help="显式允许非 fixture/tempfile checkpoint；仍禁止 /data2。默认 false。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """函数功能：命令行入口。"""
    args = parse_args(argv)
    run_entrypoint(args)


if __name__ == "__main__":
    main()
