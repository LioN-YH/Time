#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P15c Visual-specific small canonical entrypoint thin slice。

输入：
    命令行显式传入 small SampleManifest CSV、Visual mock history windows JSON、
    small expert prediction JSON 和 output dir；默认使用仓库内 P13b/P14b fixture。

输出：
    在 `output_dir/run_id/` 下写出 canonical run_dir：run metadata/status、
    inputs 引用、evaluation summary、prediction rows 和最小日志。

关键约束：
    本脚本只做 Visual branch-specific small fixture / mock feature / smoke adapter
    pattern 级别的 canonical rehearsal；不访问 `/data2`，不启动训练、
    不读取真实 checkpoint，不接真实视觉 MLP 路由器，不启动 ViT embedding，
    不修改 generic small CLI 或 TimeFuse small CLI，也不把 run_dir 传入 provider。
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
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import VisualMockFeatureProvider  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput, SampleManifest, SampleManifestRow  # noqa: E402
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
DEFAULT_HISTORY_WINDOWS_JSON = DEFAULT_FIXTURE_ROOT / "stage1_visual_feature_mock" / "history_windows.json"
DEFAULT_RUN_ID = "stage1_visual_small"
DEFAULT_FEATURE_DIM = 8


def assert_not_data2(path: Path, *, role: str) -> None:
    """函数功能：阻止 P15c Visual small entrypoint 读写 `/data2` 路径。"""
    resolved = str(path.resolve())
    if resolved == "/data2" or resolved.startswith("/data2/"):
        raise ValueError(f"P15c Visual small entrypoint 不应使用 /data2 {role}：{resolved}")


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


def load_history_windows_json(path: Path) -> dict[str, list[float]]:
    """
    函数功能：
        读取 Visual mock history window fixture，返回内存映射。

    关键约束：
        history fixture 只代表过去窗口 x；本函数不读取 checkpoint、prediction cache、
        oracle、scaler、ViT 资源或 run_dir。
    """
    assert_not_data2(path, role="history_windows_json")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("history_windows JSON 必须是非空 object")
    history_windows: dict[str, list[float]] = {}
    for sample_key, window in payload.items():
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError(f"history_windows JSON 存在非法 sample_key：{sample_key!r}")
        if not isinstance(window, list) or not window:
            raise ValueError(f"history window 必须是非空 list：sample_key={sample_key}")
        history_windows[sample_key] = [float(value) for value in window]
    return history_windows


class JsonExpertSmallProvider:
    """
    类功能：
        从 small expert prediction JSON 构造 Visual small entrypoint 的 ExpertBatch。

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
            extra={"provider_name": self.provider_name, "fixture": "p15c_visual_small_expert_json"},
        )


class SmokeOnlyVisualMLP(torch.nn.Module):
    """
    类功能：
        P15c smoke-only Visual MLP head 替身。

    输入：
        head-ready float32 `FeatureBatch.features` tensor，shape 为 `[sample, feature_dim]`。

    输出：
        未归一化 logits tensor，shape 为 `[sample, num_experts]`。

    关键约束：
        该类只服务 small entrypoint rehearsal，不是正式视觉路由 head adapter，
        不读取 checkpoint，不处理 scaler，不决定 device/DataParallel。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"MLP 维度必须为正数：input_dim={input_dim} output_dim={output_dim}")
        hidden_dim = max(4, input_dim + 1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：将 head-ready features 前向映射为专家 logits。"""
        return self.net(features)


class SmokeOnlyVisualMLPAdapter:
    """
    类功能：
        P15c 脚本局部 smoke-only adapter，将已加载 MLP 输出包装为 RouterOutput。

    输入：
        FeatureBatch、显式 model_columns、Runtime 已加载好的 MLP 和 device。

    输出：
        RouterOutput(sample_keys, model_columns, logits, weights, extra)。

    关键约束：
        adapter 不读取 prediction cache、oracle/error、run_dir、checkpoint、scaler
        或 ViT 资源；它只消费内存 FeatureBatch 并做一次 deterministic forward。
    """

    adapter_name = "SmokeOnlyVisualMLPAdapter"

    def __init__(self, *, mlp: torch.nn.Module, device: torch.device | str) -> None:
        self.mlp = mlp
        self.device = torch.device(device)
        self.mlp.to(self.device)
        self.mlp.eval()

    def predict(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """函数功能：执行 MLP forward，并沿专家维 softmax 为融合权重。"""
        columns = tuple(str(model_name) for model_name in model_columns)
        if not columns:
            raise ValueError("Visual smoke MLP adapter 需要非空 model_columns")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Visual smoke MLP adapter 收到重复 model_columns：{columns}")

        features = np.asarray(feature_batch.features)
        if features.dtype != np.float32:
            raise ValueError(f"FeatureBatch.features 必须是 head-ready float32：actual={features.dtype}")
        if features.ndim != 2:
            raise ValueError(f"FeatureBatch.features 必须是二维矩阵：actual_shape={features.shape}")
        if features.shape[0] != len(feature_batch.sample_keys):
            raise ValueError("FeatureBatch.features 样本维度必须等于 sample_keys 数量")
        if not np.all(np.isfinite(features)):
            raise ValueError("FeatureBatch.features 包含 NaN 或 Inf")

        feature_tensor = torch.from_numpy(features).to(device=self.device)
        with torch.inference_mode():
            logits_tensor = self.mlp(feature_tensor)
            if logits_tensor.ndim != 2:
                raise ValueError(f"Visual smoke MLP logits 必须是二维矩阵：actual_shape={tuple(logits_tensor.shape)}")
            if logits_tensor.shape != (len(feature_batch.sample_keys), len(columns)):
                raise ValueError(
                    "Visual smoke MLP logits shape 必须与 sample/model 维度对齐："
                    f"logits={tuple(logits_tensor.shape)} samples={len(feature_batch.sample_keys)} columns={len(columns)}"
                )
            weights_tensor = torch.softmax(logits_tensor, dim=1)

        logits = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        weights = weights_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return RouterOutput(
            sample_keys=tuple(feature_batch.sample_keys),
            model_columns=columns,
            logits=logits,
            weights=weights,
            extra={
                "adapter_name": self.adapter_name,
                "adapter_scope": "p15c_script_local_smoke_only",
                "head_source": "in_memory_deterministic_smoke_mlp_state_dict",
                "feature_contract": "visual_mock_head_ready_float32_features",
                "formal_visual_router_head": False,
                "loads_real_checkpoint": False,
                "loads_real_vit": False,
            },
        )


def build_loaded_smoke_mlp(*, input_dim: int, output_dim: int) -> SmokeOnlyVisualMLP:
    """
    函数功能：
        构造 deterministic in-memory Visual smoke MLP。

    关键约束：
        本函数不读 checkpoint 文件、不调用 checkpoint loader；固定初始化只用于 small
        rehearsal，让 RouterOutput 可复现。
    """
    torch.manual_seed(20260621)
    mlp = SmokeOnlyVisualMLP(input_dim=input_dim, output_dim=output_dim)
    for parameter in mlp.parameters():
        torch.nn.init.uniform_(parameter, a=-0.12, b=0.12)
    return mlp


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
    feature_batch: FeatureBatch,
    router_output: RouterOutput,
    result: EvaluationInputAdapterResult,
    expert_provider: JsonExpertSmallProvider,
    expected_feature_dim: int,
) -> None:
    """函数功能：集中校验 P15c small canonical chain 的保序、shape 和边界。"""
    ordered_sample_keys = manifest.sample_keys()
    if expert_batch.sample_keys != ordered_sample_keys:
        raise AssertionError("ExpertBatch 未保持 manifest sample_key 顺序")
    if feature_batch.sample_keys != ordered_sample_keys:
        raise AssertionError("FeatureBatch 未保持 manifest sample_key 顺序")
    if router_output.sample_keys != ordered_sample_keys:
        raise AssertionError("RouterOutput 未保持 manifest sample_key 顺序")
    if result.evaluation_input.sample_keys != ordered_sample_keys:
        raise AssertionError("EvaluationInput 未保持 manifest sample_key 顺序")
    if tuple(feature_batch.features.shape) != (len(ordered_sample_keys), expected_feature_dim):
        raise AssertionError(f"FeatureBatch shape 异常：{feature_batch.features.shape}")
    if feature_batch.features.dtype != np.float32:
        raise AssertionError(f"FeatureBatch dtype 必须为 float32：actual={feature_batch.features.dtype}")
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
    run_dir: Path,
    created_at: str,
) -> None:
    """函数功能：Runtime 层统一写出 canonical run_dir artifact。"""
    sample_manifest_csv = Path(args.sample_manifest_csv)
    history_windows_json = Path(args.history_windows_json)
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
        "split_strategy_name": "p15c_visual_small_split_filter",
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
        "protocol_version": "stage1_visual_small_entrypoint_v1",
        "sample_manifest_schema_version": "stage1_sample_manifest_v1",
        "evaluation_schema_version": "stage1_evaluation_summary_v1",
        "config_name": str(args.config_name),
        "branch_name": "visual_router_small",
        "created_at": created_at,
        "inputs": {
            "sample_manifest": {"reference_type": "file", "path": str(sample_manifest_csv), "artifact_path": "inputs/sample_manifest_ref.json"},
            "split_summary": "inputs/split_summary.json",
            "history_windows_json": {
                "reference_type": "file",
                "path": str(history_windows_json),
                "checksum": sha256_file(history_windows_json),
                "checksum_algorithm": "sha256",
            },
            "expert_predictions_json": {
                "reference_type": "file",
                "path": str(expert_predictions_json),
                "checksum": sha256_file(expert_predictions_json),
                "checksum_algorithm": "sha256",
            },
        },
        "visual_router": {
            "feature_provider": "VisualMockFeatureProvider",
            "feature_schema": dict(feature_batch.feature_schema),
            "head": "SmokeOnlyVisualMLPAdapter script-local deterministic in-memory MLP",
            "training": "not_started_p15c_small_rehearsal_only",
            "formal_visual_router_migration": False,
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }
    status = {
        "status": "completed",
        "current_stage": "visual_small_entrypoint",
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
    log_path = run_dir / "logs" / "visual_small_entrypoint.log"
    log_path.write_text(
        "\n".join(
            [
                f"created_at={created_at}",
                "stage=P15c Visual-specific small canonical entrypoint",
                f"sample_count={len(manifest.sample_keys())}",
                f"split_counts={manifest.split_counts()}",
                f"feature_shape={tuple(feature_batch.features.shape)}",
                f"hard_mae={evaluation_summary['metrics']['hard_mae']}",
                f"raw_soft_mae={evaluation_summary['metrics']['raw_soft_mae']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_entrypoint(args: argparse.Namespace) -> Path:
    """函数功能：执行 P15c Visual small canonical dataflow，并返回 run_dir。"""
    output_dir = Path(args.output_dir)
    sample_manifest_csv = Path(args.sample_manifest_csv)
    history_windows_json = Path(args.history_windows_json)
    expert_predictions_json = Path(args.expert_predictions_json)
    for role, path in (
        ("output_dir", output_dir),
        ("sample_manifest_csv", sample_manifest_csv),
        ("history_windows_json", history_windows_json),
        ("expert_predictions_json", expert_predictions_json),
    ):
        assert_not_data2(path, role=role)
    for role, path in (
        ("sample_manifest_csv", sample_manifest_csv),
        ("history_windows_json", history_windows_json),
        ("expert_predictions_json", expert_predictions_json),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} 不存在：{path}")

    manifest = load_sample_manifest_csv(sample_manifest_csv, split_name=str(args.split_name))
    ordered_sample_keys = manifest.sample_keys()
    history_windows = load_history_windows_json(history_windows_json)
    feature_provider = VisualMockFeatureProvider(
        history_windows=history_windows,
        history_source_name="stage1_visual_feature_mock_history_window_x",
        source=f"{history_windows_json}:in_memory",
    )
    feature_batch = feature_provider.load_batch(ordered_sample_keys)
    expert_provider = JsonExpertSmallProvider(expert_predictions_json)
    expert_batch = expert_provider.load_batch(feature_batch.sample_keys)
    mlp = build_loaded_smoke_mlp(input_dim=int(feature_batch.features.shape[1]), output_dim=len(expert_batch.model_columns))
    router_output = SmokeOnlyVisualMLPAdapter(mlp=mlp, device=torch.device("cpu")).predict(feature_batch, expert_batch.model_columns)
    result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)

    if args.strict:
        assert_strict_contract(
            manifest=manifest,
            expert_batch=expert_batch,
            feature_batch=feature_batch,
            router_output=router_output,
            result=result,
            expert_provider=expert_provider,
            expected_feature_dim=int(args.feature_dim),
        )

    created_at = now_iso()
    run_dir = create_run_dir(output_dir, run_name=str(args.run_id))
    write_runtime_artifacts(
        args=args,
        manifest=manifest,
        feature_batch=feature_batch,
        result=result,
        run_dir=run_dir,
        created_at=created_at,
    )

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
    """函数功能：解析 P15c Visual-specific small CLI 参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 P15c Visual-specific small canonical entrypoint.")
    parser.add_argument("--sample-manifest-csv", default=str(DEFAULT_SAMPLE_MANIFEST_CSV), help="small SampleManifest CSV。")
    parser.add_argument("--history-windows-json", default=str(DEFAULT_HISTORY_WINDOWS_JSON), help="small Visual mock history windows JSON。")
    parser.add_argument("--expert-predictions-json", default=str(DEFAULT_EXPERT_PREDICTIONS_JSON), help="small expert prediction JSON。")
    parser.add_argument("--output-dir", required=True, help="canonical run_dir 的父目录；P15c 禁止使用 /data2。")
    parser.add_argument("--split-name", default="test", help="使用的 split；默认 test，传 all 使用全部 small fixture。")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="output-dir 下创建的单层 run 目录名。")
    parser.add_argument("--config-name", default="96_48_S", help="写入 run metadata 的 config 名。")
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM, help="VisualMockFeatureProvider 期望输出维度，默认 8。")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启，校验 sample_key 保序、Visual feature、weights 和 provider/run_dir 边界。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """函数功能：CLI 主入口。"""
    args = parse_args(argv)
    run_entrypoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
