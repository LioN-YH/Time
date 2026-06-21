#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P14d Visual mock protocol eval smoke。

输入：
    使用 P13b `sample_manifest.csv` 的 ordered sample_keys、P14b
    `history_windows.json` 的 Visual mock history windows，以及 P13b
    `expert_predictions.json` 的小型专家预测数值参考。

输出：
    标准输出打印中文检查日志；若 Visual mock FeatureBatch + mock RouterHead +
    EvaluationInputAdapter 的纯内存协议链路在保序、shape、权重归一化、
    summary/rows 或边界约束上漂移，则抛出 AssertionError。

关键约束：
    本 smoke 只验证内存协议连接，不改正式入口，不加载真实 ViT，不接 legacy MLP，
    不写 canonical run_dir，不读取 prediction cache / oracle / run_dir，不访问 `/data2`，
    不启动训练、pressure 或 full-scale。
"""

from __future__ import annotations

import builtins
import csv
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from tests.helpers.visual_smoke_providers import VisualMockFeatureProvider  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
VISUAL_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_mock"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
HISTORY_WINDOWS_PATH = VISUAL_FIXTURE_ROOT / "history_windows.json"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

EXPECTED_FEATURE_DIM = 8
ATOL = 1e-6
DISALLOWED_RUNTIME_TOKENS = (
    "prediction cache",
    "oracle",
    "run_dir",
    "checkpoint",
    "status",
    "metadata",
    "/data2",
)


class DeterministicVisualMockRouterHead:
    """
    类功能：
        P14d smoke-only Visual mock RouterHead。

    输入：
        `FeatureBatch.features` 二维矩阵和调用方显式传入的 `model_columns`。

    输出：
        `RouterOutput`，其中 sample_keys 保持 FeatureBatch 顺序，model_columns
        与 ExpertBatch 对齐，weights 为按行 softmax 后的 `[sample, model]`。

    关键约束：
        该 head 不读取 prediction cache、oracle/error、run_dir 或 checkpoint；
        不代表正式 Visual MLP adapter，也不加载真实 ViT。
    """

    head_name = "DeterministicVisualMockRouterHead"

    def __init__(self) -> None:
        # 中文注释：固定权重只服务 smoke，让输出可复算且不会依赖训练状态。
        self._base_weight = np.asarray(
            [
                [0.20, -0.10, 0.05],
                [-0.05, 0.10, 0.15],
                [0.08, 0.04, -0.12],
                [0.03, 0.12, 0.02],
                [-0.10, 0.06, 0.14],
                [0.09, -0.08, 0.05],
                [0.04, 0.11, -0.03],
                [0.02, -0.04, 0.09],
            ],
            dtype=np.float64,
        )
        self._base_bias = np.asarray([0.03, -0.02, 0.01], dtype=np.float64)

    def predict(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """
        函数功能：
            将 Visual mock FeatureBatch 映射为 deterministic RouterOutput。

        输入：
            feature_batch: P14b mock provider 输出的 FeatureBatch。
            model_columns: ExpertBatch 提供的专家列顺序。

        输出：
            RouterOutput；weights 每行归一化为 1，shape 为 `[sample, model]`。
        """
        columns = tuple(str(model_name) for model_name in model_columns)
        if not columns:
            raise ValueError("mock RouterHead 需要非空 model_columns")
        if len(columns) != len(set(columns)):
            raise ValueError(f"mock RouterHead 收到重复 model_columns：{columns}")

        features = np.asarray(feature_batch.features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"FeatureBatch.features 必须是二维矩阵：actual={features.shape}")
        if features.shape[0] != len(feature_batch.sample_keys):
            raise ValueError(
                "FeatureBatch.features 样本维度必须等于 sample_keys 数量："
                f"features={features.shape[0]} keys={len(feature_batch.sample_keys)}"
            )
        if features.shape[1] != EXPECTED_FEATURE_DIM:
            raise ValueError(f"Visual mock feature dim 漂移：actual={features.shape[1]}")

        weight = self._adapt_columns(self._base_weight, len(columns))
        bias = self._adapt_columns(self._base_bias, len(columns))
        logits = features @ weight + bias
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return RouterOutput(
            sample_keys=tuple(feature_batch.sample_keys),
            model_columns=columns,
            logits=logits.astype(np.float64, copy=False),
            weights=weights.astype(np.float64, copy=False),
            extra={
                "head_name": self.head_name,
                "scope": "p14d_smoke_only",
                "reads_prediction_cache": False,
                "reads_oracle": False,
                "writes_run_dir": False,
            },
        )

    def _adapt_columns(self, matrix: np.ndarray, num_columns: int) -> np.ndarray:
        """
        函数功能：
            将固定三专家参数扩展或截断到当前 fixture 的专家数。

        关键约束：
            P13b small fixture 当前是三专家；这里保留泛化校验，防止未来小 fixture
            调整专家数时 head 仍能 deterministic 生成 logits。
        """
        if matrix.ndim == 1:
            if num_columns <= matrix.shape[0]:
                return matrix[:num_columns]
            repeats = int(np.ceil(num_columns / matrix.shape[0]))
            return np.tile(matrix, repeats)[:num_columns]
        if num_columns <= matrix.shape[1]:
            return matrix[:, :num_columns]
        repeats = int(np.ceil(num_columns / matrix.shape[1]))
        return np.tile(matrix, (1, repeats))[:, :num_columns]


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入 fixture 位于仓库内且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"fixture 文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P14d smoke 不应访问 /data2 fixture：{path}")


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """函数功能：从 P13b manifest 中读取 ordered sample_keys。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{sample_keys}")
    return sample_keys


def load_history_windows(path: Path) -> dict[str, list[float]]:
    """
    函数功能：
        读取 P14b Visual mock history windows。

    关键约束：
        history fixture 只保存过去窗口 x；不允许包含 future y、prediction、oracle
        或 run artifact 字段。
    """
    assert_repo_file(path)
    raw_text = path.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    for token in DISALLOWED_RUNTIME_TOKENS:
        if token in lowered:
            raise AssertionError(f"history window fixture 不应包含禁用字段或路径：{token}")

    payload = json.loads(raw_text)
    if not isinstance(payload, dict) or not payload:
        raise AssertionError("history_windows.json 必须是非空 sample_key -> window 映射")
    history_windows: dict[str, list[float]] = {}
    for sample_key, window in payload.items():
        if not isinstance(sample_key, str) or not sample_key:
            raise AssertionError(f"history window fixture 存在非法 sample_key：{sample_key!r}")
        if not isinstance(window, list) or not window:
            raise AssertionError(f"history window 必须是非空 list：sample_key={sample_key}")
        history_windows[sample_key] = [float(value) for value in window]
    return history_windows


def load_expert_batch_from_reference(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """
    函数功能：
        用 P13b expert JSON 数值参考构造最小 ExpertBatch。

    关键约束：
        `expert_predictions.json` 只是 small fixture 数值参考，不是正式
        prediction backend schema；正式路径仍应由 prediction backend/provider 提供。
    """
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError("expert_predictions.json 不是 JSON object")

    model_columns = tuple(str(model_name) for model_name in payload["model_columns"])
    if not model_columns or len(model_columns) != len(set(model_columns)):
        raise AssertionError(f"model_columns 异常：{model_columns}")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise AssertionError("expert_predictions.json 缺少 samples list")
    sample_by_key: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise AssertionError(f"sample 不是 object：{sample}")
        sample_key = str(sample["sample_key"])
        if sample_key in sample_by_key:
            raise AssertionError(f"expert_predictions.json 存在重复 sample_key：{sample_key}")
        sample_by_key[sample_key] = sample
    if set(sample_by_key) != set(ordered_sample_keys):
        raise AssertionError(
            "expert_predictions.json sample_key 集合未与 manifest 对齐："
            f"json={sorted(sample_by_key)} manifest={sorted(ordered_sample_keys)}"
        )

    y_true_rows = []
    y_pred_rows = []
    for sample_key in ordered_sample_keys:
        sample = sample_by_key[sample_key]
        y_true_rows.append(np.asarray(sample["y_true"], dtype=np.float32))
        y_pred = np.asarray(sample["y_pred"], dtype=np.float32)
        if y_pred.shape[0] != len(model_columns):
            raise AssertionError(f"{sample_key} y_pred 专家维与 model_columns 不一致：shape={y_pred.shape}")
        y_pred_rows.append(y_pred)

    y_true = np.stack(y_true_rows, axis=0).astype(np.float32)
    y_pred = np.stack(y_pred_rows, axis=0).astype(np.float32)
    if y_pred.shape[2:] != y_true.shape[1:]:
        raise AssertionError(f"ExpertBatch y_pred/y_true target shape 不一致：{y_pred.shape} vs {y_true.shape}")
    return ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in ordered_sample_keys),
        model_columns=model_columns,
        y_pred=y_pred,
        y_true=y_true,
        row_index_metadata={
            "source": "tests/fixtures/stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
            "formal_backend_schema": False,
        },
        extra={
            "provider_name": "P14dInMemoryExpertBatchReference",
            "fixture": "stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
            "formal_backend_schema": False,
        },
    )


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


@contextmanager
def forbid_head_and_evaluator_io() -> Iterator[None]:
    """
    函数功能：
        在 mock head 与 EvaluationInputAdapter 阶段阻断常见文件 IO。

    关键约束：
        manifest、history 和 expert JSON 只允许在前置 fixture loading 阶段读取；
        head/evaluator 只能消费内存协议对象，不应回读 cache 或写 run_dir。
    """
    with patch.object(builtins, "open", side_effect=fail_runtime_io), patch.object(
        Path, "open", side_effect=fail_runtime_io
    ), patch.object(Path, "read_text", side_effect=fail_runtime_io), patch.object(
        np, "load", side_effect=fail_runtime_io
    ), patch.object(
        np, "save", side_effect=fail_runtime_io
    ), patch.object(
        np, "savez", side_effect=fail_runtime_io
    ):
        yield


def fail_runtime_io(*args: object, **kwargs: object) -> object:
    """函数功能：统一失败入口，说明 P14d head/evaluator 阶段不允许文件 IO。"""
    raise AssertionError(f"P14d mock head/evaluator 阶段不应访问文件系统或 np IO：args={args} kwargs={kwargs}")


def assert_feature_and_expert_alignment(feature_batch: FeatureBatch, expert_batch: ExpertBatch) -> None:
    """函数功能：检查 FeatureBatch 与 ExpertBatch 只通过 sample_keys 显式对齐。"""
    if feature_batch.sample_keys != expert_batch.sample_keys:
        raise AssertionError(
            "FeatureBatch sample_keys 必须与 ExpertBatch manifest 顺序一致："
            f"feature={feature_batch.sample_keys} expert={expert_batch.sample_keys}"
        )
    features = np.asarray(feature_batch.features)
    if features.shape != (len(expert_batch.sample_keys), EXPECTED_FEATURE_DIM):
        raise AssertionError(f"Visual mock features shape 漂移：actual={features.shape}")
    if features.dtype != np.float32:
        raise AssertionError(f"Visual mock features dtype 漂移：actual={features.dtype}")
    if not np.all(np.isfinite(features)):
        raise AssertionError("Visual mock features 包含 NaN 或 Inf")


def assert_router_output_contract(
    *,
    router_output: RouterOutput,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> None:
    """函数功能：验证 mock RouterHead 输出与 FeatureBatch / ExpertBatch 对齐。"""
    if router_output.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"RouterOutput sample_keys 未保持 FeatureBatch 顺序：{router_output.sample_keys}")
    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError(
            "RouterOutput model_columns 必须等于 ExpertBatch.model_columns："
            f"router={router_output.model_columns} expert={expert_batch.model_columns}"
        )
    logits = np.asarray(router_output.logits)
    weights = np.asarray(router_output.weights)
    expected_shape = (len(expert_batch.sample_keys), len(expert_batch.model_columns))
    if logits.shape != expected_shape:
        raise AssertionError(f"RouterOutput logits shape 漂移：actual={logits.shape} expected={expected_shape}")
    if weights.shape != expected_shape:
        raise AssertionError(f"RouterOutput weights shape 漂移：actual={weights.shape} expected={expected_shape}")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(weights)):
        raise AssertionError("RouterOutput logits/weights 包含 NaN 或 Inf")
    if np.any(weights < 0.0):
        raise AssertionError(f"RouterOutput weights 不应为负：{weights}")
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(expected_shape[0]), rtol=0.0, atol=ATOL)


def assert_evaluation_result_contract(
    *,
    result: EvaluationInputAdapterResult,
    expert_batch: ExpertBatch,
    router_output: RouterOutput,
    ordered_sample_keys: Sequence[str],
) -> None:
    """函数功能：检查 EvaluationInputAdapter 生成 summary/rows 且保序。"""
    evaluation_input = result.evaluation_input
    if evaluation_input.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"EvaluationInput sample_keys 未保持 manifest 顺序：{evaluation_input.sample_keys}")
    if evaluation_input.model_columns != expert_batch.model_columns:
        raise AssertionError(f"EvaluationInput model_columns 漂移：{evaluation_input.model_columns}")
    if evaluation_input.y_pred is not expert_batch.y_pred or evaluation_input.y_true is not expert_batch.y_true:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch.y_pred/y_true，不应复制或重读")
    if evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")

    summary = result.summary
    if summary["num_samples"] != len(ordered_sample_keys):
        raise AssertionError(f"summary sample_count 漂移：{summary}")
    if summary["num_experts"] != len(expert_batch.model_columns):
        raise AssertionError(f"summary expert count 漂移：{summary}")
    if summary["model_columns"] != list(expert_batch.model_columns):
        raise AssertionError(f"summary model_columns 漂移：{summary}")
    for metric_name in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse", "mean_entropy", "mean_max_weight"):
        metric_value = float(summary[metric_name])
        if not np.isfinite(metric_value):
            raise AssertionError(f"summary {metric_name} 不是有限值：{metric_value}")
    selected_counts = summary["selected_counts"]
    if set(selected_counts) != set(expert_batch.model_columns):
        raise AssertionError(f"selected_counts 专家集合漂移：{selected_counts}")
    if sum(int(value) for value in selected_counts.values()) != len(ordered_sample_keys):
        raise AssertionError(f"selected_counts 总数必须等于 sample_count：{selected_counts}")

    rows = result.per_sample_rows
    if len(rows) != len(ordered_sample_keys):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(rows)} expected={len(ordered_sample_keys)}")
    if [row["sample_key"] for row in rows] != list(ordered_sample_keys):
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：{rows}")
    required_row_fields = {
        "sample_key",
        "selected_model",
        "selected_index",
        "hard_mae",
        "hard_mse",
        "raw_soft_mae",
        "raw_soft_mse",
        "max_weight",
        "weight_entropy",
    }
    for row in rows:
        if set(row) != required_row_fields:
            raise AssertionError(f"per-sample row 字段集合漂移：{row}")
        if row["selected_model"] not in expert_batch.model_columns:
            raise AssertionError(f"row selected_model 不在 model_columns 中：{row}")
        selected_index = int(row["selected_index"])
        if selected_index < 0 or selected_index >= len(expert_batch.model_columns):
            raise AssertionError(f"row selected_index 越界：{row}")


def run_smoke() -> None:
    """函数功能：执行 P14d Visual mock protocol eval smoke。"""
    print("开始 Stage 1 P14d Visual mock protocol eval smoke")
    before_outputs = snapshot_run_outputs()

    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    history_windows = load_history_windows(HISTORY_WINDOWS_PATH)
    if set(history_windows) != set(ordered_sample_keys):
        raise AssertionError(
            "history windows sample_key 集合必须等于 manifest："
            f"history={sorted(history_windows)} manifest={sorted(ordered_sample_keys)}"
        )
    expert_batch = load_expert_batch_from_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    print("通过：已读取 P13b manifest 顺序，并用 P13b expert JSON 参考构造内存 ExpertBatch")

    provider = VisualMockFeatureProvider(
        history_windows=history_windows,
        history_source_name="stage1_visual_feature_mock_history_window_x",
        source="tests/fixtures/stage1_visual_feature_mock/history_windows.json:in_memory",
    )
    feature_batch = provider.load_batch(ordered_sample_keys)
    assert_feature_and_expert_alignment(feature_batch, expert_batch)
    print("通过：VisualMockFeatureProvider 输出 FeatureBatch 且与 ExpertBatch sample_keys 对齐")

    head = DeterministicVisualMockRouterHead()
    adapter = EvaluationInputAdapter()
    with forbid_head_and_evaluator_io():
        router_output = head.predict(feature_batch, expert_batch.model_columns)
        result = adapter.evaluate(expert_batch=expert_batch, router_output=router_output)
    print("通过：mock RouterHead + EvaluationInputAdapter 阶段未调用文件 IO、np.load 或 np.save")

    assert_router_output_contract(
        router_output=router_output,
        feature_batch=feature_batch,
        expert_batch=expert_batch,
    )
    assert_evaluation_result_contract(
        result=result,
        expert_batch=expert_batch,
        router_output=router_output,
        ordered_sample_keys=ordered_sample_keys,
    )
    print(
        "通过：RouterOutput 权重归一化，summary/rows 可生成且保留 manifest 顺序，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P14d smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P14d Visual mock protocol eval smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
