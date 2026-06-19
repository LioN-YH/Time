#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P6b FusionEvaluator adapter smoke。

输入：
    默认读取 2026-06-14 的 4 sample packed golden fixture，通过
    PredictionCacheExpertProvider 显式构造 ExpertBatch，再用 golden weights
    构造 RouterOutput。

输出：
    标准输出打印中文检查日志；任一 adapter contract 或 golden 指标漂移时
    抛出 AssertionError。

关键约束：
    该 smoke 只允许 provider 阶段读取 golden fixture；进入 FusionEvaluator
    adapter 后会阻断常见文件读取 API，验证 adapter 不重新读取 prediction cache、
    不访问 oracle/TSF、不创建正式输出目录，也不写任何结果文件。
"""

from __future__ import annotations

import argparse
import builtins
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import FusionEvaluationResult, FusionEvaluator  # noqa: E402
from time_router.evaluation.metrics import compute_weight_entropy  # noqa: E402
from time_router.experts import PredictionCacheExpertProvider  # noqa: E402
from time_router.io import DEFAULT_MODEL_COLUMNS  # noqa: E402
from time_router.protocols import EvaluationInput, ExpertBatch, RouterOutput  # noqa: E402


DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "experiment_logs"
    / "run_outputs"
    / "2026-06-14_stage1_full_scale_dry_run_v2"
    / "merged_cache"
)
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"
EXPECTED_MODEL_COLUMNS = tuple(DEFAULT_MODEL_COLUMNS)
EXPECTED_SAMPLE_KEYS = (
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873",
)
EXPECTED_HARD_MODELS = ["CrossFormer", "DLinear", "PatchTST", "DLinear"]
EXPECTED_SELECTED_INDICES = [2, 0, 1, 0]
EXPECTED_SELECTED_COUNTS = {"DLinear": 2, "PatchTST": 1, "CrossFormer": 1, "ES": 0, "NaiveForecaster": 0}
EXPECTED_HARD_MAE = 0.41604843735694885
EXPECTED_HARD_MSE = 0.4563697576522827
EXPECTED_RAW_SOFT_MAE = 0.4102966785430908
EXPECTED_RAW_SOFT_MSE = 0.48815402388572693
GOLDEN_WEIGHTS = np.asarray(
    [
        [0.10, 0.20, 0.55, 0.05, 0.10],
        [0.65, 0.10, 0.15, 0.05, 0.05],
        [0.05, 0.70, 0.10, 0.10, 0.05],
        [0.30, 0.30, 0.20, 0.10, 0.10],
    ],
    dtype=np.float32,
)
EXPECTED_ROW_FIELDS = {
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


def parse_args() -> argparse.Namespace:
    """函数功能：解析只读 fixture 路径和浮点容忍度。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 FusionEvaluator adapter smoke.")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="只读 merged_cache fixture 目录，需包含 manifest.csv 和 packed npy 数组。",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="MAE/MSE 浮点比对绝对容忍度。")
    return parser.parse_args()


def assert_close(name: str, actual: float, expected: float, *, atol: float) -> None:
    """函数功能：带中文上下文的浮点 golden 断言。"""
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name} 不一致：actual={actual:.12f} expected={expected:.12f} atol={atol}")


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录正式 run_outputs 根目录下一层目录名，用于检查 adapter 不创建输出目录。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


@contextmanager
def forbid_adapter_file_reads() -> Iterator[None]:
    """
    函数功能：
        在 adapter 调用阶段阻断常见文件读取路径。

    关键约束：
        provider 已经在进入该上下文前完成 fixture 读取；如果 adapter 回读
        manifest、packed npy、oracle/TSF 或其他文件，这里会立即失败。
    """

    def fail_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("FusionEvaluator adapter 阶段不应调用 open/path.open")

    def fail_np_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("FusionEvaluator adapter 阶段不应调用 np.load 读取 prediction cache")

    with patch.object(builtins, "open", side_effect=fail_open), patch.object(
        Path, "open", side_effect=fail_open
    ), patch.object(np, "load", side_effect=fail_np_load):
        yield


def build_inputs(fixture_root: Path) -> tuple[ExpertBatch, RouterOutput]:
    """函数功能：使用 P6a provider 和 golden weights 构造 adapter 输入。"""
    provider = PredictionCacheExpertProvider(fixture_root=fixture_root, model_columns=EXPECTED_MODEL_COLUMNS)
    expert_batch = provider.load_batch(EXPECTED_SAMPLE_KEYS, verify_metrics=True)
    router_output = RouterOutput(
        sample_keys=expert_batch.sample_keys,
        model_columns=expert_batch.model_columns,
        weights=GOLDEN_WEIGHTS,
        extra={"weights_source": "stage1_golden_smoke"},
    )
    return expert_batch, router_output


def assert_adapter_result(result: FusionEvaluationResult, *, expert_batch: ExpertBatch, atol: float) -> None:
    """函数功能：验证 adapter 复算结果与 golden smoke 当前锁定口径一致。"""
    if not isinstance(result.evaluation_input, EvaluationInput):
        raise AssertionError(f"adapter 未构造 EvaluationInput：actual={type(result.evaluation_input)!r}")
    if result.evaluation_input.sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"EvaluationInput sample_keys 顺序漂移：{result.evaluation_input.sample_keys}")
    if result.evaluation_input.model_columns != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(f"EvaluationInput model_columns 漂移：{result.evaluation_input.model_columns}")
    if result.evaluation_input.y_pred is not expert_batch.y_pred:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch.y_pred，不应复制或重读")
    if result.evaluation_input.y_true is not expert_batch.y_true:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch.y_true，不应复制或重读")
    if result.evaluation_input.weights is not GOLDEN_WEIGHTS:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")

    summary = result.summary
    assert_close("summary hard_mae", float(summary["hard_mae"]), EXPECTED_HARD_MAE, atol=atol)
    assert_close("summary hard_mse", float(summary["hard_mse"]), EXPECTED_HARD_MSE, atol=atol)
    assert_close("summary raw_soft_mae", float(summary["raw_soft_mae"]), EXPECTED_RAW_SOFT_MAE, atol=atol)
    assert_close("summary raw_soft_mse", float(summary["raw_soft_mse"]), EXPECTED_RAW_SOFT_MSE, atol=atol)
    if summary["selected_counts"] != EXPECTED_SELECTED_COUNTS:
        raise AssertionError(f"summary selected_counts 漂移：{summary['selected_counts']}")
    if summary["model_columns"] != list(EXPECTED_MODEL_COLUMNS):
        raise AssertionError(f"summary model_columns 漂移：{summary['model_columns']}")
    if summary["num_samples"] != len(EXPECTED_SAMPLE_KEYS) or summary["num_experts"] != len(EXPECTED_MODEL_COLUMNS):
        raise AssertionError(f"summary 样本/专家数漂移：{summary}")

    rows = result.per_sample_rows
    if len(rows) != len(EXPECTED_SAMPLE_KEYS):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(rows)}")
    if [row["sample_key"] for row in rows] != list(EXPECTED_SAMPLE_KEYS):
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：{rows}")
    if [row["selected_model"] for row in rows] != EXPECTED_HARD_MODELS:
        raise AssertionError(f"per-sample rows selected_model 漂移：{rows}")
    if [row["selected_index"] for row in rows] != EXPECTED_SELECTED_INDICES:
        raise AssertionError(f"per-sample rows selected_index 漂移：{rows}")
    for row in rows:
        if set(row) != EXPECTED_ROW_FIELDS:
            raise AssertionError(f"per-sample row 字段集合漂移：actual={sorted(row)}")

    np.testing.assert_allclose(
        [row["hard_mae"] for row in rows],
        np.mean(np.abs(result.hard_result.fused_pred - expert_batch.y_true), axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        [row["hard_mse"] for row in rows],
        np.mean((result.hard_result.fused_pred - expert_batch.y_true) ** 2, axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        [row["raw_soft_mae"] for row in rows],
        np.mean(np.abs(result.raw_soft_result.fused_pred - expert_batch.y_true), axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        [row["raw_soft_mse"] for row in rows],
        np.mean((result.raw_soft_result.fused_pred - expert_batch.y_true) ** 2, axis=(1, 2)),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose([row["max_weight"] for row in rows], np.max(GOLDEN_WEIGHTS, axis=1), rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        [row["weight_entropy"] for row in rows],
        compute_weight_entropy(GOLDEN_WEIGHTS),
        rtol=0.0,
        atol=atol,
    )


def run_smoke(fixture_root: Path, *, atol: float) -> None:
    """函数功能：执行 FusionEvaluator adapter 最小 contract smoke。"""
    fixture_root = fixture_root.resolve()
    print(f"开始 Stage 1 FusionEvaluator adapter smoke：fixture_root={fixture_root}")
    if str(fixture_root).startswith("/data2/"):
        raise AssertionError("adapter smoke 不应访问 /data2 fixture")

    before_dirs = snapshot_run_outputs()
    expert_batch, router_output = build_inputs(fixture_root)
    if expert_batch.sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"ExpertBatch sample_keys 顺序漂移：{expert_batch.sample_keys}")
    if expert_batch.model_columns != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(f"ExpertBatch model_columns 漂移：{expert_batch.model_columns}")
    print("通过：PredictionCacheExpertProvider 显式构造 ExpertBatch，sample/model 顺序与 golden 一致")

    evaluator = FusionEvaluator()
    with forbid_adapter_file_reads():
        result = evaluator.evaluate(expert_batch=expert_batch, router_output=router_output)
        direct_result = evaluator.evaluate(evaluation_input=result.evaluation_input)
    print("通过：adapter 阶段未调用 open/path.open/np.load，不重新读取 prediction cache 或 oracle/TSF")

    assert_adapter_result(result, expert_batch=expert_batch, atol=atol)
    assert_adapter_result(direct_result, expert_batch=expert_batch, atol=atol)
    after_dirs = snapshot_run_outputs()
    if after_dirs != before_dirs:
        raise AssertionError(f"adapter 不应创建正式输出目录：before={sorted(before_dirs)} after={sorted(after_dirs)}")

    if result.diagnostics.get("adapter_name") != "FusionEvaluator":
        raise AssertionError(f"adapter diagnostics 漂移：{result.diagnostics}")
    if "lineage" not in result.diagnostics:
        raise AssertionError("adapter diagnostics 应保留来自 ExpertBatch/RouterOutput 的轻量 lineage")
    print(
        "通过：adapter 复算 hard/raw-soft summary 和 per-sample rows，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    print("完成：Stage 1 FusionEvaluator adapter smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    args = parse_args()
    run_smoke(args.fixture_root, atol=float(args.atol))


if __name__ == "__main__":
    main()
