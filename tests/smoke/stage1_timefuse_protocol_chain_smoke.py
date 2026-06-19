#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P7c TimeFuse protocol chain smoke。

输入：
    使用 golden prediction fixture 构造 ExpertBatch，使用测试内临时 TimeFuse
    feature CSV 构造 FeatureBatch，再用固定 TimeFuseLinearSoftmaxHead 权重
    构造 RouterOutput。

输出：
    标准输出打印中文检查日志；任一协议链 contract 或 deterministic 数值漂移
    时抛出 AssertionError。

关键约束：
    该 smoke 只验证已完成 adapter 可组合，不训练、不算 loss、不建 optimizer、
    不保存 checkpoint；不访问 /data2，不新增 Bash/scripts，不迁移正式
    TimeFuse fusor 或 Visual Router 入口。
"""

from __future__ import annotations

import argparse
import builtins
import csv
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.evaluation.metrics import compute_weight_entropy  # noqa: E402
from time_router.experts import PredictionCacheExpertProvider  # noqa: E402
from time_router.features import TimeFuseFeatureCacheProvider  # noqa: E402
from time_router.io import DEFAULT_MODEL_COLUMNS  # noqa: E402
from time_router.models import TimeFuseLinearSoftmaxHead  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "experiment_logs"
    / "run_outputs"
    / "2026-06-14_stage1_full_scale_dry_run_v2"
    / "merged_cache"
)
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"
MODEL_COLUMNS = tuple(DEFAULT_MODEL_COLUMNS)
SAMPLE_KEYS = (
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873",
)
CSV_SAMPLE_ORDER = (SAMPLE_KEYS[2], SAMPLE_KEYS[0], SAMPLE_KEYS[3], SAMPLE_KEYS[1])
FEATURE_COLUMNS = tuple(f"timefuse_feature_{index:02d}" for index in range(17))
FEATURE_MATRIX = np.asarray(
    [
        [0.05 * (sample_index + 1) + 0.01 * feature_index for feature_index in range(17)]
        for sample_index in range(len(SAMPLE_KEYS))
    ],
    dtype=np.float64,
)
HEAD_WEIGHT = ((np.arange(17 * len(MODEL_COLUMNS), dtype=np.float64).reshape(17, len(MODEL_COLUMNS)) - 42.0) / 200.0)
HEAD_BIAS = np.asarray([0.05, -0.02, 0.03, -0.04, 0.01], dtype=np.float64)
EXPECTED_LOGITS = FEATURE_MATRIX @ HEAD_WEIGHT + HEAD_BIAS
EXPECTED_WEIGHTS = np.exp(EXPECTED_LOGITS - np.max(EXPECTED_LOGITS, axis=1, keepdims=True))
EXPECTED_WEIGHTS = EXPECTED_WEIGHTS / np.sum(EXPECTED_WEIGHTS, axis=1, keepdims=True)
EXPECTED_SUMMARY = {
    "hard_mae": 1.0935739278793335,
    "hard_mse": 2.6132822036743164,
    "raw_soft_mae": 0.5567512691186863,
    "raw_soft_mse": 0.85302892017598,
    "selected_counts": {"DLinear": 0, "PatchTST": 0, "CrossFormer": 0, "ES": 0, "NaiveForecaster": 4},
    "mean_entropy": 1.6089299887564834,
    "mean_max_weight": 0.20782637540049725,
}
EXPECTED_SELECTED_MODELS = ["NaiveForecaster", "NaiveForecaster", "NaiveForecaster", "NaiveForecaster"]
EXPECTED_SELECTED_INDICES = [4, 4, 4, 4]
EXPECTED_ROW_HARD_MAE = [1.3940553665161133, 1.6518787145614624, 0.5880991816520691, 0.7402625679969788]
EXPECTED_ROW_HARD_MSE = [3.832878828048706, 5.20277738571167, 0.5233125686645508, 0.8941592574119568]
EXPECTED_ROW_RAW_SOFT_MAE = [0.6113177514528844, 0.946651373308128, 0.2589350126301558, 0.4101009390835768]
EXPECTED_ROW_RAW_SOFT_MSE = [1.0849780719567559, 1.8907058868745656, 0.1034200656718484, 0.3330116562007495]


def parse_args() -> argparse.Namespace:
    """函数功能：解析只读 fixture 路径和浮点容忍度。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 TimeFuse protocol chain smoke.")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="只读 merged_cache golden fixture 目录，需包含 manifest.csv 和 packed npy 数组。",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="MAE/MSE 与权重比对绝对容忍度。")
    return parser.parse_args()


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于检查链路不创建输出目录。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def write_feature_csv(feature_csv_path: Path) -> None:
    """
    函数功能：
        写入只含 sample_key 和 17 维 TimeFuse 特征的临时 fixture。

    关键约束：
        CSV 行顺序刻意不同于 SAMPLE_KEYS，用于验证 FeatureProvider 必须按
        调用方传入的 ExpertBatch.sample_keys 对齐，而不是沿用文件顺序。
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
    """函数功能：FeatureProvider 阶段只允许读取测试内临时 feature CSV。"""
    original_open = builtins.open
    original_path_open = Path.open
    allowed_path = feature_csv_path.resolve()

    def checked_open(file: object, *args: object, **kwargs: object) -> object:
        path = Path(file).resolve()
        if path != allowed_path:
            raise AssertionError(f"TimeFuseFeatureCacheProvider 只能读取临时 feature CSV，不应读取：{path}")
        return original_open(file, *args, **kwargs)

    def checked_path_open(path_self: Path, *args: object, **kwargs: object) -> object:
        path = path_self.resolve()
        if path != allowed_path:
            raise AssertionError(f"TimeFuseFeatureCacheProvider 只能读取临时 feature CSV，不应读取：{path}")
        return original_path_open(path_self, *args, **kwargs)

    with patch.object(builtins, "open", side_effect=checked_open), patch.object(
        Path, "open", checked_path_open
    ), patch.object(np, "load", side_effect=fail_file_access):
        yield


@contextmanager
def forbid_head_and_evaluator_io() -> Iterator[None]:
    """
    函数功能：
        在 head/evaluator 阶段阻断常见文件读写 API。

    关键约束：
        ExpertProvider 和 FeatureProvider 已在进入该上下文前完成读取；如果
        TimeFuseLinearSoftmaxHead 或 EvaluationInputAdapter 回读 prediction
        cache、oracle/TSF、feature CSV 或尝试写 run artifact，这里会立即失败。
    """
    with patch.object(builtins, "open", side_effect=fail_file_access), patch.object(
        Path, "open", side_effect=fail_file_access
    ), patch.object(np, "load", side_effect=fail_file_access), patch.object(
        np, "save", side_effect=fail_training_side_effect
    ), patch.object(
        np, "savez", side_effect=fail_training_side_effect
    ):
        yield


def fail_file_access(*args: object, **kwargs: object) -> object:
    """函数功能：阻断链路中不允许发生的文件读取。"""
    raise AssertionError(f"TimeFuse protocol chain 当前阶段不应访问文件系统：args={args} kwargs={kwargs}")


def fail_training_side_effect(*args: object, **kwargs: object) -> object:
    """函数功能：阻断 checkpoint/run artifact 等非 smoke 行为。"""
    raise AssertionError(f"TimeFuse protocol chain 不应产生训练或运行产物：args={args} kwargs={kwargs}")


def assert_close(name: str, actual: float, expected: float, *, atol: float) -> None:
    """函数功能：带中文上下文的浮点 golden 断言。"""
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name} 不一致：actual={actual:.12f} expected={expected:.12f} atol={atol}")


def load_chain_inputs(fixture_root: Path, feature_csv_path: Path) -> tuple[ExpertBatch, FeatureBatch]:
    """函数功能：显式构造协议链前两段输入对象。"""
    expert_provider = PredictionCacheExpertProvider(fixture_root=fixture_root, model_columns=MODEL_COLUMNS)
    expert_batch = expert_provider.load_batch(SAMPLE_KEYS, verify_metrics=True)

    with allow_only_feature_csv_reads(feature_csv_path):
        feature_provider = TimeFuseFeatureCacheProvider(
            feature_csv_path=feature_csv_path,
            feature_columns=FEATURE_COLUMNS,
            feature_schema_name="timefuse_protocol_chain_smoke_v1",
        )
        feature_batch = feature_provider.load_batch(expert_batch.sample_keys)

    return expert_batch, feature_batch


def assert_chain_result(
    *,
    expert_batch: ExpertBatch,
    feature_batch: FeatureBatch,
    router_output: RouterOutput,
    result: EvaluationInputAdapterResult,
    atol: float,
) -> None:
    """函数功能：验证链路输出保序、对齐且 deterministic。"""
    if expert_batch.sample_keys != SAMPLE_KEYS:
        raise AssertionError(f"ExpertBatch sample_keys 顺序漂移：{expert_batch.sample_keys}")
    if expert_batch.model_columns != MODEL_COLUMNS:
        raise AssertionError(f"ExpertBatch model_columns 漂移：{expert_batch.model_columns}")
    if feature_batch.sample_keys != expert_batch.sample_keys:
        raise AssertionError(
            "FeatureBatch sample_keys 必须与 ExpertBatch 对齐："
            f"feature={feature_batch.sample_keys} expert={expert_batch.sample_keys}"
        )
    np.testing.assert_allclose(feature_batch.features, FEATURE_MATRIX.astype(np.float32), rtol=0.0, atol=atol)
    if feature_batch.feature_schema.get("feature_columns") != FEATURE_COLUMNS:
        raise AssertionError(f"FeatureBatch feature_columns 漂移：{feature_batch.feature_schema}")

    if router_output.sample_keys != expert_batch.sample_keys:
        raise AssertionError(f"RouterOutput sample_keys 未保序：{router_output.sample_keys}")
    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError(f"RouterOutput model_columns 未与 ExpertBatch 对齐：{router_output.model_columns}")
    np.testing.assert_allclose(router_output.logits, EXPECTED_LOGITS, rtol=0.0, atol=atol)
    np.testing.assert_allclose(router_output.weights, EXPECTED_WEIGHTS, rtol=0.0, atol=atol)
    np.testing.assert_allclose(np.sum(router_output.weights, axis=1), np.ones(len(SAMPLE_KEYS)), rtol=0.0, atol=atol)

    evaluation_input = result.evaluation_input
    if evaluation_input.sample_keys != SAMPLE_KEYS:
        raise AssertionError(f"EvaluationInput sample_keys 漂移：{evaluation_input.sample_keys}")
    if evaluation_input.model_columns != MODEL_COLUMNS:
        raise AssertionError(f"EvaluationInput model_columns 漂移：{evaluation_input.model_columns}")
    if evaluation_input.y_pred is not expert_batch.y_pred or evaluation_input.y_true is not expert_batch.y_true:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch.y_pred/y_true，不应复制或重读")
    if evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")

    summary = result.summary
    assert_close("summary hard_mae", float(summary["hard_mae"]), EXPECTED_SUMMARY["hard_mae"], atol=atol)
    assert_close("summary hard_mse", float(summary["hard_mse"]), EXPECTED_SUMMARY["hard_mse"], atol=atol)
    assert_close("summary raw_soft_mae", float(summary["raw_soft_mae"]), EXPECTED_SUMMARY["raw_soft_mae"], atol=atol)
    assert_close("summary raw_soft_mse", float(summary["raw_soft_mse"]), EXPECTED_SUMMARY["raw_soft_mse"], atol=atol)
    assert_close("summary mean_entropy", float(summary["mean_entropy"]), EXPECTED_SUMMARY["mean_entropy"], atol=atol)
    assert_close("summary mean_max_weight", float(summary["mean_max_weight"]), EXPECTED_SUMMARY["mean_max_weight"], atol=atol)
    if summary["selected_counts"] != EXPECTED_SUMMARY["selected_counts"]:
        raise AssertionError(f"summary selected_counts 漂移：{summary['selected_counts']}")
    if summary["model_columns"] != list(MODEL_COLUMNS):
        raise AssertionError(f"summary model_columns 漂移：{summary['model_columns']}")
    if summary["num_samples"] != len(SAMPLE_KEYS) or summary["num_experts"] != len(MODEL_COLUMNS):
        raise AssertionError(f"summary 样本/专家数漂移：{summary}")

    rows = result.per_sample_rows
    if [row["sample_key"] for row in rows] != list(SAMPLE_KEYS):
        raise AssertionError(f"rows sample_key 顺序漂移：{rows}")
    if [row["selected_model"] for row in rows] != EXPECTED_SELECTED_MODELS:
        raise AssertionError(f"rows selected_model 漂移：{rows}")
    if [row["selected_index"] for row in rows] != EXPECTED_SELECTED_INDICES:
        raise AssertionError(f"rows selected_index 漂移：{rows}")
    np.testing.assert_allclose([row["hard_mae"] for row in rows], EXPECTED_ROW_HARD_MAE, rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["hard_mse"] for row in rows], EXPECTED_ROW_HARD_MSE, rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["raw_soft_mae"] for row in rows], EXPECTED_ROW_RAW_SOFT_MAE, rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["raw_soft_mse"] for row in rows], EXPECTED_ROW_RAW_SOFT_MSE, rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["max_weight"] for row in rows], np.max(EXPECTED_WEIGHTS, axis=1), rtol=0.0, atol=atol)
    np.testing.assert_allclose([row["weight_entropy"] for row in rows], compute_weight_entropy(EXPECTED_WEIGHTS), rtol=0.0, atol=atol)


def run_smoke(fixture_root: Path, *, atol: float) -> None:
    """函数功能：执行 TimeFuse protocol chain 组合性 smoke。"""
    fixture_root = fixture_root.resolve()
    print(f"开始 Stage 1 TimeFuse protocol chain smoke：fixture_root={fixture_root}")
    if str(fixture_root).startswith("/data2/"):
        raise AssertionError("protocol chain smoke 不应访问 /data2 fixture")
    before_dirs = snapshot_run_outputs()

    with TemporaryDirectory(prefix="stage1_timefuse_protocol_chain_") as tmp_dir:
        feature_csv_path = Path(tmp_dir) / "timefuse_protocol_chain_features.csv"
        write_feature_csv(feature_csv_path)
        if str(feature_csv_path.resolve()).startswith("/data2/"):
            raise AssertionError("protocol chain smoke 不应在 /data2 构造临时 feature CSV")

        expert_batch, feature_batch = load_chain_inputs(fixture_root, feature_csv_path)
        print("通过：PredictionCacheExpertProvider 与 TimeFuseFeatureCacheProvider 显式构造 ExpertBatch/FeatureBatch")

        head = TimeFuseLinearSoftmaxHead(weight=HEAD_WEIGHT, bias=HEAD_BIAS)
        adapter = EvaluationInputAdapter()
        with forbid_head_and_evaluator_io():
            router_output = head.predict(feature_batch, expert_batch.model_columns)
            result = adapter.evaluate(expert_batch=expert_batch, router_output=router_output)
        print("通过：head/evaluator 阶段未调用文件 IO、np.load 或 np.save，不回读 cache、不写产物")

        assert_chain_result(
            expert_batch=expert_batch,
            feature_batch=feature_batch,
            router_output=router_output,
            result=result,
            atol=atol,
        )
        print(
            "通过：链路输出保序且 deterministic，"
            f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
        )

    after_dirs = snapshot_run_outputs()
    if after_dirs != before_dirs:
        raise AssertionError(f"protocol chain smoke 不应创建输出目录：新增={sorted(after_dirs - before_dirs)}")
    print("完成：Stage 1 TimeFuse protocol chain smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    args = parse_args()
    run_smoke(args.fixture_root, atol=float(args.atol))


if __name__ == "__main__":
    main()
