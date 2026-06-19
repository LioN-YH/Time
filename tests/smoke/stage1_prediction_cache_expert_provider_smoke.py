#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P6a PredictionCacheExpertProvider smoke。

输入：
    默认读取 2026-06-14 的 4 sample packed golden fixture，并显式传入
    golden sample_keys。

输出：
    标准输出打印中文检查日志；任一 provider contract 或 golden 指标漂移时
    抛出 AssertionError。

关键约束：
    该 smoke 不创建正式输出目录，不访问 /data2，不训练 router，不读取
    oracle/TSF，只验证 PredictionBatchReader -> ExpertBatch 的最小适配边界。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import hard_top1_fusion, raw_soft_fusion  # noqa: E402
from time_router.experts import PredictionCacheExpertProvider  # noqa: E402
from time_router.io import DEFAULT_MODEL_COLUMNS  # noqa: E402
from time_router.protocols import ExpertBatch  # noqa: E402


DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "experiment_logs"
    / "run_outputs"
    / "2026-06-14_stage1_full_scale_dry_run_v2"
    / "merged_cache"
)
EXPECTED_MODEL_COLUMNS = tuple(DEFAULT_MODEL_COLUMNS)
EXPECTED_SAMPLE_KEYS = (
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873",
)
EXPECTED_Y_PRED_SHAPE = (4, 5, 48, 1)
EXPECTED_Y_TRUE_SHAPE = (4, 48, 1)
EXPECTED_HARD_MODELS = ["CrossFormer", "DLinear", "PatchTST", "DLinear"]
EXPECTED_HARD_MAE = 0.41604843735694885
EXPECTED_HARD_MSE = 0.4563697576522827
EXPECTED_RAW_SOFT_MAE = 0.4102966785430908
EXPECTED_RAW_SOFT_MSE = 0.48815402388572693
EXPECTED_ROW_INDICES: Dict[str, Tuple[int, int]] = {
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50": (1, 0),
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151": (0, 0),
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291": (0, 0),
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873": (1, 0),
}
GOLDEN_WEIGHTS = np.asarray(
    [
        [0.10, 0.20, 0.55, 0.05, 0.10],
        [0.65, 0.10, 0.15, 0.05, 0.05],
        [0.05, 0.70, 0.10, 0.10, 0.05],
        [0.30, 0.30, 0.20, 0.10, 0.10],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    """函数功能：解析只读 fixture 路径和浮点容忍度。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 PredictionCacheExpertProvider smoke.")
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


def run_smoke(fixture_root: Path, *, atol: float) -> None:
    """函数功能：执行 PredictionCacheExpertProvider 最小 contract smoke。"""
    fixture_root = fixture_root.resolve()
    print(f"开始 Stage 1 PredictionCacheExpertProvider smoke：fixture_root={fixture_root}")
    if str(fixture_root).startswith("/data2/"):
        raise AssertionError("provider smoke 不应访问 /data2 fixture")

    provider = PredictionCacheExpertProvider(fixture_root=fixture_root, model_columns=EXPECTED_MODEL_COLUMNS)
    for invalid_keys, expected_message in (
        ((), "非空 sample_keys"),
        ((EXPECTED_SAMPLE_KEYS[0], EXPECTED_SAMPLE_KEYS[0]), "重复 sample_key"),
    ):
        try:
            provider.load_batch(invalid_keys)
        except ValueError as exc:
            if expected_message not in str(exc):
                raise AssertionError(f"provider 拒绝非法 sample_keys 的错误信息不清晰：{exc}") from exc
        else:
            raise AssertionError(f"provider 应拒绝非法 sample_keys：{invalid_keys}")
    print("通过：provider 拒绝空 sample_keys 和重复 sample_key，不默认扫描全量 manifest")

    expert_batch = provider.load_batch(EXPECTED_SAMPLE_KEYS, verify_metrics=True)
    if not isinstance(expert_batch, ExpertBatch):
        raise AssertionError(f"provider 未返回 ExpertBatch：actual={type(expert_batch)!r}")

    if expert_batch.sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"ExpertBatch sample_keys 顺序漂移：actual={expert_batch.sample_keys}")
    if expert_batch.model_columns != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(
            f"ExpertBatch model_columns 漂移：actual={expert_batch.model_columns} expected={EXPECTED_MODEL_COLUMNS}"
        )
    if tuple(expert_batch.y_pred.shape) != EXPECTED_Y_PRED_SHAPE:
        raise AssertionError(f"ExpertBatch y_pred shape 漂移：actual={expert_batch.y_pred.shape}")
    if tuple(expert_batch.y_true.shape) != EXPECTED_Y_TRUE_SHAPE:
        raise AssertionError(f"ExpertBatch y_true shape 漂移：actual={expert_batch.y_true.shape}")
    print(f"通过：ExpertBatch 保序且 shape 正确，y_pred={expert_batch.y_pred.shape}，y_true={expert_batch.y_true.shape}")

    row_index_metadata = expert_batch.row_index_metadata
    if not isinstance(row_index_metadata, dict):
        raise AssertionError("ExpertBatch row_index_metadata 必须包含 packed row index dict")
    for sample_key in EXPECTED_SAMPLE_KEYS:
        if sample_key not in row_index_metadata:
            raise AssertionError(f"row_index_metadata 缺少 sample_key：{sample_key}")
        for model_name in EXPECTED_MODEL_COLUMNS:
            if model_name not in row_index_metadata[sample_key]:
                raise AssertionError(f"row_index_metadata 缺少专家：sample_key={sample_key} model={model_name}")
            actual_indices = tuple(row_index_metadata[sample_key][model_name])
            if actual_indices != EXPECTED_ROW_INDICES[sample_key]:
                raise AssertionError(
                    f"packed row index 漂移：sample_key={sample_key} model={model_name} "
                    f"actual={actual_indices} expected={EXPECTED_ROW_INDICES[sample_key]}"
                )
    print("通过：row_index_metadata 保留 packed_npy_v1 y_true/y_pred row index 信息")

    extra = expert_batch.extra
    if extra.get("provider_name") != "PredictionCacheExpertProvider":
        raise AssertionError(f"provider_name metadata 漂移：{extra.get('provider_name')}")
    if extra.get("array_storage") != "packed_npy_v1":
        raise AssertionError(f"array_storage metadata 漂移：{extra.get('array_storage')}")
    reader_metadata = extra.get("reader_metadata")
    if not isinstance(reader_metadata, dict):
        raise AssertionError("reader_metadata 必须是 dict")
    if reader_metadata.get("manifest_row_count") != len(EXPECTED_SAMPLE_KEYS) * len(EXPECTED_MODEL_COLUMNS):
        raise AssertionError(f"manifest_row_count 漂移：{reader_metadata.get('manifest_row_count')}")
    if reader_metadata.get("verify_metrics") is not True:
        raise AssertionError("reader_metadata 必须记录 verify_metrics=True")
    print("通过：ExpertBatch.extra 包含 provider name、array_storage 和轻量 reader metadata")

    hard_result = hard_top1_fusion(
        y_pred=expert_batch.y_pred,
        y_true=expert_batch.y_true,
        weights=GOLDEN_WEIGHTS,
        model_columns=expert_batch.model_columns,
    )
    if hard_result.selected_models != EXPECTED_HARD_MODELS:
        raise AssertionError(f"hard top-1 选择漂移：actual={hard_result.selected_models}")
    assert_close("hard top-1 MAE", hard_result.mae, EXPECTED_HARD_MAE, atol=atol)
    assert_close("hard top-1 MSE", hard_result.mse, EXPECTED_HARD_MSE, atol=atol)

    raw_soft_result = raw_soft_fusion(
        y_pred=expert_batch.y_pred,
        y_true=expert_batch.y_true,
        weights=GOLDEN_WEIGHTS,
        model_columns=expert_batch.model_columns,
    )
    assert_close("raw soft fusion MAE", raw_soft_result.mae, EXPECTED_RAW_SOFT_MAE, atol=atol)
    assert_close("raw soft fusion MSE", raw_soft_result.mse, EXPECTED_RAW_SOFT_MSE, atol=atol)
    print(
        "通过：time_router.evaluation public API 复算 golden 指标，"
        f"hard_mae={hard_result.mae:.9f}，raw_soft_mae={raw_soft_result.mae:.9f}"
    )

    print("完成：Stage 1 PredictionCacheExpertProvider smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    args = parse_args()
    run_smoke(args.fixture_root, atol=float(args.atol))


if __name__ == "__main__":
    main()
