#!/usr/bin/env python3
"""
文件功能：
    Stage 1 重构前 golden smoke。该脚本只读已有极小 packed cache fixture，
    用来锁定 sample_key 顺序、五专家顺序、数组 shape、hard top-1、raw soft
    fusion 指标，以及 packed_npy_v1 row index 读取行为。

输入：
    --fixture-root 指向已有 merged_cache 目录，默认使用 2026-06-14 的 4 sample
    full-scale dry-run packed fixture。

输出：
    标准输出打印中文检查日志；任一 golden 契约不一致时抛出 AssertionError。

关键约束：
    该脚本不生成预测、不训练 router、不写正式输出目录，只读取 fixture 现有
    manifest 和 npy 数组，适合作为 Stage 1 后续公共 batch reader 重构的等价门禁。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.common.prediction_array_io import (  # noqa: E402
    load_prediction_array,
    resolve_cache_array_path,
)


DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "experiment_logs"
    / "run_outputs"
    / "2026-06-14_stage1_full_scale_dry_run_v2"
    / "merged_cache"
)
FUSION_UTILS_PATH = (
    REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "fusion_utils.py"
)
EXPECTED_MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
EXPECTED_MANIFEST_MODEL_ORDER = ["CrossFormer", "DLinear", "ES", "NaiveForecaster", "PatchTST"]
EXPECTED_SAMPLE_KEYS = [
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873",
]
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
    """函数功能：解析 smoke fixture 路径和数值容忍度参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 golden smoke checks.")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="只读 merged_cache fixture 目录，需包含 manifest.csv 和 packed npy 数组。",
    )
    parser.add_argument("--atol", type=float, default=1e-6, help="MAE/MSE 浮点比对绝对容忍度。")
    return parser.parse_args()


def read_model_columns_from_source(source_path: Path) -> List[str]:
    """
    函数功能：
        直接从 fusion_utils.py 源码解析 MODEL_COLUMNS，避免为了一个 smoke 导入
        torch/sklearn 等训练依赖，同时仍能锁定正式专家顺序是否漂移。
    """
    text = source_path.read_text(encoding="utf-8")
    match = re.search(r"^MODEL_COLUMNS\s*=\s*(\[[^\n]+\])", text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"未能在 {source_path} 中解析 MODEL_COLUMNS")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"MODEL_COLUMNS 不是字符串列表：{value!r}")
    return value


def assert_close(name: str, actual: float, expected: float, *, atol: float) -> None:
    """函数功能：带中文上下文的浮点 golden 断言。"""
    if not np.isclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name} 不一致：actual={actual:.12f} expected={expected:.12f} atol={atol}")


def load_manifest(fixture_root: Path) -> pd.DataFrame:
    """函数功能：读取并校验极小 golden fixture manifest。"""
    manifest_path = fixture_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 golden fixture manifest：{manifest_path}")
    df = pd.read_csv(manifest_path)
    required_cols = {
        "sample_key",
        "model_name",
        "y_true_path",
        "y_pred_path",
        "array_storage",
        "y_true_row_index",
        "y_pred_row_index",
        "mae",
        "mse",
    }
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise AssertionError(f"manifest 缺少字段：{missing}")
    if len(df) != len(EXPECTED_SAMPLE_KEYS) * len(EXPECTED_MODEL_COLUMNS):
        raise AssertionError(f"manifest 行数不一致：actual={len(df)}")
    if set(df["array_storage"].astype(str)) != {"packed_npy_v1"}:
        raise AssertionError("golden fixture 必须使用 packed_npy_v1")
    return df


def make_absolute_record(row: Mapping[str, object], fixture_root: Path) -> Dict[str, object]:
    """函数功能：把 manifest 相对数组路径转换为 load_prediction_array 可读取的绝对路径。"""
    record = dict(row)
    record["y_true_path"] = str(resolve_cache_array_path(str(record["y_true_path"]), fixture_root))
    record["y_pred_path"] = str(resolve_cache_array_path(str(record["y_pred_path"]), fixture_root))
    return record


def assemble_prediction_batch(
    manifest_df: pd.DataFrame,
    fixture_root: Path,
    model_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    函数功能：
        按 sample_key golden 顺序和 MODEL_COLUMNS golden 顺序组装 batch。

    关键约束：
        manifest 中每个 sample 的原始专家行顺序不是正式动作空间顺序，本函数明确
        验证并重排到 MODEL_COLUMNS，防止未来公共 reader 误用 CSV 当前行顺序。
    """
    y_pred_batch: List[np.ndarray] = []
    y_true_batch: List[np.ndarray] = []
    for sample_key in EXPECTED_SAMPLE_KEYS:
        group = manifest_df[manifest_df["sample_key"].astype(str) == sample_key]
        manifest_order = group["model_name"].astype(str).tolist()
        if manifest_order != EXPECTED_MANIFEST_MODEL_ORDER:
            raise AssertionError(f"manifest 原始专家顺序漂移：sample_key={sample_key} order={manifest_order}")

        expected_true_row_index, expected_pred_row_index = EXPECTED_ROW_INDICES[sample_key]
        sample_predictions: List[np.ndarray] = []
        shared_y_true: np.ndarray | None = None
        for model_name in model_columns:
            model_rows = group[group["model_name"].astype(str) == model_name]
            if len(model_rows) != 1:
                raise AssertionError(f"sample/model 唯一性不满足：sample_key={sample_key} model={model_name}")
            record = make_absolute_record(model_rows.iloc[0].to_dict(), fixture_root)
            if int(record["y_true_row_index"]) != expected_true_row_index:
                raise AssertionError(f"y_true_row_index 漂移：sample_key={sample_key} model={model_name}")
            if int(record["y_pred_row_index"]) != expected_pred_row_index:
                raise AssertionError(f"y_pred_row_index 漂移：sample_key={sample_key} model={model_name}")

            y_true = load_prediction_array(record, "y_true")
            y_pred = load_prediction_array(record, "y_pred")
            if y_true.shape != (48, 1) or y_pred.shape != (48, 1):
                raise AssertionError(
                    f"数组 shape 不一致：sample_key={sample_key} model={model_name} "
                    f"y_true={y_true.shape} y_pred={y_pred.shape}"
                )
            if shared_y_true is None:
                shared_y_true = y_true
            elif not np.array_equal(shared_y_true, y_true):
                raise AssertionError(f"同一 sample_key 的五专家 y_true 不一致：{sample_key}")

            # 这里复算单专家指标，确保 row index 读取到的窗口与 manifest 行一致。
            error = y_pred - y_true
            assert_close(
                f"{sample_key}/{model_name} mae",
                float(np.mean(np.abs(error))),
                float(record["mae"]),
                atol=1e-6,
            )
            assert_close(
                f"{sample_key}/{model_name} mse",
                float(np.mean(error**2)),
                float(record["mse"]),
                atol=1e-6,
            )
            sample_predictions.append(y_pred)
        if shared_y_true is None:
            raise AssertionError(f"sample_key 无 y_true：{sample_key}")
        y_pred_batch.append(np.stack(sample_predictions, axis=0))
        y_true_batch.append(shared_y_true)
    return np.stack(y_pred_batch, axis=0), np.stack(y_true_batch, axis=0)


def run_smoke(fixture_root: Path, *, atol: float) -> None:
    """函数功能：执行全部 Stage 1 golden smoke 检查并输出中文日志。"""
    fixture_root = fixture_root.resolve()
    print(f"开始 Stage 1 golden smoke：fixture_root={fixture_root}")

    model_columns = read_model_columns_from_source(FUSION_UTILS_PATH)
    if model_columns != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(f"五专家顺序漂移：actual={model_columns} expected={EXPECTED_MODEL_COLUMNS}")
    print(f"通过：五专家顺序固定为 {model_columns}")

    manifest_df = load_manifest(fixture_root)
    sample_keys = manifest_df["sample_key"].drop_duplicates().astype(str).tolist()
    if sample_keys != EXPECTED_SAMPLE_KEYS:
        raise AssertionError(f"sample_key 顺序漂移：actual={sample_keys} expected={EXPECTED_SAMPLE_KEYS}")
    print(f"通过：sample_key 顺序固定，sample_count={len(sample_keys)}")

    y_pred, y_true = assemble_prediction_batch(manifest_df, fixture_root, model_columns)
    if tuple(y_pred.shape) != EXPECTED_Y_PRED_SHAPE:
        raise AssertionError(f"y_pred shape 漂移：actual={y_pred.shape} expected={EXPECTED_Y_PRED_SHAPE}")
    if tuple(y_true.shape) != EXPECTED_Y_TRUE_SHAPE:
        raise AssertionError(f"y_true shape 漂移：actual={y_true.shape} expected={EXPECTED_Y_TRUE_SHAPE}")
    print(f"通过：y_pred shape={y_pred.shape}，y_true shape={y_true.shape}")

    hard_indices = GOLDEN_WEIGHTS.argmax(axis=1)
    hard_models = [model_columns[int(idx)] for idx in hard_indices]
    if hard_models != EXPECTED_HARD_MODELS:
        raise AssertionError(f"hard top-1 选择漂移：actual={hard_models} expected={EXPECTED_HARD_MODELS}")
    hard_pred = y_pred[np.arange(len(EXPECTED_SAMPLE_KEYS)), hard_indices]
    hard_error = hard_pred - y_true
    hard_mae = float(np.mean(np.abs(hard_error)))
    hard_mse = float(np.mean(hard_error**2))
    assert_close("hard top-1 MAE", hard_mae, EXPECTED_HARD_MAE, atol=atol)
    assert_close("hard top-1 MSE", hard_mse, EXPECTED_HARD_MSE, atol=atol)
    print(f"通过：hard top-1 选择={hard_models}，MAE={hard_mae:.9f}，MSE={hard_mse:.9f}")

    soft_pred = (GOLDEN_WEIGHTS.reshape(GOLDEN_WEIGHTS.shape[0], GOLDEN_WEIGHTS.shape[1], 1, 1) * y_pred).sum(axis=1)
    soft_error = soft_pred - y_true
    raw_soft_mae = float(np.mean(np.abs(soft_error)))
    raw_soft_mse = float(np.mean(soft_error**2))
    assert_close("raw soft fusion MAE", raw_soft_mae, EXPECTED_RAW_SOFT_MAE, atol=atol)
    assert_close("raw soft fusion MSE", raw_soft_mse, EXPECTED_RAW_SOFT_MSE, atol=atol)
    print(f"通过：raw soft fusion MAE={raw_soft_mae:.9f}，MSE={raw_soft_mse:.9f}")

    print("完成：Stage 1 golden smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    args = parse_args()
    run_smoke(args.fixture_root, atol=float(args.atol))


if __name__ == "__main__":
    main()
