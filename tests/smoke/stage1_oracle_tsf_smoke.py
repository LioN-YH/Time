#!/usr/bin/env python3
"""
文件功能：
    Stage 1 oracle/TSF reader 小规模 smoke。该脚本只读已有 dry-run fixture，
    验证 OracleTsfReader 的 sample_key 保序、oracle/TSF join、默认禁止全扫描、
    missing_policy 缺失处理、重复检测和缺失报告。

关键约束：
    该 smoke 不训练 router、不生成 oracle/TSF、不改正式输出目录，只验证共享
    reader 的读取契约。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.data.oracle_tsf_reader import OracleTsfReader  # noqa: E402


DEFAULT_FIXTURE_ROOT = (
    REPO_ROOT
    / "experiment_logs"
    / "run_outputs"
    / "2026-06-14_stage1_full_scale_dry_run_v2"
    / "merged_cache"
)
EXPECTED_SAMPLE_KEYS = [
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873",
]
EXPECTED_ORACLE_MODELS = {
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win50": "PatchTST",
    "96_48_S__test__TEST_DATA_HOUR__item100388__ch0__win151": "DLinear",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win291": "DLinear",
    "96_48_S__vali__TEST_DATA_HOUR__item100388__ch0__win873": "DLinear",
}


def parse_args() -> argparse.Namespace:
    """函数功能：解析只读 fixture 路径。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 oracle/TSF reader smoke checks.")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="只读 merged_cache fixture 目录，需包含 window_oracle_labels_with_tsf_cell.csv 和 manifest_with_tsf_cell.csv。",
    )
    return parser.parse_args()


def assert_sample_order(frame: pd.DataFrame, expected_keys: list[str], context: str) -> None:
    """函数功能：检查 DataFrame 第一维 sample_key 顺序。"""
    actual_keys = frame["sample_key"].astype(str).tolist()
    if actual_keys != expected_keys:
        raise AssertionError(f"{context} sample_key 顺序漂移：actual={actual_keys} expected={expected_keys}")


def run_smoke(fixture_root: Path) -> None:
    """函数功能：执行 OracleTsfReader 的只读契约检查。"""
    fixture_root = fixture_root.resolve()
    print(f"开始 Stage 1 oracle/TSF smoke：fixture_root={fixture_root}")

    reader = OracleTsfReader(fixture_root=fixture_root, missing_policy="error")
    reversed_keys = list(reversed(EXPECTED_SAMPLE_KEYS))

    # 保护边界：正式入口必须显式传入 batch/shard sample_key，默认不允许无 key 全扫描。
    try:
        reader.load_oracle(None, metric="mae")
    except ValueError as exc:
        if "默认禁止全扫描" not in str(exc):
            raise AssertionError(f"allow_full_scan 默认禁止的错误信息异常：{exc}") from exc
    else:
        raise AssertionError("allow_full_scan 默认 False 时，load_oracle(None) 未被禁止")
    print("通过：allow_full_scan 默认禁止无 sample_key 全扫描")

    oracle = reader.load_oracle(reversed_keys, metric="mae")
    assert_sample_order(oracle.frame, reversed_keys, "oracle")
    oracle_models = dict(zip(oracle.frame["sample_key"], oracle.frame["oracle_model"]))
    for sample_key in reversed_keys:
        if oracle_models[sample_key] != EXPECTED_ORACLE_MODELS[sample_key]:
            raise AssertionError(
                f"oracle_model 漂移：sample_key={sample_key} "
                f"actual={oracle_models[sample_key]} expected={EXPECTED_ORACLE_MODELS[sample_key]}"
            )
    print("通过：oracle 按显式 sample_key 保序并匹配预期 label")

    tsf = reader.load_tsf(reversed_keys)
    assert_sample_order(tsf.frame, reversed_keys, "TSF")
    if set(tsf.frame["group_name"].astype(str)) != {"HIGH_HIGH_HIGH"}:
        raise AssertionError(f"TSF group_name 漂移：{tsf.frame['group_name'].tolist()}")
    print("通过：TSF enrichment 按显式 sample_key 保序且字段完整")

    joined = reader.load_joined(reversed_keys, metric="mae")
    assert_sample_order(joined.frame, reversed_keys, "joined")
    required_joined_cols = {"sample_key", "oracle_model", "oracle_value", "group_name", "forecastability_cat"}
    missing_cols = sorted(required_joined_cols.difference(joined.frame.columns))
    if missing_cols:
        raise AssertionError(f"joined 输出缺少字段：{missing_cols}")
    if joined.frame["sample_key"].duplicated().any():
        raise AssertionError("joined 输出存在重复 sample_key")
    print("通过：oracle/TSF join 不重复、不丢失，并保留 metadata 字段")

    missing_key = "missing__sample_key__for_oracle_tsf_reader_smoke"
    try:
        reader.load_joined([EXPECTED_SAMPLE_KEYS[0], missing_key], metric="mae")
    except KeyError as exc:
        if missing_key not in str(exc):
            raise AssertionError(f"missing_policy=error 缺失错误信息未包含 sample_key：{exc}") from exc
    else:
        raise AssertionError("missing_policy=error 未对缺失 sample_key 报错")
    print("通过：missing_policy=error 对缺失 sample_key 明确报错")

    report_reader = OracleTsfReader(fixture_root=fixture_root, missing_policy="report")
    reported = report_reader.load_joined([EXPECTED_SAMPLE_KEYS[0], missing_key], metric="mae")
    if reported.missing_report["oracle"]["missing_sample_keys"] != [missing_key]:
        raise AssertionError(f"oracle missing report 异常：{reported.missing_report}")
    if reported.missing_report["tsf"]["missing_sample_keys"] != [missing_key]:
        raise AssertionError(f"TSF missing report 异常：{reported.missing_report}")
    assert_sample_order(reported.frame, [EXPECTED_SAMPLE_KEYS[0], missing_key], "missing-report joined")
    print("通过：missing_policy=report 明确记录缺失 sample_key 且不打乱顺序")

    with tempfile.TemporaryDirectory(prefix="stage1_oracle_tsf_smoke_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        shutil.copy2(fixture_root / "window_oracle_labels_with_tsf_cell.csv", tmp_root / "window_oracle_labels_with_tsf_cell.csv")
        original_tsf = pd.read_csv(fixture_root / "manifest_with_tsf_cell.csv")
        conflict_row = original_tsf.head(1).copy()
        conflict_row.loc[:, "group_name"] = "CONFLICTING_TSF_CELL"
        duplicate_tsf = pd.concat([original_tsf, conflict_row], ignore_index=True)
        duplicate_tsf.to_csv(tmp_root / "manifest_with_tsf_cell.csv", index=False)
        duplicate_reader = OracleTsfReader(fixture_root=tmp_root, missing_policy="error")
        try:
            duplicate_reader.load_tsf([EXPECTED_SAMPLE_KEYS[0]])
        except ValueError as exc:
            if "重复 sample_key" not in str(exc) and "冲突元信息" not in str(exc):
                raise AssertionError(f"重复检测错误信息异常：{exc}") from exc
        else:
            raise AssertionError("冲突 TSF sample_key 未触发错误")
    print("通过：TSF 冲突重复 sample_key 会明确失败")

    print("完成：Stage 1 oracle/TSF smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    args = parse_args()
    run_smoke(args.fixture_root)


if __name__ == "__main__":
    main()
