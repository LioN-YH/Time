#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P7a TimeFuseFeatureCacheProvider smoke。

输入：
    使用测试内临时 feature CSV，显式传入 sample_keys。

输出：
    标准输出打印中文检查日志；任一 provider contract 漂移时抛出
    AssertionError。

关键约束：
    该 smoke 不读取 prediction/oracle，不访问 /data2，不创建正式输出目录，
    不做 scaler fit，不接正式 TimeFuse fusor 或 Visual Router 入口。
"""

from __future__ import annotations

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

from time_router.features import TimeFuseFeatureCacheProvider  # noqa: E402
from time_router.protocols import FeatureBatch  # noqa: E402


FEATURE_COLUMNS = tuple(f"timefuse_feature_{index:02d}" for index in range(17))
CSV_SAMPLE_KEYS = ("sample_c", "sample_a", "sample_b")
REQUESTED_SAMPLE_KEYS = ("sample_b", "sample_c")
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于检查 provider 不创建输出目录。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def write_feature_csv(feature_csv_path: Path) -> None:
    """函数功能：写入只含 sample_key 和 17 维 TimeFuse 特征的临时 fixture。"""
    with feature_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_key", *FEATURE_COLUMNS))
        writer.writeheader()
        for row_index, sample_key in enumerate(CSV_SAMPLE_KEYS):
            row = {"sample_key": sample_key}
            for feature_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = f"{row_index * 100 + feature_index:.1f}"
            writer.writerow(row)


@contextmanager
def allow_only_feature_csv_reads(feature_csv_path: Path) -> Iterator[None]:
    """
    函数功能：
        在 provider 阶段只允许读取临时 feature CSV。

    关键约束：
        如果 provider 尝试读取 prediction cache、oracle/TSF、npy 或其他路径，
        这里会立即失败，证明 P7a adapter 只读 feature。
    """

    original_open = builtins.open
    original_path_open = Path.open
    allowed_path = feature_csv_path.resolve()

    def checked_open(file: object, *args: object, **kwargs: object) -> object:
        path = Path(file).resolve()
        if path != allowed_path:
            raise AssertionError(f"TimeFuseFeatureCacheProvider 只能读取 feature CSV，不应读取：{path}")
        return original_open(file, *args, **kwargs)

    def checked_path_open(path_self: Path, *args: object, **kwargs: object) -> object:
        path = path_self.resolve()
        if path != allowed_path:
            raise AssertionError(f"TimeFuseFeatureCacheProvider 只能读取 feature CSV，不应读取：{path}")
        return original_path_open(path_self, *args, **kwargs)

    def fail_np_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("TimeFuseFeatureCacheProvider 不应调用 np.load 读取 prediction cache")

    with patch.object(builtins, "open", side_effect=checked_open), patch.object(
        Path, "open", checked_path_open
    ), patch.object(np, "load", side_effect=fail_np_load):
        yield


def run_smoke() -> None:
    """函数功能：执行 TimeFuseFeatureCacheProvider 最小 contract smoke。"""
    print("开始 Stage 1 TimeFuseFeatureCacheProvider smoke")
    before_outputs = snapshot_run_outputs()

    with TemporaryDirectory(prefix="stage1_timefuse_feature_provider_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        feature_csv_path = tmp_path / "timefuse_features.csv"
        write_feature_csv(feature_csv_path)
        if str(feature_csv_path.resolve()).startswith("/data2/"):
            raise AssertionError("provider smoke 不应访问 /data2 fixture")

        with allow_only_feature_csv_reads(feature_csv_path):
            provider = TimeFuseFeatureCacheProvider(
                feature_csv_path=feature_csv_path,
                feature_columns=FEATURE_COLUMNS,
                feature_schema_name="timefuse_single_variable_meta_v1",
            )
            feature_batch = provider.load_batch(REQUESTED_SAMPLE_KEYS)

        if not isinstance(feature_batch, FeatureBatch):
            raise AssertionError(f"provider 未返回 FeatureBatch：actual={type(feature_batch)!r}")
        if feature_batch.sample_keys != REQUESTED_SAMPLE_KEYS:
            raise AssertionError(f"FeatureBatch sample_keys 未保序：actual={feature_batch.sample_keys}")
        if tuple(feature_batch.features.shape) != (len(REQUESTED_SAMPLE_KEYS), len(FEATURE_COLUMNS)):
            raise AssertionError(f"features shape 漂移：actual={feature_batch.features.shape}")
        if feature_batch.features.dtype != np.float32:
            raise AssertionError(f"features dtype 漂移：actual={feature_batch.features.dtype}")

        expected_features = np.asarray(
            [
                [200.0 + feature_index for feature_index in range(17)],
                [0.0 + feature_index for feature_index in range(17)],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(feature_batch.features, expected_features, rtol=0.0, atol=0.0)
        print(f"通过：FeatureBatch 保序且 shape 正确，features={feature_batch.features.shape}")

        schema = feature_batch.feature_schema
        if schema.get("feature_schema_name") != "timefuse_single_variable_meta_v1":
            raise AssertionError(f"feature_schema_name 漂移：{schema}")
        if schema.get("feature_columns") != FEATURE_COLUMNS:
            raise AssertionError(f"feature_columns 漂移：{schema.get('feature_columns')}")
        if schema.get("feature_dim") != len(FEATURE_COLUMNS):
            raise AssertionError(f"feature_dim 漂移：{schema.get('feature_dim')}")
        if schema.get("source") != str(feature_csv_path):
            raise AssertionError(f"source 漂移：{schema.get('source')}")

        extra = feature_batch.extra
        if extra.get("provider_name") != "TimeFuseFeatureCacheProvider":
            raise AssertionError(f"provider_name metadata 漂移：{extra}")
        if extra.get("sample_key_column") != "sample_key":
            raise AssertionError(f"sample_key_column metadata 漂移：{extra}")
        if extra.get("num_available_rows") != len(CSV_SAMPLE_KEYS):
            raise AssertionError(f"num_available_rows metadata 漂移：{extra}")
        if extra.get("dtype") != "float32":
            raise AssertionError(f"dtype metadata 漂移：{extra}")
        print("通过：feature_schema 和 extra 只记录 feature lineage 与 provider metadata")

        for invalid_keys, expected_message in (
            ((), "非空 sample_keys"),
            ((REQUESTED_SAMPLE_KEYS[0], REQUESTED_SAMPLE_KEYS[0]), "重复 sample_key"),
        ):
            provider = TimeFuseFeatureCacheProvider(feature_csv_path=feature_csv_path, feature_columns=FEATURE_COLUMNS)
            try:
                provider.load_batch(invalid_keys)
            except ValueError as exc:
                if expected_message not in str(exc):
                    raise AssertionError(f"provider 拒绝非法 sample_keys 的错误信息不清晰：{exc}") from exc
            else:
                raise AssertionError(f"provider 应拒绝非法 sample_keys：{invalid_keys}")
        print("通过：provider 拒绝空 sample_keys 和重复 sample_key，不默认扫描全量 batch")

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"provider smoke 不应创建输出目录：新增={sorted(after_outputs - before_outputs)}")
    print("通过：provider 不读取 prediction/oracle，不创建输出目录，不写运行产物")
    print("完成：Stage 1 TimeFuseFeatureCacheProvider smoke 全部通过")


def main() -> None:
    """函数功能：脚本入口。"""
    run_smoke()


if __name__ == "__main__":
    main()
