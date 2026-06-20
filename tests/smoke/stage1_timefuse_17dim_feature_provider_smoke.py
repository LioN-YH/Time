#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P13e TimeFuse 17 维 FeatureProvider small smoke。

输入：
    使用 P13b real-derived manifest 的 ordered sample_keys，以及仓库内
    `tests/fixtures/stage1_timefuse_17dim_small/features_17d.csv` 小型 17 维
    TimeFuse-style feature CSV。

输出：
    标准输出打印中文检查日志；若 `TimeFuseFeatureCacheProvider` 输出的
    `FeatureBatch` 在 sample_key 保序、feature shape、schema metadata 或数值上
    漂移，则抛出 AssertionError。

关键约束：
    本 smoke 只验证 TimeFuse FeatureProvider -> FeatureBatch，不接
    TimeFuseLinearSoftmaxHead，不接 EvaluationInputAdapter，不写 canonical run_dir，
    不扩展 generic small entrypoint，不读取 oracle/error/prediction，也不访问 `/data2`。
"""

from __future__ import annotations

import builtins
import csv
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.features import TimeFuseFeatureCacheProvider  # noqa: E402
from time_router.protocols import FeatureBatch  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
TIMEFUSE_17D_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_timefuse_17dim_small"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
FEATURE_CSV_PATH = TIMEFUSE_17D_FIXTURE_ROOT / "features_17d.csv"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

FEATURE_COLUMNS = (
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


def assert_repo_file(path: Path) -> None:
    """函数功能：确认 smoke 输入存在于仓库内，且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"fixture 文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P13e smoke 不应访问 /data2 fixture：{path}")


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """
    函数功能：
        从 P13b real-derived sample manifest 中读取 ordered sample_keys。

    输入：
        path: `sample_manifest.csv` 路径。

    输出：
        按 manifest 行顺序排列的 sample_key tuple。
    """
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{sample_keys}")
    return sample_keys


def load_feature_reference(
    path: Path,
    *,
    ordered_sample_keys: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    """
    函数功能：
        读取 17 维 feature fixture，并按 manifest ordered sample_keys 重排参考矩阵。

    输入：
        path: `features_17d.csv` 路径。
        ordered_sample_keys: P13b manifest 行顺序。

    输出：
        `(csv_sample_keys, expected_features)`，其中 expected_features shape 为
        `[num_samples, 17]`。

    关键约束：
        本函数只读取 feature CSV，不读取 oracle label、oracle value、per-model error、
        y_true 或 prediction cache。
    """
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("features_17d.csv 不应为空")

    csv_sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if len(csv_sample_keys) != len(set(csv_sample_keys)):
        raise AssertionError(f"features_17d.csv 存在重复 sample_key：{csv_sample_keys}")
    if set(csv_sample_keys) != set(ordered_sample_keys):
        raise AssertionError(
            "features_17d.csv sample_key 集合未与 P13b manifest 对齐："
            f"csv={sorted(csv_sample_keys)} manifest={sorted(ordered_sample_keys)}"
        )
    if csv_sample_keys == tuple(ordered_sample_keys):
        raise AssertionError("features_17d.csv 行顺序应刻意不同于 manifest，用于验证 provider 保序")

    rows_by_key = {str(row["sample_key"]): row for row in rows}
    expected_rows = []
    for sample_key in ordered_sample_keys:
        row = rows_by_key[sample_key]
        expected_rows.append([float(row[column]) for column in FEATURE_COLUMNS])
    return csv_sample_keys, np.asarray(expected_rows, dtype=np.float32)


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于检查 provider 不创建 canonical run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


@contextmanager
def allow_only_feature_csv_reads(feature_csv_path: Path) -> Iterator[None]:
    """
    函数功能：
        provider 阶段只允许读取 17 维 feature CSV。

    关键约束：
        如果 provider 尝试读取 manifest、oracle/error、prediction cache、npy 或其他路径，
        这里会立即失败，避免把监督或专家误塞进 FeatureProvider。
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


def assert_feature_batch_contract(
    *,
    feature_batch: FeatureBatch,
    ordered_sample_keys: Sequence[str],
    expected_features: np.ndarray,
) -> None:
    """
    函数功能：
        验证 TimeFuseFeatureCacheProvider 输出 FeatureBatch 的核心 contract。

    输入：
        feature_batch: provider 输出。
        ordered_sample_keys: P13b manifest 行顺序。
        expected_features: 按 manifest 顺序重排后的 fixture 数值。
    """
    expected_sample_keys = tuple(ordered_sample_keys)
    if not isinstance(feature_batch, FeatureBatch):
        raise AssertionError(f"provider 未返回 FeatureBatch：actual={type(feature_batch)!r}")
    if feature_batch.sample_keys != expected_sample_keys:
        raise AssertionError(
            "FeatureBatch.sample_keys 未保持 manifest 行顺序："
            f"actual={feature_batch.sample_keys} expected={expected_sample_keys}"
        )
    if tuple(feature_batch.features.shape) != (len(expected_sample_keys), len(FEATURE_COLUMNS)):
        raise AssertionError(f"features shape 漂移：actual={feature_batch.features.shape}")
    if feature_batch.features.dtype != np.float32:
        raise AssertionError(f"features dtype 漂移：actual={feature_batch.features.dtype}")
    np.testing.assert_allclose(feature_batch.features, expected_features, rtol=0.0, atol=0.0)

    schema = feature_batch.feature_schema
    if schema.get("feature_schema_name") != "timefuse_single_variable_meta_v1":
        raise AssertionError(f"feature_schema_name 漂移：{schema}")
    if schema.get("feature_columns") != FEATURE_COLUMNS:
        raise AssertionError(f"feature_columns 漂移：{schema.get('feature_columns')}")
    if schema.get("feature_dim") != 17:
        raise AssertionError(f"feature_dim 漂移：{schema.get('feature_dim')}")
    if schema.get("source") != str(FEATURE_CSV_PATH):
        raise AssertionError(f"feature source 漂移：{schema.get('source')}")

    extra = feature_batch.extra
    if extra.get("provider_name") != "TimeFuseFeatureCacheProvider":
        raise AssertionError(f"provider metadata 漂移：{extra}")
    if extra.get("sample_key_column") != "sample_key":
        raise AssertionError(f"sample_key_column metadata 漂移：{extra}")
    if extra.get("feature_csv_path") != str(FEATURE_CSV_PATH):
        raise AssertionError(f"feature_csv_path metadata 漂移：{extra}")
    if extra.get("num_available_rows") != len(expected_sample_keys):
        raise AssertionError(f"num_available_rows metadata 漂移：{extra}")
    if extra.get("dtype") != "float32":
        raise AssertionError(f"dtype metadata 漂移：{extra}")


def run_smoke() -> None:
    """函数功能：执行 P13e TimeFuse 17 维 FeatureProvider small smoke。"""
    print("开始 Stage 1 P13e TimeFuse 17 维 FeatureProvider smoke")
    before_outputs = snapshot_run_outputs()

    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    csv_sample_keys, expected_features = load_feature_reference(
        FEATURE_CSV_PATH,
        ordered_sample_keys=ordered_sample_keys,
    )
    print("通过：P13b manifest 和 17 维 feature fixture 存在，sample_key 集合对齐且 CSV 顺序已打乱")

    with allow_only_feature_csv_reads(FEATURE_CSV_PATH):
        provider = TimeFuseFeatureCacheProvider(
            feature_csv_path=FEATURE_CSV_PATH,
            feature_columns=FEATURE_COLUMNS,
            feature_schema_name="timefuse_single_variable_meta_v1",
        )
        feature_batch = provider.load_batch(ordered_sample_keys)

    assert_feature_batch_contract(
        feature_batch=feature_batch,
        ordered_sample_keys=ordered_sample_keys,
        expected_features=expected_features,
    )
    print(f"通过：FeatureBatch 按 manifest 保序，features shape={feature_batch.features.shape}，17 维数值一致")
    print(f"通过：feature_schema/extra 记录 17 维 schema 与来源，fixture 原始顺序={csv_sample_keys}")

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P13e smoke 不应写 canonical run_dir 或输出目录：新增={sorted(after_outputs - before_outputs)}")
    print("通过：provider 阶段未读取 oracle/error/prediction，未创建 run_outputs 运行目录")
    print("完成：Stage 1 P13e TimeFuse 17 维 FeatureProvider smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
