#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P14b Visual FeatureProvider minimal mock/fixture smoke。

输入：
    使用 P13b real-derived sample_manifest.csv 的 ordered sample_keys，以及
    `tests/fixtures/stage1_visual_feature_mock/history_windows.json` 中的内存
    history window fixture。

输出：
    标准输出打印中文检查日志；若 VisualMockFeatureProvider 输出的 FeatureBatch
    在 sample_key 保序、feature shape、dtype、schema/extra 或边界上漂移，则抛出
    AssertionError。

关键约束：
    本 smoke 只验证 Visual-style provider mock -> FeatureBatch，不接 Visual
    RouterHead，不接 EvaluationInputAdapter，不写 canonical run_dir，不加载真实
    Hugging Face ViT，不访问 `/data2`，不读取 oracle/error/prediction/y_true。
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

from tests.helpers.visual_smoke_providers import DeterministicVisualEncoderStub, VisualMockFeatureProvider  # noqa: E402
from time_router.protocols import FeatureBatch  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
VISUAL_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_mock"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
HISTORY_WINDOWS_PATH = VISUAL_FIXTURE_ROOT / "history_windows.json"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"
EXPECTED_FEATURE_DIM = 8

DISALLOWED_METADATA_TOKENS = (
    "prediction",
    "oracle",
    "error",
    "y_true",
    "future",
    "run_dir",
    "metadata",
    "status",
    "checkpoint",
    "/data2",
)


def assert_repo_file(path: Path) -> None:
    """函数功能：确认 smoke 输入存在于仓库内，且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"fixture 文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P14b smoke 不应访问 /data2 fixture：{path}")


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """
    函数功能：
        从 P13b real-derived sample manifest 中读取 ordered sample_keys。
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


def load_history_windows(path: Path) -> dict[str, list[float]]:
    """
    函数功能：
        读取小型 history window fixture，并确认其中没有监督或运行产物字段。

    关键约束：
        fixture 只允许保存 sample_key 到历史窗口 x 的映射；future y、oracle、
        prediction cache、run_dir、status 等字段一律不允许出现。
    """
    assert_repo_file(path)
    raw_text = path.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    for token in DISALLOWED_METADATA_TOKENS:
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
        try:
            history_windows[sample_key] = [float(value) for value in window]
        except (TypeError, ValueError) as exc:
            raise AssertionError(f"history window 必须只包含数值：sample_key={sample_key}") from exc
    return history_windows


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于检查 provider 不创建 canonical run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


@contextmanager
def forbid_provider_file_reads() -> Iterator[None]:
    """
    函数功能：
        provider 阶段禁止任何文件读取或 np.load。

    关键约束：
        manifest/history fixture 可以在 provider 外部读取；provider 本身只能消费
        调用方注入的内存对象。如果 provider 尝试读取 prediction/oracle/run_dir 或
        任何其他路径，这里会立即失败。
    """

    def fail_open(file: object, *_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"VisualMockFeatureProvider 阶段不应读取任何文件：{file}")

    def fail_path_open(path_self: Path, *_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"VisualMockFeatureProvider 阶段不应读取任何路径：{path_self}")

    def fail_np_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("VisualMockFeatureProvider 不应调用 np.load 读取 prediction cache")

    with patch.object(builtins, "open", side_effect=fail_open), patch.object(
        Path, "open", fail_path_open
    ), patch.object(Path, "read_text", fail_path_open), patch.object(np, "load", side_effect=fail_np_load):
        yield


def expected_stub_features(
    *,
    encoder: DeterministicVisualEncoderStub,
    ordered_sample_keys: Sequence[str],
    history_windows: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """函数功能：按 manifest 顺序复算 deterministic encoder stub 的期望输出。"""
    return encoder.encode_batch([history_windows[sample_key] for sample_key in ordered_sample_keys])


def assert_no_disallowed_metadata(payload: Any, *, label: str) -> None:
    """函数功能：检查 schema/extra 中没有 supervision、prediction 或 run artifact 字段。"""
    lowered = json.dumps(payload, sort_keys=True, ensure_ascii=False).lower()
    for token in DISALLOWED_METADATA_TOKENS:
        if token in lowered:
            raise AssertionError(f"{label} 不应包含禁用字段或路径 {token!r}：{payload}")


def assert_feature_batch_contract(
    *,
    feature_batch: FeatureBatch,
    ordered_sample_keys: Sequence[str],
    expected_features: np.ndarray,
    num_available_rows: int,
) -> None:
    """
    函数功能：
        验证 VisualMockFeatureProvider 输出 FeatureBatch 的核心 contract。
    """
    expected_sample_keys = tuple(ordered_sample_keys)
    if not isinstance(feature_batch, FeatureBatch):
        raise AssertionError(f"provider 未返回 FeatureBatch：actual={type(feature_batch)!r}")
    if feature_batch.sample_keys != expected_sample_keys:
        raise AssertionError(
            "FeatureBatch.sample_keys 未保持 manifest 行顺序："
            f"actual={feature_batch.sample_keys} expected={expected_sample_keys}"
        )
    if tuple(feature_batch.features.shape) != (len(expected_sample_keys), EXPECTED_FEATURE_DIM):
        raise AssertionError(f"features shape 漂移：actual={feature_batch.features.shape}")
    if feature_batch.features.dtype != np.float32:
        raise AssertionError(f"features dtype 漂移：actual={feature_batch.features.dtype}")
    np.testing.assert_allclose(feature_batch.features, expected_features, rtol=0.0, atol=0.0)

    schema = feature_batch.feature_schema
    if schema.get("feature_schema_name") != "visual_mock_history_encoder_v1":
        raise AssertionError(f"feature_schema_name 漂移：{schema}")
    if schema.get("feature_dim") != EXPECTED_FEATURE_DIM:
        raise AssertionError(f"feature_dim 漂移：{schema.get('feature_dim')}")
    if schema.get("history_source") != "stage1_visual_feature_mock_history_window_x":
        raise AssertionError(f"history_source 漂移：{schema.get('history_source')}")
    if schema.get("pseudo_image", {}).get("variant") != "mock_not_materialized":
        raise AssertionError(f"pseudo_image 口径漂移：{schema.get('pseudo_image')}")
    encoder_stub = schema.get("encoder_stub", {})
    if encoder_stub.get("name") != "deterministic_visual_history_stats_stub_v1":
        raise AssertionError(f"encoder_stub name 漂移：{encoder_stub}")
    if encoder_stub.get("loads_real_vit") is not False:
        raise AssertionError(f"encoder_stub 不应加载真实 ViT：{encoder_stub}")
    if encoder_stub.get("uses_gpu") is not False:
        raise AssertionError(f"encoder_stub 不应使用 GPU：{encoder_stub}")
    if encoder_stub.get("uses_huggingface_cache") is not False:
        raise AssertionError(f"encoder_stub 不应使用 Hugging Face cache：{encoder_stub}")
    if schema.get("dtype") != "float32":
        raise AssertionError(f"schema dtype 漂移：{schema}")
    assert_no_disallowed_metadata(schema, label="feature_schema")

    extra = feature_batch.extra
    if extra.get("provider_name") != "VisualMockFeatureProvider":
        raise AssertionError(f"provider metadata 漂移：{extra}")
    if extra.get("source") != "tests/fixtures/stage1_visual_feature_mock/history_windows.json:in_memory":
        raise AssertionError(f"source metadata 漂移：{extra}")
    if extra.get("num_available_rows") != num_available_rows:
        raise AssertionError(f"num_available_rows metadata 漂移：{extra}")
    assert_no_disallowed_metadata(extra, label="extra")


def run_smoke() -> None:
    """函数功能：执行 P14b Visual FeatureProvider mock smoke。"""
    print("开始 Stage 1 P14b Visual FeatureProvider mock smoke")
    before_outputs = snapshot_run_outputs()

    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    history_windows = load_history_windows(HISTORY_WINDOWS_PATH)
    if set(history_windows) != set(ordered_sample_keys):
        raise AssertionError(
            "history window fixture sample_key 集合未与 P13b manifest 对齐："
            f"history={sorted(history_windows)} manifest={sorted(ordered_sample_keys)}"
        )
    print("通过：P13b manifest 和 Visual history window fixture 存在，且 sample_key 集合对齐")

    encoder = DeterministicVisualEncoderStub()
    expected_features = expected_stub_features(
        encoder=encoder,
        ordered_sample_keys=ordered_sample_keys,
        history_windows=history_windows,
    )

    with forbid_provider_file_reads():
        provider = VisualMockFeatureProvider(
            history_windows=history_windows,
            encoder=encoder,
            history_source_name="stage1_visual_feature_mock_history_window_x",
            feature_schema_name="visual_mock_history_encoder_v1",
            source="tests/fixtures/stage1_visual_feature_mock/history_windows.json:in_memory",
        )
        feature_batch = provider.load_batch(ordered_sample_keys)

    assert_feature_batch_contract(
        feature_batch=feature_batch,
        ordered_sample_keys=ordered_sample_keys,
        expected_features=expected_features,
        num_available_rows=len(history_windows),
    )
    print(f"通过：FeatureBatch 按 manifest 保序，features shape={feature_batch.features.shape}，dtype=float32")
    print("通过：feature_schema 记录 visual_mock schema、history_source、pseudo_image 和 encoder_stub 口径")

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P14b smoke 不应写 canonical run_dir 或输出目录：新增={sorted(after_outputs - before_outputs)}")
    print("通过：provider 阶段未读取文件、oracle/error/prediction/y_true/run_dir，未创建 run_outputs 运行目录")
    print("完成：Stage 1 P14b Visual FeatureProvider mock smoke 全部通过")


if __name__ == "__main__":
    run_smoke()

