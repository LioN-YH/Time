#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P18b time_router public API smoke scaffold cleanup smoke。

输入：
    仅导入 `time_router.features` public core 名称、tests helper 中的 visual
    smoke provider，以及 P18b 前保留的旧子模块兼容层；不读取 fixture、
    checkpoint、feature CSV 或 `/data2`。

输出：
    标准输出打印中文检查日志；若 smoke-only visual scaffold 仍泄漏到
    `time_router.features` public API 边界，或 helper 无法承接旧 smoke 能力，
    则抛出 AssertionError。

关键约束：
    本 smoke 只做 import boundary 检查，不启动 ViT、不导入 transformers、
    不启动训练、不访问 `/data2`，也不调用任何 entrypoint main/run 逻辑。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def assert_not_transformers_loaded(stage: str) -> None:
    """函数功能：确认 import boundary smoke 没有触发 Hugging Face transformers 导入。"""
    loaded = sorted(name for name in sys.modules if name == "transformers" or name.startswith("transformers."))
    if loaded:
        raise AssertionError(f"{stage} 不应导入 transformers：{loaded[:8]}")


def run_smoke() -> None:
    """函数功能：执行 Stage 1 P18b smoke-only visual scaffold cleanup smoke。"""
    print("开始 Stage 1 P18b time_router public API smoke scaffold cleanup smoke")
    assert_not_transformers_loaded("smoke 启动阶段")

    features = importlib.import_module("time_router.features")
    from time_router.features import LoadedFeatureScaler, VisualPrecomputedFeatureProvider  # noqa: E402

    if VisualPrecomputedFeatureProvider.__name__ != "VisualPrecomputedFeatureProvider":
        raise AssertionError("VisualPrecomputedFeatureProvider public import 异常")
    if LoadedFeatureScaler.__name__ != "LoadedFeatureScaler":
        raise AssertionError("LoadedFeatureScaler public import 异常")
    print("通过：time_router.features 长期 core provider/transform import 成功")

    smoke_only_names = {"VisualMockFeatureProvider", "DeterministicVisualEncoderStub"}
    exported = set(getattr(features, "__all__", ()))
    leaked_all = sorted(smoke_only_names & exported)
    if leaked_all:
        raise AssertionError(f"smoke-only scaffold 不应进入 time_router.features.__all__：{leaked_all}")
    leaked_attrs = sorted(name for name in smoke_only_names if hasattr(features, name))
    if leaked_attrs:
        raise AssertionError(f"smoke-only scaffold 不应保留在 time_router.features package 入口：{leaked_attrs}")
    print("通过：visual smoke scaffold 已从 time_router.features public 边界移出")

    from tests.helpers.visual_smoke_providers import (  # noqa: E402
        DeterministicVisualEncoderStub,
        VisualMockFeatureProvider,
    )

    if DeterministicVisualEncoderStub().feature_dim != 8:
        raise AssertionError("DeterministicVisualEncoderStub helper feature_dim 漂移")
    provider = VisualMockFeatureProvider(history_windows={"sample_a": [1.0, 2.0, 3.0]})
    batch = provider.load_batch(["sample_a"])
    if batch.sample_keys != ("sample_a",):
        raise AssertionError(f"VisualMockFeatureProvider helper 未保持 sample_key 顺序：{batch.sample_keys}")
    if batch.features.shape != (1, 8) or str(batch.features.dtype) != "float32":
        raise AssertionError(f"VisualMockFeatureProvider helper 输出异常：{batch.features.shape}, {batch.features.dtype}")
    print("通过：tests.helpers.visual_smoke_providers 可承接 mock/stub 能力")

    visual_mock_module = importlib.import_module("time_router.features.visual_mock")
    if not hasattr(visual_mock_module, "VisualMockFeatureProvider"):
        raise AssertionError("P18c 前旧 visual_mock 子模块兼容层缺少 VisualMockFeatureProvider")
    if getattr(visual_mock_module, "VisualMockFeatureProvider") is not VisualMockFeatureProvider:
        raise AssertionError("旧 visual_mock 子模块兼容层未指向 tests helper 实现")
    print("通过：P18c 前旧子模块兼容层仍指向 tests helper 实现")

    assert_not_transformers_loaded("完成 import 检查后")
    if any(str(Path(value).resolve()).startswith("/data2/") for value in (REPO_ROOT,)):
        raise AssertionError("P18b smoke 不应以 /data2 路径运行")
    print("完成：Stage 1 P18b time_router public API smoke scaffold cleanup smoke 全部通过")


if __name__ == "__main__":
    run_smoke()

