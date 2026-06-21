#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P18a time_router public API boundary smoke。

输入：
    只导入仓库内 `time_router` package、P17 canonical eval entrypoint 文件和
    已存在的 public API 名称；不读取 fixture、checkpoint、feature CSV 或 `/data2`。

输出：
    标准输出打印中文检查日志；若 canonical core / P17 bridge public import 漂移，
    或 smoke-only scaffold 进入 public `__all__`，则抛出 AssertionError。

关键约束：
    本 smoke 只验证 import 边界，不启动 ViT、不导入 transformers、不启动训练、
    不访问 `/data2`，也不调用任何 entrypoint main/run 逻辑。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py"


def assert_not_transformers_loaded(stage: str) -> None:
    """函数功能：确认 import boundary smoke 没有触发 Hugging Face transformers 导入。"""
    loaded = sorted(name for name in sys.modules if name == "transformers" or name.startswith("transformers."))
    if loaded:
        raise AssertionError(f"{stage} 不应导入 transformers：{loaded[:8]}")


def assert_module_all(module: ModuleType, expected_names: set[str], *, module_name: str) -> None:
    """函数功能：检查模块 __all__ 至少包含指定 public API 名称。"""
    exported = set(getattr(module, "__all__", ()))
    missing = sorted(expected_names - exported)
    if missing:
        raise AssertionError(f"{module_name} public __all__ 缺少 canonical 名称：{missing}")
    for name in expected_names:
        if not hasattr(module, name):
            raise AssertionError(f"{module_name} 缺少 public 属性：{name}")


def import_entrypoint_module() -> ModuleType:
    """
    函数功能：
        以 module import 方式加载 P17 canonical eval entrypoint，验证 import 阶段
        只建立函数/class 定义，不执行 CLI main 或访问真实路径。
    """
    spec = importlib.util.spec_from_file_location("stage1_p18a_visual_eval_canonical_import_boundary", ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法构造 entrypoint import spec：{ENTRYPOINT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_smoke() -> None:
    """函数功能：执行 Stage 1 P18a public API boundary smoke。"""
    print("开始 Stage 1 P18a time_router public API boundary smoke")
    assert_not_transformers_loaded("smoke 启动阶段")

    time_router = importlib.import_module("time_router")
    protocols = importlib.import_module("time_router.protocols")
    runtime = importlib.import_module("time_router.runtime")
    features = importlib.import_module("time_router.features")
    models = importlib.import_module("time_router.models")
    evaluation = importlib.import_module("time_router.evaluation")
    experts = importlib.import_module("time_router.experts")
    print("通过：time_router package 与子包 import 成功")

    if time_router.__name__ != "time_router":
        raise AssertionError(f"time_router 根包 import 异常：{time_router}")

    assert_module_all(
        protocols,
        {
            "SampleManifest",
            "SampleManifestRow",
            "ExpertBatch",
            "FeatureBatch",
            "RouterOutput",
            "EvaluationInput",
        },
        module_name="time_router.protocols",
    )
    assert_module_all(
        evaluation,
        {"EvaluationInputAdapter", "EvaluationInputAdapterResult", "FusionEvaluator"},
        module_name="time_router.evaluation",
    )
    assert_module_all(
        models,
        {"LoadedTorchMLPRouterHeadAdapter", "TimeFuseLinearSoftmaxHead"},
        module_name="time_router.models",
    )
    assert_module_all(
        features,
        {
            "TimeFuseFeatureCacheProvider",
            "VisualPrecomputedFeatureProvider",
            "LoadedFeatureScaler",
            "VisualFeatureChainSpec",
            "RawWindowBatch",
            "VisualEmbeddingBatch",
        },
        module_name="time_router.features",
    )
    assert_module_all(
        experts,
        {"PredictionCacheExpertProvider"},
        module_name="time_router.experts",
    )
    assert_module_all(
        runtime,
        {
            "create_run_dir",
            "write_run_metadata",
            "write_run_status",
            "extract_router_state_dict",
            "load_checkpoint_payload",
            "load_router_state_dict",
            "authorize_visual_eval_checkpoint_path",
            "authorize_visual_eval_feature_path",
            "authorize_visual_eval_scaler_path",
        },
        module_name="time_router.runtime",
    )
    print("通过：canonical core 与 P17 migration bridge public imports 可用")

    smoke_only_names = {"DeterministicVisualEncoderStub", "VisualMockFeatureProvider"}
    leaked = sorted(smoke_only_names & set(getattr(features, "__all__", ())))
    if leaked:
        raise AssertionError(f"smoke-only scaffold 不应进入 time_router.features.__all__：{leaked}")
    # 中文注释：保留属性兼容旧 smoke，但 public star-import 边界不再推广这些名字。
    for name in smoke_only_names:
        if not hasattr(features, name):
            raise AssertionError(f"P18a 兼容期仍应保留 smoke-only 属性，避免旧 smoke 断裂：{name}")
    print("通过：smoke-only Visual mock scaffold 未出现在 features.__all__，兼容属性仍可用")

    entrypoint_module = import_entrypoint_module()
    for name in (
        "JsonExpertProvider",
        "build_feature_batch",
        "import_legacy_visual_mlp_router",
        "load_sample_manifest_csv",
        "main",
    ):
        if not hasattr(entrypoint_module, name):
            raise AssertionError(f"P17 canonical eval entrypoint import 后缺少名称：{name}")
    assert_not_transformers_loaded("entrypoint import 后")
    if any(str(Path(value).resolve()).startswith("/data2/") for value in (ENTRYPOINT, REPO_ROOT)):
        raise AssertionError("P18a smoke 不应以 /data2 路径运行")
    print("通过：P17 canonical eval entrypoint 依赖的 public import 边界可加载，未导入 transformers")
    print("完成：Stage 1 P18a time_router public API boundary smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
