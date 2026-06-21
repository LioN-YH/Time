#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16h legacy VisualMLPRouter loaded-module smoke。

输入：
    使用 P13b `sample_manifest.csv` 的 ordered sample_keys、P16c
    `VisualPrecomputedFeatureProvider` 读取的 head-ready float32 FeatureBatch、
    P13b `expert_predictions.json` 构造的小型 ExpertBatch，以及内存 fake
    state_dict。

输出：
    标准输出打印中文检查日志；若 legacy VisualMLPRouter 的已加载 module 不能被
    P16a `LoadedTorchMLPRouterHeadAdapter` 消费，或 state_dict `module.` 前缀清洗、
    sample/model 保序、softmax/evaluation 边界发生漂移，则抛出 AssertionError。

关键约束：
    本 smoke 只验证“legacy VisualMLPRouter 已加载 module -> P16a adapter”边界。
    不实现 checkpoint loader，不调用 torch.load，不读取真实 checkpoint，不处理真实
    scaler，不访问 `/data2`，不启动 ViT，不迁移正式入口，不调用
    train_visual_router_online_streaming.py，也不修改 scripts/run_stage1_visual_small.py。
"""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import VisualPrecomputedFeatureProvider  # noqa: E402
from time_router.models import LoadedTorchMLPRouterHeadAdapter  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
VISUAL_PRECOMPUTED_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_precomputed_small"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
VISUAL_EMBEDDINGS_PATH = VISUAL_PRECOMPUTED_FIXTURE_ROOT / "visual_embeddings.csv"
LEGACY_MODULE_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
LEGACY_SOURCE_PATH = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router.py"
STREAMING_ENTRYPOINT_PATH = (
    REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"
)
VISUAL_SMALL_ENTRYPOINT_PATH = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
ADAPTER_SOURCE_PATH = REPO_ROOT / "time_router" / "models" / "visual_mlp_adapter.py"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

EXPECTED_FEATURE_DIM = 8
HIDDEN_DIM = 11
ATOL = 1e-6


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入文件存在且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P16h smoke 不应访问 /data2：{path}")


def load_manifest_sample_keys(path: Path) -> tuple[str, ...]:
    """函数功能：从 P13b manifest 中读取 ordered sample_keys。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    sample_keys = tuple(str(row["sample_key"]) for row in rows)
    if len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{sample_keys}")
    return sample_keys


def load_expert_batch_from_reference(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """
    函数功能：
        用 P13b expert JSON 数值参考构造最小 ExpertBatch。

    关键约束：
        P13b JSON 只是 small fixture 参考，不代表正式 prediction backend schema。
    """
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError("expert_predictions.json 不是 JSON object")

    model_columns = tuple(str(model_name) for model_name in payload["model_columns"])
    if not model_columns or len(model_columns) != len(set(model_columns)):
        raise AssertionError(f"model_columns 异常：{model_columns}")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise AssertionError("expert_predictions.json 缺少 samples list")
    sample_by_key: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise AssertionError(f"sample 不是 object：{sample}")
        sample_key = str(sample["sample_key"])
        if sample_key in sample_by_key:
            raise AssertionError(f"expert_predictions.json 存在重复 sample_key：{sample_key}")
        sample_by_key[sample_key] = sample
    if set(sample_by_key) != set(ordered_sample_keys):
        raise AssertionError(
            "expert_predictions.json sample_key 集合未与 manifest 对齐："
            f"json={sorted(sample_by_key)} manifest={sorted(ordered_sample_keys)}"
        )

    y_true_rows = []
    y_pred_rows = []
    for sample_key in ordered_sample_keys:
        sample = sample_by_key[str(sample_key)]
        y_true_rows.append(np.asarray(sample["y_true"], dtype=np.float32))
        y_pred = np.asarray(sample["y_pred"], dtype=np.float32)
        if y_pred.shape[0] != len(model_columns):
            raise AssertionError(f"{sample_key} y_pred 专家维与 model_columns 不一致：shape={y_pred.shape}")
        y_pred_rows.append(y_pred)

    return ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in ordered_sample_keys),
        model_columns=model_columns,
        y_pred=np.stack(y_pred_rows, axis=0).astype(np.float32),
        y_true=np.stack(y_true_rows, axis=0).astype(np.float32),
        row_index_metadata={
            "source": "tests/fixtures/stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
            "formal_backend_schema": False,
        },
        extra={
            "provider_name": "P16hInMemoryExpertBatchReference",
            "fixture": "stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
        },
    )


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def fail_torch_load(*args: object, **kwargs: object) -> object:
    """函数功能：若 smoke 核心路径读取 checkpoint，则立即失败。"""
    raise AssertionError(f"P16h 不应调用 torch.load 或读取 checkpoint：args={args} kwargs={kwargs}")


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    函数功能：
        清洗 DataParallel 风格 `module.` 前缀，返回可 strict load 的 state_dict。

    关键约束：
        该 helper 是 smoke-local 最小逻辑，只验证后续 Runtime loader 需要覆盖的 key
        normalization 行为；本步不把 checkpoint loader 或 path 参数加入正式 API。
    """
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key in cleaned:
            raise ValueError(f"state_dict 清洗后出现重复 key：{normalized_key}")
        cleaned[normalized_key] = value
    return cleaned


def load_legacy_visual_mlp_router_class() -> type[torch.nn.Module]:
    """
    函数功能：
        优先 import legacy VisualMLPRouter 类，并校验 constructor / forward 签名。
    """
    assert_repo_file(LEGACY_SOURCE_PATH)
    module = importlib.import_module(LEGACY_MODULE_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    if not inspect.isclass(router_cls) or not issubclass(router_cls, torch.nn.Module):
        raise AssertionError(f"legacy VisualMLPRouter 必须是 torch.nn.Module 子类：{router_cls!r}")

    init_signature = inspect.signature(router_cls.__init__)
    expected_init_params = ("self", "input_dim", "hidden_dim", "output_dim", "dropout")
    if tuple(init_signature.parameters) != expected_init_params:
        raise AssertionError(f"VisualMLPRouter constructor 签名漂移：{init_signature}")

    forward_signature = inspect.signature(router_cls.forward)
    if tuple(forward_signature.parameters) != ("self", "features"):
        raise AssertionError(f"VisualMLPRouter.forward 签名漂移：{forward_signature}")
    return router_cls


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    函数功能：
        基于 legacy module 的 key/shape 构造 deterministic in-memory fake state_dict。

    关键约束：
        这里只生成内存 tensor，不读取真实 checkpoint；数值只用于证明 strict load
        后 module 可前向并被 P16a adapter 消费。
    """
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        values = torch.linspace(
            -0.08 + index * 0.01,
            0.08 + index * 0.01,
            steps=value.numel(),
            dtype=value.dtype,
        )
        fake_state[key] = values.reshape_as(value).clone()
    expected_keys = {"network.0.weight", "network.0.bias", "network.3.weight", "network.3.bias"}
    if set(fake_state) != expected_keys:
        raise AssertionError(f"legacy VisualMLPRouter state_dict key 漂移：{sorted(fake_state)}")
    return fake_state


def build_loaded_legacy_router(
    *,
    router_cls: type[torch.nn.Module],
    input_dim: int,
    output_dim: int,
    state_dict: Mapping[str, torch.Tensor],
) -> torch.nn.Module:
    """
    函数功能：
        实例化 legacy VisualMLPRouter，并 strict 加载已清洗的内存 state_dict。
    """
    model = router_cls(input_dim=int(input_dim), hidden_dim=HIDDEN_DIM, output_dim=int(output_dim), dropout=0.0)
    cleaned_state_dict = strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(cleaned_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(
            "legacy VisualMLPRouter strict load 后仍有 key 不匹配："
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def assert_feature_and_expert_alignment(feature_batch: FeatureBatch, expert_batch: ExpertBatch) -> None:
    """函数功能：检查 head-ready FeatureBatch 与 ExpertBatch 只通过 sample_keys 对齐。"""
    if feature_batch.sample_keys != expert_batch.sample_keys:
        raise AssertionError(
            "FeatureBatch sample_keys 必须与 ExpertBatch manifest 顺序一致："
            f"feature={feature_batch.sample_keys} expert={expert_batch.sample_keys}"
        )
    features = np.asarray(feature_batch.features)
    if features.shape != (len(expert_batch.sample_keys), EXPECTED_FEATURE_DIM):
        raise AssertionError(f"head-ready features shape 漂移：actual={features.shape}")
    if features.dtype != np.float32:
        raise AssertionError(f"head-ready features dtype 必须为 float32：actual={features.dtype}")
    if not np.all(np.isfinite(features)):
        raise AssertionError("head-ready features 包含 NaN 或 Inf")


def assert_module_forward_contract(
    *,
    loaded_model: torch.nn.Module,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> None:
    """函数功能：直接验证已加载 legacy module forward 输出二维 logits。"""
    features_tensor = torch.from_numpy(np.asarray(feature_batch.features, dtype=np.float32))
    with torch.inference_mode():
        logits_tensor = loaded_model(features_tensor)
    if not isinstance(logits_tensor, torch.Tensor):
        raise AssertionError(f"legacy VisualMLPRouter forward 未返回 Tensor：{type(logits_tensor)!r}")
    expected_shape = (len(feature_batch.sample_keys), len(expert_batch.model_columns))
    if tuple(logits_tensor.shape) != expected_shape:
        raise AssertionError(f"legacy VisualMLPRouter logits shape 漂移：actual={tuple(logits_tensor.shape)}")
    if logits_tensor.ndim != 2 or not torch.isfinite(logits_tensor).all().item():
        raise AssertionError("legacy VisualMLPRouter logits 必须是二维有限 Tensor")


def assert_router_output_contract(
    *,
    router_output: RouterOutput,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> None:
    """函数功能：验证 P16a adapter 输出与 FeatureBatch / ExpertBatch 对齐。"""
    if not isinstance(router_output, RouterOutput):
        raise AssertionError(f"adapter 未返回 RouterOutput：actual={type(router_output)!r}")
    if router_output.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"RouterOutput sample_keys 未保持 FeatureBatch 顺序：{router_output.sample_keys}")
    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError(
            "RouterOutput model_columns 必须等于 ExpertBatch.model_columns："
            f"router={router_output.model_columns} expert={expert_batch.model_columns}"
        )

    logits = np.asarray(router_output.logits)
    weights = np.asarray(router_output.weights)
    expected_shape = (len(expert_batch.sample_keys), len(expert_batch.model_columns))
    if logits.shape != expected_shape:
        raise AssertionError(f"RouterOutput logits shape 漂移：actual={logits.shape} expected={expected_shape}")
    if weights.shape != expected_shape:
        raise AssertionError(f"RouterOutput weights shape 漂移：actual={weights.shape} expected={expected_shape}")
    if logits.dtype != np.float32 or weights.dtype != np.float32:
        raise AssertionError(f"RouterOutput logits/weights dtype 应为 float32：{logits.dtype} {weights.dtype}")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(weights)):
        raise AssertionError("RouterOutput logits/weights 包含 NaN 或 Inf")
    if np.any(weights < 0.0):
        raise AssertionError(f"RouterOutput weights 不应为负：{weights}")
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(expected_shape[0]), rtol=0.0, atol=ATOL)


def assert_evaluation_result_contract(
    *,
    result: EvaluationInputAdapterResult,
    expert_batch: ExpertBatch,
    router_output: RouterOutput,
    ordered_sample_keys: Sequence[str],
) -> None:
    """函数功能：检查 EvaluationInputAdapter 生成 summary/rows 且 sample_key 保序。"""
    evaluation_input = result.evaluation_input
    if evaluation_input.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"EvaluationInput sample_keys 未保持 manifest 顺序：{evaluation_input.sample_keys}")
    if evaluation_input.model_columns != expert_batch.model_columns:
        raise AssertionError(f"EvaluationInput model_columns 漂移：{evaluation_input.model_columns}")
    if evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")

    summary = result.summary
    for key in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse"):
        if key not in summary:
            raise AssertionError(f"summary 缺少 {key}：{summary}")
        if not np.isfinite(float(summary[key])):
            raise AssertionError(f"summary {key} 不是有限值：{summary[key]}")

    rows = result.per_sample_rows
    if len(rows) != len(ordered_sample_keys):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(rows)} expected={len(ordered_sample_keys)}")
    if [row["sample_key"] for row in rows] != list(ordered_sample_keys):
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：{rows}")


def assert_boundary_sources_unchanged() -> None:
    """
    函数功能：
        扫描关键源码边界，确认 P16h 没有把 checkpoint/ViT/正式入口语义下沉到 adapter。
    """
    assert_repo_file(ADAPTER_SOURCE_PATH)
    adapter_source = ADAPTER_SOURCE_PATH.read_text(encoding="utf-8")
    for token in ("/data2", "torch.load", "checkpoint_path", "scaler_path", "ViTModel", "AutoImageProcessor"):
        if token in adapter_source:
            raise AssertionError(f"P16a adapter 源码不应包含 P16h 禁止 token：{token!r}")

    assert_repo_file(VISUAL_SMALL_ENTRYPOINT_PATH)
    visual_small_source = VISUAL_SMALL_ENTRYPOINT_PATH.read_text(encoding="utf-8")
    if "LoadedTorchMLPRouterHeadAdapter" in visual_small_source:
        raise AssertionError("P16h 不应把正式 adapter 接入 scripts/run_stage1_visual_small.py")
    if "SmokeOnlyVisualMLPAdapter" not in visual_small_source:
        raise AssertionError("scripts/run_stage1_visual_small.py 应继续保留 script-local smoke adapter")

    assert_repo_file(STREAMING_ENTRYPOINT_PATH)


def run_smoke() -> None:
    """函数功能：执行 P16h legacy VisualMLPRouter loaded-module smoke。"""
    print("开始 Stage 1 P16h legacy VisualMLPRouter loaded-module smoke")
    before_outputs = snapshot_run_outputs()
    assert_boundary_sources_unchanged()

    with patch.object(torch, "load", side_effect=fail_torch_load):
        router_cls = load_legacy_visual_mlp_router_class()
        print("通过：已 import legacy VisualMLPRouter 定义，未调用 torch.load")

        ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
        expert_batch = load_expert_batch_from_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
        provider = VisualPrecomputedFeatureProvider(
            feature_source_path=VISUAL_EMBEDDINGS_PATH,
            source_name="tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv",
            provider_name="P16hVisualPrecomputedFeatureProvider",
        )
        feature_batch = provider.load_batch(ordered_sample_keys)
        assert_feature_and_expert_alignment(feature_batch, expert_batch)
        print("通过：P13b ordered sample_keys、P16c head-ready FeatureBatch 和 P13b ExpertBatch 已对齐")

        template_model = router_cls(
            input_dim=int(feature_batch.features.shape[1]),
            hidden_dim=HIDDEN_DIM,
            output_dim=len(expert_batch.model_columns),
            dropout=0.0,
        )
        normal_state_dict = build_fake_state_dict(template_model)
        module_prefixed_state_dict = {f"module.{key}": value.clone() for key, value in normal_state_dict.items()}

        normal_loaded_model = build_loaded_legacy_router(
            router_cls=router_cls,
            input_dim=int(feature_batch.features.shape[1]),
            output_dim=len(expert_batch.model_columns),
            state_dict=normal_state_dict,
        )
        prefixed_loaded_model = build_loaded_legacy_router(
            router_cls=router_cls,
            input_dim=int(feature_batch.features.shape[1]),
            output_dim=len(expert_batch.model_columns),
            state_dict=module_prefixed_state_dict,
        )
        for key in normal_state_dict:
            np.testing.assert_allclose(
                normal_loaded_model.state_dict()[key].detach().cpu().numpy(),
                prefixed_loaded_model.state_dict()[key].detach().cpu().numpy(),
                rtol=0.0,
                atol=0.0,
            )
        print("通过：normal state_dict 和 DataParallel module. 前缀 state_dict 均可清洗后 strict load")

        assert_module_forward_contract(
            loaded_model=prefixed_loaded_model,
            feature_batch=feature_batch,
            expert_batch=expert_batch,
        )
        adapter = LoadedTorchMLPRouterHeadAdapter(model=prefixed_loaded_model, device=torch.device("cpu"))
        router_output = adapter.predict(feature_batch, expert_batch.model_columns)
        result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)

    assert_router_output_contract(router_output=router_output, feature_batch=feature_batch, expert_batch=expert_batch)
    assert_evaluation_result_contract(
        result=result,
        expert_batch=expert_batch,
        router_output=router_output,
        ordered_sample_keys=ordered_sample_keys,
    )
    print(
        "通过：legacy loaded module 已被 P16a adapter 消费，EvaluationInputAdapter summary/rows 正常生成，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P16h smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P16h legacy VisualMLPRouter loaded-module smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
