#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16i legacy VisualMLPRouter tiny checkpoint payload loader smoke。

输入：
    在 tempfile 内创建 tiny checkpoint payload，覆盖 `router_state_dict`、
    `scaler_state`、`config` 和 `metadata`；使用 P13b ordered sample_keys、P16c
    `VisualPrecomputedFeatureProvider` head-ready FeatureBatch，以及 P13b
    `expert_predictions.json` 构造 ExpertBatch。

输出：
    标准输出打印中文检查日志；若 runtime checkpoint helper 不能完成“显式
    checkpoint path -> payload -> router_state_dict -> 清理 module. 前缀 -> strict
    load 到已构造 legacy VisualMLPRouter -> P16a adapter -> EvaluationInputAdapter”
    边界，或任何禁止边界被触碰，则抛出 AssertionError。

关键约束：
    本 smoke 不读取真实 checkpoint，不访问 `/data2`，不处理真实 scaler transform，
    不启动 ViT/transformers，不调用正式 streaming 入口，不修改
    scripts/run_stage1_visual_small.py，也不改变 P16a adapter 接口。
"""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import sys
import tempfile
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
from time_router.runtime import (  # noqa: E402
    extract_router_state_dict,
    load_checkpoint_payload,
    load_router_state_dict,
    strip_dataparallel_prefix,
)


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
RUNTIME_HELPER_SOURCE_PATH = REPO_ROOT / "time_router" / "runtime" / "visual_mlp_checkpoint.py"
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
        raise AssertionError(f"P16i smoke 不应访问 /data2：{path}")


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
            "provider_name": "P16iInMemoryExpertBatchReference",
            "fixture": "stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
        },
    )


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def load_legacy_visual_mlp_router_class() -> type[torch.nn.Module]:
    """函数功能：import legacy VisualMLPRouter，并校验 constructor / forward 签名。"""
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
        基于 legacy module 的 key/shape 构造 deterministic fake state_dict。

    关键约束：
        这里只生成 tiny tensor，用于写入 tempfile checkpoint；不读取真实 checkpoint。
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


def build_payload(state_dict: Mapping[str, torch.Tensor], *, payload_name: str) -> dict[str, Any]:
    """
    函数功能：
        构造 P16i tiny checkpoint payload，包含 router/scaler/config/metadata。

    关键约束：
        `scaler_state` 只作为 payload metadata 被识别，本 smoke 不执行 transform。
    """
    return {
        "router_state_dict": dict(state_dict),
        "scaler_state": {
            "kind": "metadata_only",
            "mean": [0.0] * EXPECTED_FEATURE_DIM,
            "scale": [1.0] * EXPECTED_FEATURE_DIM,
        },
        "config": {
            "input_dim": EXPECTED_FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": 5,
            "dropout": 0.0,
            "payload_name": payload_name,
        },
        "metadata": {
            "stage": "P16i",
            "source": "tempfile checkpoint payload smoke",
            "uses_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }


def save_tiny_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """函数功能：把 tiny checkpoint payload 写入 tempfile 路径。"""
    if str(path.resolve()).startswith("/data2/"):
        raise AssertionError(f"P16i smoke 不应向 /data2 写 checkpoint：{path}")
    torch.save(dict(payload), path)


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
    expected_shape = (len(feature_batch.sample_keys), len(expert_batch.model_columns))
    if not isinstance(logits_tensor, torch.Tensor):
        raise AssertionError(f"legacy VisualMLPRouter forward 未返回 Tensor：{type(logits_tensor)!r}")
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
    if logits.shape != expected_shape or weights.shape != expected_shape:
        raise AssertionError(f"RouterOutput shape 漂移：logits={logits.shape} weights={weights.shape}")
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
    if result.evaluation_input.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"EvaluationInput sample_keys 未保持 manifest 顺序：{result.evaluation_input.sample_keys}")
    if result.evaluation_input.model_columns != expert_batch.model_columns:
        raise AssertionError(f"EvaluationInput model_columns 漂移：{result.evaluation_input.model_columns}")
    if result.evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")
    for key in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse"):
        if key not in result.summary:
            raise AssertionError(f"summary 缺少 {key}：{result.summary}")
        if not np.isfinite(float(result.summary[key])):
            raise AssertionError(f"summary {key} 不是有限值：{result.summary[key]}")
    rows = result.per_sample_rows
    if len(rows) != len(ordered_sample_keys):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(rows)} expected={len(ordered_sample_keys)}")
    if [row["sample_key"] for row in rows] != list(ordered_sample_keys):
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：{rows}")


def assert_boundary_sources_unchanged() -> None:
    """
    函数功能：
        扫描关键源码边界，确认 P16i 没有把 checkpoint/ViT/正式入口语义下沉到 adapter。
    """
    assert_repo_file(ADAPTER_SOURCE_PATH)
    adapter_source = ADAPTER_SOURCE_PATH.read_text(encoding="utf-8")
    for token in ("/data2", "torch.load", "checkpoint_path", "scaler_path", "ViTModel", "AutoImageProcessor"):
        if token in adapter_source:
            raise AssertionError(f"P16a adapter 源码不应包含 P16i 禁止 token：{token!r}")

    assert_repo_file(RUNTIME_HELPER_SOURCE_PATH)
    helper_source = RUNTIME_HELPER_SOURCE_PATH.read_text(encoding="utf-8")
    for token in ("/data2/", "ViTModel", "AutoImageProcessor", "train_visual_router_online_streaming"):
        if token in helper_source:
            raise AssertionError(f"P16i runtime helper 源码不应包含禁用 token：{token!r}")
    for token in ("from time_router.protocols", "FeatureBatch("):
        if token in helper_source:
            raise AssertionError(f"P16i runtime helper 不应导入或构造 FeatureBatch：{token!r}")
    if "router_state_dict" not in helper_source or "torch.load" not in helper_source:
        raise AssertionError("P16i runtime helper 应显式实现 payload/router_state_dict 读取边界")

    assert_repo_file(VISUAL_SMALL_ENTRYPOINT_PATH)
    visual_small_source = VISUAL_SMALL_ENTRYPOINT_PATH.read_text(encoding="utf-8")
    if "LoadedTorchMLPRouterHeadAdapter" in visual_small_source:
        raise AssertionError("P16i 不应把正式 adapter 接入 scripts/run_stage1_visual_small.py")
    if "SmokeOnlyVisualMLPAdapter" not in visual_small_source:
        raise AssertionError("scripts/run_stage1_visual_small.py 应继续保留 script-local smoke adapter")

    assert_repo_file(STREAMING_ENTRYPOINT_PATH)


def assert_checkpoint_load_uses_only_tempfile(*, expected_paths: set[Path]):
    """
    函数功能：
        patch `torch.load`，确认 runtime helper 只读取本 smoke 创建的 tempfile checkpoint。
    """
    original_torch_load = torch.load
    observed_paths: list[Path] = []

    def guarded_torch_load(path: object, *args: object, **kwargs: object) -> object:
        checkpoint_path = Path(path)
        observed_paths.append(checkpoint_path.resolve())
        if checkpoint_path.resolve() not in {item.resolve() for item in expected_paths}:
            raise AssertionError(f"P16i 只允许读取 tempfile checkpoint：actual={checkpoint_path}")
        return original_torch_load(path, *args, **kwargs)

    return patch.object(torch, "load", side_effect=guarded_torch_load), observed_paths


def assert_negative_cases(
    *,
    router_cls: type[torch.nn.Module],
    input_dim: int,
    output_dim: int,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """函数功能：覆盖缺失 payload、prefix 冲突和 strict load 失败负向用例。"""
    try:
        extract_router_state_dict({"scaler_state": {"metadata_only": True}})
    except KeyError:
        pass
    else:
        raise AssertionError("缺少 router_state_dict 应报错")

    first_key = next(iter(state_dict))
    conflict_state_dict = {
        first_key: state_dict[first_key],
        f"module.{first_key}": state_dict[first_key].clone(),
    }
    try:
        strip_dataparallel_prefix(conflict_state_dict)
    except ValueError:
        pass
    else:
        raise AssertionError("module. 前缀清理后 key 冲突应 fail-fast")

    missing_key_state = dict(state_dict)
    missing_key_state.pop(first_key)
    model = router_cls(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=output_dim, dropout=0.0)
    try:
        load_router_state_dict(model, missing_key_state, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict load missing key 应由 PyTorch 报错")

    unexpected_state = dict(state_dict)
    unexpected_state["unexpected.weight"] = torch.zeros(1)
    model = router_cls(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=output_dim, dropout=0.0)
    try:
        load_router_state_dict(model, unexpected_state, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict load unexpected key 应由 PyTorch 报错")


def run_smoke() -> None:
    """函数功能：执行 P16i runtime-side tiny checkpoint payload loader smoke。"""
    print("开始 Stage 1 P16i legacy VisualMLPRouter checkpoint payload smoke")
    before_outputs = snapshot_run_outputs()
    assert_boundary_sources_unchanged()

    router_cls = load_legacy_visual_mlp_router_class()
    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    expert_batch = load_expert_batch_from_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    provider = VisualPrecomputedFeatureProvider(
        feature_source_path=VISUAL_EMBEDDINGS_PATH,
        source_name="tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv",
        provider_name="P16iVisualPrecomputedFeatureProvider",
    )
    feature_batch = provider.load_batch(ordered_sample_keys)
    assert_feature_and_expert_alignment(feature_batch, expert_batch)
    print("通过：P13b ordered sample_keys、P16c FeatureBatch 和 P13b ExpertBatch 已对齐")

    template_model = router_cls(
        input_dim=int(feature_batch.features.shape[1]),
        hidden_dim=HIDDEN_DIM,
        output_dim=len(expert_batch.model_columns),
        dropout=0.0,
    )
    normal_state_dict = build_fake_state_dict(template_model)
    prefixed_state_dict = {f"module.{key}": value.clone() for key, value in normal_state_dict.items()}

    assert_negative_cases(
        router_cls=router_cls,
        input_dim=int(feature_batch.features.shape[1]),
        output_dim=len(expert_batch.model_columns),
        state_dict=normal_state_dict,
    )
    print("通过：缺失 router_state_dict、module. key 冲突、strict missing/unexpected key 负向用例均 fail-fast")

    with tempfile.TemporaryDirectory(prefix="stage1_p16i_checkpoint_payload_") as temp_dir:
        temp_root = Path(temp_dir)
        normal_checkpoint_path = temp_root / "normal_router_payload.pt"
        prefixed_checkpoint_path = temp_root / "module_prefixed_router_payload.pt"
        save_tiny_checkpoint(normal_checkpoint_path, build_payload(normal_state_dict, payload_name="normal"))
        save_tiny_checkpoint(prefixed_checkpoint_path, build_payload(prefixed_state_dict, payload_name="module_prefixed"))

        guard, observed_load_paths = assert_checkpoint_load_uses_only_tempfile(
            expected_paths={normal_checkpoint_path, prefixed_checkpoint_path}
        )
        with guard:
            normal_payload = load_checkpoint_payload(normal_checkpoint_path, map_location="cpu")
            prefixed_payload = load_checkpoint_payload(prefixed_checkpoint_path, map_location="cpu")

        if set(observed_load_paths) != {normal_checkpoint_path.resolve(), prefixed_checkpoint_path.resolve()}:
            raise AssertionError(f"torch.load 读取路径漂移：{observed_load_paths}")
        for payload, expected_name in ((normal_payload, "normal"), (prefixed_payload, "module_prefixed")):
            if payload["config"]["payload_name"] != expected_name:
                raise AssertionError(f"checkpoint config metadata 漂移：{payload['config']}")
            if payload["scaler_state"]["kind"] != "metadata_only":
                raise AssertionError(f"scaler_state 应只作为 metadata 被识别：{payload['scaler_state']}")
        print("通过：torch.load 只读取 tempfile checkpoint，payload metadata/scaler_state 被识别且未执行 transform")

        normal_extracted_state = extract_router_state_dict(normal_payload)
        prefixed_extracted_state = extract_router_state_dict(prefixed_payload)
        if set(normal_extracted_state) != set(normal_state_dict):
            raise AssertionError(f"normal payload 提取 key 漂移：{sorted(normal_extracted_state)}")
        if set(prefixed_extracted_state) != set(normal_state_dict):
            raise AssertionError(f"module. payload 清理后 key 漂移：{sorted(prefixed_extracted_state)}")

        normal_loaded_model = router_cls(
            input_dim=int(feature_batch.features.shape[1]),
            hidden_dim=HIDDEN_DIM,
            output_dim=len(expert_batch.model_columns),
            dropout=0.0,
        )
        prefixed_loaded_model = router_cls(
            input_dim=int(feature_batch.features.shape[1]),
            hidden_dim=HIDDEN_DIM,
            output_dim=len(expert_batch.model_columns),
            dropout=0.0,
        )
        load_router_state_dict(normal_loaded_model, normal_extracted_state, strict=True)
        load_router_state_dict(prefixed_loaded_model, prefixed_extracted_state, strict=True)
        normal_loaded_model.eval()
        prefixed_loaded_model.eval()

    for key in normal_state_dict:
        np.testing.assert_allclose(
            normal_loaded_model.state_dict()[key].detach().cpu().numpy(),
            prefixed_loaded_model.state_dict()[key].detach().cpu().numpy(),
            rtol=0.0,
            atol=0.0,
        )
    print("通过：normal 与 DataParallel module. 前缀 tiny checkpoint payload 均可 strict load 到 legacy module")

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
        "通过：loaded legacy module 已被 P16a adapter 消费，EvaluationInputAdapter summary/rows 正常生成，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P16i smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P16i legacy VisualMLPRouter checkpoint payload smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
