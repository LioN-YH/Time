#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16a 正式 Visual MLP RouterHead adapter smoke。

输入：
    使用 P13b `sample_manifest.csv` 的 ordered sample_keys、P14b
    `VisualMockFeatureProvider` 生成的 head-ready float32 FeatureBatch，以及 P13b
    `expert_predictions.json` 构造的小型 ExpertBatch。

输出：
    标准输出打印中文检查日志；若 LoadedTorchMLPRouterHeadAdapter 在
    sample/model 保序、float32 边界、logits/weights shape、softmax、负向
    用例或 EvaluationInputAdapter 消费链路上漂移，则抛出 AssertionError。

关键约束：
    本 smoke 只验证“已加载 torch module + head-ready FeatureBatch ->
    RouterOutput”边界。不读取 checkpoint，不处理 scaler，不启动 ViT，不访问
    `/data2`，不迁移或调用正式 Visual Router 训练入口，也不替换 P15c
    script-local smoke adapter。
"""

from __future__ import annotations

import csv
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
from time_router.features import VisualMockFeatureProvider  # noqa: E402
from time_router.models import LoadedTorchMLPRouterHeadAdapter  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
VISUAL_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_mock"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
HISTORY_WINDOWS_PATH = VISUAL_FIXTURE_ROOT / "history_windows.json"
ADAPTER_SOURCE_PATH = REPO_ROOT / "time_router" / "models" / "visual_mlp_adapter.py"
VISUAL_SMALL_ENTRYPOINT_PATH = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

EXPECTED_FEATURE_DIM = 8
ATOL = 1e-6
DISALLOWED_SOURCE_TOKENS = (
    "/data2",
    "torch.load",
    "VisualMLPRouter",
    "ViTModel",
    "AutoImageProcessor",
    "train_visual_router_online_streaming",
    "checkpoint_path",
)


class TinyLoadedMLP(torch.nn.Module):
    """
    类功能：
        P16a smoke 使用的已加载小型 torch MLP fixture。

    输入：
        head-ready float32 features tensor，shape 为 `[sample, feature_dim]`。

    输出：
        二维 logits tensor，shape 为 `[sample, num_experts]`。

    关键约束：
        该模型只在内存中构造，模拟 Runtime 已经完成实例化和权重加载；不读取
        checkpoint，也不代表 legacy VisualMLPRouter 已完成迁移。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"MLP 维度必须为正数：input_dim={input_dim} output_dim={output_dim}")
        hidden_dim = max(4, input_dim + 2)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：将 head-ready features 前向映射为专家 logits。"""
        return self.net(features)


class BadShapeMLP(torch.nn.Module):
    """
    类功能：
        负向用例模型，故意输出错误专家维度以验证 adapter shape 校验。
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：返回 `[sample, 1]` logits，触发 model_columns 维度不匹配。"""
        return torch.zeros((features.shape[0], 1), dtype=features.dtype, device=features.device)


def assert_repo_file(path: Path) -> None:
    """函数功能：确认 fixture/source 位于仓库内且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P16a smoke 不应访问 /data2：{path}")


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


def load_history_windows(path: Path) -> dict[str, list[float]]:
    """
    函数功能：
        读取 P14b Visual mock history windows。

    关键约束：
        这里在 adapter 阶段之前读取 fixture；adapter 本身仍只消费内存
        FeatureBatch，不读取任何外部 cache 或 checkpoint。
    """
    assert_repo_file(path)
    raw_text = path.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    for token in ("checkpoint", "run_dir", "prediction cache", "oracle", "/data2"):
        if token in lowered:
            raise AssertionError(f"history window fixture 不应包含禁用字段或路径：{token}")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict) or not payload:
        raise AssertionError("history_windows.json 必须是非空 sample_key -> window 映射")
    return {str(sample_key): [float(value) for value in window] for sample_key, window in payload.items()}


def load_expert_batch_from_reference(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """
    函数功能：
        用 P13b expert JSON 数值参考构造最小 ExpertBatch。
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
            "provider_name": "P16aInMemoryExpertBatchReference",
            "fixture": "stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
        },
    )


def build_loaded_tiny_mlp(*, input_dim: int, output_dim: int) -> TinyLoadedMLP:
    """
    函数功能：
        构造固定 seed 的小型已加载 MLP。

    关键约束：
        本函数模拟 Runtime 已经持有可前向的 torch module；它不调用 torch.load，
        也不从 checkpoint path 读取权重。
    """
    torch.manual_seed(20260621)
    model = TinyLoadedMLP(input_dim=input_dim, output_dim=output_dim)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, a=-0.12, b=0.12)
    model.eval()
    return model


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def fail_torch_load(*args: object, **kwargs: object) -> object:
    """函数功能：若 adapter 或 smoke 核心路径调用 torch.load，则立即失败。"""
    raise AssertionError(f"P16a adapter 不应调用 torch.load 或读取 checkpoint：args={args} kwargs={kwargs}")


def assert_source_boundaries() -> None:
    """
    函数功能：
        扫描新增 adapter 源码，确认没有引入禁止的 runtime/legacy 依赖。
    """
    assert_repo_file(ADAPTER_SOURCE_PATH)
    source = ADAPTER_SOURCE_PATH.read_text(encoding="utf-8")
    for token in DISALLOWED_SOURCE_TOKENS:
        if token in source:
            raise AssertionError(f"P16a adapter 源码不应包含 {token!r}")

    assert_repo_file(VISUAL_SMALL_ENTRYPOINT_PATH)
    visual_small_source = VISUAL_SMALL_ENTRYPOINT_PATH.read_text(encoding="utf-8")
    if "SmokeOnlyVisualMLPAdapter" not in visual_small_source:
        raise AssertionError("P15c visual small entrypoint 应继续保留 script-local SmokeOnlyVisualMLPAdapter")
    if "LoadedTorchMLPRouterHeadAdapter" in visual_small_source and "--use-loaded-legacy-mlp" not in visual_small_source:
        raise AssertionError("P16a adapter 若接入 visual small entrypoint，必须受显式 loaded legacy CLI 控制")


def assert_feature_and_expert_alignment(feature_batch: FeatureBatch, expert_batch: ExpertBatch) -> None:
    """函数功能：检查 FeatureBatch 与 ExpertBatch 只通过 sample_keys 显式对齐。"""
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


def assert_router_output_contract(
    *,
    router_output: RouterOutput,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> None:
    """函数功能：验证正式 P16a adapter 输出与 FeatureBatch / ExpertBatch 对齐。"""
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

    expected_extra = {
        "adapter_name": "LoadedTorchMLPRouterHeadAdapter",
        "loaded_model_boundary": "runtime_supplied_torch_nn_module",
        "feature_contract": "head_ready_float32_features",
        "loads_checkpoint": False,
        "handles_scaler": False,
        "handles_vit": False,
    }
    if dict(router_output.extra) != expected_extra:
        raise AssertionError(f"RouterOutput.extra 应只记录 adapter 边界 metadata：actual={router_output.extra}")


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
    selected_counts = summary.get("selected_counts")
    if not isinstance(selected_counts, dict) or not selected_counts:
        raise AssertionError(f"summary 应包含 selected_counts：{summary}")

    rows = result.per_sample_rows
    if len(rows) != len(ordered_sample_keys):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(rows)} expected={len(ordered_sample_keys)}")
    if [row["sample_key"] for row in rows] != list(ordered_sample_keys):
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：{rows}")


def expect_raises(label: str, expected_exception: type[Exception], func: Any) -> None:
    """函数功能：执行负向用例，并确认触发预期异常类型。"""
    try:
        func()
    except expected_exception:
        print(f"通过：负向用例触发预期异常：{label}")
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"负向用例 {label} 触发了非预期异常：{type(exc)!r} {exc}") from exc
    raise AssertionError(f"负向用例未触发异常：{label}")


def assert_negative_cases(feature_batch: FeatureBatch, expert_batch: ExpertBatch) -> None:
    """函数功能：覆盖 P16a adapter 的关键失败边界。"""
    adapter = LoadedTorchMLPRouterHeadAdapter(
        model=build_loaded_tiny_mlp(input_dim=EXPECTED_FEATURE_DIM, output_dim=len(expert_batch.model_columns)),
        device="cpu",
    )
    float64_feature_batch = FeatureBatch(
        sample_keys=feature_batch.sample_keys,
        features=np.asarray(feature_batch.features, dtype=np.float64),
        feature_schema=dict(feature_batch.feature_schema),
        extra=dict(feature_batch.extra),
    )
    expect_raises(
        "feature dtype 非 float32",
        ValueError,
        lambda: adapter.predict(float64_feature_batch, expert_batch.model_columns),
    )
    expect_raises(
        "duplicate model_columns",
        ValueError,
        lambda: adapter.predict(feature_batch, tuple(expert_batch.model_columns) + (expert_batch.model_columns[0],)),
    )
    bad_shape_adapter = LoadedTorchMLPRouterHeadAdapter(model=BadShapeMLP(), device="cpu")
    expect_raises(
        "logits shape 与 model_columns 不一致",
        ValueError,
        lambda: bad_shape_adapter.predict(feature_batch, expert_batch.model_columns),
    )


def run_smoke() -> None:
    """函数功能：执行 P16a Visual MLP RouterHead adapter smoke。"""
    print("开始 Stage 1 P16a Visual MLP RouterHead adapter smoke")
    before_outputs = snapshot_run_outputs()
    assert_source_boundaries()
    print("通过：adapter 源码未引入 checkpoint、/data2、ViT、legacy VisualMLPRouter 或正式训练入口")

    ordered_sample_keys = load_manifest_sample_keys(SAMPLE_MANIFEST_PATH)
    history_windows = load_history_windows(HISTORY_WINDOWS_PATH)
    if set(history_windows) != set(ordered_sample_keys):
        raise AssertionError(
            "history windows sample_key 集合必须等于 manifest："
            f"history={sorted(history_windows)} manifest={sorted(ordered_sample_keys)}"
        )
    expert_batch = load_expert_batch_from_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    print("通过：已读取 P13b manifest 顺序，并用 P13b expert JSON 参考构造内存 ExpertBatch")

    provider = VisualMockFeatureProvider(
        history_windows=history_windows,
        history_source_name="stage1_visual_feature_mock_history_window_x",
        source="tests/fixtures/stage1_visual_feature_mock/history_windows.json:in_memory",
    )
    feature_batch = provider.load_batch(ordered_sample_keys)
    assert_feature_and_expert_alignment(feature_batch, expert_batch)
    print("通过：VisualMockFeatureProvider 输出 head-ready float32 FeatureBatch 且与 ExpertBatch sample_keys 对齐")

    loaded_model = build_loaded_tiny_mlp(
        input_dim=int(feature_batch.features.shape[1]),
        output_dim=len(expert_batch.model_columns),
    )
    adapter = LoadedTorchMLPRouterHeadAdapter(model=loaded_model, device=torch.device("cpu"))
    if "run_dir" in adapter.__dict__:
        raise AssertionError("adapter 不应接收或持有 run_dir")

    evaluator = EvaluationInputAdapter()
    with patch.object(torch, "load", side_effect=fail_torch_load):
        router_output = adapter.predict(feature_batch, expert_batch.model_columns)
        result = evaluator.evaluate(expert_batch=expert_batch, router_output=router_output)
        assert_negative_cases(feature_batch, expert_batch)
    print("通过：adapter/evaluator 阶段未调用 torch.load 或读取 checkpoint")

    assert_router_output_contract(
        router_output=router_output,
        feature_batch=feature_batch,
        expert_batch=expert_batch,
    )
    assert_evaluation_result_contract(
        result=result,
        expert_batch=expert_batch,
        router_output=router_output,
        ordered_sample_keys=ordered_sample_keys,
    )
    print(
        "通过：RouterOutput logits/weights 合法，EvaluationInputAdapter summary/rows 可生成且保序，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P16a smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P16a Visual MLP RouterHead adapter smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
