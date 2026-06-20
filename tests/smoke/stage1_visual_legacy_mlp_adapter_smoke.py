#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P14f Visual legacy MLP adapter smoke。

输入：
    使用 P13b `sample_manifest.csv` 的 ordered sample_keys、P14b
    `VisualMockFeatureProvider` 生成的 head-ready float32 FeatureBatch，以及 P13b
    `expert_predictions.json` 的小型专家预测数值参考。

输出：
    标准输出打印中文检查日志；若 smoke-only legacy MLP thin adapter 输出的
    RouterOutput 或 EvaluationInputAdapter 消费链路在保序、shape、softmax、
    有限值、summary/rows 或边界约束上漂移，则抛出 AssertionError。

关键约束：
    本 smoke 只验证 tiny head-ready FeatureBatch + 已加载小型 torch MLP
    state_dict fixture -> RouterOutput -> EvaluationInputAdapter 的内存链路。
    它不新增正式 Visual RouterHead adapter，不接正式 VisualMLPRouter，不加载真实
    checkpoint，不抽真实 ViT provider，不修改正式入口，不访问 `/data2`，不写
    canonical run_dir，不启动训练、pressure 或 full-scale。
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
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter, EvaluationInputAdapterResult  # noqa: E402
from time_router.features import VisualMockFeatureProvider  # noqa: E402
from time_router.protocols import ExpertBatch, FeatureBatch, RouterOutput  # noqa: E402


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
VISUAL_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_mock"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
HISTORY_WINDOWS_PATH = VISUAL_FIXTURE_ROOT / "history_windows.json"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

EXPECTED_FEATURE_DIM = 8
ATOL = 1e-6
DISALLOWED_RUNTIME_TOKENS = (
    "prediction cache",
    "oracle",
    "run_dir",
    "checkpoint",
    "status",
    "metadata",
    "/data2",
)


class SmokeOnlyLegacyMLP(torch.nn.Module):
    """
    类功能：
        P14f smoke-only 小型 legacy MLP 替身。

    输入：
        head-ready float32 features tensor，shape 为 `[sample, feature_dim]`。

    输出：
        未归一化 logits tensor，shape 为 `[sample, num_experts]`。

    关键约束：
        该类只服务 smoke；真实 legacy `VisualMLPRouter`、checkpoint/scaler/device
        策略仍属于正式 Runtime/entrypoint，不能由本 smoke 推断为已迁移。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"MLP 维度必须为正数：input_dim={input_dim} output_dim={output_dim}")
        hidden_dim = max(4, input_dim + 1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：将 head-ready features 前向映射为专家 logits。"""
        return self.net(features)


class SmokeOnlyLegacyMLPAdapter:
    """
    类功能：
        P14f smoke-only thin adapter，将已加载 MLP 输出包装为 RouterOutput。

    输入：
        FeatureBatch、显式 model_columns、Runtime 已加载好的 MLP 和 device。

    输出：
        RouterOutput(sample_keys, model_columns, logits, weights, extra)。

    关键约束：
        adapter 不读取 prediction cache、oracle/error、run_dir 或 checkpoint path；
        不做 scaler fit/load，不决定 dtype/DataParallel，不写 evaluation artifacts。
    """

    adapter_name = "SmokeOnlyLegacyMLPAdapter"

    def __init__(self, *, mlp: torch.nn.Module, device: torch.device | str) -> None:
        self.mlp = mlp
        self.device = torch.device(device)
        self.mlp.to(self.device)
        self.mlp.eval()

    def predict(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """
        函数功能：
            在 torch.inference_mode() 下执行 MLP forward，并 softmax 为融合权重。

        输入：
            feature_batch: 已经由 Runtime/pre-head transform 准备好的 head-ready
                float32 FeatureBatch。
            model_columns: 调用方显式传入且需与 ExpertBatch 对齐的专家列顺序。

        输出：
            RouterOutput；logits/weights 均为 numpy.float32 矩阵。
        """
        columns = tuple(str(model_name) for model_name in model_columns)
        if not columns:
            raise ValueError("legacy MLP adapter 需要非空 model_columns")
        if len(columns) != len(set(columns)):
            raise ValueError(f"legacy MLP adapter 收到重复 model_columns：{columns}")

        features = np.asarray(feature_batch.features)
        if features.dtype != np.float32:
            raise ValueError(f"FeatureBatch.features 必须是 head-ready float32：actual={features.dtype}")
        if features.ndim != 2:
            raise ValueError(f"FeatureBatch.features 必须是二维矩阵：actual_shape={features.shape}")
        if features.shape[0] != len(feature_batch.sample_keys):
            raise ValueError(
                "FeatureBatch.features 样本维度必须等于 sample_keys 数量："
                f"features={features.shape[0]} keys={len(feature_batch.sample_keys)}"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("FeatureBatch.features 包含 NaN 或 Inf")

        feature_tensor = torch.from_numpy(features).to(device=self.device)
        with torch.inference_mode():
            logits_tensor = self.mlp(feature_tensor)
            if logits_tensor.ndim != 2:
                raise ValueError(f"legacy MLP logits 必须是二维矩阵：actual_shape={tuple(logits_tensor.shape)}")
            if logits_tensor.shape[0] != len(feature_batch.sample_keys):
                raise ValueError("legacy MLP logits 样本维度未与 FeatureBatch.sample_keys 对齐")
            if logits_tensor.shape[1] != len(columns):
                raise ValueError(
                    "legacy MLP logits 专家维度必须等于 model_columns 数量："
                    f"logits={logits_tensor.shape[1]} columns={len(columns)}"
                )
            weights_tensor = torch.softmax(logits_tensor, dim=1)

        logits = logits_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        weights = weights_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        return RouterOutput(
            sample_keys=tuple(feature_batch.sample_keys),
            model_columns=columns,
            logits=logits,
            weights=weights,
            extra={
                "adapter_name": self.adapter_name,
                "adapter_scope": "p14f_smoke_only",
                "head_source": "smoke_only_loaded_torch_mlp_state_dict",
                "feature_contract": "head_ready_float32_features",
                "checkpoint_loaded_by": "test_fixture_runtime_not_adapter",
                "scaler_applied_by": "runtime_or_pre_head_not_adapter",
                "device_supplied_by": "runtime_context",
            },
        )


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入 fixture 位于仓库内且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"fixture 文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P14f smoke 不应访问 /data2 fixture：{path}")


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
        history fixture 只保存过去窗口 x；不允许包含 future y、prediction、
        oracle、checkpoint 或 run artifact 字段。
    """
    assert_repo_file(path)
    raw_text = path.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    for token in DISALLOWED_RUNTIME_TOKENS:
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
        history_windows[sample_key] = [float(value) for value in window]
    return history_windows


def load_expert_batch_from_reference(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """
    函数功能：
        用 P13b expert JSON 数值参考构造最小 ExpertBatch。

    关键约束：
        `expert_predictions.json` 只是 small fixture 数值参考，不是正式
        prediction backend schema；正式路径仍应由 prediction backend/provider 提供。
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
        sample = sample_by_key[sample_key]
        y_true_rows.append(np.asarray(sample["y_true"], dtype=np.float32))
        y_pred = np.asarray(sample["y_pred"], dtype=np.float32)
        if y_pred.shape[0] != len(model_columns):
            raise AssertionError(f"{sample_key} y_pred 专家维与 model_columns 不一致：shape={y_pred.shape}")
        y_pred_rows.append(y_pred)

    y_true = np.stack(y_true_rows, axis=0).astype(np.float32)
    y_pred = np.stack(y_pred_rows, axis=0).astype(np.float32)
    if y_pred.shape[2:] != y_true.shape[1:]:
        raise AssertionError(f"ExpertBatch y_pred/y_true target shape 不一致：{y_pred.shape} vs {y_true.shape}")
    return ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in ordered_sample_keys),
        model_columns=model_columns,
        y_pred=y_pred,
        y_true=y_true,
        row_index_metadata={
            "source": "tests/fixtures/stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
            "formal_backend_schema": False,
        },
        extra={
            "provider_name": "P14fInMemoryExpertBatchReference",
            "fixture": "stage1_real_derived_small/expert_predictions.json",
            "reference_only": True,
            "formal_backend_schema": False,
        },
    )


def build_loaded_smoke_mlp_state_dict(*, input_dim: int, output_dim: int) -> dict[str, torch.Tensor]:
    """
    函数功能：
        构造固定 seed 的小型 MLP state_dict fixture，并模拟 Runtime 已完成加载。

    关键约束：
        本函数不读 checkpoint 文件、不调用 torch.load；state_dict 只作为内存 fixture
        证明 adapter 接收“已加载 head”即可工作。
    """
    torch.manual_seed(20260620)
    fixture_model = SmokeOnlyLegacyMLP(input_dim=input_dim, output_dim=output_dim)
    for parameter in fixture_model.parameters():
        torch.nn.init.uniform_(parameter, a=-0.15, b=0.15)
    return {name: tensor.detach().clone() for name, tensor in fixture_model.state_dict().items()}


def load_smoke_mlp_from_state_dict(*, input_dim: int, output_dim: int) -> SmokeOnlyLegacyMLP:
    """函数功能：从内存 state_dict fixture 构造已加载的 smoke-only MLP。"""
    loaded_model = SmokeOnlyLegacyMLP(input_dim=input_dim, output_dim=output_dim)
    state_dict = build_loaded_smoke_mlp_state_dict(input_dim=input_dim, output_dim=output_dim)
    missing_keys, unexpected_keys = loaded_model.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise AssertionError(f"state_dict fixture 加载异常：missing={missing_keys} unexpected={unexpected_keys}")
    return loaded_model


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


@contextmanager
def forbid_adapter_and_evaluator_io() -> Iterator[None]:
    """
    函数功能：
        在 adapter/evaluator 阶段阻断文件 IO、np IO 和 torch.load。

    关键约束：
        manifest、history、expert JSON 和内存 state_dict fixture 均在前置阶段准备；
        adapter/evaluator 只能消费内存对象，不应读取 checkpoint、prediction、
        oracle、run_dir 或 `/data2`。
    """
    with patch.object(builtins, "open", side_effect=fail_runtime_io), patch.object(
        Path, "open", side_effect=fail_runtime_io
    ), patch.object(Path, "read_text", side_effect=fail_runtime_io), patch.object(
        np, "load", side_effect=fail_runtime_io
    ), patch.object(
        np, "save", side_effect=fail_runtime_io
    ), patch.object(
        np, "savez", side_effect=fail_runtime_io
    ), patch.object(
        torch, "load", side_effect=fail_runtime_io
    ):
        yield


def fail_runtime_io(*args: object, **kwargs: object) -> object:
    """函数功能：统一失败入口，说明 P14f adapter/evaluator 阶段不允许文件 IO。"""
    raise AssertionError(f"P14f adapter/evaluator 阶段不应访问文件系统、np IO 或 torch.load：args={args} kwargs={kwargs}")


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
    """函数功能：验证 smoke-only legacy MLP adapter 输出与协议对象对齐。"""
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

    extra = dict(router_output.extra)
    expected_extra = {
        "adapter_name": "SmokeOnlyLegacyMLPAdapter",
        "adapter_scope": "p14f_smoke_only",
        "head_source": "smoke_only_loaded_torch_mlp_state_dict",
        "feature_contract": "head_ready_float32_features",
        "checkpoint_loaded_by": "test_fixture_runtime_not_adapter",
        "scaler_applied_by": "runtime_or_pre_head_not_adapter",
        "device_supplied_by": "runtime_context",
    }
    if extra != expected_extra:
        raise AssertionError(f"RouterOutput.extra 只能记录 adapter/head lineage：actual={extra}")


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
    if evaluation_input.y_pred is not expert_batch.y_pred or evaluation_input.y_true is not expert_batch.y_true:
        raise AssertionError("EvaluationInput 必须复用 ExpertBatch.y_pred/y_true，不应复制或重读")
    if evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")

    summary = result.summary
    if summary["num_samples"] != len(ordered_sample_keys):
        raise AssertionError(f"summary sample_count 漂移：{summary}")
    if summary["num_experts"] != len(expert_batch.model_columns):
        raise AssertionError(f"summary expert count 漂移：{summary}")
    if summary["model_columns"] != list(expert_batch.model_columns):
        raise AssertionError(f"summary model_columns 漂移：{summary}")
    for metric_name in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse", "mean_entropy", "mean_max_weight"):
        metric_value = float(summary[metric_name])
        if not np.isfinite(metric_value):
            raise AssertionError(f"summary {metric_name} 不是有限值：{metric_value}")

    rows = result.per_sample_rows
    if len(rows) != len(ordered_sample_keys):
        raise AssertionError(f"per-sample rows 数量漂移：actual={len(rows)} expected={len(ordered_sample_keys)}")
    if [row["sample_key"] for row in rows] != list(ordered_sample_keys):
        raise AssertionError(f"per-sample rows sample_key 顺序漂移：{rows}")


def run_smoke() -> None:
    """函数功能：执行 P14f Visual legacy MLP adapter smoke。"""
    print("开始 Stage 1 P14f Visual legacy MLP adapter smoke")
    before_outputs = snapshot_run_outputs()

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

    mlp = load_smoke_mlp_from_state_dict(
        input_dim=int(feature_batch.features.shape[1]),
        output_dim=len(expert_batch.model_columns),
    )
    adapter = SmokeOnlyLegacyMLPAdapter(mlp=mlp, device=torch.device("cpu"))
    evaluator = EvaluationInputAdapter()
    with forbid_adapter_and_evaluator_io():
        router_output = adapter.predict(feature_batch, expert_batch.model_columns)
        result = evaluator.evaluate(expert_batch=expert_batch, router_output=router_output)
    print("通过：adapter/evaluator 阶段未调用文件 IO、np.load/np.save 或 torch.load")

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
        raise AssertionError(f"P14f smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P14f Visual legacy MLP adapter smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
