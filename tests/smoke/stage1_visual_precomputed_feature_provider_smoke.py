#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16c precomputed/head-ready Visual FeatureProvider minimal smoke。

输入：
    使用 P13b real-derived sample_manifest.csv 的 ordered sample_keys、P16c
    `visual_embeddings.csv` 预计算 head-ready visual embedding fixture，以及 P13b
    `expert_predictions.json` 构造的小型 ExpertBatch。

输出：
    标准输出打印中文检查日志；若 VisualPrecomputedFeatureProvider、P16a
    LoadedTorchMLPRouterHeadAdapter 或 EvaluationInputAdapter 的 sample_key 保序、
    dtype/shape、schema metadata、负向边界或 summary/rows 生成发生漂移，则抛错。

关键约束：
    本 smoke 只验证 precomputed/head-ready feature fixture -> FeatureBatch ->
    LoadedTorchMLPRouterHeadAdapter -> EvaluationInputAdapter 的最小边界。不接真实
    ViT，不做 pseudo image，不处理 scaler，不读取 checkpoint，不迁移正式入口，
    不访问 `/data2`，不创建 run_dir。
"""

from __future__ import annotations

import csv
import json
import math
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


P13B_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small"
VISUAL_PRECOMPUTED_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_precomputed_small"
SAMPLE_MANIFEST_PATH = P13B_FIXTURE_ROOT / "sample_manifest.csv"
EXPERT_REFERENCE_PATH = P13B_FIXTURE_ROOT / "expert_predictions.json"
VISUAL_EMBEDDINGS_PATH = VISUAL_PRECOMPUTED_FIXTURE_ROOT / "visual_embeddings.csv"
PROVIDER_SOURCE_PATH = REPO_ROOT / "time_router" / "features" / "visual_precomputed.py"
RUN_OUTPUTS_ROOT = REPO_ROOT / "experiment_logs" / "run_outputs"

EXPECTED_FEATURE_DIM = 8
ATOL = 1e-6
DISALLOWED_SOURCE_TOKENS = (
    "/data2",
    "torch.load",
    "ViTModel",
    "AutoImageProcessor",
    "VisualMLPRouter",
    "train_visual_router_online_streaming",
    "train_timefuse_fusor_streaming",
)


class TinyLoadedMLP(torch.nn.Module):
    """
    类功能：
        P16c smoke 使用的内存小型 torch MLP。

    输入：
        已经由 provider 输出的 head-ready float32 feature tensor。

    输出：
        二维专家 logits tensor。

    关键约束：
        该模型模拟 Runtime 已经持有可前向 module；不读取 checkpoint，不代表真实
        VisualMLPRouter 已完成迁移。
    """

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden_dim = max(4, input_dim + 1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：将 head-ready features 前向映射为专家 logits。"""
        return self.net(features)


def assert_repo_file(path: Path) -> None:
    """函数功能：确认输入文件存在于仓库内且不是 `/data2` 外部产物。"""
    if not path.is_file():
        raise AssertionError(f"文件缺失：{path}")
    resolved = str(path.resolve())
    if resolved.startswith("/data2/") or resolved == "/data2":
        raise AssertionError(f"P16c smoke 不应访问 /data2：{path}")


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    """函数功能：读取 P13b manifest 行，用于恢复 ordered sample_keys 和 split。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("sample_manifest.csv 不应为空")
    sample_keys = [str(row["sample_key"]) for row in rows]
    if len(sample_keys) != len(set(sample_keys)):
        raise AssertionError(f"sample_manifest.csv 存在重复 sample_key：{sample_keys}")
    return rows


def load_visual_embedding_reference(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, np.ndarray]]:
    """
    函数功能：
        读取 P16c visual embedding fixture，并返回文件行顺序和数值参考。

    关键约束：
        fixture 只允许包含 sample_key 与 feature_ 列，不允许出现 checkpoint、ViT、
        scaler、run_dir、prediction 或 `/data2` 等边界外信息。
    """
    assert_repo_file(path)
    raw_text = path.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    for token in ("checkpoint", "vit", "scaler", "run_dir", "oracle", "prediction", "/data2"):
        if token in lowered:
            raise AssertionError(f"visual embedding fixture 不应包含禁用字段或路径：{token}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError("visual_embeddings.csv 缺少表头")
        feature_columns = tuple(column for column in reader.fieldnames if column.startswith("feature_"))
        if len(feature_columns) != EXPECTED_FEATURE_DIM:
            raise AssertionError(f"feature 列数量漂移：{feature_columns}")
        row_order: list[str] = []
        reference: dict[str, np.ndarray] = {}
        for row in reader:
            sample_key = str(row["sample_key"])
            if not sample_key:
                raise AssertionError("visual_embeddings.csv 存在空 sample_key")
            if sample_key in reference:
                raise AssertionError(f"visual_embeddings.csv 存在重复 sample_key：{sample_key}")
            values = np.asarray([float(row[column]) for column in feature_columns], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise AssertionError(f"visual_embeddings.csv 存在非有限值：sample_key={sample_key}")
            row_order.append(sample_key)
            reference[sample_key] = values
    return tuple(row_order), feature_columns, reference


def load_expert_batch_from_reference(path: Path, ordered_sample_keys: Sequence[str]) -> ExpertBatch:
    """函数功能：用 P13b expert JSON 数值参考构造最小 ExpertBatch。"""
    assert_repo_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_columns = tuple(str(model_name) for model_name in payload["model_columns"])
    if not model_columns or len(model_columns) != len(set(model_columns)):
        raise AssertionError(f"model_columns 异常：{model_columns}")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise AssertionError("expert_predictions.json 缺少 samples list")
    sample_by_key: dict[str, Mapping[str, Any]] = {}
    for sample in samples:
        sample_key = str(sample["sample_key"])
        if sample_key in sample_by_key:
            raise AssertionError(f"expert_predictions.json 存在重复 sample_key：{sample_key}")
        sample_by_key[sample_key] = sample
    if set(sample_by_key) != set(ordered_sample_keys):
        raise AssertionError("expert_predictions.json sample_key 集合未与 manifest 对齐")

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
        },
        extra={
            "provider_name": "P16cInMemoryExpertBatchReference",
            "fixture": "stage1_real_derived_small/expert_predictions.json",
        },
    )


def build_loaded_tiny_mlp(*, input_dim: int, output_dim: int) -> TinyLoadedMLP:
    """
    函数功能：
        构造固定 seed 的已加载小型 MLP。

    关键约束：
        不调用 torch.load；本函数只初始化内存 fixture 权重。
    """
    torch.manual_seed(20260621)
    model = TinyLoadedMLP(input_dim=input_dim, output_dim=output_dim)
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, a=-0.10, b=0.10)
    model.eval()
    return model


def snapshot_run_outputs() -> set[str]:
    """函数功能：记录 run_outputs 一层目录名，用于确认 smoke 不创建 run_dir。"""
    if not RUN_OUTPUTS_ROOT.exists():
        return set()
    return {path.name for path in RUN_OUTPUTS_ROOT.iterdir()}


def fail_torch_load(*args: object, **kwargs: object) -> object:
    """函数功能：若 provider/adapter/evaluator 阶段读取 checkpoint，则立即失败。"""
    raise AssertionError(f"P16c 不应调用 torch.load 或读取 checkpoint：args={args} kwargs={kwargs}")


def assert_source_boundaries() -> None:
    """函数功能：扫描 P16c provider 源码，确认没有引入禁止 runtime/legacy 依赖。"""
    assert_repo_file(PROVIDER_SOURCE_PATH)
    source = PROVIDER_SOURCE_PATH.read_text(encoding="utf-8")
    for token in DISALLOWED_SOURCE_TOKENS:
        if token in source:
            raise AssertionError(f"P16c provider 源码不应包含 {token!r}")


def assert_feature_batch_contract(
    *,
    feature_batch: FeatureBatch,
    ordered_sample_keys: Sequence[str],
    feature_columns: Sequence[str],
    reference_by_key: Mapping[str, np.ndarray],
) -> None:
    """函数功能：验证 VisualPrecomputedFeatureProvider 输出 FeatureBatch contract。"""
    if not isinstance(feature_batch, FeatureBatch):
        raise AssertionError(f"provider 未返回 FeatureBatch：actual={type(feature_batch)!r}")
    if feature_batch.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"FeatureBatch.sample_keys 未保持 manifest 顺序：{feature_batch.sample_keys}")
    if tuple(feature_batch.features.shape) != (len(ordered_sample_keys), EXPECTED_FEATURE_DIM):
        raise AssertionError(f"features shape 漂移：actual={feature_batch.features.shape}")
    if feature_batch.features.dtype != np.float32:
        raise AssertionError(f"features dtype 必须为 float32：actual={feature_batch.features.dtype}")
    expected = np.stack([reference_by_key[str(sample_key)] for sample_key in ordered_sample_keys], axis=0)
    np.testing.assert_allclose(feature_batch.features, expected, rtol=0.0, atol=0.0)

    schema = feature_batch.feature_schema
    expected_schema_items = {
        "provider_name": "VisualPrecomputedFeatureProvider",
        "feature_schema_name": "visual_precomputed_head_ready_v1",
        "feature_dim": EXPECTED_FEATURE_DIM,
        "feature_columns": tuple(feature_columns),
        "head_ready": True,
        "loads_real_vit": False,
        "handles_scaler": False,
        "precomputed": True,
        "dtype": "float32",
    }
    for key, expected_value in expected_schema_items.items():
        if schema.get(key) != expected_value:
            raise AssertionError(f"feature_schema[{key!r}] 漂移：actual={schema.get(key)!r} expected={expected_value!r}")
    if "run_dir" in feature_batch.extra:
        raise AssertionError(f"FeatureBatch.extra 不应包含 run_dir：{feature_batch.extra}")
    if feature_batch.extra.get("num_available_rows") != len(reference_by_key):
        raise AssertionError(f"num_available_rows 漂移：{feature_batch.extra}")


def assert_router_output_contract(
    *,
    router_output: RouterOutput,
    feature_batch: FeatureBatch,
    expert_batch: ExpertBatch,
) -> None:
    """函数功能：验证 P16a adapter 可消费 P16c FeatureBatch 并输出 RouterOutput。"""
    if not isinstance(router_output, RouterOutput):
        raise AssertionError(f"adapter 未返回 RouterOutput：actual={type(router_output)!r}")
    if router_output.sample_keys != feature_batch.sample_keys:
        raise AssertionError(f"RouterOutput sample_keys 未保持 FeatureBatch 顺序：{router_output.sample_keys}")
    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError(f"RouterOutput model_columns 未与 ExpertBatch 对齐：{router_output.model_columns}")
    expected_shape = (len(feature_batch.sample_keys), len(expert_batch.model_columns))
    logits = np.asarray(router_output.logits)
    weights = np.asarray(router_output.weights)
    if logits.shape != expected_shape or weights.shape != expected_shape:
        raise AssertionError(f"RouterOutput shape 漂移：logits={logits.shape} weights={weights.shape}")
    if logits.dtype != np.float32 or weights.dtype != np.float32:
        raise AssertionError(f"RouterOutput dtype 应为 float32：{logits.dtype} {weights.dtype}")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(weights)):
        raise AssertionError("RouterOutput logits/weights 包含 NaN 或 Inf")
    np.testing.assert_allclose(np.sum(weights, axis=1), np.ones(expected_shape[0]), rtol=0.0, atol=ATOL)


def assert_evaluation_result_contract(
    *,
    result: EvaluationInputAdapterResult,
    expert_batch: ExpertBatch,
    router_output: RouterOutput,
    ordered_sample_keys: Sequence[str],
) -> None:
    """函数功能：检查 EvaluationInputAdapter 生成 summary/rows 且 per-sample rows 保序。"""
    if result.evaluation_input.sample_keys != tuple(ordered_sample_keys):
        raise AssertionError(f"EvaluationInput sample_keys 未保持 manifest 顺序：{result.evaluation_input.sample_keys}")
    if result.evaluation_input.model_columns != expert_batch.model_columns:
        raise AssertionError(f"EvaluationInput model_columns 漂移：{result.evaluation_input.model_columns}")
    if result.evaluation_input.weights is not router_output.weights:
        raise AssertionError("EvaluationInput 必须复用 RouterOutput.weights")
    for key in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse"):
        if key not in result.summary or not np.isfinite(float(result.summary[key])):
            raise AssertionError(f"summary {key} 异常：{result.summary}")
    rows = result.per_sample_rows
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


def write_duplicate_fixture(path: Path, source_rows: Sequence[dict[str, str]]) -> None:
    """函数功能：写出临时重复 sample_key fixture，用于验证 provider fail-fast。"""
    rows = list(source_rows)
    duplicate_row = dict(rows[0])
    duplicate_row["feature_0"] = "9.999"
    rows.append(duplicate_row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_nonfinite_fixture(path: Path, source_rows: Sequence[dict[str, str]]) -> None:
    """函数功能：写出临时非有限 feature fixture，用于验证 provider 数值校验。"""
    rows = [dict(row) for row in source_rows]
    rows[0]["feature_1"] = str(math.inf)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """函数功能：读取 CSV 原始行，供临时负向 fixture 改写。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_smoke() -> None:
    """函数功能：执行 P16c Visual precomputed FeatureProvider smoke。"""
    print("开始 Stage 1 P16c Visual precomputed FeatureProvider smoke")
    before_outputs = snapshot_run_outputs()
    assert_source_boundaries()
    print("通过：provider 源码未引入 /data2、checkpoint、ViT、scaler、run_dir 或正式训练入口依赖")

    manifest_rows = load_manifest_rows(SAMPLE_MANIFEST_PATH)
    ordered_sample_keys = tuple(str(row["sample_key"]) for row in manifest_rows)
    test_sample_keys = tuple(str(row["sample_key"]) for row in manifest_rows if row["split"] == "test")
    if len(test_sample_keys) != 2:
        raise AssertionError(f"P13b fixture test split 数量漂移：{test_sample_keys}")

    fixture_row_order, feature_columns, reference_by_key = load_visual_embedding_reference(VISUAL_EMBEDDINGS_PATH)
    if set(reference_by_key) != set(ordered_sample_keys):
        raise AssertionError("visual embedding fixture sample_key 集合未与 P13b manifest 对齐")
    if not set(test_sample_keys).issubset(reference_by_key):
        raise AssertionError(f"visual embedding fixture 未覆盖 test split sample_keys：{test_sample_keys}")
    if fixture_row_order == ordered_sample_keys:
        raise AssertionError("visual embedding fixture 行顺序应刻意不同于 manifest 顺序")
    print("通过：precomputed visual fixture 覆盖 P13b manifest/test split，且行顺序故意打乱")

    expert_batch = load_expert_batch_from_reference(EXPERT_REFERENCE_PATH, ordered_sample_keys)
    provider = VisualPrecomputedFeatureProvider(
        feature_source_path=VISUAL_EMBEDDINGS_PATH,
        feature_schema_name="visual_precomputed_head_ready_v1",
        source_name="tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv",
    )
    if "run_dir" in provider.__dict__:
        raise AssertionError("VisualPrecomputedFeatureProvider 不应接收或持有 run_dir")

    feature_batch = provider.load_batch(ordered_sample_keys)
    assert_feature_batch_contract(
        feature_batch=feature_batch,
        ordered_sample_keys=ordered_sample_keys,
        feature_columns=feature_columns,
        reference_by_key=reference_by_key,
    )
    print("通过：provider 输出 FeatureBatch，按 manifest 保序，features=(4, 8)，dtype=float32")

    with tempfile.TemporaryDirectory(prefix="stage1_p16c_visual_precomputed_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        source_rows = load_csv_rows(VISUAL_EMBEDDINGS_PATH)
        duplicate_path = tmp_root / "duplicate_visual_embeddings.csv"
        nonfinite_path = tmp_root / "nonfinite_visual_embeddings.csv"
        write_duplicate_fixture(duplicate_path, source_rows)
        write_nonfinite_fixture(nonfinite_path, source_rows)
        expect_raises(
            "missing sample_key",
            KeyError,
            lambda: provider.load_batch(tuple(ordered_sample_keys) + ("missing::sample",)),
        )
        expect_raises(
            "duplicate sample_key fixture",
            ValueError,
            lambda: VisualPrecomputedFeatureProvider(feature_source_path=duplicate_path),
        )
        expect_raises(
            "non-finite feature fixture",
            ValueError,
            lambda: VisualPrecomputedFeatureProvider(feature_source_path=nonfinite_path),
        )

    loaded_model = build_loaded_tiny_mlp(
        input_dim=int(feature_batch.features.shape[1]),
        output_dim=len(expert_batch.model_columns),
    )
    adapter = LoadedTorchMLPRouterHeadAdapter(model=loaded_model, device=torch.device("cpu"))
    evaluator = EvaluationInputAdapter()
    with patch.object(torch, "load", side_effect=fail_torch_load):
        router_output = adapter.predict(feature_batch, expert_batch.model_columns)
        result = evaluator.evaluate(expert_batch=expert_batch, router_output=router_output)
    print("通过：P16a adapter 可消费 P16c FeatureBatch，且阶段内未调用 torch.load")

    assert_router_output_contract(router_output=router_output, feature_batch=feature_batch, expert_batch=expert_batch)
    assert_evaluation_result_contract(
        result=result,
        expert_batch=expert_batch,
        router_output=router_output,
        ordered_sample_keys=ordered_sample_keys,
    )
    print(
        "通过：EvaluationInputAdapter summary/rows 可生成且 per-sample rows 保持 sample_key 顺序，"
        f"hard_mae={result.summary['hard_mae']:.9f}，raw_soft_mae={result.summary['raw_soft_mae']:.9f}"
    )

    after_outputs = snapshot_run_outputs()
    if after_outputs != before_outputs:
        raise AssertionError(f"P16c smoke 不应创建 canonical run_dir 或 run_outputs 目录：新增={sorted(after_outputs - before_outputs)}")
    print("完成：Stage 1 P16c Visual precomputed FeatureProvider smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
