#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P5c protocol dataclass 类型骨架的最小 smoke。

输入：
    无命令行输入；只构造内存对象。

输出：
    标准输出打印中文检查日志；任一 lightweight contract 约束不一致时抛出异常。

关键约束：
    不创建任何文件，不访问 /data2，不访问正式输出目录，不读取训练配置。
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.protocols import (  # noqa: E402
    EvaluationInput,
    ExpertBatch,
    ExperimentProtocolSpec,
    FeatureBatch,
    RouterOutput,
    SplitSpec,
)


class ShapeTrap:
    """
    功能：
        smoke 专用 object，若被协议类型访问 `.shape` 就立即失败。

    关键约束：
        用于确认 P5c dataclass 只是保存引用，不做 array/tensor 数值或形状检查。
    """

    @property
    def shape(self) -> tuple[int, ...]:
        raise AssertionError("P5c protocol dataclass 不应访问 array/tensor 的 .shape")


def assert_distinct_dicts(left: dict, right: dict, field_name: str) -> None:
    """函数功能：验证 default_factory 产生的 dict 不在两个实例之间共享。"""
    if left is right:
        raise AssertionError(f"{field_name} 在两个实例之间共享了同一个 dict")


def run_smoke() -> None:
    """函数功能：执行 P5c protocol types 的纯内存 contract smoke。"""
    print("开始 Stage 1 P5c protocol types smoke")

    sample_keys = ("sample_b", "sample_a", "sample_c")
    model_columns = ("DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster")
    train_splits = ("vali", "train_shadow")
    eval_splits = ("test", "heldout_cell")
    y_pred = ShapeTrap()
    y_true = [["普通 list 也应被原样保存"]]
    features = object()
    logits = object()
    weights = [[0.2, 0.2, 0.2, 0.2, 0.2]]

    split_spec = SplitSpec(name="vali_test", train_splits=train_splits, eval_splits=eval_splits)
    expert_batch = ExpertBatch(
        sample_keys=sample_keys,
        model_columns=model_columns,
        y_pred=y_pred,
        y_true=y_true,
        row_index_metadata={"source": "in_memory_smoke"},
    )
    feature_batch = FeatureBatch(
        sample_keys=sample_keys,
        features=features,
        feature_schema={"schema_name": "smoke_schema_v1"},
    )
    router_output_weights = RouterOutput(
        sample_keys=sample_keys,
        model_columns=model_columns,
        weights=weights,
    )
    evaluation_input_weights = EvaluationInput(
        sample_keys=sample_keys,
        model_columns=model_columns,
        y_pred=y_pred,
        y_true=y_true,
        weights=weights,
    )
    protocol_spec = ExperimentProtocolSpec(
        protocol_name="stage1_p5c_smoke",
        protocol_version="p5c_minimal_types_v1",
        stage="stage1_vali_test_router",
        config_name="96_48_S",
        model_columns=model_columns,
        runtime_contract_version="stage1_runtime_contract_v1",
        split_strategy={"ref": "split_spec"},
        expert_provider={"ref": "expert_provider_spec"},
        feature_provider={"ref": "feature_provider_spec"},
        router_head={"ref": "router_head_spec"},
        evaluator={"ref": "evaluator_spec"},
    )
    constructed = [
        split_spec,
        expert_batch,
        feature_batch,
        router_output_weights,
        evaluation_input_weights,
        protocol_spec,
    ]
    if len(constructed) != 6:
        raise AssertionError("未构造全部 P5c dataclass")
    print("通过：所有 dataclass 均可从 time_router.protocols public API 构造")

    if split_spec.train_splits != train_splits or split_spec.eval_splits != eval_splits:
        raise AssertionError("SplitSpec 未保持 train/eval split tuple 顺序")
    if expert_batch.sample_keys != sample_keys or expert_batch.model_columns != model_columns:
        raise AssertionError("ExpertBatch 未保持 sample_keys/model_columns tuple 顺序")
    if protocol_spec.model_columns != model_columns:
        raise AssertionError("ExperimentProtocolSpec 未保持 model_columns tuple 顺序")
    print("通过：sample_keys、model_columns、train_splits、eval_splits 保持 tuple 顺序")

    split_a = SplitSpec(name="a", train_splits=("vali",), eval_splits=("test",))
    split_b = SplitSpec(name="b", train_splits=("vali",), eval_splits=("test",))
    assert_distinct_dicts(split_a.extra, split_b.extra, "SplitSpec.extra")
    feature_a = FeatureBatch(sample_keys=("a",), features=object())
    feature_b = FeatureBatch(sample_keys=("b",), features=object())
    assert_distinct_dicts(feature_a.feature_schema, feature_b.feature_schema, "FeatureBatch.feature_schema")
    assert_distinct_dicts(feature_a.extra, feature_b.extra, "FeatureBatch.extra")
    protocol_a = ExperimentProtocolSpec(
        protocol_name="a",
        protocol_version="v1",
        stage="stage1",
        config_name="96_48_S",
        model_columns=("DLinear",),
        runtime_contract_version="runtime_v1",
        split_strategy="split",
        expert_provider="expert",
        feature_provider="feature",
        router_head="head",
        evaluator="evaluator",
    )
    protocol_b = ExperimentProtocolSpec(
        protocol_name="b",
        protocol_version="v1",
        stage="stage1",
        config_name="96_48_S",
        model_columns=("DLinear",),
        runtime_contract_version="runtime_v1",
        split_strategy="split",
        expert_provider="expert",
        feature_provider="feature",
        router_head="head",
        evaluator="evaluator",
    )
    assert_distinct_dicts(protocol_a.branch_specific, protocol_b.branch_specific, "ExperimentProtocolSpec.branch_specific")
    assert_distinct_dicts(protocol_a.extra, protocol_b.extra, "ExperimentProtocolSpec.extra")
    split_a.extra["mutated"] = True
    protocol_a.branch_specific["branch"] = "visual"
    if split_b.extra or protocol_b.branch_specific:
        raise AssertionError("default_factory dict 被跨实例污染")
    print("通过：extra / branch_specific / feature_schema 使用独立 default_factory dict")

    RouterOutput(sample_keys=sample_keys, model_columns=model_columns, weights=weights)
    RouterOutput(sample_keys=sample_keys, model_columns=model_columns, logits=logits)
    RouterOutput(sample_keys=sample_keys, model_columns=model_columns, logits=logits, weights=weights)
    RouterOutput(sample_keys=sample_keys, model_columns=model_columns)
    EvaluationInput(sample_keys=sample_keys, model_columns=model_columns, y_pred=y_pred, y_true=y_true, weights=weights)
    EvaluationInput(sample_keys=sample_keys, model_columns=model_columns, y_pred=y_pred, y_true=y_true, logits=logits)
    EvaluationInput(
        sample_keys=sample_keys,
        model_columns=model_columns,
        y_pred=y_pred,
        y_true=y_true,
        logits=logits,
        weights=weights,
    )
    print("通过：RouterOutput 和 EvaluationInput 支持 weights-only、logits-only、两者都有或都为空")

    if expert_batch.y_pred is not y_pred or evaluation_input_weights.y_pred is not y_pred:
        raise AssertionError("array/tensor-like object 未被原样保存")
    if expert_batch.y_true is not y_true or feature_batch.features is not features:
        raise AssertionError("普通 list/object 字段未被原样保存")
    print("通过：array/tensor 字段可保存普通 object/list，且未访问 .shape")

    print("Stage 1 P5c protocol types smoke 通过")


if __name__ == "__main__":
    run_smoke()
