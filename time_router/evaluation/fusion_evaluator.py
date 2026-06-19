"""
文件功能：
    Stage 1 P6b 最小 FusionEvaluator adapter。

设计边界：
    本模块只把 ExpertBatch + RouterOutput / EvaluationInput 适配为
    time_router.evaluation public API 可消费的内存对象，并复算 summary 与
    per-sample rows。它不读取 prediction cache、manifest、packed npy、
    oracle/TSF，也不创建 run_dir 或写任何输出文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from time_router.evaluation.metrics import FusionMetricsResult, hard_top1_fusion, raw_soft_fusion
from time_router.evaluation.prediction_rows import build_per_sample_fusion_rows
from time_router.evaluation.summary import build_fusion_summary
from time_router.protocols import EvaluationInput, ExpertBatch, RouterOutput


@dataclass(frozen=True)
class FusionEvaluationResult:
    """
    类功能：
        承载 FusionEvaluator adapter 的纯内存复算结果。

    字段说明：
        evaluation_input: 适配后的 EvaluationInput，保留 sample/model 顺序。
        hard_result/raw_soft_result: 复用 public API 得到的 fusion 结果对象。
        summary: build_fusion_summary 返回的 dict。
        per_sample_rows: build_per_sample_fusion_rows 返回的逐样本 rows。
        diagnostics: 轻量 lineage/adapter 信息，只来自输入对象而非额外 IO。
    """

    evaluation_input: EvaluationInput
    hard_result: FusionMetricsResult
    raw_soft_result: FusionMetricsResult
    summary: dict[str, Any]
    per_sample_rows: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class FusionEvaluator:
    """
    类功能：
        将 P5c protocol types 适配到 evaluation public API。

    关键约束：
        该 adapter 不拥有数据读取能力；调用方必须已经准备好 ExpertBatch 与
        RouterOutput，或直接传入 EvaluationInput。
    """

    evaluator_name = "FusionEvaluator"

    def build_evaluation_input(
        self,
        *,
        expert_batch: ExpertBatch,
        router_output: RouterOutput,
        extra: dict[str, Any] | None = None,
    ) -> EvaluationInput:
        """
        函数功能：
            从 ExpertBatch + RouterOutput 构造最小 EvaluationInput。

        输入：
            expert_batch: 包含 sample_keys/model_columns/y_pred/y_true 的专家批次。
            router_output: 包含同序 sample_keys/model_columns 和 weights/logits。
            extra: 调用方可附加的轻量 metadata。

        输出：
            EvaluationInput；y_pred/y_true 原样引用 ExpertBatch，不重新读取 cache。
        """
        _assert_same_tuple(
            "sample_keys",
            tuple(expert_batch.sample_keys),
            tuple(router_output.sample_keys),
        )
        _assert_same_tuple(
            "model_columns",
            tuple(expert_batch.model_columns),
            tuple(router_output.model_columns),
        )

        merged_extra: dict[str, Any] = {
            "adapter_name": self.evaluator_name,
            "expert_extra": dict(expert_batch.extra),
            "router_extra": dict(router_output.extra),
        }
        if expert_batch.row_index_metadata is not None:
            # 只转递 provider 已经给出的轻量 lineage，不在 adapter 内读取外部文件。
            merged_extra["row_index_metadata"] = expert_batch.row_index_metadata
        if extra:
            merged_extra.update(extra)

        return EvaluationInput(
            sample_keys=tuple(expert_batch.sample_keys),
            model_columns=tuple(expert_batch.model_columns),
            y_pred=expert_batch.y_pred,
            y_true=expert_batch.y_true,
            logits=router_output.logits,
            weights=router_output.weights,
            extra=merged_extra,
        )

    def evaluate(
        self,
        *,
        expert_batch: ExpertBatch | None = None,
        router_output: RouterOutput | None = None,
        evaluation_input: EvaluationInput | None = None,
    ) -> FusionEvaluationResult:
        """
        函数功能：
            复算 hard top-1、raw soft fusion、summary 和 per-sample rows。

        输入：
            可直接传入 evaluation_input；或传入 expert_batch + router_output 让
            adapter 先构造 EvaluationInput。

        输出：
            FusionEvaluationResult，所有结果均保存在内存中。
        """
        if evaluation_input is None:
            if expert_batch is None or router_output is None:
                raise ValueError("必须传入 evaluation_input，或同时传入 expert_batch 与 router_output")
            evaluation_input = self.build_evaluation_input(expert_batch=expert_batch, router_output=router_output)
        elif expert_batch is not None or router_output is not None:
            raise ValueError("传入 evaluation_input 时不要同时传入 expert_batch/router_output，避免口径歧义")

        if evaluation_input.weights is None:
            raise ValueError("FusionEvaluator 需要 EvaluationInput.weights 才能复算 fusion")

        hard_result = hard_top1_fusion(
            y_pred=evaluation_input.y_pred,
            y_true=evaluation_input.y_true,
            weights=evaluation_input.weights,
            model_columns=evaluation_input.model_columns,
        )
        raw_soft_result = raw_soft_fusion(
            y_pred=evaluation_input.y_pred,
            y_true=evaluation_input.y_true,
            weights=evaluation_input.weights,
            model_columns=evaluation_input.model_columns,
        )
        summary = build_fusion_summary(
            model_columns=evaluation_input.model_columns,
            hard_result=hard_result,
            raw_soft_result=raw_soft_result,
            weights=evaluation_input.weights,
        )
        per_sample_rows = build_per_sample_fusion_rows(
            sample_keys=evaluation_input.sample_keys,
            model_columns=evaluation_input.model_columns,
            hard_result=hard_result,
            raw_soft_result=raw_soft_result,
            y_true=evaluation_input.y_true,
            weights=evaluation_input.weights,
        )

        diagnostics = {
            "adapter_name": self.evaluator_name,
            "sample_keys": tuple(evaluation_input.sample_keys),
            "model_columns": tuple(evaluation_input.model_columns),
        }
        if evaluation_input.extra:
            diagnostics["lineage"] = dict(evaluation_input.extra)

        return FusionEvaluationResult(
            evaluation_input=evaluation_input,
            hard_result=hard_result,
            raw_soft_result=raw_soft_result,
            summary=summary,
            per_sample_rows=per_sample_rows,
            diagnostics=diagnostics,
        )


def _assert_same_tuple(name: str, left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    """函数功能：检查 adapter 输入两侧的顺序完全一致。"""
    if left != right:
        raise ValueError(f"{name} 不一致：expert_batch={left} router_output={right}")
