"""
文件功能：
    Stage 1 P6c FusionEvaluator 兼容包装。

设计边界：
    EvaluationInputAdapter 是 canonical adapter。本模块只保留较早 P6b 命名下的
    FusionEvaluator / FusionEvaluationResult 兼容 API，并委托
    EvaluationInputAdapter 完成 EvaluationInput 构造和 public API 复算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from time_router.evaluation.evaluation_input_adapter import EvaluationInputAdapter
from time_router.evaluation.metrics import FusionMetricsResult
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
        兼容旧 `FusionEvaluator` 调用方。

    关键约束：
        新代码应优先使用 `EvaluationInputAdapter`；本类不再维护独立 fusion
        逻辑，避免两套 adapter 并行生长。
    """

    evaluator_name = "FusionEvaluator"

    def __init__(self, adapter: EvaluationInputAdapter | None = None) -> None:
        self._adapter = adapter or EvaluationInputAdapter()

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
        compat_extra = {"compat_adapter_name": self.evaluator_name}
        if extra:
            compat_extra.update(extra)
        return self._adapter.build_evaluation_input(
            expert_batch=expert_batch,
            router_output=router_output,
            extra=compat_extra,
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

        adapter_result = self._adapter.evaluate_input(evaluation_input=evaluation_input)

        diagnostics = {
            "adapter_name": self.evaluator_name,
            "canonical_adapter_name": self._adapter.adapter_name,
            "sample_keys": tuple(evaluation_input.sample_keys),
            "model_columns": tuple(evaluation_input.model_columns),
        }
        if evaluation_input.extra:
            diagnostics["lineage"] = dict(evaluation_input.extra)

        return FusionEvaluationResult(
            evaluation_input=evaluation_input,
            hard_result=adapter_result.hard_result,
            raw_soft_result=adapter_result.raw_soft_result,
            summary=adapter_result.summary,
            per_sample_rows=adapter_result.per_sample_rows,
            diagnostics=diagnostics,
        )
