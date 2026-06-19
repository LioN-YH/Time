"""
文件功能：
    Stage 1 P6b 最小 EvaluationInput adapter。

设计边界：
    本模块只把当前 Stage 1 canonical experiment 的 ExpertBatch 与
    RouterOutput.weights 或显式 fusion weights 包装为 EvaluationInput，并调用
    time_router.evaluation public API 生成内存 summary 和 per-sample rows。
    它不读取 prediction cache、manifest、packed npy、oracle/TSF，不创建 run_dir，
    也不写任何 CSV/JSON/Parquet 文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from time_router.evaluation.metrics import FusionMetricsResult, hard_top1_fusion, raw_soft_fusion
from time_router.evaluation.prediction_rows import build_per_sample_fusion_rows
from time_router.evaluation.summary import build_fusion_summary
from time_router.protocols import EvaluationInput, ExpertBatch, RouterOutput


@dataclass(frozen=True)
class EvaluationInputAdapterResult:
    """
    类功能：
        承载 EvaluationInputAdapter 的纯内存评估输出。

    字段说明：
        evaluation_input: 由 ExpertBatch 和 weights 适配得到的 EvaluationInput。
        hard_result/raw_soft_result: evaluation public API 复算得到的融合结果。
        summary: build_fusion_summary 返回的内存 summary dict。
        per_sample_rows: build_per_sample_fusion_rows 返回的逐样本内存 rows。
        extra: 只从输入对象转递的轻量 lineage 与 adapter metadata。
    """

    evaluation_input: EvaluationInput
    hard_result: FusionMetricsResult
    raw_soft_result: FusionMetricsResult
    summary: dict[str, Any]
    per_sample_rows: list[dict[str, Any]]
    extra: dict[str, Any] = field(default_factory=dict)


class EvaluationInputAdapter:
    """
    类功能：
        将 Stage 1 protocol objects 适配到 evaluation public API。

    关键约束：
        该 adapter 只服务当前固定五专家 canonical experiment 的 smoke 验收；
        固定五专家顺序不是 Time framework 长期全局 Expert System 契约。
    """

    adapter_name = "EvaluationInputAdapter"

    def build_evaluation_input(
        self,
        *,
        expert_batch: ExpertBatch,
        router_output: RouterOutput | None = None,
        fusion_weights: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EvaluationInput:
        """
        函数功能：
            从 ExpertBatch + RouterOutput.weights 或显式 fusion weights 构造
            最小 EvaluationInput。

        输入：
            expert_batch: 已由 provider 构造好的专家预测批次。
            router_output: 可选 RouterOutput；使用其中 weights/logits 和顺序信息。
            fusion_weights: 可选显式权重矩阵；不能与 router_output 同时传入。
            extra: 调用方附加的轻量 metadata。

        输出：
            EvaluationInput；y_pred/y_true 原样引用 ExpertBatch，不复制、不重读。
        """
        if (router_output is None) == (fusion_weights is None):
            raise ValueError("必须且只能传入 router_output 或 fusion_weights 其中之一")

        logits = None
        weights = fusion_weights
        router_extra: dict[str, Any] = {}
        if router_output is not None:
            _assert_same_tuple("sample_keys", tuple(expert_batch.sample_keys), tuple(router_output.sample_keys))
            _assert_same_tuple("model_columns", tuple(expert_batch.model_columns), tuple(router_output.model_columns))
            logits = router_output.logits
            weights = router_output.weights
            router_extra = dict(router_output.extra)

        if weights is None:
            raise ValueError("EvaluationInputAdapter 需要 fusion weights 才能复算 hard/raw-soft fusion")

        merged_extra = self._build_extra(
            expert_batch=expert_batch,
            router_extra=router_extra,
            explicit_weights=router_output is None,
            user_extra=extra,
        )
        return EvaluationInput(
            sample_keys=tuple(expert_batch.sample_keys),
            model_columns=tuple(expert_batch.model_columns),
            y_pred=expert_batch.y_pred,
            y_true=expert_batch.y_true,
            logits=logits,
            weights=weights,
            extra=merged_extra,
        )

    def evaluate(
        self,
        *,
        expert_batch: ExpertBatch,
        router_output: RouterOutput | None = None,
        fusion_weights: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EvaluationInputAdapterResult:
        """
        函数功能：
            构造 EvaluationInput，并复算 hard top-1、raw soft fusion、summary
            和 per-sample rows。

        输出：
            EvaluationInputAdapterResult；所有产物均为内存对象。
        """
        evaluation_input = self.build_evaluation_input(
            expert_batch=expert_batch,
            router_output=router_output,
            fusion_weights=fusion_weights,
            extra=extra,
        )
        return self.evaluate_input(evaluation_input=evaluation_input)

    def evaluate_input(self, *, evaluation_input: EvaluationInput) -> EvaluationInputAdapterResult:
        """
        函数功能：
            对调用方已经构造好的 EvaluationInput 复算内存评估产物。

        关键约束：
            该方法是 adapter 内唯一调用 evaluation public API 的实现点；
            legacy/compat wrapper 应复用这里，避免再次复制 fusion 逻辑。
        """
        if evaluation_input.weights is None:
            raise ValueError("EvaluationInputAdapter 需要 EvaluationInput.weights 才能复算 fusion")

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

        return EvaluationInputAdapterResult(
            evaluation_input=evaluation_input,
            hard_result=hard_result,
            raw_soft_result=raw_soft_result,
            summary=summary,
            per_sample_rows=per_sample_rows,
            extra=dict(evaluation_input.extra),
        )

    def _build_extra(
        self,
        *,
        expert_batch: ExpertBatch,
        router_extra: dict[str, Any],
        explicit_weights: bool,
        user_extra: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        函数功能：
            只从 ExpertBatch / RouterOutput 已携带信息中整理轻量 lineage。
        """
        merged_extra: dict[str, Any] = {
            "adapter_name": self.adapter_name,
            "stage1_contract": "canonical_five_expert_smoke_only",
            "weights_source": "explicit_fusion_weights" if explicit_weights else "router_output_weights",
            "expert_extra": dict(expert_batch.extra),
        }
        if router_extra:
            merged_extra["router_extra"] = router_extra
        if expert_batch.row_index_metadata is not None:
            merged_extra["row_index_metadata"] = expert_batch.row_index_metadata
        if user_extra:
            merged_extra.update(user_extra)
        return merged_extra


def _assert_same_tuple(name: str, left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    """函数功能：检查两侧顺序完全一致，避免 adapter 静默重排样本或专家列。"""
    if left != right:
        raise ValueError(f"{name} 不一致：expert_batch={left} router_output={right}")
