#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P9d Visual Router evaluation adapter ExpertBatch 旁路校验 smoke。

输入：
    只使用本文件内构造的小型 numpy arrays / DataFrame，不启动 ViT，不访问
    `/data2`，不运行正式 Visual Router entrypoint。

输出：
    标准输出打印中文检查日志；若新增旁路 helper 与正式 soft_df 字段不一致，
    或失败信息缺少关键定位上下文，则抛出 AssertionError。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.evaluation import EvaluationInputAdapter  # noqa: E402
from time_router.protocols import ExpertBatch  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, compute_array_metrics  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    build_visual_router_expert_batch_for_evaluation,
    verify_evaluation_adapter_bypass_batch,
)


def _build_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        构造两个样本、五个专家、三步单通道的最小预测张量。

    关键约束：
        权重刻意让两个样本选择不同专家，覆盖 selected_model/selected_index
        与 hard/raw soft MAE/MSE 的逐样本比较。
    """
    y_true = np.asarray(
        [
            [[1.0], [2.0], [3.0]],
            [[2.0], [0.0], [-1.0]],
        ],
        dtype=np.float32,
    )
    y_pred = np.asarray(
        [
            [
                [[1.1], [2.2], [2.9]],
                [[0.8], [2.1], [3.3]],
                [[1.0], [2.0], [3.0]],
                [[1.5], [1.5], [2.5]],
                [[0.5], [2.5], [3.5]],
            ],
            [
                [[1.5], [0.2], [-1.2]],
                [[2.1], [-0.1], [-0.9]],
                [[2.4], [0.4], [-0.5]],
                [[1.9], [0.1], [-1.1]],
                [[2.0], [0.0], [-1.0]],
            ],
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            [0.10, 0.20, 0.50, 0.15, 0.05],
            [0.05, 0.15, 0.10, 0.20, 0.50],
        ],
        dtype=np.float32,
    )
    return y_pred, y_true, weights


def _build_pred_df(weights: np.ndarray) -> pd.DataFrame:
    """函数功能：构造与 Visual Router test batch 等价的 hard prediction DataFrame。"""
    rows = []
    sample_keys = ["p9d_sample_0", "p9d_sample_1"]
    for row_idx, sample_key in enumerate(sample_keys):
        selected_index = int(weights[row_idx].argmax())
        row = {
            "router_name": "visual_router_p9d_smoke",
            "config_name": "96_48_S",
            "sample_key": sample_key,
            "split": "test",
            "dataset_name": "SMOKE_DATA",
            "item_id": row_idx,
            "channel_id": 0,
            "window_index": row_idx * 10,
            "selected_model": MODEL_COLUMNS[selected_index],
            "selected_value": float(row_idx),
            "oracle_model": MODEL_COLUMNS[selected_index],
            "oracle_value": 0.0,
            "regret_to_oracle": float(row_idx),
            "oracle_label_correct": True,
            "weight_entropy": float(-(weights[row_idx] * np.log(np.clip(weights[row_idx], 1e-8, 1.0))).sum()),
            "normalized_weight_entropy": float(
                -(weights[row_idx] * np.log(np.clip(weights[row_idx], 1e-8, 1.0))).sum() / np.log(len(MODEL_COLUMNS))
            ),
            "max_weight": float(weights[row_idx].max()),
        }
        for model_idx, model_name in enumerate(MODEL_COLUMNS):
            row[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_soft_df(pred_df: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """
    函数功能：
        用与正式 add_soft_fusion_metrics 等价的数组公式生成 soft_df 字段。
    """
    rows = []
    for row_idx, row in enumerate(pred_df.to_dict(orient="records")):
        selected_index = MODEL_COLUMNS.index(str(row["selected_model"]))
        soft_pred = np.sum(y_pred[row_idx] * weights[row_idx].reshape((-1, 1, 1)), axis=0)
        hard_metrics = compute_array_metrics(y_true[row_idx], y_pred[row_idx, selected_index])
        soft_metrics = compute_array_metrics(y_true[row_idx], soft_pred)
        output_row = dict(row)
        output_row.update(
            {
                "soft_fusion_mae": soft_metrics["mae"],
                "soft_fusion_mse": soft_metrics["mse"],
                "hard_top1_mae_from_array": hard_metrics["mae"],
                "hard_top1_mse_from_array": hard_metrics["mse"],
            }
        )
        rows.append(output_row)
    return pd.DataFrame(rows)


def run_smoke() -> None:
    """函数功能：执行 P9d Visual Router evaluation adapter ExpertBatch 旁路 smoke。"""
    output_dir = REPO_ROOT / "experiment_logs" / "run_outputs" / "p9d_visual_router_expert_batch_bridge_smoke_not_created"
    if str(output_dir).startswith("/data2/"):
        raise AssertionError("P9d smoke 不应访问 /data2")

    y_pred, y_true, weights = _build_arrays()
    pred_df = _build_pred_df(weights)
    soft_df = _build_soft_df(pred_df, y_pred, y_true, weights)
    sample_keys = pred_df["sample_key"].astype(str).tolist()

    direct_expert_batch = build_visual_router_expert_batch_for_evaluation(
        sample_keys=sample_keys,
        y_pred=y_pred,
        y_true=y_true,
        model_columns=MODEL_COLUMNS,
        batch_index=6,
        output_dir=output_dir,
    )
    if direct_expert_batch.sample_keys != tuple(sample_keys):
        raise AssertionError(f"ExpertBatch.sample_keys 未保序：{direct_expert_batch.sample_keys}")
    if direct_expert_batch.model_columns != tuple(MODEL_COLUMNS):
        raise AssertionError(f"ExpertBatch.model_columns 未保序：{direct_expert_batch.model_columns}")
    if direct_expert_batch.y_pred is not y_pred or direct_expert_batch.y_true is not y_true:
        raise AssertionError("ExpertBatch builder 对 float32 y_pred/y_true 不应复制或重读")

    captured_calls: list[dict[str, object]] = []
    original_evaluate = EvaluationInputAdapter.evaluate

    def capture_evaluate(self: EvaluationInputAdapter, **kwargs: object) -> object:
        """函数功能：捕获 helper 传入 adapter 的 canonical ExpertBatch 边界。"""
        captured_calls.append(dict(kwargs))
        return original_evaluate(self, **kwargs)

    with patch.object(EvaluationInputAdapter, "evaluate", new=capture_evaluate):
        verify_evaluation_adapter_bypass_batch(
            pred_df=pred_df,
            soft_df=soft_df,
            y_pred=y_pred,
            y_true=y_true,
            output_dir=output_dir,
            batch_index=7,
            atol=1e-6,
        )

    if len(captured_calls) != 1:
        raise AssertionError(f"adapter evaluate 调用次数异常：actual={len(captured_calls)}")
    captured = captured_calls[0]
    if "evaluation_input" in captured:
        raise AssertionError("P9d helper 不应再直接传入 EvaluationInput")
    expert_batch = captured.get("expert_batch")
    if not isinstance(expert_batch, ExpertBatch):
        raise AssertionError(f"P9d helper 未通过 ExpertBatch 调用 adapter：actual={type(expert_batch)!r}")
    if expert_batch.sample_keys != tuple(sample_keys):
        raise AssertionError(f"adapter ExpertBatch.sample_keys 顺序漂移：{expert_batch.sample_keys}")
    if expert_batch.model_columns != tuple(MODEL_COLUMNS):
        raise AssertionError(f"adapter ExpertBatch.model_columns 顺序漂移：{expert_batch.model_columns}")
    if expert_batch.y_pred is not y_pred or expert_batch.y_true is not y_true:
        raise AssertionError("y_pred/y_true 必须原样进入 adapter，不应复制、重排或重读")
    np.testing.assert_allclose(captured.get("fusion_weights"), weights, rtol=0.0, atol=1e-6)
    print("通过：helper 经 ExpertBatch + fusion_weights 调用 adapter，样本/专家顺序和数组输入保持不变")
    print("通过：adapter rows 与正式 soft_df 的 selected/hard/raw-soft/权重诊断字段一致")

    bad_soft_df = soft_df.copy()
    bad_soft_df.loc[1, "soft_fusion_mae"] = float(bad_soft_df.loc[1, "soft_fusion_mae"]) + 0.25
    try:
        verify_evaluation_adapter_bypass_batch(
            pred_df=pred_df,
            soft_df=bad_soft_df,
            y_pred=y_pred,
            y_true=y_true,
            output_dir=output_dir,
            batch_index=8,
            atol=1e-6,
        )
    except AssertionError as exc:
        message = str(exc)
        required_fragments = [
            "config_name=96_48_S",
            "split=test",
            "batch_index=8",
            "row_offset=1",
            "sample_key=p9d_sample_1",
            "field=soft_fusion_mae",
            "old_value=",
            "adapter_value=",
            "output_dir=",
        ]
        missing = [fragment for fragment in required_fragments if fragment not in message]
        if missing:
            raise AssertionError(f"mismatch 错误信息缺少关键上下文：missing={missing} message={message}") from exc
        print("通过：故意 mismatch 时错误信息包含 config/split/batch/sample/字段/旧值/adapter 值/output_dir")
    else:
        raise AssertionError("故意篡改 soft_fusion_mae 后应触发旁路校验失败")


if __name__ == "__main__":
    run_smoke()
