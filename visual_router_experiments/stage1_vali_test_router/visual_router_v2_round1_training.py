#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router V2 Round 1 P2b visual-only pooling 消融提供共享训练、
    feature 读取和评估 helper。

设计边界：
    - 只读取 P2a 已生成的 sharded `.npz` feature cache，不重新生成 ViT feature；
    - 只训练 visual-only router，不拼接 RevIN aux、不修改 P2a builder/schema；
    - scaler 只在 `pilot_train` fit，`pilot_selection` 和 `diagnostic_balanced`
      只 transform 与评估；
    - prediction cache 只通过目标 sample_key 的 SQLite 子集索引按 batch 读取。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn

from time_router.io.prediction_array_io import load_prediction_arrays_grouped
from visual_router_experiments.stage1_vali_test_router.fusion_utils import EPS, MODEL_COLUMNS
from visual_router_experiments.stage1_vali_test_router.train_visual_router import VisualMLPRouter
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import SQLitePredictionIndex
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import (
    TSF_STRATA_COLUMNS,
    make_method_rows,
)


POOLING_VARIANTS = ("visual_cls_only", "visual_mean_patch_only", "visual_cls_mean_concat")
FEATURE_ARRAY_BY_VARIANT = {
    "visual_cls_only": ("cls_embedding",),
    "visual_mean_patch_only": ("mean_patch_embedding",),
    "visual_cls_mean_concat": ("cls_embedding", "mean_patch_embedding"),
}


@dataclass(frozen=True)
class PoolingFeatureSet:
    """类功能：保存一个 sample_set 按 P0 order_index 对齐后的特征与样本表。"""

    sample_set: str
    sample_df: pd.DataFrame
    features: np.ndarray


def set_seed(seed: int) -> None:
    """函数功能：固定 Python、NumPy 和 PyTorch 随机源，保证 seed 级复现。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析训练设备；auto 优先使用 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 PyTorch CUDA 不可用")
    return torch.device(device_arg)


def read_ordered_sample_csv(sample_dir: Path, sample_set: str, *, max_samples: int | None = None) -> pd.DataFrame:
    """
    函数功能：
        读取 P0 sample CSV，并严格按 `order_index` 排序。

    输入：
        sample_dir: P0 sample set 输出目录。
        sample_set: `pilot_train`、`pilot_selection` 或 `diagnostic_balanced`。
        max_samples: smoke 截断行数；正式运行必须为 None。
    """
    path = Path(sample_dir) / f"{sample_set}_sample_keys.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 P0 sample CSV：{path}")
    df = pd.read_csv(path)
    required = {"sample_set", "order_index", "sample_key", "split", *TSF_STRATA_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} 缺少字段：{missing}")
    df = df.sort_values("order_index", kind="mergesort").reset_index(drop=True)
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()
    if df.empty:
        raise ValueError(f"{path} 读取后为空")
    if df["sample_set"].astype(str).ne(str(sample_set)).any():
        raise ValueError(f"{path} 中 sample_set 与期望不一致：{sample_set}")
    expected = np.arange(len(df), dtype=np.int64)
    actual = df["order_index"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(actual, expected):
        raise ValueError(f"{path} 的 order_index 必须从 0 连续递增")
    if df["sample_key"].astype(str).duplicated().any():
        dup = df.loc[df["sample_key"].astype(str).duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"{path} 中 sample_key 重复，示例：{dup}")
    return df


def load_pooling_features(
    *,
    feature_manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
    variant: str,
) -> np.ndarray:
    """
    函数功能：
        从 P2a feature manifest 指向的 `.npz` shards 中按 P0 order_index 读取
        一个 visual-only pooling 变体。

    关键约束：
        concat 只在内存现场构造，不写回 P2a cache；调用方按变体串行运行，
        避免同时保留三份大 embedding。
    """
    if variant not in FEATURE_ARRAY_BY_VARIANT:
        raise ValueError(f"未知 pooling variant={variant}")
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].copy()
    if rows.empty:
        raise ValueError(f"P2a feature manifest 中没有 sample_set={sample_set}")
    rows = rows.sort_values("start_order_index", kind="mergesort").reset_index(drop=True)
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    feature_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    loaded_count = 0
    for row in rows.itertuples(index=False):
        if loaded_count >= wanted_count:
            break
        shard_path = Path(str(row.shard_path))
        if not shard_path.exists():
            raise FileNotFoundError(f"找不到 P2a feature shard：{shard_path}")
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            arrays = [np.asarray(data[name], dtype=np.float32) for name in FEATURE_ARRAY_BY_VARIANT[variant]]
        if len(arrays) == 1:
            shard_features = arrays[0]
        else:
            shard_features = np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
        take = min(int(shard_features.shape[0]), wanted_count - loaded_count)
        feature_parts.append(np.asarray(shard_features[:take], dtype=np.float32))
        key_parts.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded_count += take
    if loaded_count != wanted_count:
        raise ValueError(f"feature shard 样本数不足：sample_set={sample_set} expected={wanted_count} actual={loaded_count}")
    order_index = np.concatenate(order_parts, axis=0)
    expected_order = sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(order_index, expected_order):
        raise ValueError(f"{sample_set}/{variant} feature order_index 与 P0 不一致")
    if key_parts != expected_keys:
        raise ValueError(f"{sample_set}/{variant} feature sample_key 与 P0 顺序不一致")
    features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
    if features.ndim != 2 or features.shape[0] != wanted_count:
        raise ValueError(f"{sample_set}/{variant} feature shape 异常：{features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{sample_set}/{variant} feature 中存在 NaN/Inf")
    return features


def align_labels_to_samples(sample_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：把 oracle labels 严格对齐到 P0 sample order，并补齐 TSF 分层字段。"""
    label_map = label_df.set_index("sample_key", drop=False)
    missing = [key for key in sample_df["sample_key"].astype(str).tolist() if key not in label_map.index]
    if missing:
        raise ValueError(f"oracle labels 缺失 sample_key，示例：{missing[:10]}")
    aligned = label_map.loc[sample_df["sample_key"].astype(str).tolist()].reset_index(drop=True).copy()
    for col in TSF_STRATA_COLUMNS:
        aligned[col] = sample_df[col].values
    aligned["order_index"] = sample_df["order_index"].values
    return aligned


def _fetch_ordered_prediction_records(
    prediction_index: SQLitePredictionIndex,
    sample_keys: Sequence[str],
) -> Dict[Tuple[str, str], Dict[str, object]]:
    """函数功能：按 batch 查询 SQLite，并检查五专家记录完整。"""
    lookup = prediction_index.fetch_records([str(key) for key in sample_keys])
    expected = len(sample_keys) * len(MODEL_COLUMNS)
    if len(lookup) != expected:
        raise ValueError(f"prediction index batch 查询不完整：expected={expected} actual={len(lookup)}")
    return lookup


def load_prediction_batch_from_index(
    prediction_index: SQLitePredictionIndex,
    sample_keys: Sequence[str],
    *,
    error_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        从 SQLite prediction index 按 batch 读取五专家 `y_pred`、共享 `y_true`
        和 manifest 误差矩阵。

    说明：
        `load_prediction_arrays_grouped` 会按 packed `.npy` 路径分组 mmap，
        避免逐样本重复打开同一 shard 文件。
    """
    keys = [str(key) for key in sample_keys]
    lookup = _fetch_ordered_prediction_records(prediction_index, keys)
    per_model_preds: List[np.ndarray] = []
    expert_errors = np.zeros((len(keys), len(MODEL_COLUMNS)), dtype=np.float32)
    y_true: np.ndarray | None = None
    for model_idx, model_name in enumerate(MODEL_COLUMNS):
        records = [lookup[(key, model_name)] for key in keys]
        preds = load_prediction_arrays_grouped(records, "y_pred").astype(np.float32, copy=False)
        current_true = load_prediction_arrays_grouped(records, "y_true").astype(np.float32, copy=False)
        if y_true is None:
            y_true = current_true
        elif not np.array_equal(y_true, current_true):
            raise ValueError(f"batch 内 y_true 不一致：model={model_name}")
        metric_col = "mae" if error_metric == "mae" else "mse"
        expert_errors[:, model_idx] = np.asarray([float(record[metric_col]) for record in records], dtype=np.float32)
        per_model_preds.append(preds)
    assert y_true is not None
    y_pred = np.stack(per_model_preds, axis=1).astype(np.float32, copy=False)
    return y_pred, y_true, expert_errors


def train_visual_pooling_router(
    *,
    train_features_scaled: np.ndarray,
    train_sample_keys: Sequence[str],
    prediction_index: SQLitePredictionIndex,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    huber_beta: float,
    kl_tau: float,
    lambda_kl: float,
    metric: str,
) -> Tuple[VisualMLPRouter, Dict[str, object]]:
    """
    函数功能：
        使用 P2b 固定的 `fusion_huber_kl` 目标训练一个 visual-only MLP router。

    输入：
        train_features_scaled: 已由 pilot_train scaler 标准化的视觉特征。
        train_sample_keys: 与特征行顺序一致的 sample_key。

    输出：
        已训练 router 和训练损失/权重诊断 metadata。
    """
    set_seed(seed)
    router = VisualMLPRouter(
        input_dim=int(train_features_scaled.shape[1]),
        hidden_dim=int(hidden_dim),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    huber = nn.SmoothL1Loss(beta=float(huber_beta))
    keys = [str(key) for key in train_sample_keys]
    x_all = torch.from_numpy(np.asarray(train_features_scaled, dtype=np.float32))
    rng = np.random.default_rng(int(seed))
    loss_history: List[float] = []
    huber_history: List[float] = []
    kl_history: List[float] = []
    router.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(keys))
        batch_losses: List[float] = []
        batch_huber: List[float] = []
        batch_kl: List[float] = []
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_keys = [keys[int(i)] for i in idx]
            y_pred, y_true, expert_errors = load_prediction_batch_from_index(
                prediction_index,
                batch_keys,
                error_metric=metric,
            )
            batch_x = x_all[idx].to(device=device)
            batch_pred = torch.from_numpy(y_pred).to(device=device)
            batch_true = torch.from_numpy(y_true).to(device=device)
            batch_q = torch.softmax(-torch.from_numpy(expert_errors) / float(kl_tau), dim=1).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(batch_x)
            weights = torch.softmax(logits, dim=1)
            weight_shape = (weights.shape[0], weights.shape[1], *([1] * (batch_pred.ndim - 2)))
            fused = (weights.view(weight_shape) * batch_pred).sum(dim=1)
            huber_loss = huber(fused, batch_true)
            kl_loss = F.kl_div(torch.log_softmax(logits, dim=1), batch_q, reduction="batchmean")
            loss = huber_loss + float(lambda_kl) * kl_loss
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
            batch_huber.append(float(huber_loss.detach().cpu().item()))
            batch_kl.append(float(kl_loss.detach().cpu().item()))
        loss_history.append(float(np.mean(batch_losses)))
        huber_history.append(float(np.mean(batch_huber)))
        kl_history.append(float(np.mean(batch_kl)))
    with torch.inference_mode():
        logits = router(x_all.to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
    metadata = {
        "initial_train_loss": float(loss_history[0]),
        "final_train_loss": float(loss_history[-1]),
        "final_huber_loss": float(huber_history[-1]),
        "final_kl_loss": float(kl_history[-1]),
        "train_weight_entropy": float(np.mean(entropy)),
        "train_normalized_weight_entropy": float(np.mean(entropy) / math.log(len(MODEL_COLUMNS))),
        "train_mean_max_weight": float(np.mean(weights.max(axis=1))),
    }
    return router, metadata


def predict_visual_pooling_router(
    *,
    router: VisualMLPRouter,
    scaler: StandardScaler,
    features: np.ndarray,
    sample_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    variant: str,
    seed: int,
    sample_set: str,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：对一个 sample_set 输出逐样本 hard top-1 权重和 oracle 对齐字段。"""
    aligned_labels = align_labels_to_samples(sample_df, labels_df)
    x_scaled = scaler.transform(np.asarray(features, dtype=np.float32)).astype(np.float32)
    router.eval()
    with torch.inference_mode():
        logits = router(torch.from_numpy(x_scaled).to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    selected_idx = weights.argmax(axis=1)
    entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
    rows: List[Dict[str, object]] = []
    router_name = f"p2b_{variant}_seed{int(seed)}"
    for row_idx, row in enumerate(aligned_labels.itertuples(index=False)):
        selected_model = MODEL_COLUMNS[int(selected_idx[row_idx])]
        output_row: Dict[str, object] = {
            "sample_set": sample_set,
            "variant": variant,
            "seed": int(seed),
            "router_name": router_name,
            "config_name": str(row.config_name),
            "sample_key": str(row.sample_key),
            "split": str(row.split),
            "dataset_name": str(row.dataset_name),
            "item_id": int(row.item_id),
            "channel_id": int(row.channel_id),
            "window_index": int(row.window_index),
            "selected_model": selected_model,
            "selected_value": float(getattr(row, selected_model)),
            "oracle_model": str(row.oracle_model),
            "oracle_value": float(row.oracle_value),
            "regret_to_oracle": float(getattr(row, selected_model) - row.oracle_value),
            "oracle_label_correct": bool(selected_model == row.oracle_model),
            "weight_entropy": float(entropy[row_idx]),
            "normalized_weight_entropy": float(entropy[row_idx] / math.log(len(MODEL_COLUMNS))),
            "max_weight": float(weights[row_idx].max()),
        }
        for model_idx, model_name in enumerate(MODEL_COLUMNS):
            output_row[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
        for col in TSF_STRATA_COLUMNS:
            output_row[col] = getattr(row, col)
        rows.append(output_row)
    return pd.DataFrame(rows)


def add_batch_fusion_metrics(
    pred_df: pd.DataFrame,
    *,
    prediction_index: SQLitePredictionIndex,
    metric: str,
    batch_size: int,
) -> pd.DataFrame:
    """
    函数功能：
        为逐样本 router 输出追加 hard top-1 数组 MAE/MSE 和 raw soft fusion MAE/MSE。
    """
    out = pred_df.copy().reset_index(drop=True)
    hard_mae = np.zeros(len(out), dtype=np.float64)
    hard_mse = np.zeros(len(out), dtype=np.float64)
    soft_mae = np.zeros(len(out), dtype=np.float64)
    soft_mse = np.zeros(len(out), dtype=np.float64)
    sample_keys = out["sample_key"].astype(str).tolist()
    weight_matrix = out[[f"weight_{name}" for name in MODEL_COLUMNS]].to_numpy(dtype=np.float32)
    selected_indices = out["selected_model"].astype(str).map({name: idx for idx, name in enumerate(MODEL_COLUMNS)}).to_numpy()
    for start in range(0, len(out), int(batch_size)):
        stop = min(start + int(batch_size), len(out))
        keys = sample_keys[start:stop]
        y_pred, y_true, _ = load_prediction_batch_from_index(prediction_index, keys, error_metric=metric)
        weights = weight_matrix[start:stop]
        soft_pred = (weights.reshape((len(keys), len(MODEL_COLUMNS), *([1] * (y_pred.ndim - 2)))) * y_pred).sum(axis=1)
        hard_pred = y_pred[np.arange(len(keys)), selected_indices[start:stop].astype(np.int64)]
        soft_diff = soft_pred - y_true
        hard_diff = hard_pred - y_true
        axes = tuple(range(1, soft_diff.ndim))
        soft_mae[start:stop] = np.mean(np.abs(soft_diff), axis=axes)
        soft_mse[start:stop] = np.mean(soft_diff ** 2, axis=axes)
        hard_mae[start:stop] = np.mean(np.abs(hard_diff), axis=axes)
        hard_mse[start:stop] = np.mean(hard_diff ** 2, axis=axes)
    out["hard_top1_mae_from_array"] = hard_mae
    out["hard_top1_mse_from_array"] = hard_mse
    out["soft_fusion_mae"] = soft_mae
    out["soft_fusion_mse"] = soft_mse
    return out


def make_visual_pooling_method_rows(pred_df: pd.DataFrame, *, sample_set: str, variant: str, seed: int) -> pd.DataFrame:
    """函数功能：将 P2b 逐样本预测规整成 hard/raw-soft 两类统一评估行。"""
    hard = make_method_rows(
        sample_set=sample_set,
        method=f"{variant}_hard_top1",
        pred_df=pred_df,
        mae_col="hard_top1_mae_from_array",
        mse_col="hard_top1_mse_from_array",
    )
    soft = make_method_rows(
        sample_set=sample_set,
        method=f"{variant}_raw_soft_fusion",
        pred_df=pred_df,
        mae_col="soft_fusion_mae",
        mse_col="soft_fusion_mse",
    )
    rows = pd.concat([hard, soft], ignore_index=True)
    rows.insert(1, "variant", variant)
    rows.insert(2, "seed", int(seed))
    return rows


def summarize_rows_with_seed(rows: pd.DataFrame, *, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """函数功能：按 variant/seed/method 汇总 P2b 必需指标，可额外加入分层字段。"""
    by_cols = ["sample_set", "variant", "seed", "method", *group_cols]
    out_rows: List[Dict[str, object]] = []
    for keys, group in rows.groupby(by_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(by_cols, keys)}
        row.update(
            {
                "sample_count": int(len(group)),
                "MAE": float(group["mae"].mean()),
                "MSE": float(group["mse"].mean()),
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
                "weight_entropy": float(group["weight_entropy"].mean()),
                "normalized_weight_entropy": float(group["normalized_weight_entropy"].mean()),
                "mean_max_weight": float(group["mean_max_weight"].mean()),
            }
        )
        out_rows.append(row)
    return pd.DataFrame(out_rows).reset_index(drop=True)


def summarize_mean_std(seed_summary: pd.DataFrame, *, sample_set: str) -> pd.DataFrame:
    """函数功能：对三 seeds 的 selection/diagnostic 指标做 mean/std 汇总。"""
    subset = seed_summary[seed_summary["sample_set"].astype(str) == str(sample_set)].copy()
    metric_cols = [
        "MAE",
        "MSE",
        "regret_to_oracle",
        "oracle_label_accuracy",
        "weight_entropy",
        "normalized_weight_entropy",
        "mean_max_weight",
    ]
    rows: List[Dict[str, object]] = []
    for (variant, method), group in subset.groupby(["variant", "method"], sort=False):
        row: Dict[str, object] = {
            "sample_set": sample_set,
            "variant": variant,
            "method": method,
            "seed_count": int(group["seed"].nunique()),
            "sample_count_per_seed": int(group["sample_count"].iloc[0]),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["MAE_mean", "variant", "method"]).reset_index(drop=True)


def selected_model_counts_with_variant(rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：输出每个变体、seed、method 的 selected_model count 和 ratio。"""
    count_df = (
        rows.groupby(["sample_set", "variant", "seed", "method", "selected_model"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = count_df.groupby(["sample_set", "variant", "seed", "method"])["count"].transform("sum")
    count_df["ratio"] = count_df["count"] / totals
    return count_df.sort_values(["sample_set", "variant", "seed", "method", "selected_model"]).reset_index(drop=True)
