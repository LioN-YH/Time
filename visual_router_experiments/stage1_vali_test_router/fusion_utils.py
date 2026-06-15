#!/usr/bin/env python3
"""
文件功能：
    汇总 Stage 1 router / fusor 共享的专家预测读取与 hard/soft fusion 评估工具。

设计约束：
    - `MODEL_COLUMNS` 固定五专家动作空间，所有 per-config router 都复用同一专家顺序；
    - prediction manifest 只作为监督或评估数组来源，不作为 router 输入特征；
    - hard top-1 与 raw soft fusion 指标统一从数组复算，避免只依赖 CSV 中的窗口误差。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from visual_router_experiments.common.prediction_array_io import load_prediction_array, resolve_cache_array_path


MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
EPS = 1e-8
FEATURE_METADATA_COLUMNS = {
    "feature_version",
    "sample_key",
    "config_name",
    "split",
    "dataset_name",
    "item_id",
    "channel_id",
    "window_index",
    "history_length",
    "feature_type",
    "feature_dim",
}


def display_time() -> str:
    """函数功能：生成中文日志和 metadata 中使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def frame_to_markdown(df: pd.DataFrame, *, float_digits: int = 6) -> str:
    """函数功能：将 DataFrame 转为 Markdown 表格，避免依赖 tabulate。"""
    if df.empty:
        return "_无记录_"
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
        else:
            display_df[col] = display_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(display_df.columns) + " |",
        "| " + " | ".join(["---"] * len(display_df.columns)) + " |",
    ]
    for row in display_df.values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def load_feature_cache(feature_cache_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """函数功能：读取 TimeFuse-derived feature cache，并识别数值特征列。"""
    if not feature_cache_path.exists():
        raise FileNotFoundError(f"找不到 feature cache：{feature_cache_path}")
    feature_df = pd.read_csv(feature_cache_path)
    missing_metadata = sorted(FEATURE_METADATA_COLUMNS.difference(feature_df.columns))
    if missing_metadata:
        raise ValueError(f"feature cache 缺少元信息字段：{missing_metadata}")
    feature_cols = [
        col
        for col in feature_df.columns
        if col not in FEATURE_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[col])
    ]
    if not feature_cols:
        raise ValueError("feature cache 中没有可用数值特征列")
    if feature_df["sample_key"].duplicated().any():
        dup_keys = feature_df.loc[feature_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"feature cache 中 sample_key 重复，示例：{dup_keys}")
    if feature_df["feature_dim"].nunique() != 1:
        raise ValueError("feature cache 中 feature_dim 不一致")
    return feature_df, feature_cols


def join_feature_and_labels(feature_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：用 sample_key 和稳定元信息严格 join feature cache 与 oracle labels。"""
    join_cols = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    merged = feature_df.merge(labels_df, on=join_cols, how="inner", suffixes=("", "_label"))
    if len(merged) != len(feature_df) or len(merged) != len(labels_df):
        missing_feature = sorted(set(labels_df["sample_key"]) - set(feature_df["sample_key"]))
        missing_label = sorted(set(feature_df["sample_key"]) - set(labels_df["sample_key"]))
        raise ValueError(
            f"feature/label join 不完整：missing_feature={missing_feature[:10]} missing_label={missing_label[:10]}"
        )
    return merged


class TimeFuseFusor(nn.Module):
    """
    类功能：
        复刻原生 TimeFuse 的单层 fusor。

    说明：
        forward 直接返回 softmax 后的专家融合权重；训练时再用这些权重对五专家
        `y_pred` 加权求和，并通过 SmoothL1Loss 反传。
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.fc.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.fc.bias, -bound, bound)

    def forward(self, meta_features: torch.Tensor) -> torch.Tensor:
        """函数功能：输出专家 softmax 权重。"""
        logits = self.fc(meta_features)
        return torch.softmax(logits, dim=-1)


def _train_loader_from_tensors(
    x: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    """函数功能：把 numpy 张量包装为可复现的训练 DataLoader。"""
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_pred, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_true, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True, generator=generator)


def _broadcast_weights(weights: torch.Tensor, prediction_tensor: torch.Tensor) -> torch.Tensor:
    """函数功能：把 `[B, M]` 权重广播到专家预测张量 `[B, M, ...]`。"""
    weight_shape = (weights.shape[0], weights.shape[1], *([1] * (prediction_tensor.ndim - 2)))
    return weights.view(weight_shape)


def _compute_weight_statistics(weights: torch.Tensor) -> Dict[str, float]:
    """函数功能：计算权重熵与最大权重占比。"""
    clipped = torch.clamp(weights, min=EPS)
    entropy = (-(weights * torch.log(clipped))).sum(dim=1)
    max_weight = weights.max(dim=1).values
    return {
        "mean_weight_entropy": float(entropy.mean().detach().cpu().item()),
        "mean_normalized_weight_entropy": float((entropy.mean() / math.log(len(MODEL_COLUMNS))).detach().cpu().item()),
        "mean_max_weight": float(max_weight.mean().detach().cpu().item()),
    }


def train_timefuse_fusor_for_config(
    *,
    config_name: str,
    config_df: pd.DataFrame,
    feature_cols: Sequence[str],
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
    metric: str,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    seed: int,
    device: torch.device,
) -> Tuple[TimeFuseFusor, StandardScaler, Dict[str, object]]:
    """
    函数功能：
        在单个 config_name 内训练原生 TimeFuse-style fusor。

    关键约束：
        - StandardScaler 只在 vali features 上 fit；
        - 训练目标为 fused_output 与 y_true 的 SmoothL1Loss；
        - 不跨 config 共享动作空间。
    """
    vali_df = config_df[config_df["split"] == "vali"].copy()
    test_df = config_df[config_df["split"] == "test"].copy()
    if vali_df.empty or test_df.empty:
        raise ValueError(f"config_name={config_name} 需要同时包含 vali/test 样本")

    x_vali = vali_df[list(feature_cols)].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x_vali_scaled = scaler.fit_transform(x_vali).astype(np.float32)

    y_pred_vali, y_true_vali, _ = load_prediction_tensors_for_samples(
        vali_df["sample_key"].astype(str).tolist(),
        prediction_lookup,
        error_metric=str(metric),
    )

    fusor = TimeFuseFusor(input_dim=int(x_vali_scaled.shape[1]), output_dim=len(MODEL_COLUMNS)).to(device)
    loader = _train_loader_from_tensors(
        x_vali_scaled,
        y_pred_vali,
        y_true_vali,
        batch_size=batch_size,
        seed=seed,
    )
    optimizer = torch.optim.Adam(fusor.parameters(), lr=float(lr))
    criterion = torch.nn.SmoothL1Loss(beta=float(beta))

    fusor.train()
    loss_history: List[float] = []
    for _ in range(int(epochs)):
        epoch_losses: List[float] = []
        for batch_x, batch_pred, batch_true in loader:
            batch_x = batch_x.to(device=device)
            batch_pred = batch_pred.to(device=device)
            batch_true = batch_true.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            weights = fusor(batch_x)
            fused_output = (_broadcast_weights(weights, batch_pred) * batch_pred).sum(dim=1)
            loss = criterion(fused_output, batch_true)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        loss_history.append(float(np.mean(epoch_losses)))

    with torch.inference_mode():
        train_weights = fusor(torch.from_numpy(x_vali_scaled).to(device=device))
        train_pred = train_weights.argmax(dim=1).detach().cpu().numpy()
        y_vali = np.asarray([MODEL_COLUMNS.index(label) for label in vali_df["oracle_model"]], dtype=np.int64)
        y_pred_tensor = torch.from_numpy(y_pred_vali).to(device=device)
        y_true_tensor = torch.from_numpy(y_true_vali).to(device=device)
        fused_vali = (_broadcast_weights(train_weights, y_pred_tensor) * y_pred_tensor).sum(dim=1)
        train_error = fused_vali - y_true_tensor
        train_metrics = {
            "train_fusion_mae": float(train_error.abs().mean().detach().cpu().item()),
            "train_fusion_mse": float((train_error ** 2).mean().detach().cpu().item()),
        }
        train_weight_stats = _compute_weight_statistics(train_weights)

    metadata = {
        "config_name": config_name,
        "vali_sample_count": int(len(vali_df)),
        "test_sample_count": int(len(test_df)),
        "feature_dim": int(x_vali_scaled.shape[1]),
        "feature_columns": list(feature_cols),
        "loss": "SmoothL1Loss",
        "beta": float(beta),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "training_split": "vali",
        "evaluation_split": "test",
        "final_train_loss": float(loss_history[-1]),
        "initial_train_loss": float(loss_history[0]),
        **train_metrics,
        **train_weight_stats,
        "train_hard_oracle_label_accuracy": float((train_pred == y_vali).mean()),
    }
    return fusor, scaler, metadata


def predict_timefuse_fusor_for_config(
    *,
    fusor: TimeFuseFusor,
    scaler: StandardScaler,
    config_df: pd.DataFrame,
    feature_cols: Sequence[str],
    router_name: str,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：对单个 config_name 的 test split 输出 hard top-1 权重和融合路由结果。"""
    test_df = config_df[config_df["split"] == "test"].copy().reset_index(drop=True)
    x_test = test_df[list(feature_cols)].to_numpy(dtype=np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    fusor.eval()
    with torch.inference_mode():
        weights = fusor(torch.from_numpy(x_test_scaled).to(device=device)).detach().cpu().numpy()
    selected_indices = weights.argmax(axis=1)
    selected_models = [MODEL_COLUMNS[int(idx)] for idx in selected_indices]
    weight_entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
    normalized_weight_entropy = weight_entropy / np.log(len(MODEL_COLUMNS))
    max_weight = weights.max(axis=1)

    rows: List[Dict[str, object]] = []
    for row_idx, (_, row) in enumerate(test_df.iterrows()):
        selected_model = selected_models[row_idx]
        output_row: Dict[str, object] = {
            "router_name": router_name,
            "config_name": row["config_name"],
            "sample_key": row["sample_key"],
            "split": row["split"],
            "dataset_name": row["dataset_name"],
            "item_id": int(row["item_id"]),
            "channel_id": int(row["channel_id"]),
            "window_index": int(row["window_index"]),
            "selected_model": selected_model,
            "selected_value": float(row[selected_model]),
            "oracle_model": row["oracle_model"],
            "oracle_value": float(row["oracle_value"]),
            "regret_to_oracle": float(row[selected_model] - row["oracle_value"]),
            "oracle_label_correct": bool(selected_model == row["oracle_model"]),
            "weight_entropy": float(weight_entropy[row_idx]),
            "normalized_weight_entropy": float(normalized_weight_entropy[row_idx]),
            "max_weight": float(max_weight[row_idx]),
        }
        for model_idx, model_name in enumerate(MODEL_COLUMNS):
            output_row[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
        rows.append(output_row)
    return pd.DataFrame(rows)


def run_timefuse_fusor_baseline(
    *,
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
    metric: str,
    feature_cols: Sequence[str],
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """函数功能：按 config_name 独立训练并评估 TimeFuse-style fusor baseline。"""
    merged_df = join_feature_and_labels(feature_df, labels_df)
    hard_frames: List[pd.DataFrame] = []
    config_metadata: List[Dict[str, object]] = []
    for config_name, config_df in merged_df.groupby("config_name", sort=True):
        fusor, scaler, metadata = train_timefuse_fusor_for_config(
            config_name=str(config_name),
            config_df=config_df,
            feature_cols=feature_cols,
            prediction_lookup=prediction_lookup,
            metric=metric,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            beta=beta,
            seed=seed,
            device=device,
        )
        config_metadata.append(metadata)
        hard_frames.append(
            predict_timefuse_fusor_for_config(
                fusor=fusor,
                scaler=scaler,
                config_df=config_df,
                feature_cols=feature_cols,
                router_name="timefuse_style_fusor",
                device=device,
            )
        )

    hard_pred_df = pd.concat(hard_frames, ignore_index=True)
    soft_pred_df = add_soft_fusion_metrics(hard_pred_df, prediction_lookup)
    hard_summary_df = summarize_hard_predictions(hard_pred_df)
    soft_summary_df = summarize_soft_fusion(soft_pred_df)
    selected_counts_df = summarize_selected_model_counts(hard_pred_df)
    return {
        "hard_pred_df": hard_pred_df,
        "soft_pred_df": soft_pred_df,
        "hard_summary_df": hard_summary_df,
        "soft_summary_df": soft_summary_df,
        "selected_counts_df": selected_counts_df,
        "config_metadata": config_metadata,
    }


def load_prediction_lookup(prediction_manifest_path: Path) -> Mapping[Tuple[str, str], Dict[str, object]]:
    """函数功能：读取 prediction cache manifest，建立 `(sample_key, model_name) -> 路径记录`。"""
    if not prediction_manifest_path.exists():
        raise FileNotFoundError(f"找不到 prediction manifest：{prediction_manifest_path}")
    manifest_df = pd.read_csv(prediction_manifest_path)
    required_cols = {"sample_key", "model_name", "y_true_path", "y_pred_path", "mae", "mse"}
    missing_cols = sorted(required_cols.difference(manifest_df.columns))
    if missing_cols:
        raise ValueError(f"prediction manifest 缺少字段：{missing_cols}")
    duplicated = manifest_df.duplicated(["sample_key", "model_name"])
    if duplicated.any():
        dup = manifest_df.loc[duplicated, ["sample_key", "model_name"]].head(10).to_dict("records")
        raise ValueError(f"prediction manifest 中 sample_key + model_name 重复，示例：{dup}")

    manifest_dir = prediction_manifest_path.parent
    lookup: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in manifest_df.itertuples(index=False):
        record = row._asdict()
        record["y_true_path"] = resolve_cache_array_path(str(record["y_true_path"]), manifest_dir)
        record["y_pred_path"] = resolve_cache_array_path(str(record["y_pred_path"]), manifest_dir)
        lookup[(str(row.sample_key), str(row.model_name))] = record
    return lookup


def load_prediction_tensors_for_samples(
    sample_keys: Sequence[str],
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
    *,
    error_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        按 sample_key 顺序读取五专家预测数组和共享 y_true，用于 fusion loss。

    输入：
        sample_keys: 与 feature/label DataFrame 行顺序一致的 sample_key。
        prediction_lookup: `(sample_key, model_name) -> y_true/y_pred 路径` 索引。

    输出：
        - y_preds: `[N, M, ...]`，M 为五个专家；
        - y_true: `[N, ...]`；
        - expert_errors: `[N, M]`，按 `error_metric` 复算出的专家窗口误差。

    关键约束：
        训练阶段可读取当前训练 split 的 y_true/y_pred 作为 loss 监督，但这些数组
        不会作为 router 输入特征，也不会读取 test oracle 误差来调权重。
    """
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []
    all_errors: List[np.ndarray] = []
    for sample_key in sample_keys:
        missing_models = [model for model in MODEL_COLUMNS if (str(sample_key), model) not in prediction_lookup]
        if missing_models:
            raise ValueError(f"prediction manifest 缺少 sample_key={sample_key} 的专家：{missing_models}")

        sample_preds: List[np.ndarray] = []
        sample_true: Optional[np.ndarray] = None
        sample_errors: List[float] = []
        for model_name in MODEL_COLUMNS:
            record = prediction_lookup[(str(sample_key), model_name)]
            y_pred = load_prediction_array(record, "y_pred")
            current_y_true = load_prediction_array(record, "y_true")
            if sample_true is None:
                sample_true = current_y_true
            elif not np.array_equal(sample_true, current_y_true):
                raise ValueError(f"同一 sample_key 的 y_true 内容不一致：{sample_key}")
            if y_pred.shape != current_y_true.shape:
                raise ValueError(f"y_pred/y_true shape 不一致：sample_key={sample_key} model={model_name}")
            sample_preds.append(y_pred)
            diff = y_pred - current_y_true
            if error_metric == "mae":
                sample_errors.append(float(np.mean(np.abs(diff))))
            elif error_metric == "mse":
                sample_errors.append(float(np.mean(diff ** 2)))
            else:
                raise ValueError(f"未知 error_metric={error_metric}")

        assert sample_true is not None
        all_preds.append(np.stack(sample_preds, axis=0))
        all_trues.append(sample_true)
        all_errors.append(np.asarray(sample_errors, dtype=np.float32))

    y_preds = np.stack(all_preds, axis=0).astype(np.float32)
    y_true = np.stack(all_trues, axis=0).astype(np.float32)
    expert_errors = np.stack(all_errors, axis=0).astype(np.float32)
    if y_preds.ndim < 3:
        raise ValueError(f"专家预测张量维度异常：{y_preds.shape}")
    if not (np.isfinite(y_preds).all() and np.isfinite(y_true).all() and np.isfinite(expert_errors).all()):
        raise ValueError("prediction tensor 中存在非有限值")
    return y_preds, y_true, expert_errors


def compute_array_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """函数功能：基于数组计算 MAE/MSE，用于 hard top-1 和 soft fusion 统一复核。"""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape 不一致：{y_true.shape} vs {y_pred.shape}")
    error = y_pred - y_true
    return {"mae": float(np.mean(np.abs(error))), "mse": float(np.mean(error ** 2))}


def add_soft_fusion_metrics(pred_df: pd.DataFrame, prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]]) -> pd.DataFrame:
    """
    函数功能：
        根据 router/fusor 权重加权五专家预测数组，计算 raw soft fusion MAE/MSE。

    关键约束：
        只融合同一 `sample_key` 下的五个专家预测，不读取 test oracle 误差来调整权重。
    """
    rows: List[Dict[str, object]] = []
    for _, row in pred_df.iterrows():
        sample_key = str(row["sample_key"])
        missing_models = [model for model in MODEL_COLUMNS if (sample_key, model) not in prediction_lookup]
        if missing_models:
            raise ValueError(f"prediction manifest 缺少 sample_key={sample_key} 的专家：{missing_models}")

        y_true: Optional[np.ndarray] = None
        weighted_pred: Optional[np.ndarray] = None
        expert_metric_cols: Dict[str, float] = {}
        for model_name in MODEL_COLUMNS:
            record = prediction_lookup[(sample_key, model_name)]
            y_pred = load_prediction_array(record, "y_pred")
            current_y_true = load_prediction_array(record, "y_true")
            if y_true is None:
                y_true = current_y_true
            elif not np.array_equal(y_true, current_y_true):
                raise ValueError(f"同一 sample_key 的 y_true 内容不一致：{sample_key}")
            weight = float(row[f"weight_{model_name}"])
            weighted_pred = y_pred * weight if weighted_pred is None else weighted_pred + y_pred * weight
            expert_metric_cols[f"{model_name}_mae_from_manifest"] = float(record["mae"])
            expert_metric_cols[f"{model_name}_mse_from_manifest"] = float(record["mse"])

        assert y_true is not None and weighted_pred is not None
        soft_metrics = compute_array_metrics(y_true, weighted_pred)
        selected_record = prediction_lookup[(sample_key, str(row["selected_model"]))]
        hard_metrics = compute_array_metrics(y_true, load_prediction_array(selected_record, "y_pred"))

        output_row = row.to_dict()
        output_row.update(expert_metric_cols)
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


def summarize_hard_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：汇总 hard top-1 routing/fusor 的 test 指标。"""
    rows: List[Dict[str, object]] = []
    for (router_name, config_name), group in pred_df.groupby(["router_name", "config_name"], sort=True):
        rows.append(
            {
                "router_name": router_name,
                "config_name": config_name,
                "sample_count": int(len(group)),
                "selected_value": float(group["selected_value"].mean()),
                "oracle_value": float(group["oracle_value"].mean()),
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
                "mean_weight_entropy": float(group["weight_entropy"].mean()),
                "mean_normalized_weight_entropy": float(group["normalized_weight_entropy"].mean()),
                "mean_max_weight": float(group["max_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_soft_fusion(pred_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：汇总 raw soft fusion 与 hard top-1 的数组级 MAE/MSE。"""
    rows: List[Dict[str, object]] = []
    for (router_name, config_name), group in pred_df.groupby(["router_name", "config_name"], sort=True):
        rows.append(
            {
                "router_name": f"{router_name}_soft_fusion",
                "config_name": config_name,
                "sample_count": int(len(group)),
                "soft_fusion_mae": float(group["soft_fusion_mae"].mean()),
                "soft_fusion_mse": float(group["soft_fusion_mse"].mean()),
                "hard_top1_mae_from_array": float(group["hard_top1_mae_from_array"].mean()),
                "hard_top1_mse_from_array": float(group["hard_top1_mse_from_array"].mean()),
                "oracle_value": float(group["oracle_value"].mean()),
                "mean_weight_entropy": float(group["weight_entropy"].mean()),
                "mean_normalized_weight_entropy": float(group["normalized_weight_entropy"].mean()),
                "mean_max_weight": float(group["max_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_selected_model_counts(pred_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：汇总 hard top-1 选中专家分布，用于诊断权重是否塌缩到单一专家。"""
    rows: List[Dict[str, object]] = []
    for (router_name, config_name), group in pred_df.groupby(["router_name", "config_name"], sort=True):
        counts = group["selected_model"].value_counts().reindex(MODEL_COLUMNS, fill_value=0)
        for model_name, count in counts.items():
            rows.append(
                {
                    "router_name": router_name,
                    "config_name": config_name,
                    "selected_model": model_name,
                    "count": int(count),
                    "ratio": float(count / len(group)),
                }
            )
    return pd.DataFrame(rows)
