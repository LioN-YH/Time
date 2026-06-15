#!/usr/bin/env python3
"""
文件功能：
    使用冻结视觉 encoder embedding 训练 Stage 1 Visual Router。

设计约束：
    - 输入特征可以来自历史离线 embedding manifest，也可以由 online 入口以内存
      `sample_key -> embedding` 字典传入；
    - router 训练只使用 `vali` split，评估只使用 `test` split；
    - 每个 `config_name` 独立训练一个 router，不跨历史长度/预测长度共享动作空间；
    - 参考 TimeFuse 的 ModelFusor 思路，router 输出五个下游专家的 softmax 权重；
    - hard top-1 routing 取权重最大的专家，soft fusion 使用权重加权同一 sample_key
      下五个专家的预测数组；
    - 不把 test oracle 误差、未来 y 或专家误差作为输入特征。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import (  # noqa: E402
    EPS,
    MODEL_COLUMNS,
    add_soft_fusion_metrics,
    compute_array_metrics,
    load_prediction_lookup,
    load_prediction_tensors_for_samples,
    summarize_hard_predictions,
    summarize_selected_model_counts,
    summarize_soft_fusion,
)


DEFAULT_LABELS_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "window_oracle_labels_with_tsf_cell.csv"
)
DEFAULT_PREDICTION_MANIFEST_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "manifest.csv"
)
ROUTER_VERSION_BY_MODE = {
    "classification": "visual_router_mlp_v1_classification",
    "fusion_huber_kl": "visual_router_mlp_v2_fusion_huber_kl",
}


def now_token() -> str:
    """函数功能：生成 run 目录时间戳，精确到微秒避免输出目录重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata 和 Markdown 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Visual Router 训练与评估参数。"""
    parser = argparse.ArgumentParser(description="Train a TimeFuse-style MLP Visual Router for Stage 1.")
    parser.add_argument("--embedding-manifest-path", type=Path, required=True, help="ViT embedding manifest CSV。")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="window oracle labels CSV。")
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST_PATH, help="prediction cache manifest CSV，用于 soft fusion。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="oracle label 和辅助误差分布使用的指标。")
    parser.add_argument(
        "--router-mode",
        choices=["classification", "fusion_huber_kl"],
        default="fusion_huber_kl",
        help="classification 保留旧 CE 分类 baseline；fusion_huber_kl 用融合预测误差训练权重。",
    )
    parser.add_argument("--huber-beta", type=float, default=0.1, help="fusion_huber_kl 主损失 SmoothL1Loss beta。")
    parser.add_argument("--kl-tau", type=float, default=0.5, help="soft oracle q_i=softmax(-error_i/tau) 的温度。")
    parser.add_argument("--lambda-kl", type=float, default=0.1, help="KL 辅助损失权重。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认基于 output-root 生成时间戳目录。")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden dimension。")
    parser.add_argument("--dropout", type=float, default=0.10, help="MLP dropout rate。")
    parser.add_argument("--epochs", type=int, default=300, help="训练 epoch 数。")
    parser.add_argument("--batch-size", type=int, default=32, help="router 训练 batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay。")
    parser.add_argument("--seed", type=int, default=16, help="随机种子。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="训练设备。")
    parser.add_argument("--skip-soft-fusion", action="store_true", help="只评估 hard top-1，不读取预测数组。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束时打印多少行预测预览。")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析训练设备，auto 优先 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 PyTorch CUDA 不可用")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    """函数功能：固定主要随机源，保证小规模 smoke 结果便于复核。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VisualMLPRouter(nn.Module):
    """
    类功能：
        TimeFuse-style 视觉 fusor/router。

    输入：
        ViT embedding，经 vali-fitted StandardScaler 标准化。

    输出：
        logits，推理时通过 softmax 得到五个专家的融合权重。这里用小型 MLP 替代
        TimeFuse 原始 `Linear -> softmax`，以允许视觉 embedding 经过一层非线性压缩。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """函数功能：使用 Kaiming/Xavier 初始化，使小样本训练更稳定。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：前向输出未归一化 logits。"""
        return self.network(features)


def load_labels(labels_path: Path, metric: str) -> pd.DataFrame:
    """函数功能：读取 oracle labels，并筛选指定 metric。"""
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels 文件：{labels_path}")
    labels_df = pd.read_csv(labels_path)
    required_cols = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "oracle_model",
        "oracle_value",
        "metric",
        *MODEL_COLUMNS,
    }
    missing_cols = sorted(required_cols.difference(labels_df.columns))
    if missing_cols:
        raise ValueError(f"labels 文件缺少字段：{missing_cols}")
    labels_df = labels_df[labels_df["metric"] == metric].copy()
    if labels_df.empty:
        raise ValueError(f"labels 文件中没有 metric={metric} 的记录")
    if labels_df["sample_key"].duplicated().any():
        dup_keys = labels_df.loc[labels_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"metric={metric} 的 labels sample_key 重复，示例：{dup_keys}")
    return labels_df


def load_embedding_manifest(manifest_path: Path) -> pd.DataFrame:
    """函数功能：读取 ViT embedding manifest 并做基础字段校验。"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 embedding manifest：{manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    required_cols = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "embedding_path",
        "embedding_dim",
        "finite",
    }
    missing_cols = sorted(required_cols.difference(manifest_df.columns))
    if missing_cols:
        raise ValueError(f"embedding manifest 缺少字段：{missing_cols}")
    if manifest_df["sample_key"].duplicated().any():
        dup_keys = manifest_df.loc[manifest_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"embedding manifest sample_key 重复，示例：{dup_keys}")
    if not manifest_df["finite"].astype(bool).all():
        raise ValueError("embedding manifest 中存在 finite=False")
    if manifest_df["embedding_dim"].nunique() != 1:
        raise ValueError("embedding manifest 中 embedding_dim 不一致")
    return manifest_df


def resolve_feature_path(path_text: str, manifest_dir: Path) -> Path:
    """函数功能：解析 manifest 中可能为相对路径或绝对路径的 embedding_path。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return manifest_dir / path


def load_embedding_matrix(
    merged_df: pd.DataFrame,
    manifest_dir: Path,
    feature_lookup: Optional[Mapping[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    函数功能：
        按 DataFrame 顺序取得视觉特征矩阵。

    设计说明：
        离线路径从 `embedding_path` 读取 `.npy`；online router 路径则传入
        `sample_key -> embedding` 的运行内字典。二者共享后续 scaler、MLP 训练和
        hard/soft fusion 评估，避免复制训练逻辑。
    """
    features: List[np.ndarray] = []
    for row in merged_df.itertuples(index=False):
        if feature_lookup is None:
            embedding_path = resolve_feature_path(str(row.embedding_path), manifest_dir)
            if not embedding_path.exists():
                raise FileNotFoundError(f"找不到 embedding 文件：{embedding_path}")
            embedding = np.load(embedding_path).astype(np.float32)
        else:
            sample_key = str(row.sample_key)
            if sample_key not in feature_lookup:
                raise KeyError(f"online feature lookup 缺少 sample_key={sample_key}")
            embedding = np.asarray(feature_lookup[sample_key], dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError(f"embedding 非法：sample_key={row.sample_key} shape={embedding.shape}")
        features.append(embedding)
    matrix = np.stack(features, axis=0)
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix 维度异常：{matrix.shape}")
    return matrix


def join_embeddings_and_labels(embedding_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：用 sample_key 和稳定元信息严格 join embedding manifest 与 oracle labels。"""
    join_cols = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    merged_df = embedding_df.merge(labels_df, on=join_cols, how="inner", suffixes=("_embedding", ""))
    if len(merged_df) != len(embedding_df) or len(merged_df) != len(labels_df):
        missing_embedding = sorted(set(labels_df["sample_key"]) - set(embedding_df["sample_key"]))
        missing_label = sorted(set(embedding_df["sample_key"]) - set(labels_df["sample_key"]))
        raise ValueError(
            f"embedding/label join 不完整：missing_embedding={missing_embedding[:10]} missing_label={missing_label[:10]}"
        )
    return merged_df.sort_values(join_cols).reset_index(drop=True)


def make_class_weight(labels: Sequence[str], device: torch.device) -> torch.Tensor:
    """
    函数功能：
        按 vali oracle label 频次构造 balanced class weight。

    说明：
        类别缺失时对应权重为 0；由于该类不会作为 target 出现，不影响 loss，但会在
        metadata 中记录，避免静默误解为五类全都参与监督。
    """
    counts = pd.Series(labels).value_counts().reindex(MODEL_COLUMNS, fill_value=0).astype(float)
    present_count = int((counts > 0).sum())
    weights = np.zeros(len(MODEL_COLUMNS), dtype=np.float32)
    if present_count == 0:
        raise ValueError("vali labels 为空，无法构造 class weight")
    total = float(counts.sum())
    for idx, model_name in enumerate(MODEL_COLUMNS):
        if counts[model_name] > 0:
            weights[idx] = total / (present_count * counts[model_name])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def router_version_for_mode(router_mode: str) -> str:
    """函数功能：根据训练模式返回稳定 router 名称，便于比较表区分 baseline 与 fusion 版本。"""
    if router_mode not in ROUTER_VERSION_BY_MODE:
        raise ValueError(f"未知 router_mode={router_mode}")
    return ROUTER_VERSION_BY_MODE[router_mode]


def validate_training_args(args: argparse.Namespace) -> None:
    """函数功能：提前校验 fusion loss 相关超参，避免训练中出现静默 NaN。"""
    if args.huber_beta <= 0:
        raise ValueError("--huber-beta 必须为正数")
    if args.kl_tau <= 0:
        raise ValueError("--kl-tau 必须为正数")
    if args.lambda_kl < 0:
        raise ValueError("--lambda-kl 必须 >= 0")


def train_router_for_config(
    *,
    config_name: str,
    config_df: pd.DataFrame,
    manifest_dir: Path,
    prediction_lookup: Optional[Mapping[Tuple[str, str], Dict[str, object]]],
    args: argparse.Namespace,
    device: torch.device,
    feature_lookup: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[VisualMLPRouter, StandardScaler, Dict[str, object]]:
    """函数功能：对单个 config_name 训练 vali->test 视觉 MLP router。"""
    vali_df = config_df[config_df["split"] == "vali"].copy()
    test_df = config_df[config_df["split"] == "test"].copy()
    if vali_df.empty or test_df.empty:
        raise ValueError(f"config_name={config_name} 需要同时包含 vali/test 样本")

    labels_seen = sorted(vali_df["oracle_model"].unique().tolist())
    if args.router_mode == "classification" and len(labels_seen) < 2:
        raise ValueError(
            f"config_name={config_name} 的 vali oracle label 少于 2 类，无法训练分类 router；labels={labels_seen}"
        )

    x_vali = load_embedding_matrix(vali_df, manifest_dir, feature_lookup=feature_lookup)
    y_vali = np.array([MODEL_COLUMNS.index(label) for label in vali_df["oracle_model"]], dtype=np.int64)
    scaler = StandardScaler()
    x_vali_scaled = scaler.fit_transform(x_vali).astype(np.float32)

    router = VisualMLPRouter(
        input_dim=int(x_vali_scaled.shape[1]),
        hidden_dim=int(args.hidden_dim),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(args.dropout),
    ).to(device)

    if args.router_mode == "classification":
        dataset = TensorDataset(torch.from_numpy(x_vali_scaled), torch.from_numpy(y_vali))
    else:
        if prediction_lookup is None:
            raise ValueError("fusion_huber_kl 模式需要 prediction_lookup 读取五专家 y_pred/y_true")
        y_pred_vali, y_true_vali, expert_errors_vali = load_prediction_tensors_for_samples(
            vali_df["sample_key"].astype(str).tolist(),
            prediction_lookup,
            error_metric=str(args.metric),
        )
        soft_oracle_vali = torch.softmax(
            -torch.from_numpy(expert_errors_vali) / float(args.kl_tau),
            dim=1,
        ).to(dtype=torch.float32)
        dataset = TensorDataset(
            torch.from_numpy(x_vali_scaled),
            torch.from_numpy(y_pred_vali),
            torch.from_numpy(y_true_vali),
            soft_oracle_vali,
        )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, generator=generator)

    optimizer = torch.optim.AdamW(router.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    classification_criterion = nn.CrossEntropyLoss(weight=make_class_weight(vali_df["oracle_model"].tolist(), device))
    huber_criterion = nn.SmoothL1Loss(beta=float(args.huber_beta))

    loss_history: List[float] = []
    huber_history: List[float] = []
    kl_history: List[float] = []
    router.train()
    for _ in range(int(args.epochs)):
        epoch_losses = []
        epoch_huber_losses = []
        epoch_kl_losses = []
        for batch in loader:
            batch_x = batch[0].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(batch_x)
            if args.router_mode == "classification":
                batch_y = batch[1].to(device=device)
                loss = classification_criterion(logits, batch_y)
            else:
                batch_pred = batch[1].to(device=device)
                batch_true = batch[2].to(device=device)
                batch_q = batch[3].to(device=device)
                weights = torch.softmax(logits, dim=1)
                weight_shape = (weights.shape[0], weights.shape[1], *([1] * (batch_pred.ndim - 2)))
                fused_pred = (weights.view(weight_shape) * batch_pred).sum(dim=1)
                huber_loss = huber_criterion(fused_pred, batch_true)
                # KL(q || p_router)：q 来自五专家训练误差，p_router 来自视觉 embedding。
                kl_loss = F.kl_div(torch.log_softmax(logits, dim=1), batch_q, reduction="batchmean")
                loss = huber_loss + float(args.lambda_kl) * kl_loss
                epoch_huber_losses.append(float(huber_loss.detach().cpu().item()))
                epoch_kl_losses.append(float(kl_loss.detach().cpu().item()))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        loss_history.append(float(np.mean(epoch_losses)))
        if epoch_huber_losses:
            huber_history.append(float(np.mean(epoch_huber_losses)))
            kl_history.append(float(np.mean(epoch_kl_losses)))

    with torch.inference_mode():
        train_logits = router(torch.from_numpy(x_vali_scaled).to(device=device))
        train_weights = torch.softmax(train_logits, dim=1)
        train_pred = train_weights.argmax(dim=1).detach().cpu().numpy()
    train_accuracy = float((train_pred == y_vali).mean())
    train_entropy = (-(train_weights * torch.log(train_weights.clamp_min(EPS))).sum(dim=1)).detach().cpu().numpy()
    train_max_weight = train_weights.max(dim=1).values.detach().cpu().numpy()

    train_fusion_metrics: Dict[str, float] = {}
    if args.router_mode == "fusion_huber_kl":
        with torch.inference_mode():
            y_pred_tensor = torch.from_numpy(y_pred_vali).to(device=device)
            y_true_tensor = torch.from_numpy(y_true_vali).to(device=device)
            weight_shape = (train_weights.shape[0], train_weights.shape[1], *([1] * (y_pred_tensor.ndim - 2)))
            fused_train = (train_weights.view(weight_shape) * y_pred_tensor).sum(dim=1)
            train_error = fused_train - y_true_tensor
            train_fusion_metrics = {
                "train_fusion_mae": float(train_error.abs().mean().detach().cpu().item()),
                "train_fusion_mse": float((train_error ** 2).mean().detach().cpu().item()),
            }

    metadata = {
        "config_name": config_name,
        "router_mode": args.router_mode,
        "vali_sample_count": int(len(vali_df)),
        "test_sample_count": int(len(test_df)),
        "embedding_dim": int(x_vali_scaled.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "labels_seen": labels_seen,
        "label_counts": {str(k): int(v) for k, v in vali_df["oracle_model"].value_counts().reindex(MODEL_COLUMNS, fill_value=0).items()},
        "final_train_loss": float(loss_history[-1]),
        "initial_train_loss": float(loss_history[0]),
        "final_huber_loss": float(huber_history[-1]) if huber_history else None,
        "final_kl_loss": float(kl_history[-1]) if kl_history else None,
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "train_label_accuracy": train_accuracy,
        "train_mean_weight_entropy": float(np.mean(train_entropy)),
        "train_mean_normalized_weight_entropy": float(np.mean(train_entropy) / np.log(len(MODEL_COLUMNS))),
        "train_mean_max_weight": float(np.mean(train_max_weight)),
        **train_fusion_metrics,
    }
    return router, scaler, metadata


def predict_router_for_config(
    *,
    router: VisualMLPRouter,
    scaler: StandardScaler,
    config_df: pd.DataFrame,
    manifest_dir: Path,
    device: torch.device,
    router_name: str,
    feature_lookup: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """函数功能：对单个 config_name 的 test split 输出专家权重和 hard top-1 结果。"""
    test_df = config_df[config_df["split"] == "test"].copy().reset_index(drop=True)
    x_test = load_embedding_matrix(test_df, manifest_dir, feature_lookup=feature_lookup)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    router.eval()
    with torch.inference_mode():
        logits = router(torch.from_numpy(x_test_scaled).to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
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


def compare_with_baselines(
    output_dir: Path,
    labels_path: Path,
    visual_summary: pd.DataFrame,
    soft_summary: Optional[pd.DataFrame],
    metric: str,
) -> pd.DataFrame:
    """函数功能：若同目录存在 baseline/结构特征结果，则生成同表比较。"""
    rows: List[Dict[str, object]] = []
    labels_dir = labels_path.parent

    baseline_path = labels_dir / "baseline_summary.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        for row in baseline_df.itertuples(index=False):
            rows.append(
                {
                    "method": str(row.baseline),
                    "method_status": "active",
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae_like_value": float(row.selected_value),
                    "oracle_value": float(row.oracle_value),
                    "regret_to_oracle": float(row.regret_to_oracle),
                    "oracle_label_accuracy": float(row.oracle_label_accuracy),
                    "mean_weight_entropy": pd.NA,
                    "mean_normalized_weight_entropy": pd.NA,
                    "mean_max_weight": pd.NA,
                    "source": str(baseline_path),
                }
            )

    structure_path = RUN_OUTPUT_ROOT / "2026-06-13_113713_308023_visual_router_stage1_structure_feature_pilot" / "structure_router_summary.csv"
    if structure_path.exists():
        structure_df = pd.read_csv(structure_path)
        for row in structure_df.itertuples(index=False):
            # 旧结构特征 LogisticRegression 只保留作历史附录，避免后续同表比较误读为主 baseline。
            baseline_status = getattr(row, "baseline_status", "legacy_deprecated")
            rows.append(
                {
                    "method": str(row.router_name),
                    "method_status": str(baseline_status),
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae_like_value": float(row.selected_value),
                    "oracle_value": float(row.oracle_value),
                    "regret_to_oracle": float(row.regret_to_oracle),
                    "oracle_label_accuracy": float(row.oracle_label_accuracy),
                    "mean_weight_entropy": pd.NA,
                    "mean_normalized_weight_entropy": pd.NA,
                    "mean_max_weight": pd.NA,
                    "source": str(structure_path),
                }
            )

    for row in visual_summary.itertuples(index=False):
        rows.append(
            {
                "method": str(row.router_name),
                "method_status": "active",
                "config_name": str(row.config_name),
                "sample_count": int(row.sample_count),
                "mae_like_value": float(row.selected_value),
                "oracle_value": float(row.oracle_value),
                "regret_to_oracle": float(row.regret_to_oracle),
                "oracle_label_accuracy": float(row.oracle_label_accuracy),
                "mean_weight_entropy": float(row.mean_weight_entropy),
                "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                "mean_max_weight": float(row.mean_max_weight),
                "source": str(output_dir / "visual_router_summary.csv"),
            }
        )

    if soft_summary is not None:
        soft_metric_col = "soft_fusion_mae" if metric == "mae" else "soft_fusion_mse"
        for row in soft_summary.itertuples(index=False):
            soft_value = float(getattr(row, soft_metric_col))
            rows.append(
                {
                    "method": str(row.router_name),
                    "method_status": "active",
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae_like_value": soft_value,
                    "oracle_value": float(row.oracle_value),
                    "regret_to_oracle": float(soft_value - row.oracle_value),
                    "oracle_label_accuracy": pd.NA,
                    "mean_weight_entropy": float(row.mean_weight_entropy),
                    "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                    "mean_max_weight": float(row.mean_max_weight),
                    "source": str(output_dir / "visual_router_soft_fusion_summary.csv"),
                }
            )

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df

    global_rows = comparison_df[comparison_df["method"] == "global_best_single"][["config_name", "mae_like_value"]]
    global_rows = global_rows.rename(columns={"mae_like_value": "global_best_single_value"})
    comparison_df = comparison_df.merge(global_rows, on="config_name", how="left")
    comparison_df["relative_improvement_vs_global_best_single"] = (
        comparison_df["global_best_single_value"] - comparison_df["mae_like_value"]
    ) / comparison_df["global_best_single_value"]
    return comparison_df.drop(columns=["global_best_single_value"]).sort_values(["config_name", "mae_like_value"]).reset_index(drop=True)


def frame_to_markdown(df: pd.DataFrame, *, float_digits: int = 6) -> str:
    """函数功能：将 DataFrame 转成 Markdown 表格，避免依赖 tabulate。"""
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


def write_summary_md(
    *,
    output_dir: Path,
    hard_summary: pd.DataFrame,
    soft_summary: Optional[pd.DataFrame],
    selected_counts: pd.DataFrame,
    comparison_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写出中文 Markdown 摘要，记录训练口径和 smoke 结果。"""
    lines = [
        "# Stage 1 Visual Router Smoke",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## Router 口径",
        "",
        "- 输入：冻结 ViT embedding；`StandardScaler` 只在 vali split 上 fit。",
        "- 模型：TimeFuse-style 小型 MLP fusor，输出五专家 softmax 权重。",
        f"- 训练模式：`{metadata['router_mode']}`。",
        (
            "- fusion_huber_kl：用 router 权重融合五专家 `y_pred`，主损失为 "
            f"`SmoothL1Loss(beta={metadata['huber_beta']})`；KL 辅助目标为 "
            f"`softmax(-error/tau)`，`tau={metadata['kl_tau']}`，"
            f"`lambda_kl={metadata['lambda_kl']}`。"
            if metadata["router_mode"] == "fusion_huber_kl"
            else "- classification：保留旧版 oracle hard label `CrossEntropyLoss` baseline。"
        ),
        "- 评估：test split hard top-1 routing 与 soft fusion。",
        "- 约束：不使用未来 y、test oracle 误差或专家误差作为输入特征。",
        "",
        "## Hard Top-1 Summary",
        "",
        frame_to_markdown(hard_summary),
        "",
    ]
    if soft_summary is not None:
        lines.extend(["## Soft Fusion Summary", "", frame_to_markdown(soft_summary), ""])
    lines.extend(["## Top-1 选中专家分布", "", frame_to_markdown(selected_counts), ""])
    lines.extend(
        [
            "## Baseline Comparison",
            "",
            frame_to_markdown(
                comparison_df[
                    [
                        "method",
                        "method_status",
                        "config_name",
                        "sample_count",
                        "mae_like_value",
                        "oracle_value",
                        "regret_to_oracle",
                        "oracle_label_accuracy",
                        "mean_weight_entropy",
                        "mean_normalized_weight_entropy",
                        "mean_max_weight",
                        "relative_improvement_vs_global_best_single",
                    ]
                ]
                if not comparison_df.empty
                else comparison_df
            ),
            "",
            "## 输出文件",
            "",
            f"- `visual_router_predictions.csv`: `{output_dir / 'visual_router_predictions.csv'}`",
            f"- `visual_router_summary.csv`: `{output_dir / 'visual_router_summary.csv'}`",
            f"- `visual_router_soft_fusion_predictions.csv`: `{output_dir / 'visual_router_soft_fusion_predictions.csv'}`",
            f"- `visual_router_selected_model_counts.csv`: `{output_dir / 'visual_router_selected_model_counts.csv'}`",
            f"- `visual_router_comparison.csv`: `{output_dir / 'visual_router_comparison.csv'}`",
            f"- `visual_router_metadata.json`: `{output_dir / 'visual_router_metadata.json'}`",
            "",
        ]
    )
    (output_dir / "visual_router_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行视觉 MLP router 训练、test 评估和结果落盘。"""
    args = parse_args()
    validate_training_args(args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_visual_router_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    router_name = router_version_for_mode(str(args.router_mode))

    embedding_df = load_embedding_manifest(args.embedding_manifest_path)
    labels_df = load_labels(args.labels_path, args.metric)
    merged_df = join_embeddings_and_labels(embedding_df, labels_df)
    manifest_dir = args.embedding_manifest_path.parent
    prediction_lookup: Optional[Mapping[Tuple[str, str], Dict[str, object]]] = None
    if args.router_mode == "fusion_huber_kl" or not args.skip_soft_fusion:
        prediction_lookup = load_prediction_lookup(args.prediction_manifest_path)

    hard_prediction_frames: List[pd.DataFrame] = []
    config_metadata: List[Dict[str, object]] = []
    for config_name, config_df in merged_df.groupby("config_name", sort=True):
        router, scaler, metadata = train_router_for_config(
            config_name=str(config_name),
            config_df=config_df,
            manifest_dir=manifest_dir,
            prediction_lookup=prediction_lookup,
            args=args,
            device=device,
        )
        config_metadata.append(metadata)
        hard_prediction_frames.append(
            predict_router_for_config(
                router=router,
                scaler=scaler,
                config_df=config_df,
                manifest_dir=manifest_dir,
                device=device,
                router_name=router_name,
            )
        )

    hard_pred_df = pd.concat(hard_prediction_frames, ignore_index=True)
    hard_summary_df = summarize_hard_predictions(hard_pred_df)
    selected_counts_df = summarize_selected_model_counts(hard_pred_df)

    soft_pred_df: Optional[pd.DataFrame] = None
    soft_summary_df: Optional[pd.DataFrame] = None
    if not args.skip_soft_fusion:
        assert prediction_lookup is not None
        soft_pred_df = add_soft_fusion_metrics(hard_pred_df, prediction_lookup)
        soft_summary_df = summarize_soft_fusion(soft_pred_df)

    comparison_df = compare_with_baselines(output_dir, args.labels_path, hard_summary_df, soft_summary_df, args.metric)

    hard_pred_df.to_csv(output_dir / "visual_router_predictions.csv", index=False)
    hard_summary_df.to_csv(output_dir / "visual_router_summary.csv", index=False)
    selected_counts_df.to_csv(output_dir / "visual_router_selected_model_counts.csv", index=False)
    if soft_pred_df is not None and soft_summary_df is not None:
        soft_pred_df.to_csv(output_dir / "visual_router_soft_fusion_predictions.csv", index=False)
        soft_summary_df.to_csv(output_dir / "visual_router_soft_fusion_summary.csv", index=False)
    comparison_df.to_csv(output_dir / "visual_router_comparison.csv", index=False)

    run_metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "router_version": router_name,
        "router_mode": args.router_mode,
        "embedding_manifest_path": str(args.embedding_manifest_path),
        "labels_path": str(args.labels_path),
        "prediction_manifest_path": str(args.prediction_manifest_path),
        "metric": args.metric,
        "model_columns": MODEL_COLUMNS,
        "training_split": "vali",
        "evaluation_split": "test",
        "device": str(device),
        "seed": int(args.seed),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "soft_fusion_enabled": not bool(args.skip_soft_fusion),
        "config_metadata": config_metadata,
        "input_exclusions": ["future_y", "test_oracle_error_as_feature", "expert_error_as_feature"],
    }
    (output_dir / "visual_router_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_md(
        output_dir=output_dir,
        hard_summary=hard_summary_df,
        soft_summary=soft_summary_df,
        selected_counts=selected_counts_df,
        comparison_df=comparison_df,
        metadata=run_metadata,
    )

    print(f"wrote visual router outputs to {output_dir}")
    print(hard_summary_df.to_string(index=False))
    if soft_summary_df is not None:
        print(soft_summary_df.to_string(index=False))
    preview_cols = ["sample_key", "selected_model", "selected_value", "oracle_model", "oracle_value", *[f"weight_{m}" for m in MODEL_COLUMNS]]
    print(hard_pred_df[preview_cols].head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
