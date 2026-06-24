#!/usr/bin/env python3
"""
文件功能：
    运行 Visual Router V2 Round2 full-scale streaming FiLM 主线。

核心约束：
    - 输入路径固定为 `x -> Round2 layout pseudo image -> frozen ViT -> mean_patch embedding -> FiLMRouter`；
    - RevIN aux 只作为 FiLM condition，不 concat 到 visual embedding；
    - pseudo image tensor 和 ViT embedding 只存在于 batch runtime，不落盘为长期 cache；
    - prediction manifest 只通过 SQLite 子集索引按 batch 查询，不构建全量 Python lookup。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
for path in (REPO_ROOT, WORKSPACE, QUITO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if sys.path[0] != str(REPO_ROOT):
    # 当前 v2 repo 必须优先于旧 `/home/shiyuhong/Time`，否则会导入旧实验包。
    sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT))

from quito.config.training import TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402
from visual_router_experiments.common.prediction_array_io import resolve_cache_array_path  # noqa: E402
from visual_router_experiments.common.pseudo_imageization import encoder_normalize  # noqa: E402
from visual_router_experiments.common.round2_layout_registry import imageize_round2_layout  # noqa: E402
from visual_router_experiments.common.vit_embedding_utils import (  # noqa: E402
    batch_required_pairs,
    build_required_index,
    parse_period_candidate_arg,
    pool_vit_outputs,
    resolve_dtype,
)
from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import DEFAULT_ORACLE_LABELS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router import (  # noqa: E402
    DEFAULT_PREDICTION_MANIFEST_PATH,
    EPS,
    load_labels,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import (  # noqa: E402
    DEFAULT_CONFIG,
    _timer_start,
    _timer_stop,
    display_time,
    load_data_config,
    mode_from_split,
    now_token,
    resolve_device,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    SQLitePredictionIndex,
    append_csv,
    append_latency,
    build_lightweight_prediction_index,
    filter_stream_shard,
    limit_samples_per_split,
    load_prediction_tensors_from_lightweight_index,
    load_vit_model_with_retry,
    scaler_to_state,
    windows_from_labels,
    write_status,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import (  # noqa: E402
    AUX_FEATURE_COLUMNS,
    compute_revin_aux_from_x,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import add_batch_fusion_metrics  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_OUTPUT_ROOT = DATA2_RUN_OUTPUT_ROOT
DEFAULT_PREDICTION_MANIFEST = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "merged_cache"
    / "manifest.csv"
)
SCRIPT_VERSION = "visual_router_v2_round2_fullscale_streaming_film_v1"
ROUTER_NAME = "visual_router_v2_round2_spatial_panel_3view_film_mean_patch_aux_streaming"


class FiLMRouter(nn.Module):
    """类功能：用 RevIN aux 对 mean_patch visual hidden 做 FiLM 调制并输出五专家 logits。"""

    def __init__(self, visual_dim: int, aux_dim: int, hidden_dim: int, film_hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.visual_backbone = nn.Sequential(
            nn.Linear(int(visual_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(int(aux_dim), int(film_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(film_hidden_dim), int(hidden_dim) * 2),
        )
        self.head = nn.Linear(int(hidden_dim), int(output_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        """函数功能：初始化 FiLM router，并让 FiLM 分支初始接近恒等调制。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)
        last = self.film_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, visual_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        """函数功能：返回五专家 logits。"""
        hidden = self.visual_backbone(visual_features)
        gamma, beta = self.film_mlp(aux_features).chunk(2, dim=1)
        return self.head(hidden * (1.0 + gamma) + beta)


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2 fullscale streaming FiLM 参数。"""
    parser = argparse.ArgumentParser(description="Train Round2 fullscale streaming FiLM visual router.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--prediction-index-path", type=Path, default=None, help="显式复用已有 SQLite prediction index；默认按本次 sample_key 构建 subset index。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--layout-name", choices=["spatial_panel_3view"], default="spatial_panel_3view")
    parser.add_argument("--encoder-name", default="google/vit-base-patch16-224")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vit-data-parallel", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--normalization-preset", default="hf_vit_0_5")
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates")
    parser.add_argument("--period-candidates", default=None)
    parser.add_argument("--pooling", choices=["mean_patch"], default="mean_patch")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--film-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--stream-shard-index", type=int, default=0)
    parser.add_argument("--stream-shard-count", type=int, default=1)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--chunk-read-rows", type=int, default=200_000)
    parser.add_argument("--status-update-interval", type=int, default=50)
    parser.add_argument("--skip-soft-fusion", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-rows", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """函数功能：固定主要随机源，便于 smoke 和 shard 复核。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve_period_candidates(args: argparse.Namespace, history_length: int) -> Optional[List[int]]:
    """函数功能：解析固定候选周期；dynamic FFT 模式返回 None。"""
    values = parse_period_candidate_arg(args.period_candidates)
    if args.period_selection != "fixed_candidates":
        return None
    if values is not None:
        return values
    from visual_router_experiments.common.pseudo_imageization import make_default_period_candidates

    return [int(v) for v in make_default_period_candidates(int(history_length), device=torch.device("cpu")).tolist()]


def required_prediction_sample_keys(labels_df: pd.DataFrame, *, train_only: bool, skip_soft_fusion: bool) -> List[str]:
    """函数功能：只索引训练 vali 和需要 raw-soft 评估的 test sample_key。"""
    parts = [labels_df.loc[labels_df["split"].astype(str) == "vali", "sample_key"].astype(str)]
    if not train_only and not skip_soft_fusion:
        parts.append(labels_df.loc[labels_df["split"].astype(str) == "test", "sample_key"].astype(str))
    return pd.concat(parts, ignore_index=True).drop_duplicates().tolist()


def make_round2_pixel_values(
    x_cpu: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    period_candidate_values: Optional[Sequence[int]],
) -> torch.Tensor:
    """
    函数功能：
        使用 Round2 layout registry 构造 ViT pixel_values。

    说明：
        registry 输出为 [0,1] pseudo image；这里显式调用 encoder_normalize，
        保持和旧 streaming 入口的 HF ViT normalization 口径一致。
    """
    x_device = x_cpu.to(device=device, dtype=torch.float32, non_blocking=False)
    result = imageize_round2_layout(
        x_device,
        layout_name=str(args.layout_name),
        image_size=int(args.image_size),
        norm_mode=str(args.norm_mode),
        clip=float(args.clip),
        period_candidates=period_candidate_values,
        period_selection=str(args.period_selection),
    )
    return encoder_normalize(result.images.to(dtype=dtype), preset=str(args.normalization_preset))


def iter_round2_embedding_aux_batches(
    *,
    windows_df: pd.DataFrame,
    data_config,
    vit_model,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    period_candidate_values: Optional[Sequence[int]],
) -> Iterator[Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Dict[str, object]]]]:
    """函数功能：流式读取历史窗口，在线生成 mean_patch embedding 和 RevIN aux。"""
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"].astype(str), windows_df["config_name"].astype(str)))
    for split in sorted(windows_df["split"].astype(str).unique()):
        datasets = load_datasets(data_config=data_config, task=TaskType.EVALUATE, mode=mode_from_split(str(split)), cleanup=False, concat=False)
        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
            item_ids = sorted(item_id for req_split, req_dataset, item_id in required_index if req_split == split and req_dataset == dataset_name)
            for item_id in item_ids:
                item_dataset = dataset.copy() if hasattr(dataset, "copy") else None
                if item_dataset is None:
                    import copy

                    item_dataset = copy.deepcopy(dataset)
                item_dataset.select_user_data(int(item_id))
                channel_count = int(item_dataset.data.shape[0])
                required_for_item = required_index[(str(split), str(dataset_name), int(item_id))]
                for pair_batch in batch_required_pairs(required_for_item, int(args.embedding_batch_size)):
                    x_windows: List[np.ndarray] = []
                    sample_keys: List[str] = []
                    channel_ids: List[int] = []
                    window_indices: List[int] = []
                    for channel_id, window_index, sample_key in pair_batch:
                        if int(channel_id) >= channel_count:
                            raise ValueError(f"channel_id 越界：sample_key={sample_key}")
                        window_start = int(window_index)
                        window_end = window_start + int(data_config.seq_len)
                        x_window = item_dataset.data[int(channel_id), window_start:window_end, :]
                        if x_window.shape[0] != int(data_config.seq_len):
                            raise ValueError(f"历史窗口长度不完整：sample_key={sample_key} shape={x_window.shape}")
                        x_windows.append(x_window)
                        sample_keys.append(str(sample_key))
                        channel_ids.append(int(channel_id))
                        window_indices.append(int(window_index))

                    x_np = np.stack(x_windows, axis=0).astype(np.float32)
                    x_cpu = torch.from_numpy(x_np)
                    aux = compute_revin_aux_from_x(x_np, clip=float(args.clip))
                    with torch.inference_mode():
                        image_start = _timer_start(device)
                        pixel_values = make_round2_pixel_values(
                            x_cpu,
                            args=args,
                            device=device,
                            dtype=dtype,
                            period_candidate_values=period_candidate_values,
                        )
                        image_ms = _timer_stop(image_start, device)
                        forward_start = _timer_start(device)
                        outputs = vit_model(pixel_values=pixel_values)
                        embeddings = pool_vit_outputs(outputs, "mean_patch")
                        forward_ms = _timer_stop(forward_start, device)
                        embeddings_cpu = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()

                    rows: List[Dict[str, object]] = []
                    for row_idx, sample_key in enumerate(sample_keys):
                        key = PredictionCacheKey(
                            config_name=str(config_by_key[sample_key]),
                            split=str(split),
                            dataset_name=str(dataset_name),
                            item_id=int(item_id),
                            channel_id=int(channel_ids[row_idx]),
                            window_index=int(window_indices[row_idx]),
                        )
                        if key.as_string() != sample_key:
                            raise ValueError(f"sample_key 与元信息不一致：{sample_key} vs {key.as_string()}")
                        rows.append(
                            {
                                "sample_key": sample_key,
                                "config_name": str(config_by_key[sample_key]),
                                "split": str(split),
                                "dataset_name": str(dataset_name),
                                "item_id": int(item_id),
                                "channel_id": int(channel_ids[row_idx]),
                                "window_index": int(window_indices[row_idx]),
                                "history_length": int(data_config.seq_len),
                                "embedding_dim": int(embeddings_cpu.shape[1]),
                                "layout_name": str(args.layout_name),
                                "pooling": "mean_patch",
                            }
                        )
                    latency_rows = [
                        {
                            "split": str(split),
                            "dataset_name": str(dataset_name),
                            "item_id": int(item_id),
                            "batch_size": int(len(sample_keys)),
                            "imageization_ms": float(image_ms),
                            "encoder_forward_ms": float(forward_ms),
                            "imageization_per_window_ms": float(image_ms / len(sample_keys)),
                            "encoder_forward_per_window_ms": float(forward_ms / len(sample_keys)),
                            "device": str(device),
                        }
                    ]
                    yield pd.DataFrame(rows), embeddings_cpu.astype(np.float32), aux.astype(np.float32), latency_rows


def train_batch(
    *,
    router: FiLMRouter,
    optimizer: torch.optim.Optimizer,
    visual_scaler: StandardScaler,
    aux_scaler: StandardScaler,
    batch_manifest_df: pd.DataFrame,
    embeddings: np.ndarray,
    aux: np.ndarray,
    prediction_index: SQLitePredictionIndex,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    """函数功能：使用 fusion_huber_kl 目标更新一个 streaming batch。"""
    sample_keys = batch_manifest_df["sample_key"].astype(str).tolist()
    y_pred, y_true, expert_errors = load_prediction_tensors_from_lightweight_index(sample_keys, prediction_index, error_metric=str(args.metric))
    visual_scaled = visual_scaler.transform(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
    aux_scaled = aux_scaler.transform(np.asarray(aux, dtype=np.float32)).astype(np.float32)
    soft_oracle = torch.softmax(-torch.from_numpy(expert_errors) / float(args.kl_tau), dim=1).to(dtype=torch.float32)
    huber = nn.SmoothL1Loss(beta=float(args.huber_beta))
    losses: List[float] = []
    huber_losses: List[float] = []
    kl_losses: List[float] = []
    router.train()
    for start in range(0, len(sample_keys), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(sample_keys))
        batch_visual = torch.from_numpy(visual_scaled[start:stop]).to(device=device)
        batch_aux = torch.from_numpy(aux_scaled[start:stop]).to(device=device)
        batch_pred = torch.from_numpy(y_pred[start:stop]).to(device=device)
        batch_true = torch.from_numpy(y_true[start:stop]).to(device=device)
        batch_q = soft_oracle[start:stop].to(device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = router(batch_visual, batch_aux)
        weights = torch.softmax(logits, dim=1)
        weight_shape = (weights.shape[0], weights.shape[1], *([1] * (batch_pred.ndim - 2)))
        fused = (weights.view(weight_shape) * batch_pred).sum(dim=1)
        huber_loss = huber(fused, batch_true)
        kl_loss = F.kl_div(torch.log_softmax(logits, dim=1), batch_q, reduction="batchmean")
        loss = huber_loss + float(args.lambda_kl) * kl_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        huber_losses.append(float(huber_loss.detach().cpu().item()))
        kl_losses.append(float(kl_loss.detach().cpu().item()))
    return {"loss": float(np.mean(losses)), "huber_loss": float(np.mean(huber_losses)), "kl_loss": float(np.mean(kl_losses))}


def predict_batch(
    *,
    router: FiLMRouter,
    visual_scaler: StandardScaler,
    aux_scaler: StandardScaler,
    batch_manifest_df: pd.DataFrame,
    embeddings: np.ndarray,
    aux: np.ndarray,
    labels_by_key: Mapping[str, Mapping[str, object]],
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：输出一个 batch 的 hard top1 与五专家权重。"""
    visual_scaled = visual_scaler.transform(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
    aux_scaled = aux_scaler.transform(np.asarray(aux, dtype=np.float32)).astype(np.float32)
    router.eval()
    with torch.inference_mode():
        logits = router(torch.from_numpy(visual_scaled).to(device=device), torch.from_numpy(aux_scaled).to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    selected_indices = weights.argmax(axis=1)
    entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
    rows: List[Dict[str, object]] = []
    for row_idx, row in enumerate(batch_manifest_df.itertuples(index=False)):
        sample_key = str(row.sample_key)
        label_row = labels_by_key[sample_key]
        selected_model = MODEL_COLUMNS[int(selected_indices[row_idx])]
        out: Dict[str, object] = {
            "sample_set": "fullscale_test" if str(label_row["split"]) == "test" else "fullscale_vali",
            "variant": str(ROUTER_NAME),
            "layout_name": "spatial_panel_3view",
            "backend_style": "film_mean_patch_aux",
            "seed": int(seed),
            "router_name": ROUTER_NAME,
            "config_name": str(label_row["config_name"]),
            "sample_key": sample_key,
            "split": str(label_row["split"]),
            "dataset_name": str(label_row["dataset_name"]),
            "item_id": int(label_row["item_id"]),
            "channel_id": int(label_row["channel_id"]),
            "window_index": int(label_row["window_index"]),
            "selected_model": selected_model,
            "selected_value": float(label_row[selected_model]),
            "oracle_model": str(label_row["oracle_model"]),
            "oracle_value": float(label_row["oracle_value"]),
            "regret_to_oracle": float(label_row[selected_model] - label_row["oracle_value"]),
            "oracle_label_correct": bool(selected_model == label_row["oracle_model"]),
            "weight_entropy": float(entropy[row_idx]),
            "normalized_weight_entropy": float(entropy[row_idx] / math.log(len(MODEL_COLUMNS))),
            "max_weight": float(weights[row_idx].max()),
        }
        for model_idx, model_name in enumerate(MODEL_COLUMNS):
            out[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_outputs(output_dir: Path) -> pd.DataFrame:
    """函数功能：按 split 汇总 hard/raw-soft MAE、MSE 和路由行为。"""
    pred_path = output_dir / "visual_router_predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(pred_path)
    if (output_dir / "visual_router_soft_fusion_predictions.csv").exists():
        pred = pd.read_csv(output_dir / "visual_router_soft_fusion_predictions.csv")
    rows: List[Dict[str, object]] = []
    for split, group in pred.groupby("split", sort=False):
        row: Dict[str, object] = {
            "split": str(split),
            "sample_count": int(len(group)),
            "hard_top1_regret": float(group["regret_to_oracle"].mean()),
            "oracle_label_accuracy": float(group["oracle_label_correct"].astype(float).mean()),
            "weight_entropy": float(group["weight_entropy"].mean()),
            "mean_max_weight": float(group["max_weight"].mean()),
        }
        if "hard_top1_mae_from_array" in group.columns:
            row["hard_top1_MAE"] = float(group["hard_top1_mae_from_array"].mean())
            row["hard_top1_MSE"] = float(group["hard_top1_mse_from_array"].mean())
        if "soft_fusion_mae" in group.columns:
            row["raw_soft_MAE"] = float(group["soft_fusion_mae"].mean())
            row["raw_soft_MSE"] = float(group["soft_fusion_mse"].mean())
            row["raw_soft_regret"] = float((group["soft_fusion_mae"] - group["oracle_value"]).mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "visual_router_round2_fullscale_summary.csv", index=False)
    return summary


def build_subset_prediction_index(
    prediction_manifest_path: Path,
    *,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
    index_db_path: Path,
) -> SQLitePredictionIndex:
    """
    函数功能：
        为本次 train/test sample_key 构建轻量 SQLite prediction index。

    说明：
        与旧 streaming helper 的 schema 保持兼容；差异是当 expected_records 已经
        收齐时立即停止扫描，避免 smoke 为少量 sample_key 读完整个 52GB manifest。
    """
    if not prediction_manifest_path.exists():
        raise FileNotFoundError(f"找不到 prediction manifest：{prediction_manifest_path}")
    key_set = {str(key) for key in sample_keys}
    if not key_set:
        raise ValueError("prediction subset index 至少需要一个 sample_key")
    usecols = [
        "sample_key",
        "model_name",
        "y_true_path",
        "y_pred_path",
        "mae",
        "mse",
        "array_storage",
        "y_true_row_index",
        "y_pred_row_index",
    ]
    index_db_path = Path(index_db_path)
    index_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path = index_db_path.with_suffix(index_db_path.suffix + ".tmp")
    if tmp_db_path.exists():
        tmp_db_path.unlink()
    if index_db_path.exists():
        index_db_path.unlink()
    connection = sqlite3.connect(str(tmp_db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE prediction_index (
            sample_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            y_true_path TEXT NOT NULL,
            y_pred_path TEXT NOT NULL,
            mae REAL NOT NULL,
            mse REAL NOT NULL,
            array_storage TEXT,
            y_true_row_index INTEGER,
            y_pred_row_index INTEGER,
            PRIMARY KEY (sample_key, model_name)
        )
        """
    )
    expected_records = int(len(key_set) * len(MODEL_COLUMNS))
    matched_rows = 0
    rows_seen = 0
    try:
        for chunk_idx, chunk_df in enumerate(pd.read_csv(prediction_manifest_path, usecols=usecols, chunksize=int(chunk_read_rows)), start=1):
            rows_seen += int(len(chunk_df))
            matched_df = chunk_df[chunk_df["sample_key"].astype(str).isin(key_set)]
            if matched_df.empty:
                continue
            insert_rows = [
                (
                    str(row.sample_key),
                    str(row.model_name),
                    str(row.y_true_path),
                    str(row.y_pred_path),
                    float(row.mae),
                    float(row.mse),
                    str(row.array_storage) if pd.notna(row.array_storage) else None,
                    None if pd.isna(row.y_true_row_index) else int(row.y_true_row_index),
                    None if pd.isna(row.y_pred_row_index) else int(row.y_pred_row_index),
                )
                for row in matched_df.itertuples(index=False)
            ]
            try:
                connection.executemany(
                    """
                    INSERT INTO prediction_index (
                        sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                        array_storage, y_true_row_index, y_pred_row_index
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("prediction manifest 中 sample_key + model_name 存在重复") from exc
            connection.commit()
            matched_rows = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
            print(
                f"[manifest_index] chunks={chunk_idx} rows_seen={rows_seen} matched_rows={matched_rows}/{expected_records}",
                flush=True,
            )
            if matched_rows >= expected_records:
                break
        actual_records = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
        if actual_records != expected_records:
            raise ValueError(f"prediction subset 不完整：expected={expected_records} actual={actual_records} sample_keys={len(key_set)}")
        connection.execute("CREATE INDEX idx_prediction_index_sample_key ON prediction_index(sample_key)")
        connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        metadata = {
            "created_at": display_time(),
            "prediction_manifest_path": str(prediction_manifest_path),
            "target_sample_keys": int(len(key_set)),
            "expected_records": int(expected_records),
            "actual_records": int(actual_records),
            "chunk_read_rows": int(chunk_read_rows),
            "rows_seen_until_complete": int(rows_seen),
            "early_stop_when_complete": True,
        }
        connection.executemany("INSERT INTO index_metadata (key, value) VALUES (?, ?)", [(str(k), json.dumps(v, ensure_ascii=False)) for k, v in metadata.items()])
        connection.commit()
    except Exception:
        connection.close()
        if tmp_db_path.exists():
            tmp_db_path.unlink()
        raise
    connection.close()
    tmp_db_path.replace(index_db_path)
    return SQLitePredictionIndex(index_db_path, prediction_manifest_path.parent)


def main() -> None:
    """函数功能：执行 Round2 fullscale streaming FiLM 训练和 frozen test。"""
    args = parse_args()
    if int(args.epochs) <= 0:
        raise ValueError("--epochs 必须为正整数")
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    dtype = resolve_dtype(str(args.dtype), device)
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_v2_round2_fullscale_streaming_film_seed{int(args.seed)}"
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_status(output_dir, {"status": "running", "phase": "init", "seed": int(args.seed)})

    labels_df = load_labels(Path(args.labels_path), str(args.metric))
    labels_df = filter_stream_shard(labels_df, int(args.stream_shard_index), int(args.stream_shard_count))
    labels_df = limit_samples_per_split(labels_df, args.max_samples_per_split)
    windows_df = windows_from_labels(labels_df)
    train_windows_df = windows_df[windows_df["split"].astype(str) == "vali"].copy()
    test_windows_df = windows_df[windows_df["split"].astype(str) == "test"].copy()
    if train_windows_df.empty or (test_windows_df.empty and not bool(args.train_only)):
        raise ValueError("Round2 fullscale streaming 需要 vali train；非 train-only 还需要 test eval")
    labels_by_key = labels_df.set_index("sample_key").to_dict(orient="index")

    data_config = load_data_config(Path(args.config_path))
    period_candidate_values = resolve_period_candidates(args, int(data_config.seq_len))
    vit_model = load_vit_model_with_retry(args, device, dtype)
    if args.prediction_index_path is not None and Path(args.prediction_index_path).exists():
        # 正式 full-scale 已有共享 SQLite index 时直接复用，避免重复扫描 52GB manifest。
        prediction_index = SQLitePredictionIndex(Path(args.prediction_index_path), Path(args.prediction_manifest_path).parent)
        prediction_index_source = str(args.prediction_index_path)
    else:
        needed_prediction_keys = required_prediction_sample_keys(labels_df, train_only=bool(args.train_only), skip_soft_fusion=bool(args.skip_soft_fusion))
        prediction_index = build_subset_prediction_index(
            Path(args.prediction_manifest_path),
            sample_keys=needed_prediction_keys,
            chunk_read_rows=int(args.chunk_read_rows),
            index_db_path=output_dir / "prediction_manifest_index.sqlite",
        )
        prediction_index_source = str(output_dir / "prediction_manifest_index.sqlite")
    write_status(
        output_dir,
        {
            "status": "running",
            "phase": "prediction_index_ready",
            "prediction_index_source": prediction_index_source,
            "seed": int(args.seed),
        },
    )

    visual_scaler = StandardScaler()
    aux_scaler = StandardScaler()
    scaler_batches = 0
    scaler_samples = 0
    embedding_dim: Optional[int] = None
    for batch_manifest_df, embeddings, aux, latency_rows in iter_round2_embedding_aux_batches(
        windows_df=train_windows_df,
        data_config=data_config,
        vit_model=vit_model,
        args=args,
        device=device,
        dtype=dtype,
        period_candidate_values=period_candidate_values,
    ):
        visual_scaler.partial_fit(embeddings)
        aux_scaler.partial_fit(aux)
        append_csv(output_dir / "online_embedding_manifest.csv", batch_manifest_df)
        append_latency(output_dir, latency_rows, "scaler_fit")
        embedding_dim = int(embeddings.shape[1])
        scaler_batches += 1
        scaler_samples += int(len(batch_manifest_df))
        if scaler_batches % int(args.status_update_interval) == 0:
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "scaler_fit",
                    "scaler_batches": int(scaler_batches),
                    "scaler_samples": int(scaler_samples),
                    "seed": int(args.seed),
                },
            )
    write_status(output_dir, {"status": "running", "phase": "scaler_fit_completed", "scaler_batches": int(scaler_batches)})

    router = FiLMRouter(
        visual_dim=int(visual_scaler.n_features_in_),
        aux_dim=len(AUX_FEATURE_COLUMNS),
        hidden_dim=int(args.hidden_dim),
        film_hidden_dim=int(args.film_hidden_dim),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_batches = 0
    epoch_summaries: List[Dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        epoch_rows: List[Dict[str, float]] = []
        for batch_manifest_df, embeddings, aux, latency_rows in iter_round2_embedding_aux_batches(
            windows_df=train_windows_df,
            data_config=data_config,
            vit_model=vit_model,
            args=args,
            device=device,
            dtype=dtype,
            period_candidate_values=period_candidate_values,
        ):
            metrics = train_batch(
                router=router,
                optimizer=optimizer,
                visual_scaler=visual_scaler,
                aux_scaler=aux_scaler,
                batch_manifest_df=batch_manifest_df,
                embeddings=embeddings,
                aux=aux,
                prediction_index=prediction_index,
                args=args,
                device=device,
            )
            epoch_rows.append(metrics)
            append_latency(output_dir, latency_rows, f"train_epoch_{epoch}")
            total_batches += 1
            if total_batches % int(args.status_update_interval) == 0:
                write_status(output_dir, {"status": "running", "phase": "training", "epoch": int(epoch), "embedding_batches": int(total_batches)})
        epoch_summary = {
            "epoch": float(epoch),
            "loss": float(np.mean([row["loss"] for row in epoch_rows])),
            "huber_loss": float(np.mean([row["huber_loss"] for row in epoch_rows])),
            "kl_loss": float(np.mean([row["kl_loss"] for row in epoch_rows])),
        }
        epoch_summaries.append(epoch_summary)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"round2_film_seed{int(args.seed)}_epoch{epoch:04d}.pt"
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "router_name": ROUTER_NAME,
                "layout_name": str(args.layout_name),
                "backend_style": "film_mean_patch_aux",
                "seed": int(args.seed),
                "router_state_dict": router.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "visual_scaler_state": scaler_to_state(visual_scaler),
                "aux_scaler_state": scaler_to_state(aux_scaler),
                "model_columns": list(MODEL_COLUMNS),
                "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
                "epoch_summaries": epoch_summaries,
            },
            checkpoint_path,
        )
        write_status(output_dir, {"status": "running", "phase": "checkpoint_saved", "epoch": int(epoch), "latest_checkpoint_path": str(checkpoint_path)})

    if not bool(args.train_only):
        test_batches = 0
        test_samples = 0
        for batch_manifest_df, embeddings, aux, latency_rows in iter_round2_embedding_aux_batches(
            windows_df=test_windows_df,
            data_config=data_config,
            vit_model=vit_model,
            args=args,
            device=device,
            dtype=dtype,
            period_candidate_values=period_candidate_values,
        ):
            append_latency(output_dir, latency_rows, "test_predict")
            pred = predict_batch(
                router=router,
                visual_scaler=visual_scaler,
                aux_scaler=aux_scaler,
                batch_manifest_df=batch_manifest_df,
                embeddings=embeddings,
                aux=aux,
                labels_by_key=labels_by_key,
                seed=int(args.seed),
                device=device,
            )
            append_csv(output_dir / "visual_router_predictions.csv", pred)
            if not bool(args.skip_soft_fusion):
                pred_with_metrics = add_batch_fusion_metrics(pred, prediction_index=prediction_index, metric=str(args.metric), batch_size=int(args.eval_batch_size))
                append_csv(output_dir / "visual_router_soft_fusion_predictions.csv", pred_with_metrics)
            test_batches += 1
            test_samples += int(len(pred))
            if test_batches % int(args.status_update_interval) == 0:
                write_status(
                    output_dir,
                    {
                        "status": "running",
                        "phase": "test_predict",
                        "test_batches": int(test_batches),
                        "test_samples": int(test_samples),
                        "latest_checkpoint_path": str(checkpoint_path),
                        "seed": int(args.seed),
                    },
                )

    latency_path = output_dir / "online_embedding_latency_summary.csv"
    latency_df = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame()
    metadata = {
        "generated_at": display_time(),
        "script_version": SCRIPT_VERSION,
        "router_name": ROUTER_NAME,
        "output_dir": str(output_dir),
        "labels_path": str(args.labels_path),
        "prediction_manifest_path": str(args.prediction_manifest_path),
        "prediction_index_source": prediction_index_source,
        "config_path": str(args.config_path),
        "seed": int(args.seed),
        "layout_name": str(args.layout_name),
        "backend_style": "film_mean_patch_aux",
        "visual_input": "mean_patch_embedding",
        "condition_input": "revin_aux",
        "used_concat_aux": False,
        "trained_vit": False,
        "embedding_storage": "batch_runtime_only_not_saved",
        "pseudo_image_tensor_storage": "not_saved",
        "persistent_embedding_npy_written": False,
        "stream_shard_index": int(args.stream_shard_index),
        "stream_shard_count": int(args.stream_shard_count),
        "max_samples_per_split": args.max_samples_per_split,
        "train_sample_count": int(len(train_windows_df)),
        "test_sample_count": int(len(test_windows_df)),
        "epochs": int(args.epochs),
        "epoch_summaries": epoch_summaries,
        "embedding_dim": int(embedding_dim or 0),
        "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
        "period_candidates": period_candidate_values,
        "latency_mean": {
            "imageization_per_window_ms": float(latency_df["imageization_per_window_ms"].mean()) if not latency_df.empty else None,
            "encoder_forward_per_window_ms": float(latency_df["encoder_forward_per_window_ms"].mean()) if not latency_df.empty else None,
        },
        "input_exclusions": ["future_y_as_feature", "oracle_model_as_feature", "oracle_value_as_feature", "expert_error_as_feature"],
    }
    write_json(output_dir / "visual_router_metadata.json", metadata)
    summary = summarize_outputs(output_dir)
    prediction_index.close()
    write_status(output_dir, {"status": "completed", "phase": "done", "summary_rows": int(len(summary))})
    print(f"wrote Round2 fullscale streaming FiLM outputs to {output_dir}")
    if not summary.empty:
        print(summary.to_string(index=False))
    pred_path = output_dir / "visual_router_predictions.csv"
    if pred_path.exists():
        print(pd.read_csv(pred_path).head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
