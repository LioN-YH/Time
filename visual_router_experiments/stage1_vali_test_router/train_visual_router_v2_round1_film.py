#!/usr/bin/env python3
"""
文件功能：
    执行 Visual Router V2 Round 1 P2e FiLM / aux modulation pilot 消融。

实验边界：
    - 只读取 P2a sharded feature cache 中已有的 cls/mean_patch embedding 和 revin_aux；
    - RevIN aux 只通过 FiLM 的 gamma/beta 调制 visual hidden representation，
      不直接 concat 到 base visual input；
    - 不重新生成 P2a features，不保存 pseudo image tensor，不做 pilot_test；
    - 单任务模式只写独立 variant/seed 子目录，汇总模式单独生成统一 summary。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import (  # noqa: E402
    DEFAULT_ORACLE_LABELS,
    DEFAULT_PREDICTION_MANIFEST,
    load_oracle_subset,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import scaler_to_state  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_concat import (  # noqa: E402
    ensure_prediction_index,
    normalize_comparison_frame,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import TSF_STRATA_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    load_prediction_batch_from_index,
    make_visual_pooling_method_rows,
    read_ordered_sample_csv,
    resolve_device,
    selected_model_counts_with_variant,
    summarize_mean_std,
    summarize_rows_with_seed,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round1_film"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round1" / "p2e_film"
SCRIPT_VERSION = "visual_router_v2_round1_film_p2e_v1"
FILM_VARIANTS = ("film_cls_mean_concat_aux", "film_mean_patch_aux")
FEATURE_ARRAY_BY_FILM_VARIANT = {
    "film_cls_mean_concat_aux": ("cls_embedding", "mean_patch_embedding"),
    "film_mean_patch_aux": ("mean_patch_embedding",),
}
SAMPLE_SETS = ("pilot_train", "pilot_selection", "diagnostic_balanced")
EVAL_SAMPLE_SETS = ("pilot_selection", "diagnostic_balanced")


def display_time() -> str:
    """函数功能：生成写入日志、metadata 和 summary 的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于后台监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2e FiLM 单任务训练和汇总参数。"""
    parser = argparse.ArgumentParser(description="Train / aggregate Visual Router V2 Round 1 P2e FiLM variants.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--variant", choices=FILM_VARIANTS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--film-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--devices-requested", default="")
    parser.add_argument("--parallel-launcher-used", action="store_true")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke；正式运行必须省略。")
    parser.add_argument("--run-single", action="store_true", help="只运行一个 variant/seed，并写入独立 task 子目录。")
    parser.add_argument("--aggregate-only", action="store_true", help="只读取 task 子目录并生成统一汇总。")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    """函数功能：解析逗号分隔 seeds，并去重保序。"""
    seeds: List[int] = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value not in seeds:
                seeds.append(value)
    if not seeds:
        raise ValueError("--seeds 不能为空")
    return seeds


def set_seed(seed: int) -> None:
    """函数功能：固定 Python、NumPy 和 PyTorch 随机源。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_commit_hash() -> str:
    """函数功能：记录当前 repo commit hash；失败时返回 unknown。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def task_dir(output_dir: Path, variant: str, seed: int) -> Path:
    """函数功能：返回单个 variant/seed 的隔离输出目录。"""
    return Path(output_dir) / "tasks" / f"{variant}_seed{int(seed)}"


class FiLMRouter(nn.Module):
    """
    类功能：
        使用 RevIN aux 对 visual hidden representation 做 FiLM 调制的 router。

    输入：
        visual_features: 标准化后的 CLS/mean_patch 视觉特征；
        aux_features: 标准化后的 RevIN aux 特征。
    输出：
        五专家 logits；训练与评估时再 softmax 得到融合权重。
    """

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
        """函数功能：初始化 FiLM router，并让初始 gamma/beta 接近恒等调制。"""
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
        """函数功能：输出 FiLM 调制后的五专家 logits。"""
        hidden = self.visual_backbone(visual_features)
        gamma_beta = self.film_mlp(aux_features)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        hidden = hidden * (1.0 + gamma) + beta
        return self.head(hidden)


def load_film_features(
    *,
    feature_manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
    variant: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    函数功能：
        从 P2a sharded `.npz` feature cache 读取 FiLM 所需 visual base 和 revin_aux。

    关键约束：
        visual base 与 aux 分开返回；aux 不被 concat 到 visual input，只作为 FiLM
        condition 生成 gamma/beta。
    """
    if variant not in FEATURE_ARRAY_BY_FILM_VARIANT:
        raise ValueError(f"未知 FiLM variant={variant}")
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].copy()
    if rows.empty:
        raise ValueError(f"P2a feature manifest 中没有 sample_set={sample_set}")
    rows = rows.sort_values("start_order_index", kind="mergesort").reset_index(drop=True)
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    visual_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
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
            visual_arrays = [np.asarray(data[name], dtype=np.float32) for name in FEATURE_ARRAY_BY_FILM_VARIANT[variant]]
            aux_array = np.asarray(data["revin_aux"], dtype=np.float32)
        visual = visual_arrays[0] if len(visual_arrays) == 1 else np.concatenate(visual_arrays, axis=1).astype(np.float32, copy=False)
        take = min(int(visual.shape[0]), wanted_count - loaded_count)
        visual_parts.append(visual[:take])
        aux_parts.append(aux_array[:take])
        key_parts.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded_count += take
    if loaded_count != wanted_count:
        raise ValueError(f"feature shard 样本数不足：sample_set={sample_set} expected={wanted_count} actual={loaded_count}")
    order_index = np.concatenate(order_parts, axis=0)
    if not np.array_equal(order_index, sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)):
        raise ValueError(f"{sample_set}/{variant} feature order_index 与 P0 不一致")
    if key_parts != expected_keys:
        raise ValueError(f"{sample_set}/{variant} feature sample_key 与 P0 顺序不一致")
    visual_features = np.concatenate(visual_parts, axis=0).astype(np.float32, copy=False)
    aux_features = np.concatenate(aux_parts, axis=0).astype(np.float32, copy=False)
    expected_visual_dim = 1536 if variant == "film_cls_mean_concat_aux" else 768
    if visual_features.shape != (wanted_count, expected_visual_dim):
        raise ValueError(f"{sample_set}/{variant} visual feature shape 异常：{visual_features.shape}")
    if aux_features.shape[0] != wanted_count or aux_features.ndim != 2:
        raise ValueError(f"{sample_set}/{variant} aux feature shape 异常：{aux_features.shape}")
    if not np.isfinite(visual_features).all() or not np.isfinite(aux_features).all():
        raise ValueError(f"{sample_set}/{variant} feature 中存在 NaN/Inf")
    return visual_features, aux_features


def train_film_router(
    *,
    train_visual_scaled: np.ndarray,
    train_aux_scaled: np.ndarray,
    train_sample_keys: Sequence[str],
    prediction_index,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    film_hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    huber_beta: float,
    kl_tau: float,
    lambda_kl: float,
    metric: str,
) -> Tuple[FiLMRouter, Dict[str, object]]:
    """函数功能：用 fusion_huber_kl 目标训练一个 FiLM router。"""
    set_seed(seed)
    router = FiLMRouter(
        visual_dim=int(train_visual_scaled.shape[1]),
        aux_dim=int(train_aux_scaled.shape[1]),
        hidden_dim=int(hidden_dim),
        film_hidden_dim=int(film_hidden_dim),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    huber = nn.SmoothL1Loss(beta=float(huber_beta))
    keys = [str(key) for key in train_sample_keys]
    x_visual_all = torch.from_numpy(np.asarray(train_visual_scaled, dtype=np.float32))
    x_aux_all = torch.from_numpy(np.asarray(train_aux_scaled, dtype=np.float32))
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
            y_pred, y_true, expert_errors = load_prediction_batch_from_index(prediction_index, batch_keys, error_metric=metric)
            batch_visual = x_visual_all[idx].to(device=device)
            batch_aux = x_aux_all[idx].to(device=device)
            batch_pred = torch.from_numpy(y_pred).to(device=device)
            batch_true = torch.from_numpy(y_true).to(device=device)
            batch_q = torch.softmax(-torch.from_numpy(expert_errors) / float(kl_tau), dim=1).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(batch_visual, batch_aux)
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
        logits = router(x_visual_all.to(device=device), x_aux_all.to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    return router, {
        "initial_train_loss": float(loss_history[0]),
        "final_train_loss": float(loss_history[-1]),
        "final_huber_loss": float(huber_history[-1]),
        "final_kl_loss": float(kl_history[-1]),
        "train_weight_entropy": float(np.mean(entropy)),
        "train_normalized_weight_entropy": float(np.mean(entropy) / math.log(len(MODEL_COLUMNS))),
        "train_mean_max_weight": float(np.mean(weights.max(axis=1))),
    }


def predict_film_router(
    *,
    router: FiLMRouter,
    visual_scaler: StandardScaler,
    aux_scaler: StandardScaler,
    visual_features: np.ndarray,
    aux_features: np.ndarray,
    sample_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    variant: str,
    seed: int,
    sample_set: str,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：对一个 sample_set 输出逐样本 FiLM router 权重和 oracle 对齐字段。"""
    from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import align_labels_to_samples

    aligned_labels = align_labels_to_samples(sample_df, labels_df)
    visual_scaled = visual_scaler.transform(np.asarray(visual_features, dtype=np.float32)).astype(np.float32)
    aux_scaled = aux_scaler.transform(np.asarray(aux_features, dtype=np.float32)).astype(np.float32)
    router.eval()
    with torch.inference_mode():
        logits = router(torch.from_numpy(visual_scaled).to(device=device), torch.from_numpy(aux_scaled).to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    selected_idx = weights.argmax(axis=1)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    rows: List[Dict[str, object]] = []
    router_name = f"p2e_{variant}_seed{int(seed)}"
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


def run_single(args: argparse.Namespace) -> None:
    """函数功能：运行一个 FiLM variant/seed，并只写自己的隔离子目录。"""
    if args.variant is None or args.seed is None:
        raise ValueError("--run-single 必须同时提供 --variant 和 --seed")
    out_dir = task_dir(args.output_dir, args.variant, int(args.seed))
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "task_metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"单任务输出已存在；如需覆盖请传 --overwrite：{out_dir}")
    write_json(out_dir / "status.json", {"status": "started", "variant": args.variant, "seed": int(args.seed), "updated_at": display_time()})

    feature_manifest_path = Path(args.feature_dir) / "round1_feature_manifest.csv"
    train_df = read_ordered_sample_csv(args.sample_dir, "pilot_train", max_samples=args.max_samples_per_set)
    selection_df = read_ordered_sample_csv(args.sample_dir, "pilot_selection", max_samples=args.max_samples_per_set)
    diagnostic_df = read_ordered_sample_csv(args.sample_dir, "diagnostic_balanced", max_samples=args.max_samples_per_set)
    all_keys = (
        train_df["sample_key"].astype(str).tolist()
        + selection_df["sample_key"].astype(str).tolist()
        + diagnostic_df["sample_key"].astype(str).tolist()
    )
    log_stage(f"读取 oracle labels 子集：variant={args.variant} seed={args.seed}")
    label_df_all = load_oracle_subset(args.oracle_labels_path, all_keys, batch_rows=args.parquet_batch_rows)
    label_by_set = {
        "pilot_selection": label_df_all[label_df_all["sample_key"].isin(selection_df["sample_key"].astype(str))].copy(),
        "diagnostic_balanced": label_df_all[label_df_all["sample_key"].isin(diagnostic_df["sample_key"].astype(str))].copy(),
    }
    prediction_index = ensure_prediction_index(
        output_dir=args.output_dir,
        round0_dir=args.round0_dir,
        prediction_manifest_path=args.prediction_manifest_path,
        sample_keys=all_keys,
        chunk_read_rows=args.csv_chunksize,
    )
    device = resolve_device(args.device)
    try:
        log_stage(f"读取 P2a FiLM features：variant={args.variant}")
        train_visual, train_aux = load_film_features(
            feature_manifest_path=feature_manifest_path,
            sample_df=train_df,
            sample_set="pilot_train",
            variant=args.variant,
        )
        visual_scaler = StandardScaler()
        aux_scaler = StandardScaler()
        train_visual_scaled = visual_scaler.fit_transform(train_visual).astype(np.float32)
        train_aux_scaled = aux_scaler.fit_transform(train_aux).astype(np.float32)
        del train_visual, train_aux
        log_stage(f"训练 FiLM router：variant={args.variant} seed={args.seed} device={device}")
        router, train_meta = train_film_router(
            train_visual_scaled=train_visual_scaled,
            train_aux_scaled=train_aux_scaled,
            train_sample_keys=train_df["sample_key"].astype(str).tolist(),
            prediction_index=prediction_index,
            seed=int(args.seed),
            device=device,
            hidden_dim=int(args.hidden_dim),
            film_hidden_dim=int(args.film_hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            huber_beta=float(args.huber_beta),
            kl_tau=float(args.kl_tau),
            lambda_kl=float(args.lambda_kl),
            metric=str(args.metric),
        )
        checkpoint_path = out_dir / f"checkpoint_{args.variant}_seed{int(args.seed)}.pt"
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "variant": args.variant,
                "seed": int(args.seed),
                "router_state_dict": router.state_dict(),
                "visual_scaler_state": scaler_to_state(visual_scaler),
                "aux_scaler_state": scaler_to_state(aux_scaler),
                "model_columns": list(MODEL_COLUMNS),
                "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
                "hyperparameters": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "hidden_dim": int(args.hidden_dim),
                    "film_hidden_dim": int(args.film_hidden_dim),
                    "dropout": float(args.dropout),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "huber_beta": float(args.huber_beta),
                    "kl_tau": float(args.kl_tau),
                    "lambda_kl": float(args.lambda_kl),
                },
            },
            checkpoint_path,
        )
        method_rows: List[pd.DataFrame] = []
        for sample_set, sample_df in [("pilot_selection", selection_df), ("diagnostic_balanced", diagnostic_df)]:
            visual_features, aux_features = load_film_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=sample_df,
                sample_set=sample_set,
                variant=args.variant,
            )
            pred = predict_film_router(
                router=router,
                visual_scaler=visual_scaler,
                aux_scaler=aux_scaler,
                visual_features=visual_features,
                aux_features=aux_features,
                sample_df=sample_df,
                labels_df=label_by_set[sample_set],
                variant=args.variant,
                seed=int(args.seed),
                sample_set=sample_set,
                device=device,
            )
            pred = add_batch_fusion_metrics(pred, prediction_index=prediction_index, metric=str(args.metric), batch_size=int(args.eval_batch_size))
            pred.to_csv(out_dir / f"predictions_{args.variant}_seed{int(args.seed)}_{sample_set}.csv", index=False)
            method_rows.append(make_visual_pooling_method_rows(pred, sample_set=sample_set, variant=args.variant, seed=int(args.seed)))
        method_df = pd.concat(method_rows, ignore_index=True)
        seed_results = summarize_rows_with_seed(method_df)
        method_df.to_csv(out_dir / "method_rows.csv", index=False)
        seed_results.to_csv(out_dir / "seed_results.csv", index=False)
        write_json(
            out_dir / "task_metadata.json",
            {
                "status": "completed",
                "generated_at": display_time(),
                "script_version": SCRIPT_VERSION,
                "variant": args.variant,
                "seed": int(args.seed),
                "device": str(device),
                "output_dir": str(out_dir),
                "checkpoint_path": str(checkpoint_path),
                "train_metadata": train_meta,
                "constraints": {
                    "pilot_test_used_for_selection": False,
                    "pilot_test_evaluated": False,
                    "trained_new_model": True,
                    "rebuilt_p2a_feature": False,
                    "loaded_116m_prediction_manifest_to_memory": False,
                    "saved_pseudo_image_tensor": False,
                    "used_film": True,
                    "used_gating": False,
                    "used_attention": False,
                    "used_concat_aux": False,
                    "single_task_output_isolated": True,
                },
            },
        )
        write_json(out_dir / "status.json", {"status": "completed", "variant": args.variant, "seed": int(args.seed), "updated_at": display_time()})
        log_stage(f"单任务完成：{out_dir}")
    finally:
        prediction_index.close()


def build_film_stratified_summary(method_rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成 oracle_model、error_gap_quantile、dataset/TSF 等分层诊断表。"""
    frames: List[pd.DataFrame] = []
    for col in TSF_STRATA_COLUMNS:
        grouped = summarize_rows_with_seed(method_rows, group_cols=[col]).rename(columns={col: "stratum_value"})
        grouped.insert(4, "stratum_column", col)
        grouped.insert(5, "stratum_kind", "single_column")
        frames.append(grouped)
    tsf_cell = summarize_rows_with_seed(method_rows, group_cols=TSF_STRATA_COLUMNS)
    tsf_cell = tsf_cell.copy()
    tsf_cell.insert(4, "stratum_column", "tsf_cell")
    tsf_cell.insert(5, "stratum_kind", "tsf_cell")
    tsf_cell["stratum_value"] = tsf_cell[TSF_STRATA_COLUMNS].astype(str).agg("|".join, axis=1)
    frames.append(tsf_cell)
    return pd.concat(frames, ignore_index=True)


def normalize_round0_frame(path: Path) -> pd.DataFrame:
    """函数功能：把 Round0 Visual/TimeFuse/global/oracle 统一成 P2e comparison 口径。"""
    df = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    name_map = {
        "visual_router_raw_soft_fusion": ("Round0 original Visual", "raw_soft_fusion"),
        "visual_router_hard_top1": ("Round0 original Visual", "hard_top1"),
        "timefuse_raw_soft_fusion": ("Round0 TimeFuse", "raw_soft_fusion"),
        "timefuse_hard_top1": ("Round0 TimeFuse", "hard_top1"),
        "global_best_single": ("global_best_single", "single"),
        "oracle_top1": ("oracle_top1", "oracle"),
    }
    for row in df.itertuples(index=False):
        data = row._asdict()
        method = str(data["method"])
        if method not in name_map:
            continue
        variant, method_kind = name_map[method]
        rows.append(
            {
                "stage": "Round0",
                "sample_set": data["sample_set"],
                "variant": variant,
                "method": method,
                "method_kind": method_kind,
                "seed_count": 1,
                "sample_count": int(data["sample_count"]),
                "MAE_mean": float(data["MAE"]),
                "MAE_std": 0.0,
                "MSE_mean": float(data["MSE"]) if not pd.isna(data.get("MSE", np.nan)) else np.nan,
                "MSE_std": 0.0,
                "regret_to_oracle_mean": float(data["regret_to_oracle"]),
                "regret_to_oracle_std": 0.0,
                "oracle_label_accuracy_mean": float(data["oracle_label_accuracy"]),
                "oracle_label_accuracy_std": 0.0,
                "weight_entropy_mean": float(data["weight_entropy"]) if not pd.isna(data.get("weight_entropy", np.nan)) else np.nan,
                "weight_entropy_std": 0.0,
                "normalized_weight_entropy_mean": float(data["normalized_weight_entropy"]) if not pd.isna(data.get("normalized_weight_entropy", np.nan)) else np.nan,
                "normalized_weight_entropy_std": 0.0,
                "mean_max_weight_mean": float(data["mean_max_weight"]) if not pd.isna(data.get("mean_max_weight", np.nan)) else np.nan,
                "mean_max_weight_std": 0.0,
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def build_comparison(selection_summary: pd.DataFrame, diagnostic_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """函数功能：合并 P2e 和历史 baseline，生成用户要求的 comparison。"""
    frames = [
        normalize_comparison_frame(selection_summary, stage="P2e", source_path=args.output_dir / "round1_film_selection_comparison.csv"),
        normalize_comparison_frame(diagnostic_summary, stage="P2e", source_path=args.output_dir / "round1_film_diagnostic_summary.csv"),
    ]
    historical = [
        ("P2b", DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling" / "visual_pooling_selection_comparison.csv"),
        ("P2b", DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling" / "visual_pooling_diagnostic_summary.csv"),
        ("P2c", DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_aux_only" / "aux_only_selection_comparison.csv"),
        ("P2c", DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_aux_only" / "aux_only_diagnostic_summary.csv"),
        ("P2d", DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_concat" / "round1_concat_selection_comparison.csv"),
        ("P2d", DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_concat" / "round1_concat_diagnostic_summary.csv"),
    ]
    for stage, path in historical:
        if path.exists():
            frames.append(normalize_comparison_frame(pd.read_csv(path), stage=stage, source_path=path))
    for path in [Path(args.round0_dir) / "round0_selection_comparison.csv", Path(args.round0_dir) / "round0_diagnostic_balanced_summary.csv"]:
        if path.exists():
            frames.append(normalize_round0_frame(path))
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["sample_set", "method_kind", "MAE_mean", "stage", "variant"], kind="mergesort").reset_index(drop=True)


def choose_best_variant(selection_summary: pd.DataFrame) -> Dict[str, object]:
    """函数功能：按用户指定 tie-breaker 只从 pilot_selection 选择 P2e best FiLM variant。"""
    soft = selection_summary[selection_summary["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    if soft.empty:
        raise ValueError("selection summary 缺少 raw_soft_fusion 行")
    soft = soft.sort_values(
        ["MAE_mean", "MAE_std", "regret_to_oracle_mean", "MSE_mean", "weight_entropy_std", "mean_max_weight_std"],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = soft.iloc[0].to_dict()
    return {
        "best_variant": str(best["variant"]),
        "selection_basis": "pilot_selection raw-soft MAE_mean; tie-breakers MAE_std, regret_to_oracle_mean, MSE_mean, weight_entropy_std, mean_max_weight_std",
        "selected_from_sample_set": "pilot_selection",
        "diagnostic_balanced_used_for_selection": False,
        "pilot_test_used_for_selection": False,
        "best_row": {key: (float(value) if isinstance(value, (float, np.floating)) else int(value) if isinstance(value, (int, np.integer)) else value) for key, value in best.items()},
    }


def build_delta_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成用户指定的 FiLM 与历史 baseline delta summary。"""
    selection_soft = comparison[
        (comparison["sample_set"].astype(str) == "pilot_selection")
        & (comparison["method_kind"].astype(str) == "raw_soft_fusion")
    ].set_index("variant")
    pairs = [
        ("film_cls_mean_concat_aux", "visual_cls_mean_concat"),
        ("film_cls_mean_concat_aux", "cls_mean_concat_plus_aux"),
        ("film_mean_patch_aux", "visual_mean_patch_only"),
        ("film_mean_patch_aux", "mean_patch_plus_aux"),
        ("film_cls_mean_concat_aux", "film_mean_patch_aux"),
        ("film_cls_mean_concat_aux", "Round0 TimeFuse"),
        ("film_mean_patch_aux", "Round0 TimeFuse"),
    ]
    rows: List[Dict[str, object]] = []
    for left, right in pairs:
        if left not in selection_soft.index or right not in selection_soft.index:
            rows.append({"delta_name": f"{left} - {right}", "status": "missing_reference"})
            continue
        left_row = selection_soft.loc[left]
        right_row = selection_soft.loc[right]
        rows.append(
            {
                "delta_name": f"{left} - {right}",
                "sample_set": "pilot_selection",
                "method_kind": "raw_soft_fusion",
                "left_variant": left,
                "right_variant": right,
                "left_MAE_mean": float(left_row["MAE_mean"]),
                "right_MAE_mean": float(right_row["MAE_mean"]),
                "delta_MAE_mean": float(left_row["MAE_mean"] - right_row["MAE_mean"]),
                "left_MAE_std": float(left_row["MAE_std"]),
                "right_MAE_std": float(right_row["MAE_std"]),
                "delta_MAE_std": float(left_row["MAE_std"] - right_row["MAE_std"]),
                "left_MSE_mean": float(left_row["MSE_mean"]) if not pd.isna(left_row["MSE_mean"]) else np.nan,
                "right_MSE_mean": float(right_row["MSE_mean"]) if not pd.isna(right_row["MSE_mean"]) else np.nan,
                "delta_MSE_mean": float(left_row["MSE_mean"] - right_row["MSE_mean"]) if not pd.isna(left_row["MSE_mean"]) and not pd.isna(right_row["MSE_mean"]) else np.nan,
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _metric(comparison: pd.DataFrame, variant: str, col: str) -> float:
    """函数功能：读取 pilot_selection raw-soft 指定指标。"""
    row = comparison[
        (comparison["sample_set"].astype(str) == "pilot_selection")
        & (comparison["method_kind"].astype(str) == "raw_soft_fusion")
        & (comparison["variant"].astype(str) == variant)
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def write_summary_md(
    *,
    output_dir: Path,
    selection_summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    delta_summary: pd.DataFrame,
    best_variant: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 P2e summary，逐条回答用户关心的问题。"""
    cls_mae = _metric(comparison, "film_cls_mean_concat_aux", "MAE_mean")
    mean_mae = _metric(comparison, "film_mean_patch_aux", "MAE_mean")
    visual_cls = _metric(comparison, "visual_cls_mean_concat", "MAE_mean")
    concat_cls = _metric(comparison, "cls_mean_concat_plus_aux", "MAE_mean")
    visual_mean = _metric(comparison, "visual_mean_patch_only", "MAE_mean")
    concat_mean = _metric(comparison, "mean_patch_plus_aux", "MAE_mean")
    cls_std = _metric(comparison, "film_cls_mean_concat_aux", "MAE_std")
    visual_cls_std = _metric(comparison, "visual_cls_mean_concat", "MAE_std")
    cls_mse = _metric(comparison, "film_cls_mean_concat_aux", "MSE_mean")
    visual_cls_mse = _metric(comparison, "visual_cls_mean_concat", "MSE_mean")

    def better(left: float, right: float) -> str:
        return "是" if np.isfinite(left) and np.isfinite(right) and left < right else "否"

    strata = pd.read_csv(output_dir / "round1_film_stratified_summary.csv")
    oracle_strata = strata[
        (strata["sample_set"].astype(str) == "pilot_selection")
        & (strata["method"].astype(str).str.endswith("_raw_soft_fusion"))
        & (strata["stratum_column"].astype(str) == "oracle_model")
        & (strata["stratum_value"].astype(str).isin(["CrossFormer", "PatchTST"]))
    ].copy()

    lines = [
        "# Visual Router V2 Round 1 P2e FiLM / Aux Modulation Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 结论回答",
        "",
        f"1. FiLM 是否优于 `visual_cls_mean_concat`？{better(cls_mae, visual_cls)}。film_cls_mean_concat_aux selection raw-soft MAE={cls_mae:.6f} vs visual_cls_mean_concat={visual_cls:.6f}，delta={cls_mae - visual_cls:+.6f}。",
        f"2. FiLM 是否优于 `cls_mean_concat_plus_aux`？{better(cls_mae, concat_cls)}。film_cls_mean_concat_aux MAE={cls_mae:.6f} vs cls_mean_concat_plus_aux={concat_cls:.6f}，delta={cls_mae - concat_cls:+.6f}。",
        f"3. FiLM 是否优于 `visual_mean_patch_only`？{better(mean_mae, visual_mean)}。film_mean_patch_aux MAE={mean_mae:.6f} vs visual_mean_patch_only={visual_mean:.6f}，delta={mean_mae - visual_mean:+.6f}。",
        f"4. mean_patch 路线用 FiLM 是否比直接 concat aux 更好？{better(mean_mae, concat_mean)}。film_mean_patch_aux MAE={mean_mae:.6f} vs mean_patch_plus_aux={concat_mean:.6f}，delta={mean_mae - concat_mean:+.6f}。",
        f"5. FiLM 是否改善 seed stability？{better(cls_std, visual_cls_std)}。film_cls_mean_concat_aux MAE_std={cls_std:.6f} vs visual_cls_mean_concat={visual_cls_std:.6f}。",
        f"6. FiLM 是否改善 MSE tail？{better(cls_mse, visual_cls_mse)}。film_cls_mean_concat_aux MSE_mean={cls_mse:.6f} vs visual_cls_mean_concat={visual_cls_mse:.6f}。",
        "7. FiLM 是否改善 CrossFormer / PatchTST strata？见下方 oracle_model 分层表；判断以 `round1_film_stratified_summary.csv` 的三 seed 均值为准。",
        f"8. 是否值得进入下一步 frozen pilot_test eval extension？{'值得' if cls_mae < min(visual_cls, concat_cls) or mean_mae < min(visual_mean, concat_mean) else '暂不建议'}；本轮未使用也未评估 pilot_test。",
        "",
        "## Oracle Model Strata",
        "",
        frame_to_markdown(oracle_strata.sort_values(["stratum_value", "MAE"], kind="mergesort").head(20), float_digits=6),
        "",
        "## Pilot Selection Mean/Std",
        "",
        frame_to_markdown(selection_summary, float_digits=6),
        "",
        "## Diagnostic Balanced Mean/Std",
        "",
        frame_to_markdown(diagnostic_summary, float_digits=6),
        "",
        "## Delta Summary",
        "",
        frame_to_markdown(delta_summary, float_digits=6),
        "",
        "## Best FiLM Variant",
        "",
        f"- best_variant：`{best_variant['best_variant']}`",
        f"- selection_basis：{best_variant['selection_basis']}",
        "- 本轮只训练 pilot_train，只用 pilot_selection 选择，diagnostic_balanced 只诊断；pilot_test_evaluated=false。",
    ]
    (output_dir / "round1_film_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_light_summaries(output_dir: Path, summary_dir: Path) -> None:
    """函数功能：只复制轻量 summary/metadata/CSV 到 experiment_summaries。"""
    summary_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "round1_film_variant_seed_results.csv",
        "round1_film_selection_comparison.csv",
        "round1_film_diagnostic_summary.csv",
        "round1_film_selected_model_counts.csv",
        "round1_film_stratified_summary.csv",
        "round1_film_delta_summary.csv",
        "round1_film_best_variant.json",
        "round1_film_metadata.json",
        "round1_film_summary.md",
    ]
    for name in names:
        shutil.copy2(output_dir / name, summary_dir / name)


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：汇总所有 FiLM task 子目录，生成统一 P2e 结果。"""
    seeds = parse_seed_list(args.seeds)
    expected = [(variant, seed) for variant in FILM_VARIANTS for seed in seeds]
    method_frames: List[pd.DataFrame] = []
    seed_frames: List[pd.DataFrame] = []
    task_meta: List[Mapping[str, object]] = []
    missing: List[str] = []
    for variant, seed in expected:
        out_dir = task_dir(args.output_dir, variant, seed)
        for name in ["method_rows.csv", "seed_results.csv", "task_metadata.json"]:
            if not (out_dir / name).exists():
                missing.append(str(out_dir / name))
        if not missing:
            method_frames.append(pd.read_csv(out_dir / "method_rows.csv"))
            seed_frames.append(pd.read_csv(out_dir / "seed_results.csv"))
            task_meta.append(json.loads((out_dir / "task_metadata.json").read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError("FiLM task 输出不完整：" + "; ".join(missing[:20]))
    method_rows = pd.concat(method_frames, ignore_index=True)
    seed_results = pd.concat(seed_frames, ignore_index=True)
    selection_summary = summarize_mean_std(seed_results, sample_set="pilot_selection")
    diagnostic_summary = summarize_mean_std(seed_results, sample_set="diagnostic_balanced")
    selected_counts = selected_model_counts_with_variant(method_rows)
    stratified = build_film_stratified_summary(method_rows)
    best_variant = choose_best_variant(selection_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_results.to_csv(args.output_dir / "round1_film_variant_seed_results.csv", index=False)
    selection_summary.to_csv(args.output_dir / "round1_film_selection_comparison.csv", index=False)
    diagnostic_summary.to_csv(args.output_dir / "round1_film_diagnostic_summary.csv", index=False)
    selected_counts.to_csv(args.output_dir / "round1_film_selected_model_counts.csv", index=False)
    stratified.to_csv(args.output_dir / "round1_film_stratified_summary.csv", index=False)
    write_json(args.output_dir / "round1_film_best_variant.json", best_variant)
    comparison = build_comparison(selection_summary, diagnostic_summary, args)
    delta_summary = build_delta_summary(comparison)
    comparison.to_csv(args.output_dir / "round1_film_selection_comparison_with_baselines.csv", index=False)
    delta_summary.to_csv(args.output_dir / "round1_film_delta_summary.csv", index=False)

    devices_used = sorted({str(meta.get("device", "")) for meta in task_meta if str(meta.get("device", "")).strip()})
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "output_dir": str(args.output_dir),
        "summary_copy_dir": str(args.summary_copy_dir),
        "inputs": {
            "sample_dir": str(args.sample_dir),
            "round0_dir": str(args.round0_dir),
            "feature_dir": str(args.feature_dir),
            "oracle_labels_path": str(args.oracle_labels_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
        },
        "variants": list(FILM_VARIANTS),
        "feature_groups": {key: list(value) for key, value in FEATURE_ARRAY_BY_FILM_VARIANT.items()},
        "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
        "seeds": seeds,
        "constraints": {
            "pilot_test_used_for_selection": False,
            "pilot_test_evaluated": False,
            "trained_new_model": True,
            "rebuilt_p2a_feature": False,
            "loaded_116m_prediction_manifest_to_memory": False,
            "saved_pseudo_image_tensor": False,
            "used_film": True,
            "used_gating": False,
            "used_attention": False,
            "used_concat_aux": False,
            "parallel_launcher_used": bool(args.parallel_launcher_used),
            "parallel_backend": "process_per_variant_seed",
            "devices_requested": str(args.devices_requested),
            "devices_used": devices_used,
            "single_task_output_isolated": True,
            "scaler_fit_sample_set": "pilot_train",
            "best_variant_selection_sample_set": "pilot_selection",
            "diagnostic_balanced_used_for_selection": False,
        },
        "best_variant": best_variant,
        "task_metadata": task_meta,
    }
    write_json(args.output_dir / "round1_film_metadata.json", metadata)
    write_summary_md(
        output_dir=args.output_dir,
        selection_summary=selection_summary,
        diagnostic_summary=diagnostic_summary,
        comparison=comparison,
        delta_summary=delta_summary,
        best_variant=best_variant,
        metadata=metadata,
    )
    copy_light_summaries(args.output_dir, args.summary_copy_dir)
    write_json(args.output_dir / "status.json", {"status": "completed", "best_variant": best_variant, "updated_at": display_time()})
    log_stage(f"P2e FiLM aggregation outputs written to {args.output_dir}")


def run_serial(args: argparse.Namespace) -> None:
    """函数功能：不使用 launcher 时提供全量串行 fallback。"""
    seeds = parse_seed_list(args.seeds)
    for variant in FILM_VARIANTS:
        for seed in seeds:
            child_args = argparse.Namespace(**vars(args))
            child_args.variant = variant
            child_args.seed = seed
            child_args.run_single = True
            child_args.aggregate_only = False
            run_single(child_args)
    aggregate(args)


def main() -> None:
    """函数功能：根据 CLI 模式执行单任务、汇总或串行 fallback。"""
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
    elif args.run_single:
        run_single(args)
    else:
        run_serial(args)


if __name__ == "__main__":
    main()
