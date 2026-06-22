#!/usr/bin/env python3
"""
文件功能：
    在 Round2 35k small 样本上训练 panel-aware gating / residual FiLM router probe。

实验边界：
    - 只复用既有 panel pooling 35k feature cache，不重新跑 ViT；
    - 不保存 pseudo image tensor，不做 full-scale / 65k / P0；
    - selection 只用于选择 variant，test_small 只做 frozen screening；
    - baseline `film_mean_patch_aux` 保持 768 维 global mean_patch FiLM 口径。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import DEFAULT_ORACLE_LABELS, DEFAULT_PREDICTION_MANIFEST, load_oracle_subset  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import SQLitePredictionIndex, build_lightweight_prediction_index, scaler_to_state  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import (  # noqa: E402
    FiLMRouter,
    build_film_stratified_summary,
    git_commit_hash,
    predict_film_router,
    train_film_router,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round2_layout_film import read_round2_sample_set, sample_sets_from_args  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_panel_gating import (  # noqa: E402
    PANEL_GATING_SCHEMA_VERSION,
    PanelGatingConfig,
    build_panel_gating_model,
    panel_concat_to_stack,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_panel_pooling import PANEL_POOLING_SCHEMA_VERSION  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    load_prediction_batch_from_index,
    make_visual_pooling_method_rows,
    resolve_device,
    selected_model_counts_with_variant,
    summarize_mean_std,
    summarize_rows_with_seed,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_MANIFEST = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_small_samples" / "round2_small_sample_manifest.csv"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_pooling_35k_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_gating_35k_screening"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2"
PANEL_GATING_VARIANTS = ("film_mean_patch_aux", "film_panel_gated_mean_aux", "film_panel_lowrank_aux")
GATING_ONLY_VARIANTS = ("film_panel_gated_mean_aux", "film_panel_lowrank_aux")
SCRIPT_VERSION = "visual_router_v2_round2_panel_gating_35k_screening_v1"


def display_time() -> str:
    """函数功能：生成日志和 metadata 时间戳。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于后台监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seed_list(text: str) -> List[int]:
    """函数功能：解析 seeds。"""
    return [int(value) for value in parse_csv(text)]


def parse_args() -> argparse.Namespace:
    """函数功能：解析 panel gating 35k screening 参数。"""
    parser = argparse.ArgumentParser(description="Train Round2 35k panel gating/residual FiLM probe.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--artifact-prefix", default="panel_gating_35k")
    parser.add_argument("--variants", default=",".join(PANEL_GATING_VARIANTS))
    parser.add_argument("--variant", choices=PANEL_GATING_VARIANTS, default=None)
    parser.add_argument("--train-sample-set", default="round2_train_small")
    parser.add_argument("--selection-sample-set", default="round2_selection_small")
    parser.add_argument("--diagnostic-sample-set", default="round2_diagnostic_balanced_small")
    parser.add_argument("--test-sample-set", default="round2_test_small")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--film-hidden-dim", type=int, default=32)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--lowrank-dim", type=int, default=256)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke。")
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def eval_sample_sets_from_args(args: argparse.Namespace) -> Tuple[str, str, str]:
    """函数功能：返回 selection/diagnostic/test 三个评估集合。"""
    return (str(args.selection_sample_set), str(args.diagnostic_sample_set), str(args.test_sample_set))


def read_all_sample_frames(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    """函数功能：读取训练、选择、诊断、frozen test 四个 sample set。"""
    return {
        name: read_round2_sample_set(args.sample_manifest, name, max_samples=args.max_samples_per_set)
        for name in sample_sets_from_args(args)
    }


def prediction_index_path(output_dir: Path) -> Path:
    """函数功能：返回本 screening 共用 prediction subset SQLite 路径。"""
    return Path(output_dir) / "prediction_index_round2_panel_gating_subset.sqlite"


def ensure_prediction_index(args: argparse.Namespace, sample_keys: Sequence[str]) -> SQLitePredictionIndex:
    """函数功能：获取或构建当前 35k screening 所需 prediction SQLite index。"""
    index_path = prediction_index_path(args.output_dir)
    if index_path.exists():
        return SQLitePredictionIndex(index_path, args.prediction_manifest_path.parent)
    if args.run_single and not args.build_index_only:
        raise FileNotFoundError(f"prediction index 尚未构建：{index_path}")
    return build_lightweight_prediction_index(
        args.prediction_manifest_path,
        sample_keys=[str(key) for key in sample_keys],
        chunk_read_rows=int(args.csv_chunksize),
        index_db_path=index_path,
    )


def task_dir(output_dir: Path, variant: str, seed: int) -> Path:
    """函数功能：返回 variant/seed 隔离输出目录。"""
    return Path(output_dir) / "tasks" / f"{variant}_seed{int(seed)}"


def feature_manifest_path(args: argparse.Namespace) -> Path:
    """函数功能：返回 panel pooling 35k feature manifest 路径。"""
    return Path(args.feature_dir) / "round2_panel_pooling_35k_feature_manifest.csv"


def validate_feature_cache(args: argparse.Namespace) -> Dict[str, object]:
    """函数功能：确认 feature cache 覆盖四个 small sample set 且不含 pseudo image tensor。"""
    manifest_path = feature_manifest_path(args)
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 panel pooling feature manifest：{manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required_sets = list(sample_sets_from_args(args))
    coverage: Dict[str, int] = {}
    for sample_set in required_sets:
        rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)]
        if rows.empty:
            raise ValueError(f"feature cache 缺少 sample_set={sample_set}")
        coverage[sample_set] = int(rows["sample_count"].sum())
        if "saved_pseudo_image_tensor" in rows.columns and rows["saved_pseudo_image_tensor"].astype(str).str.lower().ne("false").any():
            raise ValueError(f"{sample_set} manifest 显示保存了 pseudo image tensor，不符合本实验边界")
        pooling = ",".join(rows.get("pooling_available", pd.Series(dtype=str)).astype(str).tolist())
        for feature_name in ["global_mean_patch", "panel_mean_concat", "revin_aux"]:
            if feature_name not in pooling and feature_name != "revin_aux":
                raise ValueError(f"{sample_set} pooling_available 缺少 {feature_name}")
    return {
        "feature_manifest_path": str(manifest_path),
        "covered_sample_sets": coverage,
        "saved_pseudo_image_tensor": False,
        "reran_vit": False,
    }


def load_panel_feature_triplet(
    *,
    manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        从 panel pooling 35k cache 读取 global_mean_patch、panel_mean_concat 和 revin_aux。

    输出：
        global_mean_patch `[B,768]`、panel_mean_concat `[B,2304]`、revin_aux `[B,6]`。
    """
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].sort_values("start_order_index", kind="mergesort")
    if rows.empty:
        raise ValueError(f"panel feature manifest 缺少 sample_set={sample_set}")
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    global_parts: List[np.ndarray] = []
    panel_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    loaded = 0
    for row in rows.itertuples(index=False):
        if loaded >= wanted_count:
            break
        shard_path = Path(str(row.shard_path))
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            global_array = np.asarray(data["global_mean_patch"], dtype=np.float32)
            panel_array = np.asarray(data["panel_mean_concat"], dtype=np.float32)
            aux_array = np.asarray(data["revin_aux"], dtype=np.float32)
        take = min(int(global_array.shape[0]), wanted_count - loaded)
        global_parts.append(global_array[:take])
        panel_parts.append(panel_array[:take])
        aux_parts.append(aux_array[:take])
        key_parts.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded += take
    if loaded != wanted_count or key_parts != expected_keys:
        raise ValueError(f"{sample_set} feature 数量或 sample_key 顺序不一致")
    if not np.array_equal(np.concatenate(order_parts, axis=0), sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)):
        raise ValueError(f"{sample_set} order_index 不一致")
    global_features = np.concatenate(global_parts, axis=0).astype(np.float32, copy=False)
    panel_features = np.concatenate(panel_parts, axis=0).astype(np.float32, copy=False)
    aux_features = np.concatenate(aux_parts, axis=0).astype(np.float32, copy=False)
    if global_features.shape != (wanted_count, 768):
        raise ValueError(f"{sample_set} global_mean_patch shape 异常：{global_features.shape}")
    if panel_features.shape != (wanted_count, 2304):
        raise ValueError(f"{sample_set} panel_mean_concat shape 异常：{panel_features.shape}")
    if aux_features.shape != (wanted_count, len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"{sample_set} aux feature shape 异常：{aux_features.shape}")
    if not np.isfinite(global_features).all() or not np.isfinite(panel_features).all() or not np.isfinite(aux_features).all():
        raise ValueError(f"{sample_set} feature 中存在 NaN/Inf")
    return global_features, panel_features, aux_features


class PanelGatingFiLMRouter(nn.Module):
    """
    类功能：
        将 panel-aware residual/gating 模块与 FiLMRouter 串接并端到端训练。

    输入：
        global_features: 标准化后的 `[B,768]` global mean patch。
        panel_features: 标准化后的 `[B,2304]` panel concat，会 reshape 为 `[B,3,768]`。
        aux_features: 标准化后的 RevIN aux。
    输出：
        五专家 logits。
    """

    def __init__(
        self,
        *,
        variant: str,
        aux_dim: int,
        hidden_dim: int,
        film_hidden_dim: int,
        gate_hidden_dim: int,
        lowrank_dim: int,
        init_alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        config = PanelGatingConfig(
            visual_dim=768,
            panel_count=3,
            gate_hidden_dim=int(gate_hidden_dim),
            lowrank_dim=int(lowrank_dim),
            init_alpha=float(init_alpha),
        )
        self.variant = str(variant)
        self.panel_module = build_panel_gating_model(str(variant), config)
        self.router = FiLMRouter(
            visual_dim=768,
            aux_dim=int(aux_dim),
            hidden_dim=int(hidden_dim),
            film_hidden_dim=int(film_hidden_dim),
            output_dim=len(MODEL_COLUMNS),
            dropout=float(dropout),
        )
        self.last_panel_output: Mapping[str, torch.Tensor] | None = None

    def forward(self, global_features: torch.Tensor, panel_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        """函数功能：先构造 768 维 panel-aware visual，再输出 router logits。"""
        panel_stack = panel_concat_to_stack(panel_features, panel_count=3)
        panel_output = self.panel_module(global_features, panel_stack)
        self.last_panel_output = panel_output
        return self.router(panel_output["visual"], aux_features)


def _weight_diagnostics(weights: np.ndarray) -> Dict[str, float]:
    """函数功能：计算训练集权重熵和最大权重诊断。"""
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1.0))).sum(axis=1)
    return {
        "train_weight_entropy": float(np.mean(entropy)),
        "train_normalized_weight_entropy": float(np.mean(entropy) / math.log(len(MODEL_COLUMNS))),
        "train_mean_max_weight": float(np.mean(weights.max(axis=1))),
    }


def train_panel_gating_router(
    *,
    model: PanelGatingFiLMRouter,
    train_global_scaled: np.ndarray,
    train_panel_scaled: np.ndarray,
    train_aux_scaled: np.ndarray,
    train_sample_keys: Sequence[str],
    prediction_index: SQLitePredictionIndex,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    huber_beta: float,
    kl_tau: float,
    lambda_kl: float,
    metric: str,
) -> Dict[str, object]:
    """函数功能：用 fusion_huber_kl 目标联合训练 panel module 和 FiLM router。"""
    from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import set_seed

    set_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    huber = nn.SmoothL1Loss(beta=float(huber_beta))
    keys = [str(key) for key in train_sample_keys]
    x_global_all = torch.from_numpy(np.asarray(train_global_scaled, dtype=np.float32))
    x_panel_all = torch.from_numpy(np.asarray(train_panel_scaled, dtype=np.float32))
    x_aux_all = torch.from_numpy(np.asarray(train_aux_scaled, dtype=np.float32))
    rng = np.random.default_rng(int(seed))
    loss_history: List[float] = []
    huber_history: List[float] = []
    kl_history: List[float] = []
    model.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(keys))
        batch_losses: List[float] = []
        batch_huber: List[float] = []
        batch_kl: List[float] = []
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch_keys = [keys[int(i)] for i in idx]
            y_pred, y_true, expert_errors = load_prediction_batch_from_index(prediction_index, batch_keys, error_metric=metric)
            batch_global = x_global_all[idx].to(device=device)
            batch_panel = x_panel_all[idx].to(device=device)
            batch_aux = x_aux_all[idx].to(device=device)
            batch_pred = torch.from_numpy(y_pred).to(device=device)
            batch_true = torch.from_numpy(y_true).to(device=device)
            batch_q = torch.softmax(-torch.from_numpy(expert_errors) / float(kl_tau), dim=1).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_global, batch_panel, batch_aux)
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
    model.eval()
    with torch.inference_mode():
        logits = model(
            x_global_all.to(device=device),
            x_panel_all.to(device=device),
            x_aux_all.to(device=device),
        )
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
        panel_output = model.last_panel_output or {}
        alpha = float(panel_output["alpha"].detach().cpu().item()) if "alpha" in panel_output else np.nan
    meta: Dict[str, object] = {
        "initial_train_loss": float(loss_history[0]),
        "final_train_loss": float(loss_history[-1]),
        "final_huber_loss": float(huber_history[-1]),
        "final_kl_loss": float(kl_history[-1]),
        "learned_alpha": alpha,
    }
    meta.update(_weight_diagnostics(weights))
    return meta


def predict_panel_gating_router(
    *,
    model: PanelGatingFiLMRouter,
    global_scaler: StandardScaler,
    panel_scaler: StandardScaler,
    aux_scaler: StandardScaler,
    global_features: np.ndarray,
    panel_features: np.ndarray,
    aux_features: np.ndarray,
    sample_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    variant: str,
    seed: int,
    sample_set: str,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：对一个 sample_set 输出 panel gating router 权重和 oracle 对齐字段。"""
    from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import predict_film_router

    class _Adapter(nn.Module):
        """类功能：让现有 predict_film_router 复用 panel gating 模型。"""

        def __init__(self, wrapped: PanelGatingFiLMRouter, panel_scaled: np.ndarray) -> None:
            super().__init__()
            self.wrapped = wrapped
            self.panel_scaled = torch.from_numpy(panel_scaled.astype(np.float32, copy=False))

        def eval(self):  # type: ignore[override]
            self.wrapped.eval()
            return self

        def forward(self, visual_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
            panel = self.panel_scaled.to(device=visual_features.device)
            return self.wrapped(visual_features, panel, aux_features)

    global_scaled = global_scaler.transform(np.asarray(global_features, dtype=np.float32)).astype(np.float32)
    panel_scaled = panel_scaler.transform(np.asarray(panel_features, dtype=np.float32)).astype(np.float32)
    # 现有 predict_film_router 会自行 transform visual/aux；这里用 identity scaler 避免复制整段对齐逻辑。
    identity_visual = StandardScaler()
    identity_visual.mean_ = np.zeros(global_scaled.shape[1], dtype=np.float64)
    identity_visual.scale_ = np.ones(global_scaled.shape[1], dtype=np.float64)
    identity_visual.var_ = np.ones(global_scaled.shape[1], dtype=np.float64)
    identity_visual.n_features_in_ = global_scaled.shape[1]
    identity_aux = aux_scaler
    adapter = _Adapter(model, panel_scaled)
    return predict_film_router(
        router=adapter,  # type: ignore[arg-type]
        visual_scaler=identity_visual,
        aux_scaler=identity_aux,
        visual_features=global_scaled,
        aux_features=aux_features,
        sample_df=sample_df,
        labels_df=labels_df,
        variant=variant,
        seed=seed,
        sample_set=sample_set,
        device=device,
    )


def run_build_index_only(args: argparse.Namespace) -> None:
    """函数功能：单进程预构建 small sample prediction SQLite index 并校验 feature cache。"""
    cache_report = validate_feature_cache(args)
    frames = read_all_sample_frames(args)
    keys: List[str] = []
    for name in sample_sets_from_args(args):
        keys.extend(frames[name]["sample_key"].astype(str).tolist())
    index = ensure_prediction_index(args, keys)
    index.close()
    write_json(
        args.output_dir / "prediction_index_status.json",
        {
            "status": "completed",
            "index_path": str(prediction_index_path(args.output_dir)),
            "sample_key_count": len(set(keys)),
            "feature_cache": cache_report,
            "updated_at": display_time(),
        },
    )


def run_single(args: argparse.Namespace) -> None:
    """函数功能：训练并评估一个 baseline 或 panel gating variant/seed。"""
    if args.variant is None or args.seed is None:
        raise ValueError("--run-single 必须同时提供 --variant 和 --seed")
    variant = str(args.variant)
    seed = int(args.seed)
    out_dir = task_dir(args.output_dir, variant, seed)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "task_metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"单任务输出已存在；如需覆盖请传 --overwrite：{out_dir}")
    write_json(out_dir / "status.json", {"status": "started", "variant": variant, "seed": seed, "updated_at": display_time()})

    validate_feature_cache(args)
    frames = read_all_sample_frames(args)
    train_set, _, _, _ = sample_sets_from_args(args)
    eval_sets = eval_sample_sets_from_args(args)
    all_keys = [key for name in sample_sets_from_args(args) for key in frames[name]["sample_key"].astype(str).tolist()]
    log_stage(f"读取 oracle labels：variant={variant} seed={seed}")
    labels_all = load_oracle_subset(args.oracle_labels_path, all_keys, batch_rows=args.parquet_batch_rows)
    labels_by_set = {name: labels_all[labels_all["sample_key"].isin(frames[name]["sample_key"].astype(str))].copy() for name in eval_sets}
    prediction_index = ensure_prediction_index(args, all_keys)
    device = resolve_device(args.device)
    manifest_path = feature_manifest_path(args)
    try:
        train_global, train_panel, train_aux = load_panel_feature_triplet(
            manifest_path=manifest_path,
            sample_df=frames[train_set],
            sample_set=train_set,
        )
        global_scaler = StandardScaler()
        panel_scaler = StandardScaler()
        aux_scaler = StandardScaler()
        train_global_scaled = global_scaler.fit_transform(train_global).astype(np.float32)
        train_aux_scaled = aux_scaler.fit_transform(train_aux).astype(np.float32)
        if variant == "film_mean_patch_aux":
            log_stage(f"训练 baseline FiLM router：seed={seed} device={device}")
            router, train_meta = train_film_router(
                train_visual_scaled=train_global_scaled,
                train_aux_scaled=train_aux_scaled,
                train_sample_keys=frames[train_set]["sample_key"].astype(str).tolist(),
                prediction_index=prediction_index,
                seed=seed,
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
            checkpoint_payload = {
                "router_state_dict": router.state_dict(),
                "panel_module_state_dict": None,
                "global_scaler_state": scaler_to_state(global_scaler),
                "panel_scaler_state": None,
                "aux_scaler_state": scaler_to_state(aux_scaler),
            }
        else:
            train_panel_scaled = panel_scaler.fit_transform(train_panel).astype(np.float32)
            log_stage(f"联合训练 panel gating router：variant={variant} seed={seed} device={device}")
            router = PanelGatingFiLMRouter(
                variant=variant,
                aux_dim=train_aux_scaled.shape[1],
                hidden_dim=int(args.hidden_dim),
                film_hidden_dim=int(args.film_hidden_dim),
                gate_hidden_dim=int(args.gate_hidden_dim),
                lowrank_dim=int(args.lowrank_dim),
                init_alpha=float(args.init_alpha),
                dropout=float(args.dropout),
            )
            train_meta = train_panel_gating_router(
                model=router,
                train_global_scaled=train_global_scaled,
                train_panel_scaled=train_panel_scaled,
                train_aux_scaled=train_aux_scaled,
                train_sample_keys=frames[train_set]["sample_key"].astype(str).tolist(),
                prediction_index=prediction_index,
                seed=seed,
                device=device,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                huber_beta=float(args.huber_beta),
                kl_tau=float(args.kl_tau),
                lambda_kl=float(args.lambda_kl),
                metric=str(args.metric),
            )
            checkpoint_payload = {
                "router_state_dict": router.router.state_dict(),
                "panel_module_state_dict": router.panel_module.state_dict(),
                "global_scaler_state": scaler_to_state(global_scaler),
                "panel_scaler_state": scaler_to_state(panel_scaler),
                "aux_scaler_state": scaler_to_state(aux_scaler),
            }
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "variant": variant,
                "seed": seed,
                "model_columns": list(MODEL_COLUMNS),
                "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
                "hyperparameters": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "hidden_dim": int(args.hidden_dim),
                    "film_hidden_dim": int(args.film_hidden_dim),
                    "gate_hidden_dim": int(args.gate_hidden_dim),
                    "lowrank_dim": int(args.lowrank_dim),
                    "init_alpha": float(args.init_alpha),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                    "huber_beta": float(args.huber_beta),
                    "kl_tau": float(args.kl_tau),
                    "lambda_kl": float(args.lambda_kl),
                },
                **checkpoint_payload,
            },
            out_dir / f"checkpoint_{variant}_seed{seed}.pt",
        )
        method_frames: List[pd.DataFrame] = []
        for sample_set in eval_sets:
            global_features, panel_features, aux_features = load_panel_feature_triplet(
                manifest_path=manifest_path,
                sample_df=frames[sample_set],
                sample_set=sample_set,
            )
            if variant == "film_mean_patch_aux":
                pred = predict_film_router(
                    router=router,  # type: ignore[arg-type]
                    visual_scaler=global_scaler,
                    aux_scaler=aux_scaler,
                    visual_features=global_features,
                    aux_features=aux_features,
                    sample_df=frames[sample_set],
                    labels_df=labels_by_set[sample_set],
                    variant=variant,
                    seed=seed,
                    sample_set=sample_set,
                    device=device,
                )
            else:
                pred = predict_panel_gating_router(
                    model=router,  # type: ignore[arg-type]
                    global_scaler=global_scaler,
                    panel_scaler=panel_scaler,
                    aux_scaler=aux_scaler,
                    global_features=global_features,
                    panel_features=panel_features,
                    aux_features=aux_features,
                    sample_df=frames[sample_set],
                    labels_df=labels_by_set[sample_set],
                    variant=variant,
                    seed=seed,
                    sample_set=sample_set,
                    device=device,
                )
            pred = add_batch_fusion_metrics(pred, prediction_index=prediction_index, metric=str(args.metric), batch_size=int(args.eval_batch_size))
            pred.to_csv(out_dir / f"predictions_{variant}_seed{seed}_{sample_set}.csv", index=False)
            method_frames.append(make_visual_pooling_method_rows(pred, sample_set=sample_set, variant=variant, seed=seed))
        method_rows = pd.concat(method_frames, ignore_index=True)
        seed_results = summarize_rows_with_seed(method_rows)
        method_rows.to_csv(out_dir / "method_rows.csv", index=False)
        seed_results.to_csv(out_dir / "seed_results.csv", index=False)
        write_json(
            out_dir / "task_metadata.json",
            {
                "status": "completed",
                "generated_at": display_time(),
                "script_version": SCRIPT_VERSION,
                "variant": variant,
                "seed": seed,
                "device": str(device),
                "train_metadata": train_meta,
                "constraints": {
                    "layout_fixed_to_spatial_panel_3view": True,
                    "condition_input": "revin_aux",
                    "used_film": True,
                    "used_concat_aux": False,
                    "reran_vit": False,
                    "saved_pseudo_image_tensor": False,
                    "full_scale_validation": False,
                    "expanded_65k_validation": False,
                    "p0_scale": False,
                    "test_used_for_training_or_selection": False,
                },
            },
        )
        write_json(out_dir / "status.json", {"status": "completed", "variant": variant, "seed": seed, "updated_at": display_time()})
        log_stage(f"单任务完成：{out_dir}")
    finally:
        prediction_index.close()


def raw_soft_summary(summary: pd.DataFrame, sample_set: str) -> pd.DataFrame:
    """函数功能：筛选指定 sample_set 的 raw-soft 汇总行。"""
    return summary[
        (summary["sample_set"].astype(str) == str(sample_set))
        & (summary["method"].astype(str).str.endswith("_raw_soft_fusion"))
    ].copy()


def build_key_strata_delta(stratified: pd.DataFrame, *, baseline: str = "film_mean_patch_aux") -> pd.DataFrame:
    """函数功能：生成 q5 / 重点 oracle_model / LOW_LOW_HIGH 相对 baseline 的 strata delta。"""
    raw = stratified[stratified["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    # stratified 原表保留 seed 维度；这里先压成 seed mean，避免同一 strata 多行
    # 在 delta 计算时被误当成单行。sample_count 对三个 seed 应一致，取均值即可。
    raw = (
        raw.groupby(["sample_set", "variant", "stratum_column", "stratum_value"], as_index=False)
        .agg(
            sample_count=("sample_count", "mean"),
            MAE=("MAE", "mean"),
            MSE=("MSE", "mean"),
            regret_to_oracle=("regret_to_oracle", "mean"),
            oracle_label_accuracy=("oracle_label_accuracy", "mean"),
            mean_max_weight=("mean_max_weight", "mean"),
        )
    )
    key_specs = [
        ("error_gap_quantile", "q5"),
        ("oracle_model", "PatchTST"),
        ("oracle_model", "CrossFormer"),
        ("oracle_model", "DLinear"),
        ("oracle_model", "ES"),
        ("group_name", "LOW_LOW_HIGH"),
    ]
    rows: List[Dict[str, object]] = []
    indexed = raw.set_index(["sample_set", "variant", "stratum_column", "stratum_value"], drop=False)
    for sample_set in sorted(raw["sample_set"].astype(str).unique()):
        for col, value in key_specs:
            key = (sample_set, baseline, col, value)
            if key not in indexed.index:
                continue
            base = indexed.loc[key]
            for variant in sorted(raw["variant"].astype(str).unique()):
                if variant == baseline:
                    continue
                other_key = (sample_set, variant, col, value)
                if other_key not in indexed.index:
                    continue
                row = indexed.loc[other_key]
                rows.append(
                    {
                        "sample_set": sample_set,
                        "stratum_column": col,
                        "stratum_value": value,
                        "variant": variant,
                        "sample_count": float(row["sample_count"]),
                        "MAE_mean": float(row["MAE"]),
                        "MAE_delta_vs_baseline": float(row["MAE"] - base["MAE"]),
                        "MSE_delta_vs_baseline": float(row["MSE"] - base["MSE"]),
                        "regret_delta_vs_baseline": float(row["regret_to_oracle"] - base["regret_to_oracle"]),
                        "oracle_label_accuracy_mean": float(row["oracle_label_accuracy"]),
                        "mean_max_weight_mean": float(row["mean_max_weight"]),
                    }
                )
    return pd.DataFrame(rows)


def decide_recommendation(selection_summary: pd.DataFrame, test_summary: pd.DataFrame, key_delta: pd.DataFrame) -> Tuple[str, List[str]]:
    """函数功能：按用户给定升级标准生成 65k/side/drop 判断。"""
    selection = raw_soft_summary(selection_summary, "round2_selection_small").set_index("variant")
    test = raw_soft_summary(test_summary, "round2_test_small").set_index("variant")
    reasons: List[str] = []
    if "film_mean_patch_aux" not in selection.index:
        return "Drop / defer panel-aware path", ["缺少 baseline selection raw-soft 行"]
    base = selection.loc["film_mean_patch_aux"]
    promoted: List[str] = []
    side: List[str] = []
    dropped: List[str] = []
    for variant in [v for v in selection.index.astype(str).tolist() if v != "film_mean_patch_aux"]:
        row = selection.loc[variant]
        improves_mae = float(row["MAE_mean"]) < float(base["MAE_mean"])
        no_mse_regret = float(row["MSE_mean"]) <= float(base["MSE_mean"]) and float(row["regret_to_oracle_mean"]) <= float(base["regret_to_oracle_mean"])
        std_ok = float(row["MAE_std"]) <= max(float(base["MAE_std"]) * 1.5, float(base["MAE_std"]) + 1.0e-6)
        selected_test_ok = True
        if variant in test.index and "film_mean_patch_aux" in test.index:
            selected_test_ok = float(test.loc[variant]["MAE_mean"]) <= float(test.loc["film_mean_patch_aux"]["MAE_mean"]) + 0.001
        selection_key = key_delta[(key_delta["sample_set"] == "round2_selection_small") & (key_delta["variant"] == variant)]
        q5_ok = not ((selection_key["stratum_column"] == "error_gap_quantile") & (selection_key["stratum_value"] == "q5") & (selection_key["MAE_delta_vs_baseline"] > 0)).any()
        patch_cross_ok = not ((selection_key["stratum_column"] == "oracle_model") & (selection_key["stratum_value"].isin(["PatchTST", "CrossFormer"])) & (selection_key["MAE_delta_vs_baseline"] > 0)).any()
        if improves_mae and no_mse_regret and std_ok and q5_ok and patch_cross_ok and selected_test_ok:
            promoted.append(variant)
        elif not improves_mae or not no_mse_regret or not q5_ok or not patch_cross_ok:
            dropped.append(variant)
        else:
            side.append(variant)
        reasons.append(
            f"{variant}: selection MAE delta={float(row['MAE_mean'] - base['MAE_mean']):+.6f}, "
            f"MSE delta={float(row['MSE_mean'] - base['MSE_mean']):+.6f}, "
            f"regret delta={float(row['regret_to_oracle_mean'] - base['regret_to_oracle_mean']):+.6f}, "
            f"q5_ok={q5_ok}, PatchTST/CrossFormer_ok={patch_cross_ok}, test_ok={selected_test_ok}"
        )
    if promoted:
        return "Promote to 65k expanded validation: " + ",".join(promoted), reasons
    if side:
        return "Keep as side branch", reasons
    return "Drop / defer panel-aware path", reasons


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：汇总 panel gating variant × seed 输出并写 summary。"""
    variants = [str(args.variant)] if args.variant else parse_csv(args.variants)
    seeds = parse_seed_list(args.seeds)
    _, selection_set, diagnostic_set, test_set = sample_sets_from_args(args)
    method_frames: List[pd.DataFrame] = []
    seed_frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for variant in variants:
        for seed in seeds:
            out_dir = task_dir(args.output_dir, variant, seed)
            for name in ["method_rows.csv", "seed_results.csv"]:
                if not (out_dir / name).exists():
                    missing.append(str(out_dir / name))
            if not missing:
                method_frames.append(pd.read_csv(out_dir / "method_rows.csv"))
                seed_frames.append(pd.read_csv(out_dir / "seed_results.csv"))
    if missing:
        raise FileNotFoundError("panel gating task 输出不完整：" + "; ".join(missing[:20]))
    method_rows = pd.concat(method_frames, ignore_index=True)
    seed_results = pd.concat(seed_frames, ignore_index=True)
    selection_summary = summarize_mean_std(seed_results, sample_set=selection_set)
    diagnostic_summary = summarize_mean_std(seed_results, sample_set=diagnostic_set)
    test_summary = summarize_mean_std(seed_results, sample_set=test_set)
    selected_counts = selected_model_counts_with_variant(method_rows)
    stratified = build_film_stratified_summary(method_rows)
    key_delta = build_key_strata_delta(stratified)
    recommendation, recommendation_reasons = decide_recommendation(selection_summary, test_summary, key_delta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.artifact_prefix)
    paths = {
        "variant_seed_results": output_dir / f"{prefix}_variant_seed_results.csv",
        "selection_summary": output_dir / f"{prefix}_selection_summary.csv",
        "diagnostic_summary": output_dir / f"{prefix}_diagnostic_summary.csv",
        "test_summary": output_dir / f"{prefix}_test_small_summary.csv",
        "selected_counts": output_dir / f"{prefix}_selected_model_counts.csv",
        "stratified": output_dir / f"{prefix}_stratified_summary.csv",
        "key_delta": output_dir / f"{prefix}_key_strata_delta.csv",
        "metadata": output_dir / f"{prefix}_metadata.json",
        "summary": output_dir / f"{prefix}_screening_summary.md",
    }
    seed_results.to_csv(paths["variant_seed_results"], index=False)
    selection_summary.to_csv(paths["selection_summary"], index=False)
    diagnostic_summary.to_csv(paths["diagnostic_summary"], index=False)
    test_summary.to_csv(paths["test_summary"], index=False)
    selected_counts.to_csv(paths["selected_counts"], index=False)
    stratified.to_csv(paths["stratified"], index=False)
    key_delta.to_csv(paths["key_delta"], index=False)
    selection_soft = raw_soft_summary(selection_summary, selection_set).sort_values(["MAE_mean", "MAE_std", "MSE_mean"], kind="mergesort")
    best = selection_soft.head(1).to_dict("records")
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "feature_schema_version": PANEL_POOLING_SCHEMA_VERSION,
        "panel_gating_schema_version": PANEL_GATING_SCHEMA_VERSION,
        "commit_hash": git_commit_hash(),
        "variants": variants,
        "seeds": seeds,
        "selection_sample_set": selection_set,
        "diagnostic_sample_set": diagnostic_set,
        "test_sample_set": test_set,
        "best_selection_variant": best[0] if best else None,
        "recommendation": recommendation,
        "recommendation_reasons": recommendation_reasons,
        "constraints": {
            "reran_vit": False,
            "saved_pseudo_image_tensor": False,
            "full_scale_validation": False,
            "expanded_65k_validation": False,
            "p0_scale": False,
            "test_small_used_for_training_tuning_selection_seed_or_epoch": False,
            "oracle_label_accuracy_is_primary_metric": False,
        },
    }
    write_json(paths["metadata"], metadata)
    selection_key = key_delta[key_delta["sample_set"].astype(str) == selection_set].copy()
    summary = "\n".join(
        [
            "# Visual Router V2 Round2 panel gating 35k screening",
            "",
            f"生成时间：{metadata['generated_at']}",
            "",
            "## 结论",
            "",
            f"- selection best：`{metadata['best_selection_variant']['variant'] if metadata['best_selection_variant'] else 'N/A'}`。",
            f"- 升级判断：{recommendation}。",
            "- 本结果只覆盖 Round2 35k small screening，不影响并行 full-scale 主线；full-scale 仍可继续使用 `spatial_panel_3view + film_mean_patch_aux`。",
            "- 本轮不重新跑 ViT，不保存 pseudo image tensor，不做 65k/P0/full-scale；`round2_test_small` 未用于训练、调参、选择 variant、选择 seed 或选择 epoch。",
            "",
            "## Selection Raw-Soft / Hard",
            "",
            frame_to_markdown(selection_summary, float_digits=6),
            "",
            "## Diagnostic Balanced",
            "",
            frame_to_markdown(diagnostic_summary, float_digits=6),
            "",
            "## Frozen Test Small",
            "",
            frame_to_markdown(test_summary, float_digits=6),
            "",
            "## Selection Key Strata Delta",
            "",
            "负数表示好于 baseline `film_mean_patch_aux`。",
            "",
            frame_to_markdown(selection_key, float_digits=6),
            "",
            "## Selected Model Ratio",
            "",
            frame_to_markdown(selected_counts, float_digits=6),
            "",
            "## 判断依据",
            "",
            "\n".join(f"- {line}" for line in recommendation_reasons),
            "",
            "## 产物",
            "",
            "\n".join(f"- `{path.name}`" for path in paths.values()),
            "",
        ]
    )
    paths["summary"].write_text(summary, encoding="utf-8")
    summary_dir = Path(args.summary_copy_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if Path(path).suffix in {".csv", ".json", ".md"}:
            shutil.copy2(path, summary_dir / Path(path).name)


def main() -> None:
    """函数功能：分发 build-index、single task 或 aggregate。"""
    args = parse_args()
    if args.build_index_only:
        run_build_index_only(args)
    elif args.run_single:
        run_single(args)
    elif args.aggregate_only:
        aggregate(args)
    else:
        raise ValueError("请显式指定 --build-index-only、--run-single 或 --aggregate-only")


if __name__ == "__main__":
    main()
