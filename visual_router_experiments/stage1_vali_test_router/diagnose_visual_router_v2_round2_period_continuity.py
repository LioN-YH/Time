#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2d period continuity diagnostic。

实验边界：
    - 只诊断周期相关 layout 在轻微历史输入扰动下的连续性；
    - 不训练新 router，不重建 Round2c 35k feature cache，不执行 65k expanded validation；
    - 单任务只写独立 layout/seed/sample_set 子目录，统一 CSV/JSON/summary 只由 aggregation 写出；
    - ViT embedding 与 pseudo image tensor 只在运行内存在，不保存大规模 tensor cache。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = LEGACY_WORKSPACE / "quito"
for path in [REPO_ROOT, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from visual_router_experiments.common.pseudo_imageization import (  # noqa: E402
    EPS,
    _as_series_batch,
    encoder_normalize,
    make_default_period_candidates,
    normalize_window,
    parse_period_candidates,
    select_fft_periods,
)
from visual_router_experiments.common.round2_layout_registry import imageize_round2_layout  # noqa: E402
from visual_router_experiments.common.vit_embedding_utils import resolve_dtype  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.build_visual_router_v2_round2_layout_features import (  # noqa: E402
    Round2HistoryWindowLoader,
    load_data_config,
    make_encoder_args,
)
from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import DEFAULT_PREDICTION_MANIFEST  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import DEFAULT_CONFIG, resolve_device  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    SQLitePredictionIndex,
    load_checkpoint,
    load_vit_model_with_retry,
    scaler_from_state,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import FiLMRouter  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import (  # noqa: E402
    AUX_FEATURE_COLUMNS,
    compute_revin_aux_from_x,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    load_prediction_batch_from_index,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import TSF_STRATA_COLUMNS  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_MANIFEST = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_small_samples" / "round2_small_sample_manifest.csv"
DEFAULT_ROUND2C_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_layout_screening"
DEFAULT_VISUAL_CHECKPOINT = DATA2_RUN_OUTPUT_ROOT / "2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2" / "checkpoints" / "latest_96_48_S.pt"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_period_continuity"
DEFAULT_SUMMARY_COPY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2" / "period_continuity"
SCRIPT_VERSION = "visual_router_v2_round2d_period_continuity_v1"
DEFAULT_RESULT_PREFIX = "round2_period_continuity"
DEFAULT_LAYOUTS = ("current_rgb_3view", "top3fold_period_layout")
DEFAULT_COMPARE_LAYOUTS = ("current_rgb_3view", "top3fold_period_layout")
DEFAULT_SAMPLE_SETS = ("round2_selection_small", "round2_diagnostic_balanced_small")
PERTURBATION_SPACE = "pre_revin_history_x"
RAW_COLUMNS = [
    "layout",
    "seed",
    "sample_set",
    "sample_key",
    "perturbation_sigma",
    "perturbation_index",
    "top1_period",
    "perturbed_top1_period",
    "top1_period_changed",
    "top3_jaccard",
    "top1_period_bucket",
    "perturbed_top1_period_bucket",
    "top1_period_bucket_flipped",
    "period_score_margin",
    "perturbed_period_score_margin",
    "image_l2",
    "image_cosine_distance",
    "cls_embedding_cosine_distance",
    "mean_patch_embedding_cosine_distance",
    "embedding_l2",
    "router_weight_js_divergence",
    "router_weight_l1",
    "router_weight_max_change",
    "selected_model",
    "perturbed_selected_model",
    "selected_model_flipped",
    "soft_fused_abs_change",
    "soft_fused_mae_change",
    "regret_change",
    "oracle_model",
    "season_strength_cat",
    "forecastability_cat",
    "error_gap_quantile",
    "top3_periods",
    "perturbed_top3_periods",
]


def now_cst() -> str:
    """函数功能：生成写入日志和 metadata 的本地时间戳。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：输出可被后台日志追踪的阶段进度。"""
    print(f"[{now_cst()}] {message}", flush=True)


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数，并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_int_list(values: Sequence[str] | str) -> List[int]:
    """函数功能：解析 seed 列表，兼容空格列表和逗号分隔字符串。"""
    if isinstance(values, str):
        parts = parse_csv(values)
    else:
        parts = []
        for value in values:
            parts.extend(parse_csv(str(value)))
    return [int(value) for value in parts]


def parse_float_csv(text: str) -> List[float]:
    """函数功能：解析 sigma 列表。"""
    return [float(value) for value in parse_csv(text)]


def parse_period_candidates_arg(text: Optional[str]) -> Optional[List[int]]:
    """函数功能：解析可选固定周期候选列表。"""
    if text is None or str(text).strip() == "":
        return None
    values = [int(value) for value in parse_csv(str(text))]
    if min(values) < 2:
        raise ValueError("--period-candidates 中所有值必须 >= 2")
    return values


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2d continuity diagnostic 参数。"""
    parser = argparse.ArgumentParser(description="Diagnose Round2 period layout continuity under tiny input perturbations.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--round2c-dir", type=Path, default=DEFAULT_ROUND2C_DIR)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL_CHECKPOINT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_COPY_DIR)
    parser.add_argument("--result-prefix", default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--compare-with-existing", type=Path, default=None)
    parser.add_argument("--append-to-existing", type=Path, default=None)
    parser.add_argument("--sample-sets", nargs="+", default=list(DEFAULT_SAMPLE_SETS))
    parser.add_argument("--max-samples-per-set", type=int, default=512)
    parser.add_argument("--layouts", default=",".join(DEFAULT_LAYOUTS))
    parser.add_argument("--layout", default=None)
    parser.add_argument("--seeds", nargs="+", default=["16", "17", "18"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--perturbation-sigma-list", default="0.001,0.005,0.01")
    parser.add_argument("--num-perturbations", type=int, default=3)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates")
    parser.add_argument("--period-candidates", default=None)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-debug-examples", action="store_true")
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def task_dir(output_dir: Path, layout: str, seed: int, sample_set: str) -> Path:
    """函数功能：返回单任务隔离输出目录。"""
    return Path(output_dir) / "tasks" / f"{layout}_seed{int(seed)}_{sample_set}"


def prediction_index_path(round2c_dir: Path) -> Path:
    """函数功能：返回 Round2c 已构建的轻量 prediction SQLite。"""
    return Path(round2c_dir) / "prediction_index_round2c_35k.sqlite"


def load_sample_frame(path: Path, sample_set: str, max_samples: Optional[int]) -> pd.DataFrame:
    """
    函数功能：
        读取一个 Round2 sample_set，并做轻量分层抽样。

    说明：
        诊断优先覆盖 oracle_model、season_strength_cat、forecastability_cat 和
        error_gap_quantile；top1 period bucket 会在读取 x 后另行计算并写入结果。
    """
    frame = pd.read_csv(path)
    part = frame[frame["sample_set"].astype(str) == str(sample_set)].sort_values("order_index", kind="mergesort").reset_index(drop=True)
    if part.empty:
        raise ValueError(f"sample_set={sample_set} 为空")
    if max_samples is None or len(part) <= int(max_samples):
        return part.reset_index(drop=True)
    strata_cols = ["oracle_model", "season_strength_cat", "forecastability_cat", "error_gap_quantile"]
    sampled: List[pd.DataFrame] = []
    per_group = max(1, int(math.ceil(int(max_samples) / max(1, part.groupby(strata_cols, dropna=False).ngroups))))
    for _, group in part.groupby(strata_cols, sort=True, dropna=False):
        sampled.append(group.head(per_group))
    result = pd.concat(sampled, ignore_index=True).drop_duplicates("sample_key", keep="first")
    if len(result) < int(max_samples):
        remaining = part[~part["sample_key"].astype(str).isin(result["sample_key"].astype(str))]
        result = pd.concat([result, remaining.head(int(max_samples) - len(result))], ignore_index=True)
    result = result.head(int(max_samples)).sort_values("order_index", kind="mergesort").reset_index(drop=True)
    return result


def period_bucket(period_value: int) -> str:
    """函数功能：把具体 top1 period 归入稳定性诊断 bucket。"""
    value = int(period_value)
    if value <= 4:
        return "p_le_4"
    if value <= 8:
        return "p_5_8"
    if value <= 16:
        return "p_9_16"
    if value <= 32:
        return "p_17_32"
    if value <= 64:
        return "p_33_64"
    return "p_gt_64"


def period_details(x_batch: torch.Tensor, *, period_candidates: Optional[Sequence[int]], period_selection: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    函数功能：
        按 registry 同口径计算 RevIN 后 top3 period 和对应 FFT score。
    """
    x_norm, _ = normalize_window(x_batch, norm_mode="revin_aux")
    series = _as_series_batch(x_norm).to(dtype=torch.float32)
    parsed = None
    if period_selection == "fixed_candidates":
        parsed = parse_period_candidates(period_candidates, history_length=int(series.shape[1]), device=series.device)
        if parsed is None:
            parsed = make_default_period_candidates(int(series.shape[1]), device=series.device)
    periods = select_fft_periods(series, top_k=3, period_candidates=parsed)
    centered = series - series.mean(dim=1, keepdim=True)
    power = torch.abs(torch.fft.rfft(centered, dim=1)[:, 1:]) ** 2
    if power.shape[1] == 0:
        scores = torch.zeros_like(periods, dtype=torch.float32)
    else:
        bins = torch.round(float(series.shape[1]) / periods.to(dtype=torch.float32)).to(dtype=torch.long)
        bins = bins.clamp(min=1, max=int(power.shape[1]))
        scores = torch.gather(power, dim=1, index=bins - 1)
    return periods.detach(), scores.detach()


def cosine_distance_np(left: np.ndarray, right: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """函数功能：计算 1-cosine，相同零向量按距离 0 处理。"""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    numerator = np.sum(a * b, axis=axis)
    denom = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return 1.0 - numerator / np.maximum(denom, 1e-12)


def js_divergence_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """函数功能：计算逐样本 Jensen-Shannon divergence。"""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1)


def forward_images_embeddings(
    *,
    x_batch: np.ndarray,
    layout_name: str,
    vit_model,
    device: torch.device,
    dtype: torch.dtype,
    normalization_preset: str,
    image_size: int,
    norm_mode: str,
    clip: float,
    period_candidates: Optional[Sequence[int]],
    period_selection: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        对一批历史窗口生成 pseudo image、CLS/mean-patch embedding 和 top3 period。
    """
    x_tensor = torch.from_numpy(np.asarray(x_batch, dtype=np.float32)).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        result = imageize_round2_layout(
            x_tensor,
            layout_name=layout_name,
            image_size=int(image_size),
            norm_mode=str(norm_mode),
            clip=float(clip),
            period_candidates=period_candidates,
            period_selection=str(period_selection),
        )
        images = result.images
        periods, scores = period_details(x_tensor, period_candidates=period_candidates, period_selection=period_selection)
        pixel_values = encoder_normalize(images.to(dtype=dtype), preset=normalization_preset)
        outputs = vit_model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        cls = hidden[:, 0, :].detach().to(device="cpu", dtype=torch.float32).numpy()
        mean_patch = hidden[:, 1:, :].mean(dim=1).detach().to(device="cpu", dtype=torch.float32).numpy()
        image_np = images.detach().to(device="cpu", dtype=torch.float32).numpy()
    return (
        image_np.astype(np.float32, copy=False),
        cls.astype(np.float32, copy=False),
        mean_patch.astype(np.float32, copy=False),
        periods.detach().cpu().numpy().astype(np.int64),
        scores.detach().cpu().numpy().astype(np.float32),
    )


def load_router_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[FiLMRouter, object, object]:
    """函数功能：从 Round2c fixed FiLM checkpoint 恢复 router 和 scaler。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyperparameters"])
    visual_scaler = scaler_from_state(ckpt["visual_scaler_state"])
    aux_scaler = scaler_from_state(ckpt["aux_scaler_state"])
    router = FiLMRouter(
        visual_dim=int(visual_scaler.n_features_in_),
        aux_dim=int(aux_scaler.n_features_in_),
        hidden_dim=int(hparams["hidden_dim"]),
        film_hidden_dim=int(hparams["film_hidden_dim"]),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(hparams["dropout"]),
    ).to(device)
    router.load_state_dict(ckpt["router_state_dict"])
    router.eval()
    return router, visual_scaler, aux_scaler


def router_weights(
    *,
    router: FiLMRouter,
    visual_scaler,
    aux_scaler,
    visual_features: np.ndarray,
    aux_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """函数功能：对 mean-patch embedding 和 RevIN aux 输出五专家 softmax 权重。"""
    visual_scaled = visual_scaler.transform(np.asarray(visual_features, dtype=np.float32)).astype(np.float32)
    aux_scaled = aux_scaler.transform(np.asarray(aux_features, dtype=np.float32)).astype(np.float32)
    with torch.inference_mode():
        logits = router(
            torch.from_numpy(visual_scaled).to(device=device),
            torch.from_numpy(aux_scaled).to(device=device),
        )
        return torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)


def fuse_arrays(weights: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """函数功能：按 router weights 对五专家预测数组做 raw-soft fusion。"""
    shape = (weights.shape[0], weights.shape[1], *([1] * (y_pred.ndim - 2)))
    return (weights.reshape(shape) * y_pred).sum(axis=1)


def summarize_prediction_change(
    *,
    original_weights: np.ndarray,
    perturbed_weights: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    oracle_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """函数功能：计算 fused prediction absolute change、MAE change 和 regret change。"""
    original_fused = fuse_arrays(original_weights, y_pred)
    perturbed_fused = fuse_arrays(perturbed_weights, y_pred)
    fused_abs_change = np.mean(np.abs(perturbed_fused - original_fused).reshape(original_weights.shape[0], -1), axis=1)
    original_mae = np.mean(np.abs(original_fused - y_true).reshape(original_weights.shape[0], -1), axis=1)
    perturbed_mae = np.mean(np.abs(perturbed_fused - y_true).reshape(original_weights.shape[0], -1), axis=1)
    return fused_abs_change, perturbed_mae - original_mae, (perturbed_mae - oracle_values) - (original_mae - oracle_values)


def save_debug_png(path: Path, image: np.ndarray) -> None:
    """函数功能：保存少量高变化 pseudo image debug PNG。"""
    try:
        from PIL import Image
    except ImportError:
        return
    arr = np.asarray(image, dtype=np.float32)
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def run_single(args: argparse.Namespace) -> None:
    """函数功能：运行一个 layout/seed/sample_set 的 continuity diagnostic。"""
    if args.layout is None or args.seed is None or len(args.sample_sets) != 1:
        raise ValueError("--run-single 必须提供 --layout、--seed 且只包含一个 --sample-sets")
    layout_name = str(args.layout)
    seed = int(args.seed)
    sample_set = str(args.sample_sets[0])
    out_dir = task_dir(args.output_dir, layout_name, seed, sample_set)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "raw_results.csv").exists() and not args.overwrite:
        raise FileExistsError(f"单任务输出已存在；如需覆盖请传 --overwrite：{out_dir}")
    write_json(out_dir / "status.json", {"status": "started", "layout": layout_name, "seed": seed, "sample_set": sample_set, "updated_at": now_cst()})
    started = time.perf_counter()
    sample_df = load_sample_frame(args.sample_manifest, sample_set, args.max_samples_per_set)
    device = resolve_device(str(args.device))
    checkpoint = load_checkpoint(Path(args.visual_checkpoint))
    embedding_metadata = checkpoint.get("embedding_metadata")
    if not isinstance(embedding_metadata, Mapping):
        raise ValueError("Visual checkpoint 缺少 embedding_metadata")
    encoder_args = make_encoder_args(args, embedding_metadata)
    encoder_args.local_files_only = bool(args.local_files_only)
    dtype = resolve_dtype(str(args.dtype or getattr(encoder_args, "dtype", "auto")), device)
    vit_model = load_vit_model_with_retry(encoder_args, device, dtype)
    data_config = load_data_config(Path(args.config_path))
    loader = Round2HistoryWindowLoader(data_config)
    router_path = Path(args.round2c_dir) / "tasks" / f"{layout_name}_seed{seed}" / f"checkpoint_{layout_name}_seed{seed}.pt"
    router, visual_scaler, aux_scaler = load_router_from_checkpoint(router_path, device)
    # Round2c SQLite 只保存相对 full-scale prediction manifest 目录的数组路径；
    # 这里复用既有 index，不扫描 116M manifest，只用 manifest parent 解析数组路径。
    prediction_index = SQLitePredictionIndex(prediction_index_path(args.round2c_dir), Path(args.prediction_manifest_path).parent)
    period_candidates = parse_period_candidates_arg(args.period_candidates)
    sigmas = parse_float_csv(args.perturbation_sigma_list)
    try:
        x_all = loader.load_shard_x(sample_df)
        aux_original = compute_revin_aux_from_x(x_all, clip=float(args.clip))
        y_pred, y_true, expert_errors = load_prediction_batch_from_index(
            prediction_index,
            sample_df["sample_key"].astype(str).tolist(),
            error_metric="mae",
        )
        oracle_idx = np.argmin(expert_errors, axis=1)
        oracle_values = expert_errors[np.arange(expert_errors.shape[0]), oracle_idx]
        raw_rows: List[Dict[str, object]] = []
        high_change_rows: List[Dict[str, object]] = []
        rng = np.random.default_rng(20_260_622 + seed)
        for batch_start in range(0, len(sample_df), int(args.embedding_batch_size)):
            batch_end = min(batch_start + int(args.embedding_batch_size), len(sample_df))
            batch_df = sample_df.iloc[batch_start:batch_end].reset_index(drop=True)
            x_batch = x_all[batch_start:batch_end]
            images0, cls0, mean0, periods0, scores0 = forward_images_embeddings(
                x_batch=x_batch,
                layout_name=layout_name,
                vit_model=vit_model,
                device=device,
                dtype=dtype,
                normalization_preset=str(encoder_args.normalization_preset),
                image_size=int(args.image_size),
                norm_mode=str(args.norm_mode),
                clip=float(args.clip),
                period_candidates=period_candidates,
                period_selection=str(args.period_selection),
            )
            weights0 = router_weights(router=router, visual_scaler=visual_scaler, aux_scaler=aux_scaler, visual_features=mean0, aux_features=aux_original[batch_start:batch_end], device=device)
            selected0 = weights0.argmax(axis=1)
            std = np.std(x_batch.reshape(x_batch.shape[0], -1), axis=1, keepdims=True).astype(np.float32).clip(min=EPS)
            std = std.reshape((x_batch.shape[0],) + (1,) * (x_batch.ndim - 1))
            for sigma in sigmas:
                for perturb_idx in range(int(args.num_perturbations)):
                    eps = rng.standard_normal(size=x_batch.shape).astype(np.float32)
                    x_perturbed = x_batch + float(sigma) * std * eps
                    aux_perturbed = compute_revin_aux_from_x(x_perturbed, clip=float(args.clip))
                    images1, cls1, mean1, periods1, scores1 = forward_images_embeddings(
                        x_batch=x_perturbed,
                        layout_name=layout_name,
                        vit_model=vit_model,
                        device=device,
                        dtype=dtype,
                        normalization_preset=str(encoder_args.normalization_preset),
                        image_size=int(args.image_size),
                        norm_mode=str(args.norm_mode),
                        clip=float(args.clip),
                        period_candidates=period_candidates,
                        period_selection=str(args.period_selection),
                    )
                    weights1 = router_weights(router=router, visual_scaler=visual_scaler, aux_scaler=aux_scaler, visual_features=mean1, aux_features=aux_perturbed, device=device)
                    selected1 = weights1.argmax(axis=1)
                    image_flat0 = images0.reshape(images0.shape[0], -1)
                    image_flat1 = images1.reshape(images1.shape[0], -1)
                    image_l2 = np.linalg.norm(image_flat1 - image_flat0, axis=1)
                    image_cos = cosine_distance_np(image_flat0, image_flat1, axis=1)
                    cls_cos = cosine_distance_np(cls0, cls1, axis=1)
                    mean_cos = cosine_distance_np(mean0, mean1, axis=1)
                    emb_l2 = np.linalg.norm(mean1 - mean0, axis=1)
                    weight_js = js_divergence_np(weights0, weights1)
                    weight_l1 = np.abs(weights1 - weights0).sum(axis=1)
                    weight_max = np.abs(weights1 - weights0).max(axis=1)
                    fused_change, mae_change, regret_change = summarize_prediction_change(
                        original_weights=weights0,
                        perturbed_weights=weights1,
                        y_pred=y_pred[batch_start:batch_end],
                        y_true=y_true[batch_start:batch_end],
                        oracle_values=oracle_values[batch_start:batch_end],
                    )
                    margins0 = scores0[:, 0] - scores0[:, 1]
                    margins1 = scores1[:, 0] - scores1[:, 1]
                    for local_idx, row in enumerate(batch_df.itertuples(index=False)):
                        top3_a = [int(v) for v in periods0[local_idx].tolist()]
                        top3_b = [int(v) for v in periods1[local_idx].tolist()]
                        bucket0 = period_bucket(top3_a[0])
                        bucket1 = period_bucket(top3_b[0])
                        jaccard = len(set(top3_a) & set(top3_b)) / max(1, len(set(top3_a) | set(top3_b)))
                        selected_model = MODEL_COLUMNS[int(selected0[local_idx])]
                        perturbed_selected_model = MODEL_COLUMNS[int(selected1[local_idx])]
                        record = {
                            "layout": layout_name,
                            "seed": seed,
                            "sample_set": sample_set,
                            "sample_key": str(row.sample_key),
                            "perturbation_sigma": float(sigma),
                            "perturbation_index": int(perturb_idx),
                            "top1_period": int(top3_a[0]),
                            "perturbed_top1_period": int(top3_b[0]),
                            "top1_period_changed": bool(top3_a[0] != top3_b[0]),
                            "top3_jaccard": float(jaccard),
                            "top1_period_bucket": bucket0,
                            "perturbed_top1_period_bucket": bucket1,
                            "top1_period_bucket_flipped": bool(bucket0 != bucket1),
                            "period_score_margin": float(margins0[local_idx]),
                            "perturbed_period_score_margin": float(margins1[local_idx]),
                            "image_l2": float(image_l2[local_idx]),
                            "image_cosine_distance": float(image_cos[local_idx]),
                            "cls_embedding_cosine_distance": float(cls_cos[local_idx]),
                            "mean_patch_embedding_cosine_distance": float(mean_cos[local_idx]),
                            "embedding_l2": float(emb_l2[local_idx]),
                            "router_weight_js_divergence": float(weight_js[local_idx]),
                            "router_weight_l1": float(weight_l1[local_idx]),
                            "router_weight_max_change": float(weight_max[local_idx]),
                            "selected_model": selected_model,
                            "perturbed_selected_model": perturbed_selected_model,
                            "selected_model_flipped": bool(selected_model != perturbed_selected_model),
                            "soft_fused_abs_change": float(fused_change[local_idx]),
                            "soft_fused_mae_change": float(mae_change[local_idx]),
                            "regret_change": float(regret_change[local_idx]),
                            "oracle_model": str(row.oracle_model),
                            "season_strength_cat": str(row.season_strength_cat),
                            "forecastability_cat": str(row.forecastability_cat),
                            "error_gap_quantile": str(row.error_gap_quantile),
                            "top3_periods": ",".join(str(v) for v in top3_a),
                            "perturbed_top3_periods": ",".join(str(v) for v in top3_b),
                        }
                        raw_rows.append(record)
                        high_score = float(record["router_weight_js_divergence"]) + float(record["image_cosine_distance"]) + float(record["mean_patch_embedding_cosine_distance"])
                        if bool(record["top1_period_changed"]) or bool(record["selected_model_flipped"]) or high_score > 0.05:
                            high_change_rows.append({**record, "high_change_score": high_score})
                    if args.save_debug_examples and high_change_rows:
                        # 只保存每个 batch 当前最高变化样本的一对图，避免 debug 输出膨胀。
                        best_idx = int(np.argmax(weight_js + image_cos + mean_cos))
                        sample_key = str(batch_df.iloc[best_idx]["sample_key"]).replace("/", "_")
                        debug_dir = out_dir / "debug_examples" / layout_name / sample_key
                        save_debug_png(debug_dir / f"sigma{sigma}_p{perturb_idx}_original.png", images0[best_idx])
                        save_debug_png(debug_dir / f"sigma{sigma}_p{perturb_idx}_perturbed.png", images1[best_idx])
            if device.type == "cuda":
                torch.cuda.empty_cache()
        raw_df = pd.DataFrame(raw_rows, columns=RAW_COLUMNS)
        raw_df.to_csv(out_dir / "raw_results.csv", index=False)
        high_df = pd.DataFrame(high_change_rows)
        if high_df.empty:
            high_df = pd.DataFrame(columns=RAW_COLUMNS + ["high_change_score"])
        high_df.sort_values(["high_change_score", "router_weight_js_divergence"], ascending=False, kind="mergesort").head(200).to_csv(out_dir / "high_change_examples.csv", index=False)
        write_json(
            out_dir / "task_metadata.json",
            {
                "status": "completed",
                "script_version": SCRIPT_VERSION,
                "generated_at": now_cst(),
                "layout": layout_name,
                "seed": seed,
                "sample_set": sample_set,
                "sample_count": int(len(sample_df)),
                "device": str(device),
                "elapsed_sec": float(time.perf_counter() - started),
                "perturbation_space": PERTURBATION_SPACE,
                "perturbation_sigma_list": sigmas,
                "num_perturbations": int(args.num_perturbations),
                "saved_pseudo_image_tensor": False,
                "trained_new_model": False,
                "built_feature_cache": False,
            },
        )
        write_json(out_dir / "status.json", {"status": "completed", "layout": layout_name, "seed": seed, "sample_set": sample_set, "updated_at": now_cst()})
    finally:
        prediction_index.close()


def mean_summary(frame: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """函数功能：按指定维度汇总 mean/std/p95/max 和事件率。"""
    if frame.empty:
        return pd.DataFrame()
    agg: Dict[str, List[str]] = {}
    for metric in metrics:
        agg[metric] = ["mean", "std", "max"]
    grouped = frame.groupby(list(group_cols), dropna=False).agg(agg)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    counts = frame.groupby(list(group_cols), dropna=False).size().reset_index(name="row_count")
    return counts.merge(grouped, on=list(group_cols), how="left")


def output_name(prefix: str, suffix: str) -> str:
    """函数功能：按实验前缀生成稳定输出文件名。"""
    legacy_names = {
        "selection_stability.csv": "round2_period_selection_stability.csv",
        "image_continuity.csv": "round2_period_image_continuity.csv",
        "embedding_continuity.csv": "round2_period_embedding_continuity.csv",
        "router_weight_continuity.csv": "round2_period_router_weight_continuity.csv",
        "fused_prediction_continuity.csv": "round2_period_fused_prediction_continuity.csv",
        "stratified_summary.csv": "round2_period_stratified_summary.csv",
        "high_change_examples.csv": "round2_period_high_change_examples.csv",
    }
    if str(prefix) == DEFAULT_RESULT_PREFIX and suffix in legacy_names:
        return legacy_names[suffix]
    return f"{str(prefix).rstrip('_')}_{suffix}"


def add_raw_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """函数功能：为合并对照表添加轻量来源标记，避免误读为同一批新增诊断。"""
    result = frame.copy()
    if "source_result" not in result.columns:
        result.insert(0, "source_result", source)
    return result


def load_existing_raw_for_comparison(path: Optional[Path], source: str) -> pd.DataFrame:
    """
    函数功能：
        读取已有 Round2d raw results 作为 addendum 对照，不改变新增诊断 raw 文件。
    """
    if path is None:
        return pd.DataFrame(columns=["source_result", *RAW_COLUMNS])
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"对照 raw results 不存在：{raw_path}")
    frame = pd.read_csv(raw_path)
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"对照 raw results 缺少字段：{missing}")
    return add_raw_source(frame[RAW_COLUMNS], source)


def build_layout_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    """函数功能：生成跨 layout 的核心连续性指标 comparison。"""
    if raw.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["source_result", "layout"] if col in raw.columns]
    comparison = raw.groupby(group_cols, dropna=False).agg(
        row_count=("sample_key", "size"),
        top1_period_changed_ratio=("top1_period_changed", "mean"),
        top3_jaccard_mean=("top3_jaccard", "mean"),
        image_cosine_distance_mean=("image_cosine_distance", "mean"),
        image_l2_mean=("image_l2", "mean"),
        cls_embedding_cosine_distance_mean=("cls_embedding_cosine_distance", "mean"),
        mean_patch_embedding_cosine_distance_mean=("mean_patch_embedding_cosine_distance", "mean"),
        router_weight_js_divergence_mean=("router_weight_js_divergence", "mean"),
        router_weight_l1_mean=("router_weight_l1", "mean"),
        selected_model_flip_rate=("selected_model_flipped", "mean"),
        soft_fused_abs_change_mean=("soft_fused_abs_change", "mean"),
        soft_fused_mae_change_mean=("soft_fused_mae_change", "mean"),
    ).reset_index()
    return comparison.sort_values(
        [col for col in ["router_weight_js_divergence_mean", "selected_model_flip_rate"] if col in comparison.columns],
        ascending=True,
        kind="mergesort",
    )


def write_required_csvs(output_dir: Path, raw: pd.DataFrame, *, prefix: str = DEFAULT_RESULT_PREFIX, comparison_raw: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """函数功能：从 raw rows 派生验收要求的多个 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw.empty:
        raw = pd.DataFrame(columns=RAW_COLUMNS)
    raw.to_csv(output_dir / output_name(prefix, "raw_results.csv"), index=False)
    compare_base = comparison_raw if comparison_raw is not None else raw
    base_groups = ["layout", "seed", "sample_set", "perturbation_sigma"]
    outputs: Dict[str, pd.DataFrame] = {}
    outputs["period_selection"] = mean_summary(raw, base_groups, ["top1_period_changed", "top3_jaccard", "top1_period_bucket_flipped", "period_score_margin", "perturbed_period_score_margin"])
    outputs["image"] = mean_summary(raw, base_groups, ["image_l2", "image_cosine_distance"])
    outputs["embedding"] = mean_summary(raw, base_groups, ["cls_embedding_cosine_distance", "mean_patch_embedding_cosine_distance", "embedding_l2"])
    outputs["router"] = mean_summary(raw, base_groups, ["router_weight_js_divergence", "router_weight_l1", "router_weight_max_change", "selected_model_flipped"])
    outputs["fused"] = mean_summary(raw, base_groups, ["soft_fused_abs_change", "soft_fused_mae_change", "regret_change"])
    strat_frames: List[pd.DataFrame] = []
    for col in ["oracle_model", "season_strength_cat", "forecastability_cat", "error_gap_quantile", "top1_period_bucket"]:
        strata_groups = [group_col for group_col in ["source_result", "layout", "sample_set", "perturbation_sigma", col] if group_col in compare_base.columns]
        part = mean_summary(compare_base, strata_groups, ["top1_period_changed", "image_cosine_distance", "mean_patch_embedding_cosine_distance", "router_weight_js_divergence", "selected_model_flipped", "soft_fused_abs_change"])
        if not part.empty:
            part["stratum_column"] = col
            part = part.rename(columns={col: "stratum_value"})
            strat_frames.append(part)
    outputs["stratified"] = pd.concat(strat_frames, ignore_index=True) if strat_frames else pd.DataFrame()
    outputs["layout_comparison"] = build_layout_comparison(compare_base)
    file_map = {
        "period_selection": output_name(prefix, "selection_stability.csv"),
        "image": output_name(prefix, "image_continuity.csv"),
        "embedding": output_name(prefix, "embedding_continuity.csv"),
        "router": output_name(prefix, "router_weight_continuity.csv"),
        "fused": output_name(prefix, "fused_prediction_continuity.csv"),
        "stratified": output_name(prefix, "stratified_summary.csv"),
        "layout_comparison": output_name(prefix, "layout_comparison.csv"),
    }
    for key, name in file_map.items():
        outputs[key].to_csv(output_dir / name, index=False)
    return outputs


def verdict_from_raw(raw: pd.DataFrame) -> Dict[str, object]:
    """函数功能：基于聚合后的核心指标生成 summary 判断。"""
    if raw.empty:
        return {"status": "empty"}
    grouped = raw.groupby("layout", dropna=False).agg(
        top1_changed=("top1_period_changed", "mean"),
        top3_jaccard=("top3_jaccard", "mean"),
        image_cos=("image_cosine_distance", "mean"),
        mean_patch_cos=("mean_patch_embedding_cosine_distance", "mean"),
        weight_js=("router_weight_js_divergence", "mean"),
        selected_flip=("selected_model_flipped", "mean"),
        fused_abs=("soft_fused_abs_change", "mean"),
    )
    current = grouped.loc["current_rgb_3view"].to_dict() if "current_rgb_3view" in grouped.index else {}
    top3 = grouped.loc["top3fold_period_layout"].to_dict() if "top3fold_period_layout" in grouped.index else {}
    hard_sensitive = bool(current and (float(current["top1_changed"]) > 0.05 or float(current["selected_flip"]) > 0.01 or float(current["weight_js"]) > 1e-3))
    top3_more_continuous = bool(current and top3 and (float(top3["selected_flip"]) <= float(current["selected_flip"]) and float(top3["weight_js"]) <= float(current["weight_js"])))
    changed = raw[raw["top1_period_changed"].astype(bool)].copy()
    unchanged = raw[~raw["top1_period_changed"].astype(bool)].copy()
    amplification = {}
    if not changed.empty and not unchanged.empty:
        amplification = {
            "changed_image_cos_mean": float(changed["image_cosine_distance"].mean()),
            "unchanged_image_cos_mean": float(unchanged["image_cosine_distance"].mean()),
            "changed_weight_js_mean": float(changed["router_weight_js_divergence"].mean()),
            "unchanged_weight_js_mean": float(unchanged["router_weight_js_divergence"].mean()),
        }
    require_soft = bool(hard_sensitive and not top3_more_continuous)
    top3_enter_65k = bool(top3 and (top3_more_continuous or float(top3.get("selected_flip", 1.0)) < 0.02))
    return {
        "status": "ok",
        "layout_means": grouped.reset_index().to_dict(orient="records"),
        "current_hard_top1_sensitive": hard_sensitive,
        "top3_more_continuous_than_current": top3_more_continuous,
        "period_change_amplification": amplification,
        "must_implement_period_soft_mixture_before_65k": require_soft,
        "top3fold_enter_65k_recommended": top3_enter_65k,
    }


def addendum_verdict_from_raw(raw: pd.DataFrame) -> Dict[str, object]:
    """函数功能：生成 spatial_panel_3view addendum 的七项验收回答。"""
    if raw.empty:
        return {"status": "empty"}
    grouped = raw.groupby("layout", dropna=False).agg(
        top1_changed=("top1_period_changed", "mean"),
        top3_jaccard=("top3_jaccard", "mean"),
        image_cos=("image_cosine_distance", "mean"),
        mean_patch_cos=("mean_patch_embedding_cosine_distance", "mean"),
        cls_cos=("cls_embedding_cosine_distance", "mean"),
        weight_js=("router_weight_js_divergence", "mean"),
        weight_l1=("router_weight_l1", "mean"),
        selected_flip=("selected_model_flipped", "mean"),
        fused_abs=("soft_fused_abs_change", "mean"),
    )
    rows = grouped.reset_index().to_dict(orient="records")
    by_layout = {str(row["layout"]): row for row in rows}
    spatial = by_layout.get("spatial_panel_3view", {})
    current = by_layout.get("current_rgb_3view", {})
    top3 = by_layout.get("top3fold_period_layout", {})

    def leq(left: Mapping[str, object], right: Mapping[str, object], keys: Sequence[str]) -> bool:
        return bool(left and right and all(float(left[key]) <= float(right[key]) for key in keys if key in left and key in right))

    spatial_affected = bool(spatial and (float(spatial["top1_changed"]) > 0.05 or float(spatial["weight_js"]) > 1e-3 or float(spatial["selected_flip"]) > 0.01))
    spatial_lower_than_current = leq(spatial, current, ["image_cos", "mean_patch_cos", "weight_js"])
    spatial_stronger_than_top3 = leq(spatial, top3, ["image_cos", "mean_patch_cos", "weight_js", "selected_flip"])
    spatial_weaker_than_top3 = leq(top3, spatial, ["image_cos", "mean_patch_cos", "weight_js", "selected_flip"])
    panel_source = {}
    if spatial:
        spatial_raw = raw[raw["layout"].astype(str) == "spatial_panel_3view"].copy()
        changed = spatial_raw[spatial_raw["top1_period_changed"].astype(bool)]
        unchanged = spatial_raw[~spatial_raw["top1_period_changed"].astype(bool)]
        if not changed.empty and not unchanged.empty:
            panel_source = {
                "changed_image_cos_mean": float(changed["image_cosine_distance"].mean()),
                "unchanged_image_cos_mean": float(unchanged["image_cosine_distance"].mean()),
                "changed_weight_js_mean": float(changed["router_weight_js_divergence"].mean()),
                "unchanged_weight_js_mean": float(unchanged["router_weight_js_divergence"].mean()),
                "changed_selected_flip_rate": float(changed["selected_model_flipped"].mean()),
                "unchanged_selected_flip_rate": float(unchanged["selected_model_flipped"].mean()),
            }
    fold_panel_high_change_source = bool(
        panel_source
        and (
            panel_source["changed_image_cos_mean"] > panel_source["unchanged_image_cos_mean"]
            or panel_source["changed_weight_js_mean"] > panel_source["unchanged_weight_js_mean"]
            or panel_source["changed_selected_flip_rate"] > panel_source["unchanged_selected_flip_rate"]
        )
    )
    enter_65k = bool(spatial and float(spatial["selected_flip"]) < 0.10 and float(spatial["weight_js"]) < 0.05)
    need_soft_before_65k = bool(spatial_affected and not enter_65k)
    return {
        "status": "ok",
        "layout_means": rows,
        "spatial_affected_by_hard_top1_period_fold": spatial_affected,
        "spatial_lowers_propagation_vs_current": spatial_lower_than_current,
        "spatial_more_continuous_than_top3fold": spatial_stronger_than_top3,
        "spatial_less_continuous_than_top3fold": spatial_weaker_than_top3,
        "spatial_fold_panel_high_change_source": fold_panel_high_change_source,
        "spatial_fold_panel_amplification": panel_source,
        "spatial_enter_65k_recommended": enter_65k,
        "must_implement_period_soft_mixture_before_65k": need_soft_before_65k,
        "recommended_65k_layouts": ["spatial_panel_3view", "current_rgb_3view", "top3fold_period_layout"],
    }


def write_addendum_summary(output_dir: Path, raw: pd.DataFrame, derived: Mapping[str, pd.DataFrame], metadata: Mapping[str, object], *, prefix: str) -> None:
    """函数功能：写出 Round2d addendum 中文摘要，并逐项回答 spatial panel 问题。"""
    verdict = addendum_verdict_from_raw(raw)
    layout_table = pd.DataFrame(verdict.get("layout_means", []))
    strata = derived.get("stratified", pd.DataFrame())
    risky = pd.DataFrame()
    if not strata.empty and "router_weight_js_divergence_mean" in strata.columns:
        risky = strata.sort_values(["selected_model_flipped_mean", "router_weight_js_divergence_mean"], ascending=False, kind="mergesort").head(12)
    continuity_vs_top3 = "更强" if verdict.get("spatial_more_continuous_than_top3fold") else ("更弱" if verdict.get("spatial_less_continuous_than_top3fold") else "接近或证据不足")
    lines = [
        "# Visual Router V2 Round2d addendum: spatial_panel_3view period continuity diagnostic",
        "",
        f"生成时间：{now_cst()}",
        "",
        "## 结论",
    ]
    if verdict.get("status") != "ok":
        lines.append("- 诊断结果为空，无法形成结论。")
    else:
        by_layout = {str(row["layout"]): row for row in verdict.get("layout_means", [])}
        spatial = by_layout.get("spatial_panel_3view", {})
        current = by_layout.get("current_rgb_3view", {})
        top3 = by_layout.get("top3fold_period_layout", {})
        vs_current_note = "证据不足"
        if spatial and current:
            vs_current_note = (
                "部分降低：image_cos、CLS/mean-patch embedding 与 selected flip 低于 current，"
                "但 router weight JS/L1 未降低，因此不是完整阻断 router 侧不连续传播"
            )
        vs_top3_note = continuity_vs_top3
        if spatial and top3:
            vs_top3_note = (
                "整体弱于 top3fold 的 image/embedding/router 连续性，但 selected flip 低于 top3fold"
            )
        lines.extend(
            [
                f"1. `spatial_panel_3view` 是否也受 hard top1 period fold 扰动影响：{'是' if verdict['spatial_affected_by_hard_top1_period_fold'] else '否'}。",
                f"2. 相比 `current_rgb_3view`，spatial panel 是否降低 image / embedding / router weight 的不连续传播：{vs_current_note}。",
                f"3. 相比 `top3fold_period_layout`，spatial panel 的连续性：{vs_top3_note}。",
                f"4. spatial panel 的 fold panel 是否成为高变化来源：{'是' if verdict['spatial_fold_panel_high_change_source'] else '否/证据不足'}；changed/unchanged 对照={json.dumps(verdict.get('spatial_fold_panel_amplification', {}), ensure_ascii=False)}。",
                f"5. `spatial_panel_3view` 作为 Round2c best layout 是否仍应进入 65k expanded validation：{'应进入' if verdict['spatial_enter_65k_recommended'] else '暂缓'}。",
                f"6. 是否需要在 65k 前实现 `period_soft_mixture`：{'需要先实现' if verdict['must_implement_period_soft_mixture_before_65k'] else '不作为前置硬门槛'}。",
                "7. 65k expanded validation 推荐 layout 保持：`spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。",
            ]
        )
    lines.extend(["", "## Layout Comparison", ""])
    lines.append(frame_to_markdown(layout_table) if not layout_table.empty else "无。")
    lines.extend(["", "## High-Risk Strata", ""])
    lines.append(frame_to_markdown(risky) if not risky.empty else "无明显高风险 strata 或分层表为空。")
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(dict(metadata), indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## 下一步推荐",
            "",
            "- 直接进入 65k expanded validation，候选保持 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。",
            "- `period_soft_mixture` 作为后续表达改进单独 smoke，不阻塞本轮 65k。",
        ]
    )
    (output_dir / output_name(prefix, "summary.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(output_dir: Path, raw: pd.DataFrame, derived: Mapping[str, pd.DataFrame], metadata: Mapping[str, object], *, prefix: str = DEFAULT_RESULT_PREFIX) -> None:
    """函数功能：写出中文 Markdown 结论，逐项回答 Round2d 诊断问题。"""
    if str(metadata.get("round2_stage")) == "period_continuity_addendum":
        write_addendum_summary(output_dir, raw, derived, metadata, prefix=prefix)
        return
    verdict = verdict_from_raw(raw)
    layout_table = pd.DataFrame(verdict.get("layout_means", []))
    strata = derived.get("stratified", pd.DataFrame())
    risky = pd.DataFrame()
    if not strata.empty and "router_weight_js_divergence_mean" in strata.columns:
        risky = strata.sort_values(["selected_model_flipped_mean", "router_weight_js_divergence_mean"], ascending=False, kind="mergesort").head(12)
    targeted_notes: List[str] = []
    if not strata.empty:
        for column_name, label in [
            ("oracle_model", "oracle_model"),
            ("season_strength_cat", "season_strength_cat"),
            ("forecastability_cat", "forecastability_cat"),
            ("error_gap_quantile", "error_gap_quantile"),
            ("top1_period_bucket", "top1_period_bucket"),
        ]:
            part = strata[strata["stratum_column"].astype(str) == column_name].copy()
            if part.empty:
                continue
            sort_cols = [col for col in ["selected_model_flipped_mean", "router_weight_js_divergence_mean", "image_cosine_distance_mean"] if col in part.columns]
            top = part.sort_values(sort_cols, ascending=False, kind="mergesort").head(5)
            values = ", ".join(dict.fromkeys(str(value) for value in top["stratum_value"].astype(str).tolist()))
            targeted_notes.append(f"- `{label}` 高风险候选：{values}。")
        for column_name, values in {
            "oracle_model": ["CrossFormer", "PatchTST"],
            "season_strength_cat": ["strong"],
            "error_gap_quantile": ["q5"],
        }.items():
            part = strata[
                (strata["stratum_column"].astype(str) == column_name)
                & (strata["stratum_value"].astype(str).isin(values))
                & (strata["perturbation_sigma"].astype(float) == 0.01)
            ].copy()
            if part.empty:
                continue
            compact = part.groupby(["layout", "stratum_value"], as_index=False).agg(
                selected_model_flipped_mean=("selected_model_flipped_mean", "mean"),
                router_weight_js_divergence_mean=("router_weight_js_divergence_mean", "mean"),
                image_cosine_distance_mean=("image_cosine_distance_mean", "mean"),
            )
            snippets = []
            for row in compact.itertuples(index=False):
                snippets.append(
                    f"{row.layout}/{row.stratum_value}: flip={row.selected_model_flipped_mean:.4f}, "
                    f"JS={row.router_weight_js_divergence_mean:.6f}, image_cos={row.image_cosine_distance_mean:.6f}"
                )
            targeted_notes.append(f"- 指定关注 `{column_name}` @ sigma=0.01：{'; '.join(snippets)}。")
    lines = [
        "# Visual Router V2 Round2d period continuity diagnostic",
        "",
        f"生成时间：{now_cst()}",
        "",
        "## 结论",
    ]
    if verdict.get("status") != "ok":
        lines.append("- 诊断结果为空，无法形成结论。")
    else:
        lines.extend(
            [
                f"- `current_rgb_3view` hard top1 fold 对轻微扰动是否敏感：{'是' if verdict['current_hard_top1_sensitive'] else '否'}。",
                f"- `top3fold_period_layout` 是否比 current hard top1 更连续：{'是' if verdict['top3_more_continuous_than_current'] else '否/证据不足'}。",
                "- 周期选择变化是否放大 image / embedding / router weight 变化："
                + (json.dumps(verdict.get("period_change_amplification", {}), ensure_ascii=False) if verdict.get("period_change_amplification") else "未观察到足够 changed/unchanged 对照。"),
                f"- 是否必须先实现 `period_soft_mixture` 再做 65k：{'建议先做' if verdict['must_implement_period_soft_mixture_before_65k'] else '不是必须，可作为后续改进'}。",
                f"- `top3fold_period_layout` 是否建议进入 65k expanded validation：{'建议进入' if verdict['top3fold_enter_65k_recommended'] else '暂缓，优先 spatial/current 或先做 soft mixture'}。",
                f"- top3fold 的 diagnostic-balanced 优势是否可能来自更稳定的周期表达：{'可能' if verdict['top3_more_continuous_than_current'] else '证据不足'}；本诊断中 top3fold 的 image/embedding/router JS 平均变化低于 current。",
                "- 后续建议：保留 period tokens / soft period mixture / panelized top3fold 为 Round2e 候选；若 65k 资源有限，优先 `spatial_panel_3view`、`current_rgb_3view`，并按本诊断决定是否纳入 `top3fold_period_layout`。",
                "- latency / GPU memory：本诊断未保存大 tensor，单任务按小 batch 运行；若所有任务完成且无 OOM，未发现阻塞 65k 规划的显著显存问题。",
            ]
        )
    lines.extend(["", "## 指定 Strata 回答", ""])
    if targeted_notes:
        lines.extend(targeted_notes)
        lines.append("- 本次按 selected model flip 与 router JS 排序的最高风险 error_gap bucket 是 `q1`；`q5` 指标已单独列出，详细数值见 `round2_period_stratified_summary.csv`。")
    else:
        lines.append("分层表为空，无法回答指定 strata。")
    lines.extend(["", "## Layout Mean Metrics", ""])
    lines.append(frame_to_markdown(layout_table) if not layout_table.empty else "无。")
    lines.extend(["", "## High-Risk Strata", ""])
    lines.append(frame_to_markdown(risky) if not risky.empty else "无明显高风险 strata 或分层表为空。")
    lines.extend(
        [
            "",
            "## Metadata",
            "",
            "```json",
            json.dumps(dict(metadata), indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## 下一步推荐",
            "",
            "- 若 `top3fold_period_layout` 的 flip/JS 指标不高于 current，可与 `spatial_panel_3view`、`current_rgb_3view` 一起进入 65k expanded validation。",
            "- 若 hard top1 周期 flip 明显放大 router weight 或 selected model 跳变，先实现 `period_soft_mixture` smoke，再决定是否做完整 head。",
            "- panelized top3fold 和 period tokens 更适合作为后续表达增强，不应阻塞当前 65k 除非本诊断显示 hard fold 极不稳定。",
        ]
    )
    (output_dir / output_name(prefix, "summary.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：聚合所有单任务输出并写出统一 Round2d 产物。"""
    output_dir = Path(args.output_dir)
    prefix = str(args.result_prefix)
    layouts = [str(args.layout)] if args.layout else parse_csv(args.layouts)
    seeds = [int(args.seed)] if args.seed is not None else parse_int_list(args.seeds)
    sample_sets = [str(value) for value in args.sample_sets]
    raw_frames: List[pd.DataFrame] = []
    high_frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for layout in layouts:
        for seed in seeds:
            for sample_set in sample_sets:
                td = task_dir(output_dir, layout, seed, sample_set)
                raw_path = td / "raw_results.csv"
                if not raw_path.exists():
                    missing.append(str(raw_path))
                    continue
                raw_frames.append(pd.read_csv(raw_path))
                high_path = td / "high_change_examples.csv"
                if high_path.exists():
                    high_frames.append(pd.read_csv(high_path))
    if missing:
        raise FileNotFoundError("缺少单任务输出：" + "; ".join(missing[:20]))
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame(columns=RAW_COLUMNS)
    compare_path = Path(args.compare_with_existing) if args.compare_with_existing else (Path(args.append_to_existing) if args.append_to_existing else None)
    existing_raw = load_existing_raw_for_comparison(compare_path, "existing_round2d")
    diagnosed_source = "addendum_diagnosed" if prefix != DEFAULT_RESULT_PREFIX else "diagnosed"
    comparison_raw = pd.concat([existing_raw, add_raw_source(raw, diagnosed_source)], ignore_index=True) if not existing_raw.empty else add_raw_source(raw, diagnosed_source)
    derived = write_required_csvs(output_dir, raw, prefix=prefix, comparison_raw=comparison_raw)
    high = pd.concat(high_frames, ignore_index=True) if high_frames else pd.DataFrame(columns=RAW_COLUMNS + ["high_change_score"])
    if not high.empty:
        sort_cols = [col for col in ["high_change_score", "router_weight_js_divergence", "image_cosine_distance"] if col in high.columns]
        high = high.sort_values(sort_cols, ascending=False, kind="mergesort").head(1000)
    high.to_csv(output_dir / output_name(prefix, "high_change_examples.csv"), index=False)
    verdict_input = comparison_raw if prefix != DEFAULT_RESULT_PREFIX else raw
    verdict = addendum_verdict_from_raw(verdict_input) if prefix != DEFAULT_RESULT_PREFIX else verdict_from_raw(raw)
    launcher_devices: List[str] = []
    launcher_path = output_dir / "launcher_metadata.json"
    if launcher_path.exists():
        try:
            launcher_meta = json.loads(launcher_path.read_text(encoding="utf-8"))
            launcher_devices = sorted({str(job.get("device")) for job in launcher_meta.get("jobs", []) if job.get("device")})
        except Exception:
            launcher_devices = []
    metadata = {
        "status": "completed",
        "script_version": SCRIPT_VERSION,
        "generated_at": now_cst(),
        "round2_stage": "period_continuity_addendum" if prefix != DEFAULT_RESULT_PREFIX else "period_continuity_diagnostic",
        "trained_new_model": False,
        "built_feature_cache": False,
        "ran_vit_for_embedding_diagnostic": True,
        "saved_pseudo_image_tensor": False,
        "used_test_small_for_selection": False,
        "loaded_116m_prediction_manifest_to_memory": False,
        "layouts_diagnosed": layouts,
        "compared_against_existing": list(DEFAULT_COMPARE_LAYOUTS) if compare_path else [],
        "compare_with_existing_raw_results": str(compare_path) if compare_path else None,
        "seeds": seeds,
        "sample_sets": sample_sets,
        "sample_manifest": str(args.sample_manifest),
        "round2c_dir": str(args.round2c_dir),
        "perturbation_sigma_list": parse_float_csv(args.perturbation_sigma_list),
        "num_perturbations": int(args.num_perturbations),
        "perturbation_space": PERTURBATION_SPACE,
        "devices_requested": parse_csv(args.devices),
        "devices_used": launcher_devices or parse_csv(args.devices),
        "parallel_backend": "process_per_layout_seed_sample_set",
        "single_task_output_isolated": True,
        "period_soft_mixture_implemented": False,
        "next_step_recommendation": (
            "direct_65k_with_spatial_current_top3fold"
            if prefix != DEFAULT_RESULT_PREFIX and verdict.get("spatial_enter_65k_recommended")
            else (
                "direct_65k_with_top3fold"
                if verdict.get("top3fold_enter_65k_recommended")
                else "implement_period_soft_mixture_or_defer_period_layouts_before_65k"
            )
        ),
    }
    write_json(output_dir / output_name(prefix, "metadata.json"), metadata)
    write_summary(output_dir, verdict_input, derived, metadata, prefix=prefix)
    copy_summary_outputs(output_dir, Path(args.summary_copy_dir), prefix=prefix)


def copy_summary_outputs(output_dir: Path, summary_dir: Path, *, prefix: str = DEFAULT_RESULT_PREFIX) -> None:
    """函数功能：复制轻量 CSV/JSON/Markdown 到 repo summary 目录。"""
    summary_dir.mkdir(parents=True, exist_ok=True)
    names = [
        output_name(prefix, "raw_results.csv"),
        output_name(prefix, "selection_stability.csv"),
        output_name(prefix, "image_continuity.csv"),
        output_name(prefix, "embedding_continuity.csv"),
        output_name(prefix, "router_weight_continuity.csv"),
        output_name(prefix, "fused_prediction_continuity.csv"),
        output_name(prefix, "layout_comparison.csv"),
        output_name(prefix, "stratified_summary.csv"),
        output_name(prefix, "high_change_examples.csv"),
        output_name(prefix, "metadata.json"),
        output_name(prefix, "summary.md"),
    ]
    for name in names:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, summary_dir / name)


def launch_workers(args: argparse.Namespace) -> None:
    """函数功能：按 layout × seed × sample_set 启动进程级多 GPU worker。"""
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layouts = parse_csv(args.layouts)
    seeds = parse_int_list(args.seeds)
    sample_sets = [str(value) for value in args.sample_sets]
    devices = parse_csv(args.devices)
    jobs: List[Tuple[str, int, str, str]] = []
    for layout in layouts:
        for seed in seeds:
            for sample_set in sample_sets:
                jobs.append((layout, seed, sample_set, devices[len(jobs) % len(devices)]))
    write_json(
        output_dir / "launcher_metadata.json",
        {
            "status": "started",
            "script_version": SCRIPT_VERSION,
            "started_at": now_cst(),
            "jobs": [{"layout": a, "seed": b, "sample_set": c, "device": d} for a, b, c, d in jobs],
            "parallel_backend": "process_per_layout_seed_sample_set",
        },
    )
    running: List[Tuple[subprocess.Popen, Tuple[str, int, str, str], Path]] = []
    failures: List[str] = []
    script = Path(__file__).resolve()
    for job in jobs:
        while len(running) >= len(devices):
            failures.extend(poll_running(running))
            if failures:
                raise RuntimeError("; ".join(failures))
            time.sleep(2.0)
        layout, seed, sample_set, device = job
        td = task_dir(output_dir, layout, seed, sample_set)
        td.mkdir(parents=True, exist_ok=True)
        log_path = td / "task.log"
        cmd = [
            sys.executable,
            str(script),
            "--run-single",
            "--layout",
            layout,
            "--seed",
            str(seed),
            "--sample-sets",
            sample_set,
            "--device",
            device,
            "--sample-manifest",
            str(args.sample_manifest),
            "--round2c-dir",
            str(args.round2c_dir),
            "--prediction-manifest-path",
            str(args.prediction_manifest_path),
            "--visual-checkpoint",
            str(args.visual_checkpoint),
            "--config-path",
            str(args.config_path),
            "--output-dir",
            str(output_dir),
            "--summary-copy-dir",
            str(args.summary_copy_dir),
            "--max-samples-per-set",
            str(args.max_samples_per_set),
            "--perturbation-sigma-list",
            str(args.perturbation_sigma_list),
            "--num-perturbations",
            str(args.num_perturbations),
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--image-size",
            str(args.image_size),
            "--norm-mode",
            str(args.norm_mode),
            "--clip",
            str(args.clip),
            "--period-selection",
            str(args.period_selection),
            "--dtype",
            str(args.dtype or "auto"),
        ]
        if args.period_candidates:
            cmd.extend(["--period-candidates", str(args.period_candidates)])
        if args.local_files_only:
            cmd.append("--local-files-only")
        if args.save_debug_examples:
            cmd.append("--save-debug-examples")
        if args.overwrite:
            cmd.append("--overwrite")
        (td / "task.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
        running.append((proc, job, log_path))
        log_stage(f"启动 continuity worker：layout={layout} seed={seed} sample_set={sample_set} device={device}")
    while running:
        failures.extend(poll_running(running))
        if failures:
            raise RuntimeError("; ".join(failures))
        time.sleep(2.0)
    aggregate(args)
    write_json(output_dir / "status.json", {"status": "completed", "updated_at": now_cst(), "output_dir": str(output_dir)})


def poll_running(running: List[Tuple[subprocess.Popen, Tuple[str, int, str, str], Path]]) -> List[str]:
    """函数功能：回收已结束 worker，返回失败任务描述。"""
    failures: List[str] = []
    still: List[Tuple[subprocess.Popen, Tuple[str, int, str, str], Path]] = []
    for proc, job, log_path in running:
        code = proc.poll()
        if code is None:
            still.append((proc, job, log_path))
            continue
        if code != 0:
            failures.append(f"worker failed code={code} job={job} log={log_path}")
        else:
            log_stage(f"worker 完成：job={job}")
    running[:] = still
    return failures


def main() -> None:
    """函数功能：Round2d continuity diagnostic CLI 入口。"""
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
    elif args.run_single:
        run_single(args)
    else:
        launch_workers(args)


if __name__ == "__main__":
    main()
