#!/usr/bin/env python3
"""
文件功能：
    以 streaming / shard-aware 方式训练和评估 Stage 1 Online Visual Router。

核心约束：
    - 顶层路线固定为 `x -> pseudo image -> frozen ViT -> router`；
    - 伪图像 tensor 和 ViT embedding 只在当前 batch 的运行时存在，不写 `.npy`，
      不生成长期 embedding cache；
    - `StandardScaler` 只在 vali embedding 流上 `partial_fit`，test 只做 forward；
    - `fusion_huber_kl` 训练只在 vali split 读取专家 prediction cache 作为监督，
      不把专家误差、oracle label 或未来 y 作为 router 输入；
    - 输出文件名兼容 `evaluate_soft_fusion_calibration.py`，特别是
      `visual_router_predictions.csv` 和标准 `visual_router_metadata.json`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from transformers import ViTModel


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config.training import TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402
from visual_router_experiments.common.vit_embedding_utils import (  # noqa: E402
    EMBEDDING_VERSION,
    batch_required_pairs,
    build_required_index,
    make_default_period_candidates,
    make_pseudo_images,
    parse_period_candidate_arg,
    pool_vit_outputs,
    resolve_dtype,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_PREDICTION_MANIFEST_PATH,
    EPS,
    MODEL_COLUMNS,
    VisualMLPRouter,
    add_soft_fusion_metrics,
    compare_with_baselines,
    frame_to_markdown,
    load_labels,
    load_prediction_lookup,
    load_prediction_tensors_for_samples,
    make_class_weight,
    summarize_hard_predictions,
    summarize_selected_model_counts,
    summarize_soft_fusion,
    validate_training_args,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import (  # noqa: E402
    DEFAULT_CONFIG,
    _timer_start,
    _timer_stop,
    build_train_args,
    display_time,
    load_data_config,
    make_embedding_manifest_row,
    mode_from_split,
    now_token,
    resolve_device,
)


STREAMING_ONLINE_ROUTER_VERSION = "visual_router_mlp_v3_fusion_huber_kl_online_vit_streaming"


def parse_args() -> argparse.Namespace:
    """函数功能：解析 streaming online router 参数。"""
    parser = argparse.ArgumentParser(description="Train Stage 1 Online Visual Router with streaming ViT batches.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="window oracle labels CSV。")
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST_PATH, help="五专家 prediction cache manifest CSV。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config；仅用于读取历史窗口 x。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="oracle label 和辅助误差分布使用的指标。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认生成 streaming run 目录。")
    parser.add_argument("--router-mode", choices=["classification", "fusion_huber_kl"], default="fusion_huber_kl", help="router 训练目标。")
    parser.add_argument("--huber-beta", type=float, default=0.1, help="fusion_huber_kl SmoothL1 beta。")
    parser.add_argument("--kl-tau", type=float, default=0.1, help="soft oracle temperature。")
    parser.add_argument("--lambda-kl", type=float, default=0.01, help="KL 辅助损失权重。")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden dimension。")
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout。")
    parser.add_argument("--epochs", type=int, default=1, help="streaming 训练 epoch；每个 epoch 会重新运行 vali ViT 前向。")
    parser.add_argument("--batch-size", type=int, default=32, help="router 参数更新 batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay。")
    parser.add_argument("--seed", type=int, default=16, help="随机种子。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ViT 和 router 运行设备。")
    parser.add_argument("--skip-soft-fusion", action="store_true", help="只写 hard top-1，不计算 raw soft fusion。")
    parser.add_argument("--encoder-name", default="google/vit-base-patch16-224", help="Hugging Face ViT encoder 名称或本地路径。")
    parser.add_argument("--variant", choices=["variant_a_3view", "variant_b_top3fold"], default="variant_a_3view", help="伪图像 variant。")
    parser.add_argument("--pooling", choices=["cls", "mean_patch", "pooler"], default="cls", help="ViT 输出聚合方式。")
    parser.add_argument("--normalization-preset", default="hf_vit_0_5", help="encoder 前 normalization 口径。")
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="在线 ViT 前向 batch size。")
    parser.add_argument("--image-size", type=int, default=224, help="伪图像尺寸。")
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux", help="历史窗口 normalization 口径。")
    parser.add_argument("--pixel-mode", choices=["vision"], default="vision", help="pixel 映射口径。")
    parser.add_argument("--clip", type=float, default=5.0, help="视觉 pixel 映射前截断阈值。")
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates", help="full-scale 默认固定候选周期，减少同步。")
    parser.add_argument("--period-candidates", default=None, help="逗号分隔候选周期；只在 fixed_candidates 下使用。")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto", help="encoder 前向 dtype；CPU 强制 fp32。")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地 Hugging Face cache，不联网下载。")
    parser.add_argument("--stream-shard-index", type=int, default=0, help="按 sample_key 稳定切分后的当前 streaming shard。")
    parser.add_argument("--stream-shard-count", type=int, default=1, help="streaming shard 总数；dry-run 可用来验证多 shard 口径。")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="dry-run 限制每个 split 最多样本数；None 表示不限制。")
    parser.add_argument("--chunk-read-rows", type=int, default=200000, help="预留的大 CSV chunk 读取行数，metadata 记录用。")
    parser.add_argument("--status-update-interval", type=int, default=50, help="每处理多少个 embedding batch 更新一次 status.json。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束打印多少行预测预览。")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="从已有 router checkpoint 继续训练或 eval-only。")
    parser.add_argument("--train-only", action="store_true", help="只训练并保存 checkpoint，不执行 test streaming 预测。")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """函数功能：固定主要随机源，便于 dry-run 复核。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_status(output_dir: Path, status: Mapping[str, object]) -> None:
    """函数功能：写出长任务 status.json，供断点检查和外部监控读取。"""
    payload = dict(status)
    payload["updated_at"] = display_time()
    payload["output_dir"] = str(output_dir)
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_path_string(path: Path) -> str:
    """函数功能：把输入路径规整成 checkpoint 校验使用的稳定字符串。"""
    return str(path.expanduser().resolve())


def scaler_to_state(scaler: StandardScaler) -> Dict[str, object]:
    """
    函数功能：
        将 StandardScaler 转为 torch checkpoint 可稳定保存的状态字典。

    说明：
        不直接依赖 pickle 保存完整对象，后续恢复时只重建 transform 所需字段，
        降低 sklearn 版本差异带来的不透明风险。
    """
    required_attrs = ["mean_", "scale_", "var_", "n_features_in_"]
    missing = [name for name in required_attrs if not hasattr(scaler, name)]
    if missing:
        raise ValueError(f"scaler 尚未 fit，缺少字段：{missing}")
    state: Dict[str, object] = {
        "mean_": np.asarray(scaler.mean_, dtype=np.float64),
        "scale_": np.asarray(scaler.scale_, dtype=np.float64),
        "var_": np.asarray(scaler.var_, dtype=np.float64),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": getattr(scaler, "n_samples_seen_", None),
    }
    if hasattr(scaler, "feature_names_in_"):
        state["feature_names_in_"] = np.asarray(scaler.feature_names_in_)
    return state


def scaler_from_state(state: Mapping[str, object]) -> StandardScaler:
    """函数功能：从 checkpoint 状态重建 StandardScaler，用于 resume 后直接 transform。"""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var_"], dtype=np.float64)
    scaler.n_features_in_ = int(state["n_features_in_"])
    n_samples_seen = state.get("n_samples_seen_")
    if n_samples_seen is not None:
        scaler.n_samples_seen_ = np.asarray(n_samples_seen) if isinstance(n_samples_seen, (list, tuple, np.ndarray)) else n_samples_seen
    if "feature_names_in_" in state:
        scaler.feature_names_in_ = np.asarray(state["feature_names_in_"])
    return scaler


def build_resume_signature(args: argparse.Namespace, period_candidate_values: Optional[Sequence[int]]) -> Dict[str, object]:
    """
    函数功能：
        构造影响 router 参数形状、训练目标、输入 embedding 和数据切分的严格校验签名。
    """
    return {
        "config_name": None,
        "model_columns": list(MODEL_COLUMNS),
        "router_mode": args.router_mode,
        "metric": args.metric,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "embedding_metadata": {
            "embedding_version": f"{EMBEDDING_VERSION}_online_streaming",
            "encoder_name": args.encoder_name,
            "variant": args.variant,
            "pooling": args.pooling,
            "normalization_preset": args.normalization_preset,
            "image_size": int(args.image_size),
            "norm_mode": args.norm_mode,
            "pixel_mode": args.pixel_mode,
            "clip": float(args.clip),
            "period_selection": args.period_selection,
            "period_candidates_arg": args.period_candidates,
            "period_candidates": [int(value) for value in period_candidate_values] if period_candidate_values is not None else None,
            "dtype_arg": args.dtype,
        },
        "stream_shard_index": int(args.stream_shard_index),
        "stream_shard_count": int(args.stream_shard_count),
        "labels_path": stable_path_string(args.labels_path),
        "prediction_manifest_path": stable_path_string(args.prediction_manifest_path),
        "config_path": stable_path_string(args.config_path),
    }


def load_checkpoint(path: Path) -> Dict[str, object]:
    """函数功能：加载 checkpoint，并显式关闭 weights_only 限制以读取 numpy/scaler 状态。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 resume checkpoint：{path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_checkpoint_matches(
    *,
    checkpoint: Mapping[str, object],
    expected_signature: Mapping[str, object],
    config_name: str,
) -> None:
    """
    函数功能：
        对 checkpoint 与当前命令做严格签名校验，防止不同 config、embedding 口径或数据路径误接。
    """
    signature = dict(expected_signature)
    signature["config_name"] = str(config_name)
    fields = [
        "config_name",
        "model_columns",
        "router_mode",
        "metric",
        "hidden_dim",
        "dropout",
        "lr",
        "weight_decay",
        "huber_beta",
        "kl_tau",
        "lambda_kl",
        "embedding_metadata",
        "stream_shard_index",
        "stream_shard_count",
        "labels_path",
        "prediction_manifest_path",
        "config_path",
    ]
    mismatches = []
    for field in fields:
        actual = checkpoint.get(field)
        expected = signature[field]
        if actual != expected:
            mismatches.append({"field": field, "checkpoint": actual, "current": expected})
    if mismatches:
        raise ValueError(f"resume checkpoint 与当前命令不兼容：{json.dumps(mismatches, ensure_ascii=False, indent=2)}")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """函数功能：把 optimizer state 中的 tensor 移到当前训练设备，避免 resume 后设备不一致。"""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def checkpoint_filename(config_name: str, completed_epochs: int) -> str:
    """函数功能：生成不含特殊字符的 config 级 checkpoint 文件名。"""
    safe_config = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(config_name))
    return f"router_{safe_config}_epoch_{int(completed_epochs):04d}.pt"


def save_checkpoint(
    *,
    output_dir: Path,
    config_name: str,
    router: VisualMLPRouter,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler,
    completed_epochs: int,
    args: argparse.Namespace,
    period_candidate_values: Optional[Sequence[int]],
    epoch_summaries: Sequence[Mapping[str, float]],
    scaler_batches: int,
    scaler_samples: int,
) -> Path:
    """
    函数功能：
        在每个 epoch 结束保存可续训 checkpoint，并维护 config 级 latest checkpoint。
    """
    signature = build_resume_signature(args, period_candidate_values)
    signature["config_name"] = str(config_name)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        **signature,
        "checkpoint_version": "stage1_streaming_router_checkpoint_v1",
        "router_state_dict": router.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state": scaler_to_state(scaler),
        "completed_epochs": int(completed_epochs),
        "epoch_summaries": [dict(row) for row in epoch_summaries],
        "scaler_batches": int(scaler_batches),
        "scaler_samples": int(scaler_samples),
        "saved_at": display_time(),
    }
    checkpoint_path = checkpoint_dir / checkpoint_filename(str(config_name), int(completed_epochs))
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)
    latest_path = checkpoint_dir / f"latest_{''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in str(config_name))}.pt"
    latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
    torch.save(payload, latest_tmp)
    latest_tmp.replace(latest_path)
    (checkpoint_dir / "latest_checkpoint_index.json").write_text(
        json.dumps(
            {
                "config_name": str(config_name),
                "completed_epochs": int(completed_epochs),
                "checkpoint_path": str(checkpoint_path),
                "latest_checkpoint_path": str(latest_path),
                "updated_at": display_time(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return latest_path


def cleanup_output_files(output_dir: Path, *, resume: bool, train_only: bool) -> None:
    """
    函数功能：
        清理本次会被重新生成的 CSV/summary 文件；resume 时绝不删除 checkpoints。
    """
    stale_names = [
        "visual_router_predictions.csv",
        "visual_router_soft_fusion_predictions.csv",
        "visual_router_summary.csv",
        "visual_router_soft_fusion_summary.csv",
        "visual_router_selected_model_counts.csv",
        "visual_router_comparison.csv",
        "visual_router_streaming_summary.md",
    ]
    if not resume:
        stale_names.extend(["online_embedding_manifest.csv", "online_embedding_latency_summary.csv"])
    if train_only:
        stale_names = [name for name in stale_names if name.startswith("visual_router_")]
    for stale_name in stale_names:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


def filter_stream_shard(labels_df: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    """函数功能：按 sample_key 稳定切分 streaming shard。"""
    if shard_count <= 0:
        raise ValueError("--stream-shard-count 必须为正整数")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--stream-shard-index 必须落在 [0, stream_shard_count)")
    if shard_count == 1:
        return labels_df.copy().reset_index(drop=True)
    ordered = labels_df.sort_values("sample_key").reset_index(drop=True)
    mask = (np.arange(len(ordered)) % int(shard_count)) == int(shard_index)
    return ordered.loc[mask].reset_index(drop=True)


def limit_samples_per_split(labels_df: pd.DataFrame, max_samples_per_split: Optional[int]) -> pd.DataFrame:
    """函数功能：为 full-scale dry-run 截取每个 split 的前若干 sample。"""
    if max_samples_per_split is None:
        return labels_df.copy().reset_index(drop=True)
    if max_samples_per_split <= 0:
        raise ValueError("--max-samples-per-split 必须为正整数")
    rows = []
    for (_, split), group in labels_df.sort_values("sample_key").groupby(["config_name", "split"], sort=True):
        rows.append(group.head(int(max_samples_per_split)))
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def windows_from_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：从 labels DataFrame 生成需要在线 embedding 的唯一窗口清单。"""
    required_cols = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    windows_df = labels_df[required_cols].drop_duplicates().reset_index(drop=True)
    if windows_df["sample_key"].duplicated().any():
        dup = windows_df.loc[windows_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"labels 中 sample_key 不唯一，示例：{dup}")
    return windows_df


def build_vit_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> ViTModel:
    """函数功能：构建冻结 ViT encoder，并处理本地 cache / dtype 口径。"""
    model = ViTModel.from_pretrained(
        args.encoder_name,
        local_files_only=bool(args.local_files_only),
        add_pooling_layer=args.pooling == "pooler",
    )
    model.eval().to(device=device)
    if dtype == torch.float16:
        model = model.half()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def is_retryable_model_load_error(exc: Exception) -> bool:
    """
    函数功能：识别 ViT/Hugging Face 模型加载阶段的临时错误。

    说明：
        full-scale 长任务可能在首次拉模型、读取远端缓存或访问本地镜像时遇到
        429/503、连接超时、短暂 I/O 抖动。这里仅把明显的临时错误标记为可重试，
        避免把配置错误误吞掉。
    """
    message = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = [
        "429",
        "503",
        "timeout",
        "timed out",
        "connection",
        "temporar",
        "readerror",
        "service unavailable",
        "too many requests",
        "httperror",
        "connectionreset",
    ]
    return any(marker in message for marker in retry_markers)


def load_vit_model_with_retry(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, max_attempts: int = 3) -> ViTModel:
    """函数功能：带有限指数退避的 ViT 加载，降低网络抖动对长任务的影响。"""
    last_exc: Optional[Exception] = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            return build_vit_model(args, device, dtype)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= int(max_attempts) or not is_retryable_model_load_error(exc):
                raise
            sleep_seconds = float(min(30.0, 2.0 ** (attempt - 1)))
            print(
                f"[retry] ViT 加载失败，准备重试 {attempt + 1}/{max_attempts}，"
                f"sleep={sleep_seconds:.1f}s，error={repr(exc)}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def resolve_period_candidates(args: argparse.Namespace, history_length: int) -> Optional[List[int]]:
    """函数功能：解析 fixed candidate 周期列表，并写入 metadata。"""
    values = parse_period_candidate_arg(args.period_candidates)
    if args.period_selection == "fixed_candidates" and values is None:
        values = [int(value) for value in make_default_period_candidates(history_length, device=torch.device("cpu")).tolist()]
    return values


def iter_online_embedding_batches(
    *,
    windows_df: pd.DataFrame,
    data_config,
    vit_model: ViTModel,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    period_candidate_values: Optional[Sequence[int]],
) -> Iterator[Tuple[pd.DataFrame, np.ndarray, List[Dict[str, object]]]]:
    """
    函数功能：
        流式读取 Quito 历史窗口，生成当前 batch 的 ViT embedding。

    输出：
        batch_manifest_df: 当前 batch 的 embedding metadata，不含 embedding_path；
        embeddings: 当前 batch 的 float32 embedding 矩阵；
        latency_rows: 当前 batch 的 imageization / ViT 前向耗时。

    关键约束：
        `embeddings` 只在调用方处理当前 batch 时存在；本函数不保存 `.npy`，
        也不返回全量 `sample_key -> embedding` 字典。
    """
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"].astype(str), windows_df["config_name"].astype(str)))

    for split in sorted(windows_df["split"].astype(str).unique()):
        datasets = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=mode_from_split(str(split)),
            cleanup=False,
            concat=False,
        )
        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
            item_ids = sorted(
                item_id
                for req_split, req_dataset, item_id in required_index
                if req_split == split and req_dataset == dataset_name
            )
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
                        # 只读取历史窗口 x；不访问未来 y、oracle 或专家误差作为输入。
                        x_window = item_dataset.data[int(channel_id), window_start:window_end, :]
                        if x_window.shape[0] != int(data_config.seq_len):
                            raise ValueError(f"历史窗口长度不完整：sample_key={sample_key} shape={x_window.shape}")
                        x_windows.append(x_window)
                        sample_keys.append(str(sample_key))
                        channel_ids.append(int(channel_id))
                        window_indices.append(int(window_index))

                    x_cpu = torch.from_numpy(np.stack(x_windows, axis=0)).to(dtype=torch.float32)
                    with torch.inference_mode():
                        image_start = _timer_start(device)
                        pixel_values = make_pseudo_images(
                            x_cpu,
                            variant=args.variant,
                            norm_mode=args.norm_mode,
                            pixel_mode=args.pixel_mode,
                            clip=float(args.clip),
                            image_size=int(args.image_size),
                            device=device,
                            dtype=dtype,
                            normalization_preset=args.normalization_preset,
                            period_selection=args.period_selection,
                            period_candidate_values=period_candidate_values,
                        )
                        image_ms = _timer_stop(image_start, device)

                        forward_start = _timer_start(device)
                        outputs = vit_model(pixel_values=pixel_values)
                        embeddings = pool_vit_outputs(outputs, args.pooling)
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
                            make_embedding_manifest_row(
                                sample_key=sample_key,
                                config_name=str(config_by_key[sample_key]),
                                split=str(split),
                                dataset_name=str(dataset_name),
                                item_id=int(item_id),
                                channel_id=int(channel_ids[row_idx]),
                                window_index=int(window_indices[row_idx]),
                                history_length=int(data_config.seq_len),
                                embedding_dim=int(embeddings_cpu.shape[1]),
                                args=args,
                            )
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
                    yield pd.DataFrame(rows), embeddings_cpu.astype(np.float32), latency_rows


def train_on_stream_batch(
    *,
    router: VisualMLPRouter,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler,
    batch_manifest_df: pd.DataFrame,
    embeddings: np.ndarray,
    labels_by_key: Mapping[str, Mapping[str, object]],
    prediction_lookup: Optional[Mapping[Tuple[str, str], Dict[str, object]]],
    args: SimpleNamespace,
    device: torch.device,
    class_weight: torch.Tensor,
) -> Dict[str, float]:
    """函数功能：用一个 streaming embedding batch 更新 router 参数。"""
    sample_keys = batch_manifest_df["sample_key"].astype(str).tolist()
    labels_batch = [labels_by_key[key] for key in sample_keys]
    x_scaled = scaler.transform(embeddings).astype(np.float32)
    losses: List[float] = []
    huber_losses: List[float] = []
    kl_losses: List[float] = []
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    huber_criterion = torch.nn.SmoothL1Loss(beta=float(args.huber_beta))

    if args.router_mode == "classification":
        targets = np.asarray([MODEL_COLUMNS.index(str(row["oracle_model"])) for row in labels_batch], dtype=np.int64)
        for start in range(0, len(sample_keys), int(args.batch_size)):
            stop = start + int(args.batch_size)
            batch_x = torch.from_numpy(x_scaled[start:stop]).to(device=device)
            batch_y = torch.from_numpy(targets[start:stop]).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(router(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    else:
        if prediction_lookup is None:
            raise ValueError("fusion_huber_kl 需要 prediction_lookup")
        y_pred, y_true, expert_errors = load_prediction_tensors_for_samples(sample_keys, prediction_lookup, error_metric=str(args.metric))
        soft_oracle = torch.softmax(-torch.from_numpy(expert_errors) / float(args.kl_tau), dim=1).to(dtype=torch.float32)
        for start in range(0, len(sample_keys), int(args.batch_size)):
            stop = start + int(args.batch_size)
            batch_x = torch.from_numpy(x_scaled[start:stop]).to(device=device)
            batch_pred = torch.from_numpy(y_pred[start:stop]).to(device=device)
            batch_true = torch.from_numpy(y_true[start:stop]).to(device=device)
            batch_q = soft_oracle[start:stop].to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(batch_x)
            weights = torch.softmax(logits, dim=1)
            weight_shape = (weights.shape[0], weights.shape[1], *([1] * (batch_pred.ndim - 2)))
            fused_pred = (weights.view(weight_shape) * batch_pred).sum(dim=1)
            huber_loss = huber_criterion(fused_pred, batch_true)
            kl_loss = F.kl_div(torch.log_softmax(logits, dim=1), batch_q, reduction="batchmean")
            loss = huber_loss + float(args.lambda_kl) * kl_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            huber_losses.append(float(huber_loss.detach().cpu().item()))
            kl_losses.append(float(kl_loss.detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)),
        "huber_loss": float(np.mean(huber_losses)) if huber_losses else np.nan,
        "kl_loss": float(np.mean(kl_losses)) if kl_losses else np.nan,
    }


def predict_stream_batch(
    *,
    router: VisualMLPRouter,
    scaler: StandardScaler,
    batch_manifest_df: pd.DataFrame,
    embeddings: np.ndarray,
    labels_by_key: Mapping[str, Mapping[str, object]],
    router_name: str,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：对一个 test streaming batch 输出 hard top-1 router 预测行。"""
    x_scaled = scaler.transform(embeddings).astype(np.float32)
    router.eval()
    with torch.inference_mode():
        logits = router(torch.from_numpy(x_scaled).to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    selected_indices = weights.argmax(axis=1)
    weight_entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
    normalized_weight_entropy = weight_entropy / np.log(len(MODEL_COLUMNS))
    max_weight = weights.max(axis=1)

    rows: List[Dict[str, object]] = []
    for row_idx, row in enumerate(batch_manifest_df.itertuples(index=False)):
        sample_key = str(row.sample_key)
        label_row = labels_by_key[sample_key]
        selected_model = MODEL_COLUMNS[int(selected_indices[row_idx])]
        output_row: Dict[str, object] = {
            "router_name": router_name,
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
            "weight_entropy": float(weight_entropy[row_idx]),
            "normalized_weight_entropy": float(normalized_weight_entropy[row_idx]),
            "max_weight": float(max_weight[row_idx]),
        }
        for model_idx, model_name in enumerate(MODEL_COLUMNS):
            output_row[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
        rows.append(output_row)
    return pd.DataFrame(rows)


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    """函数功能：追加写 CSV，首批自动写 header。"""
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_latency(output_dir: Path, latency_rows: List[Dict[str, object]], phase: str) -> None:
    """函数功能：追加写 embedding latency，并标注当前 streaming 阶段。"""
    latency_df = pd.DataFrame(latency_rows)
    if latency_df.empty:
        return
    latency_df["phase"] = str(phase)
    append_csv(output_dir / "online_embedding_latency_summary.csv", latency_df)


def summarize_csv_outputs(output_dir: Path, metric: str, labels_path: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """函数功能：从 streaming 写出的 CSV 生成兼容旧入口的 summary/comparison。"""
    hard_pred_df = pd.read_csv(output_dir / "visual_router_predictions.csv")
    hard_summary_df = summarize_hard_predictions(hard_pred_df)
    selected_counts_df = summarize_selected_model_counts(hard_pred_df)
    hard_summary_df.to_csv(output_dir / "visual_router_summary.csv", index=False)
    selected_counts_df.to_csv(output_dir / "visual_router_selected_model_counts.csv", index=False)

    soft_summary_df: Optional[pd.DataFrame] = None
    soft_path = output_dir / "visual_router_soft_fusion_predictions.csv"
    if soft_path.exists():
        soft_pred_df = pd.read_csv(soft_path)
        soft_summary_df = summarize_soft_fusion(soft_pred_df)
        soft_summary_df.to_csv(output_dir / "visual_router_soft_fusion_summary.csv", index=False)

    comparison_df = compare_with_baselines(output_dir, labels_path, hard_summary_df, soft_summary_df, metric)
    comparison_df.to_csv(output_dir / "visual_router_comparison.csv", index=False)
    return hard_summary_df, soft_summary_df, selected_counts_df, comparison_df


def write_summary_md(
    *,
    output_dir: Path,
    hard_summary: pd.DataFrame,
    soft_summary: Optional[pd.DataFrame],
    selected_counts: pd.DataFrame,
    comparison_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写出中文 streaming 摘要。"""
    lines = [
        "# Stage 1 Streaming Online Visual Router",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 口径",
        "",
        "- 路径：`x -> pseudo image -> frozen ViT -> router`。",
        "- 伪图像 tensor 和 ViT embedding 均为 batch 运行时对象，不保存 `.npy`，不建立长期 embedding cache。",
        "- `StandardScaler.partial_fit` 只遍历 vali embedding；test 只用于 forward 和评估。",
        f"- streaming shard: `{metadata['stream_shard_index']}/{metadata['stream_shard_count']}`。",
        f"- encoder: `{metadata['embedding_metadata']['encoder_name']}`，period_selection: `{metadata['embedding_metadata']['period_selection']}`。",
        "",
        "## Hard Top-1 Summary",
        "",
        frame_to_markdown(hard_summary),
        "",
    ]
    if soft_summary is not None:
        lines.extend(["## Soft Fusion Summary", "", frame_to_markdown(soft_summary), ""])
    lines.extend(["## Top-1 选中专家分布", "", frame_to_markdown(selected_counts), ""])
    lines.extend(["## Baseline Comparison", "", frame_to_markdown(comparison_df.head(24)), ""])
    lines.extend(
        [
            "## 输出文件",
            "",
            f"- `visual_router_predictions.csv`: `{output_dir / 'visual_router_predictions.csv'}`",
            f"- `visual_router_summary.csv`: `{output_dir / 'visual_router_summary.csv'}`",
            f"- `visual_router_metadata.json`: `{output_dir / 'visual_router_metadata.json'}`",
            f"- `status.json`: `{output_dir / 'status.json'}`",
            "",
        ]
    )
    (output_dir / "visual_router_streaming_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 streaming online router 全流程。"""
    args = parse_args()
    if int(args.epochs) < 0:
        raise ValueError("--epochs 必须 >= 0；resume/eval-only 可使用 --epochs 0")
    train_args = build_train_args(args)
    validate_training_args(train_args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_online_visual_router_streaming_96_48_s"
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    write_status(
        output_dir,
        {
            "status": "running",
            "phase": "init",
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "latest_checkpoint_path": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "completed_epochs": int(resume_checkpoint["completed_epochs"]) if resume_checkpoint is not None else 0,
        },
    )

    labels_df = load_labels(args.labels_path, args.metric)
    labels_df = filter_stream_shard(labels_df, int(args.stream_shard_index), int(args.stream_shard_count))
    labels_df = limit_samples_per_split(labels_df, args.max_samples_per_split)
    windows_df = windows_from_labels(labels_df)
    data_config = load_data_config(args.config_path)
    period_candidate_values = resolve_period_candidates(args, int(data_config.seq_len))
    vit_model = load_vit_model_with_retry(args, device, dtype)
    prediction_lookup = load_prediction_lookup(args.prediction_manifest_path) if (args.router_mode == "fusion_huber_kl" or not args.skip_soft_fusion) else None
    expected_signature = build_resume_signature(args, period_candidate_values)
    if resume_checkpoint is not None:
        config_names = sorted(labels_df["config_name"].astype(str).unique().tolist())
        if config_names != [str(resume_checkpoint.get("config_name"))]:
            raise ValueError(f"--resume-checkpoint 当前仅支持单 config 严格续训；labels config={config_names}, checkpoint config={resume_checkpoint.get('config_name')}")
    cleanup_output_files(output_dir, resume=resume_checkpoint is not None, train_only=bool(args.train_only))

    router_name = STREAMING_ONLINE_ROUTER_VERSION if args.router_mode == "fusion_huber_kl" else "visual_router_mlp_v1_classification_online_vit_streaming"
    config_metadata: List[Dict[str, object]] = []
    embedding_dim: Optional[int] = None
    total_embedding_batches = 0
    latest_checkpoint_path: Optional[Path] = Path(args.resume_checkpoint) if args.resume_checkpoint is not None else None

    for config_name, config_labels_df in labels_df.groupby("config_name", sort=True):
        config_windows_df = windows_df[windows_df["config_name"].astype(str) == str(config_name)].copy()
        vali_windows_df = config_windows_df[config_windows_df["split"] == "vali"].copy()
        test_windows_df = config_windows_df[config_windows_df["split"] == "test"].copy()
        config_labels_by_key = config_labels_df.set_index("sample_key").to_dict(orient="index")
        vali_labels_by_key = config_labels_df[config_labels_df["split"] == "vali"].set_index("sample_key").to_dict(orient="index")
        test_labels_by_key = config_labels_df[config_labels_df["split"] == "test"].set_index("sample_key").to_dict(orient="index")
        if vali_windows_df.empty or test_windows_df.empty:
            raise ValueError(f"config_name={config_name} 需要同时包含 vali/test")

        previous_completed_epochs = 0
        if resume_checkpoint is not None:
            assert_checkpoint_matches(checkpoint=resume_checkpoint, expected_signature=expected_signature, config_name=str(config_name))
            scaler = scaler_from_state(resume_checkpoint["scaler_state"])
            scaler_batches = int(resume_checkpoint.get("scaler_batches", 0))
            scaler_samples = int(resume_checkpoint.get("scaler_samples", 0))
            previous_completed_epochs = int(resume_checkpoint["completed_epochs"])
            embedding_dim = int(scaler.n_features_in_)
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "checkpoint_loaded",
                    "config_name": str(config_name),
                    "resume_checkpoint": str(args.resume_checkpoint),
                    "completed_epochs": previous_completed_epochs,
                    "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
                },
            )
        else:
            scaler = StandardScaler()
            scaler_batches = 0
            scaler_samples = 0
            for batch_manifest_df, embeddings, latency_rows in iter_online_embedding_batches(
                windows_df=vali_windows_df,
                data_config=data_config,
                vit_model=vit_model,
                args=args,
                device=device,
                dtype=dtype,
                period_candidate_values=period_candidate_values,
            ):
                scaler.partial_fit(embeddings)
                append_csv(output_dir / "online_embedding_manifest.csv", batch_manifest_df)
                append_latency(output_dir, latency_rows, "scaler_fit")
                embedding_dim = int(embeddings.shape[1])
                scaler_batches += 1
                scaler_samples += int(len(batch_manifest_df))
                total_embedding_batches += 1
            write_status(output_dir, {"status": "running", "phase": "scaler_fit_completed", "config_name": str(config_name), "scaler_samples": scaler_samples, "completed_epochs": 0, "latest_checkpoint_path": None})

        router = VisualMLPRouter(
            input_dim=int(scaler.n_features_in_),
            hidden_dim=int(args.hidden_dim),
            output_dim=len(MODEL_COLUMNS),
            dropout=float(args.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        if resume_checkpoint is not None:
            router.load_state_dict(resume_checkpoint["router_state_dict"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        class_weight = make_class_weight(
            [str(row["oracle_model"]) for row in vali_labels_by_key.values()],
            device=device,
        )

        epoch_summaries: List[Dict[str, float]] = [dict(row) for row in resume_checkpoint.get("epoch_summaries", [])] if resume_checkpoint is not None else []
        router.train()
        for local_epoch_idx in range(int(args.epochs)):
            global_epoch = previous_completed_epochs + local_epoch_idx + 1
            epoch_rows: List[Dict[str, float]] = []
            for batch_manifest_df, embeddings, latency_rows in iter_online_embedding_batches(
                windows_df=vali_windows_df,
                data_config=data_config,
                vit_model=vit_model,
                args=args,
                device=device,
                dtype=dtype,
                period_candidate_values=period_candidate_values,
            ):
                metrics = train_on_stream_batch(
                    router=router,
                    optimizer=optimizer,
                    scaler=scaler,
                    batch_manifest_df=batch_manifest_df,
                    embeddings=embeddings,
                    labels_by_key=vali_labels_by_key,
                    prediction_lookup=prediction_lookup,
                    args=train_args,
                    device=device,
                    class_weight=class_weight,
                )
                epoch_rows.append(metrics)
                append_latency(output_dir, latency_rows, f"train_epoch_{global_epoch}")
                total_embedding_batches += 1
                if total_embedding_batches % int(args.status_update_interval) == 0:
                    write_status(
                        output_dir,
                        {
                            "status": "running",
                            "phase": "training",
                            "config_name": str(config_name),
                            "epoch": int(global_epoch),
                            "current_epoch": int(global_epoch),
                            "completed_epochs": int(global_epoch - 1),
                            "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
                            "embedding_batches": total_embedding_batches,
                        },
                    )
            epoch_summary = {
                "epoch": float(global_epoch),
                "loss": float(np.nanmean([row["loss"] for row in epoch_rows])),
                "huber_loss": float(np.nanmean([row["huber_loss"] for row in epoch_rows])),
                "kl_loss": float(np.nanmean([row["kl_loss"] for row in epoch_rows])),
            }
            epoch_summaries.append(epoch_summary)
            latest_checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                config_name=str(config_name),
                router=router,
                optimizer=optimizer,
                scaler=scaler,
                completed_epochs=int(global_epoch),
                args=args,
                period_candidate_values=period_candidate_values,
                epoch_summaries=epoch_summaries,
                scaler_batches=scaler_batches,
                scaler_samples=scaler_samples,
            )
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "checkpoint_saved",
                    "config_name": str(config_name),
                    "epoch": int(global_epoch),
                    "current_epoch": int(global_epoch),
                    "completed_epochs": int(global_epoch),
                    "latest_checkpoint_path": str(latest_checkpoint_path),
                    "embedding_batches": total_embedding_batches,
                },
            )

        hard_rows_seen = 0
        if not args.train_only:
            router.eval()
            for batch_manifest_df, embeddings, latency_rows in iter_online_embedding_batches(
                windows_df=test_windows_df,
                data_config=data_config,
                vit_model=vit_model,
                args=args,
                device=device,
                dtype=dtype,
                period_candidate_values=period_candidate_values,
            ):
                append_csv(output_dir / "online_embedding_manifest.csv", batch_manifest_df)
                append_latency(output_dir, latency_rows, "test_predict")
                pred_df = predict_stream_batch(
                    router=router,
                    scaler=scaler,
                    batch_manifest_df=batch_manifest_df,
                    embeddings=embeddings,
                    labels_by_key=test_labels_by_key,
                    router_name=router_name,
                    device=device,
                )
                append_csv(output_dir / "visual_router_predictions.csv", pred_df)
                if not args.skip_soft_fusion:
                    assert prediction_lookup is not None
                    soft_df = add_soft_fusion_metrics(pred_df, prediction_lookup)
                    append_csv(output_dir / "visual_router_soft_fusion_predictions.csv", soft_df)
                hard_rows_seen += int(len(pred_df))
                total_embedding_batches += 1
        config_metadata.append(
            {
                "config_name": str(config_name),
                "router_mode": args.router_mode,
                "vali_sample_count": int(len(vali_windows_df)),
                "test_sample_count": int(len(test_windows_df)),
                "scaler_batches": int(scaler_batches),
                "scaler_samples": int(scaler_samples),
                "test_predictions": int(hard_rows_seen),
                "embedding_dim": int(scaler.n_features_in_),
                "epochs_requested_this_run": int(args.epochs),
                "previous_completed_epochs": int(previous_completed_epochs),
                "completed_epochs": int(previous_completed_epochs + int(args.epochs)),
                "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
                "streaming_epoch_summaries": epoch_summaries,
                "label_counts": {str(k): int(v) for k, v in config_labels_df[config_labels_df["split"] == "vali"]["oracle_model"].value_counts().reindex(MODEL_COLUMNS, fill_value=0).items()},
            }
        )

    latency_path = output_dir / "online_embedding_latency_summary.csv"
    latency_df = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame(columns=["imageization_per_window_ms", "encoder_forward_per_window_ms"])
    embedding_metadata = {
        "embedding_version": f"{EMBEDDING_VERSION}_online_streaming",
        "sample_count": int(len(windows_df)),
        "encoder_name": args.encoder_name,
        "variant": args.variant,
        "pooling": args.pooling,
        "normalization_preset": args.normalization_preset,
        "input_mode": "direct_pixel_values_online_streaming",
        "processor_do_rescale": "not_used",
        "image_size": int(args.image_size),
        "norm_mode": args.norm_mode,
        "pixel_mode": args.pixel_mode,
        "clip": float(args.clip),
        "period_selection": args.period_selection,
        "period_candidates_arg": args.period_candidates,
        "period_candidates": period_candidate_values,
        "device": str(device),
        "forward_dtype": str(dtype).replace("torch.", ""),
        "embedding_storage": "batch_runtime_only_not_saved",
        "saved_dtype": "not_saved",
        "embedding_dim": int(embedding_dim or 0),
        "splits": sorted(windows_df["split"].unique().tolist()),
        "config_names": sorted(windows_df["config_name"].unique().tolist()),
        "input_exclusions": ["future_y", "expert_errors_as_input", "oracle_model_as_input", "oracle_value_as_input"],
        "latency_mean": {
            "imageization_per_window_ms": float(latency_df["imageization_per_window_ms"].mean()) if not latency_df.empty else None,
            "encoder_forward_per_window_ms": float(latency_df["encoder_forward_per_window_ms"].mean()) if not latency_df.empty else None,
        },
    }
    run_metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "router_version": router_name,
        "router_mode": args.router_mode,
        "labels_path": str(args.labels_path),
        "prediction_manifest_path": str(args.prediction_manifest_path),
        "config_path": str(args.config_path),
        "local_files_only": bool(args.local_files_only),
        "metric": args.metric,
        "model_columns": MODEL_COLUMNS,
        "training_split": "vali",
        "evaluation_split": "test",
        "device": str(device),
        "seed": int(args.seed),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "epochs_semantics": "additional_epochs_this_run",
        "batch_size": int(args.batch_size),
        "embedding_batch_size": int(args.embedding_batch_size),
        "stream_shard_index": int(args.stream_shard_index),
        "stream_shard_count": int(args.stream_shard_count),
        "max_samples_per_split": args.max_samples_per_split,
        "chunk_read_rows": int(args.chunk_read_rows),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "soft_fusion_enabled": not bool(args.skip_soft_fusion),
        "train_only": bool(args.train_only),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "embedding_metadata": embedding_metadata,
        "config_metadata": config_metadata,
        "embedding_storage": "batch_runtime_only_not_saved",
        "pseudo_image_tensor_storage": "not_saved",
        "persistent_embedding_npy_written": False,
        "persistent_pseudo_image_tensor_written": False,
        "streaming_scaler_partial_fit": True,
        "input_exclusions": ["future_y_as_feature", "test_oracle_error_as_feature", "expert_error_as_feature"],
    }
    (output_dir / "visual_router_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "visual_router_online_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.train_only:
        completed_epochs = max(int(row["completed_epochs"]) for row in config_metadata) if config_metadata else 0
        write_status(
            output_dir,
            {
                "status": "completed",
                "phase": "train_only_done",
                "completed_epochs": completed_epochs,
                "current_epoch": completed_epochs,
                "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
            },
        )
        print(f"wrote train-only streaming checkpoint outputs to {output_dir}")
        print(f"latest_checkpoint_path={latest_checkpoint_path}")
    else:
        hard_summary_df, soft_summary_df, selected_counts_df, comparison_df = summarize_csv_outputs(output_dir, args.metric, args.labels_path)
        write_summary_md(
            output_dir=output_dir,
            hard_summary=hard_summary_df,
            soft_summary=soft_summary_df,
            selected_counts=selected_counts_df,
            comparison_df=comparison_df,
            metadata=run_metadata,
        )
        completed_epochs = max(int(row["completed_epochs"]) for row in config_metadata) if config_metadata else 0
        write_status(
            output_dir,
            {
                "status": "completed",
                "phase": "done",
                "router_predictions": int(hard_summary_df["sample_count"].sum()),
                "completed_epochs": completed_epochs,
                "current_epoch": completed_epochs,
                "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
            },
        )

        print(f"wrote streaming online visual router outputs to {output_dir}")
        print(hard_summary_df.to_string(index=False))
        if soft_summary_df is not None:
            print(soft_summary_df.to_string(index=False))
        pred_preview = pd.read_csv(output_dir / "visual_router_predictions.csv").head(int(args.print_rows))
        print(pred_preview.to_string(index=False))


if __name__ == "__main__":
    main()
