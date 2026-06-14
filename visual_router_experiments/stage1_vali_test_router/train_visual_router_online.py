#!/usr/bin/env python3
"""
文件功能：
    在线构建冻结 HF ViT embedding，并在同一次运行内训练 Stage 1 Visual Router。

设计约束：
    - 输入仍来自 sample_key 对齐的 Quito 历史窗口 x；
    - 在线执行 x -> pseudo image -> frozen ViT -> CLS embedding -> MLP router；
    - 不保存伪图像 tensor，不保存 ViT embedding npy；
    - 第一版允许把 vali/test embedding 暂存在运行内内存字典中，避免每个 epoch
      重复执行 ViT 前向；
    - MLP router、fusion_huber_kl/classification 训练、hard top-1 与 soft fusion
      评估复用 `train_visual_router.py`，避免复制训练评估逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import ViTModel


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config import AutoConfig  # noqa: E402
from quito.config.training import ModeType, TaskType  # noqa: E402
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
    MODEL_COLUMNS,
    add_soft_fusion_metrics,
    compare_with_baselines,
    frame_to_markdown,
    join_embeddings_and_labels,
    load_labels,
    load_prediction_lookup,
    predict_router_for_config,
    router_version_for_mode,
    set_seed,
    summarize_hard_predictions,
    summarize_selected_model_counts,
    summarize_soft_fusion,
    train_router_for_config,
    validate_training_args,
)


DEFAULT_CONFIG = (
    WORKSPACE
    / "quito"
    / "outputs"
    / "default_baseline"
    / "dlinear"
    / "96_48_S"
    / "seed_16"
    / "EVALUATE"
    / "ver_0"
    / "config.yaml"
)
DEFAULT_OFFLINE_REFERENCE_DIR = (
    RUN_OUTPUT_ROOT
    / "2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke"
)
ONLINE_ROUTER_VERSION = "visual_router_mlp_v2_fusion_huber_kl_online_vit"
EPS = 1e-8


def now_token() -> str:
    """函数功能：生成 run 目录时间戳，精确到微秒避免输出目录重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata 和 Markdown 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 online Visual Router 训练、评估和 ViT 前向参数。"""
    parser = argparse.ArgumentParser(description="Train Stage 1 Visual Router with online ViT embeddings.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="window oracle labels CSV。")
    parser.add_argument(
        "--prediction-manifest-path",
        type=Path,
        default=DEFAULT_PREDICTION_MANIFEST_PATH,
        help="五专家 prediction cache manifest CSV，用于 fusion loss 与 soft fusion 评估。",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config；仅复用 data_config 加载历史窗口 x。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="oracle label 和辅助误差分布使用的指标。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认生成 online visual router smoke 目录。")
    parser.add_argument(
        "--offline-reference-dir",
        type=Path,
        default=DEFAULT_OFFLINE_REFERENCE_DIR,
        help="可选离线 embedding 代表 router 目录；存在时写出 online/offline 指标对比。",
    )
    parser.add_argument(
        "--router-mode",
        choices=["classification", "fusion_huber_kl"],
        default="fusion_huber_kl",
        help="classification 保留旧 CE 分类 baseline；fusion_huber_kl 用融合预测误差训练权重。",
    )
    parser.add_argument("--huber-beta", type=float, default=0.1, help="fusion_huber_kl 主损失 SmoothL1Loss beta。")
    parser.add_argument("--kl-tau", type=float, default=0.1, help="soft oracle q_i=softmax(-error_i/tau) 的温度。")
    parser.add_argument("--lambda-kl", type=float, default=0.01, help="KL 辅助损失权重。")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden dimension。")
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout rate。")
    parser.add_argument("--epochs", type=int, default=300, help="训练 epoch 数。")
    parser.add_argument("--batch-size", type=int, default=32, help="router 训练 batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay。")
    parser.add_argument("--seed", type=int, default=16, help="随机种子。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ViT 和 router 运行设备。")
    parser.add_argument("--skip-soft-fusion", action="store_true", help="只评估 hard top-1，不读取预测数组。")
    parser.add_argument("--encoder-name", default="google/vit-base-patch16-224", help="Hugging Face ViT encoder 名称或本地路径。")
    parser.add_argument("--variant", choices=["variant_a_3view", "variant_b_top3fold"], default="variant_a_3view", help="伪图像 variant。")
    parser.add_argument("--pooling", choices=["cls", "mean_patch", "pooler"], default="cls", help="ViT 输出聚合方式。")
    parser.add_argument("--normalization-preset", default="hf_vit_0_5", help="encoder 前 normalization 口径。")
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="在线 ViT 前向 batch size。")
    parser.add_argument("--image-size", type=int, default=224, help="伪图像尺寸。")
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux", help="历史窗口 normalization 口径。")
    parser.add_argument("--pixel-mode", choices=["vision"], default="vision", help="pixel 映射口径。")
    parser.add_argument("--clip", type=float, default=5.0, help="视觉 pixel 映射前的对称截断阈值。")
    parser.add_argument(
        "--period-selection",
        choices=["fixed_candidates", "dynamic_fft_topk"],
        default="dynamic_fft_topk",
        help="online smoke 默认沿用旧代表 embedding 的动态 FFT 口径；1k 后续可切换 fixed_candidates 加速。",
    )
    parser.add_argument(
        "--period-candidates",
        default=None,
        help="逗号分隔候选周期；只有 --period-selection fixed_candidates 时使用。",
    )
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto", help="encoder 前向 dtype；CPU 会强制 fp32。")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地 Hugging Face 缓存，不联网下载。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束时打印多少行预测预览。")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析运行设备，auto 优先 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 PyTorch CUDA 不可用")
    return torch.device(device_arg)


def mode_from_split(split: str) -> ModeType:
    """函数功能：把 Stage 1 split 名称映射到 Quito ModeType。"""
    if split == "vali":
        return ModeType.VALID
    if split == "test":
        return ModeType.TEST
    raise ValueError(f"未知 split：{split}")


def load_data_config(config_path: Path):
    """函数功能：读取 Quito evaluate config，并返回数据配置对象。"""
    config = OmegaConf.load(config_path)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        rank=-1,
        world_size=-1,
        local_rank=-1,
    )
    del model_config, training_config
    return data_config


def load_required_windows_from_labels(labels_path: Path, metric: str) -> pd.DataFrame:
    """
    函数功能：
        从 oracle labels 中读取 online embedding 需要覆盖的唯一 sample_key 清单。

    说明：
        online 入口与离线 embedding 一样只用历史 x 构造视觉输入；这里读取 labels
        只是为了确定 120 smoke 的 sample_key 和元信息，不把 oracle label 作为 ViT
        或 router 的输入特征。
    """
    labels_df = load_labels(labels_path, metric)
    required_cols = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    windows_df = labels_df[required_cols].drop_duplicates().reset_index(drop=True)
    if windows_df.empty:
        raise ValueError(f"{labels_path} 中没有 metric={metric} 的样本")
    if windows_df["sample_key"].duplicated().any():
        dup_keys = windows_df.loc[windows_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"metric={metric} 的 sample_key 不唯一，示例：{dup_keys}")
    return windows_df


def _timer_start(device: torch.device) -> float:
    """函数功能：启动 latency 计时；CUDA 路径先同步以得到真实 wall time。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _timer_stop(start_time: float, device: torch.device) -> float:
    """函数功能：结束 latency 计时，返回毫秒。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start_time) * 1000.0


def make_embedding_manifest_row(
    *,
    sample_key: str,
    config_name: str,
    split: str,
    dataset_name: str,
    item_id: int,
    channel_id: int,
    window_index: int,
    history_length: int,
    embedding_dim: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """函数功能：构造运行内 embedding manifest 行；不包含 embedding_path。"""
    return {
        "embedding_version": f"{EMBEDDING_VERSION}_online_in_memory",
        "sample_key": sample_key,
        "config_name": config_name,
        "split": split,
        "dataset_name": dataset_name,
        "item_id": int(item_id),
        "channel_id": int(channel_id),
        "window_index": int(window_index),
        "history_length": int(history_length),
        "feature_type": "vit_embedding_online_in_memory",
        "variant": args.variant,
        "encoder_name": args.encoder_name,
        "pooling": args.pooling,
        "normalization_preset": args.normalization_preset,
        "input_mode": "direct_pixel_values_online",
        "processor_do_rescale": "not_used",
        "embedding_dim": int(embedding_dim),
        "dtype": "float32_in_memory",
        "finite": True,
    }


def build_online_embeddings(
    *,
    windows_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Mapping[str, np.ndarray], pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    函数功能：
        在线读取 Quito 历史窗口 x，前向冻结 ViT，并返回运行内 embedding 字典。

    输出：
        - feature_lookup: sample_key -> 1D float32 embedding；
        - embedding_manifest_df: 只记录样本和 encoder 口径，不含 `.npy` 路径；
        - latency_df: 每个 batch 的 imageization/encoder/in_memory_store latency；
        - embedding_metadata: 设备、dtype、样本覆盖和输入排除项。
    """
    data_config = load_data_config(args.config_path)
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"].astype(str), windows_df["config_name"].astype(str)))
    period_candidate_values = parse_period_candidate_arg(args.period_candidates)
    if args.period_selection == "fixed_candidates" and period_candidate_values is None:
        period_candidate_values = [
            int(value)
            for value in make_default_period_candidates(int(data_config.seq_len), device=torch.device("cpu")).tolist()
        ]

    # 与离线 embedding 入口保持一致：默认使用 CLS token，pooler 仅在显式请求时启用。
    model = ViTModel.from_pretrained(
        args.encoder_name,
        local_files_only=bool(args.local_files_only),
        add_pooling_layer=args.pooling == "pooler",
    )
    model.eval().to(device=device)
    if dtype == torch.float16:
        model = model.half()

    feature_lookup: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    latency_rows: List[Dict[str, object]] = []
    seen_keys: Set[str] = set()

    with torch.inference_mode():
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
                split_dataset_items = sorted(
                    item_id
                    for req_split, req_dataset, item_id in required_index
                    if req_split == split and req_dataset == dataset_name
                )
                if not split_dataset_items:
                    continue

                for item_id in split_dataset_items:
                    item_dataset = deepcopy(dataset)
                    item_dataset.select_user_data(int(item_id))
                    channel_count = int(item_dataset.data.shape[0])
                    required_for_item = required_index[(str(split), str(dataset_name), int(item_id))]

                    for pair_batch in batch_required_pairs(required_for_item, int(args.embedding_batch_size)):
                        x_windows = []
                        sample_keys = []
                        channel_ids = []
                        window_indices = []
                        for channel_id, window_index, sample_key in pair_batch:
                            if int(channel_id) >= channel_count:
                                raise ValueError(
                                    f"channel_id 越界：sample_key={sample_key} channel={channel_id} channel_count={channel_count}"
                                )
                            window_start = int(window_index)
                            window_end = window_start + int(data_config.seq_len)
                            # 只切历史 x 窗口；不访问未来 y、专家预测或 oracle error 作为输入。
                            x_window = item_dataset.data[int(channel_id), window_start:window_end, :]
                            if x_window.shape[0] != int(data_config.seq_len):
                                raise ValueError(f"历史窗口长度不完整：sample_key={sample_key} shape={x_window.shape}")
                            x_windows.append(x_window)
                            sample_keys.append(str(sample_key))
                            channel_ids.append(int(channel_id))
                            window_indices.append(int(window_index))

                        x_cpu = torch.from_numpy(np.stack(x_windows, axis=0)).to(dtype=torch.float32)
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
                        outputs = model(pixel_values=pixel_values)
                        embeddings = pool_vit_outputs(outputs, args.pooling)
                        forward_ms = _timer_stop(forward_start, device)

                        store_start = time.perf_counter()
                        embeddings_cpu = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
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

                            embedding = embeddings_cpu[row_idx].astype(np.float32, copy=True)
                            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                                raise ValueError(f"embedding 非法：sample_key={sample_key} shape={embedding.shape}")
                            feature_lookup[sample_key] = embedding
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
                                    embedding_dim=int(embedding.shape[0]),
                                    args=args,
                                )
                            )
                            seen_keys.add(sample_key)
                        store_ms = (time.perf_counter() - store_start) * 1000.0
                        latency_rows.append(
                            {
                                "split": str(split),
                                "dataset_name": str(dataset_name),
                                "item_id": int(item_id),
                                "batch_size": int(len(pair_batch)),
                                "imageization_ms": float(image_ms),
                                "encoder_forward_ms": float(forward_ms),
                                "in_memory_store_ms": float(store_ms),
                                "imageization_per_window_ms": float(image_ms / len(pair_batch)),
                                "encoder_forward_per_window_ms": float(forward_ms / len(pair_batch)),
                                "in_memory_store_per_window_ms": float(store_ms / len(pair_batch)),
                                "device": str(device),
                            }
                        )

    expected_keys = set(windows_df["sample_key"].astype(str))
    if seen_keys != expected_keys:
        raise ValueError(f"online embedding 覆盖不一致：missing={sorted(expected_keys - seen_keys)[:10]} extra={sorted(seen_keys - expected_keys)[:10]}")

    manifest_df = pd.DataFrame(rows).sort_values(
        ["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    ).reset_index(drop=True)
    latency_df = pd.DataFrame(latency_rows)
    embedding_metadata: Dict[str, object] = {
        "embedding_version": f"{EMBEDDING_VERSION}_online_in_memory",
        "sample_count": int(len(manifest_df)),
        "encoder_name": args.encoder_name,
        "variant": args.variant,
        "pooling": args.pooling,
        "pooling_detail": "last_hidden_state[:, 0] CLS token" if args.pooling == "cls" else args.pooling,
        "normalization_preset": args.normalization_preset,
        "input_mode": "direct_pixel_values_online",
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
        "embedding_storage": "in_memory_only",
        "saved_dtype": "not_saved",
        "embedding_dim": int(manifest_df["embedding_dim"].iloc[0]),
        "splits": sorted(manifest_df["split"].unique().tolist()),
        "config_names": sorted(manifest_df["config_name"].unique().tolist()),
        "input_exclusions": ["future_y", "expert_errors_as_input", "oracle_model_as_input", "oracle_value_as_input"],
    }
    return feature_lookup, manifest_df, latency_df, embedding_metadata


def validate_online_embedding_manifest(manifest_df: pd.DataFrame, expected_windows_df: pd.DataFrame) -> None:
    """函数功能：校验 online embedding manifest 字段、shape 和 sample_key 覆盖。"""
    required_cols = {
        "embedding_version",
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "feature_type",
        "variant",
        "encoder_name",
        "embedding_dim",
        "dtype",
        "finite",
    }
    missing = sorted(required_cols.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"online embedding manifest 缺少字段：{missing}")
    if manifest_df["sample_key"].duplicated().any():
        dup_keys = manifest_df.loc[manifest_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"online embedding manifest sample_key 重复，示例：{dup_keys}")
    if not manifest_df["finite"].astype(bool).all():
        raise ValueError("online embedding manifest 中存在 finite=False")
    if manifest_df["embedding_dim"].nunique() != 1:
        raise ValueError("online embedding_dim 不一致")
    if set(manifest_df["sample_key"].astype(str)) != set(expected_windows_df["sample_key"].astype(str)):
        raise ValueError("online embedding manifest 与 labels sample_key 集合不一致")


def build_train_args(args: argparse.Namespace) -> SimpleNamespace:
    """
    函数功能：
        构造可复用 `train_visual_router.py` 训练函数所需参数对象。

    说明：
        online 入口不通过 embedding manifest 读取 `.npy`，但训练超参和评估开关与离线
        入口保持同名字段，便于复用已有训练评估逻辑。
    """
    return SimpleNamespace(
        router_mode=args.router_mode,
        metric=args.metric,
        huber_beta=float(args.huber_beta),
        kl_tau=float(args.kl_tau),
        lambda_kl=float(args.lambda_kl),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        skip_soft_fusion=bool(args.skip_soft_fusion),
    )


def compare_with_offline_reference(
    *,
    output_dir: Path,
    online_hard_summary: pd.DataFrame,
    online_soft_summary: Optional[pd.DataFrame],
    offline_reference_dir: Optional[Path],
) -> pd.DataFrame:
    """函数功能：读取离线代表 router 结果，并写出 online/offline 关键指标对比。"""
    rows: List[Dict[str, object]] = []

    def add_from_run(run_name: str, run_dir: Path, hard_summary: pd.DataFrame, soft_summary: Optional[pd.DataFrame]) -> None:
        for row in hard_summary.itertuples(index=False):
            rows.append(
                {
                    "run_name": run_name,
                    "source_dir": str(run_dir),
                    "method": "hard_top1",
                    "config_name": str(row.config_name),
                    "sample_count": int(row.sample_count),
                    "mae": float(row.selected_value),
                    "oracle_mae": float(row.oracle_value),
                    "regret_to_oracle": float(row.regret_to_oracle),
                    "oracle_label_accuracy": float(row.oracle_label_accuracy),
                    "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                    "mean_max_weight": float(row.mean_max_weight),
                }
            )
        if soft_summary is not None:
            for row in soft_summary.itertuples(index=False):
                rows.append(
                    {
                        "run_name": run_name,
                        "source_dir": str(run_dir),
                        "method": "raw_soft_fusion",
                        "config_name": str(row.config_name),
                        "sample_count": int(row.sample_count),
                        "mae": float(row.soft_fusion_mae),
                        "oracle_mae": float(row.oracle_value),
                        "regret_to_oracle": float(row.soft_fusion_mae - row.oracle_value),
                        "oracle_label_accuracy": pd.NA,
                        "mean_normalized_weight_entropy": float(row.mean_normalized_weight_entropy),
                        "mean_max_weight": float(row.mean_max_weight),
                    }
                )

    add_from_run("online_in_memory_vit", output_dir, online_hard_summary, online_soft_summary)
    if offline_reference_dir is not None and offline_reference_dir.exists():
        offline_hard_path = offline_reference_dir / "visual_router_summary.csv"
        offline_soft_path = offline_reference_dir / "visual_router_soft_fusion_summary.csv"
        if offline_hard_path.exists():
            offline_hard = pd.read_csv(offline_hard_path)
            offline_soft = pd.read_csv(offline_soft_path) if offline_soft_path.exists() else None
            add_from_run("offline_embedding_reference", offline_reference_dir, offline_hard, offline_soft)

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df

    # 只对同 config/method 的 MAE 做差，方便直接检查 online 是否复现代表离线结果。
    online_values = comparison_df[comparison_df["run_name"] == "online_in_memory_vit"][
        ["method", "config_name", "mae"]
    ].rename(columns={"mae": "online_mae"})
    comparison_df = comparison_df.merge(online_values, on=["method", "config_name"], how="left")
    comparison_df["mae_delta_vs_online"] = comparison_df["mae"] - comparison_df["online_mae"]
    comparison_df = comparison_df.drop(columns=["online_mae"])
    comparison_df.to_csv(output_dir / "online_vs_offline_reference_comparison.csv", index=False)
    return comparison_df.sort_values(["config_name", "method", "run_name"]).reset_index(drop=True)


def write_online_summary_md(
    *,
    output_dir: Path,
    hard_summary: pd.DataFrame,
    soft_summary: Optional[pd.DataFrame],
    selected_counts: pd.DataFrame,
    comparison_df: pd.DataFrame,
    offline_comparison_df: pd.DataFrame,
    embedding_latency_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写出中文 Markdown 摘要，记录 online embedding 口径和 smoke 结果。"""
    latency_mean = embedding_latency_df[
        ["imageization_per_window_ms", "encoder_forward_per_window_ms", "in_memory_store_per_window_ms"]
    ].mean(numeric_only=True)
    lines = [
        "# Stage 1 Online Visual Router Smoke",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## Online Embedding 口径",
        "",
        "- 输入：Quito 历史窗口 `x`，按 `sample_key` 对齐 labels 和 prediction cache。",
        "- 路径：`x -> pseudo image -> frozen ViT -> CLS embedding -> MLP router`。",
        "- 存储：ViT embedding 仅在本次运行内存字典暂存，不保存 `.npy`；伪图像 tensor 不落盘。",
        f"- encoder: `{metadata['embedding_metadata']['encoder_name']}`，variant: `{metadata['embedding_metadata']['variant']}`。",
        f"- pooling: `{metadata['embedding_metadata']['pooling']}`，period_selection: `{metadata['embedding_metadata']['period_selection']}`。",
        f"- device: `{metadata['device']}`，forward_dtype: `{metadata['embedding_metadata']['forward_dtype']}`。",
        "",
        "## Online Embedding Latency",
        "",
        f"- imageization per-window mean ms: {float(latency_mean.get('imageization_per_window_ms', np.nan)):.6f}",
        f"- encoder forward per-window mean ms: {float(latency_mean.get('encoder_forward_per_window_ms', np.nan)):.6f}",
        f"- in-memory store per-window mean ms: {float(latency_mean.get('in_memory_store_per_window_ms', np.nan)):.6f}",
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
            "## Offline Reference Comparison",
            "",
            frame_to_markdown(offline_comparison_df) if not offline_comparison_df.empty else "_未找到离线代表结果_",
            "",
            "## 输出文件",
            "",
            f"- `online_embedding_manifest.csv`: `{output_dir / 'online_embedding_manifest.csv'}`",
            f"- `online_embedding_latency_summary.csv`: `{output_dir / 'online_embedding_latency_summary.csv'}`",
            f"- `visual_router_predictions.csv`: `{output_dir / 'visual_router_predictions.csv'}`",
            f"- `visual_router_summary.csv`: `{output_dir / 'visual_router_summary.csv'}`",
            f"- `visual_router_soft_fusion_predictions.csv`: `{output_dir / 'visual_router_soft_fusion_predictions.csv'}`",
            f"- `visual_router_soft_fusion_summary.csv`: `{output_dir / 'visual_router_soft_fusion_summary.csv'}`",
            f"- `visual_router_comparison.csv`: `{output_dir / 'visual_router_comparison.csv'}`",
            f"- `online_vs_offline_reference_comparison.csv`: `{output_dir / 'online_vs_offline_reference_comparison.csv'}`",
            f"- `visual_router_online_metadata.json`: `{output_dir / 'visual_router_online_metadata.json'}`",
            "",
        ]
    )
    (output_dir / "visual_router_online_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 online ViT embedding、router 训练、test 评估和结果落盘。"""
    args = parse_args()
    train_args = build_train_args(args)
    validate_training_args(train_args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_online_visual_router_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df = load_labels(args.labels_path, args.metric)
    windows_df = load_required_windows_from_labels(args.labels_path, args.metric)
    feature_lookup, embedding_df, embedding_latency_df, embedding_metadata = build_online_embeddings(
        windows_df=windows_df,
        args=args,
        device=device,
        dtype=dtype,
    )
    validate_online_embedding_manifest(embedding_df, windows_df)
    merged_df = join_embeddings_and_labels(embedding_df, labels_df)
    # ViT 模型构造会消耗 PyTorch RNG。为了让 online 入口复用离线 router
    # 训练函数时仍能按相同 seed 初始化 MLP 和 DataLoader shuffle，这里在
    # embedding 完成后重新固定随机源。
    set_seed(int(args.seed))

    prediction_lookup: Optional[Mapping[Tuple[str, str], Dict[str, object]]] = None
    if args.router_mode == "fusion_huber_kl" or not args.skip_soft_fusion:
        prediction_lookup = load_prediction_lookup(args.prediction_manifest_path)

    router_name = ONLINE_ROUTER_VERSION if args.router_mode == "fusion_huber_kl" else "visual_router_mlp_v1_classification_online_vit"
    hard_prediction_frames: List[pd.DataFrame] = []
    config_metadata: List[Dict[str, object]] = []
    for config_name, config_df in merged_df.groupby("config_name", sort=True):
        router, scaler, metadata = train_router_for_config(
            config_name=str(config_name),
            config_df=config_df,
            manifest_dir=output_dir,
            prediction_lookup=prediction_lookup,
            args=train_args,
            device=device,
            feature_lookup=feature_lookup,
        )
        config_metadata.append(metadata)
        hard_prediction_frames.append(
            predict_router_for_config(
                router=router,
                scaler=scaler,
                config_df=config_df,
                manifest_dir=output_dir,
                device=device,
                router_name=router_name,
                feature_lookup=feature_lookup,
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
    offline_comparison_df = compare_with_offline_reference(
        output_dir=output_dir,
        online_hard_summary=hard_summary_df,
        online_soft_summary=soft_summary_df,
        offline_reference_dir=args.offline_reference_dir,
    )

    embedding_df.to_csv(output_dir / "online_embedding_manifest.csv", index=False)
    embedding_latency_df.to_csv(output_dir / "online_embedding_latency_summary.csv", index=False)
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
        "labels_path": str(args.labels_path),
        "prediction_manifest_path": str(args.prediction_manifest_path),
        "config_path": str(args.config_path),
        "offline_reference_dir": str(args.offline_reference_dir) if args.offline_reference_dir is not None else None,
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
        "embedding_batch_size": int(args.embedding_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "soft_fusion_enabled": not bool(args.skip_soft_fusion),
        "embedding_metadata": embedding_metadata,
        "config_metadata": config_metadata,
        "embedding_storage": "in_memory_only",
        "pseudo_image_tensor_storage": "not_saved",
        "persistent_embedding_npy_written": False,
        "persistent_pseudo_image_tensor_written": False,
        "router_seed_reset_after_online_embedding": True,
        "input_exclusions": ["future_y_as_feature", "test_oracle_error_as_feature", "expert_error_as_feature"],
    }
    (output_dir / "visual_router_online_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_online_summary_md(
        output_dir=output_dir,
        hard_summary=hard_summary_df,
        soft_summary=soft_summary_df,
        selected_counts=selected_counts_df,
        comparison_df=comparison_df,
        offline_comparison_df=offline_comparison_df,
        embedding_latency_df=embedding_latency_df,
        metadata=run_metadata,
    )

    print(f"wrote online visual router outputs to {output_dir}")
    print(hard_summary_df.to_string(index=False))
    if soft_summary_df is not None:
        print(soft_summary_df.to_string(index=False))
    if not offline_comparison_df.empty:
        print(offline_comparison_df.to_string(index=False))
    preview_cols = ["sample_key", "selected_model", "selected_value", "oracle_model", "oracle_value", *[f"weight_{m}" for m in MODEL_COLUMNS]]
    print(hard_pred_df[preview_cols].head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
