#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router Stage 1 验证在线伪图像化路径。

Pilot 限制：
    - 默认只覆盖已有 96_48_S 扩大版 prediction cache pilot 中 metric=mae 的
      120 个 sample_key；
    - 只从 Quito train-based normalized 历史窗口 x 在线生成伪图像；
    - 不读取未来 y、不读取专家误差、不读取 oracle label 作为模型输入；
    - 不保存全量图像 tensor，只保存 index、metadata、latency summary 和少量 debug PNG；
    - 该脚本只验证图像化与审计口径，不训练 visual router，不作为正式结论。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image


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
from visual_router_experiments.common.pseudo_imageization import (  # noqa: E402
    audit_metadata,
    imageize_3view,
    imageize_top3fold,
    normalize_window,
    select_fft_periods,
)


PSEUDO_IMAGE_VERSION = "visual_router_online_pseudo_image_v1"
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
DEFAULT_LABELS_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "window_oracle_labels_with_tsf_cell.csv"
)


def now_token() -> str:
    """函数功能：生成 run 目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata 和 summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析在线伪图像化 pilot 参数。"""
    parser = argparse.ArgumentParser(description="Build an online pseudo-imageization pilot for Stage 1 visual router.")
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="oracle labels CSV 路径；默认使用 96_48_S 扩大版五专家 pilot。",
    )
    parser.add_argument(
        "--metric",
        default="mae",
        choices=["mae", "mse"],
        help="用于确定需要覆盖哪些 sample_key；默认 mae。",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Quito evaluate config 路径；只复用 data_config 加载历史窗口 x。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入 experiment_logs/run_outputs 下的时间戳目录。",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="每次在线图像化的窗口 batch size。")
    parser.add_argument("--image-size", type=int, default=224, help="输出伪图像尺寸，默认 224。")
    parser.add_argument(
        "--norm-mode",
        default="revin_aux",
        choices=["quito", "revin", "revin_aux"],
        help="历史窗口 normalization 口径；默认 revin_aux。",
    )
    parser.add_argument(
        "--pixel-mode",
        default="vision",
        choices=["vision"],
        help="pixel 映射口径；当前固定为 vision。",
    )
    parser.add_argument("--clip", type=float, default=5.0, help="vision pixel 映射前的对称截断阈值。")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="主验证设备；auto 在 CUDA 可用时使用 cuda。",
    )
    parser.add_argument("--max-debug-previews", type=int, default=12, help="最多保存多少个样本的 debug PNG；每个样本保存两个 variant。")
    parser.add_argument("--print-rows", type=int, default=20, help="运行结束时最多打印多少行 index 预览。")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析主图像化设备，auto 优先使用 CUDA。"""
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


def load_required_windows(labels_path: Path, metric: str) -> pd.DataFrame:
    """
    函数功能：
        读取 oracle labels，筛出指定 metric 的唯一 sample_key 元信息。

    约束：
        标签文件只提供待覆盖窗口清单；专家误差、oracle_model 和 regret 字段不作为输入。
    """
    labels_df = pd.read_csv(labels_path)
    required_cols = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "metric",
    }
    missing_cols = sorted(required_cols.difference(labels_df.columns))
    if missing_cols:
        raise ValueError(f"oracle labels 缺少字段：{missing_cols}")

    windows_df = labels_df.loc[labels_df["metric"] == metric, sorted(required_cols - {"metric"})].copy()
    windows_df = windows_df.drop_duplicates().reset_index(drop=True)
    if windows_df.empty:
        raise ValueError(f"{labels_path} 中没有 metric={metric} 的样本")
    duplicated = windows_df["sample_key"].duplicated()
    if duplicated.any():
        dup_keys = windows_df.loc[duplicated, "sample_key"].head(10).tolist()
        raise ValueError(f"metric={metric} 的 sample_key 不唯一，示例：{dup_keys}")
    return windows_df


def build_required_index(windows_df: pd.DataFrame) -> Mapping[Tuple[str, str, int], Set[Tuple[int, int, str]]]:
    """
    函数功能：
        将待覆盖窗口整理成便于按 split/dataset/item 定位 Quito 历史 x 的索引。
    """
    required: Dict[Tuple[str, str, int], Set[Tuple[int, int, str]]] = {}
    for row in windows_df.itertuples(index=False):
        group_key = (str(row.split), str(row.dataset_name), int(row.item_id))
        required.setdefault(group_key, set()).add((int(row.channel_id), int(row.window_index), str(row.sample_key)))
    return required


def _timer_start(device: torch.device) -> float:
    """函数功能：启动一次 latency 计时；CUDA 路径先同步以得到真实 wall time。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _timer_stop(start_time: float, device: torch.device) -> float:
    """函数功能：结束一次 latency 计时，返回毫秒。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start_time) * 1000.0


def generate_variants(
    x_batch: torch.Tensor,
    *,
    norm_mode: str,
    pixel_mode: str,
    clip: float,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    函数功能：
        在线生成 variant_a/variant_b，并返回审计 metadata。

    输入：
        x_batch: [B, L, 1] Quito train-normalized 历史窗口，只包含历史 x。
    """
    x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=False)
    x_norm, norm_metadata = normalize_window(x_batch, norm_mode=norm_mode)
    periods = select_fft_periods(x_norm, top_k=3)
    variant_a = imageize_3view(
        x_norm,
        image_size=image_size,
        periods=periods,
        pixel_mode=pixel_mode,
        clip=clip,
    )
    variant_b = imageize_top3fold(
        x_norm,
        image_size=image_size,
        periods=periods,
        pixel_mode=pixel_mode,
        clip=clip,
    )
    metadata = audit_metadata(x_norm=x_norm, norm_metadata=norm_metadata, periods=periods, clip=clip)
    return variant_a, variant_b, metadata


def tensor_to_png(image_tensor: torch.Tensor, output_path: Path) -> str:
    """
    函数功能：
        将 [3, H, W] 的 [0, 1] tensor 后处理成 debug PNG。

    约束：
        PNG 仅用于少量人工预览；主图像化路径不依赖 PIL。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = image_tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(output_path)
    return str(output_path)


def validate_image_batch(image_tensor: torch.Tensor, expected_batch: int, image_size: int, variant_name: str) -> Dict[str, object]:
    """函数功能：校验伪图像 batch 的 shape、finite 和 [0, 1] 范围。"""
    expected_shape = (expected_batch, 3, image_size, image_size)
    if tuple(image_tensor.shape) != expected_shape:
        raise ValueError(f"{variant_name} shape 不正确：{tuple(image_tensor.shape)} != {expected_shape}")
    if not bool(torch.isfinite(image_tensor).all().item()):
        raise ValueError(f"{variant_name} 存在 NaN 或 Inf")
    min_value = float(image_tensor.min().item())
    max_value = float(image_tensor.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise ValueError(f"{variant_name} 范围不在 [0, 1]：min={min_value} max={max_value}")
    return {
        f"{variant_name}_shape": "x".join(str(dim) for dim in image_tensor.shape[1:]),
        f"{variant_name}_min": min_value,
        f"{variant_name}_max": max_value,
        f"{variant_name}_finite": True,
    }


def batch_required_pairs(required_for_item: Set[Tuple[int, int, str]], batch_size: int) -> List[List[Tuple[int, int, str]]]:
    """函数功能：按稳定顺序将 required window 切成小 batch。"""
    sorted_pairs = sorted(required_for_item, key=lambda item: (item[0], item[1], item[2]))
    return [sorted_pairs[start : start + batch_size] for start in range(0, len(sorted_pairs), batch_size)]


def build_rows(
    *,
    data_config,
    windows_df: pd.DataFrame,
    batch_size: int,
    image_size: int,
    norm_mode: str,
    pixel_mode: str,
    clip: float,
    device: torch.device,
    output_dir: Path,
    max_debug_previews: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """
    函数功能：
        遍历 Quito vali/test 数据集，为 oracle label 指定的 sample_key 在线生成伪图像。

    关键约束：
        直接从 item_dataset.data 切片历史窗口，只读取 x 对应的 [window, window+seq_len)，
        不访问 DataLoader batch['y'] 或专家预测结果。
    """
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"], windows_df["config_name"]))
    rows: List[Dict[str, object]] = []
    latency_rows: List[Dict[str, object]] = []
    seen_keys: Set[str] = set()
    preview_count = 0

    for split in sorted(windows_df["split"].unique()):
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
                item_dataset.select_user_data(item_id)
                channel_count = int(item_dataset.data.shape[0])
                required_for_item = required_index[(str(split), str(dataset_name), int(item_id))]

                for pair_batch in batch_required_pairs(required_for_item, batch_size=batch_size):
                    x_windows = []
                    sample_keys = []
                    channel_ids = []
                    window_indices = []
                    for channel_id, window_index, sample_key in pair_batch:
                        if channel_id >= channel_count:
                            raise ValueError(
                                f"channel_id 越界：sample_key={sample_key} channel={channel_id} channel_count={channel_count}"
                            )
                        window_start = int(window_index)
                        window_end = window_start + int(data_config.seq_len)
                        # 这里只切历史 x 窗口；未来 y 和 oracle label 均不进入图像化输入。
                        x_window = item_dataset.data[int(channel_id), window_start:window_end, :]
                        if x_window.shape[0] != int(data_config.seq_len):
                            raise ValueError(f"历史窗口长度不完整：sample_key={sample_key} shape={x_window.shape}")
                        x_windows.append(x_window)
                        sample_keys.append(str(sample_key))
                        channel_ids.append(int(channel_id))
                        window_indices.append(int(window_index))

                    x_cpu = torch.from_numpy(np.stack(x_windows, axis=0)).to(dtype=torch.float32)

                    cpu_start = _timer_start(torch.device("cpu"))
                    cpu_a, cpu_b, cpu_meta = generate_variants(
                        x_cpu,
                        norm_mode=norm_mode,
                        pixel_mode=pixel_mode,
                        clip=clip,
                        image_size=image_size,
                        device=torch.device("cpu"),
                    )
                    cpu_ms = _timer_stop(cpu_start, torch.device("cpu"))
                    del cpu_a, cpu_b, cpu_meta

                    gpu_ms: Optional[float] = None
                    start = _timer_start(device)
                    variant_a, variant_b, metadata = generate_variants(
                        x_cpu,
                        norm_mode=norm_mode,
                        pixel_mode=pixel_mode,
                        clip=clip,
                        image_size=image_size,
                        device=device,
                    )
                    elapsed_ms = _timer_stop(start, device)
                    if device.type == "cuda":
                        gpu_ms = elapsed_ms
                    else:
                        cpu_ms = elapsed_ms

                    variant_a_stats = validate_image_batch(variant_a, len(pair_batch), image_size, "variant_a")
                    variant_b_stats = validate_image_batch(variant_b, len(pair_batch), image_size, "variant_b")

                    latency_rows.append(
                        {
                            "split": str(split),
                            "dataset_name": str(dataset_name),
                            "item_id": int(item_id),
                            "batch_size": int(len(pair_batch)),
                            "cpu_total_ms": float(cpu_ms),
                            "cpu_per_window_ms": float(cpu_ms / len(pair_batch)),
                            "gpu_total_ms": None if gpu_ms is None else float(gpu_ms),
                            "gpu_per_window_ms": None if gpu_ms is None else float(gpu_ms / len(pair_batch)),
                            "device": str(device),
                        }
                    )

                    top3_periods = metadata["top3_periods"].detach().cpu().numpy()
                    norm_mean = metadata["norm_mean"].detach().cpu().numpy()
                    norm_std = metadata["norm_std"].detach().cpu().numpy()
                    norm_range = metadata["norm_range"].detach().cpu().numpy()
                    clip_ratio = metadata["clip_ratio"].detach().cpu().numpy()

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

                        preview_a = ""
                        preview_b = ""
                        if preview_count < max_debug_previews:
                            safe_name = sample_key.replace("/", "_")
                            preview_a = tensor_to_png(
                                variant_a[row_idx],
                                output_dir / "debug_preview" / f"{safe_name}__variant_a_3view.png",
                            )
                            preview_b = tensor_to_png(
                                variant_b[row_idx],
                                output_dir / "debug_preview" / f"{safe_name}__variant_b_top3fold.png",
                            )
                            preview_count += 1

                        row: Dict[str, object] = {
                            "pseudo_image_version": PSEUDO_IMAGE_VERSION,
                            "sample_key": sample_key,
                            "config_name": str(config_by_key[sample_key]),
                            "split": str(split),
                            "dataset_name": str(dataset_name),
                            "item_id": int(item_id),
                            "channel_id": int(channel_ids[row_idx]),
                            "window_index": int(window_indices[row_idx]),
                            "history_length": int(data_config.seq_len),
                            "image_size": int(image_size),
                            "norm_mode": norm_mode,
                            "pixel_mode": pixel_mode,
                            "clip": float(clip),
                            "period_policy": "fft_top3",
                            "top3_periods": ";".join(str(int(v)) for v in top3_periods[row_idx].tolist()),
                            "norm_mean": float(norm_mean[row_idx]),
                            "norm_std": float(norm_std[row_idx]),
                            "norm_range": float(norm_range[row_idx]),
                            "clip_ratio": float(clip_ratio[row_idx]),
                            "debug_preview_variant_a": preview_a,
                            "debug_preview_variant_b": preview_b,
                        }
                        row.update(variant_a_stats)
                        row.update(variant_b_stats)
                        rows.append(row)
                        seen_keys.add(sample_key)

    expected_keys = set(windows_df["sample_key"])
    missing_keys = sorted(expected_keys - seen_keys)
    extra_keys = sorted(seen_keys - expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(f"伪图像化覆盖不一致：missing={missing_keys[:10]} extra={extra_keys[:10]}")

    return rows, latency_rows


def validate_index(index_df: pd.DataFrame, windows_df: pd.DataFrame) -> None:
    """函数功能：校验 imageization_index 与目标 sample_key 清单严格对齐。"""
    required_cols = {
        "pseudo_image_version",
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "history_length",
        "image_size",
        "norm_mode",
        "pixel_mode",
        "clip",
        "period_policy",
        "top3_periods",
        "norm_mean",
        "norm_std",
        "norm_range",
        "clip_ratio",
        "variant_a_shape",
        "variant_a_min",
        "variant_a_max",
        "variant_a_finite",
        "variant_b_shape",
        "variant_b_min",
        "variant_b_max",
        "variant_b_finite",
    }
    missing_cols = sorted(required_cols.difference(index_df.columns))
    if missing_cols:
        raise ValueError(f"imageization_index 缺少字段：{missing_cols}")

    if index_df["sample_key"].duplicated().any():
        dup_keys = index_df.loc[index_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"imageization_index 中 sample_key 重复，示例：{dup_keys}")

    expected_keys = set(windows_df["sample_key"])
    actual_keys = set(index_df["sample_key"])
    if expected_keys != actual_keys:
        raise ValueError(
            f"sample_key 覆盖不一致：missing={sorted(expected_keys - actual_keys)[:10]} "
            f"extra={sorted(actual_keys - expected_keys)[:10]}"
        )

    expected_key_strings = index_df.apply(
        lambda row: PredictionCacheKey(
            config_name=str(row["config_name"]),
            split=str(row["split"]),
            dataset_name=str(row["dataset_name"]),
            item_id=int(row["item_id"]),
            channel_id=int(row["channel_id"]),
            window_index=int(row["window_index"]),
        ).as_string(),
        axis=1,
    )
    if not (expected_key_strings == index_df["sample_key"]).all():
        bad_keys = index_df.loc[expected_key_strings != index_df["sample_key"], "sample_key"].head(10).tolist()
        raise ValueError(f"sample_key 与元信息不一致，示例：{bad_keys}")

    for col in ["norm_mean", "norm_std", "norm_range", "clip_ratio", "variant_a_min", "variant_a_max", "variant_b_min", "variant_b_max"]:
        values = index_df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{col} 存在 NaN 或 Inf")
    if not ((index_df["clip_ratio"] >= 0.0) & (index_df["clip_ratio"] <= 1.0)).all():
        raise ValueError("clip_ratio 不在 [0, 1]")
    if not ((index_df["variant_a_min"] >= -1e-6) & (index_df["variant_a_max"] <= 1.0 + 1e-6)).all():
        raise ValueError("variant_a 范围不在 [0, 1]")
    if not ((index_df["variant_b_min"] >= -1e-6) & (index_df["variant_b_max"] <= 1.0 + 1e-6)).all():
        raise ValueError("variant_b 范围不在 [0, 1]")

    split_dataset_counts = index_df.groupby(["split", "dataset_name"]).size()
    required_pairs = {("vali", "TEST_DATA_MIN"), ("vali", "TEST_DATA_HOUR"), ("test", "TEST_DATA_MIN"), ("test", "TEST_DATA_HOUR")}
    actual_pairs = set(split_dataset_counts.index.tolist())
    missing_pairs = sorted(required_pairs - actual_pairs)
    if missing_pairs:
        raise ValueError(f"未覆盖预期 split/dataset：{missing_pairs}")


def write_summary(output_dir: Path, index_df: pd.DataFrame, latency_df: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> None:
    """函数功能：写出简短 Markdown 摘要，便于快速查看本次 pilot 结果。"""
    split_counts = index_df.groupby(["config_name", "split", "dataset_name"]).size().reset_index(name="rows")
    count_lines = [
        "| config_name | split | dataset_name | rows |",
        "| --- | --- | --- | --- |",
    ]
    for row in split_counts.itertuples(index=False):
        count_lines.append(f"| {row.config_name} | {row.split} | {row.dataset_name} | {row.rows} |")

    latency_summary = latency_df[["cpu_per_window_ms", "gpu_per_window_ms"]].mean(numeric_only=True)
    summary_lines = [
        "# Stage 1 在线伪图像化 Pilot",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 输入",
        "",
        f"- labels_path: `{args.labels_path}`",
        f"- metric: `{args.metric}`",
        f"- config_path: `{args.config_path}`",
        "",
        "## 口径",
        "",
        f"- norm_mode: `{args.norm_mode}`",
        f"- pixel_mode: `{args.pixel_mode}`",
        f"- clip: `{args.clip}`",
        f"- image_size: `{args.image_size}`",
        f"- period_policy: `fft_top3`",
        f"- device: `{device}`",
        "",
        "## 输出",
        "",
        f"- imageization_index.csv: `{output_dir / 'imageization_index.csv'}`",
        f"- latency_summary.csv: `{output_dir / 'latency_summary.csv'}`",
        f"- metadata.json: `{output_dir / 'metadata.json'}`",
        f"- debug_preview/: `{output_dir / 'debug_preview'}`",
        f"- 样本数：{len(index_df)}",
        "",
        "## Latency",
        "",
        f"- CPU per-window mean ms: {float(latency_summary.get('cpu_per_window_ms', np.nan)):.6f}",
        f"- GPU per-window mean ms: {float(latency_summary.get('gpu_per_window_ms', np.nan)):.6f}",
        "",
        "## 分层计数",
        "",
        "\n".join(count_lines),
        "",
        "## 输入排除项",
        "",
        "- 未读取未来 y 作为图像化输入。",
        "- 未读取专家误差、regret 或 oracle_model 作为图像化输入。",
        "- 未保存全量图像 tensor cache。",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 Stage 1 在线伪图像化 pilot。"""
    args = parse_args()
    output_dir = args.output_dir or RUN_OUTPUT_ROOT / f"{now_token()}_visual_router_stage1_online_pseudo_image_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    windows_df = load_required_windows(args.labels_path, args.metric)
    data_config = load_data_config(args.config_path)
    rows, latency_rows = build_rows(
        data_config=data_config,
        windows_df=windows_df,
        batch_size=args.batch_size,
        image_size=args.image_size,
        norm_mode=args.norm_mode,
        pixel_mode=args.pixel_mode,
        clip=args.clip,
        device=device,
        output_dir=output_dir,
        max_debug_previews=args.max_debug_previews,
    )

    index_df = pd.DataFrame(rows)
    index_df = index_df.sort_values(["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]).reset_index(drop=True)
    validate_index(index_df, windows_df)
    index_df.to_csv(output_dir / "imageization_index.csv", index=False)

    latency_df = pd.DataFrame(latency_rows)
    latency_df.to_csv(output_dir / "latency_summary.csv", index=False)

    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "labels_path": str(args.labels_path),
        "metric": args.metric,
        "config_path": str(args.config_path),
        "pseudo_image_version": PSEUDO_IMAGE_VERSION,
        "sample_count": int(len(index_df)),
        "norm_mode": args.norm_mode,
        "pixel_mode": args.pixel_mode,
        "clip": float(args.clip),
        "image_size": int(args.image_size),
        "period_policy": "fft_top3",
        "variants": {
            "variant_a": ["line_raster", "top1_period_fold", "fft_power"],
            "variant_b": ["top1_period_fold", "top2_period_fold", "top3_period_fold"],
        },
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "splits": sorted(index_df["split"].unique().tolist()),
        "datasets": sorted(index_df["dataset_name"].unique().tolist()),
        "config_names": sorted(index_df["config_name"].unique().tolist()),
        "input_exclusions": [
            "future_y",
            "expert_errors",
            "oracle_model",
            "oracle_value",
            "expert_regret",
        ],
        "saved_full_tensor_cache": False,
        "debug_preview_sample_count": int((index_df["debug_preview_variant_a"] != "").sum()),
        "debug_preview_png_count": int((index_df["debug_preview_variant_a"] != "").sum() * 2),
        "clip_ratio_summary": {
            "min": float(index_df["clip_ratio"].min()),
            "mean": float(index_df["clip_ratio"].mean()),
            "max": float(index_df["clip_ratio"].max()),
        },
        "norm_summary": {
            "norm_mean_mean": float(index_df["norm_mean"].mean()),
            "norm_std_mean": float(index_df["norm_std"].mean()),
            "norm_range_mean": float(index_df["norm_range"].mean()),
        },
        "sample_metadata": index_df[
            ["sample_key", "norm_mean", "norm_std", "norm_range", "clip_ratio", "top3_periods"]
        ].to_dict(orient="records"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir, index_df, latency_df, args, device)

    print(f"wrote online pseudo-image pilot to {output_dir}")
    print(f"sample_count={len(index_df)} device={device}")
    preview_cols = ["sample_key", "top3_periods", "clip_ratio", "variant_a_shape", "variant_b_shape"]
    print(index_df[preview_cols].head(args.print_rows).to_string(index=False))
    if len(index_df) > args.print_rows:
        print(f"... omitted {len(index_df) - args.print_rows} rows")


if __name__ == "__main__":
    main()
