#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router Stage 1 构建冻结 HF ViT embedding。

设计约束：
    - 输入只来自 Quito 历史窗口 x，经在线伪图像化后送入冻结视觉 encoder；
    - 默认使用 `google/vit-base-patch16-224`，并采用 last_hidden_state 的 CLS token
      作为 768 维视觉特征；
    - 不读取未来 y、专家误差或 oracle label 作为模型输入；
    - 每个 embedding 以 `sample_key` 对齐 prediction cache / oracle labels；
    - 小规模 smoke 可以写入仓库内 `experiment_logs/run_outputs/`，大规模缓存应通过
      `--output-root` 或 `--cache-root` 写到外部输出盘。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
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
from visual_router_experiments.common.pseudo_imageization import (  # noqa: E402
    encoder_normalize,
    imageize_3view,
    imageize_top3fold,
    normalize_window,
    select_fft_periods,
)


EMBEDDING_VERSION = "visual_router_vit_embedding_v1"
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
    """函数功能：解析 ViT embedding 构建参数。"""
    parser = argparse.ArgumentParser(description="Build frozen ViT embeddings for Stage 1 Visual Router.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="oracle labels CSV，用于确定 sample_key 清单。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="选择待覆盖 sample_key 的 oracle label 指标。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config；仅复用 data_config 加载历史窗口 x。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认基于 output-root 生成时间戳目录。")
    parser.add_argument("--cache-root", type=Path, default=None, help="embedding npy 缓存根目录；默认使用 output-dir/embeddings。")
    parser.add_argument("--encoder-name", default="google/vit-base-patch16-224", help="Hugging Face ViT encoder 名称或本地路径。")
    parser.add_argument("--variant", choices=["variant_a_3view", "variant_b_top3fold"], default="variant_a_3view", help="伪图像 variant。")
    parser.add_argument("--pooling", choices=["cls", "mean_patch", "pooler"], default="cls", help="ViT 输出聚合方式；默认 last_hidden_state[:, 0]。")
    parser.add_argument("--normalization-preset", default="hf_vit_0_5", help="encoder 前 normalization 口径。")
    parser.add_argument("--batch-size", type=int, default=16, help="ViT 前向 batch size。")
    parser.add_argument("--image-size", type=int, default=224, help="伪图像尺寸。")
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux", help="历史窗口 normalization 口径。")
    parser.add_argument("--pixel-mode", choices=["vision"], default="vision", help="pixel 映射口径。")
    parser.add_argument("--clip", type=float, default=5.0, help="视觉 pixel 映射前的对称截断阈值。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="运行设备。")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto", help="encoder 前向 dtype；CPU 会强制 fp32。")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地 Hugging Face 缓存，不联网下载。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束时打印多少行 manifest 预览。")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析运行设备，auto 优先 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 PyTorch CUDA 不可用")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """函数功能：解析 ViT 前向 dtype；CPU 路径保持 fp32，避免半精度算子兼容问题。"""
    if device.type == "cpu":
        return torch.float32
    if dtype_arg == "fp32":
        return torch.float32
    return torch.float16


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
    """函数功能：从 oracle labels 中读取指定 metric 的唯一 sample_key 清单。"""
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
    if windows_df["sample_key"].duplicated().any():
        dup_keys = windows_df.loc[windows_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"metric={metric} 的 sample_key 不唯一，示例：{dup_keys}")
    return windows_df


def build_required_index(windows_df: pd.DataFrame) -> Mapping[Tuple[str, str, int], Set[Tuple[int, int, str]]]:
    """函数功能：把待覆盖窗口整理为 split/dataset/item -> channel/window/sample_key 索引。"""
    required: Dict[Tuple[str, str, int], Set[Tuple[int, int, str]]] = {}
    for row in windows_df.itertuples(index=False):
        group_key = (str(row.split), str(row.dataset_name), int(row.item_id))
        required.setdefault(group_key, set()).add((int(row.channel_id), int(row.window_index), str(row.sample_key)))
    return required


def batch_required_pairs(required_for_item: Set[Tuple[int, int, str]], batch_size: int) -> List[List[Tuple[int, int, str]]]:
    """函数功能：按稳定顺序将 required window 切成小 batch。"""
    sorted_pairs = sorted(required_for_item, key=lambda item: (item[0], item[1], item[2]))
    return [sorted_pairs[start : start + batch_size] for start in range(0, len(sorted_pairs), batch_size)]


def make_pseudo_images(
    x_batch: torch.Tensor,
    *,
    variant: str,
    norm_mode: str,
    pixel_mode: str,
    clip: float,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    normalization_preset: str,
) -> torch.Tensor:
    """
    函数功能：
        从历史窗口 batch 构造 ViT `pixel_values`。

    关键约束：
        只使用历史 x。`encoder_normalize()` 在伪图像 [0,1] 输出之后执行，
        这里不走 HF processor，避免 rescale / normalize 口径隐式变化。
    """
    x_batch = x_batch.to(device=device, dtype=torch.float32, non_blocking=False)
    x_norm, _ = normalize_window(x_batch, norm_mode=norm_mode)
    periods = select_fft_periods(x_norm, top_k=3)
    if variant == "variant_a_3view":
        images = imageize_3view(x_norm, image_size=image_size, periods=periods, pixel_mode=pixel_mode, clip=clip)
    elif variant == "variant_b_top3fold":
        images = imageize_top3fold(x_norm, image_size=image_size, periods=periods, pixel_mode=pixel_mode, clip=clip)
    else:
        raise ValueError(f"未知 variant={variant}")
    return encoder_normalize(images.to(dtype=dtype), preset=normalization_preset)


def pool_vit_outputs(outputs, pooling: str) -> torch.Tensor:
    """
    函数功能：
        将 ViT 输出聚合成单个视觉特征向量。

    说明：
        默认使用 last_hidden_state 的 CLS token，这是 Hugging Face ViT 分类头前最常见
        的图像级表示；mean_patch 可作为后续 ablation。
    """
    if pooling == "cls":
        return outputs.last_hidden_state[:, 0]
    if pooling == "mean_patch":
        return outputs.last_hidden_state[:, 1:].mean(dim=1)
    if pooling == "pooler":
        if outputs.pooler_output is None:
            raise ValueError("当前 encoder 输出没有 pooler_output，不能使用 --pooling pooler")
        return outputs.pooler_output
    raise ValueError(f"未知 pooling={pooling}")


def safe_embedding_name(sample_key: str, variant: str) -> str:
    """函数功能：把 sample_key 转成适合作为 npy 文件名的稳定名称。"""
    return f"{sample_key.replace('/', '_')}__{variant}.npy"


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


def build_embeddings(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Path]:
    """函数功能：主流程，遍历 Quito 窗口、生成伪图像、前向 ViT 并保存 embedding。"""
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_vit_embedding_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dir = args.cache_root or output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    windows_df = load_required_windows(args.labels_path, args.metric)
    data_config = load_data_config(args.config_path)
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"], windows_df["config_name"]))

    # 默认使用 last_hidden_state 的 CLS token，不需要 ViTModel 的 pooler；关闭 pooler
    # 可避免从分类 checkpoint 加载 encoder 时创建未初始化的随机 pooler 参数。
    model = ViTModel.from_pretrained(
        args.encoder_name,
        local_files_only=bool(args.local_files_only),
        add_pooling_layer=args.pooling == "pooler",
    )
    model.eval().to(device=device)
    if dtype == torch.float16:
        model = model.half()

    rows: List[Dict[str, object]] = []
    latency_rows: List[Dict[str, object]] = []
    seen_keys: Set[str] = set()

    with torch.inference_mode():
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

                    for pair_batch in batch_required_pairs(required_for_item, args.batch_size):
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
                            # 这里只切历史 x 窗口，不访问未来 y 或专家预测。
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
                            clip=args.clip,
                            image_size=args.image_size,
                            device=device,
                            dtype=dtype,
                            normalization_preset=args.normalization_preset,
                        )
                        image_ms = _timer_stop(image_start, device)

                        forward_start = _timer_start(device)
                        outputs = model(pixel_values=pixel_values)
                        embeddings = pool_vit_outputs(outputs, args.pooling)
                        forward_ms = _timer_stop(forward_start, device)
                        embeddings_cpu = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()

                        write_start = time.perf_counter()
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

                            embedding = embeddings_cpu[row_idx]
                            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                                raise ValueError(f"embedding 非法：sample_key={sample_key} shape={embedding.shape}")
                            rel_path = Path("embeddings") / safe_embedding_name(sample_key, args.variant)
                            abs_path = embedding_dir / safe_embedding_name(sample_key, args.variant)
                            np.save(abs_path, embedding.astype(np.float32))
                            rows.append(
                                {
                                    "embedding_version": EMBEDDING_VERSION,
                                    "sample_key": sample_key,
                                    "config_name": str(config_by_key[sample_key]),
                                    "split": str(split),
                                    "dataset_name": str(dataset_name),
                                    "item_id": int(item_id),
                                    "channel_id": int(channel_ids[row_idx]),
                                    "window_index": int(window_indices[row_idx]),
                                    "history_length": int(data_config.seq_len),
                                    "feature_type": "vit_embedding",
                                    "variant": args.variant,
                                    "encoder_name": args.encoder_name,
                                    "pooling": args.pooling,
                                    "normalization_preset": args.normalization_preset,
                                    "input_mode": "direct_pixel_values",
                                    "processor_do_rescale": "not_used",
                                    "embedding_path": str(rel_path if embedding_dir == output_dir / "embeddings" else abs_path),
                                    "embedding_dim": int(embedding.shape[0]),
                                    "dtype": "float32_saved",
                                    "finite": True,
                                }
                            )
                            seen_keys.add(sample_key)
                        write_ms = (time.perf_counter() - write_start) * 1000.0
                        latency_rows.append(
                            {
                                "split": str(split),
                                "dataset_name": str(dataset_name),
                                "item_id": int(item_id),
                                "batch_size": int(len(pair_batch)),
                                "imageization_ms": float(image_ms),
                                "encoder_forward_ms": float(forward_ms),
                                "write_ms": float(write_ms),
                                "imageization_per_window_ms": float(image_ms / len(pair_batch)),
                                "encoder_forward_per_window_ms": float(forward_ms / len(pair_batch)),
                                "write_per_window_ms": float(write_ms / len(pair_batch)),
                                "device": str(device),
                            }
                        )

    expected_keys = set(windows_df["sample_key"])
    if seen_keys != expected_keys:
        raise ValueError(f"embedding 覆盖不一致：missing={sorted(expected_keys - seen_keys)[:10]} extra={sorted(seen_keys - expected_keys)[:10]}")

    manifest_df = pd.DataFrame(rows).sort_values(["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]).reset_index(drop=True)
    latency_df = pd.DataFrame(latency_rows)
    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "embedding_dir": str(embedding_dir),
        "labels_path": str(args.labels_path),
        "metric": args.metric,
        "config_path": str(args.config_path),
        "embedding_version": EMBEDDING_VERSION,
        "sample_count": int(len(manifest_df)),
        "encoder_name": args.encoder_name,
        "variant": args.variant,
        "pooling": args.pooling,
        "pooling_detail": "last_hidden_state[:, 0] CLS token" if args.pooling == "cls" else args.pooling,
        "normalization_preset": args.normalization_preset,
        "input_mode": "direct_pixel_values",
        "processor_do_rescale": "not_used",
        "image_size": int(args.image_size),
        "norm_mode": args.norm_mode,
        "pixel_mode": args.pixel_mode,
        "clip": float(args.clip),
        "device": str(device),
        "forward_dtype": str(dtype).replace("torch.", ""),
        "saved_dtype": "float32",
        "embedding_dim": int(manifest_df["embedding_dim"].iloc[0]),
        "splits": sorted(manifest_df["split"].unique().tolist()),
        "config_names": sorted(manifest_df["config_name"].unique().tolist()),
        "input_exclusions": ["future_y", "expert_errors", "oracle_model", "oracle_value", "expert_regret"],
    }
    return manifest_df, latency_df, metadata, output_dir


def validate_manifest(manifest_df: pd.DataFrame, labels_path: Path, metric: str) -> None:
    """函数功能：校验 embedding manifest 字段、shape 和 sample_key 覆盖。"""
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
        "embedding_path",
        "embedding_dim",
        "dtype",
        "finite",
    }
    missing = sorted(required_cols.difference(manifest_df.columns))
    if missing:
        raise ValueError(f"embedding manifest 缺少字段：{missing}")
    if manifest_df["sample_key"].duplicated().any():
        dup_keys = manifest_df.loc[manifest_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"embedding manifest sample_key 重复，示例：{dup_keys}")
    if not manifest_df["finite"].astype(bool).all():
        raise ValueError("embedding manifest 中存在 finite=False")
    if manifest_df["embedding_dim"].nunique() != 1:
        raise ValueError("embedding_dim 不一致")

    expected_df = load_required_windows(labels_path, metric)
    if set(manifest_df["sample_key"]) != set(expected_df["sample_key"]):
        raise ValueError("embedding manifest 与 labels sample_key 集合不一致")


def write_summary(output_dir: Path, manifest_df: pd.DataFrame, latency_df: pd.DataFrame, metadata: Dict[str, object]) -> None:
    """函数功能：写出中文 Markdown 摘要。"""
    split_counts = manifest_df.groupby(["config_name", "split", "dataset_name"]).size().reset_index(name="rows")
    count_lines = ["| config_name | split | dataset_name | rows |", "| --- | --- | --- | --- |"]
    for row in split_counts.itertuples(index=False):
        count_lines.append(f"| {row.config_name} | {row.split} | {row.dataset_name} | {row.rows} |")

    latency_mean = latency_df[["imageization_per_window_ms", "encoder_forward_per_window_ms", "write_per_window_ms"]].mean(numeric_only=True)
    lines = [
        "# Stage 1 ViT Embedding Smoke",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 口径",
        "",
        f"- encoder: `{metadata['encoder_name']}`",
        f"- variant: `{metadata['variant']}`",
        f"- pooling: `{metadata['pooling']}` ({metadata['pooling_detail']})",
        f"- normalization_preset: `{metadata['normalization_preset']}`",
        f"- input_mode: `{metadata['input_mode']}`",
        f"- forward_dtype: `{metadata['forward_dtype']}`，saved_dtype: `{metadata['saved_dtype']}`",
        f"- embedding_dim: `{metadata['embedding_dim']}`",
        "",
        "## 输出",
        "",
        f"- embedding_manifest.csv: `{output_dir / 'embedding_manifest.csv'}`",
        f"- embedding_latency_summary.csv: `{output_dir / 'embedding_latency_summary.csv'}`",
        f"- embedding_metadata.json: `{output_dir / 'embedding_metadata.json'}`",
        f"- embedding_dir: `{metadata['embedding_dir']}`",
        f"- sample_count: `{metadata['sample_count']}`",
        "",
        "## Latency",
        "",
        f"- imageization per-window mean ms: {float(latency_mean.get('imageization_per_window_ms', np.nan)):.6f}",
        f"- encoder forward per-window mean ms: {float(latency_mean.get('encoder_forward_per_window_ms', np.nan)):.6f}",
        f"- write per-window mean ms: {float(latency_mean.get('write_per_window_ms', np.nan)):.6f}",
        "",
        "## 分层计数",
        "",
        "\n".join(count_lines),
        "",
        "## 输入排除项",
        "",
        "- 未读取未来 y 作为 embedding 输入。",
        "- 未读取专家误差、regret 或 oracle_model 作为 embedding 输入。",
        "",
    ]
    (output_dir / "embedding_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 Stage 1 ViT embedding 构建和校验。"""
    args = parse_args()
    manifest_df, latency_df, metadata, output_dir = build_embeddings(args)
    validate_manifest(manifest_df, args.labels_path, args.metric)
    manifest_df.to_csv(output_dir / "embedding_manifest.csv", index=False)
    latency_df.to_csv(output_dir / "embedding_latency_summary.csv", index=False)
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir, manifest_df, latency_df, metadata)

    print(f"wrote ViT embeddings to {output_dir}")
    print(f"sample_count={len(manifest_df)} embedding_dim={metadata['embedding_dim']} device={metadata['device']}")
    preview_cols = ["sample_key", "split", "variant", "encoder_name", "pooling", "embedding_dim", "embedding_path"]
    print(manifest_df[preview_cols].head(args.print_rows).to_string(index=False))


if __name__ == "__main__":
    main()
