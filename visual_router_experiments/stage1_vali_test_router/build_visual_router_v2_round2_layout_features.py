#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router V2 Round2c 的单个或多个 layout 构建 frozen ViT feature cache。

核心约束：
    - 只读取 Round2 已冻结 small sample manifest 中的历史窗口 x；
    - layout 图像化通过 Round2 registry 运行在 torch tensor path；
    - ViT encoder 固定为 Round1 visual checkpoint 中记录的 frozen ViT 口径；
    - 不读取专家 prediction/oracle label 作为图像化输入，不保存 pseudo image tensor；
    - layout 级输出目录彼此隔离，统一 manifest 只由 aggregate step 写出。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = LEGACY_WORKSPACE / "quito"
for path in [REPO_ROOT, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config.training import TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402
from visual_router_experiments.common.pseudo_imageization import encoder_normalize  # noqa: E402
from visual_router_experiments.common.round2_layout_registry import DEFAULT_ROUND2_LAYOUTS, imageize_round2_layout, list_layout_specs  # noqa: E402
from visual_router_experiments.common.vit_embedding_utils import resolve_dtype  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import DEFAULT_CONFIG, load_data_config, mode_from_split, resolve_device  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import load_checkpoint, load_vit_model_with_retry  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS, compute_revin_aux_from_x  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_MANIFEST = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_small_samples" / "round2_small_sample_manifest.csv"
DEFAULT_LAYOUT_CANDIDATES = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_small_samples" / "round2_layout_candidates.json"
DEFAULT_VISUAL_CHECKPOINT = DATA2_RUN_OUTPUT_ROOT / "2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2" / "checkpoints" / "latest_96_48_S.pt"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_layout_screening"
ROUND2_SAMPLE_SETS = (
    "round2_train_small",
    "round2_selection_small",
    "round2_diagnostic_balanced_small",
    "round2_test_small",
)
FEATURE_SCHEMA_VERSION = "visual_router_v2_round2_layout_feature_cache_v1"
SCRIPT_VERSION = "visual_router_v2_round2c_layout_feature_builder_v1"


def now_cst() -> str:
    """函数功能：生成写入 metadata/status 的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


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


def parse_period_candidates(text: Optional[str]) -> Optional[List[int]]:
    """函数功能：解析可选周期候选列表。"""
    if text is None or str(text).strip() == "":
        return None
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        return None
    if min(values) < 2:
        raise ValueError("--period-candidates 中所有值必须 >=2")
    return values


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2c layout feature cache 参数。"""
    parser = argparse.ArgumentParser(description="Build Round2c layout frozen ViT feature cache.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--layout-candidates", type=Path, default=DEFAULT_LAYOUT_CANDIDATES)
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL_CHECKPOINT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layouts", default=",".join(DEFAULT_ROUND2_LAYOUTS))
    parser.add_argument("--layout", default=None, help="单 layout worker 模式；launcher 会使用该参数。")
    parser.add_argument("--sample-sets", default=",".join(ROUND2_SAMPLE_SETS))
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke。")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates")
    parser.add_argument("--period-candidates", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：用同目录临时文件原子写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def make_encoder_args(args: argparse.Namespace, embedding_metadata: Mapping[str, object]) -> SimpleNamespace:
    """
    函数功能：
        从 Round1 checkpoint metadata 构造 frozen ViT 所需参数。
    """
    return SimpleNamespace(
        encoder_name=str(embedding_metadata["encoder_name"]),
        pooling="cls",
        dtype=str(args.dtype or embedding_metadata.get("dtype_arg", "auto")),
        local_files_only=bool(args.local_files_only),
        vit_data_parallel=False,
        normalization_preset=str(embedding_metadata["normalization_preset"]),
    )


class Round2HistoryWindowLoader:
    """
    类功能：
        按 split/dataset/item 懒加载 Quito 数据，为 Round2 sample manifest 读取历史窗口。

    约束：
        只读取 `window_index : window_index + seq_len` 的历史窗口，不访问 future y。
    """

    def __init__(self, data_config) -> None:
        self.data_config = data_config
        self._datasets_by_split: Dict[str, Dict[str, object]] = {}

    def _load_split(self, split: str) -> Dict[str, object]:
        if split not in self._datasets_by_split:
            datasets = load_datasets(
                data_config=self.data_config,
                task=TaskType.EVALUATE,
                mode=mode_from_split(str(split)),
                cleanup=False,
                concat=False,
            )
            mapping: Dict[str, object] = {}
            for dataset_idx, dataset in enumerate(datasets):
                dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
                mapping[str(dataset_name)] = dataset
            self._datasets_by_split[str(split)] = mapping
        return self._datasets_by_split[str(split)]

    def load_shard_x(self, shard_df: pd.DataFrame) -> np.ndarray:
        """函数功能：返回与 shard_df 行顺序一致的 `[N,L,C]` 历史窗口数组。"""
        seq_len = int(self.data_config.seq_len)
        indexed = shard_df.reset_index(drop=True).copy()
        indexed["row_pos"] = np.arange(len(indexed), dtype=np.int64)
        windows: List[Optional[np.ndarray]] = [None] * int(len(indexed))
        for (split, dataset_name, item_id), group in indexed.groupby(["split", "dataset_name", "item_id"], sort=False):
            split_datasets = self._load_split(str(split))
            if str(dataset_name) not in split_datasets:
                raise ValueError(f"Quito 数据集中找不到 dataset_name={dataset_name} split={split}")
            item_dataset = deepcopy(split_datasets[str(dataset_name)])
            item_dataset.select_user_data(int(item_id))
            channel_count = int(item_dataset.data.shape[0])
            for row in group.itertuples(index=False):
                key = PredictionCacheKey(
                    config_name=str(row.config_name),
                    split=str(row.split),
                    dataset_name=str(row.dataset_name),
                    item_id=int(row.item_id),
                    channel_id=int(row.channel_id),
                    window_index=int(row.window_index),
                )
                if key.as_string() != str(row.sample_key):
                    raise ValueError(f"sample_key 与稳定元信息不一致：{row.sample_key} vs {key.as_string()}")
                if int(row.channel_id) >= channel_count:
                    raise ValueError(f"channel_id 越界：sample_key={row.sample_key}")
                start = int(row.window_index)
                stop = start + seq_len
                x_window = item_dataset.data[int(row.channel_id), start:stop, :]
                if x_window.shape[0] != seq_len:
                    raise ValueError(f"历史窗口长度不完整：sample_key={row.sample_key} shape={x_window.shape}")
                windows[int(row.row_pos)] = np.asarray(x_window, dtype=np.float32)
        if any(value is None for value in windows):
            raise RuntimeError("内部错误：部分 shard 未读取到历史窗口")
        return np.stack([value for value in windows if value is not None], axis=0).astype(np.float32)


def load_round2_samples(path: Path, sample_sets: Sequence[str], max_samples_per_set: Optional[int]) -> Dict[str, pd.DataFrame]:
    """函数功能：读取并校验 Round2 small sample manifest，按 sample_set 返回有序表。"""
    frame = pd.read_csv(path)
    required = {"sample_set", "order_index", "sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{missing}")
    result: Dict[str, pd.DataFrame] = {}
    for sample_set in sample_sets:
        part = frame[frame["sample_set"].astype(str) == str(sample_set)].sort_values("order_index", kind="mergesort").reset_index(drop=True)
        if max_samples_per_set is not None:
            part = part.head(int(max_samples_per_set)).copy()
        if part.empty:
            raise ValueError(f"sample_set={sample_set} 为空")
        order_index = part["order_index"].to_numpy(dtype=np.int64, copy=False)
        expected = np.arange(0, len(part), dtype=np.int64)
        if not np.array_equal(order_index, expected):
            raise ValueError(f"{sample_set} order_index 必须从 0 连续递增")
        if part["sample_key"].astype(str).duplicated().any():
            raise ValueError(f"{sample_set} 存在重复 sample_key")
        result[str(sample_set)] = part
    return result


def write_layout_feature_shard(
    *,
    shard_path: Path,
    sample_set: str,
    layout_name: str,
    sample_keys: Sequence[str],
    order_index: np.ndarray,
    cls_embedding: np.ndarray,
    mean_patch_embedding: np.ndarray,
    revin_aux: np.ndarray,
) -> None:
    """函数功能：写出一个 Round2 layout feature shard，并保存 layout/sample_set 来源字段。"""
    sample_count = len(sample_keys)
    if cls_embedding.shape != mean_patch_embedding.shape:
        raise ValueError("cls_embedding 与 mean_patch_embedding shape 不一致")
    if cls_embedding.shape[0] != sample_count or revin_aux.shape != (sample_count, len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"feature shape 异常：cls={cls_embedding.shape} aux={revin_aux.shape} count={sample_count}")
    if not (np.isfinite(cls_embedding).all() and np.isfinite(mean_patch_embedding).all() and np.isfinite(revin_aux).all()):
        raise ValueError("feature shard 中存在 NaN/Inf")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_key=np.asarray([str(key) for key in sample_keys], dtype=object),
            order_index=np.asarray(order_index, dtype=np.int64),
            layout_name=np.asarray([str(layout_name)] * sample_count, dtype=object),
            sample_set=np.asarray([str(sample_set)] * sample_count, dtype=object),
            cls_embedding=np.asarray(cls_embedding, dtype=np.float32),
            mean_patch_embedding=np.asarray(mean_patch_embedding, dtype=np.float32),
            revin_aux=np.asarray(revin_aux, dtype=np.float32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(shard_path)


def validate_existing_shard(shard_path: Path, *, sample_keys: Sequence[str], order_index: np.ndarray, layout_name: str, sample_set: str) -> Tuple[int, int, int]:
    """函数功能：校验已有 shard 是否可安全复用。"""
    with np.load(shard_path, allow_pickle=True) as data:
        keys = [str(value) for value in data["sample_key"].tolist()]
        orders = np.asarray(data["order_index"], dtype=np.int64)
        layouts = {str(value) for value in data["layout_name"].tolist()}
        sets = {str(value) for value in data["sample_set"].tolist()}
        cls = np.asarray(data["cls_embedding"], dtype=np.float32)
        mean_patch = np.asarray(data["mean_patch_embedding"], dtype=np.float32)
        aux = np.asarray(data["revin_aux"], dtype=np.float32)
    if keys != [str(key) for key in sample_keys] or not np.array_equal(orders, np.asarray(order_index, dtype=np.int64)):
        raise ValueError(f"已有 shard 与 manifest 顺序不一致：{shard_path}")
    if layouts != {str(layout_name)} or sets != {str(sample_set)}:
        raise ValueError(f"已有 shard layout/sample_set 字段不一致：{shard_path}")
    if cls.shape != mean_patch.shape or cls.shape[0] != len(keys) or aux.shape != (len(keys), len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"已有 shard shape 异常：{shard_path}")
    if not (np.isfinite(cls).all() and np.isfinite(mean_patch).all() and np.isfinite(aux).all()):
        raise ValueError(f"已有 shard 存在 NaN/Inf：{shard_path}")
    return int(len(keys)), int(cls.shape[1]), int(aux.shape[1])


def forward_layout_features(
    *,
    x_shard: np.ndarray,
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
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]], Dict[str, object]]:
    """函数功能：对一个 shard 生成指定 layout 的 CLS 和 mean-patch ViT feature。"""
    cls_chunks: List[np.ndarray] = []
    mean_chunks: List[np.ndarray] = []
    latency_rows: List[Dict[str, object]] = []
    first_metadata: Dict[str, object] = {}
    for batch_start in range(0, int(x_shard.shape[0]), int(batch_size)):
        batch_end = min(batch_start + int(batch_size), int(x_shard.shape[0]))
        x_batch = torch.from_numpy(x_shard[batch_start:batch_end]).to(dtype=torch.float32)
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            image_start = time.perf_counter()
            result = imageize_round2_layout(
                x_batch.to(device=device, dtype=torch.float32),
                layout_name=layout_name,
                image_size=int(image_size),
                norm_mode=str(norm_mode),
                clip=float(clip),
                period_candidates=period_candidates,
                period_selection=str(period_selection),
            )
            images = result.images
            if images.ndim != 4 or images.shape[1:] != (3, int(image_size), int(image_size)):
                raise ValueError(f"{layout_name} 输出 shape 异常：{tuple(images.shape)}")
            if not torch.isfinite(images).all() or images.min() < -1e-6 or images.max() > 1.0 + 1e-6:
                raise ValueError(f"{layout_name} 输出存在非有限值或越出 [0,1]")
            pixel_values = encoder_normalize(images.to(dtype=dtype), preset=normalization_preset)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            image_ms = (time.perf_counter() - image_start) * 1000.0
            if not first_metadata:
                first_metadata = dict(result.metadata)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            outputs = vit_model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state
            cls_embedding = hidden[:, 0, :]
            mean_patch_embedding = hidden[:, 1:, :].mean(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            cls_chunks.append(cls_embedding.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float32))
            mean_chunks.append(mean_patch_embedding.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float32))
            del result, images, pixel_values, outputs, hidden, cls_embedding, mean_patch_embedding
        latency_rows.append(
            {
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "batch_size": int(batch_end - batch_start),
                "imageization_latency_ms": float(image_ms),
                "encoder_forward_ms": float(forward_ms),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(cls_chunks, axis=0).astype(np.float32), np.concatenate(mean_chunks, axis=0).astype(np.float32), latency_rows, first_metadata


def manifest_row(
    *,
    layout_name: str,
    sample_set: str,
    shard_id: int,
    shard_path: Path,
    shard_df: pd.DataFrame,
    visual_feature_dim: int,
    aux_feature_dim: int,
    file_size_mb: float,
) -> Dict[str, object]:
    """函数功能：构造 layout-level feature manifest 行。"""
    return {
        "layout_name": str(layout_name),
        "sample_set": str(sample_set),
        "shard_id": int(shard_id),
        "shard_path": str(shard_path),
        "start_order_index": int(shard_df["order_index"].iloc[0]),
        "end_order_index": int(shard_df["order_index"].iloc[-1]),
        "sample_count": int(len(shard_df)),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "visual_feature_dim": int(visual_feature_dim),
        "aux_feature_dim": int(aux_feature_dim),
        "pooling_available": "cls_embedding,mean_patch_embedding",
        "finite": True,
        "file_size_mb": float(file_size_mb),
    }


def run_layout_worker(args: argparse.Namespace, layout_name: str) -> None:
    """函数功能：构建单个 layout 的 feature cache，输出 layout-level manifest/status。"""
    output_dir = Path(args.output_dir)
    layout_dir = output_dir / "features" / layout_name
    if layout_dir.exists() and args.overwrite:
        shutil.rmtree(layout_dir)
    layout_dir.mkdir(parents=True, exist_ok=True)
    status_path = layout_dir / "layout_status.json"
    status: Dict[str, object] = {
        "status": "running",
        "layout_name": layout_name,
        "started_at": now_cst(),
        "current_sample_set": None,
        "current_shard_id": None,
        "completed_shards": 0,
        "processed_count": 0,
        "failed_reason": None,
    }
    atomic_write_json(status_path, status)
    start_time = time.perf_counter()
    try:
        sample_sets = parse_csv(args.sample_sets)
        samples_by_set = load_round2_samples(args.sample_manifest, sample_sets, args.max_samples_per_set)
        checkpoint = load_checkpoint(Path(args.visual_checkpoint))
        embedding_metadata = checkpoint.get("embedding_metadata")
        if not isinstance(embedding_metadata, Mapping):
            raise ValueError("Visual checkpoint 缺少 embedding_metadata")
        encoder_args = make_encoder_args(args, embedding_metadata)
        device = resolve_device(str(args.device))
        dtype = resolve_dtype(str(encoder_args.dtype), device)
        vit_model = load_vit_model_with_retry(encoder_args, device, dtype)
        data_config = load_data_config(Path(args.config_path))
        loader = Round2HistoryWindowLoader(data_config)
        period_candidates = parse_period_candidates(args.period_candidates)
        manifest_rows: List[Dict[str, object]] = []
        latency_rows: List[Dict[str, object]] = []
        layout_metadata: Dict[str, object] = {}

        for sample_set, sample_df in samples_by_set.items():
            set_dir = layout_dir / sample_set
            set_dir.mkdir(parents=True, exist_ok=True)
            shard_count = int(np.ceil(len(sample_df) / int(args.shard_size)))
            for shard_id in range(shard_count):
                start = shard_id * int(args.shard_size)
                end = min(start + int(args.shard_size), len(sample_df))
                shard_df = sample_df.iloc[start:end].reset_index(drop=True)
                shard_path = set_dir / f"shard_{shard_id:05d}.npz"
                sample_keys = shard_df["sample_key"].astype(str).tolist()
                order_index = shard_df["order_index"].to_numpy(dtype=np.int64, copy=False)
                status.update({"current_sample_set": sample_set, "current_shard_id": int(shard_id)})
                atomic_write_json(status_path, status)
                if shard_path.exists() and not args.overwrite:
                    sample_count, visual_dim, aux_dim = validate_existing_shard(
                        shard_path,
                        sample_keys=sample_keys,
                        order_index=order_index,
                        layout_name=layout_name,
                        sample_set=sample_set,
                    )
                    file_size_mb = shard_path.stat().st_size / (1024.0 * 1024.0)
                else:
                    x_shard = loader.load_shard_x(shard_df)
                    revin_aux = compute_revin_aux_from_x(x_shard, clip=float(args.clip))
                    cls_embedding, mean_patch_embedding, shard_latency, batch_metadata = forward_layout_features(
                        x_shard=x_shard,
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
                        batch_size=int(args.embedding_batch_size),
                    )
                    if not layout_metadata:
                        layout_metadata = batch_metadata
                    write_layout_feature_shard(
                        shard_path=shard_path,
                        sample_set=sample_set,
                        layout_name=layout_name,
                        sample_keys=sample_keys,
                        order_index=order_index,
                        cls_embedding=cls_embedding,
                        mean_patch_embedding=mean_patch_embedding,
                        revin_aux=revin_aux,
                    )
                    sample_count = int(len(sample_keys))
                    visual_dim = int(cls_embedding.shape[1])
                    aux_dim = int(revin_aux.shape[1])
                    file_size_mb = shard_path.stat().st_size / (1024.0 * 1024.0)
                    for row in shard_latency:
                        latency_rows.append({"layout_name": layout_name, "sample_set": sample_set, "shard_id": int(shard_id), **row})
                    del x_shard, revin_aux, cls_embedding, mean_patch_embedding
                manifest_rows.append(
                    manifest_row(
                        layout_name=layout_name,
                        sample_set=sample_set,
                        shard_id=shard_id,
                        shard_path=shard_path,
                        shard_df=shard_df,
                        visual_feature_dim=visual_dim,
                        aux_feature_dim=aux_dim,
                        file_size_mb=file_size_mb,
                    )
                )
                status["completed_shards"] = int(status["completed_shards"]) + 1
                status["processed_count"] = int(status["processed_count"]) + int(sample_count)
                atomic_write_json(status_path, status)

        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(layout_dir / "layout_feature_manifest.csv", index=False)
        if latency_rows:
            pd.DataFrame(latency_rows).to_csv(layout_dir / "layout_feature_latency.csv", index=False)
        metadata = {
            "status": "completed",
            "generated_at": now_cst(),
            "elapsed_sec": float(time.perf_counter() - start_time),
            "script_version": SCRIPT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "layout_name": layout_name,
            "sample_manifest": str(args.sample_manifest),
            "layout_candidates": str(args.layout_candidates),
            "visual_checkpoint": str(args.visual_checkpoint),
            "config_path": str(args.config_path),
            "device": str(device),
            "dtype": str(dtype),
            "sample_sets": sample_sets,
            "sample_counts": {name: int(len(frame)) for name, frame in samples_by_set.items()},
            "shard_size": int(args.shard_size),
            "embedding_batch_size": int(args.embedding_batch_size),
            "imageization": {
                "image_size": int(args.image_size),
                "norm_mode": str(args.norm_mode),
                "clip": float(args.clip),
                "period_selection": str(args.period_selection),
                "period_candidates": period_candidates,
                "normalization_preset": str(encoder_args.normalization_preset),
            },
            "layout_metadata_first_batch": layout_metadata,
            "constraints": {
                "history_x_only_for_imageization": True,
                "read_expert_prediction_as_feature": False,
                "read_oracle_label_as_feature": False,
                "saved_pseudo_image_tensor": False,
                "trained_router_or_encoder": False,
                "mean_patch_excludes_cls": True,
            },
        }
        atomic_write_json(layout_dir / "layout_feature_metadata.json", metadata)
        status.update({"status": "completed", "current_sample_set": None, "current_shard_id": None, "elapsed_sec": metadata["elapsed_sec"]})
        atomic_write_json(status_path, status)
    except Exception as exc:  # noqa: BLE001
        status.update({"status": "failed", "failed_reason": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(status_path, status)
        raise


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：单进程聚合所有 layout-level manifest/metadata，写出统一产物。"""
    output_dir = Path(args.output_dir)
    layouts = [str(args.layout)] if args.layout else parse_csv(args.layouts)
    manifest_frames: List[pd.DataFrame] = []
    latency_frames: List[pd.DataFrame] = []
    layout_metadata: Dict[str, object] = {}
    missing: List[str] = []
    for layout_name in layouts:
        layout_dir = output_dir / "features" / layout_name
        for name in ["layout_feature_manifest.csv", "layout_feature_metadata.json", "layout_status.json"]:
            if not (layout_dir / name).exists():
                missing.append(str(layout_dir / name))
        if missing:
            continue
        manifest_frames.append(pd.read_csv(layout_dir / "layout_feature_manifest.csv"))
        if (layout_dir / "layout_feature_latency.csv").exists():
            latency_frames.append(pd.read_csv(layout_dir / "layout_feature_latency.csv"))
        layout_metadata[layout_name] = json.loads((layout_dir / "layout_feature_metadata.json").read_text(encoding="utf-8"))
    if missing:
        raise FileNotFoundError("layout feature 输出不完整：" + "; ".join(missing[:20]))
    manifest_df = pd.concat(manifest_frames, ignore_index=True)
    layout_rank = {name: idx for idx, name in enumerate(layouts)}
    set_rank = {name: idx for idx, name in enumerate(parse_csv(args.sample_sets))}
    manifest_df["_layout_rank"] = manifest_df["layout_name"].map(layout_rank)
    manifest_df["_set_rank"] = manifest_df["sample_set"].map(set_rank)
    manifest_df = manifest_df.sort_values(["_layout_rank", "_set_rank", "start_order_index"], kind="mergesort").drop(columns=["_layout_rank", "_set_rank"]).reset_index(drop=True)
    manifest_df.to_csv(output_dir / "round2_layout_feature_manifest.csv", index=False)
    size_df = manifest_df[["layout_name", "sample_set", "shard_id", "shard_path", "sample_count", "file_size_mb"]].copy()
    size_df["cumulative_size_mb"] = size_df["file_size_mb"].cumsum()
    size_df.to_csv(output_dir / "round2_layout_feature_cache_size_summary.csv", index=False)
    if latency_frames:
        pd.concat(latency_frames, ignore_index=True).to_csv(output_dir / "round2_layout_feature_latency.csv", index=False)
    metadata = {
        "status": "completed",
        "generated_at": now_cst(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "sample_manifest": str(args.sample_manifest),
        "layout_candidates": str(args.layout_candidates),
        "layouts": layouts,
        "sample_sets": parse_csv(args.sample_sets),
        "total_sample_rows": int(manifest_df["sample_count"].sum()),
        "total_cache_size_mb": float(size_df["file_size_mb"].sum()),
        "registry_specs": list_layout_specs(),
        "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
        "visual_feature_columns": ["cls_embedding", "mean_patch_embedding"],
        "layout_metadata": layout_metadata,
        "constraints": {
            "layout_level_workers": True,
            "unified_manifest_written_by_aggregate_only": True,
            "saved_pseudo_image_tensor": False,
            "loaded_116m_prediction_manifest": False,
            "trained_router_or_encoder": False,
        },
    }
    atomic_write_json(output_dir / "round2_layout_feature_metadata.json", metadata)
    lines = [
        "# Visual Router V2 Round2c Layout Feature Cache Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- layouts：{', '.join(layouts)}",
        f"- sample_sets：{', '.join(parse_csv(args.sample_sets))}",
        f"- shard 数：{len(manifest_df)}",
        f"- feature rows：{int(manifest_df['sample_count'].sum())}",
        f"- 总缓存大小：{float(size_df['file_size_mb'].sum()):.3f} MB",
        "",
        "本步骤只从历史窗口 x 生成 layout pseudo image，并通过 frozen ViT 提取 CLS 与 mean_patch embedding；未保存大规模 pseudo image tensor，未读取专家 prediction 或 oracle label 作为图像化输入。",
    ]
    (output_dir / "round2_layout_feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """函数功能：根据 CLI 模式执行 layout worker、串行 worker 或 aggregation。"""
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
        return
    layouts = [str(args.layout)] if args.layout else parse_csv(args.layouts)
    for layout_name in layouts:
        run_layout_worker(args, layout_name)
    aggregate(args)


if __name__ == "__main__":
    main()
