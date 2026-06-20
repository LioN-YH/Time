#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router V2 Round 1 RevIN aux 与 pooling 消融构建 sharded pilot
    feature cache。

核心约束：
    - 只处理 P0 固定 sample CSV，默认不处理 pilot_test；
    - 严格按 P0 `order_index` 顺序写出 shard；
    - 只使用历史窗口 x 生成 pseudo image、frozen ViT CLS/mean-patch feature 和
      6 维 RevIN aux；
    - 不训练 router/head/encoder，不读取 prediction manifest，不保存 pseudo image tensor。
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
from visual_router_experiments.common.vit_embedding_utils import make_pseudo_images, resolve_dtype  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import (  # noqa: E402
    DEFAULT_CONFIG,
    _timer_start,
    _timer_stop,
    load_data_config,
    mode_from_split,
    resolve_device,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    load_checkpoint,
    load_vit_model_with_retry,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import (  # noqa: E402
    AUX_FEATURE_COLUMNS,
    DEFAULT_ROUND1_SAMPLE_SETS,
    FEATURE_SCHEMA_VERSION,
    FINAL_TEST_ONLY_SAMPLE_SET,
    atomic_write_json,
    compute_revin_aux_from_x,
    load_and_validate_sample_csv,
    validate_existing_shard,
    write_feature_shard_atomic,
)


DEFAULT_P0_SAMPLE_DIR = Path("/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples")
DEFAULT_ROUND0_DIR = Path("/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0")
DEFAULT_VISUAL_CHECKPOINT = Path(
    "/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt"
)
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features")
SCRIPT_VERSION = "visual_router_v2_round1_feature_builder_v1"


def now_cst() -> str:
    """函数功能：返回 metadata/status 使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round 1 feature cache builder 参数。"""
    parser = argparse.ArgumentParser(description="Build Visual Router V2 Round 1 sharded pilot feature cache.")
    parser.add_argument("--p0-sample-dir", type=Path, default=DEFAULT_P0_SAMPLE_DIR, help="P0 pilot sample set 输出目录。")
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR, help="P1 Round 0 输出目录，仅用于 metadata lineage。")
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL_CHECKPOINT, help="Visual full-scale checkpoint。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config，用于重新加载历史窗口 x。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Round 1 feature cache 输出目录。")
    parser.add_argument("--sample-sets", nargs="+", default=list(DEFAULT_ROUND1_SAMPLE_SETS), help="要处理的 sample set；默认不含 pilot_test。")
    parser.add_argument("--include-pilot-test-final-test-only", action="store_true", help="显式允许生成 pilot_test，并在 metadata 标记 final_test_only。")
    parser.add_argument("--shard-size", type=int, default=2000, help="每个 feature shard 的样本数。")
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="ViT 前向 batch size。")
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="smoke 截断每个 sample_set 的样本数。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ViT 运行设备。")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default=None, help="覆盖 checkpoint 中的 dtype_arg；默认使用 checkpoint 口径。")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地 Hugging Face cache，不联网下载。")
    parser.add_argument("--vit-data-parallel", action="store_true", help="CUDA 多卡可用时对冻结 ViT 前向使用 DataParallel。")
    parser.add_argument("--overwrite", action="store_true", help="允许重算并覆盖已有 shard。")
    parser.add_argument("--status-update-interval", type=int, default=1, help="每完成多少个 shard 更新一次 status.json。")
    return parser.parse_args()


def make_encoder_args(args: argparse.Namespace, embedding_metadata: Mapping[str, object]) -> SimpleNamespace:
    """
    函数功能：
        从 checkpoint embedding metadata 构造 frozen ViT / pseudo image 所需参数。

    说明：
        Round 1 cache 必须复用当前 Visual Router checkpoint 对应的伪图像口径，因此默认
        不从命令行自由指定 variant/norm/pooling，而是直接读取 checkpoint。
    """
    dtype_arg = str(args.dtype or embedding_metadata.get("dtype_arg", "auto"))
    return SimpleNamespace(
        encoder_name=str(embedding_metadata["encoder_name"]),
        variant=str(embedding_metadata["variant"]),
        pooling="cls",
        normalization_preset=str(embedding_metadata["normalization_preset"]),
        embedding_batch_size=int(args.embedding_batch_size),
        image_size=int(embedding_metadata["image_size"]),
        norm_mode=str(embedding_metadata["norm_mode"]),
        pixel_mode=str(embedding_metadata["pixel_mode"]),
        clip=float(embedding_metadata["clip"]),
        period_selection=str(embedding_metadata["period_selection"]),
        period_candidates=None,
        dtype=dtype_arg,
        local_files_only=bool(args.local_files_only),
        vit_data_parallel=bool(args.vit_data_parallel),
    )


class HistoryWindowLoader:
    """
    类功能：
        按 split/dataset/item 懒加载 Quito 数据，并为一个 shard 按原始行顺序取历史窗口 x。

    约束：
        只读取 `window_index : window_index + seq_len` 的历史窗口，不访问预测区间 y。
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
        """函数功能：返回与 shard_df 行顺序一致的 `[N, seq_len, C]` 历史窗口数组。"""
        seq_len = int(self.data_config.seq_len)
        x_windows: List[Optional[np.ndarray]] = [None] * int(len(shard_df))
        indexed = shard_df.reset_index(drop=True).copy()
        # pandas.itertuples 会改写以下划线开头的字段名，因此使用普通列名保存
        # shard 内行号，确保取回的窗口严格写回原始 order_index 顺序。
        indexed["row_pos"] = np.arange(len(indexed), dtype=np.int64)
        group_cols = ["split", "dataset_name", "item_id"]
        for (split, dataset_name, item_id), group in indexed.groupby(group_cols, sort=False):
            split_datasets = self._load_split(str(split))
            if str(dataset_name) not in split_datasets:
                raise ValueError(f"Quito 数据集中找不到 dataset_name={dataset_name} split={split}")
            base_dataset = split_datasets[str(dataset_name)]
            item_dataset = base_dataset.copy() if hasattr(base_dataset, "copy") else deepcopy(base_dataset)
            item_dataset.select_user_data(int(item_id))
            channel_count = int(item_dataset.data.shape[0])
            for row in group.itertuples(index=False):
                sample_key = str(row.sample_key)
                key = PredictionCacheKey(
                    config_name=str(row.config_name),
                    split=str(row.split),
                    dataset_name=str(row.dataset_name),
                    item_id=int(row.item_id),
                    channel_id=int(row.channel_id),
                    window_index=int(row.window_index),
                )
                if key.as_string() != sample_key:
                    raise ValueError(f"sample_key 与稳定元信息不一致：{sample_key} vs {key.as_string()}")
                if int(row.channel_id) >= channel_count:
                    raise ValueError(f"channel_id 越界：sample_key={sample_key}")
                window_start = int(row.window_index)
                window_end = window_start + seq_len
                x_window = item_dataset.data[int(row.channel_id), window_start:window_end, :]
                if x_window.shape[0] != seq_len:
                    raise ValueError(f"历史窗口长度不完整：sample_key={sample_key} shape={x_window.shape}")
                x_windows[int(row.row_pos)] = np.asarray(x_window, dtype=np.float32)
        if any(value is None for value in x_windows):
            raise RuntimeError("内部错误：部分 shard 行未填充历史窗口")
        return np.stack([value for value in x_windows if value is not None], axis=0).astype(np.float32)


def forward_visual_features(
    *,
    x_shard: np.ndarray,
    vit_model,
    encoder_args: SimpleNamespace,
    device: torch.device,
    dtype: torch.dtype,
    period_candidate_values: Optional[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    函数功能：
        对一个 shard 分 batch 执行 pseudo image 与 frozen ViT 前向，返回 CLS 和
        patch-token mean pooling。
    """
    cls_chunks: List[np.ndarray] = []
    mean_patch_chunks: List[np.ndarray] = []
    latency_rows: List[Dict[str, object]] = []
    batch_size = int(encoder_args.embedding_batch_size)
    for batch_start in range(0, int(x_shard.shape[0]), batch_size):
        batch_end = min(batch_start + batch_size, int(x_shard.shape[0]))
        x_cpu = torch.from_numpy(x_shard[batch_start:batch_end]).to(dtype=torch.float32)
        with torch.inference_mode():
            image_start = _timer_start(device)
            pixel_values = make_pseudo_images(
                x_cpu,
                variant=encoder_args.variant,
                norm_mode=encoder_args.norm_mode,
                pixel_mode=encoder_args.pixel_mode,
                clip=float(encoder_args.clip),
                image_size=int(encoder_args.image_size),
                device=device,
                dtype=dtype,
                normalization_preset=encoder_args.normalization_preset,
                period_selection=encoder_args.period_selection,
                period_candidate_values=period_candidate_values,
            )
            image_ms = _timer_stop(image_start, device)
            forward_start = _timer_start(device)
            outputs = vit_model(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state
            cls_embedding = hidden[:, 0, :]
            # mean_patch 明确只聚合 patch tokens，不包含 CLS token。
            mean_patch_embedding = hidden[:, 1:, :].mean(dim=1)
            forward_ms = _timer_stop(forward_start, device)
            cls_chunks.append(cls_embedding.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float32))
            mean_patch_chunks.append(mean_patch_embedding.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float32))
            del pixel_values, outputs, hidden, cls_embedding, mean_patch_embedding
        if device.type == "cuda":
            torch.cuda.empty_cache()
        latency_rows.append(
            {
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "batch_size": int(batch_end - batch_start),
                "imageization_ms": float(image_ms),
                "encoder_forward_ms": float(forward_ms),
            }
        )
    cls = np.concatenate(cls_chunks, axis=0).astype(np.float32)
    mean_patch = np.concatenate(mean_patch_chunks, axis=0).astype(np.float32)
    return cls, mean_patch, latency_rows


def sample_csv_path(sample_dir: Path, sample_set: str) -> Path:
    """函数功能：根据 sample_set 返回 P0 CSV 路径。"""
    return Path(sample_dir) / f"{sample_set}_sample_keys.csv"


def make_manifest_row(
    *,
    sample_set: str,
    shard_id: int,
    shard_path: Path,
    shard_df: pd.DataFrame,
    visual_feature_dim: int,
    aux_feature_dim: int,
    encoder_args: SimpleNamespace,
    file_size_mb: float,
    final_test_only: bool,
) -> Dict[str, object]:
    """函数功能：构造 round1_feature_manifest.csv 的单行记录。"""
    return {
        "sample_set": str(sample_set),
        "shard_id": int(shard_id),
        "shard_path": str(shard_path),
        "start_order_index": int(shard_df["order_index"].iloc[0]),
        "end_order_index": int(shard_df["order_index"].iloc[-1]),
        "sample_count": int(len(shard_df)),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "visual_feature_dim": int(visual_feature_dim),
        "aux_feature_dim": int(aux_feature_dim),
        "encoder_name": str(encoder_args.encoder_name),
        "pseudo_image_variant": str(encoder_args.variant),
        "norm_mode": str(encoder_args.norm_mode),
        "pooling_available": "cls_embedding,mean_patch_embedding",
        "finite": True,
        "file_size_mb": float(file_size_mb),
        "final_test_only": bool(final_test_only),
    }


def rebuild_size_summary(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按 manifest 重建 cache size summary，并计算累计大小。"""
    rows: List[Dict[str, object]] = []
    cumulative = 0.0
    for row in manifest_df.itertuples(index=False):
        cumulative += float(row.file_size_mb)
        rows.append(
            {
                "sample_set": str(row.sample_set),
                "shard_id": int(row.shard_id),
                "shard_path": str(row.shard_path),
                "sample_count": int(row.sample_count),
                "file_size_mb": float(row.file_size_mb),
                "cumulative_size_mb": float(cumulative),
            }
        )
    return pd.DataFrame(rows)


def simple_markdown_table(frame: pd.DataFrame) -> str:
    """函数功能：不依赖 tabulate 生成小型 Markdown 表格，便于 quito 环境直接运行。"""
    if frame.empty:
        return "| 无 |\\n| --- |"
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_chinese_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    metadata: Mapping[str, object],
    manifest_df: pd.DataFrame,
    size_df: pd.DataFrame,
    smoke_mode: bool,
) -> None:
    """函数功能：写出中文 Round 1 feature cache summary。"""
    counts = manifest_df.groupby("sample_set")["sample_count"].sum().reset_index()
    total_size = float(size_df["file_size_mb"].sum()) if not size_df.empty else 0.0
    lines = [
        "# Visual Router V2 Round 1 Feature Cache Summary",
        "",
        f"生成时间：{now_cst()}",
        "",
        "## 输入路径",
        "",
        f"- P0 sample set：`{args.p0_sample_dir}`",
        f"- P1 Round 0 output：`{args.round0_dir}`",
        f"- Visual checkpoint：`{args.visual_checkpoint}`",
        f"- Quito config：`{args.config_path}`",
        "",
        "## 输出结构",
        "",
        f"- 输出目录：`{output_dir}`",
        "- `features/<sample_set>/shard_XXXXX.npz`：sharded feature cache",
        "- `round1_feature_manifest.csv`：按 sample_set/order_index 恢复顺序的 shard manifest",
        "- `round1_feature_cache_size_summary.csv`：shard 与累计缓存大小",
        "- `round1_feature_metadata.json`、`status.json`：运行参数、状态和 lineage",
        "",
        "## Feature Schema",
        "",
        f"- schema version：`{FEATURE_SCHEMA_VERSION}`",
        "- 每个 shard 包含 `sample_key`、`order_index`、`cls_embedding`、`mean_patch_embedding`、`revin_aux`。",
        "- `cls_embedding` 来自 ViT CLS token。",
        "- `mean_patch_embedding` 只对 patch tokens 求均值，即 `last_hidden_state[:, 1:, :].mean(dim=1)`，不包含 CLS token。",
        f"- `revin_aux` 为 6 维 raw aux：{', '.join(AUX_FEATURE_COLUMNS)}；只由历史窗口 x 计算。",
        "- 本步骤不 fit scaler，不保存 `cls_mean_concat_embedding`，后续训练时按需 concat。",
        "",
        "## Sample Counts",
        "",
        simple_markdown_table(counts),
        "",
        "## Cache Size",
        "",
        f"- 总缓存大小：{total_size:.3f} MB",
        f"- shard 数：{len(manifest_df)}",
        "",
        "## Smoke 与正式运行结果",
        "",
        f"- 当前运行模式：{'smoke' if smoke_mode else '正式 P2a'}",
        f"- `status.json` 状态：`{metadata.get('status')}`",
        f"- 所有 shard finite：{bool(manifest_df['finite'].all()) if not manifest_df.empty else False}",
        f"- resume/skip existing 机制：`--overwrite` 未开启时会校验已有 shard 并跳过；本次 skipped_shards={metadata.get('skipped_shards', 0)}。",
        "",
        "## 使用边界",
        "",
        "该 cache 只覆盖 Round 1 pilot 的固定 P0 sample sets，用于 RevIN aux 与 pooling 消融。它不是 full-scale embedding cache：未处理全量 2327 万样本、未读取 116M prediction manifest、未保存 pseudo image tensor，默认也不生成 pilot_test feature，避免把 final test 用于架构选择。",
        "",
    ]
    (output_dir / "round1_feature_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    """函数功能：执行 Round 1 sharded feature cache 构建与最终校验。"""
    args = parse_args()
    start_time = time.perf_counter()
    if int(args.shard_size) <= 0:
        raise ValueError("--shard-size 必须大于 0")
    if int(args.embedding_batch_size) <= 0:
        raise ValueError("--embedding-batch-size 必须大于 0")
    sample_sets = [str(value) for value in args.sample_sets]
    if FINAL_TEST_ONLY_SAMPLE_SET in sample_sets and not bool(args.include_pilot_test_final_test_only):
        raise ValueError("默认禁止处理 pilot_test；如确需 final test feature，必须显式传入 --include-pilot-test-final-test-only")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, object] = {
        "status": "running",
        "phase": "init",
        "started_at": now_cst(),
        "script": str(Path(__file__).resolve()),
        "current_sample_set": None,
        "current_shard_id": None,
        "processed_count": 0,
        "completed_shards": 0,
        "skipped_shards": 0,
        "failed_reason": None,
    }
    atomic_write_json(output_dir / "status.json", status)

    try:
        checkpoint = load_checkpoint(Path(args.visual_checkpoint))
        embedding_metadata = checkpoint.get("embedding_metadata")
        if not isinstance(embedding_metadata, Mapping):
            raise ValueError("Visual checkpoint 缺少 embedding_metadata")
        encoder_args = make_encoder_args(args, embedding_metadata)
        device = resolve_device(str(args.device))
        dtype = resolve_dtype(str(encoder_args.dtype), device)
        data_config = load_data_config(Path(args.config_path))
        period_candidate_values = embedding_metadata.get("period_candidates")
        if period_candidate_values is not None:
            period_candidate_values = [int(value) for value in period_candidate_values]
        vit_model = load_vit_model_with_retry(encoder_args, device, dtype)
        loader = HistoryWindowLoader(data_config)

        manifest_rows: List[Dict[str, object]] = []
        latency_rows: List[Dict[str, object]] = []
        expected_counts: Dict[str, int] = {}
        status["phase"] = "process_shards"
        atomic_write_json(output_dir / "status.json", status)

        for sample_set in sample_sets:
            final_test_only = sample_set == FINAL_TEST_ONLY_SAMPLE_SET
            sample_df = load_and_validate_sample_csv(
                sample_csv_path(Path(args.p0_sample_dir), sample_set),
                sample_set=sample_set,
                max_samples=args.max_samples_per_set,
            )
            expected_counts[sample_set] = int(len(sample_df))
            set_dir = output_dir / "features" / sample_set
            set_dir.mkdir(parents=True, exist_ok=True)
            shard_count = int(np.ceil(len(sample_df) / int(args.shard_size)))
            for shard_id in range(shard_count):
                start = shard_id * int(args.shard_size)
                end = min(start + int(args.shard_size), len(sample_df))
                shard_df = sample_df.iloc[start:end].reset_index(drop=True)
                shard_path = set_dir / f"shard_{shard_id:05d}.npz"
                sample_keys = shard_df["sample_key"].astype(str).tolist()
                order_index = shard_df["order_index"].to_numpy(dtype=np.int64, copy=False)
                status.update(
                    {
                        "current_sample_set": sample_set,
                        "current_shard_id": int(shard_id),
                        "phase": "process_shards",
                    }
                )
                atomic_write_json(output_dir / "status.json", status)

                if shard_path.exists() and not bool(args.overwrite):
                    sample_count, visual_dim, aux_dim = validate_existing_shard(
                        shard_path=shard_path,
                        expected_sample_keys=sample_keys,
                        expected_order_index=order_index,
                    )
                    status["skipped_shards"] = int(status["skipped_shards"]) + 1
                    file_size_mb = shard_path.stat().st_size / (1024.0 * 1024.0)
                else:
                    x_shard = loader.load_shard_x(shard_df)
                    revin_aux = compute_revin_aux_from_x(x_shard, clip=float(encoder_args.clip))
                    cls_embedding, mean_patch_embedding, shard_latency = forward_visual_features(
                        x_shard=x_shard,
                        vit_model=vit_model,
                        encoder_args=encoder_args,
                        device=device,
                        dtype=dtype,
                        period_candidate_values=period_candidate_values,
                    )
                    write_feature_shard_atomic(
                        shard_path=shard_path,
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
                        latency_rows.append({"sample_set": sample_set, "shard_id": int(shard_id), **row})
                    del x_shard, revin_aux, cls_embedding, mean_patch_embedding

                manifest_rows.append(
                    make_manifest_row(
                        sample_set=sample_set,
                        shard_id=shard_id,
                        shard_path=shard_path,
                        shard_df=shard_df,
                        visual_feature_dim=visual_dim,
                        aux_feature_dim=aux_dim,
                        encoder_args=encoder_args,
                        file_size_mb=file_size_mb,
                        final_test_only=final_test_only,
                    )
                )
                status["completed_shards"] = int(status["completed_shards"]) + 1
                status["processed_count"] = int(status["processed_count"]) + int(sample_count)
                if int(status["completed_shards"]) % int(args.status_update_interval) == 0:
                    atomic_write_json(output_dir / "status.json", status)

        manifest_df = pd.DataFrame(manifest_rows)
        if manifest_df.empty:
            raise ValueError("未生成任何 feature shard")
        # 保持命令行 sample_set 顺序，同时保证每个集合内可按 order_index 恢复 P0 CSV。
        sample_set_rank = {name: idx for idx, name in enumerate(sample_sets)}
        manifest_df["_sample_set_rank"] = manifest_df["sample_set"].map(sample_set_rank)
        manifest_df = (
            manifest_df.sort_values(["_sample_set_rank", "start_order_index"], kind="mergesort")
            .drop(columns=["_sample_set_rank"])
            .reset_index(drop=True)
        )
        count_by_set = manifest_df.groupby("sample_set")["sample_count"].sum().to_dict()
        if {str(k): int(v) for k, v in count_by_set.items()} != expected_counts:
            raise ValueError(f"manifest sample_count 与 P0 CSV 不一致：manifest={count_by_set} expected={expected_counts}")
        manifest_path = output_dir / "round1_feature_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        size_df = rebuild_size_summary(manifest_df)
        size_df.to_csv(output_dir / "round1_feature_cache_size_summary.csv", index=False)
        if latency_rows:
            pd.DataFrame(latency_rows).to_csv(output_dir / "round1_feature_latency.csv", index=False)

        elapsed_sec = time.perf_counter() - start_time
        metadata: Dict[str, object] = {
            "status": "completed",
            "generated_at": now_cst(),
            "elapsed_sec": float(elapsed_sec),
            "script": str(Path(__file__).resolve()),
            "script_version": SCRIPT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "p0_sample_dir": str(args.p0_sample_dir),
            "round0_dir": str(args.round0_dir),
            "visual_checkpoint": str(args.visual_checkpoint),
            "config_path": str(args.config_path),
            "output_dir": str(output_dir),
            "sample_sets": sample_sets,
            "sample_counts": expected_counts,
            "default_excludes_pilot_test": FINAL_TEST_ONLY_SAMPLE_SET not in sample_sets,
            "final_test_only_sets": [FINAL_TEST_ONLY_SAMPLE_SET] if FINAL_TEST_ONLY_SAMPLE_SET in sample_sets else [],
            "max_samples_per_set": args.max_samples_per_set,
            "shard_size": int(args.shard_size),
            "embedding_batch_size": int(args.embedding_batch_size),
            "device": str(device),
            "dtype": str(dtype),
            "local_files_only": bool(args.local_files_only),
            "overwrite": bool(args.overwrite),
            "skipped_shards": int(status["skipped_shards"]),
            "completed_shards": int(status["completed_shards"]),
            "total_cache_size_mb": float(size_df["file_size_mb"].sum()),
            "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
            "visual_feature_columns": ["cls_embedding", "mean_patch_embedding"],
            "embedding_metadata_from_checkpoint": dict(embedding_metadata),
            "feature_constraints": {
                "mean_patch_excludes_cls": True,
                "revin_aux_source": "history_x_only",
                "fit_scaler": False,
                "pseudo_image_tensor_saved": False,
                "read_prediction_manifest": False,
                "train_router_or_encoder": False,
                "full_scale_embedding_cache": False,
            },
        }
        atomic_write_json(output_dir / "round1_feature_metadata.json", metadata)
        status.update(
            {
                "status": "completed",
                "phase": "done",
                "current_sample_set": None,
                "current_shard_id": None,
                "failed_reason": None,
                "elapsed_sec": float(elapsed_sec),
            }
        )
        atomic_write_json(output_dir / "status.json", status)
        write_chinese_summary(
            output_dir=output_dir,
            args=args,
            metadata=metadata,
            manifest_df=manifest_df,
            size_df=size_df,
            smoke_mode=args.max_samples_per_set is not None,
        )
        print(json.dumps({"status": "completed", "output_dir": str(output_dir), "sample_counts": expected_counts}, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        status.update({"status": "failed", "phase": "failed", "failed_reason": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(output_dir / "status.json", status)
        raise


if __name__ == "__main__":
    run()
