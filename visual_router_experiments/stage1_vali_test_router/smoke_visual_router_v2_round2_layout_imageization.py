#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2b layout registry / GPU tensor imageization smoke。

核心约束：
    - 只读取 Round2 small sample manifest 中少量样本的历史 x；
    - 不读取 future y、专家 prediction、oracle label 或 116M prediction manifest；
    - 不训练 router，不运行 frozen ViT，不保存大规模 pseudo image tensor；
    - 主 imageization 路径通过 `round2_layout_registry` 使用 torch tensor 操作。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
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
from visual_router_experiments.common.round2_layout_registry import (  # noqa: E402
    DEFAULT_ROUND2_LAYOUTS,
    DEFERRED_ROUND2_LAYOUTS,
    imageize_round2_layout,
    list_layout_specs,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import (  # noqa: E402
    DEFAULT_CONFIG,
    load_data_config,
    mode_from_split,
)


DEFAULT_SAMPLE_MANIFEST = Path(
    "/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_small_sample_manifest.csv"
)
DEFAULT_LAYOUT_CANDIDATES = Path(
    "/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_layout_candidates.json"
)
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_imageization_smoke")
DEFAULT_SUMMARY_COPY_DIR = Path("experiment_summaries/visual_router_v2_round2/layout_imageization_smoke")
DEFAULT_SAMPLE_COUNTS = {
    "round2_train_small": 128,
    "round2_selection_small": 64,
    "round2_diagnostic_balanced_small": 64,
    "round2_test_small": 64,
}
SCRIPT_VERSION = "visual_router_v2_round2b_layout_imageization_smoke_v1"


def now_cst() -> str:
    """函数功能：返回用于 metadata / summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2b imageization smoke 参数。"""
    parser = argparse.ArgumentParser(description="Smoke Round2 layout registry and tensor imageization.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST, help="Round2 small sample manifest CSV。")
    parser.add_argument("--layout-candidates", type=Path, default=DEFAULT_LAYOUT_CANDIDATES, help="Round2 layout candidates JSON，仅用于 lineage。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config，用于读取历史窗口 x。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="smoke 输出目录。")
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_COPY_DIR, help="轻量 summary 复制目录。")
    parser.add_argument("--layouts", default=",".join(DEFAULT_ROUND2_LAYOUTS), help="逗号分隔 layout 名称。")
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="覆盖所有 sample_set 的 smoke 样本数。")
    parser.add_argument("--batch-size", type=int, default=32, help="每次 imageization 的 batch size。")
    parser.add_argument("--device", default="cuda:0", help="imageization 设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--image-size", type=int, default=224, help="输出伪图像 H/W。")
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux", help="历史 x normalization 口径。")
    parser.add_argument("--clip", type=float, default=5.0, help="视觉 pixel 映射截断阈值。")
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates", help="周期选择口径。")
    parser.add_argument("--period-candidates", default=None, help="可选逗号分隔周期候选；默认使用 registry 默认候选。")
    parser.add_argument("--save-debug-thumbnails", action="store_true", help="保存少量 debug PNG。")
    parser.add_argument("--debug-thumbnail-count", type=int, default=8, help="每个 layout 最多保存多少张 debug PNG。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖既有输出目录。")
    return parser.parse_args()


def parse_layouts(text: str) -> List[str]:
    """函数功能：解析逗号分隔 layout 参数，并保持输入顺序去重。"""
    layouts: List[str] = []
    for part in str(text).split(","):
        name = part.strip()
        if name and name not in layouts:
            layouts.append(name)
    if not layouts:
        raise ValueError("--layouts 解析后为空")
    return layouts


def parse_period_candidates(text: Optional[str]) -> Optional[List[int]]:
    """函数功能：解析 CLI 中的周期候选列表。"""
    if text is None or str(text).strip() == "":
        return None
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        return None
    if min(values) < 2:
        raise ValueError("--period-candidates 中所有值必须 >=2")
    return values


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析并验证 imageization 设备。"""
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求设备 {device_arg}，但当前 torch.cuda.is_available() 为 False")
    return requested


def timer_start(device: torch.device) -> float:
    """函数功能：开始计时；CUDA 路径先同步以减少异步误差。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def timer_stop(start: float, device: torch.device) -> float:
    """函数功能：结束计时并返回毫秒。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0


class SmokeHistoryWindowLoader:
    """
    类功能：
        按 split/dataset/item 懒加载 Quito 数据，为 smoke 样本读取历史窗口 x。

    约束：
        只切片 `window_index : window_index + seq_len`，不访问预测区间 y。
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

    def load_x_in_order(self, sample_df: pd.DataFrame) -> np.ndarray:
        """函数功能：返回与 sample_df 当前行顺序一致的 `[N, seq_len, C]` 历史窗口数组。"""
        seq_len = int(self.data_config.seq_len)
        indexed = sample_df.reset_index(drop=True).copy()
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
            raise RuntimeError("内部错误：部分 smoke 样本没有读取到历史窗口")
        return np.stack([value for value in windows if value is not None], axis=0).astype(np.float32)


def select_smoke_samples(manifest: pd.DataFrame, max_samples_per_set: Optional[int]) -> pd.DataFrame:
    """
    函数功能：
        按默认或 CLI 覆盖样本数，从每个 Round2 sample set 取前若干样本。

    说明：
        使用已冻结 manifest 的 order_index 顺序，避免本 smoke 重新抽样引入不可复现差异。
    """
    frames: List[pd.DataFrame] = []
    for sample_set, default_count in DEFAULT_SAMPLE_COUNTS.items():
        limit = int(max_samples_per_set) if max_samples_per_set is not None else int(default_count)
        part = manifest[manifest["sample_set"].astype(str) == sample_set].sort_values("order_index").head(limit).copy()
        if len(part) != limit:
            raise ValueError(f"{sample_set} 样本不足：需要 {limit}，实际 {len(part)}")
        frames.append(part)
    selected = pd.concat(frames, ignore_index=True)
    if selected["sample_key"].astype(str).duplicated().any():
        raise ValueError("smoke 选中样本存在重复 sample_key")
    return selected


def tensor_value_stats(images: torch.Tensor) -> Dict[str, object]:
    """函数功能：计算 layout 输出 tensor 的 shape、dtype、range 和 finite 统计。"""
    detached = images.detach()
    finite = torch.isfinite(detached)
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "min": float(detached.min().cpu().item()),
        "max": float(detached.max().cpu().item()),
        "mean": float(detached.mean().cpu().item()),
        "std": float(detached.std(unbiased=False).cpu().item()),
        "finite_count": int(finite.sum().cpu().item()),
        "total_count": int(detached.numel()),
        "finite_ratio": float(finite.to(dtype=torch.float32).mean().cpu().item()),
        "in_range_0_1": bool((detached.min() >= -1e-6).cpu().item() and (detached.max() <= 1.0 + 1e-6).cpu().item()),
    }


def save_debug_thumbnail(images: torch.Tensor, path: Path) -> None:
    """
    函数功能：
        将单张 `[3,H,W]` pseudo image 保存为 PNG，用于人工检查。

    约束：
        这是 debug-only 路径，允许 detach 到 CPU；主 smoke 不保存大规模 tensor。
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    array = images.detach().clamp(0.0, 1.0).to(device="cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def summarize_period_metadata(layout_name: str, metadata: Mapping[str, object], batch_size: int) -> Dict[str, object]:
    """函数功能：把 layout side metadata 中的周期信息压平成 CSV 行。"""
    period_summary = metadata.get("period_summary", {})
    padding = metadata.get("padding", {})
    return {
        "layout_name": layout_name,
        "batch_size": int(batch_size),
        "selected_periods_shape": json.dumps(metadata.get("selected_periods_shape", []), ensure_ascii=False),
        "period_score_mean_by_rank": json.dumps(period_summary.get("period_score_mean_by_rank", []), ensure_ascii=False) if isinstance(period_summary, Mapping) else "[]",
        "period_score_max_by_rank": json.dumps(period_summary.get("period_score_max_by_rank", []), ensure_ascii=False) if isinstance(period_summary, Mapping) else "[]",
        "top1_period_bucket_counts": json.dumps(period_summary.get("top1_period_bucket_counts", {}), ensure_ascii=False, sort_keys=True) if isinstance(period_summary, Mapping) else "{}",
        "padding_mask_available": bool(padding.get("padding_mask_available", False)) if isinstance(padding, Mapping) else False,
        "padding_mask_used_as_vit_input": bool(padding.get("padding_mask_used_as_vit_input", False)) if isinstance(padding, Mapping) else False,
        "pad_count_max": padding.get("pad_count_max", "") if isinstance(padding, Mapping) else "",
        "pad_count_mean": padding.get("pad_count_mean", "") if isinstance(padding, Mapping) else "",
        "padded_sample_ratio": padding.get("padded_sample_ratio", "") if isinstance(padding, Mapping) else "",
    }


def write_markdown_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    selected_df: pd.DataFrame,
    result_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    value_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写出中文 smoke 摘要，显式回答 Round2b 验收问题。"""
    def markdown_table(frame: pd.DataFrame) -> str:
        """函数功能：避免依赖 tabulate，渲染小型 Markdown 表格。"""
        if frame.empty:
            return ""
        headers = [str(col) for col in frame.columns]
        rows = []
        for row in frame.itertuples(index=False, name=None):
            rows.append([str(value) for value in row])
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            table_lines.append("| " + " | ".join(row) + " |")
        return "\n".join(table_lines)

    lines: List[str] = []
    lines.append("# Visual Router V2 Round2b layout imageization smoke")
    lines.append("")
    lines.append(f"生成时间：{now_cst()}")
    lines.append("")
    lines.append("## 输入与边界")
    lines.append("")
    lines.append(f"- sample manifest：`{args.sample_manifest}`")
    lines.append(f"- layout candidates：`{args.layout_candidates}`")
    lines.append(f"- 输出目录：`{output_dir}`")
    lines.append(f"- 设备：`{metadata['device']}`")
    lines.append(f"- image size：`{args.image_size}`")
    lines.append(f"- smoke sample count：`{len(selected_df)}`")
    lines.append("- 本步未读取 future y、专家 prediction、oracle label 或 116M prediction manifest。")
    lines.append("- 本步未训练 router，未运行 ViT，未保存大规模 pseudo image tensor。")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    passed_layouts = result_df[result_df["status"] == "passed"]["layout_name"].tolist()
    failed_layouts = result_df[result_df["status"] != "passed"]["layout_name"].tolist()
    lines.append(f"- 已通过 layout：{', '.join(passed_layouts) if passed_layouts else '无'}。")
    lines.append(f"- 未通过 layout：{', '.join(failed_layouts) if failed_layouts else '无'}。")
    range_ok = bool(value_df["in_range_0_1"].all()) if not value_df.empty else False
    finite_ok = bool((value_df["finite_ratio"] == 1.0).all()) if not value_df.empty else False
    shape_ok = bool(result_df["shape_ok"].all()) if not result_df.empty else False
    lines.append(f"- shape 检查：{'通过' if shape_ok else '未通过'}；finite 检查：{'通过' if finite_ok else '未通过'}；[0,1] range 检查：{'通过' if range_ok else '未通过'}。")
    lines.append("- 主 imageization 入口为 torch tensor path；debug PNG 才将少量生成后 tensor detach 到 CPU。")
    lines.append("- `top3fold_period_layout` 当前是 channel-packed top1/top2/top3 fold，通过 registry adapter 复用 `imageize_top3fold`，不再暴露旧 `variant_b_top3fold` 参数语义。")
    lines.append("")
    lines.append("## Layout Latency")
    lines.append("")
    if not latency_df.empty:
        latency_view = latency_df[["layout_name", "batch_count", "sample_count", "total_time_ms", "samples_per_sec", "cpu_fallback"]].copy()
        lines.append(markdown_table(latency_view))
    lines.append("")
    lines.append("## Protocol 3.3 覆盖")
    lines.append("")
    lines.append("- 插值 / resize：metadata 记录每个 layout 使用 `linear_for_1d_profile; bilinear_for_2d_panel_or_fold`，`antialias=false`，并记录 `L=96 -> H/W=224` 映射。")
    lines.append("- value / difference bands：`line_difference_band` 使用 `first_diff=x[t]-x[t-1]`，首元素补 0；channel2 使用 `abs(first_diff)` 的窗口内 min-max band。")
    lines.append("- padding mask：period fold/top3fold layout 记录 padding mask 可用性、padding 是否输入 ViT、pad_count 统计；本步不把 mask 输入 ViT。")
    lines.append("- absolute FFT energy：`fft_absolute_energy` 使用 `abs(rfft(centered_x)[1:])**2` 和 `log1p(abs_energy)`；模型输入 profile 做窗口内归一化，absolute/log energy 原始统计写入 metadata。")
    lines.append("")
    lines.append("## Protocol 3.4 后续准备")
    lines.append("")
    lines.append("- top3fold metadata 已记录 top1/top2/top3 selected periods、period score summary 和 hard period bucket counts。")
    lines.append("- 下一步 continuity diagnostic：对输入 x 加轻微扰动，比较 pseudo image cosine/L2 distance、ViT embedding cosine distance、router weights JS divergence、selected model flip rate，并对比 hard_top1_fold、top3fold、period_soft_mixture。")
    lines.append("")
    lines.append("## 进入下一步条件")
    lines.append("")
    ready = shape_ok and range_ok and finite_ok and not failed_layouts
    lines.append(f"- 是否具备进入 35k small feature cache screening 的条件：{'是' if ready else '否'}。")
    lines.append("- 下一步仍需在 feature cache builder 中接入本 registry，并固定后端为 Round1 `film_mean_patch_aux` 风格：mean_patch_embedding + revin_aux FiLM。")
    (output_dir / "round2_layout_imageization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """函数功能：执行 Round2b layout imageization smoke 并写出所有要求产物。"""
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在，需传入 --overwrite：{output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layouts = parse_layouts(args.layouts)
    period_candidates = parse_period_candidates(args.period_candidates)
    device = resolve_device(args.device)
    data_config = load_data_config(args.config_path)
    manifest = pd.read_csv(args.sample_manifest)
    selected_df = select_smoke_samples(manifest, args.max_samples_per_set)
    selected_df.to_csv(output_dir / "round2_layout_imageization_smoke_samples.csv", index=False)

    loader = SmokeHistoryWindowLoader(data_config)
    x_np = loader.load_x_in_order(selected_df)
    x_cpu = torch.from_numpy(x_np).to(dtype=torch.float32)

    result_rows: List[Dict[str, object]] = []
    latency_rows: List[Dict[str, object]] = []
    value_rows: List[Dict[str, object]] = []
    shape_rows: List[Dict[str, object]] = []
    period_rows: List[Dict[str, object]] = []
    layout_metadata: Dict[str, object] = {}
    thumbnail_saved: Dict[str, int] = {}

    for layout_name in layouts:
        total_ms = 0.0
        sample_count = 0
        batch_count = 0
        first_metadata: Optional[Mapping[str, object]] = None
        first_stats: Optional[Dict[str, object]] = None
        status = "passed"
        error_message = ""
        try:
            for batch_start in range(0, int(x_cpu.shape[0]), int(args.batch_size)):
                batch_end = min(batch_start + int(args.batch_size), int(x_cpu.shape[0]))
                x_batch = x_cpu[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=False)
                with torch.inference_mode():
                    start = timer_start(device)
                    result = imageize_round2_layout(
                        x_batch,
                        layout_name=layout_name,
                        image_size=int(args.image_size),
                        norm_mode=str(args.norm_mode),
                        clip=float(args.clip),
                        period_candidates=period_candidates,
                        period_selection=str(args.period_selection),
                    )
                    elapsed_ms = timer_stop(start, device)
                images = result.images
                stats = tensor_value_stats(images)
                shape_ok = images.ndim == 4 and images.shape[1] == 3 and images.shape[2] == int(args.image_size) and images.shape[3] == int(args.image_size)
                if not shape_ok or not stats["in_range_0_1"] or stats["finite_ratio"] != 1.0:
                    status = "failed"
                    error_message = f"shape_ok={shape_ok} in_range={stats['in_range_0_1']} finite_ratio={stats['finite_ratio']}"
                if first_metadata is None:
                    first_metadata = result.metadata
                    first_stats = stats
                value_rows.append({"layout_name": layout_name, "batch_start": batch_start, **stats})
                shape_rows.append(
                    {
                        "layout_name": layout_name,
                        "batch_start": batch_start,
                        "batch_size": int(images.shape[0]),
                        "channels": int(images.shape[1]),
                        "height": int(images.shape[2]),
                        "width": int(images.shape[3]),
                        "shape_ok": bool(shape_ok),
                        "dtype": str(images.dtype),
                        "device": str(images.device),
                    }
                )
                period_rows.append(summarize_period_metadata(layout_name, result.metadata, int(images.shape[0])))
                if args.save_debug_thumbnails:
                    saved = thumbnail_saved.get(layout_name, 0)
                    remaining = max(0, int(args.debug_thumbnail_count) - saved)
                    for row_idx in range(min(remaining, int(images.shape[0]))):
                        global_idx = batch_start + row_idx
                        save_debug_thumbnail(images[row_idx], output_dir / "debug_thumbnails" / layout_name / f"{global_idx:04d}.png")
                        saved += 1
                    thumbnail_saved[layout_name] = saved
                total_ms += float(elapsed_ms)
                sample_count += int(images.shape[0])
                batch_count += 1
                del images, result
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = repr(exc)
            first_metadata = first_metadata or {}
        samples_per_sec = float(sample_count) / (total_ms / 1000.0) if total_ms > 0 else 0.0
        metadata_obj = dict(first_metadata or {})
        layout_metadata[layout_name] = metadata_obj
        latency_rows.append(
            {
                "layout_name": layout_name,
                "device": str(device),
                "batch_size": int(args.batch_size),
                "batch_count": int(batch_count),
                "sample_count": int(sample_count),
                "total_time_ms": total_ms,
                "samples_per_sec": samples_per_sec,
                "optional_vit_preprocessing_time_ms": "",
                "cpu_fallback": bool(metadata_obj.get("cpu_fallback", False)),
                "explicit_cpu_gpu_transfer_note": str(metadata_obj.get("explicit_cpu_gpu_transfer_in_main_path", "")),
            }
        )
        expected_shape = [sample_count if sample_count else None, 3, int(args.image_size), int(args.image_size)]
        actual_shape = first_stats.get("shape", []) if first_stats else []
        result_rows.append(
            {
                "layout_name": layout_name,
                "status": status,
                "sample_count": int(sample_count),
                "expected_shape": json.dumps(expected_shape),
                "first_batch_shape": json.dumps(actual_shape),
                "shape_ok": bool(status == "passed" and actual_shape[1:] == [3, int(args.image_size), int(args.image_size)]),
                "dtype": first_stats.get("dtype", "") if first_stats else "",
                "device": first_stats.get("device", str(device)) if first_stats else str(device),
                "value_min": first_stats.get("min", "") if first_stats else "",
                "value_max": first_stats.get("max", "") if first_stats else "",
                "finite_ratio": first_stats.get("finite_ratio", "") if first_stats else "",
                "error_message": error_message,
            }
        )

    result_df = pd.DataFrame(result_rows)
    latency_df = pd.DataFrame(latency_rows)
    value_df = pd.DataFrame(value_rows)
    shape_df = pd.DataFrame(shape_rows)
    period_df = pd.DataFrame(period_rows)
    result_df.to_csv(output_dir / "round2_layout_imageization_smoke_results.csv", index=False)
    latency_df.to_csv(output_dir / "round2_layout_imageization_latency.csv", index=False)
    value_df.to_csv(output_dir / "round2_layout_imageization_value_stats.csv", index=False)
    shape_df.to_csv(output_dir / "round2_layout_imageization_shape_check.csv", index=False)
    period_df.to_csv(output_dir / "round2_layout_imageization_period_stats.csv", index=False)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "created_at": now_cst(),
        "status": "completed" if (result_df["status"] == "passed").all() else "failed",
        "sample_manifest": str(args.sample_manifest),
        "layout_candidates": str(args.layout_candidates),
        "config_path": str(args.config_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "layouts_requested": layouts,
        "default_layouts": list(DEFAULT_ROUND2_LAYOUTS),
        "deferred_layouts": list(DEFERRED_ROUND2_LAYOUTS),
        "registry_specs": list_layout_specs(),
        "sample_counts": {key: int(value) for key, value in selected_df.groupby("sample_set").size().to_dict().items()},
        "constraints": {
            "read_history_x": True,
            "read_future_y": False,
            "read_expert_prediction": False,
            "read_oracle_label_as_feature": False,
            "read_116m_prediction_manifest": False,
            "trained_router": False,
            "ran_vit": False,
            "saved_large_pseudo_image_tensor": False,
            "debug_thumbnails_detach_to_cpu_only": bool(args.save_debug_thumbnails),
        },
        "layout_metadata": layout_metadata,
    }
    (output_dir / "round2_layout_imageization_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(
        output_dir=output_dir,
        args=args,
        selected_df=selected_df,
        result_df=result_df,
        latency_df=latency_df,
        value_df=value_df,
        metadata=metadata,
    )

    summary_copy_dir = args.summary_copy_dir
    summary_copy_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "round2_layout_imageization_smoke_results.csv",
        "round2_layout_imageization_latency.csv",
        "round2_layout_imageization_value_stats.csv",
        "round2_layout_imageization_shape_check.csv",
        "round2_layout_imageization_period_stats.csv",
        "round2_layout_imageization_metadata.json",
        "round2_layout_imageization_summary.md",
    ]:
        shutil.copy2(output_dir / name, summary_copy_dir / name)
    if args.save_debug_thumbnails and (output_dir / "debug_thumbnails").exists():
        target = summary_copy_dir / "debug_thumbnails"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(output_dir / "debug_thumbnails", target)


if __name__ == "__main__":
    main()
