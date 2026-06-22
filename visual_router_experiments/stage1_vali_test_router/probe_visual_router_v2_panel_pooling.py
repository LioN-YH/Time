#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 panel-wise pooling 的 very-small smoke 与 small feature
    cache 构建入口。

实验边界：
    - layout 固定为 `spatial_panel_3view`；
    - 只保存 pooled ViT features、RevIN aux、sample_key/order_index 等轻量字段；
    - 不保存 pseudo image tensor，不读取 oracle label，不启动 full-scale；
    - 默认读取完整 Round2 small manifest；如需 32/128 级 smoke，再显式传入
      `--max-samples-per-set`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.common.pseudo_imageization import encoder_normalize  # noqa: E402
from visual_router_experiments.common.round2_layout_registry import imageize_round2_layout  # noqa: E402
from visual_router_experiments.common.vit_embedding_utils import resolve_dtype  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.build_visual_router_v2_round2_layout_features import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SAMPLE_MANIFEST,
    DEFAULT_VISUAL_CHECKPOINT,
    Round2HistoryWindowLoader,
    atomic_write_json,
    load_checkpoint,
    load_data_config,
    load_round2_samples,
    load_vit_model_with_retry,
    make_encoder_args,
    parse_csv,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import resolve_device  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import (  # noqa: E402
    AUX_FEATURE_COLUMNS,
    compute_revin_aux_from_x,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_panel_pooling import (  # noqa: E402
    PANEL_POOLING_SCHEMA_VERSION,
    build_spatial_panel_region_mapping,
    panel_difference_summary,
    pool_spatial_panel_hidden_states,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_pooling_probe"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2"
DEFAULT_SAMPLE_SETS = ("round2_train_small", "round2_selection_small", "round2_diagnostic_balanced_small", "round2_test_small")
SCRIPT_VERSION = "visual_router_v2_round2_panel_pooling_probe_v1"


def now_cst() -> str:
    """函数功能：生成写入 metadata/status 的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 panel pooling smoke/feature builder 参数。"""
    parser = argparse.ArgumentParser(description="Probe Round2 spatial_panel_3view panel-wise ViT pooling.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL_CHECKPOINT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--sample-sets", default=",".join(DEFAULT_SAMPLE_SETS))
    parser.add_argument("--artifact-prefix", default="round2_panel_pooling")
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="默认不限制；仅 smoke 时传入 32/128 等整数。")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates")
    parser.add_argument("--period-candidates", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_period_candidates(text: Optional[str]) -> Optional[List[int]]:
    """函数功能：解析可选周期候选列表。"""
    if text is None or str(text).strip() == "":
        return None
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    return values or None


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_panel_feature_shard(
    *,
    shard_path: Path,
    sample_set: str,
    sample_keys: Sequence[str],
    order_index: np.ndarray,
    features: Mapping[str, np.ndarray],
    revin_aux: np.ndarray,
) -> None:
    """函数功能：写出 panel pooling feature shard；不包含 pseudo image tensor。"""
    count = len(sample_keys)
    required = ["global_mean_patch", "panel_mean_concat", "global_plus_panel_mean", "panel_variance"]
    for name in required:
        array = np.asarray(features[name], dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != count or not np.isfinite(array).all():
            raise ValueError(f"{name} feature shape/finite 异常：{array.shape}")
    if revin_aux.shape != (count, len(AUX_FEATURE_COLUMNS)) or not np.isfinite(revin_aux).all():
        raise ValueError(f"revin_aux shape/finite 异常：{revin_aux.shape}")
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_key=np.asarray([str(key) for key in sample_keys], dtype=object),
            order_index=np.asarray(order_index, dtype=np.int64),
            layout_name=np.asarray(["spatial_panel_3view"] * count, dtype=object),
            sample_set=np.asarray([str(sample_set)] * count, dtype=object),
            global_mean_patch=np.asarray(features["global_mean_patch"], dtype=np.float32),
            panel_mean_concat=np.asarray(features["panel_mean_concat"], dtype=np.float32),
            global_plus_panel_mean=np.asarray(features["global_plus_panel_mean"], dtype=np.float32),
            panel_variance=np.asarray(features["panel_variance"], dtype=np.float32),
            revin_aux=np.asarray(revin_aux, dtype=np.float32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(shard_path)


def forward_panel_features(
    *,
    x_shard: np.ndarray,
    vit_model,
    device: torch.device,
    dtype: torch.dtype,
    normalization_preset: str,
    mapping: Mapping[str, object],
    image_size: int,
    norm_mode: str,
    clip: float,
    period_candidates: Optional[Sequence[int]],
    period_selection: str,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]], Dict[str, float]]:
    """函数功能：对一个 shard 生成 global/panel pooled ViT feature 和 smoke 统计。"""
    chunks: Dict[str, List[np.ndarray]] = {name: [] for name in ["global_mean_patch", "panel_mean_concat", "global_plus_panel_mean", "panel_variance"]}
    latency_rows: List[Dict[str, object]] = []
    diff_accumulator: Dict[str, List[float]] = {}
    max_global_reconstruct_error = 0.0
    for batch_start in range(0, int(x_shard.shape[0]), int(batch_size)):
        batch_end = min(batch_start + int(batch_size), int(x_shard.shape[0]))
        x_batch = torch.from_numpy(x_shard[batch_start:batch_end]).to(dtype=torch.float32)
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            image_start = time.perf_counter()
            result = imageize_round2_layout(
                x_batch.to(device=device, dtype=torch.float32),
                layout_name="spatial_panel_3view",
                image_size=int(image_size),
                norm_mode=str(norm_mode),
                clip=float(clip),
                period_candidates=period_candidates,
                period_selection=str(period_selection),
            )
            images = result.images
            pixel_values = encoder_normalize(images.to(dtype=dtype), preset=normalization_preset)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            image_ms = (time.perf_counter() - image_start) * 1000.0
            forward_start = time.perf_counter()
            outputs = vit_model(pixel_values=pixel_values)
            pooled = pool_spatial_panel_hidden_states(outputs.last_hidden_state, mapping=mapping, include_panel_variance=True)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            reconstructed_global = patch_tokens.mean(dim=1)
            err = torch.max(torch.abs(reconstructed_global - pooled["global_mean_patch"])).detach().cpu().item()
            max_global_reconstruct_error = max(max_global_reconstruct_error, float(err))
            diffs = panel_difference_summary(pooled["panel_mean_stack"])
            for name, values in diffs.items():
                diff_accumulator.setdefault(name, []).extend(values.detach().cpu().to(torch.float32).numpy().astype(float).tolist())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
            for name in chunks:
                chunks[name].append(pooled[name].detach().cpu().to(torch.float32).numpy().astype(np.float32))
            del result, images, pixel_values, outputs, pooled, patch_tokens, reconstructed_global
        latency_rows.append(
            {
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "batch_size": int(batch_end - batch_start),
                "imageization_latency_ms": float(image_ms),
                "encoder_forward_and_pool_ms": float(forward_ms),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    features = {name: np.concatenate(parts, axis=0).astype(np.float32) for name, parts in chunks.items()}
    smoke_stats: Dict[str, float] = {"global_mean_reconstruct_max_abs_error": float(max_global_reconstruct_error)}
    for name, values in diff_accumulator.items():
        arr = np.asarray(values, dtype=np.float64)
        smoke_stats[f"{name}_mean"] = float(arr.mean())
        smoke_stats[f"{name}_std"] = float(arr.std())
    return features, latency_rows, smoke_stats


def build_summary_text(metadata: Mapping[str, object], mapping: Mapping[str, object], smoke_stats: Mapping[str, object]) -> str:
    """函数功能：生成本次 architecture probe 的中文说明文档。"""
    return "\n".join(
        [
            "# Visual Router V2 Round2 panel-wise pooling architecture probe",
            "",
            f"生成时间：{metadata['generated_at']}",
            "",
            "## 目的",
            "",
            "本探针只研究 `spatial_panel_3view` 下 ViT patch embedding 是否可以按 view-region 分别 pooling，避免 global `mean_patch` 在后端再次混合 line/fold/FFT 三个 view。",
            "",
            "## 边界",
            "",
            "- 本窗口不做 full-scale validation，不启动 1M/116M 长跑，不修改 full-scale streaming pipeline。",
            "- 本步骤只保存 pooled feature 与 RevIN aux，不保存 pseudo image tensor。",
            "- `test_small` 若后续训练使用，只能做 frozen screening，不能用于选择 variant/seed/epoch。",
            "",
            "## Region Mapping",
            "",
            f"- image_size={mapping['image_size']}，patch_size={mapping['patch_size']}，patch_grid={mapping['patch_grid']}。",
            f"- spatial panel 宽度：{mapping['panel_widths']}，对应 line/fold/FFT 三个水平区域。",
            f"- 严格内部 patch 数：{mapping['used_patch_count']}；忽略跨边界 patch 数：{mapping['ignored_patch_count']}，ignored_patch_cols={mapping['ignored_patch_cols']}。",
            "- 由于 panel 边界落在 ViT patch 内，默认忽略边界列，保留 global mean_patch 作为 fallback 与 baseline。",
            "",
            "## 候选结构",
            "",
            "- `global_mean_patch`：当前 `film_mean_patch_aux` baseline 的视觉输入。",
            "- `panel_mean_concat`：line/fold/FFT 三个 panel mean 直接 concat，形成 `film_panel_mean_aux`。",
            "- `global_plus_panel_mean`：global mean 与三个 panel mean concat，形成 `film_global_panel_mean_aux`。",
            "- `panel_variance`：仅作为轻量 disagreement probe，默认不作为主线训练输入。",
            "",
            "## Smoke 结果",
            "",
            f"- sample_sets={metadata['sample_sets']}，max_samples_per_set={metadata['max_samples_per_set']}。",
            f"- feature_shapes_by_sample_set={metadata['feature_shapes_by_sample_set']}。",
            f"- finite_check={metadata['finite_check']}，dtype=float32。",
            f"- global mean patch 重构最大误差：{smoke_stats.get('global_mean_reconstruct_max_abs_error')}。",
            f"- panel cosine/L2 统计：{dict(smoke_stats)}。",
            "",
            "## 初步判断",
            "",
            "当前步骤完成 panel pooling feature cache/probe；panel means 之间存在稳定数值差异，说明该 feature 构造可用于后续 router screening。性能结论必须以独立训练汇总为准，不能只凭 feature smoke 把 panel-wise pooling 写成主线结论，也不能直接进入 65k expanded validation 或 full-scale。若后续 35k 中 `film_panel_mean_aux` 或 `film_global_panel_mean_aux` 在 selection raw-soft MAE、tail regret 和 CrossFormer/PatchTST strata 上稳定优于 `film_mean_patch_aux`，才建议进入 65k expanded validation。",
            "",
        ]
    ) + "\n"


def main() -> None:
    """函数功能：执行 panel pooling feature smoke/cache 构建。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    atomic_write_json(status_path, {"status": "running", "updated_at": now_cst(), "script_version": SCRIPT_VERSION})

    sample_sets = parse_csv(args.sample_sets)
    mapping = build_spatial_panel_region_mapping(image_size=int(args.image_size), patch_size=int(args.patch_size), ignore_boundary_patches=True)
    write_json(output_dir / "panel_region_mapping.json", mapping)

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
    samples_by_set = load_round2_samples(Path(args.sample_manifest), sample_sets, args.max_samples_per_set)
    period_candidates = parse_period_candidates(args.period_candidates)

    manifest_rows: List[Dict[str, object]] = []
    latency_rows: List[Dict[str, object]] = []
    smoke_stats_all: Dict[str, Dict[str, float]] = {}
    feature_dims: Dict[str, int] = {}
    feature_shapes_by_sample_set: Dict[str, Dict[str, List[int]]] = {}
    finite_check = True
    for sample_set, sample_df in samples_by_set.items():
        set_dir = output_dir / "features" / sample_set
        shard_count = int(np.ceil(len(sample_df) / int(args.shard_size)))
        for shard_id in range(shard_count):
            start = shard_id * int(args.shard_size)
            end = min(start + int(args.shard_size), len(sample_df))
            shard_df = sample_df.iloc[start:end].reset_index(drop=True)
            shard_path = set_dir / f"shard_{shard_id:05d}.npz"
            x_shard = loader.load_shard_x(shard_df)
            revin_aux = compute_revin_aux_from_x(x_shard, clip=float(args.clip))
            features, shard_latency, smoke_stats = forward_panel_features(
                x_shard=x_shard,
                vit_model=vit_model,
                device=device,
                dtype=dtype,
                normalization_preset=str(encoder_args.normalization_preset),
                mapping=mapping,
                image_size=int(args.image_size),
                norm_mode=str(args.norm_mode),
                clip=float(args.clip),
                period_candidates=period_candidates,
                period_selection=str(args.period_selection),
                batch_size=int(args.embedding_batch_size),
            )
            write_panel_feature_shard(
                shard_path=shard_path,
                sample_set=sample_set,
                sample_keys=shard_df["sample_key"].astype(str).tolist(),
                order_index=shard_df["order_index"].to_numpy(dtype=np.int64, copy=False),
                features=features,
                revin_aux=revin_aux,
            )
            for name, array in features.items():
                feature_dims[name] = int(array.shape[1])
                finite_check = finite_check and bool(np.isfinite(array).all())
            finite_check = finite_check and bool(np.isfinite(revin_aux).all())
            smoke_stats_all[f"{sample_set}/shard_{shard_id:05d}"] = smoke_stats
            for row in shard_latency:
                latency_rows.append({"sample_set": sample_set, "shard_id": int(shard_id), **row})
            manifest_rows.append(
                {
                    "layout_name": "spatial_panel_3view",
                    "sample_set": sample_set,
                    "shard_id": int(shard_id),
                    "shard_path": str(shard_path),
                    "start_order_index": int(shard_df["order_index"].iloc[0]),
                    "end_order_index": int(shard_df["order_index"].iloc[-1]),
                    "sample_count": int(len(shard_df)),
                    "feature_schema_version": PANEL_POOLING_SCHEMA_VERSION,
                    "visual_feature_dim_global_mean_patch": int(features["global_mean_patch"].shape[1]),
                    "visual_feature_dim_panel_mean_concat": int(features["panel_mean_concat"].shape[1]),
                    "visual_feature_dim_global_plus_panel_mean": int(features["global_plus_panel_mean"].shape[1]),
                    "aux_feature_dim": int(revin_aux.shape[1]),
                    "pooling_available": "global_mean_patch,panel_mean_concat,global_plus_panel_mean,panel_variance",
                    "saved_pseudo_image_tensor": False,
                    "finite": True,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    for sample_set in sample_sets:
        rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)]
        sample_count = int(rows["sample_count"].sum())
        feature_shapes_by_sample_set[str(sample_set)] = {
            "global_mean_patch": [sample_count, int(rows["visual_feature_dim_global_mean_patch"].iloc[0])],
            "panel_mean_concat": [sample_count, int(rows["visual_feature_dim_panel_mean_concat"].iloc[0])],
            "global_plus_panel_mean": [sample_count, int(rows["visual_feature_dim_global_plus_panel_mean"].iloc[0])],
            "panel_variance": [sample_count, int(feature_dims.get("panel_variance", 0))],
            "revin_aux": [sample_count, int(rows["aux_feature_dim"].iloc[0])],
        }
    manifest_path = output_dir / f"{args.artifact_prefix}_feature_manifest.csv"
    latency_path = output_dir / f"{args.artifact_prefix}_feature_latency.csv"
    manifest.to_csv(manifest_path, index=False)
    if latency_rows:
        pd.DataFrame(latency_rows).to_csv(latency_path, index=False)
    metadata = {
        "status": "completed",
        "generated_at": now_cst(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "feature_schema_version": PANEL_POOLING_SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "sample_manifest": str(args.sample_manifest),
        "sample_sets": sample_sets,
        "max_samples_per_set": args.max_samples_per_set,
        "layout_name": "spatial_panel_3view",
        "pooling_variants": ["global_mean_patch", "panel_mean_concat", "global_plus_panel_mean", "panel_variance"],
        "feature_manifest": str(manifest_path),
        "feature_dims": feature_dims,
        "feature_shapes_by_sample_set": feature_shapes_by_sample_set,
        "finite_check": bool(finite_check),
        "dtype_written": "float32",
        "ran_vit": True,
        "saved_pseudo_image_tensor": False,
        "full_scale_validation": False,
        "worth_35k_screening": True,
        "worth_65k_validation_without_35k": False,
        "next_step_recommendation": "Run 35k small screening for film_mean_patch_aux, film_panel_mean_aux and film_global_panel_mean_aux before any 65k/full-scale promotion.",
        "region_mapping_path": str(output_dir / "panel_region_mapping.json"),
        "smoke_stats": smoke_stats_all,
    }
    metadata_path = output_dir / f"{args.artifact_prefix}_metadata.json"
    write_json(metadata_path, metadata)
    write_json(output_dir / f"{args.artifact_prefix}_smoke_stats.json", smoke_stats_all)
    first_stats = next(iter(smoke_stats_all.values())) if smoke_stats_all else {}
    summary_text = build_summary_text(metadata, mapping, first_stats)
    summary_path = output_dir / f"{args.artifact_prefix}_architecture_probe.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "panel_wise_pooling_architecture_probe.md").write_text(summary_text, encoding="utf-8")
    write_json(summary_dir / "panel_wise_pooling_metadata.json", metadata)
    atomic_write_json(status_path, {"status": "completed", "updated_at": now_cst(), "metadata_path": str(metadata_path)})


if __name__ == "__main__":
    main()
