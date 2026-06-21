#!/usr/bin/env python3
"""
文件功能：
    构建 Visual Router V2 Round2 view layout small screening 的固定小样本集，
    并记录第一轮 pseudo image layout candidates 与 top3fold 复用审计。

输入：
    - full-scale window oracle labels parquet；
    - full-scale sample TSF enrichment parquet；
    - 可选 P0 pilot sample CSV，用于记录 overlap ratio。

输出：
    - round2_train_small / selection / diagnostic / test 四个 sample_key CSV；
    - round2_small_sample_manifest.csv；
    - round2_layout_candidates.json；
    - round2_top3fold_reuse_audit.md；
    - round2_small_sample_metadata.json；
    - round2_small_screening_summary.md。

关键约束：
    本脚本只冻结 screening 协议和样本索引，不生成 full feature cache，不训练 router，
    不运行 ViT，不保存 pseudo image tensor，也不读取 116M prediction manifest。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set

import pandas as pd

from build_visual_router_v2_pilot_samples import (
    MODEL_ORDER,
    OUTPUT_COLS,
    TSF_COLS,
    attach_tsf,
    build_coverage_summary,
    collect_diagnostic_rows,
    collect_main_sets_and_gap_boundaries,
    gap_quantile_label,
    load_tsf_subset,
    rows_to_frame,
    validate_inputs,
)


DEFAULT_FULL_SCALE_ROOT = Path(
    "/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher"
)
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples")
DEFAULT_LIGHT_SUMMARY_DIR = Path("experiment_summaries/visual_router_v2_round2/small_samples")
DEFAULT_P0_SAMPLE_DIR = Path("/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples")
SCRIPT_VERSION = "visual_router_v2_round2_small_sample_builder_v1"


def now_cst() -> str:
    """函数功能：返回 metadata、summary 和日志使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2 small screening sample builder 参数。"""
    parser = argparse.ArgumentParser(description="Build Visual Router V2 Round2 small screening sample sets.")
    parser.add_argument(
        "--oracle-labels-path",
        type=Path,
        default=DEFAULT_FULL_SCALE_ROOT / "oracle_labels_full_scale_2026-06-16" / "window_oracle_labels.parquet",
        help="full-scale window_oracle_labels.parquet 路径。",
    )
    parser.add_argument(
        "--tsf-enrichment-path",
        type=Path,
        default=DEFAULT_FULL_SCALE_ROOT / "tsf_enrichment_full_scale_2026-06-16" / "sample_tsf_enrichment.parquet",
        help="full-scale sample_tsf_enrichment.parquet 路径。",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Round2 小样本输出目录。")
    parser.add_argument("--light-summary-dir", type=Path, default=DEFAULT_LIGHT_SUMMARY_DIR, help="轻量 summary 复制目录。")
    parser.add_argument("--p0-sample-dir", type=Path, default=DEFAULT_P0_SAMPLE_DIR, help="P0 pilot sample 目录，用于 overlap 统计。")
    parser.add_argument("--seed", type=int, default=20260621, help="固定抽样 seed。")
    parser.add_argument("--train-size", type=int, default=20_000, help="round2_train_small vali window 数。")
    parser.add_argument("--selection-size", type=int, default=5_000, help="round2_selection_small vali window 数。")
    parser.add_argument("--diagnostic-balanced-size", type=int, default=5_000, help="round2_diagnostic_balanced_small vali window 数。")
    parser.add_argument("--test-size", type=int, default=5_000, help="round2_test_small test window 数。")
    parser.add_argument("--gap-quantile-reservoir-size", type=int, default=500_000, help="估计 error_gap 分位边界的稳定哈希 reservoir 大小。")
    parser.add_argument("--batch-size", type=int, default=250_000, help="Parquet 扫描 batch 行数。")
    parser.add_argument("--no-copy-light-summary", action="store_true", help="只写外部输出目录，不复制轻量 summary 到仓库。")
    return parser.parse_args()


def build_layout_candidates() -> List[Dict[str, object]]:
    """
    函数功能：
        定义 Round2 第一轮 view layout screening 的候选 registry。

    说明：
        `implementation_status` 只描述本步可执行性边界；本脚本不会生成伪图像或 ViT
        feature cache，避免把 protocol freeze 与训练/特征构建混在同一步。
    """
    common_time = "1-1.5h per 35k samples on current P2a-like single-layout cache estimate"
    return [
        {
            "layout_name": "current_rgb_3view",
            "layout_family": "rgb_channel_mixed_multiview",
            "input_source": "RevIN-normalized history x; line raster, top1 period fold, FFT power",
            "pseudo_image_size": "224x224",
            "channel_design": "3 semantic views packed into ViT RGB channels",
            "panel_design": "no spatial panel; channel-packed baseline",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": False,
            "uses_frequency_information": True,
            "uses_period_folding": True,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium: channel mixing may encode dataset-specific view interactions",
            "expected_helped_strata": ["general baseline", "season_strength_cat", "PatchTST/CrossFormer strata"],
            "expected_failure_modes": ["RGB channel interference", "view semantics not aligned with natural-image pretraining"],
            "implementation_status": "existing_round1_baseline_via_variant_a_3view",
            "estimated_single_layout_feature_cache_time": common_time,
            "default_in_round2a": True,
            "notes": "Round1 当前对照口径；用于锚定所有 Round2 layout 改动。",
        },
        {
            "layout_name": "spatial_panel_3view",
            "layout_family": "spatial_panel_multiview",
            "input_source": "same three semantic views as current_rgb_3view",
            "pseudo_image_size": "224x224",
            "channel_design": "grayscale panels copied or lightly encoded into RGB-compatible tensor",
            "panel_design": "three views arranged as horizontal or vertical spatial panels",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": False,
            "uses_frequency_information": True,
            "uses_period_folding": True,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium: panel position and reduced effective resolution may leak layout bias",
            "expected_helped_strata": ["mean_patch stability", "PatchTST stratum", "CrossFormer stratum"],
            "expected_failure_modes": ["per-view resolution loss", "ViT positional bias to panel location"],
            "implementation_status": "new_layout_candidate_to_implement_before_feature_cache",
            "estimated_single_layout_feature_cache_time": common_time,
            "default_in_round2a": True,
            "notes": "核心问题是 view separation 是否优于 RGB channel mixing。",
        },
        {
            "layout_name": "line_only",
            "layout_family": "minimal_shape",
            "input_source": "RevIN-normalized history x",
            "pseudo_image_size": "224x224",
            "channel_design": "single line raster replicated or encoded into ViT-compatible 3 channels",
            "panel_design": "single full-canvas line plot",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": False,
            "uses_frequency_information": False,
            "uses_period_folding": False,
            "uses_scale_statistics": False,
            "shortcut_risk": "low: minimal shape-only signal",
            "expected_helped_strata": ["forecastability_cat", "shape-dominant datasets"],
            "expected_failure_modes": ["missing local volatility", "missing frequency/seasonality information"],
            "implementation_status": "new_layout_candidate_to_implement_before_feature_cache",
            "estimated_single_layout_feature_cache_time": common_time,
            "default_in_round2a": True,
            "notes": "低 shortcut 风险、最可解释的 shape baseline。",
        },
        {
            "layout_name": "line_difference_band",
            "layout_family": "shape_plus_local_change",
            "input_source": "RevIN-normalized history x and first-difference / rolling local-change signal",
            "pseudo_image_size": "224x224",
            "channel_design": "line view plus difference/volatility band in RGB-compatible channels or panels",
            "panel_design": "line with auxiliary local-change band, exact packing to freeze in feature-cache step",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": True,
            "uses_frequency_information": False,
            "uses_period_folding": False,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium-low: local change can amplify noise but is not a direct dataset id",
            "expected_helped_strata": ["PatchTST stratum", "CrossFormer stratum", "high error_gap_quantile"],
            "expected_failure_modes": ["noise amplification", "difference band dominates main shape"],
            "implementation_status": "new_layout_candidate_to_implement_before_feature_cache",
            "estimated_single_layout_feature_cache_time": common_time,
            "default_in_round2a": True,
            "notes": "用于测试局部变化、突变和 patch-level pattern 是否改善 routing。",
        },
        {
            "layout_name": "fft_absolute_energy",
            "layout_family": "frequency_energy",
            "input_source": "FFT absolute or log energy computed from RevIN-normalized history x",
            "pseudo_image_size": "224x224",
            "channel_design": "frequency energy profile rendered as ViT-compatible image",
            "panel_design": "frequency profile or heatmap; does not preserve phase",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": False,
            "uses_frequency_information": True,
            "uses_period_folding": False,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium-high: frequency distribution may encode dataset shortcut",
            "expected_helped_strata": ["season_strength_cat", "periodic datasets", "PatchTST stratum"],
            "expected_failure_modes": ["loss of time-local information", "dataset frequency shortcut"],
            "implementation_status": "partially_existing_fft_power_view_can_be_reused_or_refactored",
            "estimated_single_layout_feature_cache_time": common_time,
            "default_in_round2a": True,
            "notes": "区别于 top3fold：这是频域统计图，不显式保留周期内局部形状。",
        },
        {
            "layout_name": "top3fold_period_layout",
            "layout_family": "period_aware_spatial_layout",
            "input_source": "RevIN-normalized history x with FFT top-3 period candidates",
            "pseudo_image_size": "224x224",
            "channel_design": "top1/top2/top3 period-folded views in ViT-compatible tensor",
            "panel_design": "period-aware folded spatial layout; current implementation uses channels",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": False,
            "uses_frequency_information": True,
            "uses_period_folding": True,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium-high: unstable period estimates or top-k period can become dataset shortcut",
            "expected_helped_strata": ["season_strength_cat", "periodic datasets", "PatchTST stratum", "forecastability_cat"],
            "expected_failure_modes": ["period estimate instability", "hard top-k period shortcut", "weak non-periodic series"],
            "implementation_status": "existing_imageize_top3fold_available_needs_round2_registry_adapter",
            "estimated_single_layout_feature_cache_time": common_time,
            "default_in_round2a": True,
            "notes": "保留周期内局部形状，是 FFT energy 的互补 period-aware spatial layout。",
        },
        {
            "layout_name": "period_soft_mixture",
            "layout_family": "period_aware_soft_mixture",
            "input_source": "multiple period candidates and their relative strengths",
            "pseudo_image_size": "224x224",
            "channel_design": "soft mixture of folded period views, exact weighting deferred",
            "panel_design": "deferred; may mix or panelize multiple period candidates",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": False,
            "uses_frequency_information": True,
            "uses_period_folding": True,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium: soft weights reduce hard shortcut but still expose period spectrum",
            "expected_helped_strata": ["multi-period series", "unstable seasonality", "forecastability_cat"],
            "expected_failure_modes": ["mixture blur", "extra hyperparameter coupling", "hard to compare with first-round layouts"],
            "implementation_status": "deferred_design_only",
            "estimated_single_layout_feature_cache_time": "deferred; expected near top3fold unless multiple ViT passes are introduced",
            "default_in_round2a": False,
            "notes": "本步只记录，不纳入第一轮默认 feature cache screening。",
        },
        {
            "layout_name": "independent_view_encoder",
            "layout_family": "architecture_level_multiview_encoder",
            "input_source": "multiple pseudo image views, each independently encoded by frozen ViT",
            "pseudo_image_size": "224x224 per view",
            "channel_design": "each view encoded separately, embeddings aggregated after ViT",
            "panel_design": "no shared image panel; view separation happens at encoder level",
            "uses_revin_normalized_shape": True,
            "uses_difference_or_volatility": True,
            "uses_frequency_information": True,
            "uses_period_folding": True,
            "uses_scale_statistics": False,
            "shortcut_risk": "medium: cleaner view separation but changes architecture and compute budget",
            "expected_helped_strata": ["view interference cases", "mean_patch stability", "seasonal and local-change strata"],
            "expected_failure_modes": ["2-3x or view-count-times feature cache cost", "architecture confounded with layout screening"],
            "implementation_status": "deferred_architecture_candidate_only",
            "estimated_single_layout_feature_cache_time": "not recommended first round; roughly 2-3x if each view passes ViT separately",
            "default_in_round2a": False,
            "notes": "这是架构级候选，不应与第一轮 layout screening 混跑。",
        },
    ]


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出 UTF-8 JSON，保留中文字段可读性。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_round2_sample_sets(frames: Mapping[str, pd.DataFrame]) -> Dict[str, object]:
    """函数功能：校验 Round2 四个样本集的唯一性、split 边界和互斥关系。"""
    per_set_counts = {name: int(len(frame)) for name, frame in frames.items()}
    per_set_duplicate_counts = {name: int(frame["sample_key"].duplicated().sum()) for name, frame in frames.items()}
    split_values = {name: sorted(frame["split"].unique().tolist()) for name, frame in frames.items()}
    all_keys: List[str] = []
    for frame in frames.values():
        all_keys.extend(frame["sample_key"].astype(str).tolist())
    cross_set_duplicate_count = len(all_keys) - len(set(all_keys))
    train_keys = set(frames["round2_train_small"]["sample_key"])
    selection_keys = set(frames["round2_selection_small"]["sample_key"])
    validation = {
        "status": "passed",
        "generated_at": now_cst(),
        "per_set_counts": per_set_counts,
        "per_set_duplicate_counts": per_set_duplicate_counts,
        "cross_set_duplicate_count": int(cross_set_duplicate_count),
        "split_values": split_values,
        "round2_train_selection_intersection_count": int(len(train_keys.intersection(selection_keys))),
        "all_outputs_have_order_index": all(
            bool((frame["order_index"].to_numpy() == range(len(frame))).all()) for frame in frames.values()
        ),
    }
    checks = [
        all(count == 0 for count in per_set_duplicate_counts.values()),
        cross_set_duplicate_count == 0,
        split_values["round2_train_small"] == ["vali"],
        split_values["round2_selection_small"] == ["vali"],
        split_values["round2_diagnostic_balanced_small"] == ["vali"],
        split_values["round2_test_small"] == ["test"],
        validation["round2_train_selection_intersection_count"] == 0,
        bool(validation["all_outputs_have_order_index"]),
    ]
    if not all(checks):
        validation["status"] = "failed"
    return validation


def load_sample_key_set(csv_path: Path) -> Optional[Set[str]]:
    """函数功能：读取轻量 sample_key 集合；路径不存在时返回 None 并让 metadata 记录缺失。"""
    if not csv_path.exists():
        return None
    return set(pd.read_csv(csv_path, usecols=["sample_key"])["sample_key"].astype(str))


def compute_p0_overlap(frames: Mapping[str, pd.DataFrame], p0_sample_dir: Path) -> Dict[str, object]:
    """
    函数功能：
        统计 Round2 small sample 与 P0 pilot sample sets 的 overlap ratio。

    说明：
        P0 三个主集合加诊断集合约 27.5 万行，读取 sample_key 集合成本可控；这里不触碰
        116M prediction manifest。
    """
    p0_files = {
        "pilot_train": p0_sample_dir / "pilot_train_sample_keys.csv",
        "pilot_selection": p0_sample_dir / "pilot_selection_sample_keys.csv",
        "diagnostic_balanced": p0_sample_dir / "diagnostic_balanced_sample_keys.csv",
        "pilot_test": p0_sample_dir / "pilot_test_sample_keys.csv",
    }
    p0_sets = {name: load_sample_key_set(path) for name, path in p0_files.items()}
    available = {name: keys for name, keys in p0_sets.items() if keys is not None}
    if not available:
        return {"status": "not_available", "p0_sample_dir": str(p0_sample_dir), "details": {}}

    p0_union: Set[str] = set().union(*available.values())
    details: Dict[str, object] = {}
    for round2_name, frame in frames.items():
        keys = set(frame["sample_key"].astype(str))
        row = {
            "round2_count": int(len(keys)),
            "overlap_with_p0_any_count": int(len(keys.intersection(p0_union))),
            "overlap_with_p0_any_ratio": float(len(keys.intersection(p0_union)) / len(keys)) if keys else 0.0,
        }
        for p0_name, p0_keys in available.items():
            count = int(len(keys.intersection(p0_keys)))
            row[f"overlap_with_{p0_name}_count"] = count
            row[f"overlap_with_{p0_name}_ratio"] = float(count / len(keys)) if keys else 0.0
        details[round2_name] = row
    return {
        "status": "computed",
        "p0_sample_dir": str(p0_sample_dir),
        "available_p0_sets": sorted(available),
        "details": details,
    }


def write_top3fold_reuse_audit(path: Path) -> None:
    """函数功能：写出已有 top3fold 实现位置、输入输出和 Round2 复用边界。"""
    text = f"""# Round2 top3fold 复用审计

生成时间：{now_cst()}

## 结论

已有 top3fold 实现可复用，路径为 `visual_router_experiments/common/pseudo_imageization.py`。当前实现已经支持从历史窗口 `x` 计算 FFT top-k 周期，并把 top1/top2/top3 period fold 作为 3 个 ViT 输入通道输出。

## 已有实现位置

- `select_fft_periods(x, top_k=3, period_candidates=None)`：基于历史窗口 FFT power 选择每个样本的 top-k 周期；支持动态 FFT top-k 和固定候选周期桶。
- `make_default_period_candidates(history_length, device=...)` / `parse_period_candidates(...)`：用于大规模 online 路径的固定候选周期解析。
- `_fold_fixed_period_batch(series, period, image_size, pixel_mode, clip)`：按指定周期对一批序列进行 padding、fold 和双线性 resize。
- `_period_fold_batch(series, periods, period_column, image_size, pixel_mode, clip)`：按每个样本选中的周期分桶批量 fold。
- `imageize_top3fold(x, image_size=224, periods=None, period_candidates=None, pixel_mode="vision", clip=5.0)`：构造 `[B, 3, H, W]` top3fold 伪图像，三个 channel 分别为 top1/top2/top3 fold。
- `visual_router_experiments/common/vit_embedding_utils.py::make_pseudo_images(...)`：当前仅通过 `variant_b_top3fold` 暴露 top3fold 入口。

## 输入输出

- 输入：历史窗口 tensor，支持 `[B, L, 1]` 或可被 `_as_series_batch` 归一成 `[B, L]` 的形式；Round1 正式路径先执行 RevIN-style window normalization。
- 输出：`imageize_top3fold` 返回 `[B, 3, image_size, image_size]`，数值范围 `[0, 1]`；进入 ViT 前由 `encoder_normalize(..., preset="hf_vit_0_5")` 标准化。
- 数据边界：只使用历史 `x`，不读取未来 `y`、专家预测或 oracle 标签作为输入特征。

## Round2 复用边界

- 可直接复用的部分：FFT 周期选择、固定候选周期、period fold 批处理、ViT-compatible tensor 输出。
- 需要补的部分：为 Round2 layout screening 增加 layout registry/adapter，使 `top3fold_period_layout` 与 `current_rgb_3view`、`spatial_panel_3view`、`line_only` 等候选共用同一 feature-cache 入口参数。
- 不建议本步做的部分：不要在 small sample builder 中生成 pseudo image tensor，不要跑 frozen ViT，不要把 top3fold 与 independent view encoder 同时作为一个混合因素测试。

## 风险

- hard top-k period 在周期估计不稳时可能放大噪声；
- top-k 周期本身可能形成 dataset shortcut；
- 当前 `variant_b_top3fold` 是 channel-packed 设计，若 Round2 想测试 spatial panel top3fold，需要新增 layout，而不是把它混同为已有实现。
"""
    path.write_text(text, encoding="utf-8")


def write_screening_summary(
    path: Path,
    *,
    metadata: Mapping[str, object],
    validation: Mapping[str, object],
    p0_overlap: Mapping[str, object],
) -> None:
    """函数功能：写出便于人工审阅的 Round2 small screening 中文摘要。"""
    counts = metadata["sample_counts"]
    default_layouts = ", ".join(metadata["default_layout_set"])
    deferred_layouts = ", ".join(metadata["deferred_layout_set"])
    text = f"""# Visual Router V2 Round2 Small Screening Summary

生成时间：{metadata["generated_at"]}

## 本步产物

本步只冻结 Round2 view layout small screening 的样本集、layout candidate registry 和 top3fold 复用审计。未训练 router，未运行 ViT，未生成 feature cache，未保存 pseudo image tensor。

## 样本集

| sample_set | split | count | 用途 |
| --- | --- | ---: | --- |
| round2_train_small | vali | {counts["round2_train_small"]} | 后续小样本 layout router 训练 |
| round2_selection_small | vali | {counts["round2_selection_small"]} | 后续 layout/seed/epoch/hparam 选择 |
| round2_diagnostic_balanced_small | vali | {counts["round2_diagnostic_balanced_small"]} | oracle expert balanced 诊断，不参与选择 |
| round2_test_small | test | {counts["round2_test_small"]} | frozen screening only，不参与训练或选择 |

验证状态：`{validation["status"]}`；跨集合 sample_key 重复数：`{validation["cross_set_duplicate_count"]}`；train/selection 交集：`{validation["round2_train_selection_intersection_count"]}`。

## Layout candidates

第一轮默认 layout set：{default_layouts}

第一轮暂缓：{deferred_layouts}

后端 router/head 固定为 Round1 最强路线：`film_mean_patch_aux`，即 mean_patch visual embedding + RevIN aux FiLM modulation；主指标仍为 raw-soft MAE / MSE / regret，oracle-label accuracy 只作解释指标。

## P0 overlap

P0 overlap 状态：`{p0_overlap["status"]}`。详细比例见 `round2_small_sample_metadata.json` 的 `p0_overlap` 字段。

## 后续耗时估计

- P2a 约 200k feature cache ≈ 5h；
- P2d/P2e final_test_only 约 75k feature cache/eval ≈ 2.5h；
- 35k samples 的单 layout small feature cache 预计约 1-1.5h；
- 5 个 layout 顺序跑约 5-8h；
- 若使用 3 张 GPU 进程级并行，wall time 预计约 2-3h；
- `independent_view_encoder` 若每个 view 单独过 ViT，可能接近 2-3 倍成本，不建议第一轮默认执行。
"""
    path.write_text(text, encoding="utf-8")


def copy_light_summary(output_dir: Path, light_summary_dir: Path, files: Sequence[str]) -> Dict[str, str]:
    """函数功能：把可随仓库审阅的轻量文件复制到 experiment_summaries。"""
    light_summary_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for filename in files:
        source = output_dir / filename
        target = light_summary_dir / filename
        shutil.copy2(source, target)
        copied[filename] = str(target)
    return copied


def main() -> None:
    """函数功能：执行 Round2 small screening 样本集与协议资产构建。"""
    args = parse_args()
    start = time.time()
    validate_inputs(args.oracle_labels_path, args.tsf_enrichment_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, selection_rows, test_rows, gap_boundaries, scan_counters = collect_main_sets_and_gap_boundaries(
        oracle_path=args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        train_size=args.train_size,
        selection_size=args.selection_size,
        test_size=args.test_size,
        gap_reservoir_size=args.gap_quantile_reservoir_size,
    )
    main_keys = {row.sample_key for row in train_rows + selection_rows + test_rows}
    diagnostic_rows = collect_diagnostic_rows(
        oracle_path=args.oracle_labels_path,
        seed=args.seed,
        batch_size=args.batch_size,
        total_size=args.diagnostic_balanced_size,
        excluded_keys=main_keys,
    )

    raw_frames = {
        "round2_train_small": rows_to_frame("round2_train_small", train_rows, gap_boundaries),
        "round2_selection_small": rows_to_frame("round2_selection_small", selection_rows, gap_boundaries),
        "round2_diagnostic_balanced_small": rows_to_frame(
            "round2_diagnostic_balanced_small",
            diagnostic_rows,
            gap_boundaries,
        ),
        "round2_test_small": rows_to_frame("round2_test_small", test_rows, gap_boundaries),
    }
    all_selected_keys = set().union(*(set(frame["sample_key"]) for frame in raw_frames.values()))
    tsf_subset = load_tsf_subset(args.tsf_enrichment_path, all_selected_keys, args.batch_size)
    frames = {name: attach_tsf(frame, tsf_subset) for name, frame in raw_frames.items()}

    output_files: Dict[str, str] = {}
    for name, frame in frames.items():
        output_path = args.output_dir / f"{name}_sample_keys.csv"
        frame.to_csv(output_path, index=False)
        output_files[f"{name}_sample_keys"] = str(output_path)

    manifest = pd.concat([frames[name] for name in frames], ignore_index=True)
    manifest_path = args.output_dir / "round2_small_sample_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    coverage = build_coverage_summary(frames)
    coverage_path = args.output_dir / "round2_coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)

    validation = validate_round2_sample_sets(frames)
    validation_path = args.output_dir / "round2_validation_summary.json"
    write_json(validation_path, validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"Round2 样本集验证失败，详见 {validation_path}")

    layout_candidates = build_layout_candidates()
    layout_path = args.output_dir / "round2_layout_candidates.json"
    write_json(layout_path, {"layout_candidates": layout_candidates})

    top3fold_audit_path = args.output_dir / "round2_top3fold_reuse_audit.md"
    write_top3fold_reuse_audit(top3fold_audit_path)

    p0_overlap = compute_p0_overlap(frames, args.p0_sample_dir)
    default_layout_set = [item["layout_name"] for item in layout_candidates if item["default_in_round2a"]]
    deferred_layout_set = [item["layout_name"] for item in layout_candidates if not item["default_in_round2a"]]
    sample_counts = {name: int(len(frame)) for name, frame in frames.items()}
    metadata = {
        "status": "completed",
        "script_version": SCRIPT_VERSION,
        "generated_at": now_cst(),
        "elapsed_sec": round(time.time() - start, 3),
        "script": str(Path(__file__).resolve()),
        "round2_stage": "small_sample_builder",
        "trained_model": False,
        "built_feature_cache": False,
        "ran_vit": False,
        "saved_pseudo_image_tensor": False,
        "used_pilot_test_for_selection": False,
        "test_small_used_for_selection": False,
        "loaded_116m_prediction_manifest_to_memory": False,
        "oracle_labels_path": str(args.oracle_labels_path),
        "tsf_enrichment_path": str(args.tsf_enrichment_path),
        "output_dir": str(args.output_dir),
        "light_summary_dir": None if args.no_copy_light_summary else str(args.light_summary_dir),
        "sample_source": {
            "round2_train_small": "vali metric=mae oracle labels, stable hash natural distribution",
            "round2_selection_small": "vali metric=mae oracle labels, disjoint stable hash natural distribution",
            "round2_diagnostic_balanced_small": "vali metric=mae oracle labels, oracle_model balanced, excluded from selection",
            "round2_test_small": "test metric=mae oracle labels, frozen screening only",
        },
        "sample_counts": sample_counts,
        "optional_expanded_plan": {
            "round2_train_small": 30_000,
            "round2_selection_small": 10_000,
            "round2_diagnostic_balanced_small": 10_000,
            "round2_test_small": 15_000,
            "status": "recorded_only_not_built",
        },
        "sample_set_boundaries": {
            "round2_train_small_and_round2_selection_small_disjoint": True,
            "round2_diagnostic_balanced_small_used_for_selection": False,
            "round2_test_small_split": "test",
            "round2_test_small_used_for_training_tuning_or_selection": False,
            "all_sets_cross_disjoint": True,
        },
        "hash_seed_or_sampling_rule": {
            "seed": int(args.seed),
            "main_sets": "按 seed+sample_key 的 pandas stable hash score 取每个 split 的最小 N 个；train 和 selection 来自同一 vali 排序的相邻不重叠切片。",
            "diagnostic_seed": int(args.seed + 10_003),
            "error_gap_quantile": "基于 metric=mae oracle 行的稳定哈希 reservoir 估计五分位边界。",
        },
        "oracle_balance_rule": {
            "sample_set": "round2_diagnostic_balanced_small",
            "models": MODEL_ORDER,
            "per_model_target": int(math.ceil(args.diagnostic_balanced_size / len(MODEL_ORDER))),
            "final_size": int(args.diagnostic_balanced_size),
            "selection_usage": "diagnostic_only_not_for_layout_selection",
        },
        "diagnostic_fields": ["sample_key", "dataset_name", "group_name", "oracle_model", "error_gap_quantile", *TSF_COLS],
        "layout_candidates": [item["layout_name"] for item in layout_candidates],
        "default_layout_set": default_layout_set,
        "deferred_layout_set": deferred_layout_set,
        "top3fold_existing_implementation_found": True,
        "top3fold_existing_paths": [
            "visual_router_experiments/common/pseudo_imageization.py::select_fft_periods",
            "visual_router_experiments/common/pseudo_imageization.py::imageize_top3fold",
            "visual_router_experiments/common/vit_embedding_utils.py::make_pseudo_images",
        ],
        "round1_fixed_backend_for_future_screening": {
            "variant": "film_mean_patch_aux",
            "base_visual_input": "mean_patch_embedding",
            "condition_input": "revin_aux",
            "conditioning": "FiLM gamma/beta modulation of visual hidden representation",
            "main_metrics": ["raw-soft MAE", "raw-soft MSE", "raw-soft regret"],
            "interpretation_metric": "oracle-label accuracy",
        },
        "time_estimates_for_protocol": {
            "p2a_200k_feature_cache": "about 5h",
            "p2d_p2e_75k_final_test_only_feature_cache_eval": "about 2.5h",
            "round2_35k_single_layout_feature_cache": "about 1-1.5h",
            "five_layouts_sequential": "about 5-8h",
            "three_gpu_process_parallel_wall_time": "about 2-3h",
            "independent_view_encoder": "deferred; may be 2-3x if each view separately passes ViT",
        },
        "gap_quantile_boundaries": gap_boundaries,
        "oracle_scan_counters": scan_counters,
        "p0_overlap": p0_overlap,
        "validation": validation,
        "output_files": {
            **output_files,
            "round2_small_sample_manifest": str(manifest_path),
            "round2_coverage_summary": str(coverage_path),
            "round2_validation_summary": str(validation_path),
            "round2_layout_candidates": str(layout_path),
            "round2_top3fold_reuse_audit": str(top3fold_audit_path),
        },
    }

    metadata_path = args.output_dir / "round2_small_sample_metadata.json"
    write_json(metadata_path, metadata)

    summary_path = args.output_dir / "round2_small_screening_summary.md"
    write_screening_summary(summary_path, metadata=metadata, validation=validation, p0_overlap=p0_overlap)
    metadata["output_files"]["round2_small_sample_metadata"] = str(metadata_path)
    metadata["output_files"]["round2_small_screening_summary"] = str(summary_path)
    write_json(metadata_path, metadata)

    if not args.no_copy_light_summary:
        copied = copy_light_summary(
            args.output_dir,
            args.light_summary_dir,
            files=[
                "round2_layout_candidates.json",
                "round2_top3fold_reuse_audit.md",
                "round2_small_sample_metadata.json",
                "round2_small_screening_summary.md",
                "round2_coverage_summary.csv",
                "round2_validation_summary.json",
            ],
        )
        metadata["light_summary_files"] = copied
        write_json(metadata_path, metadata)
        shutil.copy2(metadata_path, args.light_summary_dir / "round2_small_sample_metadata.json")

    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
