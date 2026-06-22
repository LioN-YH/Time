#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 panel-aware gating / residual architecture smoke。

实验边界：
    - 只读取既有 35k panel pooling feature cache；
    - 不重新跑 ViT，不保存 pseudo image tensor，不训练 router；
    - 默认只取 32 个样本做 shape/finite/gate/norm smoke；
    - `test_small` 不参与设计选择，默认 smoke 使用 selection split 的少量样本。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.visual_router_v2_panel_gating import (  # noqa: E402
    PANEL_GATING_SCHEMA_VERSION,
    PANEL_GATING_VARIANTS,
    PanelGatingConfig,
    build_panel_gating_model,
    panel_concat_to_stack,
    summarize_probe_output,
    validate_panel_inputs,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_pooling_35k_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_gating_architecture_probe"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2"
DEFAULT_ARTIFACT_PREFIX = "panel_gating"
SCRIPT_VERSION = "visual_router_v2_round2_panel_gating_probe_v1"


def now_cst() -> str:
    """函数功能：生成写入 metadata/status/summary 的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数并保持原始顺序去重。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_args() -> argparse.Namespace:
    """函数功能：解析 panel gating architecture smoke 参数。"""
    parser = argparse.ArgumentParser(description="Probe Round2 panel-aware gating/residual representations.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument("--sample-set", default="round2_selection_small")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--variants", default=",".join(PANEL_GATING_VARIANTS))
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--lowrank-dim", type=int, default=256)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve_manifest_path(feature_dir: Path, feature_manifest: Path | None) -> Path:
    """函数功能：定位既有 panel pooling feature manifest。"""
    if feature_manifest is not None:
        return Path(feature_manifest)
    matches = sorted(Path(feature_dir).glob("*feature_manifest.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"无法唯一定位 feature manifest：feature_dir={feature_dir} matches={matches}")
    return matches[0]


def load_feature_smoke_batch(
    *,
    feature_manifest_path: Path,
    sample_set: str,
    max_samples: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """
    函数功能：
        从既有 panel pooling feature shard 中读取少量样本，构造 gating smoke 输入。
    """
    if int(max_samples) <= 0:
        raise ValueError(f"max_samples 必须为正数，实际={max_samples}")
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].sort_values("start_order_index", kind="mergesort")
    if rows.empty:
        raise ValueError(f"feature manifest 缺少 sample_set={sample_set}")
    global_parts: List[np.ndarray] = []
    panel_concat_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    shard_paths: List[str] = []
    loaded = 0
    for row in rows.itertuples(index=False):
        if loaded >= int(max_samples):
            break
        shard_path = Path(str(row.shard_path))
        with np.load(shard_path, allow_pickle=True) as data:
            take = min(int(data["global_mean_patch"].shape[0]), int(max_samples) - loaded)
            global_parts.append(np.asarray(data["global_mean_patch"][:take], dtype=np.float32))
            panel_concat_parts.append(np.asarray(data["panel_mean_concat"][:take], dtype=np.float32))
            aux_parts.append(np.asarray(data["revin_aux"][:take], dtype=np.float32))
            key_parts.extend([str(value) for value in data["sample_key"][:take].tolist()])
            order_parts.append(np.asarray(data["order_index"][:take], dtype=np.int64))
        shard_paths.append(str(shard_path))
        loaded += take
    if loaded != int(max_samples):
        raise ValueError(f"实际读取样本数不足：loaded={loaded} max_samples={max_samples}")
    global_mean_patch = torch.from_numpy(np.concatenate(global_parts, axis=0).astype(np.float32, copy=False))
    panel_mean_concat = torch.from_numpy(np.concatenate(panel_concat_parts, axis=0).astype(np.float32, copy=False))
    revin_aux = torch.from_numpy(np.concatenate(aux_parts, axis=0).astype(np.float32, copy=False))
    panel_mean_stack = panel_concat_to_stack(panel_mean_concat)
    validate_panel_inputs(global_mean_patch, panel_mean_stack)
    if not torch.isfinite(revin_aux).all():
        raise ValueError("revin_aux 中存在 NaN/Inf")
    metadata = {
        "feature_manifest_path": str(feature_manifest_path),
        "sample_set": str(sample_set),
        "max_samples": int(max_samples),
        "loaded_samples": int(loaded),
        "sample_key_head": key_parts[:5],
        "order_index_head": [int(value) for value in np.concatenate(order_parts, axis=0)[:5].tolist()],
        "source_shard_paths": shard_paths,
        "global_mean_patch_shape": [int(value) for value in global_mean_patch.shape],
        "panel_mean_concat_shape": [int(value) for value in panel_mean_concat.shape],
        "panel_mean_stack_shape": [int(value) for value in panel_mean_stack.shape],
        "revin_aux_shape": [int(value) for value in revin_aux.shape],
        "finite_check": bool(
            torch.isfinite(global_mean_patch).all()
            and torch.isfinite(panel_mean_stack).all()
            and torch.isfinite(revin_aux).all()
        ),
    }
    return {
        "global_mean_patch": global_mean_patch,
        "panel_mean_stack": panel_mean_stack,
        "revin_aux": revin_aux,
    }, metadata


def baseline_report(global_mean_patch: torch.Tensor, panel_mean_stack: torch.Tensor) -> Dict[str, object]:
    """函数功能：记录 baseline global mean 与原始 panel residual 的尺度关系。"""
    panel_delta = panel_mean_stack - global_mean_patch.unsqueeze(1)
    mean_panel_residual = panel_delta.mean(dim=1)
    global_norm = torch.linalg.vector_norm(global_mean_patch, ord=2, dim=1).clamp_min(1.0e-12)
    residual_norm = torch.linalg.vector_norm(mean_panel_residual, ord=2, dim=1)
    panel_to_global = torch.linalg.vector_norm(panel_mean_stack, ord=2, dim=2) / global_norm.unsqueeze(1)
    return {
        "variant": "film_mean_patch_aux",
        "role": "baseline global mean_patch; panel stack is diagnostic only",
        "visual_shape": [int(value) for value in global_mean_patch.shape],
        "visual_finite": bool(torch.isfinite(global_mean_patch).all().detach().cpu().item()),
        "global_norm_mean": float(global_norm.mean().detach().cpu().item()),
        "mean_panel_residual_norm_mean": float(residual_norm.mean().detach().cpu().item()),
        "mean_panel_residual_to_global_norm_ratio_mean": float((residual_norm / global_norm).mean().detach().cpu().item()),
        "panel_token_to_global_norm_ratio_mean": float(panel_to_global.mean().detach().cpu().item()),
        "panel_token_to_global_norm_ratio_max": float(panel_to_global.max().detach().cpu().item()),
    }


def build_summary_text(metadata: Mapping[str, object], variant_summaries: Sequence[Mapping[str, object]]) -> str:
    """函数功能：生成中文 architecture probe 摘要。"""
    rows = []
    for item in variant_summaries:
        rows.append(
            "| {variant} | {shape} | {finite} | {alpha} | {delta_ratio:.6f} | {residual_ratio:.6f} |".format(
                variant=item.get("variant"),
                shape=item.get("visual_shape"),
                finite=item.get("visual_finite"),
                alpha=item.get("alpha", "n/a"),
                delta_ratio=float(item.get("visual_delta_to_global_norm_ratio_mean", 0.0)),
                residual_ratio=float(item.get("panel_residual_to_global_norm_ratio_mean", item.get("mean_panel_residual_to_global_norm_ratio_mean", 0.0))),
            )
        )
    gate_lines = []
    for item in variant_summaries:
        if "gate_min" in item:
            gate_lines.append(
                f"- `{item['variant']}` gate range=[{item['gate_min']:.6f}, {item['gate_max']:.6f}], mean={item['gate_mean']:.6f}, valid={item['gate_range_valid']}。"
            )
        if "attention_min" in item:
            gate_lines.append(
                f"- `{item['variant']}` attention range=[{item['attention_min']:.6f}, {item['attention_max']:.6f}], row_sum_max_abs_error={item['attention_row_sum_max_abs_error']:.6e}。"
            )
    return "\n".join(
        [
            "# Visual Router V2 Round2 panel-aware gating architecture probe",
            "",
            f"生成时间：{metadata['generated_at']}",
            "",
            "## 目的",
            "",
            "本探针在不继续高维 panel concat 的前提下，验证 panel 信息能否作为 `global_mean_patch` 的轻量 gate / residual 调制信号。它只检查 architecture contract 和表示尺度，不给出 35k 性能结论。",
            "",
            "## 背景",
            "",
            "35k panel concat screening 中，selection raw-soft 主指标仍由 baseline `film_mean_patch_aux` 最优：`film_global_panel_mean_aux` 的 selection raw-soft MAE delta 为 +0.000578，`film_panel_mean_aux` 为 +0.001768。两种 panel concat 在 diagnostic/test_small 有局部收益，但 test_small 不能用于选择设计；selection 的 q5 high-error、PatchTST 与 CrossFormer strata 存在退化。因此直接用 2304/3072 维 panel concat 替代或扩展 visual representation 不进入 65k。",
            "",
            "## 最小设计",
            "",
            "- baseline `film_mean_patch_aux`：继续使用 768 维 `global_mean_patch`，RevIN aux 仍通过 FiLM 注入。",
            "- `film_panel_gated_mean_aux`：panel stack 只生成 3 个 sigmoid gate；panel residual 为 `sum_i gate_i * (panel_i - global)` 的归一化加权和；最终 `visual = global + alpha * residual`，输出 768 维。",
            "- `film_panel_lowrank_aux`：2304 维 panel concat 只进入 256 维 bottleneck adapter，再生成 768 维 residual，与 global residual merge。",
            "- `film_panel_attention_aux`：三 panel token 只做极小 softmax attention，输出 768 维 residual，不引入复杂 transformer。",
            "",
            "这些设计把 panel 信息限制为 residual 调制，保留 global mean_patch fallback，可降低高维 concat 带来的过拟合、尺度膨胀和 seed 间专家分配波动风险。",
            "",
            "## Smoke 输入",
            "",
            f"- feature_manifest={metadata['feature_manifest_path']}",
            f"- sample_set={metadata['sample_set']}，max_samples={metadata['max_samples']}，loaded_samples={metadata['loaded_samples']}。",
            f"- global_mean_patch_shape={metadata['global_mean_patch_shape']}，panel_mean_stack_shape={metadata['panel_mean_stack_shape']}，revin_aux_shape={metadata['revin_aux_shape']}。",
            f"- finite_check={metadata['finite_check']}，saved_pseudo_image_tensor=false，rerun_vit=false，full_scale_validation=false。",
            "",
            "## Smoke 结果",
            "",
            "| variant | visual_shape | finite | alpha | visual_delta/global mean | panel_residual/global mean |",
            "| --- | --- | --- | --- | ---: | ---: |",
            *rows,
            "",
            "Gate / attention 检查：",
            "",
            *(gate_lines or ["- 无 gate/attention 字段。"]),
            "",
            "## 判断",
            "",
            "small smoke 证明三个 panel-aware residual/gating candidate 都能在既有 35k panel pooling cache 上产生 finite 的 768 维 visual representation，gate/attention 范围合法，`alpha=0.1` 下 visual delta 相对 global norm 较小，未观察到表示尺度爆炸。",
            "",
            "下一步不应直接进入 65k 或 full-scale。若继续推进，建议另开目标只在 35k small screening 上比较 `film_panel_gated_mean_aux` 与 baseline 的 selection raw-soft MAE/MSE/regret，并重点审计 q5 high-error、PatchTST、CrossFormer strata；只有 selection 主指标和关键 strata 同时不退化，才考虑更大规模验证。",
            "",
        ]
    ) + "\n"


def main() -> None:
    """函数功能：执行 panel-aware gating 表征 smoke 并写出轻量产物。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录已存在且非空：{output_dir}；如需覆盖请传 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = resolve_manifest_path(Path(args.feature_dir), args.feature_manifest)
    tensors, input_metadata = load_feature_smoke_batch(
        feature_manifest_path=feature_manifest_path,
        sample_set=str(args.sample_set),
        max_samples=int(args.max_samples),
    )
    torch.manual_seed(int(args.seed))
    global_mean_patch = tensors["global_mean_patch"]
    panel_mean_stack = tensors["panel_mean_stack"]
    config = PanelGatingConfig(
        visual_dim=int(global_mean_patch.shape[1]),
        panel_count=int(panel_mean_stack.shape[1]),
        gate_hidden_dim=int(args.gate_hidden_dim),
        lowrank_dim=int(args.lowrank_dim),
        init_alpha=float(args.init_alpha),
    )
    variants = parse_csv(args.variants)
    unsupported = [variant for variant in variants if variant not in PANEL_GATING_VARIANTS]
    if unsupported:
        raise ValueError(f"未知 variants={unsupported}，支持 {PANEL_GATING_VARIANTS}")
    variant_summaries: List[Dict[str, object]] = [baseline_report(global_mean_patch, panel_mean_stack)]
    for variant in variants:
        model = build_panel_gating_model(variant, config)
        model.eval()
        with torch.inference_mode():
            output = model(global_mean_patch, panel_mean_stack)
        variant_summaries.append(summarize_probe_output(variant=variant, output=output, global_mean_patch=global_mean_patch))
    metadata: Dict[str, object] = {
        "status": "completed",
        "generated_at": now_cst(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "schema_version": PANEL_GATING_SCHEMA_VERSION,
        "variants": variants,
        "seed": int(args.seed),
        "config": {
            "visual_dim": int(config.visual_dim),
            "panel_count": int(config.panel_count),
            "gate_hidden_dim": int(config.gate_hidden_dim),
            "lowrank_dim": int(config.lowrank_dim),
            "init_alpha": float(config.init_alpha),
        },
        "rerun_vit": False,
        "saved_pseudo_image_tensor": False,
        "trained_router": False,
        "full_scale_validation": False,
        "launched_65k": False,
        "test_small_used_for_design_selection": False,
        "feature_cache_reused": True,
        **input_metadata,
        "variant_summaries": variant_summaries,
    }
    summary_text = build_summary_text(metadata, variant_summaries)
    metadata_path = output_dir / f"{args.artifact_prefix}_metadata.json"
    summary_path = output_dir / f"{args.artifact_prefix}_architecture_probe.md"
    write_json(metadata_path, metadata)
    summary_path.write_text(summary_text, encoding="utf-8")
    write_json(output_dir / "status.json", {"status": "completed", "generated_at": metadata["generated_at"], "metadata_path": str(metadata_path), "summary_path": str(summary_path)})
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_copy = summary_dir / "panel_gating_architecture_probe.md"
    metadata_copy = summary_dir / "panel_gating_metadata.json"
    summary_copy.write_text(summary_text, encoding="utf-8")
    write_json(metadata_copy, metadata)
    print(json.dumps({"status": "completed", "summary": str(summary_copy), "metadata": str(metadata_copy)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
