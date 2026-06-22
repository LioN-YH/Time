#!/usr/bin/env python3
"""
文件功能：
    使用 Round2 panel-wise pooling feature cache 训练轻量 FiLM router probe。

实验边界：
    - 只消费 `probe_visual_router_v2_panel_pooling.py` 生成的 small feature cache；
    - 候选仅为 `film_mean_patch_aux`、`film_panel_mean_aux`、
      `film_global_panel_mean_aux`；
    - aux 仍只通过 FiLM 注入，不回到 direct concat aux；
    - 不构建 full-scale cache，不保存 pseudo image tensor，不用 test 做选择。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import DEFAULT_ORACLE_LABELS, DEFAULT_PREDICTION_MANIFEST, load_oracle_subset  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import SQLitePredictionIndex, build_lightweight_prediction_index, scaler_to_state  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round1_film import (  # noqa: E402
    FiLMRouter,
    build_film_stratified_summary,
    git_commit_hash,
    predict_film_router,
    train_film_router,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_v2_round2_layout_film import (  # noqa: E402
    build_delta_summary,
    read_round2_sample_set,
    sample_sets_from_args,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_panel_pooling import PANEL_POOLING_SCHEMA_VERSION  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    add_batch_fusion_metrics,
    make_visual_pooling_method_rows,
    resolve_device,
    selected_model_counts_with_variant,
    summarize_mean_std,
    summarize_rows_with_seed,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_MANIFEST = DATA2_RUN_OUTPUT_ROOT / "2026-06-21_visual_router_v2_round2_small_samples" / "round2_small_sample_manifest.csv"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_pooling_probe"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-22_visual_router_v2_round2_panel_pooling_probe_train"
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2"
PANEL_VARIANTS = ("film_mean_patch_aux", "film_panel_mean_aux", "film_global_panel_mean_aux")
FEATURE_ARRAY_BY_PANEL_VARIANT = {
    "film_mean_patch_aux": "global_mean_patch",
    "film_panel_mean_aux": "panel_mean_concat",
    "film_global_panel_mean_aux": "global_plus_panel_mean",
}
SCRIPT_VERSION = "visual_router_v2_round2_panel_pooling_train_probe_v1"


def display_time() -> str:
    """函数功能：生成日志和 metadata 时间戳。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seed_list(text: str) -> List[int]:
    """函数功能：解析 seeds。"""
    return [int(value) for value in parse_csv(text)]


def parse_args() -> argparse.Namespace:
    """函数功能：解析 panel pooling FiLM probe 参数。"""
    parser = argparse.ArgumentParser(description="Train panel-wise pooling FiLM probe.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--artifact-prefix", default="round2_panel_pooling")
    parser.add_argument("--variants", default=",".join(PANEL_VARIANTS))
    parser.add_argument("--variant", choices=PANEL_VARIANTS, default=None)
    parser.add_argument("--train-sample-set", default="round2_train_small")
    parser.add_argument("--selection-sample-set", default="round2_selection_small")
    parser.add_argument("--diagnostic-sample-set", default="round2_diagnostic_balanced_small")
    parser.add_argument("--test-sample-set", default="round2_test_small")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--film-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke。")
    parser.add_argument("--run-single", action="store_true")
    parser.add_argument("--build-index-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def eval_sample_sets_from_args(args: argparse.Namespace) -> Tuple[str, str, str]:
    """函数功能：返回 selection/diagnostic/test 三个评估集合。"""
    return (str(args.selection_sample_set), str(args.diagnostic_sample_set), str(args.test_sample_set))


def read_all_sample_frames(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    """函数功能：读取训练、选择、诊断、frozen test 四个 sample set。"""
    return {
        name: read_round2_sample_set(args.sample_manifest, name, max_samples=args.max_samples_per_set)
        for name in sample_sets_from_args(args)
    }


def prediction_index_path(output_dir: Path) -> Path:
    """函数功能：返回本 probe 共用 prediction subset SQLite 路径。"""
    return Path(output_dir) / "prediction_index_round2_panel_pooling_subset.sqlite"


def ensure_prediction_index(args: argparse.Namespace, sample_keys: Sequence[str]) -> SQLitePredictionIndex:
    """函数功能：获取或构建当前 small probe 所需 prediction SQLite index。"""
    index_path = prediction_index_path(args.output_dir)
    if index_path.exists():
        return SQLitePredictionIndex(index_path, args.prediction_manifest_path.parent)
    if args.run_single and not args.build_index_only:
        raise FileNotFoundError(f"prediction index 尚未构建：{index_path}")
    return build_lightweight_prediction_index(
        args.prediction_manifest_path,
        sample_keys=[str(key) for key in sample_keys],
        chunk_read_rows=int(args.csv_chunksize),
        index_db_path=index_path,
    )


def task_dir(output_dir: Path, variant: str, seed: int) -> Path:
    """函数功能：返回 variant/seed 隔离输出目录。"""
    return Path(output_dir) / "tasks" / f"{variant}_seed{int(seed)}"


def load_panel_features(
    *,
    feature_manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
    variant: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """函数功能：读取指定 panel pooling variant 的 visual feature 和 revin_aux。"""
    if variant not in FEATURE_ARRAY_BY_PANEL_VARIANT:
        raise ValueError(f"未知 panel variant={variant}")
    feature_name = FEATURE_ARRAY_BY_PANEL_VARIANT[variant]
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].sort_values("start_order_index", kind="mergesort")
    if rows.empty:
        raise ValueError(f"panel feature manifest 缺少 sample_set={sample_set}")
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    visual_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    loaded = 0
    for row in rows.itertuples(index=False):
        if loaded >= wanted_count:
            break
        shard_path = Path(str(row.shard_path))
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            visual = np.asarray(data[feature_name], dtype=np.float32)
            aux = np.asarray(data["revin_aux"], dtype=np.float32)
        take = min(int(visual.shape[0]), wanted_count - loaded)
        visual_parts.append(visual[:take])
        aux_parts.append(aux[:take])
        key_parts.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded += take
    if loaded != wanted_count or key_parts != expected_keys:
        raise ValueError(f"{variant}/{sample_set} feature 数量或 sample_key 顺序不一致")
    if not np.array_equal(np.concatenate(order_parts, axis=0), sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)):
        raise ValueError(f"{variant}/{sample_set} order_index 不一致")
    visual_features = np.concatenate(visual_parts, axis=0).astype(np.float32, copy=False)
    aux_features = np.concatenate(aux_parts, axis=0).astype(np.float32, copy=False)
    if visual_features.ndim != 2 or visual_features.shape[0] != wanted_count:
        raise ValueError(f"{variant}/{sample_set} visual feature shape 异常：{visual_features.shape}")
    if aux_features.shape != (wanted_count, len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"{variant}/{sample_set} aux feature shape 异常：{aux_features.shape}")
    if not np.isfinite(visual_features).all() or not np.isfinite(aux_features).all():
        raise ValueError(f"{variant}/{sample_set} feature 中存在 NaN/Inf")
    return visual_features, aux_features


def run_build_index_only(args: argparse.Namespace) -> None:
    """函数功能：单进程预构建 small sample prediction SQLite index。"""
    frames = read_all_sample_frames(args)
    keys: List[str] = []
    for name in sample_sets_from_args(args):
        keys.extend(frames[name]["sample_key"].astype(str).tolist())
    index = ensure_prediction_index(args, keys)
    index.close()
    write_json(args.output_dir / "prediction_index_status.json", {"status": "completed", "index_path": str(prediction_index_path(args.output_dir)), "sample_key_count": len(set(keys)), "updated_at": display_time()})


def run_single(args: argparse.Namespace) -> None:
    """函数功能：训练并评估一个 panel pooling variant/seed。"""
    if args.variant is None or args.seed is None:
        raise ValueError("--run-single 必须同时提供 --variant 和 --seed")
    variant = str(args.variant)
    seed = int(args.seed)
    out_dir = task_dir(args.output_dir, variant, seed)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = read_all_sample_frames(args)
    train_set, _, _, _ = sample_sets_from_args(args)
    eval_sets = eval_sample_sets_from_args(args)
    all_keys = [key for name in sample_sets_from_args(args) for key in frames[name]["sample_key"].astype(str).tolist()]
    labels_all = load_oracle_subset(args.oracle_labels_path, all_keys, batch_rows=args.parquet_batch_rows)
    labels_by_set = {name: labels_all[labels_all["sample_key"].isin(frames[name]["sample_key"].astype(str))].copy() for name in eval_sets}
    prediction_index = ensure_prediction_index(args, all_keys)
    device = resolve_device(args.device)
    feature_manifest_path = Path(args.feature_dir) / f"{args.artifact_prefix}_feature_manifest.csv"
    try:
        train_visual, train_aux = load_panel_features(
            feature_manifest_path=feature_manifest_path,
            sample_df=frames[train_set],
            sample_set=train_set,
            variant=variant,
        )
        visual_scaler = StandardScaler()
        aux_scaler = StandardScaler()
        train_visual_scaled = visual_scaler.fit_transform(train_visual).astype(np.float32)
        train_aux_scaled = aux_scaler.fit_transform(train_aux).astype(np.float32)
        router, train_meta = train_film_router(
            train_visual_scaled=train_visual_scaled,
            train_aux_scaled=train_aux_scaled,
            train_sample_keys=frames[train_set]["sample_key"].astype(str).tolist(),
            prediction_index=prediction_index,
            seed=seed,
            device=device,
            hidden_dim=int(args.hidden_dim),
            film_hidden_dim=int(args.film_hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            huber_beta=float(args.huber_beta),
            kl_tau=float(args.kl_tau),
            lambda_kl=float(args.lambda_kl),
            metric=str(args.metric),
        )
        torch.save(
            {
                "script_version": SCRIPT_VERSION,
                "variant": variant,
                "feature_name": FEATURE_ARRAY_BY_PANEL_VARIANT[variant],
                "seed": seed,
                "router_state_dict": router.state_dict(),
                "visual_scaler_state": scaler_to_state(visual_scaler),
                "aux_scaler_state": scaler_to_state(aux_scaler),
            },
            out_dir / f"checkpoint_{variant}_seed{seed}.pt",
        )
        method_frames: List[pd.DataFrame] = []
        for sample_set in eval_sets:
            visual_features, aux_features = load_panel_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=frames[sample_set],
                sample_set=sample_set,
                variant=variant,
            )
            pred = predict_film_router(
                router=router,
                visual_scaler=visual_scaler,
                aux_scaler=aux_scaler,
                visual_features=visual_features,
                aux_features=aux_features,
                sample_df=frames[sample_set],
                labels_df=labels_by_set[sample_set],
                variant=variant,
                seed=seed,
                sample_set=sample_set,
                device=device,
            )
            pred = add_batch_fusion_metrics(pred, prediction_index=prediction_index, metric=str(args.metric), batch_size=int(args.eval_batch_size))
            pred.to_csv(out_dir / f"predictions_{variant}_seed{seed}_{sample_set}.csv", index=False)
            method_frames.append(make_visual_pooling_method_rows(pred, sample_set=sample_set, variant=variant, seed=seed))
        method_rows = pd.concat(method_frames, ignore_index=True)
        seed_results = summarize_rows_with_seed(method_rows)
        method_rows.to_csv(out_dir / "method_rows.csv", index=False)
        seed_results.to_csv(out_dir / "seed_results.csv", index=False)
        write_json(
            out_dir / "task_metadata.json",
            {
                "status": "completed",
                "generated_at": display_time(),
                "script_version": SCRIPT_VERSION,
                "variant": variant,
                "feature_name": FEATURE_ARRAY_BY_PANEL_VARIANT[variant],
                "seed": seed,
                "train_metadata": train_meta,
                "constraints": {
                    "layout_fixed_to_spatial_panel_3view": True,
                    "condition_input": "revin_aux",
                    "used_film": True,
                    "used_concat_aux": False,
                    "saved_pseudo_image_tensor": False,
                    "full_scale_validation": False,
                    "test_used_for_selection": False,
                },
            },
        )
    finally:
        prediction_index.close()


def aggregate(args: argparse.Namespace) -> None:
    """函数功能：汇总 panel pooling variant × seed 输出并写轻量 summary。"""
    variants = [str(args.variant)] if args.variant else parse_csv(args.variants)
    seeds = parse_seed_list(args.seeds)
    _, selection_set, diagnostic_set, test_set = sample_sets_from_args(args)
    method_frames: List[pd.DataFrame] = []
    seed_frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for variant in variants:
        for seed in seeds:
            out_dir = task_dir(args.output_dir, variant, seed)
            for name in ["method_rows.csv", "seed_results.csv"]:
                if not (out_dir / name).exists():
                    missing.append(str(out_dir / name))
            if not missing:
                method_frames.append(pd.read_csv(out_dir / "method_rows.csv"))
                seed_frames.append(pd.read_csv(out_dir / "seed_results.csv"))
    if missing:
        raise FileNotFoundError("panel pooling task 输出不完整：" + "; ".join(missing[:20]))
    method_rows = pd.concat(method_frames, ignore_index=True)
    seed_results = pd.concat(seed_frames, ignore_index=True)
    selection_summary = summarize_mean_std(seed_results, sample_set=selection_set)
    diagnostic_summary = summarize_mean_std(seed_results, sample_set=diagnostic_set)
    test_summary = summarize_mean_std(seed_results, sample_set=test_set)
    selected_counts = selected_model_counts_with_variant(method_rows)
    stratified = build_film_stratified_summary(method_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.artifact_prefix)
    paths = {
        "variant_seed_results": output_dir / f"{prefix}_variant_seed_results.csv",
        "selection_summary": output_dir / f"{prefix}_selection_summary.csv",
        "diagnostic_summary": output_dir / f"{prefix}_diagnostic_summary.csv",
        "test_summary": output_dir / f"{prefix}_test_summary.csv",
        "selected_counts": output_dir / f"{prefix}_selected_model_counts.csv",
        "stratified": output_dir / f"{prefix}_stratified_summary.csv",
        "metadata": output_dir / f"{prefix}_training_metadata.json",
        "summary": output_dir / f"{prefix}_training_summary.md",
    }
    seed_results.to_csv(paths["variant_seed_results"], index=False)
    selection_summary.to_csv(paths["selection_summary"], index=False)
    diagnostic_summary.to_csv(paths["diagnostic_summary"], index=False)
    test_summary.to_csv(paths["test_summary"], index=False)
    selected_counts.to_csv(paths["selected_counts"], index=False)
    stratified.to_csv(paths["stratified"], index=False)
    best = selection_summary.sort_values(["MAE_mean", "variant"], kind="mergesort").head(1).to_dict("records")
    verdict = "需要进入 65k expanded validation" if best and best[0]["variant"] != "film_mean_patch_aux" else "暂不升级到 65k，保持 global mean_patch baseline"
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "feature_schema_version": PANEL_POOLING_SCHEMA_VERSION,
        "commit_hash": git_commit_hash(),
        "variants": variants,
        "seeds": seeds,
        "selection_sample_set": selection_set,
        "diagnostic_sample_set": diagnostic_set,
        "test_sample_set": test_set,
        "best_selection_variant": best[0] if best else None,
        "next_step_recommendation": verdict,
        "saved_pseudo_image_tensor": False,
        "full_scale_validation": False,
        "test_used_for_selection": False,
    }
    write_json(paths["metadata"], metadata)
    summary = "\n".join(
        [
            "# Visual Router V2 Round2 panel-wise pooling training probe",
            "",
            f"生成时间：{metadata['generated_at']}",
            "",
            "## 结论",
            "",
            f"- selection best：`{metadata['best_selection_variant']['variant'] if metadata['best_selection_variant'] else 'N/A'}`。",
            f"- 下一步判断：{verdict}。",
            "- 本汇总只覆盖 small probe，不代表 full-scale 结论。",
            "",
            "## Selection",
            "",
            frame_to_markdown(selection_summary, float_digits=6),
            "",
            "## Diagnostic Balanced",
            "",
            frame_to_markdown(diagnostic_summary, float_digits=6),
            "",
            "## Frozen Test",
            "",
            frame_to_markdown(test_summary, float_digits=6),
            "",
            "## Selected Model Ratio",
            "",
            frame_to_markdown(selected_counts, float_digits=6),
            "",
        ]
    )
    paths["summary"].write_text(summary, encoding="utf-8")
    summary_dir = Path(args.summary_copy_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if Path(path).suffix in {".csv", ".json", ".md"}:
            shutil.copy2(path, summary_dir / Path(path).name)


def main() -> None:
    """函数功能：分发 build-index、single task 或 aggregate。"""
    args = parse_args()
    if args.build_index_only:
        run_build_index_only(args)
    elif args.run_single:
        run_single(args)
    elif args.aggregate_only:
        aggregate(args)
    else:
        raise ValueError("请显式指定 --build-index-only、--run-single 或 --aggregate-only")


if __name__ == "__main__":
    main()

