#!/usr/bin/env python3
"""
文件功能：
    执行 Visual Router V2 Round 1 P2b visual-only pooling 消融。

本脚本只做三个 visual-only 变体：
    - visual_cls_only
    - visual_mean_patch_only
    - visual_cls_mean_concat

边界约束：
    - 不做 RevIN aux-only、visual+aux concat、feature probe 或 ViT finetune；
    - 不重新生成 P2a features，不修改 P2a builder/schema；
    - 只用 pilot_train 训练和 fit scaler，只用 pilot_selection 选择 best variant；
    - diagnostic_balanced 只作为诊断展示，不参与架构选择。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.evaluate_visual_router_v2_round0 import (  # noqa: E402
    DEFAULT_ORACLE_LABELS,
    DEFAULT_PREDICTION_MANIFEST,
    load_oracle_subset,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    SQLitePredictionIndex,
    build_lightweight_prediction_index,
    scaler_to_state,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import TSF_STRATA_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_round1_training import (  # noqa: E402
    POOLING_VARIANTS,
    add_batch_fusion_metrics,
    load_pooling_features,
    make_visual_pooling_method_rows,
    predict_visual_pooling_router,
    read_ordered_sample_csv,
    resolve_device,
    selected_model_counts_with_variant,
    summarize_mean_std,
    summarize_rows_with_seed,
    train_visual_pooling_router,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_visual_pooling"
SCRIPT_VERSION = "visual_router_v2_round1_visual_pooling_p2b_v1"


def display_time() -> str:
    """函数功能：生成写入 metadata/log 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2b visual-only pooling 消融参数。"""
    parser = argparse.ArgumentParser(description="Train Visual Router V2 Round 1 P2b visual-only pooling ablation.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--seeds", default="16,17,18", help="逗号分隔 seeds，正式 P2b 固定为 16,17,18。")
    parser.add_argument("--epochs", type=int, default=3, help="每个 seed 训练 epoch 数，P2b 建议 2-3。")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae"], default="mae")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke；正式运行必须省略。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖 P2b 输出目录中本脚本产物。")
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    """函数功能：解析逗号分隔 seed 列表，并去重保序。"""
    seeds: List[int] = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in seeds:
            seeds.append(value)
    if not seeds:
        raise ValueError("--seeds 不能为空")
    return seeds


def git_commit_hash() -> str:
    """函数功能：记录当前 repo commit hash；失败不影响实验执行。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 JSON。"""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于后台日志监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def required_output_names() -> List[str]:
    """函数功能：列出本脚本会写出的顶层 P2b 产物。"""
    return [
        "visual_pooling_variant_seed_results.csv",
        "visual_pooling_selection_comparison.csv",
        "visual_pooling_diagnostic_summary.csv",
        "visual_pooling_selected_model_counts.csv",
        "visual_pooling_stratified_summary.csv",
        "visual_pooling_best_variant.json",
        "visual_pooling_metadata.json",
        "visual_pooling_summary.md",
        "status.json",
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """函数功能：创建输出目录，并在未显式 overwrite 时避免覆盖既有 P2b 产物。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in required_output_names() if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出目录已有 P2b 产物；如需覆盖请传 --overwrite：{existing}")
    if overwrite:
        for name in required_output_names():
            path = output_dir / name
            if path.exists():
                path.unlink()
        for path in output_dir.glob("checkpoint_*.pt"):
            path.unlink()
        for path in output_dir.glob("predictions_*.csv"):
            path.unlink()


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写 status.json 供外部窗口监控。"""
    data = dict(payload)
    data["updated_at"] = display_time()
    data["output_dir"] = str(output_dir)
    write_json(output_dir / "status.json", data)


def ensure_prediction_index(
    *,
    output_dir: Path,
    round0_dir: Path,
    prediction_manifest_path: Path,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
) -> SQLitePredictionIndex:
    """
    函数功能：
        取得 P2b 需要的 prediction SQLite 子集索引。

    说明：
        优先复用 Round 0 vali subset index；若样本数或覆盖不完整，则在 P2b
        输出目录中建立自己的 SQLite，不覆盖 P1/Round0 产物。
    """
    round0_index = Path(round0_dir) / "prediction_index_round0_vali.sqlite"
    # Round0 vali index 只覆盖 pilot_selection + diagnostic_balanced。正式 P2b 还
    # 需要 pilot_train，因此只有目标 key 规模不超过 Round0 覆盖范围时才尝试复用；
    # 否则必须在 P2b 自己的输出目录构建完整索引，避免训练阶段缺失 pilot_train。
    if round0_index.exists() and len(sample_keys) <= 50_000:
        index = SQLitePredictionIndex(round0_index, prediction_manifest_path.parent)
        lookup = index.fetch_records(sample_keys[: min(32, len(sample_keys))])
        if len(lookup) == min(32, len(sample_keys)) * len(MODEL_COLUMNS):
            log_stage(f"复用 Round0 prediction subset index：{round0_index}")
            return index
        index.close()
    p2b_index_path = Path(output_dir) / "prediction_index_p2b_train_selection_diagnostic.sqlite"
    if p2b_index_path.exists():
        log_stage(f"复用 P2b prediction subset index：{p2b_index_path}")
        return SQLitePredictionIndex(p2b_index_path, prediction_manifest_path.parent)
    log_stage("构建 P2b prediction subset SQLite index")
    return build_lightweight_prediction_index(
        prediction_manifest_path,
        sample_keys=sample_keys,
        chunk_read_rows=int(chunk_read_rows),
        index_db_path=p2b_index_path,
    )


def choose_best_variant(selection_mean_std: pd.DataFrame) -> Dict[str, object]:
    """
    函数功能：
        只基于 pilot_selection 选择 best visual pooling。

    选择口径：
        主指标使用 raw soft fusion MAE mean；若并列则依次看 hard top-1 MAE mean、
        raw soft fusion regret mean 和 oracle-label accuracy mean。
    """
    soft = selection_mean_std[selection_mean_std["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    if soft.empty:
        raise ValueError("selection mean/std 中没有 raw soft fusion 行，无法选择 best variant")
    hard = selection_mean_std[selection_mean_std["method"].astype(str).str.endswith("_hard_top1")].copy()
    hard_lookup = hard.set_index("variant")["MAE_mean"].to_dict()
    soft["hard_top1_MAE_mean"] = soft["variant"].map(hard_lookup)
    soft = soft.sort_values(
        ["MAE_mean", "hard_top1_MAE_mean", "regret_to_oracle_mean", "oracle_label_accuracy_mean"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    best = soft.iloc[0].to_dict()
    return {
        "best_variant": str(best["variant"]),
        "selection_basis": "pilot_selection raw_soft_fusion MAE_mean; tie-breakers hard_top1 MAE_mean, regret_to_oracle_mean, oracle_label_accuracy_mean",
        "selected_from_sample_set": "pilot_selection",
        "diagnostic_balanced_used_for_selection": False,
        "pilot_test_used_for_selection": False,
        "best_row": {key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value) for key, value in best.items()},
    }


def write_summary_md(
    *,
    output_dir: Path,
    selection_summary: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    best_variant: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写中文 P2b summary，直接回答用户验收问题。"""
    selection_soft = selection_summary[selection_summary["method"].astype(str).str.endswith("_raw_soft_fusion")].copy()
    selection_hard = selection_summary[selection_summary["method"].astype(str).str.endswith("_hard_top1")].copy()
    soft_by_variant = selection_soft.set_index("variant")
    cls_mae = float(soft_by_variant.loc["visual_cls_only", "MAE_mean"])
    mean_mae = float(soft_by_variant.loc["visual_mean_patch_only", "MAE_mean"])
    concat_mae = float(soft_by_variant.loc["visual_cls_mean_concat", "MAE_mean"])
    best_name = str(best_variant["best_variant"])
    round0_selection = pd.read_csv(Path(metadata["inputs"]["round0_dir"]) / "round0_selection_comparison.csv")
    round0_visual_soft = float(round0_selection.loc[round0_selection["method"] == "visual_router_raw_soft_fusion", "MAE"].iloc[0])
    round0_visual_hard = float(round0_selection.loc[round0_selection["method"] == "visual_router_hard_top1", "MAE"].iloc[0])
    best_soft_mae = float(soft_by_variant.loc[best_name, "MAE_mean"])
    hard_by_variant = selection_hard.set_index("variant")
    best_hard_mae = float(hard_by_variant.loc[best_name, "MAE_mean"])

    lines = [
        "# Visual Router V2 Round 1 P2b Visual Pooling Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 结论回答",
        "",
        f"1. mean_patch 是否优于 CLS？{'是' if mean_mae < cls_mae else '否'}。pilot_selection raw-soft MAE：mean_patch={mean_mae:.6f}，CLS={cls_mae:.6f}。",
        f"2. CLS+mean concat 是否优于单一 pooling？{'是' if concat_mae < min(cls_mae, mean_mae) else '否'}。concat raw-soft MAE={concat_mae:.6f}，最佳单一 pooling={min(cls_mae, mean_mae):.6f}。",
        f"3. visual-only pooling 变体相对 P1 Round 0 Visual baseline 是否有改善？{'是' if best_soft_mae < round0_visual_soft or best_hard_mae < round0_visual_hard else '否'}。best={best_name}，raw-soft MAE={best_soft_mae:.6f} vs Round0 Visual raw-soft={round0_visual_soft:.6f}；hard MAE={best_hard_mae:.6f} vs Round0 Visual hard={round0_visual_hard:.6f}。",
        f"4. 是否建议后续 visual+aux concat 使用哪个 visual pooling？建议使用 `{best_name}`，依据为 pilot_selection raw-soft MAE mean 最低；diagnostic_balanced 未参与选择。",
        "",
        "## Pilot Selection Mean/Std",
        "",
        frame_to_markdown(selection_summary),
        "",
        "## Diagnostic Balanced Mean/Std",
        "",
        frame_to_markdown(diagnostic_summary),
        "",
        "## Best Variant",
        "",
        f"- best_variant：`{best_name}`",
        f"- selection_basis：{best_variant['selection_basis']}",
        "- 本轮未使用 pilot_test；未训练 ViT；未使用 RevIN aux 或 visual+aux concat。",
        "",
    ]
    (output_dir / "visual_pooling_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 P2b 三变体三 seeds 训练、评估和汇总写盘。"""
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    prepare_output_dir(args.output_dir, overwrite=bool(args.overwrite))
    write_status(args.output_dir, {"status": "started", "script_version": SCRIPT_VERSION})

    feature_manifest_path = Path(args.feature_dir) / "round1_feature_manifest.csv"
    if not feature_manifest_path.exists():
        raise FileNotFoundError(f"找不到 P2a feature manifest：{feature_manifest_path}")
    train_df = read_ordered_sample_csv(args.sample_dir, "pilot_train", max_samples=args.max_samples_per_set)
    selection_df = read_ordered_sample_csv(args.sample_dir, "pilot_selection", max_samples=args.max_samples_per_set)
    diagnostic_df = read_ordered_sample_csv(args.sample_dir, "diagnostic_balanced", max_samples=args.max_samples_per_set)
    all_eval_keys = (
        train_df["sample_key"].astype(str).tolist()
        + selection_df["sample_key"].astype(str).tolist()
        + diagnostic_df["sample_key"].astype(str).tolist()
    )
    log_stage("读取 oracle labels 子集")
    label_df_all = load_oracle_subset(args.oracle_labels_path, all_eval_keys, batch_rows=args.parquet_batch_rows)
    train_labels = label_df_all[label_df_all["sample_key"].isin(train_df["sample_key"].astype(str))].copy()
    selection_labels = label_df_all[label_df_all["sample_key"].isin(selection_df["sample_key"].astype(str))].copy()
    diagnostic_labels = label_df_all[label_df_all["sample_key"].isin(diagnostic_df["sample_key"].astype(str))].copy()
    prediction_index = ensure_prediction_index(
        output_dir=args.output_dir,
        round0_dir=args.round0_dir,
        prediction_manifest_path=args.prediction_manifest_path,
        sample_keys=all_eval_keys,
        chunk_read_rows=args.csv_chunksize,
    )
    device = resolve_device(args.device)
    all_method_rows: List[pd.DataFrame] = []
    train_metadata_rows: List[Dict[str, object]] = []
    try:
        for variant in POOLING_VARIANTS:
            log_stage(f"读取 P2a features：variant={variant}")
            train_features = load_pooling_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=train_df,
                sample_set="pilot_train",
                variant=variant,
            )
            selection_features = load_pooling_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=selection_df,
                sample_set="pilot_selection",
                variant=variant,
            )
            diagnostic_features = load_pooling_features(
                feature_manifest_path=feature_manifest_path,
                sample_df=diagnostic_df,
                sample_set="diagnostic_balanced",
                variant=variant,
            )
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_features).astype(np.float32)
            del train_features
            for seed in seeds:
                log_stage(f"训练 variant={variant} seed={seed}")
                router, train_meta = train_visual_pooling_router(
                    train_features_scaled=train_scaled,
                    train_sample_keys=train_df["sample_key"].astype(str).tolist(),
                    prediction_index=prediction_index,
                    seed=int(seed),
                    device=device,
                    hidden_dim=int(args.hidden_dim),
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
                checkpoint_path = args.output_dir / f"checkpoint_{variant}_seed{int(seed)}.pt"
                torch.save(
                    {
                        "script_version": SCRIPT_VERSION,
                        "variant": variant,
                        "seed": int(seed),
                        "router_state_dict": router.state_dict(),
                        "scaler_state": scaler_to_state(scaler),
                        "model_columns": list(MODEL_COLUMNS),
                        "hyperparameters": {
                            "epochs": int(args.epochs),
                            "batch_size": int(args.batch_size),
                            "hidden_dim": int(args.hidden_dim),
                            "dropout": float(args.dropout),
                            "lr": float(args.lr),
                            "weight_decay": float(args.weight_decay),
                            "huber_beta": float(args.huber_beta),
                            "kl_tau": float(args.kl_tau),
                            "lambda_kl": float(args.lambda_kl),
                        },
                    },
                    checkpoint_path,
                )
                train_metadata_rows.append({"variant": variant, "seed": int(seed), "checkpoint_path": str(checkpoint_path), **train_meta})

                pred_frames: List[pd.DataFrame] = []
                for sample_set, sample_df, labels_df, features in [
                    ("pilot_selection", selection_df, selection_labels, selection_features),
                    ("diagnostic_balanced", diagnostic_df, diagnostic_labels, diagnostic_features),
                ]:
                    pred = predict_visual_pooling_router(
                        router=router,
                        scaler=scaler,
                        features=features,
                        sample_df=sample_df,
                        labels_df=labels_df,
                        variant=variant,
                        seed=int(seed),
                        sample_set=sample_set,
                        device=device,
                    )
                    pred = add_batch_fusion_metrics(
                        pred,
                        prediction_index=prediction_index,
                        metric=str(args.metric),
                        batch_size=int(args.eval_batch_size),
                    )
                    pred.to_csv(args.output_dir / f"predictions_{variant}_seed{int(seed)}_{sample_set}.csv", index=False)
                    pred_frames.append(make_visual_pooling_method_rows(pred, sample_set=sample_set, variant=variant, seed=int(seed)))
                all_method_rows.append(pd.concat(pred_frames, ignore_index=True))
                write_status(
                    args.output_dir,
                    {
                        "status": "running",
                        "completed_variant": variant,
                        "completed_seed": int(seed),
                        "method_rows": int(sum(len(frame) for frame in all_method_rows)),
                    },
                )
            del train_scaled, selection_features, diagnostic_features
    finally:
        prediction_index.close()

    method_rows = pd.concat(all_method_rows, ignore_index=True)
    seed_results = summarize_rows_with_seed(method_rows)
    selection_mean_std = summarize_mean_std(seed_results, sample_set="pilot_selection")
    diagnostic_mean_std = summarize_mean_std(seed_results, sample_set="diagnostic_balanced")
    selected_counts = selected_model_counts_with_variant(method_rows)
    stratified = summarize_rows_with_seed(method_rows, group_cols=TSF_STRATA_COLUMNS)
    best_variant = choose_best_variant(selection_mean_std)

    seed_results.to_csv(args.output_dir / "visual_pooling_variant_seed_results.csv", index=False)
    selection_mean_std.to_csv(args.output_dir / "visual_pooling_selection_comparison.csv", index=False)
    diagnostic_mean_std.to_csv(args.output_dir / "visual_pooling_diagnostic_summary.csv", index=False)
    selected_counts.to_csv(args.output_dir / "visual_pooling_selected_model_counts.csv", index=False)
    stratified.to_csv(args.output_dir / "visual_pooling_stratified_summary.csv", index=False)
    write_json(args.output_dir / "visual_pooling_best_variant.json", best_variant)

    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "commit_hash": git_commit_hash(),
        "inputs": {
            "sample_dir": str(args.sample_dir),
            "round0_dir": str(args.round0_dir),
            "feature_dir": str(args.feature_dir),
            "feature_manifest_path": str(feature_manifest_path),
            "oracle_labels_path": str(args.oracle_labels_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
        },
        "output_dir": str(args.output_dir),
        "variants": list(POOLING_VARIANTS),
        "seeds": [int(seed) for seed in seeds],
        "sample_counts": {
            "pilot_train": int(len(train_df)),
            "pilot_selection": int(len(selection_df)),
            "diagnostic_balanced": int(len(diagnostic_df)),
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "huber_beta": float(args.huber_beta),
            "kl_tau": float(args.kl_tau),
            "lambda_kl": float(args.lambda_kl),
            "metric": str(args.metric),
            "device": str(device),
            "max_samples_per_set": args.max_samples_per_set,
        },
        "constraints": {
            "visual_only": True,
            "used_revin_aux": False,
            "used_visual_aux_concat": False,
            "trained_vit": False,
            "regenerated_p2a_features": False,
            "modified_p2a_builder_or_schema": False,
            "scaler_fit_sample_set": "pilot_train",
            "best_variant_selection_sample_set": "pilot_selection",
            "diagnostic_balanced_used_for_selection": False,
            "pilot_test_used": False,
        },
        "train_metadata": train_metadata_rows,
        "best_variant": best_variant,
    }
    write_json(args.output_dir / "visual_pooling_metadata.json", metadata)
    write_summary_md(
        output_dir=args.output_dir,
        selection_summary=selection_mean_std,
        diagnostic_summary=diagnostic_mean_std,
        best_variant=best_variant,
        metadata=metadata,
    )
    write_status(args.output_dir, {"status": "completed", "best_variant": best_variant})
    log_stage(f"P2b visual pooling outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
