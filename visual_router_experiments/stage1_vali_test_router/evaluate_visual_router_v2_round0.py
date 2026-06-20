#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round 0 统一 evaluator。

本脚本只做 evaluator、baseline 对齐、诊断汇总和文档：
    - 不训练新模型；
    - 不修改 Visual Router / TimeFuse-style 正式入口；
    - 不覆盖 P0 样本目录或 full-scale 结果目录；
    - 不全量读取 116M prediction manifest。

输出目录默认：
    `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/`
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import (  # noqa: E402
    MODEL_COLUMNS,
    TimeFuseFusor,
    frame_to_markdown,
)
from visual_router_experiments.stage1_vali_test_router.stage1_timefuse_fusor_streaming_reader import (  # noqa: E402
    infer_feature_columns,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router import (  # noqa: E402
    VisualMLPRouter,
    add_soft_fusion_metrics,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import (  # noqa: E402
    build_train_args,
    load_data_config,
    resolve_device as resolve_visual_device,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    build_lightweight_prediction_index,
    iter_online_embedding_batches,
    load_checkpoint as load_visual_checkpoint,
    load_vit_model_with_retry,
    predict_stream_batch,
    resolve_period_candidates,
    scaler_from_state as visual_scaler_from_state,
    windows_from_labels,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import (  # noqa: E402
    resolve_dtype,
)
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import (  # noqa: E402
    TSF_STRATA_COLUMNS,
    add_oracle_and_global_rows,
    align_with_sample_frame,
    choose_global_best_model,
    compare_round0_direction,
    extract_csv_rows_by_sample_keys,
    make_method_rows,
    paired_diagnostics,
    paired_summary,
    read_sample_csv,
    selected_model_counts,
    summarize_method_rows,
)


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_VISUAL_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt"
DEFAULT_TIMEFUSE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-18_stage1_timefuse_fusor_full_scale_gpu23"
DEFAULT_ORACLE_LABELS = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "oracle_labels_full_scale_2026-06-16"
    / "window_oracle_labels.parquet"
)
DEFAULT_FEATURE_SHARD_ROOT = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "timefuse_feature_cache_full_scale_launcher"
    / "shards"
)
DEFAULT_VISUAL_CHECKPOINT = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2"
    / "checkpoints"
    / "latest_96_48_S.pt"
)
DEFAULT_TIMEFUSE_CHECKPOINT = DEFAULT_TIMEFUSE_DIR / "checkpoints" / "latest_timefuse_fusor.pt"
DEFAULT_PREDICTION_INDEX = DEFAULT_VISUAL_DIR / "prediction_manifest_index.sqlite"
DEFAULT_PREDICTION_MANIFEST = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "merged_cache"
    / "manifest.csv"
)


def display_time() -> str:
    """函数功能：生成中文 metadata 与 summary 使用的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round 0 evaluator 参数。"""
    parser = argparse.ArgumentParser(description="Evaluate Visual Router V2 Round 0 pilot sample sets.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    parser.add_argument("--timefuse-dir", type=Path, default=DEFAULT_TIMEFUSE_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--feature-shard-root", type=Path, default=DEFAULT_FEATURE_SHARD_ROOT)
    parser.add_argument("--visual-checkpoint", type=Path, default=DEFAULT_VISUAL_CHECKPOINT)
    parser.add_argument("--timefuse-checkpoint", type=Path, default=DEFAULT_TIMEFUSE_CHECKPOINT)
    parser.add_argument("--prediction-index-path", type=Path, default=DEFAULT_PREDICTION_INDEX)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--csv-chunksize", type=int, default=200_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--feature-chunksize", type=int, default=200_000)
    parser.add_argument("--visual-batch-size", type=int, default=64)
    parser.add_argument("--visual-embedding-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 Round 0 输出文件。")
    return parser.parse_args()


def git_commit_hash() -> str:
    """函数功能：记录当前 repo commit hash；失败时写 unknown 但不中断 evaluator。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写 JSON。"""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def log_stage(message: str) -> None:
    """函数功能：输出阶段进度，便于长时间 I/O 抽取时判断是否仍在推进。"""
    print(f"[{display_time()}] {message}", flush=True)


def load_oracle_subset(labels_path: Path, sample_keys: Sequence[str], *, batch_rows: int) -> pd.DataFrame:
    """
    函数功能：
        从 full-scale oracle parquet 中只抽取 P0 sample_key 的 MAE 标签行。

    说明：
        这里读取的是 evaluator 所需 oracle/error baseline 证据，不作为 router
        deployable feature；函数按 sample_key 子集扫描 parquet，不读取 prediction manifest。
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    key_order = {str(key): idx for idx, key in enumerate(sample_keys)}
    value_set = pa.array(list(key_order), type=pa.string())
    columns = [
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "metric",
        "oracle_model",
        "oracle_value",
        *MODEL_COLUMNS,
    ]
    rows: List[pd.DataFrame] = []
    matched_count = 0
    pf = pq.ParquetFile(labels_path)
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=int(batch_rows), columns=columns), start=1):
        table = pa.Table.from_batches([batch])
        mask = pc.and_(pc.equal(table["metric"], "mae"), pc.is_in(table["sample_key"], value_set=value_set))
        filtered = table.filter(mask)
        if filtered.num_rows:
            rows.append(filtered.to_pandas())
            matched_count += int(filtered.num_rows)
        if batch_idx == 1 or batch_idx % 25 == 0:
            log_stage(f"oracle subset scan batches={batch_idx} matched={matched_count}/{len(key_order)}")
        if matched_count >= len(key_order):
            break
    if not rows:
        raise ValueError("oracle labels 中没有命中 P0 sample_key")
    df = pd.concat(rows, ignore_index=True)
    if df["sample_key"].duplicated().any():
        dup = df.loc[df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"oracle labels 子集 sample_key 重复，示例={dup}")
    present_keys = set(df["sample_key"].astype(str))
    missing = [key for key in sample_keys if str(key) not in present_keys]
    if missing:
        raise ValueError(f"oracle labels 子集缺失 sample_key，missing_count={len(missing)} 示例={missing[:10]}")
    df["_order_index"] = df["sample_key"].astype(str).map(key_order)
    return df.sort_values("_order_index").drop(columns=["_order_index"]).reset_index(drop=True)


def load_prediction_index_records(index_path: Path, sample_keys: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, object]]:
    """函数功能：从 eval-only 已有 SQLite 子集索引读取指定 sample_key 的五专家数组路径。"""
    if not index_path.exists():
        raise FileNotFoundError(f"找不到 prediction SQLite index：{index_path}")
    import sqlite3

    key_set = {str(key) for key in sample_keys}
    records: Dict[Tuple[str, str], Dict[str, object]] = {}
    connection = sqlite3.connect(str(index_path))
    connection.row_factory = sqlite3.Row
    try:
        chunk = 900
        key_list = sorted(key_set)
        for start in range(0, len(key_list), chunk):
            part = key_list[start : start + chunk]
            placeholders = ",".join(["?"] * len(part))
            rows = connection.execute(
                f"SELECT * FROM prediction_index WHERE sample_key IN ({placeholders})",
                part,
            ).fetchall()
            for row in rows:
                record = dict(row)
                records[(str(record["sample_key"]), str(record["model_name"]))] = record
    finally:
        connection.close()
    expected = len(key_set) * len(MODEL_COLUMNS)
    if len(records) != expected:
        raise ValueError(f"prediction index 子集不完整：expected={expected} actual={len(records)}")
    return records


def ensure_prediction_lookup(
    *,
    args: argparse.Namespace,
    sample_keys: Sequence[str],
) -> Dict[Tuple[str, str], Dict[str, object]]:
    """
    函数功能：
        取得 selection/diagnostic 所需的 prediction record 子集。

    说明：
        现有 Visual eval-only SQLite 只覆盖 test 时，先尝试复用；若不完整，则
        在 Round 0 输出目录下为 P0 vali 样本建立专用 subset SQLite index。
        该路径只写 50k sample_key 的 25 万条专家记录，不全量加载 manifest。
    """
    try:
        return load_prediction_index_records(args.prediction_index_path, sample_keys)
    except ValueError as exc:
        log_stage(f"现有 prediction index 不覆盖 vali 子集，改建 Round 0 subset index：{exc}")
    round0_index_path = args.output_dir / "prediction_index_round0_vali.sqlite"
    if round0_index_path.exists():
        log_stage(f"复用已存在 Round 0 subset prediction index：{round0_index_path}")
        from visual_router_experiments.stage1_vali_test_router.train_visual_router_online_streaming import SQLitePredictionIndex

        index = SQLitePredictionIndex(round0_index_path, args.prediction_manifest_path.parent)
    else:
        index = build_lightweight_prediction_index(
            args.prediction_manifest_path,
            sample_keys=sample_keys,
            chunk_read_rows=int(args.csv_chunksize),
            index_db_path=round0_index_path,
        )
    try:
        lookup = index.fetch_records(sample_keys)
    finally:
        index.close()
    expected = len(set(str(key) for key in sample_keys)) * len(MODEL_COLUMNS)
    if len(lookup) != expected:
        raise ValueError(f"Round 0 subset prediction index 查询不完整：expected={expected} actual={len(lookup)}")
    return lookup


def load_timefuse_checkpoint(path: Path) -> Tuple[TimeFuseFusor, StandardScaler, List[str]]:
    """函数功能：加载已训练 TimeFuse-style checkpoint，恢复 fusor/scaler/feature columns。"""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    feature_cols = [str(col) for col in checkpoint["feature_columns"]]
    scaler_state = checkpoint["scaler_state"]
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(scaler_state["mean_"], dtype=np.float64)
    scaler.scale_ = np.asarray(scaler_state["scale_"], dtype=np.float64)
    scaler.var_ = np.asarray(scaler_state["var_"], dtype=np.float64)
    scaler.n_features_in_ = int(scaler_state["n_features_in_"])
    if scaler_state.get("n_samples_seen_") is not None:
        scaler.n_samples_seen_ = scaler_state["n_samples_seen_"]
    fusor = TimeFuseFusor(input_dim=len(feature_cols), output_dim=len(MODEL_COLUMNS))
    fusor.load_state_dict(checkpoint["fusor_state_dict"])
    fusor.eval()
    return fusor, scaler, feature_cols


def load_feature_subset(feature_shard_root: Path, sample_keys: Sequence[str], *, chunksize: int) -> pd.DataFrame:
    """函数功能：从 64 个 TimeFuse feature shard 中抽取 P0 vali 目标样本。"""
    key_order = {str(key): idx for idx, key in enumerate(sample_keys)}
    rows: List[pd.DataFrame] = []
    for shard_idx, feature_path in enumerate(sorted(feature_shard_root.glob("sample_shard_*_of_0064/feature_cache.csv")), start=1):
        for chunk in pd.read_csv(feature_path, chunksize=int(chunksize)):
            matched = chunk[chunk["sample_key"].astype(str).isin(key_order)].copy()
            if not matched.empty:
                rows.append(matched)
        if shard_idx == 1 or shard_idx % 8 == 0:
            current = sum(len(frame) for frame in rows)
            log_stage(f"feature subset scan shards={shard_idx} matched={current}/{len(key_order)}")
    if not rows:
        raise ValueError("TimeFuse feature shard 中没有命中 P0 sample_key")
    df = pd.concat(rows, ignore_index=True)
    if df["sample_key"].duplicated().any():
        dup = df.loc[df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"TimeFuse feature 子集 sample_key 重复，示例={dup}")
    present_keys = set(df["sample_key"].astype(str))
    missing = [key for key in sample_keys if str(key) not in present_keys]
    if missing:
        raise ValueError(f"TimeFuse feature 子集缺失 sample_key，missing_count={len(missing)} 示例={missing[:10]}")
    df["_order_index"] = df["sample_key"].astype(str).map(key_order)
    return df.sort_values("_order_index").drop(columns=["_order_index"]).reset_index(drop=True)


def evaluate_timefuse_subset(
    *,
    sample_set: str,
    sample_df: pd.DataFrame,
    label_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    fusor: TimeFuseFusor,
    scaler: StandardScaler,
    feature_cols: Sequence[str],
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：在 P0 vali 子集上用已训练 TimeFuse checkpoint forward 生成 hard/soft 指标。"""
    aligned_labels = align_with_sample_frame(sample_df, label_df, required_col="oracle_model")
    aligned_features = align_with_sample_frame(sample_df, feature_df, required_col="feature_version")
    x = scaler.transform(aligned_features[list(feature_cols)].to_numpy(dtype=np.float32)).astype(np.float32)
    fusor = fusor.to(device)
    with torch.inference_mode():
        weights = fusor(torch.from_numpy(x).to(device=device)).detach().cpu().numpy()
    selected_idx = weights.argmax(axis=1)
    entropy = -(weights * np.log(np.clip(weights, 1e-8, 1.0))).sum(axis=1)
    pred_df = aligned_labels.copy()
    pred_df["router_name"] = "timefuse_style_fusor_streaming"
    pred_df["selected_model"] = [MODEL_COLUMNS[int(idx)] for idx in selected_idx]
    pred_df["selected_value"] = [float(row[model]) for (_, row), model in zip(pred_df.iterrows(), pred_df["selected_model"])]
    pred_df["regret_to_oracle"] = pred_df["selected_value"].astype(float) - pred_df["oracle_value"].astype(float)
    pred_df["oracle_label_correct"] = pred_df["selected_model"].astype(str) == pred_df["oracle_model"].astype(str)
    pred_df["weight_entropy"] = entropy
    pred_df["normalized_weight_entropy"] = entropy / math.log(len(MODEL_COLUMNS))
    pred_df["max_weight"] = weights.max(axis=1)
    for idx, model_name in enumerate(MODEL_COLUMNS):
        pred_df[f"weight_{model_name}"] = weights[:, idx]
    pred_key_set = set(pred_df["sample_key"].astype(str))
    soft_lookup = {key: value for key, value in prediction_lookup.items() if key[0] in pred_key_set}
    pred_df = add_soft_fusion_metrics(pred_df, soft_lookup)
    hard_rows = make_method_rows(
        sample_set=sample_set,
        method="timefuse_hard_top1",
        pred_df=pred_df,
        mae_col="hard_top1_mae_from_array",
        mse_col="hard_top1_mse_from_array",
    )
    soft_rows = make_method_rows(
        sample_set=sample_set,
        method="timefuse_raw_soft_fusion",
        pred_df=pred_df,
        mae_col="soft_fusion_mae",
        mse_col="soft_fusion_mse",
    )
    return pd.concat([hard_rows, soft_rows], ignore_index=True)


def evaluate_visual_subset(
    *,
    sample_set: str,
    sample_df: pd.DataFrame,
    label_df: pd.DataFrame,
    checkpoint_path: Path,
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    """函数功能：在 P0 vali 子集上用已训练 Visual checkpoint 和在线 ViT forward 生成指标。"""
    cache_path = output_dir / f"round0_cache_{sample_set}_visual_rows.csv"
    if cache_path.exists():
        log_stage(f"复用 Visual subset cache：{cache_path}")
        return pd.read_csv(cache_path)
    checkpoint = load_visual_checkpoint(checkpoint_path)
    visual_args = SimpleNamespace(
        labels_path=args.oracle_labels_path,
        prediction_manifest_path=DEFAULT_VISUAL_DIR / "unused_manifest.csv",
        config_path=Path(checkpoint["config_path"]),
        metric="mae",
        router_mode=checkpoint["router_mode"],
        huber_beta=float(checkpoint["huber_beta"]),
        kl_tau=float(checkpoint["kl_tau"]),
        lambda_kl=float(checkpoint["lambda_kl"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
        epochs=0,
        batch_size=int(args.visual_batch_size),
        lr=float(checkpoint["lr"]),
        weight_decay=float(checkpoint["weight_decay"]),
        seed=16,
        device=args.device,
        skip_soft_fusion=False,
        encoder_name=checkpoint["embedding_metadata"]["encoder_name"],
        variant=checkpoint["embedding_metadata"]["variant"],
        pooling=checkpoint["embedding_metadata"]["pooling"],
        normalization_preset=checkpoint["embedding_metadata"]["normalization_preset"],
        embedding_batch_size=int(args.visual_embedding_batch_size),
        image_size=int(checkpoint["embedding_metadata"]["image_size"]),
        norm_mode=checkpoint["embedding_metadata"]["norm_mode"],
        pixel_mode=checkpoint["embedding_metadata"]["pixel_mode"],
        clip=float(checkpoint["embedding_metadata"]["clip"]),
        period_selection=checkpoint["embedding_metadata"]["period_selection"],
        period_candidates=None,
        dtype=checkpoint["embedding_metadata"].get("dtype_arg", "auto"),
        local_files_only=bool(args.local_files_only),
        vit_data_parallel=True,
        stream_shard_index=0,
        stream_shard_count=1,
        max_samples_per_split=None,
        chunk_read_rows=200_000,
        status_update_interval=50,
        print_rows=0,
        resume_checkpoint=checkpoint_path,
        train_only=False,
        verify_evaluation_adapter=False,
        verify_training_expert_batch=False,
    )
    device = resolve_visual_device(visual_args.device)
    dtype = resolve_dtype(visual_args.dtype, device)
    data_config = load_data_config(visual_args.config_path)
    period_candidate_values = resolve_period_candidates(visual_args, int(data_config.seq_len))
    vit_model = load_vit_model_with_retry(visual_args, device, dtype)
    scaler = visual_scaler_from_state(checkpoint["scaler_state"])
    router = VisualMLPRouter(
        input_dim=int(scaler.n_features_in_),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    labels = align_with_sample_frame(sample_df, label_df, required_col="oracle_model")
    labels["split"] = labels["split"].astype(str)
    labels_by_key = labels.set_index("sample_key").to_dict(orient="index")
    windows_df = windows_from_labels(labels)
    pred_frames: List[pd.DataFrame] = []
    for batch_manifest_df, embeddings, _ in iter_online_embedding_batches(
        windows_df=windows_df,
        data_config=data_config,
        vit_model=vit_model,
        args=visual_args,
        device=device,
        dtype=dtype,
        period_candidate_values=period_candidate_values,
    ):
        pred_frames.append(
            predict_stream_batch(
                router=router,
                scaler=scaler,
                batch_manifest_df=batch_manifest_df,
                embeddings=embeddings,
                labels_by_key=labels_by_key,
                router_name="visual_router_mlp_v3_fusion_huber_kl_online_vit_streaming",
                device=device,
            )
        )
    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df = align_with_sample_frame(sample_df, pred_df)
    pred_key_set = set(pred_df["sample_key"].astype(str))
    soft_lookup = {key: value for key, value in prediction_lookup.items() if key[0] in pred_key_set}
    soft_df = add_soft_fusion_metrics(pred_df, soft_lookup)
    soft_df = align_with_sample_frame(sample_df, soft_df)
    hard_rows = make_method_rows(
        sample_set=sample_set,
        method="visual_router_hard_top1",
        pred_df=soft_df,
        mae_col="hard_top1_mae_from_array",
        mse_col="hard_top1_mse_from_array",
    )
    soft_rows = make_method_rows(
        sample_set=sample_set,
        method="visual_router_raw_soft_fusion",
        pred_df=soft_df,
        mae_col="soft_fusion_mae",
        mse_col="soft_fusion_mse",
    )
    rows = pd.concat([hard_rows, soft_rows], ignore_index=True)
    rows.to_csv(cache_path, index=False)
    return rows


def make_test_rows(
    *,
    sample_df: pd.DataFrame,
    visual_dir: Path,
    timefuse_dir: Path,
    chunksize: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """函数功能：从 full-scale test 逐样本 CSV 抽取 Visual/TimeFuse 的 P0 子集行。"""
    keys = sample_df["sample_key"].astype(str).tolist()
    visual_soft = extract_csv_rows_by_sample_keys(
        visual_dir / "visual_router_soft_fusion_predictions.csv",
        sample_keys=keys,
        chunksize=chunksize,
    )
    visual_soft = align_with_sample_frame(sample_df, visual_soft)
    timefuse = extract_csv_rows_by_sample_keys(
        timefuse_dir / "timefuse_fusor_predictions.csv",
        sample_keys=keys,
        chunksize=chunksize,
    )
    timefuse = align_with_sample_frame(sample_df, timefuse)
    visual_rows = pd.concat(
        [
            make_method_rows(
                sample_set="pilot_test",
                method="visual_router_hard_top1",
                pred_df=visual_soft,
                mae_col="hard_top1_mae_from_array",
                mse_col="hard_top1_mse_from_array",
            ),
            make_method_rows(
                sample_set="pilot_test",
                method="visual_router_raw_soft_fusion",
                pred_df=visual_soft,
                mae_col="soft_fusion_mae",
                mse_col="soft_fusion_mse",
            ),
        ],
        ignore_index=True,
    )
    timefuse_rows = pd.concat(
        [
            make_method_rows(
                sample_set="pilot_test",
                method="timefuse_hard_top1",
                pred_df=timefuse,
                mae_col="hard_top1_mae_from_array",
                mse_col="hard_top1_mse_from_array",
            ),
            make_method_rows(
                sample_set="pilot_test",
                method="timefuse_raw_soft_fusion",
                pred_df=timefuse,
                mae_col="soft_fusion_mae",
                mse_col="soft_fusion_mse",
            ),
        ],
        ignore_index=True,
    )
    return visual_rows, timefuse_rows


def write_summary_md(
    *,
    output_dir: Path,
    main_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    direction_ok: bool,
    direction_messages: Sequence[str],
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写 Round 0 中文结论摘要。"""
    lines = [
        "# Visual Router V2 Round 0 Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## P0 v1 代表性结论",
        "",
        f"- 是否足够代表 full-scale：{'是' if direction_ok else '否'}。",
        f"- 是否建议进入 Round 1：{'可以进入 Round 1' if direction_ok else '不能进入 Round 1'}。",
    ]
    if not direction_ok:
        lines.append("- 调整建议：重新构造 P0 样本集，增加对 full-scale 关键方向的约束，至少同时校验 MAE、MSE、oracle-label accuracy 和 selected_model 分布。")
    lines.extend(["", "## 方向检查", ""])
    lines.extend([f"- {msg}" for msg in direction_messages])
    lines.extend(["", "## pilot_test 主表", "", frame_to_markdown(main_df), ""])
    lines.extend(["## pilot_selection 参照表", "", frame_to_markdown(selection_df), ""])
    lines.extend(["## paired diagnostic 汇总", "", frame_to_markdown(paired_summary(paired_df)), ""])
    lines.extend(
        [
            "## 输入与边界",
            "",
            f"- P0 sample dir：`{metadata['inputs']['sample_dir']}`",
            f"- Visual full-scale dir：`{metadata['inputs']['visual_dir']}`",
            f"- TimeFuse full-scale dir：`{metadata['inputs']['timefuse_dir']}`",
            f"- commit hash：`{metadata['commit_hash']}`",
            "- 本轮未训练新模型，未覆盖 P0 或 full-scale 输出目录。",
            "",
        ]
    )
    (output_dir / "round0_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 Round 0 全流程并写出验收要求的产物。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        "round0_main_comparison.csv",
        "round0_selection_comparison.csv",
        "round0_diagnostic_balanced_summary.csv",
        "round0_selected_model_counts.csv",
        "round0_stratified_summary.csv",
        "round0_paired_diagnostics.csv",
        "round0_metadata.json",
        "round0_summary.md",
    ]
    existing = [name for name in required_outputs if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"输出目录已有 Round 0 产物；如需覆盖请传 --overwrite：{existing}")

    log_stage("读取 P0 sample CSV")
    pilot_test = read_sample_csv(args.sample_dir / "pilot_test_sample_keys.csv")
    pilot_selection = read_sample_csv(args.sample_dir / "pilot_selection_sample_keys.csv")
    diagnostic = read_sample_csv(args.sample_dir / "diagnostic_balanced_sample_keys.csv")
    all_keys = (
        pilot_test["sample_key"].astype(str).tolist()
        + pilot_selection["sample_key"].astype(str).tolist()
        + diagnostic["sample_key"].astype(str).tolist()
    )
    log_stage("抽取 oracle labels 子集")
    label_df_all = load_oracle_subset(args.oracle_labels_path, all_keys, batch_rows=args.parquet_batch_rows)
    selection_labels = label_df_all[label_df_all["sample_key"].isin(pilot_selection["sample_key"].astype(str))].copy()
    global_best_model = choose_global_best_model(selection_labels)

    log_stage("抽取 pilot_test Visual/TimeFuse full-scale 逐样本结果")
    visual_test_rows, timefuse_test_rows = make_test_rows(
        sample_df=pilot_test,
        visual_dir=args.visual_dir,
        timefuse_dir=args.timefuse_dir,
        chunksize=args.csv_chunksize,
    )
    test_label_df = label_df_all[label_df_all["sample_key"].isin(pilot_test["sample_key"].astype(str))].copy()
    test_baseline_rows = add_oracle_and_global_rows(
        sample_set="pilot_test",
        sample_df=pilot_test,
        label_df=test_label_df,
        global_best_model=global_best_model,
    )
    test_rows = pd.concat([visual_test_rows, timefuse_test_rows, test_baseline_rows], ignore_index=True)

    eval_keys = pilot_selection["sample_key"].astype(str).tolist() + diagnostic["sample_key"].astype(str).tolist()
    log_stage("读取 prediction SQLite 子集记录")
    prediction_lookup = ensure_prediction_lookup(args=args, sample_keys=eval_keys)
    log_stage("抽取 TimeFuse feature 子集")
    feature_df = load_feature_subset(args.feature_shard_root, eval_keys, chunksize=args.feature_chunksize)
    timefuse_model, timefuse_scaler, feature_cols = load_timefuse_checkpoint(args.timefuse_checkpoint)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    log_stage("重算 pilot_selection Visual checkpoint forward")
    visual_selection_rows = evaluate_visual_subset(
        sample_set="pilot_selection",
        sample_df=pilot_selection,
        label_df=selection_labels,
        checkpoint_path=args.visual_checkpoint,
        prediction_lookup=prediction_lookup,
        args=args,
        output_dir=args.output_dir,
    )
    log_stage("重算 pilot_selection TimeFuse checkpoint forward")
    timefuse_selection_rows = evaluate_timefuse_subset(
        sample_set="pilot_selection",
        sample_df=pilot_selection,
        label_df=selection_labels,
        feature_df=feature_df[feature_df["sample_key"].isin(pilot_selection["sample_key"].astype(str))].copy(),
        fusor=timefuse_model,
        scaler=timefuse_scaler,
        feature_cols=feature_cols,
        prediction_lookup=prediction_lookup,
        device=device,
    )
    selection_baseline_rows = add_oracle_and_global_rows(
        sample_set="pilot_selection",
        sample_df=pilot_selection,
        label_df=selection_labels,
        global_best_model=global_best_model,
    )
    selection_rows = pd.concat([visual_selection_rows, timefuse_selection_rows, selection_baseline_rows], ignore_index=True)

    diagnostic_labels = label_df_all[label_df_all["sample_key"].isin(diagnostic["sample_key"].astype(str))].copy()
    log_stage("重算 diagnostic_balanced Visual checkpoint forward")
    visual_diag_rows = evaluate_visual_subset(
        sample_set="diagnostic_balanced",
        sample_df=diagnostic,
        label_df=diagnostic_labels,
        checkpoint_path=args.visual_checkpoint,
        prediction_lookup=prediction_lookup,
        args=args,
        output_dir=args.output_dir,
    )
    log_stage("重算 diagnostic_balanced TimeFuse checkpoint forward")
    timefuse_diag_rows = evaluate_timefuse_subset(
        sample_set="diagnostic_balanced",
        sample_df=diagnostic,
        label_df=diagnostic_labels,
        feature_df=feature_df[feature_df["sample_key"].isin(diagnostic["sample_key"].astype(str))].copy(),
        fusor=timefuse_model,
        scaler=timefuse_scaler,
        feature_cols=feature_cols,
        prediction_lookup=prediction_lookup,
        device=device,
    )
    diag_baseline_rows = add_oracle_and_global_rows(
        sample_set="diagnostic_balanced",
        sample_df=diagnostic,
        label_df=diagnostic_labels,
        global_best_model=global_best_model,
    )
    diagnostic_rows = pd.concat([visual_diag_rows, timefuse_diag_rows, diag_baseline_rows], ignore_index=True)

    log_stage("汇总 comparison、selected counts、stratified summary 和 paired diagnostics")
    main_comparison = summarize_method_rows(test_rows)
    selection_comparison = summarize_method_rows(selection_rows)
    diagnostic_summary = summarize_method_rows(diagnostic_rows)
    selected_counts = selected_model_counts(pd.concat([test_rows, selection_rows, diagnostic_rows], ignore_index=True))
    stratified_summary = summarize_method_rows(
        pd.concat([test_rows, selection_rows, diagnostic_rows], ignore_index=True),
        group_cols=TSF_STRATA_COLUMNS,
    )
    paired_test = paired_diagnostics(sample_set="pilot_test", visual_rows=visual_test_rows, timefuse_rows=timefuse_test_rows)
    paired_selection = paired_diagnostics(sample_set="pilot_selection", visual_rows=visual_selection_rows, timefuse_rows=timefuse_selection_rows)
    paired_diag = paired_diagnostics(sample_set="diagnostic_balanced", visual_rows=visual_diag_rows, timefuse_rows=timefuse_diag_rows)
    paired_all = pd.concat([paired_test, paired_selection, paired_diag], ignore_index=True)

    full_scale_ref = {
        "visual_hard_mae": 0.5615367653135453,
        "visual_soft_mae": 0.5174675759559787,
        "visual_soft_mse": 143.567498,
        "timefuse_hard_mae": 0.4594660364735913,
        "timefuse_soft_mae": 0.4473909307898316,
        "timefuse_soft_mse": 181.4316851408799,
    }
    direction_ok, direction_messages = compare_round0_direction(main_comparison, full_scale_ref)

    main_comparison.to_csv(args.output_dir / "round0_main_comparison.csv", index=False)
    selection_comparison.to_csv(args.output_dir / "round0_selection_comparison.csv", index=False)
    diagnostic_summary.to_csv(args.output_dir / "round0_diagnostic_balanced_summary.csv", index=False)
    selected_counts.to_csv(args.output_dir / "round0_selected_model_counts.csv", index=False)
    stratified_summary.to_csv(args.output_dir / "round0_stratified_summary.csv", index=False)
    paired_all.to_csv(args.output_dir / "round0_paired_diagnostics.csv", index=False)

    metadata = {
        "generated_at": display_time(),
        "commit_hash": git_commit_hash(),
        "inputs": {
            "sample_dir": str(args.sample_dir),
            "sample_metadata_path": str(args.sample_dir / "sample_set_metadata.json"),
            "visual_dir": str(args.visual_dir),
            "timefuse_dir": str(args.timefuse_dir),
            "oracle_labels_path": str(args.oracle_labels_path),
            "visual_checkpoint": str(args.visual_checkpoint),
            "timefuse_checkpoint": str(args.timefuse_checkpoint),
            "prediction_index_path": str(args.prediction_index_path),
            "prediction_manifest_path": str(args.prediction_manifest_path),
            "round0_prediction_index_path": str(args.output_dir / "prediction_index_round0_vali.sqlite"),
            "feature_shard_root": str(args.feature_shard_root),
        },
        "sample_counts": {
            "pilot_train": int(pd.read_csv(args.sample_dir / "pilot_train_sample_keys.csv", usecols=["sample_key"]).shape[0]),
            "pilot_selection": int(len(pilot_selection)),
            "pilot_test": int(len(pilot_test)),
            "diagnostic_balanced": int(len(diagnostic)),
        },
        "global_best_single_source": "pilot_selection_vali",
        "global_best_single_model": global_best_model,
        "direction_ok": bool(direction_ok),
        "direction_messages": list(direction_messages),
        "constraints": {
            "trained_new_model": False,
            "modified_formal_entrypoints": False,
            "loaded_full_prediction_manifest": False,
            "built_subset_prediction_index": True,
            "subset_prediction_index_sample_keys": int(len(set(str(key) for key in eval_keys))),
        },
    }
    write_json(args.output_dir / "round0_metadata.json", metadata)
    write_summary_md(
        output_dir=args.output_dir,
        main_df=main_comparison,
        selection_df=selection_comparison,
        paired_df=paired_all,
        direction_ok=direction_ok,
        direction_messages=direction_messages,
        metadata=metadata,
    )
    print(f"wrote Round 0 outputs to {args.output_dir}")
    print(f"direction_ok={direction_ok}")


if __name__ == "__main__":
    main()
