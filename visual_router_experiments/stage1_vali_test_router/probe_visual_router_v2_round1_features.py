#!/usr/bin/env python3
"""
文件功能：
    对 Visual Router V2 Round 1 P2a sharded feature cache 做只读诊断 probe。

设计边界：
    - 只读取 P2a `.npz` feature shard，不重新生成 ViT feature 或伪图像 tensor；
    - 只训练轻量线性 probe，不训练 Visual Router routing head，也不做 hard/soft fusion；
    - scaler、one-hot encoder 和分类器只在 `pilot_train` fit；
    - `pilot_selection` 用于主评估和结论，`diagnostic_balanced` 只做额外诊断；
    - 所有小型 label/cache 只写入本脚本独立输出目录，避免覆盖 P0/P1/P2a/P2b/P2c。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_feature_probe"
DEFAULT_ORACLE_LABELS = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "oracle_labels_full_scale_2026-06-16"
    / "window_oracle_labels.parquet"
)
DEFAULT_TSF_ENRICHMENT = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "tsf_enrichment_full_scale_2026-06-16"
    / "sample_tsf_enrichment.parquet"
)

SAMPLE_SETS = ("pilot_train", "pilot_selection", "diagnostic_balanced")
EVAL_SAMPLE_SETS = ("pilot_selection", "diagnostic_balanced")
FEATURE_GROUPS = {
    "cls_embedding": ("cls_embedding",),
    "mean_patch_embedding": ("mean_patch_embedding",),
    "cls_mean_concat": ("cls_embedding", "mean_patch_embedding"),
    "revin_aux": ("revin_aux",),
    "cls_plus_aux": ("cls_embedding", "revin_aux"),
    "mean_patch_plus_aux": ("mean_patch_embedding", "revin_aux"),
}
REQUIRED_FEATURE_GROUPS = ("cls_embedding", "mean_patch_embedding", "cls_mean_concat", "revin_aux")
STRUCTURE_TARGETS = (
    "error_gap_quantile",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
    "cluster",
    "group_name",
)
TSF_BASELINE_COLUMNS = (
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
    "cluster",
    "group_name",
)


def display_time() -> str:
    """函数功能：生成中文日志和 metadata 中使用的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2probe 命令行参数。"""
    parser = argparse.ArgumentParser(description="Probe Visual Router V2 Round 1 P2a features.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--tsf-enrichment-path", type=Path, default=DEFAULT_TSF_ENRICHMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="smoke 截断；正式运行保持 None。")
    parser.add_argument("--feature-groups", nargs="+", default=list(FEATURE_GROUPS), choices=sorted(FEATURE_GROUPS))
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--max-iter", type=int, default=20, help="线性 SGD probe 的最大 epoch 数；smoke/正式均不做重模型调参。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有 P2probe 输出文件。")
    return parser.parse_args()


def git_commit_hash() -> str:
    """函数功能：记录当前 repo commit hash；失败时写 unknown 但不中断 probe。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写 JSON，保留中文字段。"""
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """函数功能：创建独立输出目录，并在默认情况下拒绝覆盖已有结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        "feature_probe_expert_suitability_results.csv",
        "feature_probe_structure_results.csv",
        "feature_probe_shortcut_baselines.csv",
        "feature_probe_confusion_matrices.csv",
        "feature_probe_within_dataset_summary.csv",
        "feature_probe_metadata.json",
        "feature_probe_summary.md",
    ]
    existing = [name for name in required_outputs if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出目录已有 probe 结果，避免并行覆盖：{output_dir} existing={existing}；如需重跑请显式加 --overwrite")
    for name in required_outputs:
        path = output_dir / name
        if path.exists() and overwrite:
            path.unlink()


def read_sample_csv(sample_dir: Path, sample_set: str, *, max_samples: int | None) -> pd.DataFrame:
    """
    函数功能：
        读取 P0 sample CSV，并校验 probe 所需 oracle/TSF label 已按 order_index 对齐。
    """
    path = Path(sample_dir) / f"{sample_set}_sample_keys.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 P0 sample CSV：{path}")
    df = pd.read_csv(path)
    required = {
        "sample_set",
        "order_index",
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "oracle_model",
        "error_gap",
        *STRUCTURE_TARGETS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} 缺少 probe 必需字段：{missing}")
    df = df.sort_values("order_index", kind="mergesort").reset_index(drop=True)
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()
    if df.empty:
        raise ValueError(f"{path} 读取后为空")
    if df["sample_set"].astype(str).ne(sample_set).any():
        raise ValueError(f"{path} 中 sample_set 与期望不一致：{sample_set}")
    expected_order = np.arange(len(df), dtype=np.int64)
    actual_order = df["order_index"].to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(actual_order, expected_order):
        raise ValueError(f"{path} 的 order_index 必须从 0 连续递增")
    if df["sample_key"].astype(str).duplicated().any():
        dup = df.loc[df["sample_key"].astype(str).duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"{path} 中 sample_key 重复，示例={dup}")
    for col in ("oracle_model", *STRUCTURE_TARGETS):
        if df[col].isna().any():
            raise ValueError(f"{path} 字段 {col} 存在缺失，不能作为 probe label")
    return df


def load_feature_group(
    *,
    feature_manifest_path: Path,
    sample_df: pd.DataFrame,
    sample_set: str,
    feature_group: str,
) -> np.ndarray:
    """
    函数功能：
        从 P2a shard 读取一个 feature group，并严格检查 sample_key/order_index 对齐。

    关键约束：
        concat 只在内存中临时构造，不写回 P2a feature cache。
    """
    manifest = pd.read_csv(feature_manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == str(sample_set)].copy()
    if rows.empty:
        raise ValueError(f"P2a feature manifest 中没有 sample_set={sample_set}")
    rows = rows.sort_values("start_order_index", kind="mergesort").reset_index(drop=True)
    wanted_count = int(len(sample_df))
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    expected_order = sample_df["order_index"].to_numpy(dtype=np.int64, copy=False)
    array_names = FEATURE_GROUPS[feature_group]
    features: List[np.ndarray] = []
    sample_keys: List[str] = []
    order_parts: List[np.ndarray] = []
    loaded_count = 0
    for row in rows.itertuples(index=False):
        if loaded_count >= wanted_count:
            break
        shard_path = Path(str(row.shard_path))
        if not shard_path.exists():
            raise FileNotFoundError(f"找不到 P2a feature shard：{shard_path}")
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            arrays = [np.asarray(data[name], dtype=np.float32) for name in array_names]
        shard_features = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1).astype(np.float32, copy=False)
        take = min(int(shard_features.shape[0]), wanted_count - loaded_count)
        features.append(np.asarray(shard_features[:take], dtype=np.float32))
        sample_keys.extend(shard_keys[:take])
        order_parts.append(shard_order[:take])
        loaded_count += take
    if loaded_count != wanted_count:
        raise ValueError(f"{sample_set}/{feature_group} feature 数量不足：expected={wanted_count} actual={loaded_count}")
    merged_order = np.concatenate(order_parts, axis=0)
    if not np.array_equal(merged_order, expected_order):
        raise ValueError(f"{sample_set}/{feature_group} order_index 与 P0 不一致")
    if sample_keys != expected_keys:
        raise ValueError(f"{sample_set}/{feature_group} sample_key 与 P0 顺序不一致")
    x = np.concatenate(features, axis=0).astype(np.float32, copy=False)
    if x.ndim != 2 or x.shape[0] != wanted_count:
        raise ValueError(f"{sample_set}/{feature_group} feature shape 异常：{x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{sample_set}/{feature_group} feature 中存在 NaN/Inf")
    return x


def train_numeric_probe(
    x_train: np.ndarray,
    y_train: Sequence[object],
    *,
    seed: int,
    max_iter: int,
) -> Pipeline:
    """函数功能：训练 StandardScaler + log-loss SGD 线性 probe。"""
    unique_labels = np.unique(np.asarray(y_train).astype(str))
    if len(unique_labels) < 2:
        # 某些结构标签在 smoke 或正式 P0 子集中可能天然只有一个取值，
        # 此时 probe 退化为常量分类器，仍记录恢复该标签的平凡上限。
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=int(max_iter),
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
            random_state=int(seed),
            n_jobs=-1,
            average=True,
        )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train).astype(str))
    return pipe


def make_onehot_probe(
    categorical_columns: Sequence[str],
    *,
    seed: int,
    max_iter: int,
) -> Pipeline:
    """
    函数功能：
        构造 categorical-only probe。OneHotEncoder 只在 train fit，
        handle_unknown='ignore' 用于 held-out 类别诊断。
    """
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(categorical_columns))],
        remainder="drop",
        sparse_threshold=1.0,
    )
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=int(max_iter),
        n_jobs=-1,
        random_state=int(seed),
    )
    return Pipeline([("onehot", preprocessor), ("clf", clf)])


def safe_log_loss(y_true: Sequence[str], proba: np.ndarray, labels: Sequence[str]) -> float:
    """函数功能：按显式 label 顺序手算 cross entropy，避免 sklearn 类别排序假设。"""
    y = np.asarray(y_true).astype(str)
    label_to_idx = {str(label): idx for idx, label in enumerate(labels)}
    indices = np.asarray([label_to_idx[str(value)] for value in y], dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return float(-np.log(np.clip(p[np.arange(len(indices)), indices], 1e-12, 1.0)).mean())


def top2_recall(y_true: Sequence[str], proba: np.ndarray, labels: Sequence[str]) -> float:
    """函数功能：计算 oracle expert 是否落在 top-2 probability 候选中。"""
    label_array = np.asarray(list(labels)).astype(str)
    top_k = np.argsort(np.asarray(proba), axis=1)[:, -2:]
    y = np.asarray(y_true).astype(str)
    return float(np.mean([y[i] in set(label_array[top_k[i]]) for i in range(len(y))]))


def classification_metrics(
    *,
    model_name: str,
    probe_kind: str,
    sample_set: str,
    target_name: str,
    y_true: Sequence[object],
    y_pred: Sequence[object],
    proba: np.ndarray,
    labels: Sequence[str],
    include_top2: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    """函数功能：统一生成分类 probe 指标、per-class recall 和 confusion matrix 长表。"""
    y_true_arr = np.asarray(y_true).astype(str)
    y_pred_arr = np.asarray(y_pred).astype(str)
    labels_str = [str(label) for label in labels]
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    if len(np.unique(np.concatenate([y_true_arr, y_pred_arr]))) < 2:
        # 单类结构标签没有“类间平衡”的含义，直接用 accuracy 记录平凡恢复表现。
        balanced_accuracy = accuracy
        macro_f1 = accuracy
        per_class_recalls = np.asarray([accuracy], dtype=np.float64)
    else:
        balanced_accuracy = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels_str, average="macro", zero_division=0))
        per_class_recalls = recall_score(y_true_arr, y_pred_arr, labels=labels_str, average=None, zero_division=0)
    row: Dict[str, object] = {
        "probe_kind": probe_kind,
        "model_name": model_name,
        "sample_set": sample_set,
        "target_name": target_name,
        "sample_count": int(len(y_true_arr)),
        "class_count": int(len(labels_str)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "cross_entropy": safe_log_loss(y_true_arr, proba, labels_str),
        "top2_recall": top2_recall(y_true_arr, proba, labels_str) if include_top2 else np.nan,
    }
    recall_rows = [
        {
            **row,
            "metric_scope": "per_class_recall",
            "class_label": str(label),
            "per_class_recall": float(value),
        }
        for label, value in zip(labels_str, per_class_recalls)
    ]
    label_to_cm_idx = {str(label): idx for idx, label in enumerate(labels_str)}
    cm = np.zeros((len(labels_str), len(labels_str)), dtype=np.int64)
    for true_value, pred_value in zip(y_true_arr, y_pred_arr):
        if str(true_value) in label_to_cm_idx and str(pred_value) in label_to_cm_idx:
            cm[label_to_cm_idx[str(true_value)], label_to_cm_idx[str(pred_value)]] += 1
    cm_rows: List[Dict[str, object]] = []
    for i, true_label in enumerate(labels_str):
        for j, pred_label in enumerate(labels_str):
            cm_rows.append(
                {
                    "probe_kind": probe_kind,
                    "model_name": model_name,
                    "sample_set": sample_set,
                    "target_name": target_name,
                    "true_label": str(true_label),
                    "pred_label": str(pred_label),
                    "count": int(cm[i, j]),
                }
            )
    return row, recall_rows, cm_rows


def evaluate_pipeline(
    *,
    pipeline: Pipeline,
    x_eval: object,
    y_eval: Sequence[object],
    labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """函数功能：按固定类别顺序输出预测标签和概率矩阵。"""
    pred = pipeline.predict(x_eval).astype(str)
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, SGDClassifier):
        scores = np.asarray(pipeline.decision_function(x_eval), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(scores, -80.0, 80.0))
        raw_proba = exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-12, None)
    else:
        raw_proba = np.asarray(pipeline.predict_proba(x_eval), dtype=np.float64)
    class_to_idx = {str(label): idx for idx, label in enumerate(clf.classes_.astype(str))}
    proba = np.zeros((len(pred), len(labels)), dtype=np.float64)
    for out_idx, label in enumerate(labels):
        if str(label) in class_to_idx:
            proba[:, out_idx] = raw_proba[:, class_to_idx[str(label)]]
    row_sum = proba.sum(axis=1, keepdims=True)
    if not np.allclose(row_sum, 1.0):
        # 训练集应覆盖固定 label；这里保留防御式归一化，避免 held-out 类别导致概率异常。
        proba = proba / np.clip(row_sum, 1e-12, None)
    if not np.isfinite(proba).all():
        # 概率只用于 probe 诊断指标；异常时回退为均匀分布并保留 hard prediction 指标。
        proba = np.full((len(pred), len(labels)), 1.0 / max(len(labels), 1), dtype=np.float64)
    return pred, proba


def run_expert_feature_probes(
    *,
    sample_frames: Mapping[str, pd.DataFrame],
    feature_manifest_path: Path,
    feature_groups: Sequence[str],
    seed: int,
    max_iter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """函数功能：运行 cls/mean/concat/aux 等 feature group 对 oracle expert 的 suitability probe。"""
    result_rows: List[Dict[str, object]] = []
    recall_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    within_rows: List[Dict[str, object]] = []
    labels = list(MODEL_COLUMNS)
    y_train = sample_frames["pilot_train"]["oracle_model"].astype(str).to_numpy()
    for feature_group in feature_groups:
        print(f"[{display_time()}] expert suitability feature_group={feature_group}", flush=True)
        x_train = load_feature_group(
            feature_manifest_path=feature_manifest_path,
            sample_df=sample_frames["pilot_train"],
            sample_set="pilot_train",
            feature_group=feature_group,
        )
        pipeline = train_numeric_probe(x_train, y_train, seed=seed, max_iter=max_iter)
        del x_train
        for sample_set in EVAL_SAMPLE_SETS:
            x_eval = load_feature_group(
                feature_manifest_path=feature_manifest_path,
                sample_df=sample_frames[sample_set],
                sample_set=sample_set,
                feature_group=feature_group,
            )
            y_eval = sample_frames[sample_set]["oracle_model"].astype(str).to_numpy()
            pred, proba = evaluate_pipeline(pipeline=pipeline, x_eval=x_eval, y_eval=y_eval, labels=labels)
            row, per_class, cms = classification_metrics(
                model_name=feature_group,
                probe_kind="feature_probe",
                sample_set=sample_set,
                target_name="oracle_model",
                y_true=y_eval,
                y_pred=pred,
                proba=proba,
                labels=labels,
                include_top2=True,
            )
            row["feature_dim"] = int(x_eval.shape[1])
            result_rows.append(row)
            recall_rows.extend(per_class)
            cm_rows.extend(cms)
            eval_df = sample_frames[sample_set][["sample_key", "dataset_name", "oracle_model"]].copy()
            eval_df["pred"] = pred
            for dataset_name, group in eval_df.groupby("dataset_name", sort=True):
                if len(group) < 2:
                    continue
                within_rows.append(
                    {
                        "probe_kind": "feature_probe",
                        "model_name": feature_group,
                        "sample_set": sample_set,
                        "target_name": "oracle_model",
                        "dataset_name": str(dataset_name),
                        "sample_count": int(len(group)),
                        "oracle_expert_accuracy": float(accuracy_score(group["oracle_model"].astype(str), group["pred"].astype(str))),
                        "macro_f1": float(
                            f1_score(
                                group["oracle_model"].astype(str),
                                group["pred"].astype(str),
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                    }
                )
            del x_eval
    return pd.DataFrame(result_rows), pd.DataFrame(recall_rows), pd.DataFrame(cm_rows), pd.DataFrame(within_rows)


def run_structure_feature_probes(
    *,
    sample_frames: Mapping[str, pd.DataFrame],
    feature_manifest_path: Path,
    feature_groups: Sequence[str],
    seed: int,
    max_iter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """函数功能：运行 feature group 到 TSF/结构语义标签的分类 probe。"""
    result_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    for target_name in STRUCTURE_TARGETS:
        labels = sorted(sample_frames["pilot_train"][target_name].astype(str).unique().tolist())
        y_train = sample_frames["pilot_train"][target_name].astype(str).to_numpy()
        for feature_group in feature_groups:
            print(f"[{display_time()}] structure target={target_name} feature_group={feature_group}", flush=True)
            x_train = load_feature_group(
                feature_manifest_path=feature_manifest_path,
                sample_df=sample_frames["pilot_train"],
                sample_set="pilot_train",
                feature_group=feature_group,
            )
            pipeline = train_numeric_probe(x_train, y_train, seed=seed, max_iter=max_iter)
            del x_train
            for sample_set in EVAL_SAMPLE_SETS:
                x_eval = load_feature_group(
                    feature_manifest_path=feature_manifest_path,
                    sample_df=sample_frames[sample_set],
                    sample_set=sample_set,
                    feature_group=feature_group,
                )
                y_eval = sample_frames[sample_set][target_name].astype(str).to_numpy()
                pred, proba = evaluate_pipeline(pipeline=pipeline, x_eval=x_eval, y_eval=y_eval, labels=labels)
                row, _per_class, cms = classification_metrics(
                    model_name=feature_group,
                    probe_kind="structure_probe",
                    sample_set=sample_set,
                    target_name=target_name,
                    y_true=y_eval,
                    y_pred=pred,
                    proba=proba,
                    labels=labels,
                    include_top2=False,
                )
                row["feature_dim"] = int(x_eval.shape[1])
                result_rows.append(row)
                cm_rows.extend(cms)
                del x_eval
    return pd.DataFrame(result_rows), pd.DataFrame(cm_rows)


def train_frequency_prior(train_labels: Sequence[object]) -> Tuple[str, Dict[str, float]]:
    """函数功能：构建最简单的 train frequency prior baseline。"""
    counts = pd.Series(np.asarray(train_labels).astype(str)).value_counts(normalize=True)
    majority = str(counts.idxmax())
    return majority, {str(label): float(value) for label, value in counts.items()}


def frequency_prior_predict(sample_count: int, prior: Mapping[str, float], labels: Sequence[str], majority: str) -> Tuple[np.ndarray, np.ndarray]:
    """函数功能：输出 frequency prior 的固定预测与概率。"""
    proba = np.zeros((int(sample_count), len(labels)), dtype=np.float64)
    for idx, label in enumerate(labels):
        proba[:, idx] = float(prior.get(str(label), 0.0))
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = proba / np.clip(row_sum, 1e-12, None)
    pred = np.asarray([majority] * int(sample_count), dtype=str)
    return pred, proba


def run_shortcut_baselines(
    *,
    sample_frames: Mapping[str, pd.DataFrame],
    seed: int,
    max_iter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """函数功能：运行 dataset-only、TSF-only 和 train frequency prior oracle_model baseline。"""
    result_rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    within_rows: List[Dict[str, object]] = []
    labels = list(MODEL_COLUMNS)
    y_train = sample_frames["pilot_train"]["oracle_model"].astype(str).to_numpy()
    baselines: List[Tuple[str, object, object]] = []
    dataset_pipe = make_onehot_probe(["dataset_name"], seed=seed, max_iter=max_iter)
    tsf_pipe = make_onehot_probe(TSF_BASELINE_COLUMNS, seed=seed, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        dataset_pipe.fit(sample_frames["pilot_train"][["dataset_name"]].astype(str), y_train)
        tsf_pipe.fit(sample_frames["pilot_train"][list(TSF_BASELINE_COLUMNS)].astype(str), y_train)
    baselines.append(("dataset_name_only", dataset_pipe, ["dataset_name"]))
    baselines.append(("tsf_only", tsf_pipe, list(TSF_BASELINE_COLUMNS)))
    majority, prior = train_frequency_prior(y_train)
    for sample_set in EVAL_SAMPLE_SETS:
        y_eval = sample_frames[sample_set]["oracle_model"].astype(str).to_numpy()
        for baseline_name, pipeline, columns in baselines:
            x_eval = sample_frames[sample_set][columns].astype(str)
            pred, proba = evaluate_pipeline(pipeline=pipeline, x_eval=x_eval, y_eval=y_eval, labels=labels)
            row, _per_class, cms = classification_metrics(
                model_name=baseline_name,
                probe_kind="shortcut_baseline",
                sample_set=sample_set,
                target_name="oracle_model",
                y_true=y_eval,
                y_pred=pred,
                proba=proba,
                labels=labels,
                include_top2=True,
            )
            row["feature_dim"] = len(columns)
            result_rows.append(row)
            cm_rows.extend(cms)
            eval_df = sample_frames[sample_set][["dataset_name", "oracle_model"]].copy()
            eval_df["pred"] = pred
            for dataset_name, group in eval_df.groupby("dataset_name", sort=True):
                within_rows.append(
                    {
                        "probe_kind": "shortcut_baseline",
                        "model_name": baseline_name,
                        "sample_set": sample_set,
                        "target_name": "oracle_model",
                        "dataset_name": str(dataset_name),
                        "sample_count": int(len(group)),
                        "oracle_expert_accuracy": float(accuracy_score(group["oracle_model"].astype(str), group["pred"].astype(str))),
                        "macro_f1": float(
                            f1_score(
                                group["oracle_model"].astype(str),
                                group["pred"].astype(str),
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                    }
                )
        pred, proba = frequency_prior_predict(len(y_eval), prior, labels, majority)
        row, _per_class, cms = classification_metrics(
            model_name="train_frequency_prior",
            probe_kind="shortcut_baseline",
            sample_set=sample_set,
            target_name="oracle_model",
            y_true=y_eval,
            y_pred=pred,
            proba=proba,
            labels=labels,
            include_top2=True,
        )
        row["feature_dim"] = 0
        result_rows.append(row)
        cm_rows.extend(cms)
    return pd.DataFrame(result_rows), pd.DataFrame(cm_rows), pd.DataFrame(within_rows)


def compact_metric_table(df: pd.DataFrame, *, sample_set: str, target_name: str, top_n: int | None = None) -> pd.DataFrame:
    """函数功能：为中文 summary 生成按主指标排序的小表。"""
    subset = df[(df["sample_set"] == sample_set) & (df["target_name"] == target_name)].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(["accuracy", "macro_f1"], ascending=False)
    cols = ["model_name", "accuracy", "macro_f1", "balanced_accuracy", "top2_recall", "cross_entropy", "sample_count"]
    cols = [col for col in cols if col in subset.columns]
    if top_n is not None:
        subset = subset.head(int(top_n))
    return subset[cols].reset_index(drop=True)


def write_summary(
    *,
    output_path: Path,
    expert_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    shortcut_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：用中文回答 P2probe 验收要求中的六个诊断问题。"""
    selection_expert = expert_df[(expert_df["sample_set"] == "pilot_selection") & (expert_df["target_name"] == "oracle_model")]
    selection_shortcut = shortcut_df[(shortcut_df["sample_set"] == "pilot_selection") & (shortcut_df["target_name"] == "oracle_model")]
    best_visual = selection_expert[selection_expert["model_name"].isin(["cls_embedding", "mean_patch_embedding", "cls_mean_concat"])]
    best_visual_row = best_visual.sort_values(["accuracy", "macro_f1"], ascending=False).head(1)
    shortcut_top = selection_shortcut.sort_values(["accuracy", "macro_f1"], ascending=False).head(1)
    cls_row = selection_expert[selection_expert["model_name"] == "cls_embedding"].head(1)
    mean_row = selection_expert[selection_expert["model_name"] == "mean_patch_embedding"].head(1)
    aux_row = selection_expert[selection_expert["model_name"] == "revin_aux"].head(1)
    structure_selection = structure_df[structure_df["sample_set"] == "pilot_selection"].copy()
    visual_structure = structure_selection[structure_selection["model_name"].isin(["cls_embedding", "mean_patch_embedding", "cls_mean_concat"])]
    aux_structure = structure_selection[structure_selection["model_name"] == "revin_aux"]
    visual_structure_mean = float(visual_structure["macro_f1"].mean()) if not visual_structure.empty else np.nan
    aux_structure_mean = float(aux_structure["macro_f1"].mean()) if not aux_structure.empty else np.nan

    def fmt_metric(row: pd.DataFrame, name: str) -> str:
        if row.empty:
            return f"{name}=缺失"
        value = row.iloc[0]
        return f"{name} accuracy={float(value['accuracy']):.4f}, macroF1={float(value['macro_f1']):.4f}, top2={float(value['top2_recall']):.4f}"

    evidence_bias = "视觉表示有结构语义增量"
    if not best_visual_row.empty and not shortcut_top.empty:
        if float(best_visual_row.iloc[0]["accuracy"]) <= float(shortcut_top.iloc[0]["accuracy"]) + 0.01:
            evidence_bias = "主要风险仍是 dataset/TSF/expert shortcut，需要谨慎"
    lines = [
        "# Visual Router V2 Round 1 P2probe 摘要",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 输入与边界",
        "",
        f"- P0 sample set：`{metadata['sample_dir']}`",
        f"- P2a feature cache：`{metadata['feature_dir']}`",
        f"- 输出目录：`{metadata['output_dir']}`",
        "- 本 probe 只训练 sklearn 线性分类器，不训练 Visual Router routing head，不生成 ViT feature，不读取 prediction manifest。",
        "",
        "## Expert Suitability 主结果（pilot_selection）",
        "",
        frame_to_markdown(compact_metric_table(pd.concat([expert_df, shortcut_df], ignore_index=True), sample_set="pilot_selection", target_name="oracle_model")),
        "",
        "## Structure Semantics 主结果（pilot_selection，各 target 最优三项）",
        "",
    ]
    for target_name in STRUCTURE_TARGETS:
        lines.extend([f"### {target_name}", "", frame_to_markdown(compact_metric_table(structure_df, sample_set="pilot_selection", target_name=target_name, top_n=3)), ""])
    lines.extend(
        [
            "## 验收问题回答",
            "",
            f"1. visual embedding 是否能预测 oracle expert，是否优于 dataset/TSF shortcut baseline？{fmt_metric(best_visual_row, 'best_visual')}；{fmt_metric(shortcut_top, 'best_shortcut')}。结论以 selection 表为准。",
            f"2. mean_patch 是否比 CLS 含有更多 expert suitability 信息？{fmt_metric(cls_row, 'CLS')}；{fmt_metric(mean_row, 'mean_patch')}。",
            f"3. visual embedding 是否能恢复 TSF/结构语义标签？visual 结构 probe 平均 macroF1={visual_structure_mean:.4f}；需要逐 target 查看上表，尤其 cluster/group_name 用于 shortcut 风险参考。",
            f"4. revin_aux 的信息强度与 visual embedding 相比如何？{fmt_metric(aux_row, 'revin_aux')}；结构 probe 中 revin_aux 平均 macroF1={aux_structure_mean:.4f}。",
            f"5. 当前证据更支持“视觉表示有结构语义增量”，还是“主要是 dataset/TSF/expert shortcut”？当前自动判读：{evidence_bias}。如果 dataset/TSF-only 接近或超过 visual，需要在 P2d 前优先做 group split/held-out dataset 复验。",
            "6. 对 P2d visual+aux concat 和 Round 2 view/imageization 消融的启发：若 cls_plus_aux/mean_patch_plus_aux 相比单独 visual 与 aux 有稳定增益，P2d concat 值得推进；若 mean_patch 优于 CLS，Round 2 应优先保留 patch-token 聚合视角；若结构标签恢复主要集中在 dataset/group_name，Round 2 需要设计更强 held-out dataset/cell 消融来排除 shortcut。",
            "",
            "## 输出文件",
            "",
            "- `feature_probe_expert_suitability_results.csv`",
            "- `feature_probe_structure_results.csv`",
            "- `feature_probe_shortcut_baselines.csv`",
            "- `feature_probe_confusion_matrices.csv`",
            "- `feature_probe_within_dataset_summary.csv`",
            "- `feature_probe_metadata.json`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """函数功能：执行 P2probe 全流程并写出验收所需 artifact。"""
    args = parse_args()
    prepare_output_dir(args.output_dir, overwrite=bool(args.overwrite))
    feature_manifest_path = args.feature_dir / "round1_feature_manifest.csv"
    feature_metadata_path = args.feature_dir / "round1_feature_metadata.json"
    if not feature_manifest_path.exists():
        raise FileNotFoundError(f"找不到 P2a feature manifest：{feature_manifest_path}")
    if not feature_metadata_path.exists():
        raise FileNotFoundError(f"找不到 P2a feature metadata：{feature_metadata_path}")
    missing_required = sorted(set(REQUIRED_FEATURE_GROUPS).difference(args.feature_groups))
    if missing_required:
        raise ValueError(f"feature_groups 必须至少覆盖验收要求：missing={missing_required}")

    sample_frames = {
        sample_set: read_sample_csv(args.sample_dir, sample_set, max_samples=args.max_samples_per_set)
        for sample_set in SAMPLE_SETS
    }
    label_subset_dir = args.output_dir / "label_subset_cache"
    label_subset_dir.mkdir(parents=True, exist_ok=True)
    for sample_set, frame in sample_frames.items():
        # 小型 label subset 只包含 P0 sample keys 和 probe 标签，便于复核，不包含 future y 或专家预测数组。
        keep_cols = [
            "sample_set",
            "order_index",
            "sample_key",
            "config_name",
            "split",
            "dataset_name",
            "oracle_model",
            "error_gap",
            *STRUCTURE_TARGETS,
        ]
        frame[keep_cols].to_csv(label_subset_dir / f"{sample_set}_probe_labels.csv", index=False)

    expert_df, expert_recall_df, expert_cm_df, within_feature_df = run_expert_feature_probes(
        sample_frames=sample_frames,
        feature_manifest_path=feature_manifest_path,
        feature_groups=args.feature_groups,
        seed=int(args.seed),
        max_iter=int(args.max_iter),
    )
    structure_df, structure_cm_df = run_structure_feature_probes(
        sample_frames=sample_frames,
        feature_manifest_path=feature_manifest_path,
        feature_groups=[name for name in args.feature_groups if name in REQUIRED_FEATURE_GROUPS],
        seed=int(args.seed),
        max_iter=int(args.max_iter),
    )
    shortcut_df, shortcut_cm_df, within_shortcut_df = run_shortcut_baselines(
        sample_frames=sample_frames,
        seed=int(args.seed),
        max_iter=int(args.max_iter),
    )
    expert_out = expert_df.copy()
    expert_out.to_csv(args.output_dir / "feature_probe_expert_suitability_results.csv", index=False)
    structure_df.to_csv(args.output_dir / "feature_probe_structure_results.csv", index=False)
    shortcut_df.to_csv(args.output_dir / "feature_probe_shortcut_baselines.csv", index=False)
    confusion_df = pd.concat([expert_cm_df, structure_cm_df, shortcut_cm_df], ignore_index=True)
    confusion_df.to_csv(args.output_dir / "feature_probe_confusion_matrices.csv", index=False)
    within_df = pd.concat([within_feature_df, within_shortcut_df], ignore_index=True)
    within_df.to_csv(args.output_dir / "feature_probe_within_dataset_summary.csv", index=False)
    expert_recall_df.to_csv(args.output_dir / "feature_probe_per_expert_recall.csv", index=False)

    p2a_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "script": str(Path(__file__).resolve()),
        "script_version": "visual_router_v2_round1_p2probe_v1",
        "git_commit": git_commit_hash(),
        "sample_dir": str(args.sample_dir),
        "round0_dir": str(args.round0_dir),
        "feature_dir": str(args.feature_dir),
        "feature_manifest_path": str(feature_manifest_path),
        "oracle_labels_path": str(args.oracle_labels_path),
        "tsf_enrichment_path": str(args.tsf_enrichment_path),
        "output_dir": str(args.output_dir),
        "sample_counts": {name: int(len(frame)) for name, frame in sample_frames.items()},
        "max_samples_per_set": args.max_samples_per_set,
        "feature_groups": list(args.feature_groups),
        "required_feature_groups": list(REQUIRED_FEATURE_GROUPS),
        "structure_targets": list(STRUCTURE_TARGETS),
        "shortcut_baselines": ["dataset_name_only", "tsf_only", "train_frequency_prior"],
        "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
        "model_columns": list(MODEL_COLUMNS),
        "seed": int(args.seed),
        "max_iter": int(args.max_iter),
        "train_protocol": {
            "fit_sample_set": "pilot_train",
            "model_selection_sample_set": "pilot_selection",
            "extra_diagnostic_sample_set": "diagnostic_balanced",
            "uses_pilot_test": False,
            "scaler_fit_on": "pilot_train_only",
            "probe_model": "StandardScaler + class-balanced SGDClassifier(log_loss) for dense features; OneHotEncoder + class-balanced LogisticRegression for shortcut baselines",
        },
        "constraints": {
            "read_p2a_npz_only": True,
            "regenerate_vit_features": False,
            "save_pseudo_image_tensor": False,
            "read_prediction_manifest": False,
            "train_visual_router_head": False,
            "train_hard_or_soft_fusion": False,
            "modify_p2a_schema_or_builder": False,
        },
        "p2a_feature_metadata_status": p2a_metadata.get("status"),
        "p2a_feature_schema_version": p2a_metadata.get("feature_schema_version"),
    }
    write_json(args.output_dir / "feature_probe_metadata.json", metadata)
    write_summary(
        output_path=args.output_dir / "feature_probe_summary.md",
        expert_df=expert_df,
        structure_df=structure_df,
        shortcut_df=shortcut_df,
        metadata=metadata,
    )
    print(f"[{display_time()}] P2probe completed output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
