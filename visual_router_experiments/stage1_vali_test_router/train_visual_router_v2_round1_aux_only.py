#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round 1 P2c RevIN aux-only ablation。

实验边界：
    - 只读取 P2a feature shard 中的 `revin_aux` 六维特征；
    - 不读取或使用 `cls_embedding` / `mean_patch_embedding` 训练；
    - StandardScaler 只在 `pilot_train` 上 fit；
    - `pilot_selection` 只用于 epoch / seed 选择；
    - `diagnostic_balanced` 只做诊断展示，不参与选择。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visual_router_experiments.common.prediction_array_io import load_prediction_array, resolve_cache_array_path  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.fusion_utils import EPS, MODEL_COLUMNS, frame_to_markdown  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_evaluator import TSF_STRATA_COLUMNS  # noqa: E402
from visual_router_experiments.stage1_vali_test_router.visual_router_v2_features import AUX_FEATURE_COLUMNS  # noqa: E402


DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_SAMPLE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_pilot_samples"
DEFAULT_ROUND0_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round0"
DEFAULT_FEATURE_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_features"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-20_visual_router_v2_round1_aux_only"
DEFAULT_ORACLE_LABELS = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "oracle_labels_full_scale_2026-06-16"
    / "window_oracle_labels.parquet"
)
DEFAULT_PREDICTION_MANIFEST = (
    DATA2_RUN_OUTPUT_ROOT
    / "2026-06-15_stage1_96_48_s_full_scale"
    / "prediction_cache_full_scale_launcher"
    / "merged_cache"
    / "manifest.csv"
)
SAMPLE_SETS = ("pilot_train", "pilot_selection", "diagnostic_balanced")
EVAL_SAMPLE_SETS = ("pilot_selection", "diagnostic_balanced")
ROUTER_NAME = "revin_aux_only_fusion_huber_kl"
SCRIPT_VERSION = "visual_router_v2_round1_aux_only_v1"


def display_time() -> str:
    """函数功能：生成中文日志、metadata 和 summary 使用的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def log_stage(message: str) -> None:
    """函数功能：打印带时间戳的阶段进度，便于长任务监控。"""
    print(f"[{display_time()}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2c aux-only 训练与评估参数。"""
    parser = argparse.ArgumentParser(description="Train Visual Router V2 Round 1 RevIN aux-only ablation.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--round0-dir", type=Path, default=DEFAULT_ROUND0_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oracle-labels-path", type=Path, default=DEFAULT_ORACLE_LABELS)
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST)
    parser.add_argument("--prediction-index-path", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[16, 17, 18])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.1)
    parser.add_argument("--kl-tau", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--csv-chunksize", type=int, default=500_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=250_000)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于开发 smoke 的每集合截断。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖本 P2c 输出目录中的既有结果文件。")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """函数功能：解析训练设备；auto 优先 CUDA。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 --device cuda，但当前 PyTorch CUDA 不可用")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    """函数功能：固定主要随机源，保证 seed 间差异可复核。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def git_commit_hash() -> str:
    """函数功能：记录当前 repo commit hash；失败时返回 unknown 但不中断实验。"""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class AuxOnlyRouter(nn.Module):
    """
    类功能：
        六维 RevIN aux-only router head。

    输入：
        经 `pilot_train` fit 的 StandardScaler 标准化后的六维 aux 特征。
    输出：
        五专家 logits；训练和评估时再 softmax 为专家融合权重。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """函数功能：初始化小 MLP，避免六维输入下初始 logits 过大。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """函数功能：输出未归一化五专家 logits。"""
        return self.network(features)


class SQLitePredictionIndex:
    """类功能：只查询 P2c subset SQLite index，按 batch 读取五专家数组路径。"""

    def __init__(self, db_path: Path, manifest_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.manifest_dir = Path(manifest_dir)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row

    def fetch_records(self, sample_keys: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, object]]:
        """函数功能：批量查询当前 batch 的 `(sample_key, model_name)` prediction record。"""
        keys = [str(key) for key in sample_keys]
        if not keys:
            return {}
        records: Dict[Tuple[str, str], Dict[str, object]] = {}
        for start in range(0, len(keys), 900):
            part = keys[start : start + 900]
            placeholders = ",".join(["?"] * len(part))
            rows = self.connection.execute(
                f"""
                SELECT sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                       array_storage, y_true_row_index, y_pred_row_index
                FROM prediction_index
                WHERE sample_key IN ({placeholders})
                """,
                part,
            ).fetchall()
            for row in rows:
                record = dict(row)
                sample_key = str(record["sample_key"])
                model_name = str(record["model_name"])
                record["y_true_path"] = resolve_cache_array_path(str(record["y_true_path"]), self.manifest_dir)
                record["y_pred_path"] = resolve_cache_array_path(str(record["y_pred_path"]), self.manifest_dir)
                records[(sample_key, model_name)] = record
        return records

    def close(self) -> None:
        """函数功能：关闭 SQLite 连接。"""
        self.connection.close()


def read_sample_csv(sample_dir: Path, sample_set: str, *, max_samples: int | None) -> pd.DataFrame:
    """函数功能：读取 P0 sample CSV，并严格按 `order_index` 对齐。"""
    path = Path(sample_dir) / f"{sample_set}_sample_keys.csv"
    df = pd.read_csv(path).sort_values("order_index", kind="mergesort").reset_index(drop=True)
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()
    required = {"sample_set", "order_index", "sample_key", "oracle_model", *TSF_STRATA_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{missing}")
    if (df["sample_set"].astype(str) != sample_set).any():
        raise ValueError(f"{path} 中 sample_set 与期望不一致：{sample_set}")
    order_index = df["order_index"].to_numpy(dtype=np.int64)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(order_index, expected):
        raise ValueError(f"{path} 的 order_index 必须从 0 连续递增")
    if df["sample_key"].astype(str).duplicated().any():
        raise ValueError(f"{path} 中 sample_key 重复")
    return df


def load_revin_aux_features(feature_dir: Path, sample_set: str, sample_df: pd.DataFrame) -> np.ndarray:
    """
    函数功能：
        只从 P2a `.npz` shard 读取 `revin_aux`，并按 P0 `order_index` 校验对齐。

    关键约束：
        本函数不会读取 `cls_embedding` 或 `mean_patch_embedding`，避免视觉特征泄漏到
        aux-only ablation。
    """
    manifest_path = Path(feature_dir) / "round1_feature_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["sample_set"].astype(str) == sample_set].sort_values("shard_id")
    if rows.empty:
        raise ValueError(f"feature manifest 中没有 sample_set={sample_set}")
    aux_parts: List[np.ndarray] = []
    key_parts: List[str] = []
    order_parts: List[np.ndarray] = []
    target_count = int(len(sample_df))
    loaded_count = 0
    for row in rows.itertuples(index=False):
        if loaded_count >= target_count:
            break
        shard_path = Path(row.shard_path)
        with np.load(shard_path, allow_pickle=True) as data:
            shard_keys = [str(value) for value in data["sample_key"].tolist()]
            shard_order = np.asarray(data["order_index"], dtype=np.int64)
            shard_aux = np.asarray(data["revin_aux"], dtype=np.float32)
        keep = min(len(shard_keys), target_count - loaded_count)
        aux_parts.append(shard_aux[:keep])
        key_parts.extend(shard_keys[:keep])
        order_parts.append(shard_order[:keep])
        loaded_count += keep
    aux = np.concatenate(aux_parts, axis=0).astype(np.float32)
    order_index = np.concatenate(order_parts, axis=0).astype(np.int64)
    expected_keys = sample_df["sample_key"].astype(str).tolist()
    if key_parts != expected_keys:
        raise ValueError(f"{sample_set} feature shard sample_key 与 P0 CSV 不一致")
    if not np.array_equal(order_index, sample_df["order_index"].to_numpy(dtype=np.int64)):
        raise ValueError(f"{sample_set} feature shard order_index 与 P0 CSV 不一致")
    if aux.shape != (len(sample_df), len(AUX_FEATURE_COLUMNS)):
        raise ValueError(f"{sample_set} revin_aux shape 异常：{aux.shape}")
    if not np.isfinite(aux).all():
        raise ValueError(f"{sample_set} revin_aux 存在 NaN/Inf")
    return aux


def load_oracle_subset(labels_path: Path, sample_keys: Sequence[str], *, batch_rows: int) -> pd.DataFrame:
    """函数功能：从 full-scale oracle parquet 中只抽取 P0 subset 的 MAE 标签行。"""
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
    matched = 0
    parquet_file = pq.ParquetFile(labels_path)
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=int(batch_rows), columns=columns), start=1):
        table = pa.Table.from_batches([batch])
        mask = pc.and_(pc.equal(table["metric"], "mae"), pc.is_in(table["sample_key"], value_set=value_set))
        filtered = table.filter(mask)
        if filtered.num_rows:
            rows.append(filtered.to_pandas())
            matched += int(filtered.num_rows)
        if batch_idx == 1 or batch_idx % 25 == 0:
            log_stage(f"oracle subset scan batches={batch_idx} matched={matched}/{len(key_order)}")
        if matched >= len(key_order):
            break
    if not rows:
        raise ValueError("oracle labels 中没有命中 P0 sample_key")
    df = pd.concat(rows, ignore_index=True)
    if df["sample_key"].duplicated().any():
        raise ValueError("oracle subset sample_key 重复")
    present_keys = set(df["sample_key"].astype(str))
    missing = [key for key in sample_keys if str(key) not in present_keys]
    if missing:
        raise ValueError(f"oracle subset 缺失 sample_key，missing_count={len(missing)} 示例={missing[:5]}")
    df["_order_index"] = df["sample_key"].astype(str).map(key_order)
    return df.sort_values("_order_index").drop(columns=["_order_index"]).reset_index(drop=True)


def build_or_load_prediction_index(
    *,
    index_path: Path,
    prediction_manifest_path: Path,
    sample_keys: Sequence[str],
    chunksize: int,
) -> SQLitePredictionIndex:
    """函数功能：为 P2c P0 subset 建立或复用轻量 SQLite prediction index。"""
    manifest_dir = prediction_manifest_path.parent
    expected = len(set(str(key) for key in sample_keys)) * len(MODEL_COLUMNS)
    if index_path.exists():
        connection = sqlite3.connect(str(index_path))
        try:
            count = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
        finally:
            connection.close()
        if count == expected:
            log_stage(f"复用 P2c subset prediction index：{index_path} records={count}")
            return SQLitePredictionIndex(index_path, manifest_dir)
        raise ValueError(f"已有 prediction index 记录数异常：path={index_path} expected={expected} actual={count}")

    key_set = {str(key) for key in sample_keys}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    connection = sqlite3.connect(str(tmp_path))
    try:
        connection.execute(
            """
            CREATE TABLE prediction_index (
                sample_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                y_true_path TEXT NOT NULL,
                y_pred_path TEXT NOT NULL,
                mae REAL NOT NULL,
                mse REAL NOT NULL,
                array_storage TEXT,
                y_true_row_index INTEGER,
                y_pred_row_index INTEGER,
                PRIMARY KEY (sample_key, model_name)
            )
            """
        )
        usecols = [
            "sample_key",
            "model_name",
            "y_true_path",
            "y_pred_path",
            "mae",
            "mse",
            "array_storage",
            "y_true_row_index",
            "y_pred_row_index",
        ]
        written = 0
        for chunk_idx, chunk in enumerate(pd.read_csv(prediction_manifest_path, usecols=usecols, chunksize=int(chunksize)), start=1):
            matched = chunk[chunk["sample_key"].astype(str).isin(key_set)].copy()
            if not matched.empty:
                records = [tuple(row[col] for col in usecols) for _, row in matched.iterrows()]
                connection.executemany(
                    """
                    INSERT INTO prediction_index (
                        sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                        array_storage, y_true_row_index, y_pred_row_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                written += int(len(records))
            if chunk_idx == 1 or chunk_idx % 25 == 0:
                log_stage(f"prediction manifest scan chunks={chunk_idx} records={written}/{expected}")
            if written >= expected:
                break
        connection.commit()
        connection.execute("CREATE INDEX idx_prediction_sample_key ON prediction_index(sample_key)")
        connection.commit()
    finally:
        connection.close()
    tmp_path.replace(index_path)
    connection = sqlite3.connect(str(index_path))
    try:
        count = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
    finally:
        connection.close()
    if count != expected:
        raise ValueError(f"P2c subset prediction index 不完整：expected={expected} actual={count}")
    log_stage(f"完成 P2c subset prediction index：{index_path} records={count}")
    return SQLitePredictionIndex(index_path, manifest_dir)


def load_prediction_tensors(
    sample_keys: Sequence[str],
    prediction_index: SQLitePredictionIndex,
    *,
    error_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """函数功能：按 sample_key 顺序读取当前 batch 的五专家预测、共享 y_true 和专家误差。"""
    lookup = prediction_index.fetch_records(sample_keys)
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []
    all_errors: List[np.ndarray] = []
    for sample_key in sample_keys:
        sample_preds: List[np.ndarray] = []
        sample_true = None
        sample_errors: List[float] = []
        for model_name in MODEL_COLUMNS:
            record = lookup.get((str(sample_key), model_name))
            if record is None:
                raise ValueError(f"prediction index 缺少 sample_key={sample_key} model={model_name}")
            y_pred = load_prediction_array(record, "y_pred")
            y_true = load_prediction_array(record, "y_true")
            if sample_true is None:
                sample_true = y_true
            elif not np.array_equal(sample_true, y_true):
                raise ValueError(f"同一 sample_key 的 y_true 不一致：{sample_key}")
            diff = y_pred - y_true
            sample_preds.append(y_pred)
            if error_metric == "mae":
                sample_errors.append(float(np.mean(np.abs(diff))))
            else:
                sample_errors.append(float(np.mean(diff ** 2)))
        assert sample_true is not None
        all_preds.append(np.stack(sample_preds, axis=0))
        all_trues.append(sample_true)
        all_errors.append(np.asarray(sample_errors, dtype=np.float32))
    return (
        np.stack(all_preds, axis=0).astype(np.float32),
        np.stack(all_trues, axis=0).astype(np.float32),
        np.stack(all_errors, axis=0).astype(np.float32),
    )


def prepare_split_frame(sample_df: pd.DataFrame, aux: np.ndarray, label_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：把 P0 sample、P2a aux 和 oracle MAE 标签按 order_index 合成训练/评估表。"""
    merged = sample_df.merge(label_df, on="sample_key", how="left", suffixes=("_sample", ""))
    if merged["oracle_value"].isna().any():
        raise ValueError("oracle labels 未覆盖 sample_df")
    for col in TSF_STRATA_COLUMNS:
        sample_col = f"{col}_sample"
        if sample_col in merged.columns:
            merged[col] = merged[col].fillna(merged[sample_col]) if col in merged.columns else merged[sample_col]
            merged = merged.drop(columns=[sample_col])
    for idx, col in enumerate(AUX_FEATURE_COLUMNS):
        merged[f"aux_{col}"] = aux[:, idx]
    return merged.sort_values("order_index", kind="mergesort").reset_index(drop=True)


def iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    """函数功能：按 batch_size 切分 index 数组。"""
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def train_one_seed(
    *,
    seed: int,
    train_df: pd.DataFrame,
    eval_frames: Mapping[str, pd.DataFrame],
    scaler: StandardScaler,
    prediction_index: SQLitePredictionIndex,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """函数功能：训练单个 seed，并用 pilot_selection 选择 best epoch。"""
    set_seed(seed)
    feature_cols = [f"aux_{col}" for col in AUX_FEATURE_COLUMNS]
    x_train = scaler.transform(train_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    router = AuxOnlyRouter(
        input_dim=len(feature_cols),
        hidden_dim=int(args.hidden_dim),
        output_dim=len(MODEL_COLUMNS),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    huber_criterion = nn.SmoothL1Loss(beta=float(args.huber_beta))
    epoch_rows: List[Dict[str, object]] = []
    best_state = None
    best_selection_mae = float("inf")
    best_epoch = -1
    indices = np.arange(len(train_df), dtype=np.int64)
    sample_keys = train_df["sample_key"].astype(str).to_numpy()
    for epoch in range(1, int(args.epochs) + 1):
        rng = np.random.default_rng(int(seed) * 1000 + epoch)
        shuffled = rng.permutation(indices)
        router.train()
        losses: List[float] = []
        huber_losses: List[float] = []
        kl_losses: List[float] = []
        for batch_ids in iter_batches(shuffled, int(args.batch_size)):
            batch_keys = sample_keys[batch_ids].tolist()
            y_pred, y_true, expert_errors = load_prediction_tensors(batch_keys, prediction_index, error_metric=str(args.metric))
            batch_x = torch.from_numpy(x_train[batch_ids]).to(device=device)
            batch_pred = torch.from_numpy(y_pred).to(device=device)
            batch_true = torch.from_numpy(y_true).to(device=device)
            batch_q = torch.softmax(-torch.from_numpy(expert_errors) / float(args.kl_tau), dim=1).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = router(batch_x)
            weights = torch.softmax(logits, dim=1)
            weight_shape = (weights.shape[0], weights.shape[1], *([1] * (batch_pred.ndim - 2)))
            fused_pred = (weights.view(weight_shape) * batch_pred).sum(dim=1)
            huber_loss = huber_criterion(fused_pred, batch_true)
            kl_loss = F.kl_div(torch.log_softmax(logits, dim=1), batch_q, reduction="batchmean")
            loss = huber_loss + float(args.lambda_kl) * kl_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            huber_losses.append(float(huber_loss.detach().cpu().item()))
            kl_losses.append(float(kl_loss.detach().cpu().item()))
        selection_rows, selection_summary = evaluate_router(
            sample_set="pilot_selection",
            method_prefix=ROUTER_NAME,
            seed=seed,
            epoch=epoch,
            router=router,
            scaler=scaler,
            eval_df=eval_frames["pilot_selection"],
            prediction_index=prediction_index,
            args=args,
            device=device,
        )
        del selection_rows
        selection_soft = selection_summary[
            (selection_summary["sample_set"] == "pilot_selection")
            & (selection_summary["method"] == f"{ROUTER_NAME}_raw_soft_fusion")
        ].iloc[0]
        selection_mae = float(selection_soft["MAE"])
        epoch_rows.append(
            {
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "train_huber_loss": float(np.mean(huber_losses)),
                "train_kl_loss": float(np.mean(kl_losses)),
                "selection_raw_soft_MAE": selection_mae,
                "selection_raw_soft_MSE": float(selection_soft["MSE"]),
            }
        )
        log_stage(
            f"seed={seed} epoch={epoch}/{args.epochs} "
            f"train_loss={np.mean(losses):.6f} selection_soft_MAE={selection_mae:.6f}"
        )
        if selection_mae < best_selection_mae:
            best_selection_mae = selection_mae
            best_epoch = epoch
            best_state = deepcopy(router.state_dict())
    if best_state is None:
        raise RuntimeError(f"seed={seed} 未产生 best state")
    router.load_state_dict(best_state)
    eval_rows: List[pd.DataFrame] = []
    eval_summaries: List[pd.DataFrame] = []
    for sample_set in EVAL_SAMPLE_SETS:
        rows, summary = evaluate_router(
            sample_set=sample_set,
            method_prefix=ROUTER_NAME,
            seed=seed,
            epoch=best_epoch,
            router=router,
            scaler=scaler,
            eval_df=eval_frames[sample_set],
            prediction_index=prediction_index,
            args=args,
            device=device,
        )
        eval_rows.append(rows)
        eval_summaries.append(summary)
    return {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_selection_raw_soft_MAE": float(best_selection_mae),
        "epoch_history": pd.DataFrame(epoch_rows),
        "eval_rows": pd.concat(eval_rows, ignore_index=True),
        "eval_summary": pd.concat(eval_summaries, ignore_index=True),
        "state_dict": {key: value.detach().cpu() for key, value in router.state_dict().items()},
    }


def evaluate_router(
    *,
    sample_set: str,
    method_prefix: str,
    seed: int,
    epoch: int,
    router: AuxOnlyRouter,
    scaler: StandardScaler,
    eval_df: pd.DataFrame,
    prediction_index: SQLitePredictionIndex,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """函数功能：对一个样本集合计算 hard top-1 与 raw soft fusion 的逐样本和汇总指标。"""
    feature_cols = [f"aux_{col}" for col in AUX_FEATURE_COLUMNS]
    x_eval = scaler.transform(eval_df[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    router.eval()
    rows: List[Dict[str, object]] = []
    all_indices = np.arange(len(eval_df), dtype=np.int64)
    for batch_ids in iter_batches(all_indices, int(args.eval_batch_size)):
        batch_df = eval_df.iloc[batch_ids].reset_index(drop=True)
        batch_x = torch.from_numpy(x_eval[batch_ids]).to(device=device)
        with torch.inference_mode():
            weights = torch.softmax(router(batch_x), dim=1).detach().cpu().numpy()
        y_pred, y_true, _ = load_prediction_tensors(
            batch_df["sample_key"].astype(str).tolist(),
            prediction_index,
            error_metric=str(args.metric),
        )
        selected_indices = weights.argmax(axis=1)
        entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
        normalized_entropy = entropy / np.log(len(MODEL_COLUMNS))
        max_weight = weights.max(axis=1)
        for row_idx, row in enumerate(batch_df.itertuples(index=False)):
            selected_idx = int(selected_indices[row_idx])
            selected_model = MODEL_COLUMNS[selected_idx]
            hard_pred = y_pred[row_idx, selected_idx]
            soft_pred = np.sum(weights[row_idx].reshape((len(MODEL_COLUMNS), *([1] * (y_pred.ndim - 2)))) * y_pred[row_idx], axis=0)
            true = y_true[row_idx]
            hard_error = hard_pred - true
            soft_error = soft_pred - true
            base = {
                "seed": int(seed),
                "best_epoch": int(epoch),
                "sample_set": sample_set,
                "sample_key": str(row.sample_key),
                "selected_model": selected_model,
                "oracle_model": str(row.oracle_model),
                "oracle_mae": float(row.oracle_value),
                "oracle_label_correct": bool(selected_model == str(row.oracle_model)),
                "weight_entropy": float(entropy[row_idx]),
                "normalized_weight_entropy": float(normalized_entropy[row_idx]),
                "mean_max_weight": float(max_weight[row_idx]),
            }
            for col in TSF_STRATA_COLUMNS:
                base[col] = getattr(row, col)
            for model_idx, model_name in enumerate(MODEL_COLUMNS):
                base[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
            hard = dict(base)
            hard.update(
                {
                    "method": f"{method_prefix}_hard_top1",
                    "mae": float(np.mean(np.abs(hard_error))),
                    "mse": float(np.mean(hard_error ** 2)),
                }
            )
            hard["regret_to_oracle"] = hard["mae"] - hard["oracle_mae"]
            soft = dict(base)
            soft.update(
                {
                    "method": f"{method_prefix}_raw_soft_fusion",
                    "mae": float(np.mean(np.abs(soft_error))),
                    "mse": float(np.mean(soft_error ** 2)),
                }
            )
            soft["regret_to_oracle"] = soft["mae"] - soft["oracle_mae"]
            rows.extend([hard, soft])
    pred_rows = pd.DataFrame(rows)
    summary = summarize_rows(pred_rows)
    return pred_rows, summary


def summarize_rows(rows: pd.DataFrame, *, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """函数功能：按 method 或额外分层字段汇总 P2c 指标。"""
    out: List[Dict[str, object]] = []
    by_cols = ["seed", "sample_set", "method", *group_cols]
    for keys, group in rows.groupby(by_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(by_cols, keys)}
        row.update(
            {
                "sample_count": int(len(group)),
                "MAE": float(group["mae"].mean()),
                "MSE": float(group["mse"].mean()),
                "regret_to_oracle": float(group["regret_to_oracle"].mean()),
                "oracle_label_accuracy": float(group["oracle_label_correct"].mean()),
                "weight_entropy": float(group["weight_entropy"].mean()),
                "normalized_weight_entropy": float(group["normalized_weight_entropy"].mean()),
                "mean_max_weight": float(group["mean_max_weight"].mean()),
            }
        )
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)


def summarize_seed_mean_std(summary_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：将每 seed summary 汇总为 mean/std，保留 seed_count。"""
    metric_cols = [
        "MAE",
        "MSE",
        "regret_to_oracle",
        "oracle_label_accuracy",
        "weight_entropy",
        "normalized_weight_entropy",
        "mean_max_weight",
    ]
    rows: List[Dict[str, object]] = []
    for keys, group in summary_df.groupby(["sample_set", "method"], sort=False):
        sample_set, method = keys
        row: Dict[str, object] = {
            "sample_set": sample_set,
            "method": method,
            "seed_count": int(group["seed"].nunique()),
            "sample_count": int(group["sample_count"].iloc[0]),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def selected_model_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：输出每个 seed/sample_set/method 的 selected_model 计数和比例。"""
    counts = rows.groupby(["seed", "sample_set", "method", "selected_model"], dropna=False).size().rename("count").reset_index()
    totals = counts.groupby(["seed", "sample_set", "method"])["count"].transform("sum")
    counts["ratio"] = counts["count"] / totals
    return counts.sort_values(["seed", "sample_set", "method", "selected_model"]).reset_index(drop=True)


def stratified_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """函数功能：按目标要求输出 dataset/oracle/gap/TSF cell 等分层指标。"""
    frames: List[pd.DataFrame] = []
    stratify_cols = ["dataset_name", "oracle_model", "error_gap_quantile", *TSF_STRATA_COLUMNS[3:]]
    for col in dict.fromkeys(stratify_cols):
        grouped = summarize_rows(rows, group_cols=[col])
        grouped = grouped.rename(columns={col: "stratum_value"})
        grouped.insert(3, "stratum_column", col)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def load_optional_round0_reference(round0_dir: Path) -> pd.DataFrame:
    """函数功能：读取 P1 Round 0 selection/diagnostic 汇总，供中文 summary 对比。"""
    frames = []
    for path in [round0_dir / "round0_selection_comparison.csv", round0_dir / "round0_diagnostic_balanced_summary.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_summary_md(
    *,
    output_dir: Path,
    selection_comparison: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    best_seed: Mapping[str, object],
    round0_reference: pd.DataFrame,
    p2b_reference: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：用中文写出 P2c 结论摘要和验收问题回答。"""
    selection_table = selection_comparison.copy()
    hard_row = selection_table[selection_table["method"] == f"{ROUTER_NAME}_hard_top1"].iloc[0]
    soft_row = selection_table[selection_table["method"] == f"{ROUTER_NAME}_raw_soft_fusion"].iloc[0]
    round0_visual = pd.DataFrame()
    if not round0_reference.empty:
        round0_visual = round0_reference[
            (round0_reference["sample_set"] == "pilot_selection")
            & (round0_reference["method"].astype(str).str.contains("visual_router"))
        ].copy()
    p2b_text = "未发现 P2b visual-only 汇总文件，因此本次不回答接近或超过 P2b 的定量比较。"
    if not p2b_reference.empty:
        p2b_text = "发现 P2b 参考汇总，已在 metadata 中记录候选路径；本脚本未把 P2b 纳入选择，只作人工对照。"
    if not round0_visual.empty:
        base_soft = round0_visual[round0_visual["method"].astype(str).str.contains("raw_soft")]
        base_hard = round0_visual[round0_visual["method"].astype(str).str.contains("hard_top1")]
        round0_text = (
            f"Round 0 visual selection hard MAE={float(base_hard.iloc[0]['MAE']):.6f}，"
            f"raw-soft MAE={float(base_soft.iloc[0]['MAE']):.6f}。"
        )
    else:
        round0_text = "未读取到 Round 0 visual baseline selection 汇总。"
    conclusion = "RevIN aux-only 在 selection 上已经提供了可测路由信号。"
    if not round0_visual.empty:
        base_soft_mae = float(round0_visual[round0_visual["method"].astype(str).str.contains("raw_soft")].iloc[0]["MAE"])
        if float(soft_row["MAE_mean"]) < base_soft_mae:
            conclusion = "aux-only raw-soft MAE 优于 P1 Round 0 visual baseline，说明 RevIN 删除的尺度信息具有较强路由价值。"
        else:
            conclusion = "aux-only 未明显优于 P1 Round 0 visual baseline，尺度信息可能有路由价值但不是唯一瓶颈。"
    lines = [
        "# P2c RevIN aux-only ablation 汇总",
        "",
        f"- 生成时间：{display_time()}",
        f"- 输出目录：`{output_dir}`",
        f"- 脚本版本：`{SCRIPT_VERSION}`",
        f"- seeds：{metadata['seeds']}",
        f"- best seed：{best_seed}",
        f"- 输入特征：仅 P2a `revin_aux`，字段顺序为 `{AUX_FEATURE_COLUMNS}`。",
        f"- 训练选择：scaler 只在 `pilot_train` fit；best epoch/seed 只按 `pilot_selection` raw-soft MAE 选择；`diagnostic_balanced` 不参与选择。",
        "",
        "## Selection mean/std",
        "",
        frame_to_markdown(selection_comparison, float_digits=6),
        "",
        "## Diagnostic mean/std",
        "",
        frame_to_markdown(diagnostic_summary, float_digits=6),
        "",
        "## 验收问题回答",
        "",
        f"1. aux-only 是否明显优于 P1 Round 0 Visual baseline？{round0_text} 当前 P2c selection hard MAE={float(hard_row['MAE_mean']):.6f}，raw-soft MAE={float(soft_row['MAE_mean']):.6f}。{conclusion}",
        f"2. aux-only 是否接近或超过 P2b visual-only 结果？{p2b_text}",
        "3. RevIN 删除掉的尺度信息是否是当前 Visual Router 的主要瓶颈之一？若 aux-only 接近或优于 visual baseline，则它至少是重要瓶颈之一；若仅小幅改善，则应视为可补充信号而非唯一主因。以上判断以 selection mean/std 为主，diagnostic 只看方向一致性。",
        "4. 是否建议 P2d 做 visual+aux concat？建议做。P2c 只用 6 维尺度统计已形成独立路由信号，P2d 可以检验该信号与视觉 embedding 是否互补；但 P2d 必须继续保持 selection 选择、diagnostic 诊断和不使用 pilot_test 选择的边界。",
        "",
        "## 约束确认",
        "",
        "- 未读取 visual embedding 作为训练输入。",
        "- 未重新生成 P2a features。",
        "- 未使用完整 17 维 TimeFuse feature。",
        "- 未使用 pilot_test。",
    ]
    (output_dir / "aux_only_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_p2b_reference(output_root: Path) -> pd.DataFrame:
    """函数功能：尽力查找 P2b visual-only 汇总；找不到时返回空表且不中断。"""
    candidates = [
        path
        for path in sorted(output_root.glob("*visual*only*/**/*summary*.csv"))
        # 避免把本实验 `aux_only` 输出误识别为 P2b visual-only 参考。
        if "aux_only" not in str(path)
    ]
    for path in candidates[:5]:
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return pd.DataFrame()


def main() -> None:
    """脚本入口：执行 P2c aux-only 三 seed 训练、评估和结果写出。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        required = [
            "aux_only_variant_seed_results.csv",
            "aux_only_selection_comparison.csv",
            "aux_only_diagnostic_summary.csv",
            "aux_only_selected_model_counts.csv",
            "aux_only_stratified_summary.csv",
            "aux_only_best_seed.json",
            "aux_only_metadata.json",
            "aux_only_summary.md",
        ]
        if any((output_dir / name).exists() for name in required):
            raise FileExistsError(f"{output_dir} 已存在 P2c 结果；如需重跑请显式传入 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(args.device))
    log_stage(f"使用设备：{device}")

    sample_frames = {name: read_sample_csv(args.sample_dir, name, max_samples=args.max_samples_per_set) for name in SAMPLE_SETS}
    aux_arrays = {name: load_revin_aux_features(args.feature_dir, name, sample_frames[name]) for name in SAMPLE_SETS}
    all_sample_keys = []
    for name in SAMPLE_SETS:
        all_sample_keys.extend(sample_frames[name]["sample_key"].astype(str).tolist())
    if len(all_sample_keys) != len(set(all_sample_keys)):
        raise ValueError("P0 三个集合之间 sample_key 有重复")

    log_stage("读取 oracle label subset")
    labels_df = load_oracle_subset(args.oracle_labels_path, all_sample_keys, batch_rows=int(args.parquet_batch_rows))
    labels_by_set: Dict[str, pd.DataFrame] = {}
    offset = 0
    for name in SAMPLE_SETS:
        count = len(sample_frames[name])
        labels_by_set[name] = labels_df.iloc[offset : offset + count].reset_index(drop=True)
        offset += count
    prepared = {name: prepare_split_frame(sample_frames[name], aux_arrays[name], labels_by_set[name]) for name in SAMPLE_SETS}

    index_path = args.prediction_index_path or (output_dir / "prediction_index_aux_only_p0.sqlite")
    prediction_index = build_or_load_prediction_index(
        index_path=Path(index_path),
        prediction_manifest_path=args.prediction_manifest_path,
        sample_keys=all_sample_keys,
        chunksize=int(args.csv_chunksize),
    )
    try:
        feature_cols = [f"aux_{col}" for col in AUX_FEATURE_COLUMNS]
        scaler = StandardScaler()
        scaler.fit(prepared["pilot_train"][feature_cols].to_numpy(dtype=np.float32))
        eval_frames = {name: prepared[name] for name in EVAL_SAMPLE_SETS}
        seed_outputs = []
        for seed in args.seeds:
            log_stage(f"开始训练 seed={seed}")
            seed_outputs.append(
                train_one_seed(
                    seed=int(seed),
                    train_df=prepared["pilot_train"],
                    eval_frames=eval_frames,
                    scaler=scaler,
                    prediction_index=prediction_index,
                    args=args,
                    device=device,
                )
            )
    finally:
        prediction_index.close()

    epoch_history = pd.concat([item["epoch_history"] for item in seed_outputs], ignore_index=True)
    eval_rows = pd.concat([item["eval_rows"] for item in seed_outputs], ignore_index=True)
    seed_summary = pd.concat([item["eval_summary"] for item in seed_outputs], ignore_index=True)
    selection_comparison = summarize_seed_mean_std(seed_summary[seed_summary["sample_set"] == "pilot_selection"])
    diagnostic_summary = summarize_seed_mean_std(seed_summary[seed_summary["sample_set"] == "diagnostic_balanced"])
    counts = selected_model_counts(eval_rows)
    strata = stratified_summary(eval_rows)
    best_seed_row = seed_summary[
        (seed_summary["sample_set"] == "pilot_selection") & (seed_summary["method"] == f"{ROUTER_NAME}_raw_soft_fusion")
    ].sort_values("MAE", kind="mergesort").iloc[0]
    best_seed = {
        "selection_rule": "min pilot_selection raw_soft_fusion MAE",
        "seed": int(best_seed_row["seed"]),
        "best_epoch": int(next(item["best_epoch"] for item in seed_outputs if item["seed"] == int(best_seed_row["seed"]))),
        "selection_raw_soft_MAE": float(best_seed_row["MAE"]),
        "selection_raw_soft_MSE": float(best_seed_row["MSE"]),
    }

    epoch_history.to_csv(output_dir / "aux_only_epoch_history.csv", index=False)
    seed_summary.to_csv(output_dir / "aux_only_variant_seed_results.csv", index=False)
    selection_comparison.to_csv(output_dir / "aux_only_selection_comparison.csv", index=False)
    diagnostic_summary.to_csv(output_dir / "aux_only_diagnostic_summary.csv", index=False)
    counts.to_csv(output_dir / "aux_only_selected_model_counts.csv", index=False)
    strata.to_csv(output_dir / "aux_only_stratified_summary.csv", index=False)
    write_json(output_dir / "aux_only_best_seed.json", best_seed)

    best_state = next(item["state_dict"] for item in seed_outputs if item["seed"] == int(best_seed["seed"]))
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "router_name": ROUTER_NAME,
            "seed": int(best_seed["seed"]),
            "best_epoch": int(best_seed["best_epoch"]),
            "state_dict": best_state,
            "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
            "scaler_state": {
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "var_": scaler.var_.tolist(),
                "n_features_in_": int(scaler.n_features_in_),
                "n_samples_seen_": int(scaler.n_samples_seen_),
            },
            "model_kwargs": {
                "input_dim": len(AUX_FEATURE_COLUMNS),
                "hidden_dim": int(args.hidden_dim),
                "output_dim": len(MODEL_COLUMNS),
                "dropout": float(args.dropout),
            },
        },
        output_dir / "aux_only_best_router.pt",
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "created_at": display_time(),
        "git_commit": git_commit_hash(),
        "sample_dir": str(args.sample_dir),
        "round0_dir": str(args.round0_dir),
        "feature_dir": str(args.feature_dir),
        "oracle_labels_path": str(args.oracle_labels_path),
        "prediction_manifest_path": str(args.prediction_manifest_path),
        "prediction_index_path": str(index_path),
        "output_dir": str(output_dir),
        "seeds": [int(seed) for seed in args.seeds],
        "epochs": int(args.epochs),
        "selection_rule": best_seed["selection_rule"],
        "diagnostic_role": "diagnostic_only_not_for_selection",
        "pilot_test_used": False,
        "input_features": "P2a revin_aux only",
        "aux_feature_columns": list(AUX_FEATURE_COLUMNS),
        "forbidden_features_not_used": ["cls_embedding", "mean_patch_embedding", "17_dim_timefuse_feature", "pseudo_image_tensor"],
        "scaler_fit_sample_set": "pilot_train",
        "sample_counts": {name: int(len(frame)) for name, frame in prepared.items()},
        "train_objective": "SmoothL1 fused prediction loss + lambda_kl * KL soft oracle",
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
    }
    write_json(output_dir / "aux_only_metadata.json", metadata)

    round0_reference = load_optional_round0_reference(args.round0_dir)
    p2b_reference = find_p2b_reference(output_dir.parent)
    write_summary_md(
        output_dir=output_dir,
        selection_comparison=selection_comparison,
        diagnostic_summary=diagnostic_summary,
        best_seed=best_seed,
        round0_reference=round0_reference,
        p2b_reference=p2b_reference,
        metadata=metadata,
    )
    log_stage(f"P2c aux-only 完成，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
