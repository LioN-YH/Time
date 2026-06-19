#!/usr/bin/env python3
"""
文件功能：
    以 streaming / shard-aware 方式训练和评估 Stage 1 Online Visual Router。

核心约束：
    - 顶层路线固定为 `x -> pseudo image -> frozen ViT -> router`；
    - 伪图像 tensor 和 ViT embedding 只在当前 batch 的运行时存在，不写 `.npy`，
      不生成长期 embedding cache；
    - `StandardScaler` 只在 vali embedding 流上 `partial_fit`，test 只做 forward；
    - `fusion_huber_kl` 训练只在 vali split 读取专家 prediction cache 作为监督，
      不把专家误差、oracle label 或未来 y 作为 router 输入；
    - 输出文件名兼容 `evaluate_soft_fusion_calibration.py`，特别是
      `visual_router_predictions.csv` 和标准 `visual_router_metadata.json`。

内存优化（2026-06-16）：
    - 不再预加载全量 prediction_lookup dict（会导致 OOM）；
    - 改为 SQLite 磁盘索引 + batch 级查询，只让当前训练 batch 的
      prediction record 进入 Python 内存。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from transformers import ViTModel


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config.training import TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from time_router.evaluation import EvaluationInputAdapter  # noqa: E402
from time_router.protocols import ExpertBatch  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402
from visual_router_experiments.common.prediction_array_io import resolve_cache_array_path  # noqa: E402
from visual_router_experiments.common.vit_embedding_utils import (  # noqa: E402
    EMBEDDING_VERSION,
    batch_required_pairs,
    build_required_index,
    make_default_period_candidates,
    make_pseudo_images,
    parse_period_candidate_arg,
    pool_vit_outputs,
    resolve_dtype,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_PREDICTION_MANIFEST_PATH,
    EPS,
    MODEL_COLUMNS,
    VisualMLPRouter,
    add_soft_fusion_metrics,
    compare_with_baselines,
    frame_to_markdown,
    load_labels,
    load_prediction_tensors_for_samples,
    make_class_weight,
    summarize_hard_predictions,
    summarize_selected_model_counts,
    summarize_soft_fusion,
    validate_training_args,
)
from visual_router_experiments.stage1_vali_test_router.train_visual_router_online import (  # noqa: E402
    DEFAULT_CONFIG,
    _timer_start,
    _timer_stop,
    build_train_args,
    display_time,
    load_data_config,
    make_embedding_manifest_row,
    mode_from_split,
    now_token,
    resolve_device,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import (  # noqa: E402
    load_prediction_array,
)


STREAMING_ONLINE_ROUTER_VERSION = "visual_router_mlp_v3_fusion_huber_kl_online_vit_streaming"


def parse_args() -> argparse.Namespace:
    """函数功能：解析 streaming online router 参数。"""
    parser = argparse.ArgumentParser(description="Train Stage 1 Online Visual Router with streaming ViT batches.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="window oracle labels CSV。")
    parser.add_argument("--prediction-manifest-path", type=Path, default=DEFAULT_PREDICTION_MANIFEST_PATH, help="五专家 prediction cache manifest CSV。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config；仅用于读取历史窗口 x。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="oracle label 和辅助误差分布使用的指标。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认生成 streaming run 目录。")
    parser.add_argument("--router-mode", choices=["classification", "fusion_huber_kl"], default="fusion_huber_kl", help="router 训练目标。")
    parser.add_argument("--huber-beta", type=float, default=0.1, help="fusion_huber_kl SmoothL1 beta。")
    parser.add_argument("--kl-tau", type=float, default=0.1, help="soft oracle temperature。")
    parser.add_argument("--lambda-kl", type=float, default=0.01, help="KL 辅助损失权重。")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden dimension。")
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout。")
    parser.add_argument("--epochs", type=int, default=1, help="streaming 训练 epoch；每个 epoch 会重新运行 vali ViT 前向。")
    parser.add_argument("--batch-size", type=int, default=32, help="router 参数更新 batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay。")
    parser.add_argument("--seed", type=int, default=16, help="随机种子。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ViT 和 router 运行设备。")
    parser.add_argument("--skip-soft-fusion", action="store_true", help="只写 hard top-1，不计算 raw soft fusion。")
    parser.add_argument("--encoder-name", default="google/vit-base-patch16-224", help="Hugging Face ViT encoder 名称或本地路径。")
    parser.add_argument("--variant", choices=["variant_a_3view", "variant_b_top3fold"], default="variant_a_3view", help="伪图像 variant。")
    parser.add_argument("--pooling", choices=["cls", "mean_patch", "pooler"], default="cls", help="ViT 输出聚合方式。")
    parser.add_argument("--normalization-preset", default="hf_vit_0_5", help="encoder 前 normalization 口径。")
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="在线 ViT 前向 batch size。")
    parser.add_argument("--image-size", type=int, default=224, help="伪图像尺寸。")
    parser.add_argument("--norm-mode", choices=["quito", "revin", "revin_aux"], default="revin_aux", help="历史窗口 normalization 口径。")
    parser.add_argument("--pixel-mode", choices=["vision"], default="vision", help="pixel 映射口径。")
    parser.add_argument("--clip", type=float, default=5.0, help="视觉 pixel 映射前截断阈值。")
    parser.add_argument("--period-selection", choices=["fixed_candidates", "dynamic_fft_topk"], default="fixed_candidates", help="full-scale 默认固定候选周期，减少同步。")
    parser.add_argument("--period-candidates", default=None, help="逗号分隔候选周期；只在 fixed_candidates 下使用。")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16"], default="auto", help="encoder 前向 dtype；CPU 强制 fp32。")
    parser.add_argument("--local-files-only", action="store_true", help="只使用本地 Hugging Face cache，不联网下载。")
    parser.add_argument("--vit-data-parallel", action="store_true", help="CUDA 多卡可用时用 DataParallel 并行冻结 ViT 前向。")
    parser.add_argument("--stream-shard-index", type=int, default=0, help="按 sample_key 稳定切分后的当前 streaming shard。")
    parser.add_argument("--stream-shard-count", type=int, default=1, help="streaming shard 总数；dry-run 可用来验证多 shard 口径。")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="dry-run 限制每个 split 最多样本数；None 表示不限制。")
    parser.add_argument("--chunk-read-rows", type=int, default=200000, help="预留的大 CSV chunk 读取行数，metadata 记录用。")
    parser.add_argument("--status-update-interval", type=int, default=50, help="每处理多少个 embedding batch 更新一次 status.json。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束打印多少行预测预览。")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="从已有 router checkpoint 继续训练或 eval-only。")
    parser.add_argument("--train-only", action="store_true", help="只训练并保存 checkpoint，不执行 test streaming 预测。")
    parser.add_argument(
        "--verify-evaluation-adapter",
        action="store_true",
        help="默认关闭；仅在 test evaluation batch 内用 EvaluationInputAdapter 旁路复算并校验 hard/raw soft 指标。",
    )
    parser.add_argument(
        "--verify-training-expert-batch",
        action="store_true",
        help="默认关闭；仅在 fusion_huber_kl training batch 内用 ExpertBatch 旁路复算 expert_errors。",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """函数功能：固定主要随机源，便于 dry-run 复核。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_status(output_dir: Path, status: Mapping[str, object]) -> None:
    """函数功能：写出长任务 status.json，供断点检查和外部监控读取。"""
    payload = dict(status)
    payload["updated_at"] = display_time()
    payload["output_dir"] = str(output_dir)
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_path_string(path: Path) -> str:
    """函数功能：把输入路径规整成 checkpoint 校验使用的稳定字符串。"""
    return str(path.expanduser().resolve())


def scaler_to_state(scaler: StandardScaler) -> Dict[str, object]:
    """
    函数功能：
        将 StandardScaler 转为 torch checkpoint 可稳定保存的状态字典。

    说明：
        不直接依赖 pickle 保存完整对象，后续恢复时只重建 transform 所需字段，
        降低 sklearn 版本差异带来的不透明风险。
    """
    required_attrs = ["mean_", "scale_", "var_", "n_features_in_"]
    missing = [name for name in required_attrs if not hasattr(scaler, name)]
    if missing:
        raise ValueError(f"scaler 尚未 fit，缺少字段：{missing}")
    state: Dict[str, object] = {
        "mean_": np.asarray(scaler.mean_, dtype=np.float64),
        "scale_": np.asarray(scaler.scale_, dtype=np.float64),
        "var_": np.asarray(scaler.var_, dtype=np.float64),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": getattr(scaler, "n_samples_seen_", None),
    }
    if hasattr(scaler, "feature_names_in_"):
        state["feature_names_in_"] = np.asarray(scaler.feature_names_in_)
    return state


def scaler_from_state(state: Mapping[str, object]) -> StandardScaler:
    """函数功能：从 checkpoint 状态重建 StandardScaler，用于 resume 后直接 transform。"""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var_"], dtype=np.float64)
    scaler.n_features_in_ = int(state["n_features_in_"])
    n_samples_seen = state.get("n_samples_seen_")
    if n_samples_seen is not None:
        scaler.n_samples_seen_ = np.asarray(n_samples_seen) if isinstance(n_samples_seen, (list, tuple, np.ndarray)) else n_samples_seen
    if "feature_names_in_" in state:
        scaler.feature_names_in_ = np.asarray(state["feature_names_in_"])
    return scaler


def build_resume_signature(args: argparse.Namespace, period_candidate_values: Optional[Sequence[int]]) -> Dict[str, object]:
    """
    函数功能：
        构造影响 router 参数形状、训练目标、输入 embedding 和数据切分的严格校验签名。
    """
    return {
        "config_name": None,
        "model_columns": list(MODEL_COLUMNS),
        "router_mode": args.router_mode,
        "metric": args.metric,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "embedding_metadata": {
            "embedding_version": f"{EMBEDDING_VERSION}_online_streaming",
            "encoder_name": args.encoder_name,
            "variant": args.variant,
            "pooling": args.pooling,
            "normalization_preset": args.normalization_preset,
            "image_size": int(args.image_size),
            "norm_mode": args.norm_mode,
            "pixel_mode": args.pixel_mode,
            "clip": float(args.clip),
            "period_selection": args.period_selection,
            "period_candidates_arg": args.period_candidates,
            "period_candidates": [int(value) for value in period_candidate_values] if period_candidate_values is not None else None,
            "dtype_arg": args.dtype,
        },
        "stream_shard_index": int(args.stream_shard_index),
        "stream_shard_count": int(args.stream_shard_count),
        "labels_path": stable_path_string(args.labels_path),
        "prediction_manifest_path": stable_path_string(args.prediction_manifest_path),
        "config_path": stable_path_string(args.config_path),
    }


def load_checkpoint(path: Path) -> Dict[str, object]:
    """函数功能：加载 checkpoint，并显式关闭 weights_only 限制以读取 numpy/scaler 状态。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 resume checkpoint：{path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_checkpoint_matches(
    *,
    checkpoint: Mapping[str, object],
    expected_signature: Mapping[str, object],
    config_name: str,
) -> None:
    """
    函数功能：
        对 checkpoint 与当前命令做严格签名校验，防止不同 config、embedding 口径或数据路径误接。
    """
    signature = dict(expected_signature)
    signature["config_name"] = str(config_name)
    fields = [
        "config_name",
        "model_columns",
        "router_mode",
        "metric",
        "hidden_dim",
        "dropout",
        "lr",
        "weight_decay",
        "huber_beta",
        "kl_tau",
        "lambda_kl",
        "embedding_metadata",
        "stream_shard_index",
        "stream_shard_count",
        "labels_path",
        "prediction_manifest_path",
        "config_path",
    ]
    mismatches = []
    for field in fields:
        actual = checkpoint.get(field)
        expected = signature[field]
        if actual != expected:
            mismatches.append({"field": field, "checkpoint": actual, "current": expected})
    if mismatches:
        raise ValueError(f"resume checkpoint 与当前命令不兼容：{json.dumps(mismatches, ensure_ascii=False, indent=2)}")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """函数功能：把 optimizer state 中的 tensor 移到当前训练设备，避免 resume 后设备不一致。"""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def checkpoint_filename(config_name: str, completed_epochs: int) -> str:
    """函数功能：生成不含特殊字符的 config 级 checkpoint 文件名。"""
    safe_config = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(config_name))
    return f"router_{safe_config}_epoch_{int(completed_epochs):04d}.pt"


def save_checkpoint(
    *,
    output_dir: Path,
    config_name: str,
    router: VisualMLPRouter,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler,
    completed_epochs: int,
    args: argparse.Namespace,
    period_candidate_values: Optional[Sequence[int]],
    epoch_summaries: Sequence[Mapping[str, float]],
    scaler_batches: int,
    scaler_samples: int,
) -> Path:
    """
    函数功能：
        在每个 epoch 结束保存可续训 checkpoint，并维护 config 级 latest checkpoint。
    """
    signature = build_resume_signature(args, period_candidate_values)
    signature["config_name"] = str(config_name)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        **signature,
        "checkpoint_version": "stage1_streaming_router_checkpoint_v1",
        "router_state_dict": router.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state": scaler_to_state(scaler),
        "completed_epochs": int(completed_epochs),
        "epoch_summaries": [dict(row) for row in epoch_summaries],
        "scaler_batches": int(scaler_batches),
        "scaler_samples": int(scaler_samples),
        "saved_at": display_time(),
    }
    checkpoint_path = checkpoint_dir / checkpoint_filename(str(config_name), int(completed_epochs))
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)
    latest_path = checkpoint_dir / f"latest_{''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in str(config_name))}.pt"
    latest_tmp = latest_path.with_suffix(latest_path.suffix + ".tmp")
    torch.save(payload, latest_tmp)
    latest_tmp.replace(latest_path)
    (checkpoint_dir / "latest_checkpoint_index.json").write_text(
        json.dumps(
            {
                "config_name": str(config_name),
                "completed_epochs": int(completed_epochs),
                "checkpoint_path": str(checkpoint_path),
                "latest_checkpoint_path": str(latest_path),
                "updated_at": display_time(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return latest_path


def cleanup_output_files(output_dir: Path, *, resume: bool, train_only: bool) -> None:
    """
    函数功能：
        清理本次会被重新生成的 CSV/summary 文件；resume 时绝不删除 checkpoints。
    """
    stale_names = [
        "visual_router_predictions.csv",
        "visual_router_soft_fusion_predictions.csv",
        "visual_router_summary.csv",
        "visual_router_soft_fusion_summary.csv",
        "visual_router_selected_model_counts.csv",
        "visual_router_comparison.csv",
        "visual_router_streaming_summary.md",
    ]
    if not resume:
        stale_names.extend(["online_embedding_manifest.csv", "online_embedding_latency_summary.csv"])
    if train_only:
        stale_names = [name for name in stale_names if name.startswith("visual_router_")]
    for stale_name in stale_names:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


def required_prediction_sample_keys(labels_df: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    """
    函数功能：
        根据本次训练/评估模式推导真正需要索引 prediction manifest 的 sample_key。

    说明：
        full-scale merged manifest 约 52GB，不能直接为 vali/test 全量同时建立 Python
        lookup。train-only 单轮训练只需要 vali split 的五专家预测数组作为 fusion loss
        监督；eval-only 才需要 test split 用于 raw soft fusion 复算。
    """
    needed: List[pd.Series] = []
    if args.router_mode == "fusion_huber_kl" and int(args.epochs) > 0:
        needed.append(labels_df.loc[labels_df["split"].astype(str) == "vali", "sample_key"].astype(str))
    if not args.train_only and not args.skip_soft_fusion:
        needed.append(labels_df.loc[labels_df["split"].astype(str) == "test", "sample_key"].astype(str))
    if not needed:
        return []
    # 保留原始出现顺序即可；full-scale 数百万 sample_key 不需要额外排序，避免启动阶段
    # 产生一份巨大的临时列表和排序开销。
    return pd.concat(needed, ignore_index=True).astype(str).drop_duplicates().tolist()


class SQLitePredictionIndex:
    """
    类功能：
        封装 full-scale prediction manifest 的磁盘索引。

    说明：
        full-scale vali 需要约 4675 万条 `(sample_key, model_name)` 记录。
        这些记录不能以 Python dict 常驻内存；SQLite 查询会把内存规模限制在
        当前 embedding batch 对应的几百条记录。
    """

    def __init__(self, db_path: Path, manifest_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.manifest_dir = Path(manifest_dir)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row

    def fetch_records(self, sample_keys: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, object]]:
        """
        函数功能：
            批量查询当前 batch 的五专家 prediction record。

        输入：
            sample_keys 为当前 embedding batch 的 sample_key 列表。

        输出：
            与旧 `prediction_lookup` 兼容的 `(sample_key, model_name) -> record`
            字典，但只包含当前 batch 的记录。
        """
        keys = [str(key) for key in sample_keys]
        if not keys:
            return {}
        placeholders = ",".join(["?"] * len(keys))
        rows = self.connection.execute(
            f"""
            SELECT sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                   array_storage, y_true_row_index, y_pred_row_index
            FROM prediction_index
            WHERE sample_key IN ({placeholders})
            """,
            keys,
        ).fetchall()
        records: Dict[Tuple[str, str], Dict[str, object]] = {}
        for row in rows:
            record = dict(row)
            sample_key = str(record["sample_key"])
            model_name = str(record["model_name"])
            # manifest 中多数路径为相对路径；查询时再解析，避免磁盘索引重复写入长绝对路径。
            record["y_true_path"] = resolve_cache_array_path(str(record["y_true_path"]), self.manifest_dir)
            record["y_pred_path"] = resolve_cache_array_path(str(record["y_pred_path"]), self.manifest_dir)
            records[(sample_key, model_name)] = record
        return records

    def close(self) -> None:
        """函数功能：显式关闭 SQLite 连接，便于长任务结束时释放文件句柄。"""
        self.connection.close()


def build_lightweight_prediction_index(
    prediction_manifest_path: Path,
    *,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
    index_db_path: Path,
) -> SQLitePredictionIndex:
    """
    函数功能：
        分块扫描 prediction manifest，只为指定 sample_key 建立 SQLite 磁盘索引。

    关键优化（2026-06-16）：
        - 不在 Python 内存中保存千万级 `(sample_key, model_name)` dict；
        - 只把当前 chunk 的匹配行批量写入 SQLite；
        - 训练时按 embedding batch 查询几百条 record，再即时读取 packed npy 单行。

    返回：
        SQLitePredictionIndex，可用 `fetch_records()` 查询当前 batch 的 record。
    """
    if not prediction_manifest_path.exists():
        raise FileNotFoundError(f"找不到 prediction manifest：{prediction_manifest_path}")
    key_set = {str(key) for key in sample_keys}
    if not key_set:
        raise ValueError("SQLite prediction index 需要至少一个 sample_key")
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
    manifest_dir = prediction_manifest_path.parent
    index_db_path = Path(index_db_path)
    index_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path = index_db_path.with_suffix(index_db_path.suffix + ".tmp")
    if tmp_db_path.exists():
        tmp_db_path.unlink()
    if index_db_path.exists():
        index_db_path.unlink()

    connection = sqlite3.connect(str(tmp_db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
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
    rows_seen = 0
    matched_rows = 0
    for chunk_idx, chunk_df in enumerate(
        pd.read_csv(prediction_manifest_path, usecols=usecols, chunksize=int(chunk_read_rows)),
        start=1,
    ):
        rows_seen += int(len(chunk_df))
        matched_df = chunk_df[chunk_df["sample_key"].astype(str).isin(key_set)]
        if matched_df.empty:
            continue
        matched_rows += int(len(matched_df))
        insert_rows = [
            (
                str(row.sample_key),
                str(row.model_name),
                str(row.y_true_path),
                str(row.y_pred_path),
                float(row.mae),
                float(row.mse),
                str(row.array_storage) if pd.notna(row.array_storage) else None,
                None if pd.isna(row.y_true_row_index) else int(row.y_true_row_index),
                None if pd.isna(row.y_pred_row_index) else int(row.y_pred_row_index),
            )
            for row in matched_df.itertuples(index=False)
        ]
        try:
            connection.executemany(
                """
                INSERT INTO prediction_index (
                    sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                    array_storage, y_true_row_index, y_pred_row_index
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("prediction manifest 中 sample_key + model_name 存在重复") from exc
        connection.commit()
        if chunk_idx == 1 or chunk_idx % 25 == 0:
            print(
                f"[manifest_index] chunks={chunk_idx} rows_seen={rows_seen} "
                f"matched_rows={matched_rows} target_sample_keys={len(key_set)}",
                flush=True,
            )
    expected_records = len(key_set) * len(MODEL_COLUMNS)
    actual_records = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
    if actual_records != expected_records:
        connection.close()
        raise ValueError(
            f"prediction manifest 子集不完整：expected_records={expected_records} actual={actual_records} "
            f"sample_keys={len(key_set)}"
        )
    connection.execute("CREATE INDEX idx_prediction_index_sample_key ON prediction_index(sample_key)")
    metadata = {
        "created_at": display_time(),
        "prediction_manifest_path": str(prediction_manifest_path),
        "target_sample_keys": int(len(key_set)),
        "expected_records": int(expected_records),
        "actual_records": int(actual_records),
        "chunk_read_rows": int(chunk_read_rows),
        "manifest_dir": str(manifest_dir),
    }
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
        [(str(key), json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
    )
    connection.commit()
    connection.close()
    tmp_db_path.replace(index_db_path)
    print(
        f"[manifest_index] sqlite_index_ready path={index_db_path} records={actual_records} "
        f"target_sample_keys={len(key_set)}",
        flush=True,
    )
    return SQLitePredictionIndex(index_db_path, manifest_dir)


def load_prediction_lookup_for_sample_keys(
    prediction_manifest_path: Path,
    *,
    sample_keys: Sequence[str],
    chunk_read_rows: int,
) -> Mapping[Tuple[str, str], Dict[str, object]]:
    """
    函数功能：
        【已弃用】分块扫描 prediction manifest，只为指定 sample_key 建立 lookup。
    
    注意：此函数在 full-scale 场景下会导致 OOM（~117 GB），已被 
    build_lightweight_prediction_index 替代。保留此函数仅为向后兼容。
    """
    import warnings
    warnings.warn(
        "load_prediction_lookup_for_sample_keys is deprecated for full-scale scenarios. "
        "Use build_lightweight_prediction_index instead to avoid OOM.",
        DeprecationWarning,
        stacklevel=2,
    )
    # 原有实现保持不变，但不再使用
    if not prediction_manifest_path.exists():
        raise FileNotFoundError(f"找不到 prediction manifest：{prediction_manifest_path}")
    key_set = {str(key) for key in sample_keys}
    if not key_set:
        return {}
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
    manifest_dir = prediction_manifest_path.parent
    lookup: Dict[Tuple[str, str], Dict[str, object]] = {}
    rows_seen = 0
    matched_rows = 0
    for chunk_idx, chunk_df in enumerate(
        pd.read_csv(prediction_manifest_path, usecols=usecols, chunksize=int(chunk_read_rows)),
        start=1,
    ):
        rows_seen += int(len(chunk_df))
        matched_df = chunk_df[chunk_df["sample_key"].astype(str).isin(key_set)]
        if matched_df.empty:
            continue
        matched_rows += int(len(matched_df))
        for row in matched_df.itertuples(index=False):
            record = row._asdict()
            sample_key = str(record["sample_key"])
            model_name = str(record["model_name"])
            lookup_key = (sample_key, model_name)
            if lookup_key in lookup:
                raise ValueError(f"prediction manifest 中 sample_key + model_name 重复：{lookup_key}")
            record["sample_key"] = sample_key
            record["model_name"] = model_name
            record["y_true_path"] = resolve_cache_array_path(str(record["y_true_path"]), manifest_dir)
            record["y_pred_path"] = resolve_cache_array_path(str(record["y_pred_path"]), manifest_dir)
            lookup[lookup_key] = record
        if chunk_idx == 1 or chunk_idx % 25 == 0:
            print(
                f"[manifest_lookup] chunks={chunk_idx} rows_seen={rows_seen} "
                f"matched_rows={matched_rows} target_sample_keys={len(key_set)}",
                flush=True,
            )
    expected_records = len(key_set) * len(MODEL_COLUMNS)
    if len(lookup) != expected_records:
        raise ValueError(
            f"prediction manifest 子集不完整：expected_records={expected_records} actual={len(lookup)} "
            f"sample_keys={len(key_set)}"
        )
    return lookup


def filter_stream_shard(labels_df: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    """函数功能：按 sample_key 稳定切分 streaming shard。"""
    if shard_count <= 0:
        raise ValueError("--stream-shard-count 必须为正整数")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--stream-shard-index 必须落在 [0, stream_shard_count)")
    if shard_count == 1:
        return labels_df.copy().reset_index(drop=True)
    ordered = labels_df.sort_values("sample_key").reset_index(drop=True)
    mask = (np.arange(len(ordered)) % int(shard_count)) == int(shard_index)
    return ordered.loc[mask].reset_index(drop=True)


def limit_samples_per_split(labels_df: pd.DataFrame, max_samples_per_split: Optional[int]) -> pd.DataFrame:
    """函数功能：为 full-scale dry-run 截取每个 split 的前若干 sample。"""
    if max_samples_per_split is None:
        return labels_df.copy().reset_index(drop=True)
    if max_samples_per_split <= 0:
        raise ValueError("--max-samples-per-split 必须为正整数")
    rows = []
    for (_, split), group in labels_df.sort_values("sample_key").groupby(["config_name", "split"], sort=True):
        rows.append(group.head(int(max_samples_per_split)))
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def windows_from_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：从 labels DataFrame 生成需要在线 embedding 的唯一窗口清单。"""
    required_cols = ["sample_key", "config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    windows_df = labels_df[required_cols].drop_duplicates().reset_index(drop=True)
    if windows_df["sample_key"].duplicated().any():
        dup = windows_df.loc[windows_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"labels 中 sample_key 不唯一，示例：{dup}")
    return windows_df


def build_vit_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> ViTModel:
    """函数功能：构建冻结 ViT encoder，并处理本地 cache / dtype 口径。"""
    model = ViTModel.from_pretrained(
        args.encoder_name,
        local_files_only=bool(args.local_files_only),
        add_pooling_layer=args.pooling == "pooler",
    )
    model.eval().to(device=device)
    if dtype == torch.float16:
        model = model.half()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if bool(getattr(args, "vit_data_parallel", False)) and device.type == "cuda" and torch.cuda.device_count() > 1:
        # 只并行冻结 ViT encoder 的前向，router/scaler/checkpoint 仍保持单进程语义；
        # 这样可以利用多卡加速 embedding 生成，同时避免多进程训练状态同步复杂化。
        model = torch.nn.DataParallel(model)
    return model


def is_retryable_model_load_error(exc: Exception) -> bool:
    """
    函数功能：识别 ViT/Hugging Face 模型加载阶段的临时错误。

    说明：
        full-scale 长任务可能在首次拉模型、读取远端缓存或访问本地镜像时遇到
        429/503、连接超时、短暂 I/O 抖动。这里仅把明显的临时错误标记为可重试，
        避免把配置错误误吞掉。
    """
    message = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = [
        "429",
        "503",
        "timeout",
        "timed out",
        "connection",
        "temporar",
        "readerror",
        "service unavailable",
        "too many requests",
        "httperror",
        "connectionreset",
    ]
    return any(marker in message for marker in retry_markers)


def load_vit_model_with_retry(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, max_attempts: int = 3) -> ViTModel:
    """函数功能：带有限指数退避的 ViT 加载，降低网络抖动对长任务的影响。"""
    last_exc: Optional[Exception] = None
    for attempt in range(1, int(max_attempts) + 1):
        try:
            return build_vit_model(args, device, dtype)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= int(max_attempts) or not is_retryable_model_load_error(exc):
                raise
            sleep_seconds = float(min(30.0, 2.0 ** (attempt - 1)))
            print(
                f"[retry] ViT 加载失败，准备重试 {attempt + 1}/{max_attempts}，"
                f"sleep={sleep_seconds:.1f}s，error={repr(exc)}",
                flush=True,
            )
            time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def resolve_period_candidates(args: argparse.Namespace, history_length: int) -> Optional[List[int]]:
    """函数功能：解析 fixed candidate 周期列表，并写入 metadata。"""
    values = parse_period_candidate_arg(args.period_candidates)
    if args.period_selection == "fixed_candidates" and values is None:
        values = [int(value) for value in make_default_period_candidates(history_length, device=torch.device("cpu")).tolist()]
    return values


def iter_online_embedding_batches(
    *,
    windows_df: pd.DataFrame,
    data_config,
    vit_model: ViTModel,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    period_candidate_values: Optional[Sequence[int]],
) -> Iterator[Tuple[pd.DataFrame, np.ndarray, List[Dict[str, object]]]]:
    """
    函数功能：
        流式读取 Quito 历史窗口，生成当前 batch 的 ViT embedding。

    输出：
        batch_manifest_df: 当前 batch 的 embedding metadata，不含 embedding_path；
        embeddings: 当前 batch 的 float32 embedding 矩阵；
        latency_rows: 当前 batch 的 imageization / ViT 前向耗时。

    关键约束：
        `embeddings` 只在调用方处理当前 batch 时存在；本函数不保存 `.npy`，
        也不返回全量 `sample_key -> embedding` 字典。
    """
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"].astype(str), windows_df["config_name"].astype(str)))

    for split in sorted(windows_df["split"].astype(str).unique()):
        datasets = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=mode_from_split(str(split)),
            cleanup=False,
            concat=False,
        )
        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
            item_ids = sorted(
                item_id
                for req_split, req_dataset, item_id in required_index
                if req_split == split and req_dataset == dataset_name
            )
            for item_id in item_ids:
                item_dataset = dataset.copy() if hasattr(dataset, "copy") else None
                if item_dataset is None:
                    import copy

                    item_dataset = copy.deepcopy(dataset)
                item_dataset.select_user_data(int(item_id))
                channel_count = int(item_dataset.data.shape[0])
                required_for_item = required_index[(str(split), str(dataset_name), int(item_id))]
                for pair_batch in batch_required_pairs(required_for_item, int(args.embedding_batch_size)):
                    x_windows: List[np.ndarray] = []
                    sample_keys: List[str] = []
                    channel_ids: List[int] = []
                    window_indices: List[int] = []
                    for channel_id, window_index, sample_key in pair_batch:
                        if int(channel_id) >= channel_count:
                            raise ValueError(f"channel_id 越界：sample_key={sample_key}")
                        window_start = int(window_index)
                        window_end = window_start + int(data_config.seq_len)
                        # 只读取历史窗口 x；不访问未来 y、oracle 或专家误差作为输入。
                        x_window = item_dataset.data[int(channel_id), window_start:window_end, :]
                        if x_window.shape[0] != int(data_config.seq_len):
                            raise ValueError(f"历史窗口长度不完整：sample_key={sample_key} shape={x_window.shape}")
                        x_windows.append(x_window)
                        sample_keys.append(str(sample_key))
                        channel_ids.append(int(channel_id))
                        window_indices.append(int(window_index))

                    x_cpu = torch.from_numpy(np.stack(x_windows, axis=0)).to(dtype=torch.float32)
                    with torch.inference_mode():
                        image_start = _timer_start(device)
                        pixel_values = make_pseudo_images(
                            x_cpu,
                            variant=args.variant,
                            norm_mode=args.norm_mode,
                            pixel_mode=args.pixel_mode,
                            clip=float(args.clip),
                            image_size=int(args.image_size),
                            device=device,
                            dtype=dtype,
                            normalization_preset=args.normalization_preset,
                            period_selection=args.period_selection,
                            period_candidate_values=period_candidate_values,
                        )
                        image_ms = _timer_stop(image_start, device)

                        forward_start = _timer_start(device)
                        outputs = vit_model(pixel_values=pixel_values)
                        embeddings = pool_vit_outputs(outputs, args.pooling)
                        forward_ms = _timer_stop(forward_start, device)
                        embeddings_cpu = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()

                    rows: List[Dict[str, object]] = []
                    for row_idx, sample_key in enumerate(sample_keys):
                        key = PredictionCacheKey(
                            config_name=str(config_by_key[sample_key]),
                            split=str(split),
                            dataset_name=str(dataset_name),
                            item_id=int(item_id),
                            channel_id=int(channel_ids[row_idx]),
                            window_index=int(window_indices[row_idx]),
                        )
                        if key.as_string() != sample_key:
                            raise ValueError(f"sample_key 与元信息不一致：{sample_key} vs {key.as_string()}")
                        rows.append(
                            make_embedding_manifest_row(
                                sample_key=sample_key,
                                config_name=str(config_by_key[sample_key]),
                                split=str(split),
                                dataset_name=str(dataset_name),
                                item_id=int(item_id),
                                channel_id=int(channel_ids[row_idx]),
                                window_index=int(window_indices[row_idx]),
                                history_length=int(data_config.seq_len),
                                embedding_dim=int(embeddings_cpu.shape[1]),
                                args=args,
                            )
                        )
                    latency_rows = [
                        {
                            "split": str(split),
                            "dataset_name": str(dataset_name),
                            "item_id": int(item_id),
                            "batch_size": int(len(sample_keys)),
                            "imageization_ms": float(image_ms),
                            "encoder_forward_ms": float(forward_ms),
                            "imageization_per_window_ms": float(image_ms / len(sample_keys)),
                            "encoder_forward_per_window_ms": float(forward_ms / len(sample_keys)),
                            "device": str(device),
                        }
                    ]
                    yield pd.DataFrame(rows), embeddings_cpu.astype(np.float32), latency_rows


def load_prediction_tensors_from_lightweight_index(
    sample_keys: Sequence[str],
    prediction_index: SQLitePredictionIndex,
    *,
    error_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        从轻量级路径索引即时读取五专家预测数组和共享 y_true，用于 fusion loss。
    
    关键优化（2026-06-16）：
        - 不再依赖全量 prediction_lookup dict；
        - 只从 SQLite 查询当前 batch 的 record；
        - 按 packed row index 即时读取 `.npy` 单行，避免整块数组进入内存。
    
    输入：
        sample_keys: 与 feature/label DataFrame 行顺序一致的 sample_key。
        prediction_index: SQLitePredictionIndex 磁盘索引。
    
    输出：
        - y_preds: `[N, M, ...]`，M 为五个专家；
        - y_true: `[N, ...]`；
        - expert_errors: `[N, M]`，按 `error_metric` 复算出的专家窗口误差。
    """
    batch_lookup = prediction_index.fetch_records(sample_keys)
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []
    all_errors: List[np.ndarray] = []
    for sample_key in sample_keys:
        missing_models = [model for model in MODEL_COLUMNS if (str(sample_key), model) not in batch_lookup]
        if missing_models:
            raise ValueError(f"prediction index 缺少 sample_key={sample_key} 的专家：{missing_models}")

        sample_preds: List[np.ndarray] = []
        sample_true: Optional[np.ndarray] = None
        sample_errors: List[float] = []
        for model_name in MODEL_COLUMNS:
            record = batch_lookup[(str(sample_key), model_name)]
            y_pred = load_prediction_array(record, "y_pred")
            current_y_true = load_prediction_array(record, "y_true")
            if sample_true is None:
                sample_true = current_y_true
            elif not np.array_equal(sample_true, current_y_true):
                raise ValueError(f"同一 sample_key 的 y_true 内容不一致：{sample_key}")
            if y_pred.shape != current_y_true.shape:
                raise ValueError(f"y_pred/y_true shape 不一致：sample_key={sample_key} model={model_name}")
            sample_preds.append(y_pred)
            diff = y_pred - current_y_true
            if error_metric == "mae":
                sample_errors.append(float(np.mean(np.abs(diff))))
            elif error_metric == "mse":
                sample_errors.append(float(np.mean(diff ** 2)))
            else:
                raise ValueError(f"未知 error_metric={error_metric}")

        assert sample_true is not None
        all_preds.append(np.stack(sample_preds, axis=0))
        all_trues.append(sample_true)
        all_errors.append(np.asarray(sample_errors, dtype=np.float32))

    y_preds = np.stack(all_preds, axis=0).astype(np.float32)
    y_true = np.stack(all_trues, axis=0).astype(np.float32)
    expert_errors = np.stack(all_errors, axis=0).astype(np.float32)
    if y_preds.ndim < 3:
        raise ValueError(f"专家预测张量维度异常：{y_preds.shape}")
    if not (np.isfinite(y_preds).all() and np.isfinite(y_true).all() and np.isfinite(expert_errors).all()):
        raise ValueError("prediction tensor 中存在非有限值")
    return y_preds, y_true, expert_errors


def verify_training_expert_errors_from_expert_batch(
    *,
    sample_keys: Sequence[str],
    y_pred: np.ndarray,
    y_true: np.ndarray,
    legacy_expert_errors: np.ndarray,
    model_columns: Sequence[str],
    error_metric: str,
    training_batch_index: int | None = None,
    epoch: int | None = None,
    output_dir: Path | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """
    函数功能：
        P9f training loss ExpertBatch 旁路校验：把当前 legacy SQLite path
        已读取出的 y_pred/y_true 包装为 ExpertBatch，并显式复算 expert_errors。

    输入：
        sample_keys/model_columns: 当前 training batch 的样本和专家顺序。
        y_pred/y_true: 正式 fusion_huber_kl loss 已经读取出的预测和真实值数组。
        legacy_expert_errors: 正式训练仍然使用的旧路径 expert_errors。
        error_metric: 只支持 mae/mse，与当前训练监督 metric 对齐。
        training_batch_index/epoch/output_dir: 仅用于失败定位，不写任何文件。

    输出：
        无返回值；若 ExpertBatch.y_pred/y_true 复算结果与旧路径不一致则抛错。

    关键约束：
        本 helper 不返回替代 loss，不参与反传，不改变 optimizer/scheduler/scaler/
        checkpoint，也不把 expert_errors 纳入通用 ExpertBatch contract。
    """
    metric = str(error_metric)
    if metric not in {"mae", "mse"}:
        raise ValueError(f"verify_training_expert_errors_from_expert_batch 只支持 mae/mse，actual={metric}")

    expert_batch = ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in sample_keys),
        model_columns=tuple(str(model_name) for model_name in model_columns),
        y_pred=np.asarray(y_pred, dtype=np.float32),
        y_true=np.asarray(y_true, dtype=np.float32),
        extra={
            "source": "train_visual_router_online_streaming.verify_training_expert_errors_from_expert_batch",
            "phase": "training",
            "router_mode": "fusion_huber_kl",
            "metric": metric,
            "training_batch_index": None if training_batch_index is None else int(training_batch_index),
            "epoch": None if epoch is None else int(epoch),
            "output_dir": None if output_dir is None else str(output_dir),
        },
    )
    expert_y_pred = np.asarray(expert_batch.y_pred, dtype=np.float32)
    expert_y_true = np.asarray(expert_batch.y_true, dtype=np.float32)
    legacy_errors = np.asarray(legacy_expert_errors, dtype=np.float32)
    if expert_y_pred.shape[0] != len(expert_batch.sample_keys):
        raise ValueError(f"ExpertBatch y_pred 样本维度异常：shape={expert_y_pred.shape} sample_count={len(expert_batch.sample_keys)}")
    if expert_y_pred.shape[1] != len(expert_batch.model_columns):
        raise ValueError(f"ExpertBatch y_pred 专家维度异常：shape={expert_y_pred.shape} model_count={len(expert_batch.model_columns)}")
    if expert_y_true.shape[0] != len(expert_batch.sample_keys):
        raise ValueError(f"ExpertBatch y_true 样本维度异常：shape={expert_y_true.shape} sample_count={len(expert_batch.sample_keys)}")

    diff = expert_y_pred - expert_y_true[:, None, ...]
    reduce_axes = tuple(range(2, diff.ndim))
    if metric == "mae":
        recomputed_errors = np.mean(np.abs(diff), axis=reduce_axes, dtype=np.float32)
    else:
        recomputed_errors = np.mean(diff ** 2, axis=reduce_axes, dtype=np.float32)
    recomputed_errors = np.asarray(recomputed_errors, dtype=np.float32)
    if legacy_errors.shape != recomputed_errors.shape:
        raise AssertionError(
            "training ExpertBatch expert_errors shape 不一致："
            "phase=training router_mode=fusion_huber_kl "
            f"metric={metric} batch_index={training_batch_index} training_batch_index={training_batch_index} "
            f"legacy_shape={legacy_errors.shape} expert_batch_shape={recomputed_errors.shape} "
            f"output_dir={output_dir}"
        )

    mismatch_mask = ~np.isclose(legacy_errors, recomputed_errors, rtol=float(rtol), atol=float(atol))
    if np.any(mismatch_mask):
        sample_idx, expert_idx = np.argwhere(mismatch_mask)[0].tolist()
        sample_key = expert_batch.sample_keys[int(sample_idx)]
        model_name = expert_batch.model_columns[int(expert_idx)]
        legacy_value = float(legacy_errors[int(sample_idx), int(expert_idx)])
        recomputed_value = float(recomputed_errors[int(sample_idx), int(expert_idx)])
        raise AssertionError(
            "training ExpertBatch expert_errors 旁路校验不一致："
            "phase=training "
            "router_mode=fusion_huber_kl "
            f"metric={metric} "
            f"batch_index={training_batch_index} "
            f"training_batch_index={training_batch_index} "
            f"epoch={epoch} "
            f"sample_key={sample_key} "
            f"model_name={model_name} "
            f"expert_index={int(expert_idx)} "
            f"old_value={legacy_value} "
            f"legacy_value={legacy_value} "
            f"expert_batch_value={recomputed_value} "
            f"recomputed_value={recomputed_value} "
            f"output_dir={output_dir}"
        )


def train_on_stream_batch(
    *,
    router: VisualMLPRouter,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler,
    batch_manifest_df: pd.DataFrame,
    embeddings: np.ndarray,
    labels_by_key: Mapping[str, Mapping[str, object]],
    prediction_lookup: Optional[Mapping[Tuple[str, str], Dict[str, object]]],
    prediction_index: Optional[SQLitePredictionIndex],
    args: SimpleNamespace,
    device: torch.device,
    class_weight: torch.Tensor,
    training_batch_index: int | None = None,
) -> Dict[str, float]:
    """函数功能：用一个 streaming embedding batch 更新 router 参数。"""
    sample_keys = batch_manifest_df["sample_key"].astype(str).tolist()
    labels_batch = [labels_by_key[key] for key in sample_keys]
    x_scaled = scaler.transform(embeddings).astype(np.float32)
    losses: List[float] = []
    huber_losses: List[float] = []
    kl_losses: List[float] = []
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    huber_criterion = torch.nn.SmoothL1Loss(beta=float(args.huber_beta))

    if args.router_mode == "classification":
        if bool(getattr(args, "verify_training_expert_batch", False)):
            raise ValueError("--verify-training-expert-batch 只适用于 router_mode=fusion_huber_kl，不支持 classification")
        targets = np.asarray([MODEL_COLUMNS.index(str(row["oracle_model"])) for row in labels_batch], dtype=np.int64)
        for start in range(0, len(sample_keys), int(args.batch_size)):
            stop = start + int(args.batch_size)
            batch_x = torch.from_numpy(x_scaled[start:stop]).to(device=device)
            batch_y = torch.from_numpy(targets[start:stop]).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(router(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
    else:
        # 优先使用 SQLite 轻量级索引（batch 查询），回退到旧的全量 lookup。
        if prediction_index is not None:
            y_pred, y_true, expert_errors = load_prediction_tensors_from_lightweight_index(
                sample_keys, prediction_index, error_metric=str(args.metric)
            )
        elif prediction_lookup is not None:
            y_pred, y_true, expert_errors = load_prediction_tensors_for_samples(
                sample_keys, prediction_lookup, error_metric=str(args.metric)
            )
        else:
            raise ValueError("fusion_huber_kl 需要 prediction_index 或 prediction_lookup")

        if bool(getattr(args, "verify_training_expert_batch", False)):
            verify_training_expert_errors_from_expert_batch(
                sample_keys=sample_keys,
                y_pred=y_pred,
                y_true=y_true,
                legacy_expert_errors=expert_errors,
                model_columns=MODEL_COLUMNS,
                error_metric=str(args.metric),
                training_batch_index=training_batch_index,
                epoch=getattr(args, "current_epoch", None),
                output_dir=getattr(args, "output_dir", None),
            )

        soft_oracle = torch.softmax(-torch.from_numpy(expert_errors) / float(args.kl_tau), dim=1).to(dtype=torch.float32)
        for start in range(0, len(sample_keys), int(args.batch_size)):
            stop = start + int(args.batch_size)
            batch_x = torch.from_numpy(x_scaled[start:stop]).to(device=device)
            batch_pred = torch.from_numpy(y_pred[start:stop]).to(device=device)
            batch_true = torch.from_numpy(y_true[start:stop]).to(device=device)
            batch_q = soft_oracle[start:stop].to(device=device)
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
    return {
        "loss": float(np.mean(losses)),
        "huber_loss": float(np.mean(huber_losses)) if huber_losses else np.nan,
        "kl_loss": float(np.mean(kl_losses)) if kl_losses else np.nan,
    }


def predict_stream_batch(
    *,
    router: VisualMLPRouter,
    scaler: StandardScaler,
    batch_manifest_df: pd.DataFrame,
    embeddings: np.ndarray,
    labels_by_key: Mapping[str, Mapping[str, object]],
    router_name: str,
    device: torch.device,
) -> pd.DataFrame:
    """函数功能：对一个 test streaming batch 输出 hard top-1 router 预测行。"""
    x_scaled = scaler.transform(embeddings).astype(np.float32)
    router.eval()
    with torch.inference_mode():
        logits = router(torch.from_numpy(x_scaled).to(device=device))
        weights = torch.softmax(logits, dim=1).detach().cpu().numpy()
    selected_indices = weights.argmax(axis=1)
    weight_entropy = -(weights * np.log(np.clip(weights, EPS, 1.0))).sum(axis=1)
    normalized_weight_entropy = weight_entropy / np.log(len(MODEL_COLUMNS))
    max_weight = weights.max(axis=1)

    rows: List[Dict[str, object]] = []
    for row_idx, row in enumerate(batch_manifest_df.itertuples(index=False)):
        sample_key = str(row.sample_key)
        label_row = labels_by_key[sample_key]
        selected_model = MODEL_COLUMNS[int(selected_indices[row_idx])]
        output_row: Dict[str, object] = {
            "router_name": router_name,
            "config_name": str(label_row["config_name"]),
            "sample_key": sample_key,
            "split": str(label_row["split"]),
            "dataset_name": str(label_row["dataset_name"]),
            "item_id": int(label_row["item_id"]),
            "channel_id": int(label_row["channel_id"]),
            "window_index": int(label_row["window_index"]),
            "selected_model": selected_model,
            "selected_value": float(label_row[selected_model]),
            "oracle_model": str(label_row["oracle_model"]),
            "oracle_value": float(label_row["oracle_value"]),
            "regret_to_oracle": float(label_row[selected_model] - label_row["oracle_value"]),
            "oracle_label_correct": bool(selected_model == label_row["oracle_model"]),
            "weight_entropy": float(weight_entropy[row_idx]),
            "normalized_weight_entropy": float(normalized_weight_entropy[row_idx]),
            "max_weight": float(max_weight[row_idx]),
        }
        for model_idx, model_name in enumerate(MODEL_COLUMNS):
            output_row[f"weight_{model_name}"] = float(weights[row_idx, model_idx])
        rows.append(output_row)
    return pd.DataFrame(rows)


def _load_evaluation_arrays_for_batch(
    *,
    sample_keys: Sequence[str],
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    函数功能：
        按当前 test batch 的 sample_key 顺序读取五专家 y_pred 和共享 y_true。

    关键约束：
        这是 P9d ExpertBatch bridge 的 legacy arrays 来源，只读取当前 batch
        已经用于 `add_soft_fusion_metrics(...)` 的 prediction_lookup，不建立新
        provider，不改变正式训练、CSV、summary、metadata 或 status schema。
    """
    y_pred, y_true, _ = load_prediction_tensors_for_samples(
        [str(sample_key) for sample_key in sample_keys],
        prediction_lookup,
        error_metric="mae",
    )
    return y_pred, y_true


def _adapter_mismatch_message(
    *,
    field_name: str,
    old_value: object,
    adapter_value: object,
    row: Mapping[str, object],
    batch_index: int,
    row_offset: int,
    output_dir: Path,
) -> str:
    """
    函数功能：
        统一生成 adapter 旁路校验失败信息，保证定位信息足够直接。
    """
    return (
        "EvaluationInputAdapter 旁路校验不一致："
        f"config_name={row.get('config_name')} "
        f"split={row.get('split')} "
        f"batch_index={int(batch_index)} "
        f"row_offset={int(row_offset)} "
        f"sample_key={row.get('sample_key')} "
        f"field={field_name} "
        f"old_value={old_value} "
        f"adapter_value={adapter_value} "
        f"output_dir={output_dir}"
    )


def build_visual_router_expert_batch_for_evaluation(
    *,
    sample_keys: Sequence[str],
    y_pred: np.ndarray,
    y_true: np.ndarray,
    model_columns: Sequence[str],
    batch_index: int,
    output_dir: Path,
    row_index_metadata: object | None = None,
    extra: Mapping[str, object] | None = None,
) -> ExpertBatch:
    """
    函数功能：
        为 Visual Router evaluation adapter 旁路构造 canonical ExpertBatch。

    输入：
        sample_keys/model_columns: 当前 legacy batch 已经确定的样本和专家顺序。
        y_pred/y_true: 当前 batch 已读取出的专家预测和共享真实值数组。
        batch_index/output_dir: 只进入 lightweight lineage，供失败定位。
        row_index_metadata/extra: 可选轻量 metadata，不触发任何文件读取或写出。

    输出：
        ExpertBatch；数组只做 float32 视图/转换，不读取 manifest、不读取 prediction
        cache、不创建 run_dir、不计算 loss 或 evaluation。

    关键约束：
        P9d 只把 P9b 的旁路输入收敛到 ExpertBatch 边界。这里不代表正式接入
        PredictionCacheExpertProvider，也不替换 Visual Router legacy SQLite batch arrays。
    """
    merged_extra: dict[str, object] = {
        "source": "train_visual_router_online_streaming.verify_evaluation_adapter_bypass_batch",
        "expert_batch_source": "visual_router_legacy_sqlite_batch_arrays",
        "batch_index": int(batch_index),
        "output_dir": str(output_dir),
    }
    if extra:
        merged_extra.update(dict(extra))
    return ExpertBatch(
        sample_keys=tuple(str(sample_key) for sample_key in sample_keys),
        model_columns=tuple(str(model_name) for model_name in model_columns),
        y_pred=np.asarray(y_pred, dtype=np.float32),
        y_true=np.asarray(y_true, dtype=np.float32),
        row_index_metadata=row_index_metadata,
        extra=merged_extra,
    )


def verify_evaluation_adapter_bypass_batch(
    *,
    pred_df: pd.DataFrame,
    soft_df: pd.DataFrame,
    prediction_lookup: Mapping[Tuple[str, str], Dict[str, object]] | None = None,
    y_pred: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    output_dir: Path,
    batch_index: int,
    atol: float = 1e-5,
) -> None:
    """
    函数功能：
        在 Visual Router test evaluation batch 内旁路调用 EvaluationInputAdapter，
        逐样本校验 adapter 复算 rows 与现有正式 soft_df 字段一致。

    输入：
        pred_df: `predict_stream_batch(...)` 生成的 hard top-1 batch rows。
        soft_df: `add_soft_fusion_metrics(...)` 生成的正式 raw soft batch rows。
        prediction_lookup: 当前 batch 从 SQLite index 查询出的专家 prediction record。
        y_pred/y_true: 可选内存数组，供 smoke 测试避免访问磁盘 prediction cache。
        output_dir/batch_index: 仅用于失败信息定位，不写入任何 artifact。

    输出：
        无返回值；不一致时抛出 AssertionError。

    关键约束：
        该 helper 只做 P9d 内存旁路验证，不替换正式 append/write 逻辑，不修改
        VisualFeatureProvider、ViT provider、VisualMLPRouter、training loop 或
        fusion_huber_kl loss；ExpertBatch 只包装当前 batch 已经读取出的 legacy
        SQLite arrays，不接入 PredictionCacheExpertProvider。
    """
    if pred_df.empty:
        return
    sample_keys = pred_df["sample_key"].astype(str).tolist()
    if soft_df["sample_key"].astype(str).tolist() != sample_keys:
        adapter_keys = soft_df["sample_key"].astype(str).tolist()
        mismatch_offset = next(
            (idx for idx, (old_key, adapter_key) in enumerate(zip(sample_keys, adapter_keys)) if old_key != adapter_key),
            min(len(sample_keys), len(adapter_keys)),
        )
        context_row = pred_df.iloc[min(mismatch_offset, len(pred_df) - 1)].to_dict()
        raise AssertionError(
            _adapter_mismatch_message(
                field_name="sample_key_order",
                old_value=sample_keys,
                adapter_value=adapter_keys,
                row=context_row,
                batch_index=batch_index,
                row_offset=mismatch_offset,
                output_dir=output_dir,
            )
        )
    if y_pred is None or y_true is None:
        if prediction_lookup is None:
            raise ValueError("verify_evaluation_adapter_bypass_batch 需要 prediction_lookup 或显式 y_pred/y_true")
        y_pred, y_true = _load_evaluation_arrays_for_batch(sample_keys=sample_keys, prediction_lookup=prediction_lookup)

    # P9d 仍从正式 hard prediction rows 恢复 router 权重，soft_df 只作为旧路径指标对照。
    weights = pred_df[[f"weight_{model_name}" for model_name in MODEL_COLUMNS]].to_numpy(dtype=np.float32)
    expert_batch = build_visual_router_expert_batch_for_evaluation(
        sample_keys=sample_keys,
        model_columns=tuple(MODEL_COLUMNS),
        y_pred=y_pred,
        y_true=y_true,
        batch_index=batch_index,
        output_dir=output_dir,
        extra={
            "verification_scope": "visual_router_evaluation_adapter_bypass",
        },
    )
    try:
        adapter_result = EvaluationInputAdapter().evaluate(
            expert_batch=expert_batch,
            fusion_weights=weights,
            extra={
                "verification_scope": "visual_router_evaluation_adapter_bypass",
            },
        )
    except Exception as exc:
        preview_keys = sample_keys[:5]
        raise RuntimeError(
            "EvaluationInputAdapter 旁路复算失败："
            f"batch_index={int(batch_index)} sample_key_preview={preview_keys} output_dir={output_dir}"
        ) from exc

    adapter_rows = adapter_result.per_sample_rows
    if len(adapter_rows) != len(soft_df):
        context_row = soft_df.iloc[0].to_dict()
        raise AssertionError(
            _adapter_mismatch_message(
                field_name="row_count",
                old_value=len(soft_df),
                adapter_value=len(adapter_rows),
                row=context_row,
                batch_index=batch_index,
                row_offset=0,
                output_dir=output_dir,
            )
        )

    comparison_fields = {
        "selected_model": "selected_model",
        "hard_top1_mae_from_array": "hard_mae",
        "hard_top1_mse_from_array": "hard_mse",
        "soft_fusion_mae": "raw_soft_mae",
        "soft_fusion_mse": "raw_soft_mse",
        "max_weight": "max_weight",
        "weight_entropy": "weight_entropy",
    }
    for row_offset, (old_row, adapter_row) in enumerate(zip(soft_df.to_dict(orient="records"), adapter_rows)):
        if str(adapter_row["sample_key"]) != str(old_row["sample_key"]):
            raise AssertionError(
                _adapter_mismatch_message(
                    field_name="sample_key",
                    old_value=old_row["sample_key"],
                    adapter_value=adapter_row["sample_key"],
                    row=old_row,
                    batch_index=batch_index,
                    row_offset=row_offset,
                    output_dir=output_dir,
                )
            )
        old_selected_index = MODEL_COLUMNS.index(str(old_row["selected_model"]))
        if int(adapter_row["selected_index"]) != old_selected_index:
            raise AssertionError(
                _adapter_mismatch_message(
                    field_name="selected_index",
                    old_value=old_selected_index,
                    adapter_value=adapter_row["selected_index"],
                    row=old_row,
                    batch_index=batch_index,
                    row_offset=row_offset,
                    output_dir=output_dir,
                )
            )
        for old_field, adapter_field in comparison_fields.items():
            old_value = old_row[old_field]
            adapter_value = adapter_row[adapter_field]
            if isinstance(old_value, str) or isinstance(adapter_value, str):
                if str(old_value) != str(adapter_value):
                    raise AssertionError(
                        _adapter_mismatch_message(
                            field_name=old_field,
                            old_value=old_value,
                            adapter_value=adapter_value,
                            row=old_row,
                            batch_index=batch_index,
                            row_offset=row_offset,
                            output_dir=output_dir,
                        )
                    )
                continue
            if not np.isclose(float(old_value), float(adapter_value), rtol=1e-5, atol=float(atol)):
                raise AssertionError(
                    _adapter_mismatch_message(
                        field_name=old_field,
                        old_value=old_value,
                        adapter_value=adapter_value,
                        row=old_row,
                        batch_index=batch_index,
                        row_offset=row_offset,
                        output_dir=output_dir,
                    )
                )


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    """函数功能：追加写 CSV，首批自动写 header。"""
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_latency(output_dir: Path, latency_rows: List[Dict[str, object]], phase: str) -> None:
    """函数功能：追加写 embedding latency，并标注当前 streaming 阶段。"""
    latency_df = pd.DataFrame(latency_rows)
    if latency_df.empty:
        return
    latency_df["phase"] = str(phase)
    append_csv(output_dir / "online_embedding_latency_summary.csv", latency_df)


def summarize_csv_outputs(output_dir: Path, metric: str, labels_path: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """函数功能：从 streaming 写出的 CSV 生成兼容旧入口的 summary/comparison。"""
    hard_pred_df = pd.read_csv(output_dir / "visual_router_predictions.csv")
    hard_summary_df = summarize_hard_predictions(hard_pred_df)
    selected_counts_df = summarize_selected_model_counts(hard_pred_df)
    hard_summary_df.to_csv(output_dir / "visual_router_summary.csv", index=False)
    selected_counts_df.to_csv(output_dir / "visual_router_selected_model_counts.csv", index=False)

    soft_summary_df: Optional[pd.DataFrame] = None
    soft_path = output_dir / "visual_router_soft_fusion_predictions.csv"
    if soft_path.exists():
        soft_pred_df = pd.read_csv(soft_path)
        soft_summary_df = summarize_soft_fusion(soft_pred_df)
        soft_summary_df.to_csv(output_dir / "visual_router_soft_fusion_summary.csv", index=False)

    comparison_df = compare_with_baselines(output_dir, labels_path, hard_summary_df, soft_summary_df, metric)
    comparison_df.to_csv(output_dir / "visual_router_comparison.csv", index=False)
    return hard_summary_df, soft_summary_df, selected_counts_df, comparison_df


def write_summary_md(
    *,
    output_dir: Path,
    hard_summary: pd.DataFrame,
    soft_summary: Optional[pd.DataFrame],
    selected_counts: pd.DataFrame,
    comparison_df: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：写出中文 streaming 摘要。"""
    lines = [
        "# Stage 1 Streaming Online Visual Router",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 口径",
        "",
        "- 路径：`x -> pseudo image -> frozen ViT -> router`。",
        "- 伪图像 tensor 和 ViT embedding 均为 batch 运行时对象，不保存 `.npy`，不建立长期 embedding cache。",
        "- `StandardScaler.partial_fit` 只遍历 vali embedding；test 只用于 forward 和评估。",
        f"- streaming shard: `{metadata['stream_shard_index']}/{metadata['stream_shard_count']}`。",
        f"- encoder: `{metadata['embedding_metadata']['encoder_name']}`，period_selection: `{metadata['embedding_metadata']['period_selection']}`。",
        "",
        "## Hard Top-1 Summary",
        "",
        frame_to_markdown(hard_summary),
        "",
    ]
    if soft_summary is not None:
        lines.extend(["## Soft Fusion Summary", "", frame_to_markdown(soft_summary), ""])
    lines.extend(["## Top-1 选中专家分布", "", frame_to_markdown(selected_counts), ""])
    lines.extend(["## Baseline Comparison", "", frame_to_markdown(comparison_df.head(24)), ""])
    lines.extend(
        [
            "## 输出文件",
            "",
            f"- `visual_router_predictions.csv`: `{output_dir / 'visual_router_predictions.csv'}`",
            f"- `visual_router_summary.csv`: `{output_dir / 'visual_router_summary.csv'}`",
            f"- `visual_router_metadata.json`: `{output_dir / 'visual_router_metadata.json'}`",
            f"- `status.json`: `{output_dir / 'status.json'}`",
            "",
        ]
    )
    (output_dir / "visual_router_streaming_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 streaming online router 全流程。"""
    args = parse_args()
    if int(args.epochs) < 0:
        raise ValueError("--epochs 必须 >= 0；resume/eval-only 可使用 --epochs 0")
    if bool(args.verify_evaluation_adapter) and bool(args.skip_soft_fusion):
        raise ValueError("--verify-evaluation-adapter 需要 raw soft fusion 对齐证据，不能与 --skip-soft-fusion 同时使用")
    if bool(args.verify_training_expert_batch) and str(args.router_mode) == "classification":
        raise ValueError("--verify-training-expert-batch 只适用于 router_mode=fusion_huber_kl，不支持 classification")
    train_args = build_train_args(args)
    validate_training_args(train_args)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_online_visual_router_streaming_96_48_s"
    output_dir.mkdir(parents=True, exist_ok=True)
    # P9f 旁路校验只需要轻量诊断上下文；这些字段不进入正式输出 schema。
    train_args.verify_training_expert_batch = bool(args.verify_training_expert_batch)
    train_args.output_dir = output_dir
    resume_checkpoint = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    write_status(
        output_dir,
        {
            "status": "running",
            "phase": "init",
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "latest_checkpoint_path": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
            "completed_epochs": int(resume_checkpoint["completed_epochs"]) if resume_checkpoint is not None else 0,
        },
    )

    labels_df = load_labels(args.labels_path, args.metric)
    labels_df = filter_stream_shard(labels_df, int(args.stream_shard_index), int(args.stream_shard_count))
    labels_df = limit_samples_per_split(labels_df, args.max_samples_per_split)
    windows_df = windows_from_labels(labels_df)
    data_config = load_data_config(args.config_path)
    period_candidate_values = resolve_period_candidates(args, int(data_config.seq_len))
    vit_model = load_vit_model_with_retry(args, device, dtype)
    required_prediction_keys = required_prediction_sample_keys(labels_df, args)
    
    # 内存优化（2026-06-16）：使用 SQLite 磁盘索引替代全量 prediction_lookup。
    # 关键点是避免千万级 Python dict 常驻内存；训练时只查询当前 batch 的 record。
    prediction_index = (
        build_lightweight_prediction_index(
            args.prediction_manifest_path,
            sample_keys=required_prediction_keys,
            chunk_read_rows=int(args.chunk_read_rows),
            index_db_path=output_dir / "prediction_manifest_index.sqlite",
        )
        if required_prediction_keys
        else None
    )
    prediction_lookup = None  # 不再使用全量 lookup
    
    expected_signature = build_resume_signature(args, period_candidate_values)
    if resume_checkpoint is not None:
        config_names = sorted(labels_df["config_name"].astype(str).unique().tolist())
        if config_names != [str(resume_checkpoint.get("config_name"))]:
            raise ValueError(f"--resume-checkpoint 当前仅支持单 config 严格续训；labels config={config_names}, checkpoint config={resume_checkpoint.get('config_name')}")
    cleanup_output_files(output_dir, resume=resume_checkpoint is not None, train_only=bool(args.train_only))

    router_name = STREAMING_ONLINE_ROUTER_VERSION if args.router_mode == "fusion_huber_kl" else "visual_router_mlp_v1_classification_online_vit_streaming"
    config_metadata: List[Dict[str, object]] = []
    embedding_dim: Optional[int] = None
    total_embedding_batches = 0
    latest_checkpoint_path: Optional[Path] = Path(args.resume_checkpoint) if args.resume_checkpoint is not None else None

    for config_name, config_labels_df in labels_df.groupby("config_name", sort=True):
        config_windows_df = windows_df[windows_df["config_name"].astype(str) == str(config_name)].copy()
        vali_windows_df = config_windows_df[config_windows_df["split"] == "vali"].copy()
        test_windows_df = config_windows_df[config_windows_df["split"] == "test"].copy()
        config_labels_by_key = config_labels_df.set_index("sample_key").to_dict(orient="index")
        vali_labels_by_key = config_labels_df[config_labels_df["split"] == "vali"].set_index("sample_key").to_dict(orient="index")
        test_labels_by_key = config_labels_df[config_labels_df["split"] == "test"].set_index("sample_key").to_dict(orient="index")
        if vali_windows_df.empty or test_windows_df.empty:
            raise ValueError(f"config_name={config_name} 需要同时包含 vali/test")

        previous_completed_epochs = 0
        if resume_checkpoint is not None:
            assert_checkpoint_matches(checkpoint=resume_checkpoint, expected_signature=expected_signature, config_name=str(config_name))
            scaler = scaler_from_state(resume_checkpoint["scaler_state"])
            scaler_batches = int(resume_checkpoint.get("scaler_batches", 0))
            scaler_samples = int(resume_checkpoint.get("scaler_samples", 0))
            previous_completed_epochs = int(resume_checkpoint["completed_epochs"])
            embedding_dim = int(scaler.n_features_in_)
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "checkpoint_loaded",
                    "config_name": str(config_name),
                    "resume_checkpoint": str(args.resume_checkpoint),
                    "completed_epochs": previous_completed_epochs,
                    "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
                },
            )
        else:
            scaler = StandardScaler()
            scaler_batches = 0
            scaler_samples = 0
            for batch_manifest_df, embeddings, latency_rows in iter_online_embedding_batches(
                windows_df=vali_windows_df,
                data_config=data_config,
                vit_model=vit_model,
                args=args,
                device=device,
                dtype=dtype,
                period_candidate_values=period_candidate_values,
            ):
                scaler.partial_fit(embeddings)
                append_csv(output_dir / "online_embedding_manifest.csv", batch_manifest_df)
                append_latency(output_dir, latency_rows, "scaler_fit")
                embedding_dim = int(embeddings.shape[1])
                scaler_batches += 1
                scaler_samples += int(len(batch_manifest_df))
                total_embedding_batches += 1
            write_status(output_dir, {"status": "running", "phase": "scaler_fit_completed", "config_name": str(config_name), "scaler_samples": scaler_samples, "completed_epochs": 0, "latest_checkpoint_path": None})

        router = VisualMLPRouter(
            input_dim=int(scaler.n_features_in_),
            hidden_dim=int(args.hidden_dim),
            output_dim=len(MODEL_COLUMNS),
            dropout=float(args.dropout),
        ).to(device)
        optimizer = torch.optim.AdamW(router.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        if resume_checkpoint is not None:
            router.load_state_dict(resume_checkpoint["router_state_dict"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        class_weight = make_class_weight(
            [str(row["oracle_model"]) for row in vali_labels_by_key.values()],
            device=device,
        )

        epoch_summaries: List[Dict[str, float]] = [dict(row) for row in resume_checkpoint.get("epoch_summaries", [])] if resume_checkpoint is not None else []
        router.train()
        for local_epoch_idx in range(int(args.epochs)):
            global_epoch = previous_completed_epochs + local_epoch_idx + 1
            epoch_rows: List[Dict[str, float]] = []
            train_args.current_epoch = int(global_epoch)
            for epoch_batch_index, (batch_manifest_df, embeddings, latency_rows) in enumerate(iter_online_embedding_batches(
                windows_df=vali_windows_df,
                data_config=data_config,
                vit_model=vit_model,
                args=args,
                device=device,
                dtype=dtype,
                period_candidate_values=period_candidate_values,
            )):
                metrics = train_on_stream_batch(
                    router=router,
                    optimizer=optimizer,
                    scaler=scaler,
                    batch_manifest_df=batch_manifest_df,
                    embeddings=embeddings,
                    labels_by_key=vali_labels_by_key,
                    prediction_lookup=prediction_lookup,
                    prediction_index=prediction_index,
                    args=train_args,
                    device=device,
                    class_weight=class_weight,
                    training_batch_index=epoch_batch_index,
                )
                epoch_rows.append(metrics)
                append_latency(output_dir, latency_rows, f"train_epoch_{global_epoch}")
                total_embedding_batches += 1
                if total_embedding_batches % int(args.status_update_interval) == 0:
                    write_status(
                        output_dir,
                        {
                            "status": "running",
                            "phase": "training",
                            "config_name": str(config_name),
                            "epoch": int(global_epoch),
                            "current_epoch": int(global_epoch),
                            "completed_epochs": int(global_epoch - 1),
                            "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
                            "embedding_batches": total_embedding_batches,
                        },
                    )
            epoch_summary = {
                "epoch": float(global_epoch),
                "loss": float(np.nanmean([row["loss"] for row in epoch_rows])),
                "huber_loss": float(np.nanmean([row["huber_loss"] for row in epoch_rows])),
                "kl_loss": float(np.nanmean([row["kl_loss"] for row in epoch_rows])),
            }
            epoch_summaries.append(epoch_summary)
            latest_checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                config_name=str(config_name),
                router=router,
                optimizer=optimizer,
                scaler=scaler,
                completed_epochs=int(global_epoch),
                args=args,
                period_candidate_values=period_candidate_values,
                epoch_summaries=epoch_summaries,
                scaler_batches=scaler_batches,
                scaler_samples=scaler_samples,
            )
            write_status(
                output_dir,
                {
                    "status": "running",
                    "phase": "checkpoint_saved",
                    "config_name": str(config_name),
                    "epoch": int(global_epoch),
                    "current_epoch": int(global_epoch),
                    "completed_epochs": int(global_epoch),
                    "latest_checkpoint_path": str(latest_checkpoint_path),
                    "embedding_batches": total_embedding_batches,
                },
            )

        hard_rows_seen = 0
        test_batch_index = 0
        if not args.train_only:
            router.eval()
            for batch_manifest_df, embeddings, latency_rows in iter_online_embedding_batches(
                windows_df=test_windows_df,
                data_config=data_config,
                vit_model=vit_model,
                args=args,
                device=device,
                dtype=dtype,
                period_candidate_values=period_candidate_values,
            ):
                append_csv(output_dir / "online_embedding_manifest.csv", batch_manifest_df)
                append_latency(output_dir, latency_rows, "test_predict")
                pred_df = predict_stream_batch(
                    router=router,
                    scaler=scaler,
                    batch_manifest_df=batch_manifest_df,
                    embeddings=embeddings,
                    labels_by_key=test_labels_by_key,
                    router_name=router_name,
                    device=device,
                )
                append_csv(output_dir / "visual_router_predictions.csv", pred_df)
                if not args.skip_soft_fusion:
                    if prediction_index is None:
                        raise ValueError("soft fusion 需要 prediction_index")
                    soft_lookup = prediction_index.fetch_records(pred_df["sample_key"].astype(str).tolist())
                    soft_df = add_soft_fusion_metrics(pred_df, soft_lookup)
                    test_batch_index += 1
                    if bool(args.verify_evaluation_adapter):
                        verify_evaluation_adapter_bypass_batch(
                            pred_df=pred_df,
                            soft_df=soft_df,
                            prediction_lookup=soft_lookup,
                            output_dir=output_dir,
                            batch_index=test_batch_index,
                        )
                    append_csv(output_dir / "visual_router_soft_fusion_predictions.csv", soft_df)
                hard_rows_seen += int(len(pred_df))
                total_embedding_batches += 1
        config_metadata.append(
            {
                "config_name": str(config_name),
                "router_mode": args.router_mode,
                "vali_sample_count": int(len(vali_windows_df)),
                "test_sample_count": int(len(test_windows_df)),
                "scaler_batches": int(scaler_batches),
                "scaler_samples": int(scaler_samples),
                "test_predictions": int(hard_rows_seen),
                "embedding_dim": int(scaler.n_features_in_),
                "epochs_requested_this_run": int(args.epochs),
                "previous_completed_epochs": int(previous_completed_epochs),
                "completed_epochs": int(previous_completed_epochs + int(args.epochs)),
                "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
                "streaming_epoch_summaries": epoch_summaries,
                "label_counts": {str(k): int(v) for k, v in config_labels_df[config_labels_df["split"] == "vali"]["oracle_model"].value_counts().reindex(MODEL_COLUMNS, fill_value=0).items()},
            }
        )

    latency_path = output_dir / "online_embedding_latency_summary.csv"
    latency_df = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame(columns=["imageization_per_window_ms", "encoder_forward_per_window_ms"])
    embedding_metadata = {
        "embedding_version": f"{EMBEDDING_VERSION}_online_streaming",
        "sample_count": int(len(windows_df)),
        "encoder_name": args.encoder_name,
        "variant": args.variant,
        "pooling": args.pooling,
        "normalization_preset": args.normalization_preset,
        "input_mode": "direct_pixel_values_online_streaming",
        "processor_do_rescale": "not_used",
        "image_size": int(args.image_size),
        "norm_mode": args.norm_mode,
        "pixel_mode": args.pixel_mode,
        "clip": float(args.clip),
        "period_selection": args.period_selection,
        "period_candidates_arg": args.period_candidates,
        "period_candidates": period_candidate_values,
        "device": str(device),
        "forward_dtype": str(dtype).replace("torch.", ""),
        "embedding_storage": "batch_runtime_only_not_saved",
        "saved_dtype": "not_saved",
        "embedding_dim": int(embedding_dim or 0),
        "splits": sorted(windows_df["split"].unique().tolist()),
        "config_names": sorted(windows_df["config_name"].unique().tolist()),
        "input_exclusions": ["future_y", "expert_errors_as_input", "oracle_model_as_input", "oracle_value_as_input"],
        "latency_mean": {
            "imageization_per_window_ms": float(latency_df["imageization_per_window_ms"].mean()) if not latency_df.empty else None,
            "encoder_forward_per_window_ms": float(latency_df["encoder_forward_per_window_ms"].mean()) if not latency_df.empty else None,
        },
    }
    run_metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "router_version": router_name,
        "router_mode": args.router_mode,
        "labels_path": str(args.labels_path),
        "prediction_manifest_path": str(args.prediction_manifest_path),
        "config_path": str(args.config_path),
        "local_files_only": bool(args.local_files_only),
        "metric": args.metric,
        "model_columns": MODEL_COLUMNS,
        "training_split": "vali",
        "evaluation_split": "test",
        "device": str(device),
        "seed": int(args.seed),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "epochs_semantics": "additional_epochs_this_run",
        "batch_size": int(args.batch_size),
        "embedding_batch_size": int(args.embedding_batch_size),
        "stream_shard_index": int(args.stream_shard_index),
        "stream_shard_count": int(args.stream_shard_count),
        "max_samples_per_split": args.max_samples_per_split,
        "chunk_read_rows": int(args.chunk_read_rows),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "huber_beta": float(args.huber_beta),
        "kl_tau": float(args.kl_tau),
        "lambda_kl": float(args.lambda_kl),
        "soft_fusion_enabled": not bool(args.skip_soft_fusion),
        "train_only": bool(args.train_only),
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
        "checkpoint_dir": str(output_dir / "checkpoints"),
        "embedding_metadata": embedding_metadata,
        "config_metadata": config_metadata,
        "embedding_storage": "batch_runtime_only_not_saved",
        "pseudo_image_tensor_storage": "not_saved",
        "persistent_embedding_npy_written": False,
        "persistent_pseudo_image_tensor_written": False,
        "streaming_scaler_partial_fit": True,
        "input_exclusions": ["future_y_as_feature", "test_oracle_error_as_feature", "expert_error_as_feature"],
    }
    (output_dir / "visual_router_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "visual_router_online_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.train_only:
        completed_epochs = max(int(row["completed_epochs"]) for row in config_metadata) if config_metadata else 0
        write_status(
            output_dir,
            {
                "status": "completed",
                "phase": "train_only_done",
                "completed_epochs": completed_epochs,
                "current_epoch": completed_epochs,
                "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
            },
        )
        print(f"wrote train-only streaming checkpoint outputs to {output_dir}")
        print(f"latest_checkpoint_path={latest_checkpoint_path}")
    else:
        hard_summary_df, soft_summary_df, selected_counts_df, comparison_df = summarize_csv_outputs(output_dir, args.metric, args.labels_path)
        write_summary_md(
            output_dir=output_dir,
            hard_summary=hard_summary_df,
            soft_summary=soft_summary_df,
            selected_counts=selected_counts_df,
            comparison_df=comparison_df,
            metadata=run_metadata,
        )
        completed_epochs = max(int(row["completed_epochs"]) for row in config_metadata) if config_metadata else 0
        write_status(
            output_dir,
            {
                "status": "completed",
                "phase": "done",
                "router_predictions": int(hard_summary_df["sample_count"].sum()),
                "completed_epochs": completed_epochs,
                "current_epoch": completed_epochs,
                "latest_checkpoint_path": str(latest_checkpoint_path) if latest_checkpoint_path is not None else None,
            },
        )

        print(f"wrote streaming online visual router outputs to {output_dir}")
        print(hard_summary_df.to_string(index=False))
        if soft_summary_df is not None:
            print(soft_summary_df.to_string(index=False))
        pred_preview = pd.read_csv(output_dir / "visual_router_predictions.csv").head(int(args.print_rows))
        print(pred_preview.to_string(index=False))


if __name__ == "__main__":
    main()
