#!/usr/bin/env python3
"""
文件功能：
    基于 Stage 1 full-scale sample manifest shard 构建 TimeFuse-derived 单变量
    元特征 cache。

输入：
    - `sample_manifest_full_scale/sample_shards/*.csv` 中的单个 sample shard；
    - Quito evaluate config，用于按 sample manifest 重新加载历史窗口 x。

输出：
    - `feature_cache.csv`：每个 sample_key 一行，包含 17 维 TimeFuse-derived 单变量特征；
    - `metadata.json` / `status.json` / `main.log` / `latency_summary.csv`。

关键约束：
    - 特征只使用历史窗口 `x`，不读取未来 `y`、专家预测或 oracle label；
    - 该脚本是正式 full-scale 入口，不依赖 pilot 入口；
    - 支持 `--resume`，已完成 shard 会直接跳过，未完成 shard 会保留完整 item 组后续跑。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.signal import periodogram
from scipy.stats import entropy, kurtosis, skew
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, adfuller
from torch.utils.data import DataLoader, Subset


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
DEFAULT_CONFIG = (
    QUITO_DIR
    / "outputs"
    / "default_baseline"
    / "dlinear"
    / "96_48_S"
    / "seed_16"
    / "EVALUATE"
    / "ver_0"
    / "config.yaml"
)

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config import AutoConfig  # noqa: E402
from quito.config.training import ModeType, TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402


FEATURE_VERSION = "timefuse_single_variable_meta_v1"
FEATURE_TYPE = "numeric_structure_timefuse_single_variable"
EPS = 1e-12

FEATURE_COLUMNS = [
    "mean",
    "std",
    "min",
    "max",
    "skewness",
    "kurtosis",
    "autocorrelation_mean",
    "stationarity",
    "rate_of_change_mean",
    "rate_of_change_std",
    "autoreg_coef_mean",
    "residual_std_mean",
    "frequency_mean",
    "frequency_peak",
    "spectral_entropy",
    "spectral_skewness",
    "spectral_kurtosis",
]

FEATURE_CACHE_COLUMNS = [
    "feature_version",
    "sample_key",
    "config_name",
    "split",
    "dataset_name",
    "item_id",
    "channel_id",
    "window_index",
    "history_length",
    "pred_length",
    "feature_type",
    "feature_dim",
    *FEATURE_COLUMNS,
]


def now_token() -> str:
    """函数功能：生成默认输出目录时间戳。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入日志和 metadata 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析单 shard TimeFuse feature cache 构建参数。"""
    parser = argparse.ArgumentParser(
        description="Build TimeFuse-derived single-variable feature cache from a Stage 1 sample manifest shard."
    )
    parser.add_argument("--sample-manifest-path", type=Path, required=True, help="单个 sample manifest shard CSV。")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config 路径。")
    parser.add_argument("--output-root", type=Path, default=DATA2_RUN_OUTPUT_ROOT, help="默认输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录。")
    parser.add_argument("--resume", action="store_true", help="跳过已完成 shard；未完成 shard 保留完整 item 组后续跑。")
    parser.add_argument("--batch-size", type=int, default=512, help="DataLoader batch size。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers。")
    parser.add_argument("--max-samples", type=int, default=None, help="只处理前 N 个 sample，用于 smoke/dry-run。")
    parser.add_argument("--print-rows", type=int, default=8, help="运行结束时打印多少行预览。")
    return parser.parse_args()


def mode_from_split(split: str) -> ModeType:
    """函数功能：把 Stage 1 split 名称映射到 Quito ModeType。"""
    if split == "vali":
        return ModeType.VALID
    if split == "test":
        return ModeType.TEST
    raise ValueError(f"未知 split：{split}")


def load_data_config(config_path: Path):
    """函数功能：读取 Quito config，并返回 data_config。"""
    config = OmegaConf.load(str(config_path))
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        rank=-1,
        world_size=-1,
        local_rank=-1,
    )
    del model_config, training_config
    return data_config


def append_main_log(output_dir: Path, message: str) -> None:
    """函数功能：向当前 shard 的主日志追加一行。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出可被 launcher/handoff 读取的状态文件。"""
    status = dict(payload)
    status["updated_at"] = display_time()
    status["output_dir"] = str(output_dir)
    status["pid"] = int(os.getpid())
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_frame(path: Path, frame: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> None:
    """函数功能：追加写 CSV；首批自动写表头。"""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    frame.to_csv(path, mode="a", header=write_header, index=False)


def load_sample_manifest(sample_manifest_path: Path, max_samples: Optional[int]) -> pd.DataFrame:
    """函数功能：读取并校验 sample manifest shard。"""
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"找不到 sample manifest：{sample_manifest_path}")
    sample_df = pd.read_csv(sample_manifest_path)
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("--max-samples 必须为正整数")
        sample_df = sample_df.head(int(max_samples)).copy()
    required = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "history_length",
        "pred_length",
    }
    missing = sorted(required.difference(sample_df.columns))
    if missing:
        raise ValueError(f"sample manifest 缺少字段：{missing}")
    if sample_df.empty:
        raise ValueError("sample manifest 为空")
    if sample_df["sample_key"].duplicated().any():
        dup = sample_df.loc[sample_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"sample manifest 中 sample_key 重复，示例：{dup}")

    expected_keys = []
    for row in sample_df.itertuples(index=False):
        key = PredictionCacheKey(
            config_name=str(row.config_name),
            split=str(row.split),
            dataset_name=str(row.dataset_name),
            item_id=int(row.item_id),
            channel_id=int(row.channel_id),
            window_index=int(row.window_index),
        )
        expected_keys.append(key.as_string())
    bad = sample_df["sample_key"].astype(str).to_numpy() != np.asarray(expected_keys, dtype=object)
    if bool(np.any(bad)):
        first_bad = sample_df.loc[bad].iloc[0]
        raise ValueError(f"sample_key 与元信息不一致：{first_bad['sample_key']}")

    return sample_df.sort_values(
        ["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    ).reset_index(drop=True)


def make_item_dataset_view(dataset, item_id: int):
    """
    函数功能：
        从已加载 Quito dataset 构造单个 item 的轻量视图。

    设计说明：
        只替换 `data/id_mask`，避免每个 item 重新读取底层 parquet；该口径与正式
        prediction cache builder 保持一致。
    """
    if getattr(dataset, "id_mask", None) is None:
        raise ValueError(f"dataset={getattr(dataset, 'name', '<unknown>')} 缺少 id_mask")
    item_dataset = copy.copy(dataset)
    mask = dataset.id_mask == int(item_id)
    item_dataset.data = dataset.data[mask].reshape(-1, dataset.data.shape[1], dataset.data.shape[-1])
    item_dataset.id_mask = dataset.id_mask[mask].reshape(-1, dataset.id_mask.shape[1], dataset.id_mask.shape[-1])
    return item_dataset


def build_required_index(sample_df: pd.DataFrame) -> Dict[Tuple[str, str, int], Dict[Tuple[int, int], Dict[str, object]]]:
    """
    函数功能：
        将 sample manifest 整理成 `(split, dataset_name, item_id) -> (channel, window) -> row`。
    """
    required: Dict[Tuple[str, str, int], Dict[Tuple[int, int], Dict[str, object]]] = {}
    for row in sample_df.itertuples(index=False):
        group_key = (str(row.split), str(row.dataset_name), int(row.item_id))
        pair = (int(row.channel_id), int(row.window_index))
        required.setdefault(group_key, {})[pair] = {
            "sample_key": str(row.sample_key),
            "config_name": str(row.config_name),
            "split": str(row.split),
            "dataset_name": str(row.dataset_name),
            "item_id": int(row.item_id),
            "channel_id": int(row.channel_id),
            "window_index": int(row.window_index),
            "history_length": int(row.history_length),
            "pred_length": int(row.pred_length),
        }
    return required


def safe_float(value: float) -> float:
    """函数功能：把 NaN/Inf 统一替换为 0，避免 downstream scaler 传播非法值。"""
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def nanmean_or_zero(values: np.ndarray) -> float:
    """函数功能：计算有限值均值；全非法时返回 0。"""
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return 0.0
    return safe_float(np.mean(values[finite_mask]))


def nanstd_or_zero(values: np.ndarray) -> float:
    """函数功能：计算有限值标准差；全非法时返回 0。"""
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return 0.0
    return safe_float(np.std(values[finite_mask]))


def safe_skew(values: np.ndarray) -> float:
    """函数功能：静默计算 skew；常数序列或数值病态时返回 0。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return safe_float(skew(values))
    except Exception:
        return 0.0


def safe_kurtosis(values: np.ndarray) -> float:
    """函数功能：静默计算 kurtosis；常数序列或数值病态时返回 0。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return safe_float(kurtosis(values))
    except Exception:
        return 0.0


def extract_timefuse_single_variable_features(x_window: np.ndarray) -> Dict[str, float]:
    """
    函数功能：
        从单变量历史窗口 x 中提取 17 个 TimeFuse-derived 元特征。

    输入：
        x_window: Quito train-normalized 后的历史窗口，形状通常为 `[history_length, 1]`。

    输出：
        `FEATURE_COLUMNS` 定义的 17 维数值特征。

    关键约束：
        本函数只接收历史 `x_window`，不会读取未来 y、专家预测或 oracle label。
    """
    series = np.asarray(x_window, dtype=np.float64).reshape(-1)
    if series.size < 3:
        raise ValueError(f"历史窗口长度过短，无法提取 TimeFuse 元特征：{series.size}")

    features: Dict[str, float] = {
        "mean": safe_float(np.mean(series)),
        "std": safe_float(np.std(series)),
        "min": safe_float(np.min(series)),
        "max": safe_float(np.max(series)),
        "skewness": safe_skew(series),
        "kurtosis": safe_kurtosis(series),
    }

    # 与 TimeFuse/meta_feature.py 保持一致：单变量时取 lag-1 ACF。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acf_values = acf(series, nlags=10, fft=True)
        features["autocorrelation_mean"] = safe_float(acf_values[1])
    except Exception:
        features["autocorrelation_mean"] = 0.0

    # TimeFuse 的 stationarity 是 ADF p-value < 0.05；单变量时结果为 0/1。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(series)
        features["stationarity"] = safe_float(float(adf_result[1] < 0.05))
    except Exception:
        features["stationarity"] = 0.0

    safe_denominator = np.where(series[:-1] == 0, np.nan, series[:-1])
    rate_of_change = np.diff(series) / safe_denominator
    features["rate_of_change_mean"] = nanmean_or_zero(rate_of_change)
    features["rate_of_change_std"] = nanstd_or_zero(rate_of_change)

    # TimeFuse landmarker 使用 AR(1) 系数和残差标准差；常数序列或拟合失败时置零。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autoreg_model = AutoReg(series, lags=1).fit()
        features["autoreg_coef_mean"] = safe_float(autoreg_model.params[1])
        features["residual_std_mean"] = safe_float(np.std(autoreg_model.resid))
    except Exception:
        features["autoreg_coef_mean"] = 0.0
        features["residual_std_mean"] = 0.0

    freqs, psd = periodogram(series)
    psd = np.asarray(psd, dtype=np.float64)
    features["frequency_mean"] = safe_float(np.mean(psd))
    features["frequency_peak"] = safe_float(freqs[int(np.argmax(psd))]) if psd.size else 0.0
    features["spectral_entropy"] = safe_float(entropy(psd + EPS))
    features["spectral_skewness"] = safe_skew(psd)
    features["spectral_kurtosis"] = safe_kurtosis(psd)

    return {name: safe_float(features.get(name, 0.0)) for name in FEATURE_COLUMNS}


def required_group_counts(sample_df: pd.DataFrame) -> Dict[Tuple[str, str, int], int]:
    """函数功能：统计每个 item 组在当前 shard 中应包含多少 sample_key。"""
    counts = (
        sample_df.groupby(["split", "dataset_name", "item_id"], sort=False)
        .size()
        .reset_index(name="rows")
    )
    return {
        (str(row.split), str(row.dataset_name), int(row.item_id)): int(row.rows)
        for row in counts.itertuples(index=False)
    }


def completed_groups_from_existing_cache(
    feature_cache_path: Path,
    sample_df: pd.DataFrame,
    *,
    resume: bool,
) -> Tuple[set[Tuple[str, str, int]], int]:
    """
    函数功能：
        读取已有 feature cache，保留完整 item 组并返回可跳过的 group。

    说明：
        full-scale shard 可能在 item 中途失败。为了避免重复 sample_key，本函数会在
        `--resume` 时把不完整 item 组从 CSV 中剪掉，只保留已经完整写出的组。
    """
    if not resume or not feature_cache_path.exists():
        return set(), 0

    existing_df = pd.read_csv(feature_cache_path)
    if existing_df.empty:
        feature_cache_path.unlink()
        return set(), 0
    missing = sorted(set(FEATURE_CACHE_COLUMNS).difference(existing_df.columns))
    if missing:
        raise ValueError(f"已有 feature cache 缺少字段，不能安全 resume：{missing}")
    if existing_df["sample_key"].duplicated().any():
        dup = existing_df.loc[existing_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"已有 feature cache 中 sample_key 重复，不能安全 resume：{dup}")

    expected_keys = set(sample_df["sample_key"].astype(str))
    extra_keys = sorted(set(existing_df["sample_key"].astype(str)) - expected_keys)
    if extra_keys:
        raise ValueError(f"已有 feature cache 含有不属于当前 sample manifest 的 key，示例：{extra_keys[:10]}")

    numeric_values = existing_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("已有 feature cache 中存在 NaN/Inf，不能安全 resume")

    expected_counts = required_group_counts(sample_df)
    existing_counts = (
        existing_df.groupby(["split", "dataset_name", "item_id"], sort=False)
        .size()
        .reset_index(name="rows")
    )
    completed_groups = set()
    for row in existing_counts.itertuples(index=False):
        key = (str(row.split), str(row.dataset_name), int(row.item_id))
        if int(row.rows) == int(expected_counts.get(key, -1)):
            completed_groups.add(key)

    keep_mask = [
        (str(row.split), str(row.dataset_name), int(row.item_id)) in completed_groups
        for row in existing_df.itertuples(index=False)
    ]
    kept_df = existing_df.loc[keep_mask, FEATURE_CACHE_COLUMNS].copy()
    if len(kept_df) != len(existing_df):
        backup_path = feature_cache_path.with_suffix(".before_resume_prune.csv")
        existing_df.to_csv(backup_path, index=False)
        if kept_df.empty:
            feature_cache_path.unlink()
        else:
            kept_df.to_csv(feature_cache_path, index=False)
    return completed_groups, int(len(kept_df))


def final_validate_and_sort(output_dir: Path, sample_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：对完成后的 feature cache 做全量一致性校验并排序落盘。"""
    feature_cache_path = output_dir / "feature_cache.csv"
    if not feature_cache_path.exists():
        raise FileNotFoundError(f"缺少 feature_cache.csv：{feature_cache_path}")
    feature_df = pd.read_csv(feature_cache_path)
    missing = sorted(set(FEATURE_CACHE_COLUMNS).difference(feature_df.columns))
    if missing:
        raise ValueError(f"feature cache 缺少字段：{missing}")
    if len(feature_df) != len(sample_df):
        raise ValueError(f"feature cache 行数不一致：actual={len(feature_df)} expected={len(sample_df)}")
    if feature_df["sample_key"].duplicated().any():
        dup = feature_df.loc[feature_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"feature cache 中 sample_key 重复，示例：{dup}")
    expected_keys = set(sample_df["sample_key"].astype(str))
    actual_keys = set(feature_df["sample_key"].astype(str))
    if actual_keys != expected_keys:
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        raise ValueError(f"feature cache 覆盖不一致：missing={missing_keys[:10]} extra={extra_keys[:10]}")
    numeric_values = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("feature cache 中存在 NaN/Inf")
    if set(feature_df["feature_version"].astype(str).unique()) != {FEATURE_VERSION}:
        raise ValueError("feature_version 不一致")
    if set(feature_df["feature_type"].astype(str).unique()) != {FEATURE_TYPE}:
        raise ValueError("feature_type 不一致")
    if set(feature_df["feature_dim"].astype(int).unique()) != {len(FEATURE_COLUMNS)}:
        raise ValueError("feature_dim 与 FEATURE_COLUMNS 数量不一致")

    expected_key_by_row = []
    for row in feature_df.itertuples(index=False):
        expected_key_by_row.append(
            PredictionCacheKey(
                config_name=str(row.config_name),
                split=str(row.split),
                dataset_name=str(row.dataset_name),
                item_id=int(row.item_id),
                channel_id=int(row.channel_id),
                window_index=int(row.window_index),
            ).as_string()
        )
    if not (feature_df["sample_key"].astype(str).to_numpy() == np.asarray(expected_key_by_row, dtype=object)).all():
        raise ValueError("feature cache 中存在 sample_key 与稳定元信息不一致的记录")

    feature_df = feature_df.sort_values(
        ["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    ).reset_index(drop=True)
    feature_df.loc[:, FEATURE_CACHE_COLUMNS].to_csv(feature_cache_path, index=False)
    return feature_df


def build_feature_cache(
    *,
    sample_df: pd.DataFrame,
    data_config,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    resume: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    函数功能：
        遍历 sample manifest 指定窗口，重新加载历史 x 并写出 feature cache。
    """
    required_index = build_required_index(sample_df)
    feature_cache_path = output_dir / "feature_cache.csv"
    completed_groups, resumed_rows = completed_groups_from_existing_cache(
        feature_cache_path,
        sample_df,
        resume=resume,
    )
    if completed_groups:
        append_main_log(output_dir, f"resume keep completed groups={len(completed_groups)} rows={resumed_rows}")

    rows_written = int(resumed_rows)
    latency_rows: List[Dict[str, object]] = []
    total_groups = int(len(required_index))
    processed_groups = int(len(completed_groups))
    start_time = time.perf_counter()

    write_status(
        output_dir,
        {
            "status": "running",
            "phase": "building_features",
            "sample_count": int(len(sample_df)),
            "rows_written": int(rows_written),
            "processed_group_count": int(processed_groups),
            "total_group_count": int(total_groups),
            "feature_version": FEATURE_VERSION,
            "feature_dim": len(FEATURE_COLUMNS),
        },
    )

    for split in sorted(sample_df["split"].astype(str).unique()):
        append_main_log(output_dir, f"load datasets for split={split}")
        datasets = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=mode_from_split(split),
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
            if not item_ids:
                continue

            for item_id in item_ids:
                group_key = (str(split), str(dataset_name), int(item_id))
                if group_key in completed_groups:
                    continue

                item_dataset = make_item_dataset_view(dataset, int(item_id))
                channel_count = int(item_dataset.data.shape[0])
                len_per_channel = len(item_dataset) // channel_count
                required_for_item = required_index[group_key]
                required_entries = sorted(
                    [
                        (int(channel_id), int(window_index), dict(row_info))
                        for (channel_id, window_index), row_info in required_for_item.items()
                    ],
                    key=lambda value: (value[0], value[1], value[2]["sample_key"]),
                )
                required_indices = [
                    channel_id * len_per_channel + window_index
                    for channel_id, window_index, _ in required_entries
                ]
                for global_index in required_indices:
                    if global_index < 0 or global_index >= len(item_dataset):
                        raise ValueError(
                            f"sample manifest 中存在越界窗口：split={split} dataset={dataset_name} "
                            f"item_id={item_id} global_index={global_index} len_item_dataset={len(item_dataset)}"
                        )

                dataloader = DataLoader(
                    Subset(item_dataset, required_indices),
                    batch_size=int(batch_size),
                    shuffle=False,
                    num_workers=int(num_workers),
                )
                group_rows: List[Dict[str, object]] = []
                row_cursor = 0
                group_start = time.perf_counter()
                for batch in dataloader:
                    # 这里只读取历史窗口 x；batch 中即使存在 y，也不访问、不写入、不作为特征输入。
                    x_batch = batch["x"].detach().cpu().numpy()
                    batch_entries = required_entries[row_cursor : row_cursor + int(x_batch.shape[0])]
                    row_cursor += int(x_batch.shape[0])
                    for row_in_batch in range(int(x_batch.shape[0])):
                        channel_id, window_index, row_info = batch_entries[row_in_batch]
                        key = PredictionCacheKey(
                            config_name=str(row_info["config_name"]),
                            split=str(row_info["split"]),
                            dataset_name=str(row_info["dataset_name"]),
                            item_id=int(row_info["item_id"]),
                            channel_id=int(channel_id),
                            window_index=int(window_index),
                        )
                        if key.as_string() != str(row_info["sample_key"]):
                            raise ValueError(f"sample_key 与元信息不一致：{row_info['sample_key']} vs {key.as_string()}")
                        feature_values = extract_timefuse_single_variable_features(x_batch[row_in_batch])
                        row = {
                            "feature_version": FEATURE_VERSION,
                            "sample_key": str(row_info["sample_key"]),
                            "config_name": str(row_info["config_name"]),
                            "split": str(row_info["split"]),
                            "dataset_name": str(row_info["dataset_name"]),
                            "item_id": int(row_info["item_id"]),
                            "channel_id": int(channel_id),
                            "window_index": int(window_index),
                            "history_length": int(row_info["history_length"]),
                            "pred_length": int(row_info["pred_length"]),
                            "feature_type": FEATURE_TYPE,
                            "feature_dim": len(FEATURE_COLUMNS),
                        }
                        row.update(feature_values)
                        group_rows.append(row)

                if len(group_rows) != len(required_entries):
                    raise ValueError(
                        f"feature rows 不完整：group={group_key} actual={len(group_rows)} expected={len(required_entries)}"
                    )
                group_df = pd.DataFrame(group_rows)
                append_frame(feature_cache_path, group_df, columns=FEATURE_CACHE_COLUMNS)
                rows_written += int(len(group_df))
                processed_groups += 1

                elapsed = time.perf_counter() - group_start
                latency_rows.append(
                    {
                        "split": split,
                        "dataset_name": dataset_name,
                        "item_id": int(item_id),
                        "sample_count": int(len(group_df)),
                        "elapsed_seconds": float(elapsed),
                        "seconds_per_sample": float(elapsed / max(1, len(group_df))),
                    }
                )

                if processed_groups % 25 == 0 or rows_written == len(sample_df):
                    total_elapsed = time.perf_counter() - start_time
                    write_status(
                        output_dir,
                        {
                            "status": "running",
                            "phase": "building_features",
                            "sample_count": int(len(sample_df)),
                            "rows_written": int(rows_written),
                            "processed_group_count": int(processed_groups),
                            "total_group_count": int(total_groups),
                            "elapsed_seconds": float(total_elapsed),
                            "rows_per_second": float((rows_written - resumed_rows) / max(total_elapsed, 1e-9)),
                            "feature_version": FEATURE_VERSION,
                            "feature_dim": len(FEATURE_COLUMNS),
                        },
                    )

    latency_df = pd.DataFrame(latency_rows)
    if not latency_df.empty:
        append_frame(output_dir / "latency_summary.csv", latency_df)
    feature_df = final_validate_and_sort(output_dir, sample_df)
    total_elapsed = time.perf_counter() - start_time
    metadata = {
        "generated_at": display_time(),
        "sample_manifest_path": None,
        "sample_count": int(len(feature_df)),
        "feature_version": FEATURE_VERSION,
        "feature_type": FEATURE_TYPE,
        "feature_columns": FEATURE_COLUMNS,
        "feature_dim": len(FEATURE_COLUMNS),
        "history_lengths": sorted(int(value) for value in feature_df["history_length"].unique().tolist()),
        "pred_lengths": sorted(int(value) for value in feature_df["pred_length"].unique().tolist()),
        "splits": sorted(str(value) for value in feature_df["split"].unique().tolist()),
        "config_names": sorted(str(value) for value in feature_df["config_name"].unique().tolist()),
        "elapsed_seconds": float(total_elapsed),
        "rows_per_second": float((len(feature_df) - resumed_rows) / max(total_elapsed, 1e-9)),
        "resumed_rows": int(resumed_rows),
        "input_exclusions": [
            "future_y",
            "expert_predictions",
            "oracle_label",
            "prediction_cache_manifest",
            "tsf_enrichment_label_file",
        ],
        "gpu_strategy": "not_used_cpu_only_statsmodels_scipy_numpy_features",
        "timefuse_source": "TimeFuse/meta_feature.py single-variable subset; multivariate covariance/spectral_variation removed",
        "excluded_timefuse_features": [
            "spectral_variation",
            "covariance_mean",
            "covariance_max",
            "covariance_min",
            "covariance_std",
        ],
    }
    return feature_df, latency_df, metadata


def write_summary(output_dir: Path, feature_df: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    """函数功能：写出 shard 级 Markdown 摘要。"""
    counts = (
        feature_df.groupby(["config_name", "split", "dataset_name"])
        .size()
        .reset_index(name="rows")
    )

    def frame_to_markdown(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_无记录_"
        lines = [
            "| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        for row in frame.astype(str).values.tolist():
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    lines = [
        "# Stage 1 TimeFuse Feature Cache Shard",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 输入",
        "",
        f"- sample_manifest: `{metadata['sample_manifest_path']}`",
        "- 特征只使用历史窗口 `x`；不读取未来 `y`、专家预测、oracle label。",
        "",
        "## 输出",
        "",
        f"- feature_cache.csv: `{output_dir / 'feature_cache.csv'}`",
        f"- metadata.json: `{output_dir / 'metadata.json'}`",
        f"- status.json: `{output_dir / 'status.json'}`",
        f"- sample_count: `{metadata['sample_count']}`",
        f"- feature_dim: `{metadata['feature_dim']}`",
        f"- rows_per_second: `{metadata['rows_per_second']:.3f}`",
        "",
        "## 覆盖统计",
        "",
        frame_to_markdown(counts),
        "",
        "## 特征列",
        "",
        "```text",
        "\n".join(FEATURE_COLUMNS),
        "```",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def completed_shard_is_valid(output_dir: Path, expected_sample_count: int) -> bool:
    """
    函数功能：
        判断 `--resume` 场景下当前 shard 是否已经完整完成。
    """
    status_path = output_dir / "status.json"
    feature_cache_path = output_dir / "feature_cache.csv"
    if not status_path.exists() or not feature_cache_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if status.get("status") != "completed":
        return False
    if int(status.get("sample_count", -1)) != int(expected_sample_count):
        return False
    try:
        row_count = sum(1 for _ in feature_cache_path.open("r", encoding="utf-8")) - 1
    except Exception:
        return False
    return int(row_count) == int(expected_sample_count)


def main() -> None:
    """函数功能：执行单个 sample shard 的 TimeFuse feature cache 构建。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_stage1_timefuse_feature_cache_shard"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "main.log").exists():
        (output_dir / "main.log").write_text("", encoding="utf-8")

    sample_df = load_sample_manifest(args.sample_manifest_path, args.max_samples)
    if args.resume and completed_shard_is_valid(output_dir, expected_sample_count=len(sample_df)):
        append_main_log(output_dir, "skip completed shard because --resume is set")
        print(f"skip completed shard: {output_dir}")
        return

    append_main_log(output_dir, "start TimeFuse feature cache shard build")
    write_status(
        output_dir,
        {
            "status": "running",
            "phase": "init",
            "sample_manifest_path": str(args.sample_manifest_path),
            "sample_count": int(len(sample_df)),
            "feature_version": FEATURE_VERSION,
            "feature_dim": len(FEATURE_COLUMNS),
        },
    )
    try:
        data_config = load_data_config(args.config_path)
        feature_df, latency_df, metadata = build_feature_cache(
            sample_df=sample_df,
            data_config=data_config,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            resume=bool(args.resume),
        )
        metadata = dict(metadata)
        metadata.update(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "sample_manifest_path": str(args.sample_manifest_path),
                "config_path": str(args.config_path),
                "batch_size": int(args.batch_size),
                "num_workers": int(args.num_workers),
                "max_samples": args.max_samples,
                "latency_row_count": int(len(latency_df)),
            }
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_summary(output_dir, feature_df, metadata)
        write_status(output_dir, metadata)
        append_main_log(output_dir, "completed TimeFuse feature cache shard build")
    except Exception as exc:
        write_status(
            output_dir,
            {
                "status": "failed",
                "phase": "error",
                "sample_manifest_path": str(args.sample_manifest_path),
                "sample_count": int(len(sample_df)),
                "error": repr(exc),
            },
        )
        append_main_log(output_dir, f"failed TimeFuse feature cache shard build: {repr(exc)}")
        raise

    preview_cols = ["sample_key", "feature_dim", *FEATURE_COLUMNS[:6]]
    print(f"wrote TimeFuse feature cache shard to {output_dir}")
    print(f"sample_count={len(feature_df)} feature_dim={len(FEATURE_COLUMNS)}")
    print(feature_df[preview_cols].head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
