#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router Stage 1 生成小规模 TimeFuse-derived 单变量元特征 cache。

Pilot 限制：
    - 默认只覆盖已有 96_48_S prediction cache pilot 的 oracle label 样本；
    - 只从 Quito train-normalized 后的历史窗口 x 提取特征；
    - 不读取未来 y、不读取专家误差作为特征、不使用 oracle label 计算特征；
    - 该脚本用于构造非视觉数值 baseline，不作为结构特征工程主线。

特征口径：
    借鉴 TimeFuse/meta_feature.py，但删除多变量/跨变量特征：
    spectral_variation、covariance_mean、covariance_max、covariance_min、covariance_std。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.signal import periodogram
from scipy.stats import entropy, kurtosis, skew
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, adfuller
from torch.utils.data import DataLoader


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

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

DEFAULT_CONFIG = (
    WORKSPACE
    / "quito"
    / "outputs"
    / "default_baseline"
    / "dlinear"
    / "96_48_S"
    / "seed_16"
    / "EVALUATE"
    / "ver_0"
    / "config.yaml"
)

DEFAULT_LABELS_PATH = (
    RUN_OUTPUT_ROOT
    / "2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot"
    / "window_oracle_labels_with_tsf_cell.csv"
)

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


def now_token() -> str:
    """函数功能：生成 run 目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 TimeFuse-derived 单变量元特征 cache pilot 参数。"""
    parser = argparse.ArgumentParser(
        description="Build a TimeFuse-derived single-variable meta-feature cache for Stage 1 pilot."
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="oracle labels CSV 路径；默认使用 96_48_S 扩大版五专家 pilot。",
    )
    parser.add_argument(
        "--metric",
        default="mae",
        choices=["mae", "mse"],
        help="用于确定需要覆盖哪些 sample_key 的 oracle label 口径；默认 mae。",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Quito evaluate config 路径；只用于复用同口径 data_config 加载历史窗口 x。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入 experiment_logs/run_outputs 下的时间戳目录。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="DataLoader batch size。",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=20,
        help="运行结束时最多打印的 feature cache 行数。",
    )
    return parser.parse_args()


def mode_from_split(split: str) -> ModeType:
    """函数功能：把 Stage 1 split 名称映射到 Quito ModeType。"""
    if split == "vali":
        return ModeType.VALID
    if split == "test":
        return ModeType.TEST
    raise ValueError(f"未知 split：{split}")


def load_data_config(config_path: Path):
    """函数功能：读取 Quito evaluate config，并返回数据配置对象。"""
    config = OmegaConf.load(config_path)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        rank=-1,
        world_size=-1,
        local_rank=-1,
    )
    del model_config, training_config
    return data_config


def load_required_windows(labels_path: Path, metric: str) -> pd.DataFrame:
    """
    函数功能：
        读取 oracle labels，筛出指定 metric 的唯一 sample_key 元信息。

    约束：
        这里不读取 oracle_model 或专家误差作为特征，只用标签文件提供待覆盖窗口清单。
    """
    labels_df = pd.read_csv(labels_path)
    required_cols = {
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "metric",
    }
    missing_cols = sorted(required_cols.difference(labels_df.columns))
    if missing_cols:
        raise ValueError(f"oracle labels 缺少字段：{missing_cols}")

    windows_df = labels_df.loc[labels_df["metric"] == metric, sorted(required_cols - {"metric"})].copy()
    windows_df = windows_df.drop_duplicates().reset_index(drop=True)
    if windows_df.empty:
        raise ValueError(f"{labels_path} 中没有 metric={metric} 的样本")
    duplicated = windows_df["sample_key"].duplicated()
    if duplicated.any():
        dup_keys = windows_df.loc[duplicated, "sample_key"].head(10).tolist()
        raise ValueError(f"metric={metric} 的 sample_key 不唯一，示例：{dup_keys}")
    return windows_df


def build_required_index(windows_df: pd.DataFrame) -> Mapping[Tuple[str, str, int], Set[Tuple[int, int, str]]]:
    """
    函数功能：
        将待覆盖窗口整理成便于遍历 Quito dataset 的索引。

    返回：
        key 为 (split, dataset_name, item_id)，value 为 {(channel_id, window_index, sample_key)}。
    """
    required: Dict[Tuple[str, str, int], Set[Tuple[int, int, str]]] = {}
    for row in windows_df.itertuples(index=False):
        group_key = (str(row.split), str(row.dataset_name), int(row.item_id))
        required.setdefault(group_key, set()).add((int(row.channel_id), int(row.window_index), str(row.sample_key)))
    return required


def safe_float(value: float) -> float:
    """函数功能：将 NaN/Inf 统一替换为 0，避免 feature cache 传播非法数值。"""
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def nanmean_or_zero(values: np.ndarray) -> float:
    """函数功能：计算 nanmean；若全为 NaN 或结果非法则返回 0。"""
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return 0.0
    return safe_float(np.mean(values[finite_mask]))


def nanstd_or_zero(values: np.ndarray) -> float:
    """函数功能：计算 nanstd；若全为 NaN 或结果非法则返回 0。"""
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return 0.0
    return safe_float(np.std(values[finite_mask]))


def extract_timefuse_single_variable_features(x_window: np.ndarray) -> Dict[str, float]:
    """
    函数功能：
        从单变量历史窗口中提取 TimeFuse-derived 元特征。

    输入：
        x_window: Quito train-normalized 后的历史窗口，形状可为 [history_length] 或
            [history_length, 1]。

    输出：
        FEATURE_COLUMNS 定义的 17 个单变量元特征。

    关键约束：
        只使用历史 x，不使用未来 y、专家预测误差或 oracle label。
    """
    series = np.asarray(x_window, dtype=np.float64).reshape(-1)
    if series.size < 3:
        raise ValueError(f"历史窗口长度过短，无法提取 TimeFuse 元特征：{series.size}")

    features: Dict[str, float] = {
        "mean": safe_float(np.mean(series)),
        "std": safe_float(np.std(series)),
        "min": safe_float(np.min(series)),
        "max": safe_float(np.max(series)),
        "skewness": safe_float(skew(series)),
        "kurtosis": safe_float(kurtosis(series)),
    }

    # TimeFuse 对每个变量算 lag-1 ACF 后取均值；当前 Stage 1 S 口径只有一个变量。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acf_values = acf(series, nlags=10, fft=True)
        features["autocorrelation_mean"] = safe_float(acf_values[1])
    except Exception:
        features["autocorrelation_mean"] = 0.0

    # TimeFuse stationarity 是 ADF p-value < 0.05 的变量比例；单变量时为 0/1。
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

    # TimeFuse 的 landmarker 是 AR(1) 系数和残差标准差；常数序列或拟合失败时置零。
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
    features["spectral_skewness"] = safe_float(skew(psd))
    features["spectral_kurtosis"] = safe_float(kurtosis(psd))

    return {name: safe_float(features.get(name, 0.0)) for name in FEATURE_COLUMNS}


def validate_feature_cache(feature_df: pd.DataFrame, windows_df: pd.DataFrame) -> None:
    """
    函数功能：
        校验 feature cache 与 oracle label 样本清单严格对齐。
    """
    metadata_cols = {
        "feature_version",
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "history_length",
        "feature_type",
        "feature_dim",
    }
    required_cols = metadata_cols.union(FEATURE_COLUMNS)
    missing_cols = sorted(required_cols.difference(feature_df.columns))
    if missing_cols:
        raise ValueError(f"feature cache 缺少字段：{missing_cols}")

    if feature_df["sample_key"].duplicated().any():
        dup_keys = feature_df.loc[feature_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"feature cache 中 sample_key 重复，示例：{dup_keys}")

    expected_keys = set(windows_df["sample_key"])
    actual_keys = set(feature_df["sample_key"])
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys or extra_keys:
        raise ValueError(
            f"feature cache 覆盖不一致：missing={missing_keys[:10]} extra={extra_keys[:10]}"
        )

    expected_key_strings = feature_df.apply(
        lambda row: PredictionCacheKey(
            config_name=str(row["config_name"]),
            split=str(row["split"]),
            dataset_name=str(row["dataset_name"]),
            item_id=int(row["item_id"]),
            channel_id=int(row["channel_id"]),
            window_index=int(row["window_index"]),
        ).as_string(),
        axis=1,
    )
    if not (expected_key_strings == feature_df["sample_key"]).all():
        bad_rows = feature_df.loc[expected_key_strings != feature_df["sample_key"], "sample_key"].head(10).tolist()
        raise ValueError(f"sample_key 与元信息不一致，示例：{bad_rows}")

    numeric_values = feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_values).all():
        raise ValueError("feature cache 中存在 NaN 或 Inf")

    if set(feature_df["feature_version"].unique()) != {FEATURE_VERSION}:
        raise ValueError("feature_version 不一致")
    if set(feature_df["feature_type"].unique()) != {FEATURE_TYPE}:
        raise ValueError("feature_type 不一致")
    if set(feature_df["feature_dim"].unique()) != {len(FEATURE_COLUMNS)}:
        raise ValueError("feature_dim 与 FEATURE_COLUMNS 数量不一致")


def build_feature_rows(
    *,
    data_config,
    windows_df: pd.DataFrame,
    batch_size: int,
) -> List[Dict[str, object]]:
    """
    函数功能：
        遍历 Quito vali/test 数据集，为 oracle label 指定的 sample_key 提取历史 x 元特征。
    """
    required_index = build_required_index(windows_df)
    config_by_key = dict(zip(windows_df["sample_key"], windows_df["config_name"]))
    rows: List[Dict[str, object]] = []
    seen_keys: Set[str] = set()

    for split in sorted(windows_df["split"].unique()):
        datasets = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=mode_from_split(str(split)),
            cleanup=False,
            concat=False,
        )

        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
            split_dataset_items = sorted(
                item_id
                for req_split, req_dataset, item_id in required_index
                if req_split == split and req_dataset == dataset_name
            )
            if not split_dataset_items:
                continue

            for item_id in split_dataset_items:
                item_dataset = deepcopy(dataset)
                item_dataset.select_user_data(item_id)
                channel_count = int(item_dataset.data.shape[0])
                len_per_channel = len(item_dataset) // channel_count
                required_for_item = required_index[(str(split), str(dataset_name), int(item_id))]
                required_pairs = {(channel_id, window_index) for channel_id, window_index, _ in required_for_item}
                sample_key_by_pair = {
                    (channel_id, window_index): sample_key
                    for channel_id, window_index, sample_key in required_for_item
                }

                dataloader = DataLoader(
                    item_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                )

                for batch_idx, batch in enumerate(dataloader):
                    x_batch = batch["x"].cpu().numpy()
                    for row_in_batch in range(x_batch.shape[0]):
                        global_sample_index = batch_idx * batch_size + row_in_batch
                        channel_id = global_sample_index // len_per_channel
                        window_index = global_sample_index % len_per_channel
                        pair = (int(channel_id), int(window_index))
                        if pair not in required_pairs:
                            continue

                        sample_key = sample_key_by_pair[pair]
                        feature_values = extract_timefuse_single_variable_features(x_batch[row_in_batch])
                        row: Dict[str, object] = {
                            "feature_version": FEATURE_VERSION,
                            "sample_key": sample_key,
                            "config_name": str(config_by_key[sample_key]),
                            "split": str(split),
                            "dataset_name": str(dataset_name),
                            "item_id": int(item_id),
                            "channel_id": int(channel_id),
                            "window_index": int(window_index),
                            "history_length": int(x_batch[row_in_batch].shape[0]),
                            "feature_type": FEATURE_TYPE,
                            "feature_dim": len(FEATURE_COLUMNS),
                        }
                        row.update(feature_values)
                        rows.append(row)
                        seen_keys.add(sample_key)

                    if len(seen_keys.intersection(sample_key_by_pair.values())) == len(sample_key_by_pair):
                        break

    return rows


def write_summary(output_dir: Path, feature_df: pd.DataFrame, labels_path: Path, metric: str) -> None:
    """函数功能：写出简短 Markdown 摘要，便于快速查看本次 pilot 结果。"""
    split_counts = feature_df.groupby(["config_name", "split", "dataset_name"]).size().reset_index(name="rows")
    # pandas.to_markdown 依赖可选包 tabulate；为了让 pilot 在 Quito 环境中少依赖，
    # 这里手写一个很小的 Markdown 表格。
    split_count_lines = [
        "| config_name | split | dataset_name | rows |",
        "| --- | --- | --- | --- |",
    ]
    for row in split_counts.itertuples(index=False):
        split_count_lines.append(f"| {row.config_name} | {row.split} | {row.dataset_name} | {row.rows} |")
    summary_lines = [
        "# Stage 1 TimeFuse-derived 单变量元特征 Pilot",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 输入",
        "",
        f"- labels_path: `{labels_path}`",
        f"- metric: `{metric}`",
        "",
        "## 输出",
        "",
        f"- feature_cache.csv: `{output_dir / 'feature_cache.csv'}`",
        f"- 样本数：{len(feature_df)}",
        f"- 特征数：{len(FEATURE_COLUMNS)}",
        "",
        "## 特征列",
        "",
        "```text",
        "\n".join(FEATURE_COLUMNS),
        "```",
        "",
        "## 分层计数",
        "",
        "\n".join(split_count_lines),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 TimeFuse-derived 单变量元特征 cache pilot。"""
    args = parse_args()
    output_dir = args.output_dir or RUN_OUTPUT_ROOT / f"{now_token()}_visual_router_stage1_structure_feature_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_df = load_required_windows(args.labels_path, args.metric)
    data_config = load_data_config(args.config_path)
    rows = build_feature_rows(
        data_config=data_config,
        windows_df=windows_df,
        batch_size=args.batch_size,
    )
    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.sort_values(["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"])
    validate_feature_cache(feature_df, windows_df)
    feature_df.to_csv(output_dir / "feature_cache.csv", index=False)

    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "labels_path": str(args.labels_path),
        "metric": args.metric,
        "config_path": str(args.config_path),
        "feature_version": FEATURE_VERSION,
        "feature_type": FEATURE_TYPE,
        "feature_columns": FEATURE_COLUMNS,
        "feature_dim": len(FEATURE_COLUMNS),
        "sample_count": int(len(feature_df)),
        "splits": sorted(feature_df["split"].unique().tolist()),
        "config_names": sorted(feature_df["config_name"].unique().tolist()),
        "pilot_scope": "timefuse_single_variable_meta_feature_cache",
        "excluded_timefuse_features": [
            "spectral_variation",
            "covariance_mean",
            "covariance_max",
            "covariance_min",
            "covariance_std",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir, feature_df, args.labels_path, args.metric)

    print(f"wrote structure feature cache pilot to {output_dir}")
    print(f"sample_count={len(feature_df)} feature_dim={len(FEATURE_COLUMNS)}")
    preview_cols = ["sample_key", "feature_dim"] + FEATURE_COLUMNS[:6]
    print(feature_df[preview_cols].head(args.print_rows).to_string(index=False))
    if len(feature_df) > args.print_rows:
        print(f"... omitted {len(feature_df) - args.print_rows} rows")


if __name__ == "__main__":
    main()
