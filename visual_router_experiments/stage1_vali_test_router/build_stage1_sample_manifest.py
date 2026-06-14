#!/usr/bin/env python3
"""
文件功能：
    为 Stage 1 Visual Router 中等规模实验生成 manifest-only 样本清单。

输入：
    - Quito evaluate config，用于读取真实 split/dataset/item/channel/window 边界；
    - 可选 TSF cell 映射，用于让 item 抽样尽量覆盖不同结构 cell。

输出：
    - sample_manifest.csv：每个待路由窗口一行，不包含专家预测或 embedding；
    - sampling_metadata.json：记录抽样参数、候选窗口规模和磁盘/时间成本估算；
    - sampling_summary.md：中文摘要；
    - status.json：便于后台或交接时快速查看本步骤状态。

关键约束：
    - 本脚本不启动专家模型推理，不保存 y_true/y_pred，不保存伪图像 tensor；
    - 默认只生成 `96_48_S` 的 1k sample_key，vali/test 分开且 dataset/item/window 尽量均衡；
    - `sample_key` 必须遵守 Stage 1 cache contract，后续 prediction cache、embedding、
      router 训练和 calibration 都应以该清单为同一来源。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
DEFAULT_CLUSTER_PATH = QUITO_DIR / "examples" / "datasets" / "cluster_data" / "item_clusters.csv"
DEFAULT_CONFIG_PATH = (
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
MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config import AutoConfig  # noqa: E402
from quito.config.training import ModeType, TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402


def now_token() -> str:
    """函数功能：生成输出目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata/status/summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 manifest-only 抽样参数。"""
    parser = argparse.ArgumentParser(description="Build Stage 1 sample manifest for 96_48_S medium Visual Router run.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH, help="Quito evaluate config，用于加载数据边界。")
    parser.add_argument("--config-name", default="96_48_S", help="写入 sample_key 的 config_name。")
    parser.add_argument("--cluster-path", type=Path, default=DEFAULT_CLUSTER_PATH, help="item_id 到 TSF cell 的映射 CSV。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认生成时间戳目录。")
    parser.add_argument("--splits", nargs="+", choices=["vali", "test"], default=["vali", "test"], help="需要抽样的 split。")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="需要抽样的 dataset_name；为空表示使用 config 中全部 evaluate dataset。",
    )
    parser.add_argument("--items-per-dataset", type=int, default=50, help="每个 split/dataset 抽取多少 item。")
    parser.add_argument("--windows-per-item", type=int, default=5, help="每个 item/channel 抽取多少等距 window。")
    parser.add_argument("--channels-per-item", type=int, default=1, help="每个 item 抽取多少 channel；默认只取 ch0 延续当前 smoke 口径。")
    parser.add_argument(
        "--item-strategy",
        choices=["tsf_balanced", "even_spaced"],
        default="tsf_balanced",
        help="item 抽样策略；默认按 TSF cell 均衡配额后在 cell 内等距取样。",
    )
    parser.add_argument("--print-rows", type=int, default=20, help="运行结束时打印多少行清单预览。")
    return parser.parse_args()


def mode_from_split(split: str) -> ModeType:
    """函数功能：把 Stage 1 split 名称映射到 Quito ModeType。"""
    if split == "vali":
        return ModeType.VALID
    if split == "test":
        return ModeType.TEST
    raise ValueError(f"未知 split：{split}")


def load_data_config(config_path: Path):
    """函数功能：读取 Quito config，并返回数据配置。"""
    config = OmegaConf.load(config_path)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        rank=-1,
        world_size=-1,
        local_rank=-1,
    )
    del model_config, training_config
    return data_config


def load_cluster_frame(cluster_path: Path) -> pd.DataFrame:
    """
    函数功能：
        读取 TSF cell 映射，返回后续清单中需要保留的字段。

    说明：
        若 cluster 文件不存在，直接报错。当前 Stage 1 分层分析依赖这些字段，缺失时
        不应悄悄退化为无分层抽样。
    """
    if not cluster_path.exists():
        raise FileNotFoundError(f"找不到 TSF cell 映射：{cluster_path}")
    cluster_df = pd.read_csv(cluster_path)
    keep_cols = [
        "item_id",
        "cluster",
        "group_name",
        "forecastability_cat",
        "season_strength_cat",
        "trend_strength_cat",
        "cv_cat",
        "missing_ratio_cat",
    ]
    missing = sorted(set(keep_cols).difference(cluster_df.columns))
    if missing:
        raise ValueError(f"cluster 映射缺少字段：{missing}")
    return cluster_df[keep_cols].drop_duplicates("item_id").copy()


def center_spaced_indices(length: int, count: int) -> List[int]:
    """
    函数功能：
        从 `[0, length)` 中取 count 个中心等距位置。

    设计说明：
        用分桶中心而不是前缀，避免中等规模样本只覆盖每条序列开头窗口。
    """
    if count <= 0:
        raise ValueError("count 必须为正整数")
    if length <= 0:
        raise ValueError("length 必须为正整数")
    if count > length:
        raise ValueError(f"请求 {count} 个位置，但候选长度只有 {length}")
    raw = np.floor((np.arange(count, dtype=np.float64) + 0.5) * float(length) / float(count)).astype(int)
    raw = np.clip(raw, 0, length - 1)
    values = sorted(set(int(x) for x in raw.tolist()))
    # 当 length/count 比例极端时理论上可能因取整重复，这里用顺序补齐保证数量准确。
    if len(values) < count:
        for candidate in range(length):
            if candidate not in values:
                values.append(candidate)
            if len(values) == count:
                break
        values = sorted(values)
    return values


def allocate_balanced_quotas(group_sizes: Mapping[str, int], total: int) -> Dict[str, int]:
    """
    函数功能：
        在 TSF cell 之间分配 item 抽样配额，尽量均衡且不超过各组容量。
    """
    if total <= 0:
        raise ValueError("total 必须为正整数")
    available = {str(group): int(size) for group, size in group_sizes.items() if int(size) > 0}
    if not available:
        raise ValueError("没有可抽样 item")
    if sum(available.values()) < total:
        raise ValueError(f"候选 item 总数 {sum(available.values())} 小于请求数量 {total}")

    quotas = {group: 0 for group in sorted(available)}
    for _ in range(total):
        candidates = [group for group, size in available.items() if quotas[group] < size]
        # 优先补当前配额最少的组；平手时按组名稳定排序，保证完全可复现。
        chosen = min(candidates, key=lambda group: (quotas[group], group))
        quotas[chosen] += 1
    return quotas


def select_items_even_spaced(item_ids: Sequence[int], count: int) -> List[int]:
    """函数功能：在排序 item_id 上做中心等距抽样。"""
    sorted_ids = sorted(int(item_id) for item_id in item_ids)
    indices = center_spaced_indices(len(sorted_ids), count)
    return [sorted_ids[index] for index in indices]


def select_items_tsf_balanced(item_ids: Sequence[int], count: int, cluster_df: pd.DataFrame) -> List[int]:
    """
    函数功能：
        先按 TSF cell 分配均衡配额，再在每个 cell 内对 item_id 做等距抽样。
    """
    item_frame = pd.DataFrame({"item_id": sorted(int(item_id) for item_id in item_ids)})
    merged = item_frame.merge(cluster_df[["item_id", "group_name"]], on="item_id", how="left")
    missing_count = int(merged["group_name"].isna().sum())
    if missing_count:
        missing_items = merged.loc[merged["group_name"].isna(), "item_id"].head(10).tolist()
        raise ValueError(f"有 {missing_count} 个 item 缺少 TSF cell 映射，示例：{missing_items}")

    group_to_items = {
        str(group): group_df["item_id"].astype(int).sort_values().tolist()
        for group, group_df in merged.groupby("group_name", sort=True)
    }
    quotas = allocate_balanced_quotas({group: len(values) for group, values in group_to_items.items()}, count)
    selected: List[int] = []
    for group in sorted(group_to_items):
        quota = quotas.get(group, 0)
        if quota:
            selected.extend(select_items_even_spaced(group_to_items[group], quota))
    return sorted(selected)


def dataset_channel_counts(dataset) -> Tuple[Dict[int, List[int]], int]:
    """
    函数功能：
        从 Quito dataset 的 id_mask 中恢复 item_id -> 可用 channel_id 列表。

    关键约束：
        当前 `S` 配置会把 item-channel 展平到样本轴。已有 pilot 中 `channel_id`
        表示某个 item 被 `select_user_data()` 后的行序号；这里按同一口径保留。
    """
    if getattr(dataset, "id_mask", None) is None:
        raise ValueError(f"dataset={getattr(dataset, 'name', '<unknown>')} 缺少 id_mask，无法恢复 item/channel")
    id_axis = dataset.id_mask[:, 0, 0].astype(int)
    mapping: Dict[int, List[int]] = {}
    for item_id in sorted(int(item_id) for item_id in np.unique(id_axis).tolist()):
        count = int((id_axis == item_id).sum())
        mapping[item_id] = list(range(count))
    len_per_channel = int(dataset.data.shape[1] - int(dataset.seq_len) - int(dataset.forecast_horizon) + 1)
    if len_per_channel <= 0:
        raise ValueError(f"dataset={dataset.name} 的 len_per_channel 非法：{len_per_channel}")
    return mapping, len_per_channel


def enrich_rows(rows: List[Dict[str, object]], cluster_df: pd.DataFrame) -> pd.DataFrame:
    """函数功能：为 sample manifest 补充 TSF cell 字段并排序。"""
    frame = pd.DataFrame(rows)
    enriched = frame.merge(cluster_df, on="item_id", how="left")
    if enriched["group_name"].isna().any():
        bad_items = enriched.loc[enriched["group_name"].isna(), "item_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"sample manifest 存在缺失 TSF cell 的 item，示例：{bad_items}")
    sort_cols = ["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]
    return enriched.sort_values(sort_cols).reset_index(drop=True)


def estimate_costs(sample_count: int, pred_length: int, channel_count: int = 1) -> Dict[str, object]:
    """
    函数功能：
        根据样本数给出 prediction cache / embedding cache 的粗略成本估算。

    说明：
        估算分为理论数组 payload 和按当前 120 sample_key pilot 的 `du` 实测线性外推。
        小 `.npy` 文件的文件系统块开销远高于 payload，因此两者都记录。
    """
    npy_payload_bytes = int(pred_length) * int(channel_count) * 4
    # 当前 48x1 float32 小数组实际 .npy 文件约 320 bytes；不同文件系统块会让 du 更大。
    approximate_npy_file_bytes = max(320, npy_payload_bytes + 128)
    duplicate_y_true_files = sample_count * len(MODEL_COLUMNS) * 2
    shared_y_true_files = sample_count * (len(MODEL_COLUMNS) + 1)
    reference_cache_du_mib_per_sample = 5.9 / 120.0
    embedding_dim = 768
    return {
        "sample_count": int(sample_count),
        "prediction_manifest_rows": int(sample_count * len(MODEL_COLUMNS)),
        "prediction_cache_duplicate_y_true_file_count": int(duplicate_y_true_files),
        "prediction_cache_shared_y_true_file_count": int(shared_y_true_files),
        "prediction_cache_duplicate_y_true_logical_mib": float(duplicate_y_true_files * approximate_npy_file_bytes / (1024**2)),
        "prediction_cache_shared_y_true_logical_mib": float(shared_y_true_files * approximate_npy_file_bytes / (1024**2)),
        "prediction_cache_duplicate_y_true_du_estimate_mib_from_120_pilot": float(reference_cache_du_mib_per_sample * sample_count),
        "vit_embedding_float32_mib": float(sample_count * embedding_dim * 4 / (1024**2)),
        "vit_embedding_float16_mib": float(sample_count * embedding_dim * 2 / (1024**2)),
        "online_embedding_long_term_embedding_npy_mib": 0.0,
        "fixed_candidates_vit_smoke_latency_note": (
            "当前 120 sample_key fixed_candidates smoke 去 warm-up 后端到端约 1.405 ms/window；"
            "1k 样本纯 ViT 前向通常不是主要瓶颈，模型加载、I/O 和专家 prediction cache 更重要。"
        ),
        "prediction_cache_time_note": (
            "1k 五专家 cache 应后台运行并按专家或 shard 绑定 GPU/CPU；"
            "当前没有正式 1k 实测时间，预计深度专家推理为分钟级，ES 统计模型可能成为 CPU 侧瓶颈。"
        ),
    }


def build_manifest(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """函数功能：执行真实数据边界枚举和均衡 sample_key 抽样。"""
    data_config = load_data_config(args.config_path)
    cluster_df = load_cluster_frame(args.cluster_path)
    requested_datasets = set(args.datasets) if args.datasets else None
    all_rows: List[Dict[str, object]] = []
    candidate_stats: List[Dict[str, object]] = []

    for split in args.splits:
        datasets = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=mode_from_split(str(split)),
            cleanup=False,
            concat=False,
        )
        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
            if requested_datasets is not None and dataset_name not in requested_datasets:
                continue
            item_to_channels, len_per_channel = dataset_channel_counts(dataset)
            candidate_item_ids = sorted(item_to_channels)
            if args.item_strategy == "tsf_balanced":
                selected_items = select_items_tsf_balanced(candidate_item_ids, int(args.items_per_dataset), cluster_df)
            else:
                selected_items = select_items_even_spaced(candidate_item_ids, int(args.items_per_dataset))

            windows = center_spaced_indices(len_per_channel, int(args.windows_per_item))
            candidate_stats.append(
                {
                    "split": split,
                    "dataset_name": dataset_name,
                    "candidate_item_count": int(len(candidate_item_ids)),
                    "candidate_channel_rows": int(sum(len(values) for values in item_to_channels.values())),
                    "len_per_channel": int(len_per_channel),
                    "candidate_window_count": int(sum(len(values) for values in item_to_channels.values()) * len_per_channel),
                    "selected_item_count": int(len(selected_items)),
                    "selected_windows_per_item": int(len(windows)),
                    "selected_window_indices": windows,
                }
            )

            for item_id in selected_items:
                available_channels = item_to_channels[int(item_id)]
                if int(args.channels_per_item) > len(available_channels):
                    raise ValueError(
                        f"item_id={item_id} 只有 {len(available_channels)} 个 channel，"
                        f"但请求 channels_per_item={args.channels_per_item}"
                    )
                selected_channels = available_channels[: int(args.channels_per_item)]
                for channel_id in selected_channels:
                    for window_index in windows:
                        key = PredictionCacheKey(
                            config_name=str(args.config_name),
                            split=str(split),
                            dataset_name=str(dataset_name),
                            item_id=int(item_id),
                            channel_id=int(channel_id),
                            window_index=int(window_index),
                        )
                        all_rows.append(
                            {
                                "sample_key": key.as_string(),
                                "config_name": str(args.config_name),
                                "split": str(split),
                                "dataset_name": str(dataset_name),
                                "item_id": int(item_id),
                                "channel_id": int(channel_id),
                                "window_index": int(window_index),
                                "history_length": int(data_config.seq_len),
                                "pred_length": int(data_config.forecast_horizon),
                                "selection_strategy": str(args.item_strategy),
                            }
                        )

    if not all_rows:
        raise ValueError("没有生成任何 sample_key，请检查 --splits/--datasets 参数")
    manifest_df = enrich_rows(all_rows, cluster_df)
    if manifest_df["sample_key"].duplicated().any():
        dup = manifest_df.loc[manifest_df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"sample_key 重复，示例：{dup}")

    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "config_path": str(args.config_path),
        "config_name": str(args.config_name),
        "cluster_path": str(args.cluster_path),
        "splits": list(args.splits),
        "datasets": sorted(manifest_df["dataset_name"].unique().tolist()),
        "items_per_dataset": int(args.items_per_dataset),
        "windows_per_item": int(args.windows_per_item),
        "channels_per_item": int(args.channels_per_item),
        "item_strategy": str(args.item_strategy),
        "sample_count": int(len(manifest_df)),
        "history_length": int(data_config.seq_len),
        "pred_length": int(data_config.forecast_horizon),
        "candidate_stats": candidate_stats,
        "cost_estimates": estimate_costs(len(manifest_df), int(data_config.forecast_horizon)),
        "input_exclusions": ["expert_predictions", "future_y", "oracle_label", "embedding_npy", "pseudo_image_tensor_cache"],
    }
    return manifest_df, metadata


def frame_to_markdown(df: pd.DataFrame) -> str:
    """函数功能：将小型 DataFrame 转成 Markdown 表格，避免额外依赖。"""
    if df.empty:
        return "_无记录_"
    display_df = df.copy()
    lines = [
        "| " + " | ".join(display_df.columns) + " |",
        "| " + " | ".join(["---"] * len(display_df.columns)) + " |",
    ]
    for row in display_df.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_outputs(output_dir: Path, manifest_df: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    """函数功能：写出 sample manifest、metadata、summary 和 status。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.csv"
    metadata_path = output_dir / "sampling_metadata.json"
    summary_path = output_dir / "sampling_summary.md"
    status_path = output_dir / "status.json"

    manifest_df.to_csv(manifest_path, index=False)
    metadata_path.write_text(json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    coverage_df = (
        manifest_df.groupby(["config_name", "split", "dataset_name"])
        .agg(
            sample_count=("sample_key", "count"),
            item_count=("item_id", "nunique"),
            channel_count=("channel_id", "nunique"),
            min_window=("window_index", "min"),
            max_window=("window_index", "max"),
        )
        .reset_index()
    )
    tsf_df = manifest_df.groupby(["split", "dataset_name", "group_name"]).size().reset_index(name="sample_count")
    estimates = metadata["cost_estimates"]
    lines = [
        "# Stage 1 96_48_S Sample Manifest",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 抽样口径",
        "",
        f"- config_name: `{metadata['config_name']}`",
        f"- split: `{', '.join(metadata['splits'])}`",
        f"- item_strategy: `{metadata['item_strategy']}`",
        f"- items_per_dataset: `{metadata['items_per_dataset']}`",
        f"- windows_per_item: `{metadata['windows_per_item']}`，使用每条序列 split 内中心等距窗口。",
        f"- channels_per_item: `{metadata['channels_per_item']}`，当前保持 ch0 口径。",
        f"- sample_count: `{metadata['sample_count']}`",
        "",
        "## 覆盖统计",
        "",
        frame_to_markdown(coverage_df),
        "",
        "## TSF Cell 分布",
        "",
        frame_to_markdown(tsf_df),
        "",
        "## 成本估算",
        "",
        f"- prediction manifest 行数：`{estimates['prediction_manifest_rows']}`。",
        f"- 旧 pilot 重复 y_true 小文件口径线性外推目录占用：约 `{estimates['prediction_cache_duplicate_y_true_du_estimate_mib_from_120_pilot']:.2f} MiB`。",
        f"- 若共享 y_true，小数组逻辑体积约 `{estimates['prediction_cache_shared_y_true_logical_mib']:.2f} MiB`。",
        f"- ViT embedding float32 长期缓存约 `{estimates['vit_embedding_float32_mib']:.2f} MiB`，fp16 约 `{estimates['vit_embedding_float16_mib']:.2f} MiB`。",
        "- online embedding 正式路线不长期保存 embedding npy；如需 smoke cache，应优先写到 `/data2/syh/Time/cache_shards/`。",
        "",
        "## 输出文件",
        "",
        f"- sample_manifest.csv: `{manifest_path}`",
        f"- sampling_metadata.json: `{metadata_path}`",
        f"- status.json: `{status_path}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    status = {
        "status": "completed",
        "updated_at": display_time(),
        "output_dir": str(output_dir),
        "sample_manifest_path": str(manifest_path),
        "sample_count": int(len(manifest_df)),
        "summary_path": str(summary_path),
    }
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    """函数功能：执行 Stage 1 manifest-only 抽样。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_sample_manifest_96_48_s_1k"
    manifest_df, metadata = build_manifest(args)
    metadata["output_dir"] = str(output_dir)
    write_outputs(output_dir, manifest_df, metadata)

    print(f"wrote sample manifest to {output_dir}")
    print(f"sample_count={len(manifest_df)}")
    preview_cols = ["sample_key", "split", "dataset_name", "item_id", "channel_id", "window_index", "group_name"]
    print(manifest_df[preview_cols].head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
