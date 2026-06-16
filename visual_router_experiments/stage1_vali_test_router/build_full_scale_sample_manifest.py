#!/usr/bin/env python3
"""
文件功能：
    为 Stage 1 `96_48_S` 正式 full-scale 实验流式生成全候选窗口 sample manifest。

输入：
    - Quito evaluate config，用于读取 vali/test split 的真实 item/channel/window 边界；
    - TSF cell 映射，用于给每个 item 补充分层分析元信息；
    - sample shard 数量，用于把千万级 sample_key 拆成多个可独立消费的 CSV。

输出：
    - `sample_shards/sample_shard_XXXX_of_NNNN.csv`：prediction cache launcher 直接消费的分片 manifest；
    - `sample_manifest_shard_index.csv`：每个分片的路径和样本数；
    - `sampling_metadata.json`、`sampling_summary.md`、`status.json`。

关键约束：
    - 本脚本枚举所有候选窗口，不做 item/window 抽样；
    - 不启动专家推理，不保存 y_true/y_pred，不保存伪图像 tensor 或 ViT embedding；
    - 采用流式写 CSV，避免在内存中构造 2000 万级 DataFrame；
    - 每个 sample_key 仍遵守 Stage 1 cache contract：
      `config__split__dataset__item{item_id}__ch{channel_id}__win{window_index}`。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from omegaconf import OmegaConf


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
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
SHARD_DIR_NAME = "sample_shards"
SHARD_INDEX_NAME = "sample_manifest_shard_index.csv"

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config import AutoConfig  # noqa: E402
from quito.config.training import ModeType, TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import PredictionCacheKey  # noqa: E402


MANIFEST_COLUMNS = [
    "sample_key",
    "config_name",
    "split",
    "dataset_name",
    "item_id",
    "channel_id",
    "window_index",
    "history_length",
    "pred_length",
    "selection_strategy",
    "cluster",
    "group_name",
    "forecastability_cat",
    "season_strength_cat",
    "trend_strength_cat",
    "cv_cat",
    "missing_ratio_cat",
]


def now_token() -> str:
    """函数功能：生成输出目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata/status/summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 full-scale manifest 生成参数。"""
    parser = argparse.ArgumentParser(description="Build full-scale Stage 1 sample manifest shards for 96_48_S.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH, help="Quito evaluate config。")
    parser.add_argument("--config-name", default="96_48_S", help="写入 sample_key 的 config_name。")
    parser.add_argument("--cluster-path", type=Path, default=DEFAULT_CLUSTER_PATH, help="item_id 到 TSF cell 的映射 CSV。")
    parser.add_argument("--output-root", type=Path, default=DATA2_RUN_OUTPUT_ROOT, help="输出根目录；正式 full-scale 默认写 /data2。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录。")
    parser.add_argument("--splits", nargs="+", choices=["vali", "test"], default=["vali", "test"], help="需要枚举的 split。")
    parser.add_argument("--datasets", nargs="+", default=None, help="需要枚举的 dataset_name；为空表示 config 中全部 evaluate dataset。")
    parser.add_argument("--sample-shard-count", type=int, default=64, help="sample manifest 分片数量。")
    parser.add_argument("--print-rows", type=int, default=8, help="运行结束时打印多少行 shard index 预览。")
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
    loaded = OmegaConf.load(str(config_path))
    data_config, model_config, training_config = AutoConfig.from_config(
        config=loaded,
        rank=-1,
        world_size=-1,
        local_rank=-1,
    )
    del model_config, training_config
    return data_config


def load_cluster_lookup(cluster_path: Path) -> Dict[int, Dict[str, object]]:
    """
    函数功能：
        读取 TSF cell 映射并转换为 item_id -> 元信息字典。

    说明：
        full-scale 全候选窗口会覆盖大量 item。若有 item 缺少 TSF cell 映射，后续
        baseline 分层会出错，因此这里选择提前失败，不静默填空值。
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
    deduped = cluster_df[keep_cols].drop_duplicates("item_id").copy()
    return {int(row.item_id): row._asdict() for row in deduped.itertuples(index=False)}


def dataset_channel_counts(dataset) -> Tuple[Dict[int, List[int]], int]:
    """
    函数功能：
        从 Quito dataset 的 id_mask 中恢复 item_id -> channel_id 列表和每通道窗口数。

    关键假设：
        当前 `S` 配置会把 item-channel 展平到样本轴；channel_id 表示同一个 item 在
        `select_user_data()` 后的行序号。该口径与 1k/pilot manifest 保持一致。
    """
    if getattr(dataset, "id_mask", None) is None:
        raise ValueError(f"dataset={getattr(dataset, 'name', '<unknown>')} 缺少 id_mask，无法恢复 item/channel")
    id_axis = dataset.id_mask[:, 0, 0].astype(int)
    mapping: Dict[int, List[int]] = {}
    for item_id in sorted(int(item_id) for item_id in pd.unique(id_axis).tolist()):
        count = int((id_axis == item_id).sum())
        mapping[item_id] = list(range(count))
    len_per_channel = int(dataset.data.shape[1] - int(dataset.seq_len) - int(dataset.forecast_horizon) + 1)
    if len_per_channel <= 0:
        raise ValueError(f"dataset={dataset.name} 的 len_per_channel 非法：{len_per_channel}")
    return mapping, len_per_channel


def estimate_payload_bytes(sample_count: int, pred_length: int, channel_count: int = 1) -> Dict[str, object]:
    """函数功能：估算 packed prediction cache 和长期 embedding 的理论 payload。"""
    single_array_bytes = int(sample_count) * int(pred_length) * int(channel_count) * 4
    return {
        "sample_count": int(sample_count),
        "prediction_manifest_rows": int(sample_count * len(MODEL_COLUMNS)),
        "packed_shared_y_true_gib": float(single_array_bytes / (1024**3)),
        "packed_five_expert_y_pred_gib": float(single_array_bytes * len(MODEL_COLUMNS) / (1024**3)),
        "packed_y_true_plus_y_pred_gib": float(single_array_bytes * (len(MODEL_COLUMNS) + 1) / (1024**3)),
        "vit_embedding_fp32_gib_if_cached": float(sample_count * 768 * 4 / (1024**3)),
        "vit_embedding_fp16_gib_if_cached": float(sample_count * 768 * 2 / (1024**3)),
        "formal_embedding_cache_policy": "not_saved; streaming router only keeps batch runtime tensors",
    }


def open_shard_writers(output_dir: Path, shard_count: int) -> Tuple[List[csv.DictWriter], List[object], List[Path]]:
    """函数功能：打开所有 shard CSV writer，并写入表头。"""
    shard_dir = output_dir / SHARD_DIR_NAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    writers: List[csv.DictWriter] = []
    handles: List[object] = []
    paths: List[Path] = []
    for shard_index in range(int(shard_count)):
        shard_path = shard_dir / f"sample_shard_{shard_index:04d}_of_{int(shard_count):04d}.csv"
        handle = shard_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writers.append(writer)
        handles.append(handle)
        paths.append(shard_path)
    return writers, handles, paths


def close_handles(handles: Sequence[object]) -> None:
    """函数功能：关闭已打开的 shard CSV 文件句柄。"""
    for handle in handles:
        handle.close()


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出 status.json，供长任务监控和 handoff 使用。"""
    status = dict(payload)
    status["updated_at"] = display_time()
    status["output_dir"] = str(output_dir)
    (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_main_log(output_dir: Path, message: str) -> None:
    """函数功能：向 manifest 生成目录追加主日志。"""
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def build_manifest_shards(args: argparse.Namespace, output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    函数功能：
        枚举全候选窗口并按 round-robin 写入 shard CSV。

    设计说明：
        这里使用全局递增 sample ordinal 分片，而不是 hash。这样每个分片规模几乎
        完全一致，也不需要先把 2000 万 sample_key 排序加载到内存。
    """
    if int(args.sample_shard_count) <= 0:
        raise ValueError("--sample-shard-count 必须为正整数")
    data_config = load_data_config(args.config_path)
    cluster_lookup = load_cluster_lookup(args.cluster_path)
    requested_datasets = set(args.datasets) if args.datasets else None
    writers, handles, shard_paths = open_shard_writers(output_dir, int(args.sample_shard_count))
    shard_counts = [0 for _ in shard_paths]
    candidate_stats: List[Dict[str, object]] = []
    total_rows = 0
    first_rows: List[Dict[str, object]] = []

    try:
        for split in args.splits:
            append_main_log(output_dir, f"load datasets for split={split}")
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
                candidate_count = int(sum(len(channels) for channels in item_to_channels.values()) * len_per_channel)
                append_main_log(
                    output_dir,
                    f"enumerate split={split} dataset={dataset_name} items={len(item_to_channels)} windows={candidate_count}",
                )
                dataset_rows = 0
                for item_id in sorted(item_to_channels):
                    if int(item_id) not in cluster_lookup:
                        raise ValueError(f"item_id={item_id} 缺少 TSF cell 映射")
                    cluster_row = cluster_lookup[int(item_id)]
                    for channel_id in item_to_channels[int(item_id)]:
                        for window_index in range(int(len_per_channel)):
                            key = PredictionCacheKey(
                                config_name=str(args.config_name),
                                split=str(split),
                                dataset_name=str(dataset_name),
                                item_id=int(item_id),
                                channel_id=int(channel_id),
                                window_index=int(window_index),
                            )
                            row = {
                                "sample_key": key.as_string(),
                                "config_name": str(args.config_name),
                                "split": str(split),
                                "dataset_name": str(dataset_name),
                                "item_id": int(item_id),
                                "channel_id": int(channel_id),
                                "window_index": int(window_index),
                                "history_length": int(data_config.seq_len),
                                "pred_length": int(data_config.forecast_horizon),
                                "selection_strategy": "all_candidate_windows",
                                "cluster": int(cluster_row["cluster"]),
                                "group_name": str(cluster_row["group_name"]),
                                "forecastability_cat": str(cluster_row["forecastability_cat"]),
                                "season_strength_cat": str(cluster_row["season_strength_cat"]),
                                "trend_strength_cat": str(cluster_row["trend_strength_cat"]),
                                "cv_cat": str(cluster_row["cv_cat"]),
                                "missing_ratio_cat": str(cluster_row["missing_ratio_cat"]),
                            }
                            shard_index = total_rows % int(args.sample_shard_count)
                            writers[shard_index].writerow(row)
                            shard_counts[shard_index] += 1
                            total_rows += 1
                            dataset_rows += 1
                            if len(first_rows) < int(args.print_rows):
                                first_rows.append(dict(row))
                candidate_stats.append(
                    {
                        "split": str(split),
                        "dataset_name": str(dataset_name),
                        "candidate_item_count": int(len(item_to_channels)),
                        "candidate_channel_rows": int(sum(len(channels) for channels in item_to_channels.values())),
                        "len_per_channel": int(len_per_channel),
                        "candidate_window_count": int(candidate_count),
                        "written_rows": int(dataset_rows),
                    }
                )
                write_status(
                    output_dir,
                    {
                        "status": "running",
                        "phase": "enumerating",
                        "current_split": str(split),
                        "current_dataset": str(dataset_name),
                        "rows_written": int(total_rows),
                    },
                )
    finally:
        close_handles(handles)

    if total_rows <= 0:
        raise ValueError("没有生成任何 sample_key，请检查 --splits/--datasets 参数")

    shard_index_rows = []
    for shard_index, shard_path in enumerate(shard_paths):
        shard_index_rows.append(
            {
                "shard_index": int(shard_index),
                "shard_count": int(args.sample_shard_count),
                "sample_count": int(shard_counts[shard_index]),
                "sample_manifest_path": str(shard_path),
            }
        )
    shard_index_df = pd.DataFrame(shard_index_rows)
    shard_index_df.to_csv(output_dir / SHARD_INDEX_NAME, index=False)
    preview_df = pd.DataFrame(first_rows)
    if not preview_df.empty:
        preview_df.to_csv(output_dir / "sample_manifest_preview.csv", index=False)

    metadata: Dict[str, object] = {
        "status": "completed",
        "generated_at": display_time(),
        "config_path": str(args.config_path),
        "config_name": str(args.config_name),
        "cluster_path": str(args.cluster_path),
        "splits": list(args.splits),
        "datasets": [str(row["dataset_name"]) for row in candidate_stats],
        "sample_shard_count": int(args.sample_shard_count),
        "sample_count": int(total_rows),
        "history_length": int(data_config.seq_len),
        "pred_length": int(data_config.forecast_horizon),
        "selection_strategy": "all_candidate_windows",
        "candidate_stats": candidate_stats,
        "shard_index_path": str(output_dir / SHARD_INDEX_NAME),
        "sample_shard_dir": str(output_dir / SHARD_DIR_NAME),
        "sample_shard_min_count": int(min(shard_counts)),
        "sample_shard_max_count": int(max(shard_counts)),
        "sample_shard_counts": [int(value) for value in shard_counts],
        "cost_estimates": estimate_payload_bytes(int(total_rows), int(data_config.forecast_horizon)),
        "input_exclusions": ["expert_predictions", "future_y", "oracle_label", "embedding_npy", "pseudo_image_tensor_cache"],
    }
    return shard_index_df, metadata


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """函数功能：将小型 DataFrame 转成 Markdown 表格，避免额外依赖。"""
    if frame.empty:
        return "_无记录_"
    display = frame.copy()
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_outputs(output_dir: Path, shard_index_df: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    """函数功能：写 metadata、summary 和 completed status。"""
    (output_dir / "sampling_metadata.json").write_text(json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate_df = pd.DataFrame(metadata["candidate_stats"])
    estimates = metadata["cost_estimates"]
    summary_lines = [
        "# Stage 1 96_48_S Full-Scale Sample Manifest",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 枚举口径",
        "",
        f"- config_name: `{metadata['config_name']}`",
        f"- selection_strategy: `{metadata['selection_strategy']}`",
        f"- sample_count: `{metadata['sample_count']}`",
        f"- sample_shard_count: `{metadata['sample_shard_count']}`",
        f"- shard size: `{metadata['sample_shard_min_count']}` - `{metadata['sample_shard_max_count']}`",
        "- 不做 item/window 抽样；每个 item 的所有 channel 和所有可用 window 均进入候选集。",
        "- 本步骤不生成专家预测、oracle label、ViT embedding 或伪图像 tensor cache。",
        "",
        "## 候选窗口规模",
        "",
        frame_to_markdown(candidate_df),
        "",
        "## 成本估算",
        "",
        f"- prediction manifest 行数：`{estimates['prediction_manifest_rows']}`。",
        f"- packed 共享 y_true 理论 payload：`{estimates['packed_shared_y_true_gib']:.3f} GiB`。",
        f"- packed 五专家 y_pred 理论 payload：`{estimates['packed_five_expert_y_pred_gib']:.3f} GiB`。",
        f"- y_true + y_pred 理论 payload：`{estimates['packed_y_true_plus_y_pred_gib']:.3f} GiB`，不含 CSV/副本/文件系统开销。",
        f"- 若错误长期缓存 ViT embedding，fp32 约 `{estimates['vit_embedding_fp32_gib_if_cached']:.3f} GiB`；正式路线不保存。",
        "",
        "## 输出文件",
        "",
        f"- shard index: `{output_dir / SHARD_INDEX_NAME}`",
        f"- shard dir: `{output_dir / SHARD_DIR_NAME}`",
        f"- metadata: `{output_dir / 'sampling_metadata.json'}`",
        f"- status: `{output_dir / 'status.json'}`",
        "",
        "## Shard 预览",
        "",
        frame_to_markdown(shard_index_df.head(12)),
        "",
    ]
    (output_dir / "sampling_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    write_status(output_dir, dict(metadata))


def main() -> None:
    """函数功能：执行 full-scale 全候选窗口 manifest 分片生成。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_sample_manifest_96_48_s_full_scale"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main.log").write_text("", encoding="utf-8")
    append_main_log(output_dir, "start full-scale sample manifest generation")
    write_status(output_dir, {"status": "running", "phase": "init"})
    try:
        shard_index_df, metadata = build_manifest_shards(args, output_dir)
        metadata["output_dir"] = str(output_dir)
        write_outputs(output_dir, shard_index_df, metadata)
        append_main_log(output_dir, "completed full-scale sample manifest generation")
    except Exception as exc:
        write_status(output_dir, {"status": "failed", "phase": "error", "error": repr(exc)})
        append_main_log(output_dir, f"failed full-scale sample manifest generation: {repr(exc)}")
        raise

    print(f"wrote full-scale manifest shards to {output_dir}")
    print(f"sample_count={metadata['sample_count']} shard_count={metadata['sample_shard_count']}")
    print(shard_index_df.head(int(args.print_rows)).to_string(index=False))


if __name__ == "__main__":
    main()
