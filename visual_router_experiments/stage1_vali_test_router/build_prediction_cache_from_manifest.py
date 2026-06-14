#!/usr/bin/env python3
"""
文件功能：
    基于 Stage 1 sample manifest 构建 window-level prediction cache shard。

输入：
    - `build_stage1_sample_manifest.py` 生成的 sample_manifest.csv；
    - 一个或多个 Quito evaluate config，每个 config 对应一个冻结专家。

输出：
    - manifest.csv：当前 shard 的 `(sample_key, model_name)` 预测记录；
    - arrays/y_true/*.npy：每个 sample_key 共享一份真实未来数组；
    - arrays/y_pred/{model_name}/*.npy：每个专家的预测数组；
    - metadata.json / status.json / main_summary.md。

关键约束：
    - 本脚本不负责跨 shard 合并，避免多个后台进程写同一个输出文件；
    - 每个 shard 内同一 sample_key 共享 y_true_path，符合 Stage 1 cache contract；
    - 支持按专家或按 sample_key shard 并发，launcher 应为每个进程设置独立 output-dir；
    - 深度模型可通过 `CUDA_VISIBLE_DEVICES=<gpu>` + `--local-rank 0` 绑定单卡；
      ES/SNaive 等统计模型建议保持 `--local-rank -1` 走 CPU。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset


WORKSPACE = Path("/home/shiyuhong/Time")
QUITO_DIR = WORKSPACE / "quito"
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
MODEL_DISPLAY_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
DEFAULT_FIVE_EXPERT_CONFIGS = [
    QUITO_DIR / "outputs" / "default_baseline" / "dlinear" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    QUITO_DIR / "outputs" / "default_baseline" / "patchtst" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    QUITO_DIR / "outputs" / "default_baseline" / "crossformer" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    QUITO_DIR / "outputs" / "statistical_baseline" / "es" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    QUITO_DIR / "outputs" / "statistical_baseline" / "snaive" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
]

for path in [WORKSPACE, QUITO_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quito.config import AutoConfig  # noqa: E402
from quito.config.training import ModeType, TaskType  # noqa: E402
from quito.datasets import load_datasets  # noqa: E402
from quito.models import AutoModel  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import (  # noqa: E402
    PredictionCacheKey,
    make_prediction_record,
    records_to_frame,
    validate_manifest_frame,
)


def now_token() -> str:
    """函数功能：生成输出目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata/status/summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 prediction cache shard 构建参数。"""
    parser = argparse.ArgumentParser(description="Build Stage 1 prediction cache shard from sample_manifest.csv.")
    parser.add_argument("--sample-manifest-path", type=Path, required=True, help="sample_manifest.csv 路径。")
    parser.add_argument("--config-paths", type=Path, nargs="+", default=DEFAULT_FIVE_EXPERT_CONFIGS, help="专家 evaluate config 列表。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="只运行指定专家名称，例如 DLinear PatchTST；为空表示运行 config-paths 中全部专家。",
    )
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="仅写入 metadata，实际 manifest 同时包含 MAE/MSE。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认生成时间戳目录。")
    parser.add_argument("--shard-index", type=int, default=0, help="当前 sample_key shard 编号，从 0 开始。")
    parser.add_argument("--shard-count", type=int, default=1, help="sample_key shard 总数。")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers。")
    parser.add_argument("--local-rank", type=int, default=-1, help="Quito 模型 local_rank；深度模型单卡绑定时设为 0。")
    parser.add_argument("--device-note", default=None, help="写入 metadata 的人工说明，例如 CUDA_VISIBLE_DEVICES=2。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束时打印多少行 manifest 预览。")
    return parser.parse_args()


def mode_from_split(split: str) -> ModeType:
    """函数功能：把 Stage 1 split 名称映射到 Quito ModeType。"""
    if split == "vali":
        return ModeType.VALID
    if split == "test":
        return ModeType.TEST
    raise ValueError(f"未知 split：{split}")


def config_name_from_path(config_path: Path) -> str:
    """函数功能：从已有 evaluate config 路径中提取 config_name。"""
    parts = config_path.resolve().parts
    for part in ["96_48_S", "576_288_S", "1024_512_S"]:
        if part in parts:
            return part
    return config_path.parent.name


def normalize_checkpoint_selection(model_name: str, checkpoint_path) -> str:
    """函数功能：为 manifest 记录专家 checkpoint 选择口径。"""
    if model_name in {"ES", "NaiveForecaster"}:
        return "not_applicable_statistical_model"
    return "validation_mae_best_or_config_defined"


def relative_to_output(path: Path, output_dir: Path) -> Path:
    """函数功能：生成写入 manifest 的相对路径。"""
    return path.relative_to(output_dir)


def save_array_once(path: Path, array: np.ndarray, existing_cache: Dict[Path, np.ndarray]) -> None:
    """
    函数功能：
        保存数组，若同一路径已保存则校验内容一致。

    说明：
        同一 sample_key 的 y_true 会被多个专家复用。这里避免重复写同一个文件，也避免
        专家之间由于数据对齐错误写出不一致 y_true。
    """
    array = np.asarray(array, dtype=np.float32)
    if path in existing_cache:
        if not np.array_equal(existing_cache[path], array):
            raise ValueError(f"同一路径重复保存但内容不一致：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))
    existing_cache[path] = array.copy()


def save_array(path: Path, array: np.ndarray) -> None:
    """函数功能：保存单个专家预测数组。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array, dtype=np.float32))


def make_item_dataset_view(dataset, item_id: int):
    """
    函数功能：
        从已加载 Quito dataset 构造单个 item 的轻量视图。

    设计说明：
        `select_user_data()` 会原地修改 dataset。这里浅拷贝 dataset 元信息，只替换
        `data/id_mask`，避免每个 item 重新读取 parquet，也避免破坏原 dataset。
    """
    if getattr(dataset, "id_mask", None) is None:
        raise ValueError(f"dataset={getattr(dataset, 'name', '<unknown>')} 缺少 id_mask")
    item_dataset = copy.copy(dataset)
    mask = dataset.id_mask == int(item_id)
    item_dataset.data = dataset.data[mask].reshape(-1, dataset.data.shape[1], dataset.data.shape[-1])
    item_dataset.id_mask = dataset.id_mask[mask].reshape(-1, dataset.id_mask.shape[1], dataset.id_mask.shape[-1])
    return item_dataset


def load_sample_manifest(sample_manifest_path: Path) -> pd.DataFrame:
    """函数功能：读取并校验 sample manifest。"""
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"找不到 sample manifest：{sample_manifest_path}")
    df = pd.read_csv(sample_manifest_path)
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
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"sample manifest 缺少字段：{missing}")
    if df["sample_key"].duplicated().any():
        dup = df.loc[df["sample_key"].duplicated(), "sample_key"].head(10).tolist()
        raise ValueError(f"sample manifest 存在重复 sample_key，示例：{dup}")
    expected = df.apply(
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
    bad = df["sample_key"].astype(str) != expected
    if bad.any():
        bad_row = df.loc[bad].iloc[0]
        raise ValueError(f"sample_key 与字段不一致：{bad_row['sample_key']}")
    return df.sort_values(["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index"]).reset_index(drop=True)


def filter_sample_shard(sample_df: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    """函数功能：按 sample_key 稳定顺序切分 shard。"""
    if shard_count <= 0:
        raise ValueError("--shard-count 必须为正整数")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index 必须落在 [0, shard_count)")
    if shard_count == 1:
        return sample_df.copy().reset_index(drop=True)
    ordered = sample_df.sort_values("sample_key").reset_index(drop=True)
    mask = (np.arange(len(ordered)) % int(shard_count)) == int(shard_index)
    return ordered.loc[mask].reset_index(drop=True)


def resolve_config_paths(config_paths: Sequence[Path], requested_models: Optional[Sequence[str]]) -> List[Path]:
    """
    函数功能：
        根据 `--models` 过滤专家 config。
    """
    if requested_models is None:
        return [Path(path) for path in config_paths]
    wanted = set(str(model) for model in requested_models)
    resolved: List[Path] = []
    for config_path in config_paths:
        model_name = config_name_from_model_path(Path(config_path))
        if model_name in wanted:
            resolved.append(Path(config_path))
    found = {config_name_from_model_path(path) for path in resolved}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"--models 请求的专家没有匹配 config：{missing}")
    return resolved


def config_name_from_model_path(config_path: Path) -> str:
    """函数功能：读取 config_path 对应的 model_name。"""
    config = OmegaConf.load(config_path)
    _, model_config, _ = AutoConfig.from_config(config=config, rank=-1, world_size=-1, local_rank=-1)
    return str(model_config.model_name)


def prepare_model(config_path: Path, local_rank: int):
    """
    函数功能：
        读取 Quito evaluate config 并加载冻结专家模型。

    返回：
        data_config、model_config、training_config、model。
    """
    config = OmegaConf.load(config_path)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        rank=local_rank if local_rank >= 0 else -1,
        world_size=1,
        local_rank=local_rank,
    )
    model = AutoModel.from_config(model_config, local_rank=local_rank)
    # Quito 的 AutoModel 会设置 model.device，但不会在所有路径下主动搬迁参数。
    # 这里显式迁移，保证 `CUDA_VISIBLE_DEVICES=<gpu> --local-rank 0` 单卡路径中
    # batch tensor 与模型权重位于同一设备。
    model = model.to(model.device)
    model.metrics = training_config.eval_metrics
    model.eval()
    return data_config, model_config, training_config, model


def build_required_index(sample_df: pd.DataFrame, config_name: str) -> Mapping[Tuple[str, str, int], Dict[Tuple[int, int], str]]:
    """
    函数功能：
        将待覆盖窗口整理为 `(split, dataset_name, item_id) -> (channel, window) -> sample_key`。
    """
    config_df = sample_df[sample_df["config_name"] == config_name].copy()
    if config_df.empty:
        raise ValueError(f"sample manifest 中没有 config_name={config_name} 的样本")
    required: Dict[Tuple[str, str, int], Dict[Tuple[int, int], str]] = {}
    for row in config_df.itertuples(index=False):
        group_key = (str(row.split), str(row.dataset_name), int(row.item_id))
        pair = (int(row.channel_id), int(row.window_index))
        required.setdefault(group_key, {})[pair] = str(row.sample_key)
    return required


def build_cache_for_model(
    *,
    sample_df: pd.DataFrame,
    data_config,
    model,
    model_name: str,
    config_name: str,
    checkpoint_selection: str,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    shared_y_true_cache: Dict[Path, np.ndarray],
) -> Tuple[List, List[Dict[str, object]]]:
    """函数功能：为单个专家生成当前 shard 的 prediction cache records。"""
    required_index = build_required_index(sample_df, config_name=config_name)
    records = []
    latency_rows: List[Dict[str, object]] = []
    arrays_dir = output_dir / "arrays"

    for split in sorted(sample_df["split"].astype(str).unique()):
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
                item_dataset = make_item_dataset_view(dataset, int(item_id))
                channel_count = int(item_dataset.data.shape[0])
                len_per_channel = len(item_dataset) // channel_count
                required_for_item = required_index[(split, dataset_name, int(item_id))]
                required_entries = sorted(
                    [
                        (int(channel_id), int(window_index), str(sample_key))
                        for (channel_id, window_index), sample_key in required_for_item.items()
                    ],
                    key=lambda value: (value[0], value[1], value[2]),
                )
                required_indices = [
                    channel_id * len_per_channel + window_index
                    for channel_id, window_index, _ in required_entries
                ]
                for global_index in required_indices:
                    if global_index < 0 or global_index >= len(item_dataset):
                        raise ValueError(
                            f"sample manifest 中存在越界窗口：item_id={item_id} global_index={global_index} "
                            f"len_item_dataset={len(item_dataset)}"
                        )

                dataloader = DataLoader(
                    Subset(item_dataset, required_indices),
                    batch_size=int(batch_size),
                    shuffle=False,
                    num_workers=int(num_workers),
                )
                seen_pairs = set()
                row_cursor = 0
                start_time = time.perf_counter()
                with torch.no_grad():
                    for batch in dataloader:
                        batch_entries = required_entries[row_cursor : row_cursor + int(batch["x"].shape[0])]
                        row_cursor += int(batch["x"].shape[0])
                        # 只对 sample manifest 指定窗口做专家前向，避免等距抽样时扫描完整 item。
                        _, predictions = model.eval_step(batch)
                        y_true_batch = batch["y"][:, -model.forecast_horizon :, :].detach().cpu().numpy()
                        y_pred_batch = predictions.detach().cpu().numpy()

                        for row_in_batch in range(y_pred_batch.shape[0]):
                            channel_id, window_index, sample_key = batch_entries[row_in_batch]
                            pair = (int(channel_id), int(window_index))
                            key = PredictionCacheKey(
                                config_name=config_name,
                                split=split,
                                dataset_name=dataset_name,
                                item_id=int(item_id),
                                channel_id=int(channel_id),
                                window_index=int(window_index),
                            )
                            if key.as_string() != sample_key:
                                raise ValueError(f"sample_key 与元信息不一致：{sample_key} vs {key.as_string()}")

                            y_true = y_true_batch[row_in_batch]
                            y_pred = y_pred_batch[row_in_batch]
                            y_true_path = arrays_dir / "y_true" / split / dataset_name / f"{sample_key}__y_true.npy"
                            y_pred_path = arrays_dir / "y_pred" / model_name / split / dataset_name / f"{sample_key}__y_pred.npy"
                            save_array_once(y_true_path, y_true, shared_y_true_cache)
                            save_array(y_pred_path, y_pred)
                            record = make_prediction_record(
                                key=key,
                                history_length=model.seq_len,
                                pred_length=model.forecast_horizon,
                                model_name=model_name,
                                expert_version=str(getattr(model.config, "checkpoint_path", "not_applicable")),
                                checkpoint_selection=checkpoint_selection,
                                y_true_path=relative_to_output(y_true_path, output_dir),
                                y_pred_path=relative_to_output(y_pred_path, output_dir),
                                y_true=y_true,
                                y_pred=y_pred,
                            )
                            records.append(record)
                            seen_pairs.add(pair)

                missing_pairs = sorted(set(required_for_item.keys()) - seen_pairs)
                if missing_pairs:
                    raise ValueError(
                        f"model={model_name} split={split} dataset={dataset_name} item={item_id} "
                        f"缺少 {len(missing_pairs)} 个窗口，示例={missing_pairs[:10]}"
                    )
                elapsed = time.perf_counter() - start_time
                latency_rows.append(
                    {
                        "model_name": model_name,
                        "config_name": config_name,
                        "split": split,
                        "dataset_name": dataset_name,
                        "item_id": int(item_id),
                        "sample_count": int(len(required_entries)),
                        "elapsed_seconds": float(elapsed),
                        "seconds_per_sample": float(elapsed / max(1, len(required_entries))),
                    }
                )
    return records, latency_rows


def write_summary(output_dir: Path, manifest_df: pd.DataFrame, latency_df: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    """函数功能：写出中文 Markdown 摘要。"""
    coverage = (
        manifest_df.groupby(["config_name", "split", "dataset_name", "model_name"])
        .size()
        .reset_index(name="rows")
    )

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

    lines = [
        "# Stage 1 Prediction Cache Shard",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 口径",
        "",
        f"- sample_manifest: `{metadata['sample_manifest_path']}`",
        f"- shard: `{metadata['shard_index']}/{metadata['shard_count']}`",
        f"- models: `{', '.join(metadata['model_names'])}`",
        f"- local_rank: `{metadata['local_rank']}`，device_note: `{metadata.get('device_note')}`",
        f"- sample_count: `{metadata['sample_count']}`，manifest_rows: `{metadata['record_count']}`",
        "",
        "## 覆盖统计",
        "",
        frame_to_markdown(coverage),
        "",
        "## Latency",
        "",
        frame_to_markdown(latency_df),
        "",
        "## 输出文件",
        "",
        f"- manifest.csv: `{output_dir / 'manifest.csv'}`",
        f"- metadata.json: `{output_dir / 'metadata.json'}`",
        f"- status.json: `{output_dir / 'status.json'}`",
        f"- latency_summary.csv: `{output_dir / 'latency_summary.csv'}`",
        "",
    ]
    (output_dir / "main_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 prediction cache shard 构建。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_prediction_cache_96_48_s_1k_shard"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(
        json.dumps({"status": "running", "updated_at": display_time(), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sample_df_all = load_sample_manifest(args.sample_manifest_path)
    sample_df = filter_sample_shard(sample_df_all, int(args.shard_index), int(args.shard_count))
    config_paths = resolve_config_paths(args.config_paths, args.models)
    all_records = []
    all_latency_rows: List[Dict[str, object]] = []
    model_names: List[str] = []
    config_names: List[str] = []
    shared_y_true_cache: Dict[Path, np.ndarray] = {}
    started_at = display_time()

    try:
        for config_path in config_paths:
            data_config, model_config, training_config, model = prepare_model(config_path, int(args.local_rank))
            del training_config
            model_name = str(model_config.model_name)
            config_name = config_name_from_path(config_path)
            model_names.append(model_name)
            config_names.append(config_name)
            checkpoint_selection = normalize_checkpoint_selection(model_name, getattr(model_config, "checkpoint_path", None))
            model_records, latency_rows = build_cache_for_model(
                sample_df=sample_df,
                data_config=data_config,
                model=model,
                model_name=model_name,
                config_name=config_name,
                checkpoint_selection=checkpoint_selection,
                output_dir=output_dir,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                shared_y_true_cache=shared_y_true_cache,
            )
            all_records.extend(model_records)
            all_latency_rows.extend(latency_rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        manifest_df = records_to_frame(all_records)
        validate_manifest_frame(
            manifest_df,
            expected_models=model_names if len(model_names) == len(MODEL_DISPLAY_ORDER) else None,
            require_shared_y_true_path=True,
        )
        latency_df = pd.DataFrame(all_latency_rows)
        manifest_df.to_csv(output_dir / "manifest.csv", index=False)
        latency_df.to_csv(output_dir / "latency_summary.csv", index=False)

        metadata: Dict[str, object] = {
            "generated_at": display_time(),
            "started_at": started_at,
            "status": "completed",
            "output_dir": str(output_dir),
            "sample_manifest_path": str(args.sample_manifest_path),
            "sample_manifest_total_count": int(len(sample_df_all)),
            "sample_count": int(len(sample_df)),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "config_paths": [str(path) for path in config_paths],
            "config_names": sorted(set(config_names)),
            "model_names": model_names,
            "metric": args.metric,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "local_rank": int(args.local_rank),
            "device_note": args.device_note,
            "record_count": int(len(manifest_df)),
            "shared_y_true_path": True,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary(output_dir, manifest_df, latency_df, metadata)
        (output_dir / "status.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"wrote prediction cache shard to {output_dir}")
        print(f"sample_count={len(sample_df)} record_count={len(manifest_df)} models={model_names}")
        preview_cols = ["sample_key", "model_name", "mae", "mse", "y_true_path", "y_pred_path"]
        print(manifest_df[preview_cols].head(int(args.print_rows)).to_string(index=False))
    except Exception as exc:
        status = {
            "status": "failed",
            "updated_at": display_time(),
            "output_dir": str(output_dir),
            "error": repr(exc),
        }
        (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
