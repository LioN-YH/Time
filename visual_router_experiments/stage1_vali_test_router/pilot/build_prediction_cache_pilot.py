#!/usr/bin/env python3
"""
文件功能：
    为 Visual Router Stage 1 生成小规模 window-level prediction cache pilot。

默认行为：
    - 使用 DLinear 的 96_48_S evaluate config；也可通过 --config-paths 输入多个专家 config；
    - 只导出 test split；
    - 只选第一个 item；
    - 只保存前 2 个 window；
    - 输出 manifest.csv、metadata.json 和 y_true/y_pred 的 .npy 文件。

设计说明：
    该脚本先验证 cache key、y_true/y_pred 形状、窗口级 MAE/MSE 和数组落盘口径，
    不用于全量正式结果。全量五专家 cache 应在 pilot 验证通过后另行扩展。
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from omegaconf import OmegaConf
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
from quito.models import AutoModel  # noqa: E402
from visual_router_experiments.common.prediction_cache_schema import (  # noqa: E402
    PredictionCacheKey,
    make_prediction_record,
    records_to_frame,
    validate_manifest_frame,
)


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

DEFAULT_FIVE_EXPERT_CONFIGS = [
    WORKSPACE / "quito" / "outputs" / "default_baseline" / "dlinear" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    WORKSPACE / "quito" / "outputs" / "default_baseline" / "patchtst" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    WORKSPACE / "quito" / "outputs" / "default_baseline" / "crossformer" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    WORKSPACE / "quito" / "outputs" / "statistical_baseline" / "es" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
    WORKSPACE / "quito" / "outputs" / "statistical_baseline" / "snaive" / "96_48_S" / "seed_16" / "EVALUATE" / "ver_0" / "config.yaml",
]


def now_token() -> str:
    """函数功能：生成 run 目录中的时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 pilot cache builder 参数。"""
    parser = argparse.ArgumentParser(description="Build a small prediction cache pilot for Stage 1.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Quito evaluate config 路径；默认使用 DLinear 96_48_S 已完成 evaluate config。",
    )
    parser.add_argument(
        "--config-paths",
        type=Path,
        nargs="+",
        default=None,
        help="多个 Quito evaluate config 路径；提供后会覆盖 --config-path，用于五专家 pilot。",
    )
    parser.add_argument(
        "--five-expert-96-48-s",
        action="store_true",
        help="使用当前已有五专家 96_48_S evaluate config 运行 pilot。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["vali", "test"],
        default=["test"],
        help="需要导出的 split；pilot 默认只导出 test。",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=1,
        help="每个 dataset 最多导出的 item 数。",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=2,
        help="每个 item-channel 最多导出的 window 数。",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=1,
        help="每个 item 最多导出的 channel 数；pilot 默认只导出 channel 0。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="pilot DataLoader batch size。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入 experiment_logs/run_outputs 下的时间戳目录。",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=20,
        help="运行结束时最多打印的 manifest 行数，避免扩大 pilot 后输出过长。",
    )
    return parser.parse_args()


def resolve_config_paths(args: argparse.Namespace) -> List[Path]:
    """函数功能：根据命令行参数解析本次 pilot 需要加载的专家 config 列表。"""
    if args.five_expert_96_48_s:
        return DEFAULT_FIVE_EXPERT_CONFIGS
    if args.config_paths:
        return args.config_paths
    return [args.config_path]


def mode_from_split(split: str) -> ModeType:
    """函数功能：把 Stage 1 split 名称映射到 Quito ModeType。"""
    if split == "vali":
        return ModeType.VALID
    if split == "test":
        return ModeType.TEST
    raise ValueError(f"未知 split：{split}")


def config_name_from_path(config_path: Path) -> str:
    """函数功能：从已有 evaluate config 路径中提取配置名，例如 96_48_S。"""
    parts = config_path.resolve().parts
    for part in ["96_48_S", "576_288_S", "1024_512_S"]:
        if part in parts:
            return part
    return config_path.parent.name


def relative_to_output(path: Path, output_dir: Path) -> Path:
    """函数功能：生成写入 manifest 的相对路径，保证输出目录可整体移动。"""
    return path.relative_to(output_dir)


def prepare_model(config_path: Path) -> tuple:
    """
    函数功能：
        读取 Quito config 并加载冻结专家模型。

    返回：
        data_config、model_config、training_config、model。
    """
    config = OmegaConf.load(config_path)
    data_config, model_config, training_config = AutoConfig.from_config(
        config=config,
        rank=-1,
        world_size=-1,
        local_rank=-1,
    )
    model = AutoModel.from_config(model_config, local_rank=-1)
    model.device = "cpu"
    model = model.to("cpu")
    model.metrics = training_config.eval_metrics
    model.eval()
    return data_config, model_config, training_config, model


def normalize_checkpoint_selection(model_name: str, checkpoint_path) -> str:
    """函数功能：为 manifest 记录专家 checkpoint 选择口径。"""
    if model_name in {"ES", "NaiveForecaster"}:
        return "not_applicable_statistical_model"
    return "validation_mae_best_or_config_defined"


def selected_user_ids(dataset, max_items: int) -> List[int]:
    """函数功能：按稳定顺序选择少量 item，避免 pilot 意外跑全量。"""
    user_ids = sorted(int(user_id) for user_id in dataset.get_all_ids())
    return user_ids[:max_items]


def save_array(path: Path, array: np.ndarray) -> None:
    """函数功能：保存单个窗口数组，父目录不存在时自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def build_cache_for_split(
    *,
    split: str,
    data_config,
    training_config,
    model,
    model_name: str,
    config_name: str,
    checkpoint_selection: str,
    output_dir: Path,
    max_items: int,
    max_windows: int,
    max_channels: int,
    batch_size: int,
) -> List:
    """函数功能：为单个 split 生成小规模 prediction cache records。"""
    datasets = load_datasets(
        data_config=data_config,
        task=TaskType.EVALUATE,
        mode=mode_from_split(split),
        cleanup=False,
        concat=False,
    )
    records = []
    arrays_dir = output_dir / "arrays"

    for dataset_idx, dataset in enumerate(datasets):
        dataset_name = getattr(dataset, "name", None) or f"dataset_{dataset_idx}"
        for item_id in selected_user_ids(dataset, max_items=max_items):
            item_dataset = deepcopy(dataset)
            item_dataset.select_user_data(item_id)
            # S 配置下 item-channel 被展开到样本轴。select_user_data 后 data.shape[0]
            # 就是该 item 的 channel 序列数，__getitem__ 的全局样本序号可反推 channel/window。
            channel_count = int(item_dataset.data.shape[0])
            selected_channel_count = min(channel_count, max_channels)
            len_per_channel = len(item_dataset) // channel_count

            dataloader = DataLoader(
                item_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            saved_by_channel = {channel_id: 0 for channel_id in range(selected_channel_count)}
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    loss_dict, predictions = model.eval_step(batch)
                    del loss_dict
                    y_true_batch = batch["y"][:, -model.forecast_horizon :, :].cpu().numpy()
                    y_pred_batch = predictions.cpu().numpy()

                    for row_in_batch in range(y_pred_batch.shape[0]):
                        if all(count >= max_windows for count in saved_by_channel.values()):
                            break

                        global_sample_index = batch_idx * batch_size + row_in_batch
                        channel_id = global_sample_index // len_per_channel
                        window_index = global_sample_index % len_per_channel
                        if channel_id >= selected_channel_count:
                            continue
                        if saved_by_channel[channel_id] >= max_windows:
                            continue

                        key = PredictionCacheKey(
                            config_name=config_name,
                            split=split,
                            dataset_name=dataset_name,
                            item_id=item_id,
                            channel_id=channel_id,
                            window_index=window_index,
                        )
                        sample_key = key.as_string()
                        y_true = y_true_batch[row_in_batch]
                        y_pred = y_pred_batch[row_in_batch]
                        y_true_path = arrays_dir / split / dataset_name / model_name / f"{sample_key}__y_true.npy"
                        y_pred_path = arrays_dir / split / dataset_name / model_name / f"{sample_key}__y_pred.npy"
                        save_array(y_true_path, y_true)
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
                        saved_by_channel[channel_id] += 1

                    if all(count >= max_windows for count in saved_by_channel.values()):
                        break

    return records


def main() -> None:
    """函数功能：执行小规模 prediction cache pilot 并写出 manifest。"""
    args = parse_args()
    output_dir = args.output_dir or RUN_OUTPUT_ROOT / f"{now_token()}_visual_router_stage1_prediction_cache_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_paths = resolve_config_paths(args)
    all_records = []
    model_names = []
    config_names = []
    for config_path in config_paths:
        data_config, model_config, training_config, model = prepare_model(config_path)
        config_name = config_name_from_path(config_path)
        model_name = model_config.model_name
        model_names.append(model_name)
        config_names.append(config_name)
        checkpoint_selection = normalize_checkpoint_selection(
            model_name=model_name,
            checkpoint_path=getattr(model_config, "checkpoint_path", None),
        )

        for split in args.splits:
            split_records = build_cache_for_split(
                split=split,
                data_config=data_config,
                training_config=training_config,
                model=model,
                model_name=model_name,
                config_name=config_name,
                checkpoint_selection=checkpoint_selection,
                output_dir=output_dir,
                max_items=args.max_items,
                max_windows=args.max_windows,
                max_channels=args.max_channels,
                batch_size=args.batch_size,
            )
            all_records.extend(split_records)

        # 主动释放模型引用，避免五专家 pilot 在小显存/内存环境中堆积。
        del model

    manifest_df = records_to_frame(all_records)
    validate_manifest_frame(manifest_df)
    manifest_df.to_csv(output_dir / "manifest.csv", index=False)

    metadata: Dict[str, object] = {
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "config_paths": [str(path) for path in config_paths],
        "config_names": sorted(set(config_names)),
        "model_names": model_names,
        "splits": args.splits,
        "max_items": args.max_items,
        "max_windows": args.max_windows,
        "max_channels": args.max_channels,
        "batch_size": args.batch_size,
        "record_count": int(len(manifest_df)),
        "pilot_scope": "small_window_cache_validation",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"wrote prediction cache pilot to {output_dir}")
    preview_cols = ["sample_key", "model_name", "mae", "mse", "y_true_path", "y_pred_path"]
    print(f"record_count={len(manifest_df)}")
    print(manifest_df[preview_cols].head(args.print_rows).to_string(index=False))
    if len(manifest_df) > args.print_rows:
        print(f"... omitted {len(manifest_df) - args.print_rows} rows")


if __name__ == "__main__":
    main()
