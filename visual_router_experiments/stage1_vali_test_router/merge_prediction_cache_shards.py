#!/usr/bin/env python3
"""
文件功能：
    合并 Stage 1 prediction cache shard，并在合并前后做完整性校验。

输入：
    - 一个或多个 `build_prediction_cache_from_manifest.py` 输出目录；
    - 每个目录必须包含 manifest.csv 和数组文件。

输出：
    - 合并后的 manifest.csv；
    - 去重后的 arrays/y_true/*.npy；
    - 复制后的 arrays/y_pred/{model_name}/*.npy；
    - metadata.json、status.json、merge_summary.md。

关键约束：
    - 合并前检查 `sample_key + model_name` 不重复；
    - 合并后要求每个 sample_key 覆盖五专家；
    - 同一 sample_key 的 y_true 内容必须一致，合并目录中只保留一份共享 y_true；
    - 本脚本只做精确复制，不删除原 shard。
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
MODEL_DISPLAY_ORDER = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.common.prediction_cache_schema import validate_manifest_frame  # noqa: E402
from visual_router_experiments.common.prediction_array_io import (  # noqa: E402
    PACKED_NPY_STORAGE,
    load_prediction_array,
    resolve_cache_array_path,
)


def now_token() -> str:
    """函数功能：生成输出目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata/status/summary 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 shard 合并参数。"""
    parser = argparse.ArgumentParser(description="Merge Stage 1 prediction cache shards.")
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True, help="待合并 shard 输出目录列表。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="合并输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录；默认生成时间戳目录。")
    parser.add_argument("--expected-models", nargs="+", default=MODEL_DISPLAY_ORDER, help="每个 sample_key 必须覆盖的专家列表。")
    parser.add_argument("--print-rows", type=int, default=10, help="运行结束时打印多少行 manifest 预览。")
    return parser.parse_args()


def resolve_array_path(path_text: str, shard_dir: Path) -> Path:
    """函数功能：解析 shard manifest 中的相对或绝对数组路径。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return shard_dir / path


def safe_copy_array(src: Path, dst: Path) -> None:
    """
    函数功能：复制数组文件，若目标已存在则在内容一致时跳过、不一致时覆盖。

    说明：
        full-scale merge 可能因为上一次失败而留下半成品目标文件。这里选择覆盖而
        不报错，目的是让同一批 shard 在同一输出目录下可恢复、可重复执行；真正的
        合并契约仍由 manifest 校验和 shard 输入一致性约束来保证。
    """
    if not src.exists():
        raise FileNotFoundError(f"找不到数组文件：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if filecmp.cmp(src, dst, shallow=False):
            return
    shutil.copy2(src, dst)


def load_shard_manifest(shard_dir: Path) -> pd.DataFrame:
    """函数功能：读取单个 shard manifest 并补充来源目录字段。"""
    manifest_path = shard_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 shard manifest：{manifest_path}")
    df = pd.read_csv(manifest_path)
    df["source_shard_dir"] = str(shard_dir)
    return df


def copy_and_rewrite_paths(combined_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    函数功能：
        复制 shard 数组到合并目录，并重写 manifest 的 y_true/y_pred 相对路径。
    """
    if "array_storage" in combined_df.columns and (combined_df["array_storage"].astype(str) == PACKED_NPY_STORAGE).any():
        return copy_packed_and_rewrite_paths(combined_df, output_dir)

    return copy_per_sample_and_rewrite_paths(combined_df, output_dir)


def copy_per_sample_and_rewrite_paths(combined_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """函数功能：兼容早期 per-sample `.npy` shard 的合并路径。"""
    rewritten_rows: List[Dict[str, object]] = []
    y_true_by_sample: Dict[str, np.ndarray] = {}
    copied_files: Dict[Path, Path] = {}

    for row in combined_df.itertuples(index=False):
        row_dict = row._asdict()
        shard_dir = Path(str(row_dict.pop("source_shard_dir")))
        sample_key = str(row_dict["sample_key"])
        model_name = str(row_dict["model_name"])

        src_true = resolve_cache_array_path(str(row_dict["y_true_path"]), shard_dir)
        src_pred = resolve_cache_array_path(str(row_dict["y_pred_path"]), shard_dir)
        dst_true = output_dir / Path(str(row_dict["y_true_path"]))
        dst_pred = output_dir / Path(str(row_dict["y_pred_path"]))

        true_record = dict(row_dict)
        true_record["y_true_path"] = str(src_true)
        true_record["y_pred_path"] = str(src_pred)
        current_true = load_prediction_array(true_record, "y_true")
        if sample_key in y_true_by_sample:
            if not np.array_equal(y_true_by_sample[sample_key], current_true):
                raise ValueError(f"sample_key={sample_key} 的 y_true 在 shard 间不一致")
        else:
            y_true_by_sample[sample_key] = np.asarray(current_true, dtype=np.float32).copy()

        if src_true not in copied_files:
            safe_copy_array(src_true, dst_true)
            copied_files[src_true] = dst_true
        if src_pred not in copied_files:
            safe_copy_array(src_pred, dst_pred)
            copied_files[src_pred] = dst_pred

        row_dict["y_true_path"] = str(dst_true.relative_to(output_dir))
        row_dict["y_pred_path"] = str(dst_pred.relative_to(output_dir))
        rewritten_rows.append(row_dict)

    return pd.DataFrame(rewritten_rows)


def _source_token_map(combined_df: pd.DataFrame) -> Dict[str, str]:
    """函数功能：为每个来源 shard 生成稳定、路径友好的 token。"""
    shard_dirs = sorted(set(combined_df["source_shard_dir"].astype(str).tolist()))
    return {shard_dir: f"source_{idx:04d}" for idx, shard_dir in enumerate(shard_dirs)}


def copy_packed_and_rewrite_paths(combined_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    函数功能：
        合并 packed_npy_v1 shard，并重建合并目录中的共享 y_true packed 文件。

    设计说明：
        不同 sample shard 的 `arrays/packed/y_true/.../y_true.npy` 相对路径相同，
        但内容不同，不能简单复制到同一个目标路径。这里按 sample_key 重新分配
        merged y_true row index；y_pred packed 文件则按来源 shard token 复制，保留
        原 row index。
    """
    rewritten_rows: List[Dict[str, object]] = []
    source_tokens = _source_token_map(combined_df)
    copied_pred_files: Dict[Path, Path] = {}
    true_buffers: Dict[Tuple[str, str], List[np.ndarray]] = {}
    true_ref_by_sample: Dict[str, Tuple[str, str, int, np.ndarray]] = {}

    for row in combined_df.itertuples(index=False):
        row_dict = row._asdict()
        shard_dir = Path(str(row_dict.pop("source_shard_dir")))
        source_token = source_tokens[str(shard_dir)]
        sample_key = str(row_dict["sample_key"])
        split = str(row_dict["split"])
        dataset_name = str(row_dict["dataset_name"])

        src_true = resolve_cache_array_path(str(row_dict["y_true_path"]), shard_dir)
        src_pred = resolve_cache_array_path(str(row_dict["y_pred_path"]), shard_dir)
        source_record = dict(row_dict)
        source_record["y_true_path"] = str(src_true)
        source_record["y_pred_path"] = str(src_pred)
        current_true = load_prediction_array(source_record, "y_true")

        true_key = (split, dataset_name)
        if sample_key in true_ref_by_sample:
            old_split, old_dataset, row_index, old_true = true_ref_by_sample[sample_key]
            if old_split != split or old_dataset != dataset_name:
                raise ValueError(f"sample_key={sample_key} 的 split/dataset 不一致")
            if not np.array_equal(old_true, current_true):
                raise ValueError(f"sample_key={sample_key} 的 y_true 在 shard 间不一致")
        else:
            buffer = true_buffers.setdefault(true_key, [])
            row_index = len(buffer)
            buffer.append(np.asarray(current_true, dtype=np.float32).copy())
            true_ref_by_sample[sample_key] = (split, dataset_name, row_index, np.asarray(current_true, dtype=np.float32).copy())

        dst_pred = output_dir / "arrays" / "source_shards" / source_token / Path(str(row_dict["y_pred_path"]))
        if src_pred not in copied_pred_files:
            safe_copy_array(src_pred, dst_pred)
            copied_pred_files[src_pred] = dst_pred

        _, _, true_row_index, _ = true_ref_by_sample[sample_key]
        dst_true_rel = Path("arrays") / "packed" / "y_true" / split / dataset_name / "y_true.npy"
        row_dict["y_true_path"] = str(dst_true_rel)
        row_dict["y_pred_path"] = str(copied_pred_files[src_pred].relative_to(output_dir))
        row_dict["array_storage"] = PACKED_NPY_STORAGE
        row_dict["y_true_row_index"] = int(true_row_index)
        rewritten_rows.append(row_dict)

    for (split, dataset_name), arrays in true_buffers.items():
        dst_true = output_dir / "arrays" / "packed" / "y_true" / split / dataset_name / "y_true.npy"
        dst_true.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_true, np.stack(arrays, axis=0).astype(np.float32))

    return pd.DataFrame(rewritten_rows)


def validate_merged_manifest(manifest_df: pd.DataFrame, expected_models: List[str]) -> None:
    """函数功能：执行合并后 manifest 契约校验。"""
    duplicate_count = int(manifest_df.duplicated(["sample_key", "model_name"]).sum())
    if duplicate_count:
        dup = manifest_df.loc[manifest_df.duplicated(["sample_key", "model_name"]), ["sample_key", "model_name"]].head(10)
        raise ValueError(f"合并前发现 {duplicate_count} 条 sample_key/model_name 重复，示例：{dup.to_dict('records')}")
    validate_manifest_frame(
        manifest_df,
        expected_models=expected_models,
        require_shared_y_true_path=True,
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


def write_summary(output_dir: Path, manifest_df: pd.DataFrame, metadata: Mapping[str, object]) -> None:
    """函数功能：写出中文 Markdown 合并摘要。"""
    coverage = (
        manifest_df.groupby(["config_name", "split", "dataset_name", "model_name"])
        .size()
        .reset_index(name="rows")
    )
    model_counts = manifest_df.groupby("sample_key")["model_name"].nunique().value_counts().reset_index()
    model_counts.columns = ["model_count_per_sample", "sample_count"]
    lines = [
        "# Stage 1 Prediction Cache Merge Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 输入 Shard",
        "",
        "\n".join(f"- `{path}`" for path in metadata["shard_dirs"]),
        "",
        "## 覆盖统计",
        "",
        frame_to_markdown(coverage),
        "",
        "## 每个 Sample 的专家数",
        "",
        frame_to_markdown(model_counts),
        "",
        "## 输出文件",
        "",
        f"- manifest.csv: `{output_dir / 'manifest.csv'}`",
        f"- metadata.json: `{output_dir / 'metadata.json'}`",
        f"- status.json: `{output_dir / 'status.json'}`",
        "",
    ]
    (output_dir / "merge_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """函数功能：执行 Stage 1 prediction cache shard 合并。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_prediction_cache_merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    status_path.write_text(
        json.dumps({"status": "running", "updated_at": display_time(), "output_dir": str(output_dir)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        shard_frames = [load_shard_manifest(path) for path in args.shard_dirs]
        combined_df = pd.concat(shard_frames, ignore_index=True)
        pre_duplicate_count = int(combined_df.duplicated(["sample_key", "model_name"]).sum())
        if pre_duplicate_count:
            raise ValueError(f"输入 shard 存在 {pre_duplicate_count} 条 sample_key/model_name 重复")

        manifest_df = copy_and_rewrite_paths(combined_df, output_dir)
        validate_merged_manifest(manifest_df, list(args.expected_models))
        manifest_df = manifest_df.sort_values(["config_name", "split", "dataset_name", "item_id", "channel_id", "window_index", "model_name"])
        manifest_df.to_csv(output_dir / "manifest.csv", index=False)

        metadata: Dict[str, object] = {
            "status": "completed",
            "generated_at": display_time(),
            "output_dir": str(output_dir),
            "shard_dirs": [str(path) for path in args.shard_dirs],
            "expected_models": list(args.expected_models),
            "sample_count": int(manifest_df["sample_key"].nunique()),
            "record_count": int(len(manifest_df)),
            "shared_y_true_path": True,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        status_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_summary(output_dir, manifest_df, metadata)

        print(f"wrote merged prediction cache to {output_dir}")
        print(f"sample_count={manifest_df['sample_key'].nunique()} record_count={len(manifest_df)}")
        preview_cols = ["sample_key", "model_name", "mae", "mse", "y_true_path", "y_pred_path"]
        print(manifest_df[preview_cols].head(int(args.print_rows)).to_string(index=False))
    except Exception as exc:
        status = {
            "status": "failed",
            "updated_at": display_time(),
            "output_dir": str(output_dir),
            "error": repr(exc),
        }
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
