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
    """函数功能：复制数组文件，若目标已存在则校验内容一致。"""
    if not src.exists():
        raise FileNotFoundError(f"找不到数组文件：{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        src_array = np.load(src).astype(np.float32)
        dst_array = np.load(dst).astype(np.float32)
        if not np.array_equal(src_array, dst_array):
            raise ValueError(f"目标数组已存在但内容不一致：{dst}")
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
    rewritten_rows: List[Dict[str, object]] = []
    y_true_by_sample: Dict[str, Path] = {}

    for row in combined_df.itertuples(index=False):
        row_dict = row._asdict()
        shard_dir = Path(str(row_dict.pop("source_shard_dir")))
        sample_key = str(row_dict["sample_key"])
        model_name = str(row_dict["model_name"])

        src_true = resolve_array_path(str(row_dict["y_true_path"]), shard_dir)
        src_pred = resolve_array_path(str(row_dict["y_pred_path"]), shard_dir)
        dst_true = output_dir / "arrays" / "y_true" / str(row_dict["split"]) / str(row_dict["dataset_name"]) / f"{sample_key}__y_true.npy"
        dst_pred = (
            output_dir
            / "arrays"
            / "y_pred"
            / model_name
            / str(row_dict["split"])
            / str(row_dict["dataset_name"])
            / f"{sample_key}__y_pred.npy"
        )

        if sample_key in y_true_by_sample:
            # 同一 sample_key 可能来自不同专家 shard；复制前先确认 y_true 内容一致。
            existing = y_true_by_sample[sample_key]
            if not np.array_equal(np.load(existing).astype(np.float32), np.load(src_true).astype(np.float32)):
                raise ValueError(f"sample_key={sample_key} 的 y_true 在 shard 间不一致")
        else:
            safe_copy_array(src_true, dst_true)
            y_true_by_sample[sample_key] = dst_true
        safe_copy_array(src_pred, dst_pred)

        row_dict["y_true_path"] = str(dst_true.relative_to(output_dir))
        row_dict["y_pred_path"] = str(dst_pred.relative_to(output_dir))
        rewritten_rows.append(row_dict)

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
