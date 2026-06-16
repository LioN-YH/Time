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
import re
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


def source_token_map_from_dirs(shard_dirs: List[Path]) -> Dict[str, str]:
    """函数功能：为流式合并输入目录生成与旧逻辑一致的稳定 source token。"""
    return {str(shard_dir): f"source_{idx:04d}" for idx, shard_dir in enumerate(sorted(shard_dirs))}


def parse_shard_identity(shard_dir: Path) -> Tuple[str, int]:
    """
    函数功能：从标准 shard 目录解析专家名和 sample shard 编号。

    输入目录形如 `shards/DLinear/sample_shard_0000_of_0064`。早期 dry-run
    使用 `sample_shard_0000`，这里也兼容，方便用历史小样本回归正式逻辑。
    """
    match = re.search(r"sample_shard_(\d{4})(?:_of_\d{4})?$", shard_dir.name)
    if not match:
        raise ValueError(f"无法从 shard 目录名解析 sample shard 编号：{shard_dir}")
    return shard_dir.parent.name, int(match.group(1))


def shard_manifest_is_packed(shard_dir: Path) -> bool:
    """函数功能：只读取 shard manifest 首行，判断是否为 packed_npy_v1。"""
    manifest_path = shard_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 shard manifest：{manifest_path}")
    preview = pd.read_csv(manifest_path, usecols=["array_storage"], nrows=1)
    if preview.empty:
        raise ValueError(f"shard manifest 为空：{manifest_path}")
    return str(preview["array_storage"].iloc[0]) == PACKED_NPY_STORAGE


def all_shards_are_packed(shard_dirs: List[Path]) -> bool:
    """函数功能：判断当前 merge 输入是否全为 full-scale 推荐的 packed shard。"""
    return bool(shard_dirs) and all(shard_manifest_is_packed(path) for path in shard_dirs)


def build_shard_grid(shard_dirs: List[Path], expected_models: List[str]) -> Dict[int, Dict[str, Path]]:
    """
    函数功能：按 sample shard 编号组织五专家输入目录，并检查专家集合完整。

    这样流式合并可以一次只读同一个 sample shard 下的五专家 manifest，避免
    full-scale 场景把 1.16 亿行 manifest 全量加载进单个 DataFrame。
    """
    grid: Dict[int, Dict[str, Path]] = {}
    for shard_dir in shard_dirs:
        model_name, shard_index = parse_shard_identity(shard_dir)
        if model_name in grid.setdefault(shard_index, {}):
            raise ValueError(f"重复的 model/sample shard 输入：model={model_name} shard={shard_index:04d}")
        grid[shard_index][model_name] = shard_dir

    expected_set = set(expected_models)
    for shard_index, by_model in sorted(grid.items()):
        actual_set = set(by_model)
        if actual_set != expected_set:
            raise ValueError(
                f"sample_shard={shard_index:04d} 专家集合不完整："
                f"expected={sorted(expected_set)} actual={sorted(actual_set)}"
            )
    return grid


def output_true_path(split: str, dataset_name: str) -> str:
    """函数功能：生成 merged cache 中共享 y_true packed 文件的相对路径。"""
    return str(Path("arrays") / "packed" / "y_true" / split / dataset_name / "y_true.npy")


def safe_relative_fragment(path_text: str) -> Path:
    """
    函数功能：把 manifest 中的数组路径转换成可拼到 source token 之后的相对片段。

    full-scale shard 当前写相对路径；这里兼容绝对路径，避免 `Path` 遇到绝对路径
    时吞掉前缀目录。
    """
    path = Path(path_text)
    if path.is_absolute():
        return Path(*path.parts[1:])
    return path


def validate_single_packed_manifest(df: pd.DataFrame, shard_dir: Path, model_name: str) -> None:
    """函数功能：校验单个 packed shard manifest 的基础字段和单专家唯一性。"""
    if df.empty:
        raise ValueError(f"shard manifest 为空：{shard_dir / 'manifest.csv'}")
    if set(df["array_storage"].astype(str).unique()) != {PACKED_NPY_STORAGE}:
        raise ValueError(f"shard 不是纯 packed_npy_v1：{shard_dir}")
    if set(df["model_name"].astype(str).unique()) != {model_name}:
        raise ValueError(f"shard model_name 与目录不一致：dir={model_name} values={df['model_name'].unique()}")
    duplicate_count = int(df.duplicated(["sample_key", "model_name"]).sum())
    if duplicate_count:
        raise ValueError(f"shard 内 sample_key/model_name 重复：{shard_dir} duplicate_count={duplicate_count}")
    if int(df["sample_key"].nunique()) != int(len(df)):
        raise ValueError(f"单专家 shard 内 sample_key 不唯一：{shard_dir}")


def validate_model_alignment(reference_df: pd.DataFrame, current_df: pd.DataFrame, shard_dir: Path) -> None:
    """
    函数功能：确认同一 sample shard 下不同专家的窗口顺序和共享 y_true 引用一致。

    full-scale launcher 对每个专家使用同一个 sample manifest shard。若这些稳定字段
    不一致，后续按 offset 重写共享 y_true 行号就不再可靠，必须立即失败。
    """
    stable_cols = [
        "sample_key",
        "config_name",
        "split",
        "dataset_name",
        "item_id",
        "channel_id",
        "window_index",
        "history_length",
        "pred_length",
        "y_true_path",
    ]
    if list(reference_df[stable_cols].columns) != stable_cols:
        raise ValueError("reference manifest 缺少稳定字段")
    if not reference_df[stable_cols].equals(current_df[stable_cols]):
        mismatch_mask = (reference_df[stable_cols] != current_df[stable_cols]).any(axis=1)
        first_bad = int(mismatch_mask.to_numpy().nonzero()[0][0]) if mismatch_mask.any() else -1
        raise ValueError(f"五专家 manifest 稳定字段不一致：{shard_dir} first_bad_row={first_bad}")


def validate_packed_row_indices(row_indices: np.ndarray, array: np.ndarray, array_path: Path, context: str) -> None:
    """
    函数功能：校验 manifest 引用的 packed row index 均能在目标数组中读取。

    说明：
        full-scale shard 曾出现过已完成 shard 被恢复逻辑重复追加数组尾部的情况。
        这种情况下整文件 shape 可能大于 manifest 实际引用行数，但只要 row index
        范围内的数据一致，merge 应以 manifest 契约为准，并忽略未引用尾部。
    """
    if row_indices.size == 0:
        raise ValueError(f"packed manifest 没有引用任何行：{context} path={array_path}")
    if int(row_indices.min()) < 0:
        raise ValueError(f"packed row index 出现负数：{context} path={array_path}")
    if int(row_indices.max()) >= int(array.shape[0]):
        raise ValueError(
            f"packed row index 超出数组范围：{context} path={array_path} "
            f"array_rows={array.shape[0]} max_index={int(row_indices.max())}"
        )
    if int(np.unique(row_indices).size) != int(row_indices.size):
        raise ValueError(f"packed row index 在同一文件内重复：{context} path={array_path}")


def packed_rows_view_or_copy(array: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    """
    函数功能：按 manifest row index 读取 packed 行，连续索引走切片避免大规模拷贝。

    full-scale shard 的 row index 正常都是 `0..N-1` 连续前缀；只有在非常规索引时
    才退回 fancy indexing。这样既保留 row-index 契约，又避免每个 shard 额外复制
    数百 MB 的 y_true 数据。
    """
    if row_indices.size and int(row_indices[0]) == 0 and int(row_indices[-1]) == int(row_indices.size) - 1:
        expected = np.arange(int(row_indices.size), dtype=np.int64)
        if np.array_equal(row_indices, expected):
            return array[: int(row_indices.size)]
    return array[row_indices]


def validate_y_true_arrays_match(reference_dir: Path, current_dir: Path, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> None:
    """
    函数功能：按 packed 文件比较同一 sample shard 下不同专家的 y_true 数组内容。

    这里按各自 manifest 引用的 packed row index 比较，而不是要求不同专家的
    row index 数字完全相同。断点恢复可能让某个专家引用重复追加后的后半段；
    只要 sample 顺序和 y_true 内容一致，merged cache 会重新写共享 y_true。
    """
    for path_text, group in reference_df.groupby("y_true_path", sort=True):
        current_group = current_df.loc[group.index]
        ref_path = resolve_cache_array_path(path_text, reference_dir)
        cur_path = resolve_cache_array_path(path_text, current_dir)
        ref_array = np.load(ref_path, mmap_mode="r")
        cur_array = np.load(cur_path, mmap_mode="r")
        if ref_array.shape[1:] != cur_array.shape[1:] or ref_array.dtype != cur_array.dtype:
            raise ValueError(
                f"y_true packed 文件尾部 shape/dtype 不一致：ref={ref_path} {ref_array.shape} {ref_array.dtype} "
                f"current={cur_path} {cur_array.shape} {cur_array.dtype}"
            )
        ref_indices = group["y_true_row_index"].to_numpy(dtype=np.int64)
        cur_indices = current_group["y_true_row_index"].to_numpy(dtype=np.int64)
        validate_packed_row_indices(ref_indices, ref_array, ref_path, f"reference y_true {reference_dir}")
        validate_packed_row_indices(cur_indices, cur_array, cur_path, f"current y_true {current_dir}")
        if not np.array_equal(packed_rows_view_or_copy(ref_array, ref_indices), packed_rows_view_or_copy(cur_array, cur_indices)):
            raise ValueError(f"y_true packed 文件引用行内容不一致：ref={ref_path} current={cur_path}")


def collect_true_array_plan(
    shard_grid: Mapping[int, Mapping[str, Path]],
    reference_model: str,
) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], Tuple[np.dtype, Tuple[int, ...]]]]:
    """
    函数功能：第一遍扫描 reference expert manifest，统计 merged y_true 的目标形状。

    只读取 manifest 和 `.npy` header，不读取数组主体；真正的数据拷贝在第二遍完成。
    """
    totals: Dict[Tuple[str, str], int] = {}
    shapes: Dict[Tuple[str, str], Tuple[np.dtype, Tuple[int, ...]]] = {}
    for shard_index in sorted(shard_grid):
        shard_dir = shard_grid[shard_index][reference_model]
        df = pd.read_csv(shard_dir / "manifest.csv", usecols=["split", "dataset_name", "y_true_path", "y_true_row_index"])
        for (split, dataset_name, path_text), group in df.groupby(["split", "dataset_name", "y_true_path"], sort=True):
            key = (str(split), str(dataset_name))
            src_path = resolve_cache_array_path(str(path_text), shard_dir)
            array = np.load(src_path, mmap_mode="r")
            expected_rows = int(group["y_true_row_index"].nunique())
            if int(array.shape[0]) != expected_rows:
                raise ValueError(f"packed y_true 行数与 manifest 不一致：{src_path} array_rows={array.shape[0]} manifest_rows={expected_rows}")
            totals[key] = totals.get(key, 0) + int(array.shape[0])
            tail = tuple(int(v) for v in array.shape[1:])
            dtype_tail = (array.dtype, tail)
            if key in shapes and shapes[key] != dtype_tail:
                raise ValueError(f"同一 split/dataset 的 y_true shape/dtype 不一致：key={key} old={shapes[key]} new={dtype_tail}")
            shapes[key] = dtype_tail
    return totals, shapes


def create_true_memmaps(
    output_dir: Path,
    totals: Mapping[Tuple[str, str], int],
    shapes: Mapping[Tuple[str, str], Tuple[np.dtype, Tuple[int, ...]]],
) -> Dict[Tuple[str, str], np.memmap]:
    """函数功能：为 merged cache 预创建共享 y_true `.npy` memmap。"""
    memmaps: Dict[Tuple[str, str], np.memmap] = {}
    for (split, dataset_name), total_rows in sorted(totals.items()):
        dtype, tail_shape = shapes[(split, dataset_name)]
        dst_path = output_dir / output_true_path(split, dataset_name)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        memmaps[(split, dataset_name)] = np.lib.format.open_memmap(
            dst_path,
            mode="w+",
            dtype=dtype,
            shape=(int(total_rows), *tail_shape),
        )
    return memmaps


def copy_packed_array_selected(src_path: Path, dst_path: Path, row_indices: np.ndarray, context: str) -> None:
    """
    函数功能：把 packed 数组按 manifest 引用行复制到目标路径，并压缩成连续行。

    说明：
        merged cache 的 source shard 文件是新文件，可以把源 shard 中任意合法 row
        index 压缩成 `0..N-1`。这能同时处理“重复尾部未引用”和“manifest 引用后半段”
        两类恢复残留，最终 merged manifest 会同步重写 y_pred_row_index。
    """
    if not src_path.exists():
        raise FileNotFoundError(f"找不到数组文件：{src_path}")
    src_array = np.load(src_path, mmap_mode="r")
    validate_packed_row_indices(row_indices, src_array, src_path, context)
    target_rows = int(row_indices.size)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        # 这里复用同一输出目录中的既有 source shard 文件。它们由同一个正式
        # merge 命令上一次失败前写出；只要 header 与 manifest 可引用范围一致，
        # 就无需再次深比较整块 y_pred，避免前 14 个已写 shard 重启时重复读数 GB。
        dst_array = np.load(dst_path, mmap_mode="r")
        if dst_array.shape == (target_rows, *src_array.shape[1:]) and dst_array.dtype == src_array.dtype:
            return
    sequential_prefix = (
        int(row_indices[0]) == 0
        and int(row_indices[-1]) == target_rows - 1
        and np.array_equal(row_indices, np.arange(target_rows, dtype=np.int64))
    )
    if sequential_prefix and int(src_array.shape[0]) == target_rows:
        safe_copy_array(src_path, dst_path)
        return
    np.save(dst_path, np.asarray(packed_rows_view_or_copy(src_array, row_indices), dtype=src_array.dtype))


def copy_y_pred_arrays_for_shard(df: pd.DataFrame, shard_dir: Path, output_dir: Path, source_token: str) -> None:
    """函数功能：复制一个专家 shard 的 packed y_pred 文件到 merged cache source_shards 目录。"""
    for path_text, group in df.groupby("y_pred_path", sort=True):
        src_path = resolve_cache_array_path(path_text, shard_dir)
        dst_rel = Path("arrays") / "source_shards" / source_token / safe_relative_fragment(path_text)
        row_indices = group["y_pred_row_index"].to_numpy(dtype=np.int64)
        copy_packed_array_selected(src_path, output_dir / dst_rel, row_indices, f"y_pred {shard_dir}")


def rewrite_packed_manifest_for_output(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    source_token: str,
    shard_true_offsets: Mapping[Tuple[str, str], int],
    model_order: int,
) -> pd.DataFrame:
    """
    函数功能：把单专家 shard manifest 改写为 merged cache manifest 口径。

    - `y_true_row_index` 从 shard 内局部行号改成 split/dataset 全局行号；
    - `y_pred_path` 加上 source shard token，保留原 packed row index；
    - 增加临时排序列，让同一个 sample_key 的五专家记录在输出 CSV 中相邻。
    """
    rewritten = df.copy()
    reference_true_row_index = reference_df["y_true_row_index"].astype(np.int64)
    for (split, dataset_name), offset in shard_true_offsets.items():
        mask = (rewritten["split"].astype(str) == split) & (rewritten["dataset_name"].astype(str) == dataset_name)
        if mask.any():
            rewritten.loc[mask, "y_true_path"] = output_true_path(split, dataset_name)
            # merged y_true 只从 reference expert 写入一次。其它专家即使源 shard
            # 引用了重复追加后的后半段，也必须改写到 reference 的共享行号。
            rewritten.loc[mask, "y_true_row_index"] = reference_true_row_index.loc[mask] + int(offset)
    rewritten["y_pred_path"] = "arrays/source_shards/" + source_token + "/" + rewritten["y_pred_path"].astype(str)
    for _path_text, group in rewritten.groupby("y_pred_path", sort=True):
        # y_pred 已按当前 manifest 顺序复制成新的 compact packed 文件，因此 merged
        # manifest 中的 row index 也重写为该文件内的连续局部行号。
        rewritten.loc[group.index, "y_pred_row_index"] = np.arange(len(group), dtype=np.int64)
    rewritten["_sample_order"] = np.arange(len(rewritten), dtype=np.int64)
    rewritten["_model_order"] = int(model_order)
    return rewritten


def write_streaming_merge_summary(
    output_dir: Path,
    coverage_rows: Mapping[Tuple[str, str, str, str], int],
    sample_count: int,
    metadata: Mapping[str, object],
) -> None:
    """函数功能：为流式 packed merge 写出与旧摘要等价的中文 Markdown 摘要。"""
    coverage = pd.DataFrame(
        [
            {
                "config_name": key[0],
                "split": key[1],
                "dataset_name": key[2],
                "model_name": key[3],
                "rows": value,
            }
            for key, value in sorted(coverage_rows.items())
        ]
    )
    model_counts = pd.DataFrame([{"model_count_per_sample": len(metadata["expected_models"]), "sample_count": sample_count}])
    shard_dirs = list(metadata["shard_dirs"])
    preview_paths = shard_dirs[:5] + (["..."] if len(shard_dirs) > 10 else []) + shard_dirs[-5:]
    lines = [
        "# Stage 1 Prediction Cache Merge Summary",
        "",
        f"生成时间：{metadata['generated_at']}",
        "",
        "## 输入 Shard",
        "",
        f"- shard_count: `{len(shard_dirs)}`",
        "\n".join(f"- `{path}`" for path in preview_paths),
        "",
        "## 合并策略",
        "",
        f"- merge_strategy: `{metadata['merge_strategy']}`",
        "- 说明：按 sample shard 流式读取五专家 manifest，整块比较 packed y_true，重建 merged y_true 全局 row index；不逐行重复打开 `.npy`。",
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


def merge_packed_shards_streaming(shard_dirs: List[Path], output_dir: Path, expected_models: List[str]) -> Dict[str, object]:
    """
    函数功能：针对 full-scale `packed_npy_v1` shard 执行流式合并。

    输入：
        shard_dirs: status.json `merge_command` 传入的 320 个 shard 输出目录。
        output_dir: merged cache 输出目录。
        expected_models: 每个 sample_key 必须覆盖的专家集合。

    输出：
        metadata 字典，同时落盘 manifest、共享 y_true、复制后的 y_pred、metadata/status/summary。

    关键约束：
        该函数是现有正式 merge 脚本的 full-scale 执行分支，不引入新的命令入口；
        它保持 `sample_key + model_name` 唯一、五专家完整和共享 y_true 一致的校验
        口径，只把实现从全量 DataFrame + 逐行数组读取改为按 sample shard 分块。
    """
    shard_grid = build_shard_grid(shard_dirs, expected_models)
    source_tokens = source_token_map_from_dirs(shard_dirs)
    reference_model = expected_models[0]
    totals, shapes = collect_true_array_plan(shard_grid, reference_model)
    true_memmaps = create_true_memmaps(output_dir, totals, shapes)
    true_offsets: Dict[Tuple[str, str], int] = {key: 0 for key in totals}
    coverage_rows: Dict[Tuple[str, str, str, str], int] = {}
    manifest_path = output_dir / "manifest.csv"
    first_write = True
    total_records = 0
    total_samples = 0

    for shard_position, shard_index in enumerate(sorted(shard_grid), start=1):
        by_model = shard_grid[shard_index]
        model_frames: Dict[str, pd.DataFrame] = {}
        for model_name in expected_models:
            shard_dir = by_model[model_name]
            df = pd.read_csv(shard_dir / "manifest.csv")
            validate_single_packed_manifest(df, shard_dir, model_name)
            model_frames[model_name] = df

        reference_df = model_frames[reference_model]
        total_samples += int(reference_df["sample_key"].nunique())
        for model_name in expected_models[1:]:
            validate_model_alignment(reference_df, model_frames[model_name], by_model[model_name])
            validate_y_true_arrays_match(by_model[reference_model], by_model[model_name], reference_df, model_frames[model_name])

        shard_true_offsets: Dict[Tuple[str, str], int] = {}
        for (split, dataset_name, path_text), group in reference_df.groupby(["split", "dataset_name", "y_true_path"], sort=True):
            key = (str(split), str(dataset_name))
            offset = true_offsets[key]
            shard_true_offsets[key] = offset
            src_path = resolve_cache_array_path(str(path_text), by_model[reference_model])
            src_array = np.load(src_path, mmap_mode="r")
            row_count = int(src_array.shape[0])
            expected_rows = int(group["y_true_row_index"].nunique())
            if row_count != expected_rows:
                raise ValueError(f"y_true packed 文件行数与 manifest 不一致：{src_path} array_rows={row_count} manifest_rows={expected_rows}")
            true_memmaps[key][offset : offset + row_count] = src_array
            true_offsets[key] = offset + row_count

        rewritten_frames: List[pd.DataFrame] = []
        output_columns = list(reference_df.columns)
        for model_order, model_name in enumerate(expected_models):
            shard_dir = by_model[model_name]
            df = model_frames[model_name]
            copy_y_pred_arrays_for_shard(df, shard_dir, output_dir, source_tokens[str(shard_dir)])
            for (config_name, split, dataset_name), count in df.groupby(["config_name", "split", "dataset_name"]).size().items():
                key = (str(config_name), str(split), str(dataset_name), model_name)
                coverage_rows[key] = coverage_rows.get(key, 0) + int(count)
            rewritten_frames.append(
                rewrite_packed_manifest_for_output(df, reference_df, source_tokens[str(shard_dir)], shard_true_offsets, model_order)
            )

        combined = pd.concat(rewritten_frames, ignore_index=True)
        combined = combined.sort_values(["_sample_order", "_model_order"], kind="mergesort")
        combined = combined[output_columns]
        combined.to_csv(manifest_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        total_records += int(len(combined))
        print(
            f"[{display_time()}] merged sample_shard={shard_index:04d} "
            f"progress={shard_position}/{len(shard_grid)} records_written={total_records}",
            flush=True,
        )

    for key, offset in true_offsets.items():
        if int(offset) != int(totals[key]):
            raise ValueError(f"merged y_true 行数不完整：key={key} written={offset} expected={totals[key]}")
        true_memmaps[key].flush()

    metadata: Dict[str, object] = {
        "status": "completed",
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "shard_dirs": [str(path) for path in shard_dirs],
        "expected_models": list(expected_models),
        "sample_count": int(total_samples),
        "record_count": int(total_records),
        "shared_y_true_path": True,
        "array_storage": PACKED_NPY_STORAGE,
        "merge_strategy": "packed_npy_v1_streaming_by_sample_shard",
        "true_array_rows": {f"{split}/{dataset_name}": int(count) for (split, dataset_name), count in sorted(totals.items())},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "status.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_streaming_merge_summary(output_dir, coverage_rows, int(total_samples), metadata)
    return metadata


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
        if all_shards_are_packed(list(args.shard_dirs)):
            metadata = merge_packed_shards_streaming(list(args.shard_dirs), output_dir, list(args.expected_models))
            print(f"wrote merged prediction cache to {output_dir}")
            print(f"sample_count={metadata['sample_count']} record_count={metadata['record_count']}")
            preview_cols = ["sample_key", "model_name", "mae", "mse", "y_true_path", "y_pred_path"]
            preview_df = pd.read_csv(output_dir / "manifest.csv", nrows=int(args.print_rows), usecols=preview_cols)
            print(preview_df.to_string(index=False))
            return

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
