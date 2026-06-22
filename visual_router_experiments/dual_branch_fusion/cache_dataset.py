#!/usr/bin/env python3
"""
文件功能：
    读取并对齐 PatchTST frozen prediction/cache 与 fixed visual embedding cache。

支持格式：
    - 单个 `.npz` 文件；
    - 包含多个 `.npz` shard 的目录，递归读取后按 sample_key 对齐。

关键约束：
    PatchTST cache 必须提供 `sample_key`、`split`、`y_patchtst`、`y_true`，并建议
    提供 `h_ts`；visual cache 必须提供 `sample_key` 和视觉表示字段。读取过程只消费
    已有 cache，不生成视觉输入、不运行 ViT。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch


PATCHTST_PRED_KEYS = ("y_patchtst", "patchtst_pred", "y_pred", "prediction")
PATCHTST_FEATURE_KEYS = ("h_ts", "patchtst_hidden", "ts_embedding", "ts_feature")
VISUAL_FEATURE_KEYS = ("h_vis", "visual_embedding", "mean_patch_embedding", "cls_embedding", "visual_feature")
Y_TRUE_KEYS = ("y_true", "target", "label")


@dataclass(frozen=True)
class AlignedDualBranchCache:
    """类功能：保存按 sample_key 和 split 对齐后的双分支训练/评估数组。"""

    sample_key: np.ndarray
    split: np.ndarray
    h_ts: np.ndarray
    h_vis: np.ndarray
    y_patchtst: np.ndarray
    y_true: np.ndarray


class DualBranchTensorDataset(torch.utils.data.Dataset):
    """类功能：把对齐后的 numpy 数组包装为 PyTorch Dataset。"""

    def __init__(self, cache: AlignedDualBranchCache, indices: Sequence[int]) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.h_ts = torch.from_numpy(cache.h_ts[self.indices].astype(np.float32, copy=False))
        self.h_vis = torch.from_numpy(cache.h_vis[self.indices].astype(np.float32, copy=False))
        self.y_patchtst = torch.from_numpy(cache.y_patchtst[self.indices].astype(np.float32, copy=False))
        self.y_true = torch.from_numpy(cache.y_true[self.indices].astype(np.float32, copy=False))

    def __len__(self) -> int:
        """函数功能：返回样本数。"""
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """函数功能：返回单样本 h_ts、h_vis、PatchTST 预测和真实值。"""
        return self.h_ts[item], self.h_vis[item], self.y_patchtst[item], self.y_true[item]


def _first_existing_key(keys: Iterable[str], available: Iterable[str], *, role: str) -> str:
    """函数功能：在多个候选字段名中选择 cache 实际提供的字段。"""
    available_set = set(available)
    for key in keys:
        if key in available_set:
            return key
    raise ValueError(f"cache 缺少 {role} 字段，候选={list(keys)} available={sorted(available_set)}")


def _list_npz_files(path: Path) -> List[Path]:
    """函数功能：把单文件或目录输入解析为稳定排序的 npz 文件列表。"""
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"只支持 .npz cache 文件：{path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"cache 路径不存在：{path}")
    files = sorted(p for p in path.rglob("*.npz") if p.is_file())
    if not files:
        raise ValueError(f"目录中没有 .npz cache shard：{path}")
    return files


def _load_npz_records(path: Path, *, field_candidates: Mapping[str, Sequence[str]]) -> Dict[str, np.ndarray]:
    """
    函数功能：
        读取单个 npz 文件，并按逻辑字段名重命名为统一 schema。
    """
    # 既有 Round2 feature cache 的 sample_key/layout_name/sample_set 使用 object array
    # 保存；这里仅从本地实验产物读取这些元信息，并立即转为字符串数组。
    with np.load(path, allow_pickle=True) as data:
        available = list(data.files)
        result: Dict[str, np.ndarray] = {}
        for logical_name, candidates in field_candidates.items():
            key = _first_existing_key(candidates, available, role=logical_name)
            result[logical_name] = np.asarray(data[key])
    return result


def _concat_records(records: List[Dict[str, np.ndarray]], *, source_name: str) -> Dict[str, np.ndarray]:
    """函数功能：合并多个 shard，并检查每个字段第一维样本数一致。"""
    if not records:
        raise ValueError(f"{source_name} 没有可合并记录")
    fields = list(records[0].keys())
    merged: Dict[str, np.ndarray] = {}
    for field in fields:
        arrays = [record[field] for record in records]
        first_dims = [int(array.shape[0]) for array in arrays]
        for record, first_dim in zip(records, first_dims):
            for other_field, other_array in record.items():
                if int(other_array.shape[0]) != first_dim:
                    raise ValueError(f"{source_name} shard 字段第一维不一致：field={other_field} shape={other_array.shape}")
        merged[field] = np.concatenate(arrays, axis=0)
    return merged


def load_patchtst_cache(path: Path) -> Dict[str, np.ndarray]:
    """函数功能：读取 PatchTST frozen prediction/cache。"""
    records = [
        _load_npz_records(
            file_path,
            field_candidates={
                "sample_key": ("sample_key", "sample_keys"),
                "split": ("split", "splits", "sample_set"),
                "h_ts": PATCHTST_FEATURE_KEYS,
                "y_patchtst": PATCHTST_PRED_KEYS,
                "y_true": Y_TRUE_KEYS,
            },
        )
        for file_path in _list_npz_files(path)
    ]
    return _concat_records(records, source_name="PatchTST cache")


def load_visual_cache(path: Path) -> Dict[str, np.ndarray]:
    """函数功能：读取 fixed visual embedding cache。"""
    records = [
        _load_npz_records(
            file_path,
            field_candidates={
                "sample_key": ("sample_key", "sample_keys"),
                "h_vis": VISUAL_FEATURE_KEYS,
            },
        )
        for file_path in _list_npz_files(path)
    ]
    return _concat_records(records, source_name="visual cache")


def _string_array(values: np.ndarray) -> np.ndarray:
    """函数功能：把 npz 中的 bytes/str/object-like 字段稳定转为 unicode 字符串数组。"""
    return np.asarray(values).astype(str)


def align_patchtst_and_visual_cache(
    *,
    patchtst_cache_path: Path,
    visual_embedding_cache_path: Path,
    required_splits: Sequence[str],
) -> AlignedDualBranchCache:
    """
    函数功能：
        按 sample_key 对齐 PatchTST cache 与 visual cache，并验证 split 覆盖。

    关键约束：
        对齐后的样本必须来自两类 cache 的交集；如果任一 required split 为空，直接报错，
        避免 baseline 与 dual-branch 在不同 split 或不同子集上评估。
    """
    patch = load_patchtst_cache(patchtst_cache_path)
    visual = load_visual_cache(visual_embedding_cache_path)

    patch_keys = _string_array(patch["sample_key"])
    visual_keys = _string_array(visual["sample_key"])
    if len(set(patch_keys.tolist())) != len(patch_keys):
        raise ValueError("PatchTST cache 中 sample_key 重复")
    if len(set(visual_keys.tolist())) != len(visual_keys):
        raise ValueError("visual cache 中 sample_key 重复")

    visual_index = {key: idx for idx, key in enumerate(visual_keys.tolist())}
    patch_keep: List[int] = []
    visual_take: List[int] = []
    for patch_idx, key in enumerate(patch_keys.tolist()):
        visual_idx = visual_index.get(key)
        if visual_idx is not None:
            patch_keep.append(patch_idx)
            visual_take.append(visual_idx)
    if not patch_keep:
        raise ValueError("PatchTST cache 与 visual cache 没有可对齐的 sample_key")

    split = _string_array(patch["split"])[patch_keep]
    for split_name in required_splits:
        if not np.any(split == str(split_name)):
            raise ValueError(f"对齐 cache 中 split={split_name!r} 为空")

    y_patchtst = np.asarray(patch["y_patchtst"])[patch_keep].astype(np.float32, copy=False)
    y_true = np.asarray(patch["y_true"])[patch_keep].astype(np.float32, copy=False)
    if y_patchtst.shape != y_true.shape:
        raise ValueError(f"PatchTST 预测与 y_true shape 不一致：{y_patchtst.shape} vs {y_true.shape}")

    h_ts = np.asarray(patch["h_ts"])[patch_keep].astype(np.float32, copy=False)
    h_vis = np.asarray(visual["h_vis"])[visual_take].astype(np.float32, copy=False)
    if h_ts.ndim != 2 or h_vis.ndim != 2:
        raise ValueError(f"h_ts/h_vis 必须为二维特征：h_ts={h_ts.shape} h_vis={h_vis.shape}")
    if not np.all(np.isfinite(h_ts)) or not np.all(np.isfinite(h_vis)):
        raise ValueError("h_ts 或 h_vis 包含 NaN/Inf")
    if not np.all(np.isfinite(y_patchtst)) or not np.all(np.isfinite(y_true)):
        raise ValueError("y_patchtst 或 y_true 包含 NaN/Inf")

    return AlignedDualBranchCache(
        sample_key=patch_keys[patch_keep],
        split=split,
        h_ts=h_ts,
        h_vis=h_vis,
        y_patchtst=y_patchtst,
        y_true=y_true,
    )


def split_indices(cache: AlignedDualBranchCache, split_name: str) -> np.ndarray:
    """函数功能：返回指定 split 的样本下标，并在为空时报错。"""
    indices = np.flatnonzero(cache.split == str(split_name))
    if indices.size == 0:
        raise ValueError(f"split={split_name!r} 没有样本")
    return indices.astype(np.int64)
