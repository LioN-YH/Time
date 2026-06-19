#!/usr/bin/env python3
"""
文件功能：
    为 Stage 1 `96_48_S` full-scale TimeFuse-style fusor 提供
    streaming / shard-aware 数据读取层和 smoke test 入口。

设计约束：
    - 输入特征来自已完成的 TimeFuse feature shard，按 batch 流式读取；
    - oracle labels parquet 和 prediction manifest 均先落成 shard-local SQLite
      索引，再按当前 batch 的 sample_key 查询；
    - 不做全量 manifest lookup、不做全量 DataFrame join，也不把 116M 行
      prediction manifest 读入内存；
    - 五专家 prediction array 支持 `packed_npy_v1`，读取时只按 row index 取
      当前 sample 的单行；
    - 该文件只验证 fusor 的输入输出契约，不启动正式训练。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
FULL_SCALE_ROOT = DATA2_RUN_OUTPUT_ROOT / "2026-06-15_stage1_96_48_s_full_scale"
DEFAULT_FEATURE_SHARD_PATH = (
    FULL_SCALE_ROOT
    / "timefuse_feature_cache_full_scale_launcher"
    / "shards"
    / "sample_shard_0000_of_0064"
    / "feature_cache.csv"
)
DEFAULT_ORACLE_LABELS_PATH = (
    FULL_SCALE_ROOT
    / "prediction_cache_full_scale_launcher"
    / "oracle_labels_full_scale_2026-06-16"
    / "window_oracle_labels.parquet"
)
DEFAULT_PREDICTION_SHARD_ROOT = FULL_SCALE_ROOT / "prediction_cache_full_scale_launcher" / "shards"
DEFAULT_OUTPUT_ROOT = DATA2_RUN_OUTPUT_ROOT

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from visual_router_experiments.common.prediction_array_io import (  # noqa: E402
    PACKED_NPY_STORAGE,
    PER_SAMPLE_NPY_STORAGE,
    load_prediction_array,
    resolve_cache_array_path,
)
from visual_router_experiments.stage1_vali_test_router.build_timefuse_feature_cache_from_manifest import (  # noqa: E402
    FEATURE_CACHE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_TYPE,
    FEATURE_VERSION,
)
from visual_router_experiments.stage1_vali_test_router.fusion_utils import MODEL_COLUMNS  # noqa: E402


ORACLE_COLUMNS = [
    "sample_key",
    "config_name",
    "split",
    "dataset_name",
    "item_id",
    "channel_id",
    "window_index",
    "metric",
    "oracle_model",
    "oracle_value",
    *MODEL_COLUMNS,
]

PREDICTION_MANIFEST_COLUMNS = [
    "sample_key",
    "model_name",
    "y_true_path",
    "y_pred_path",
    "mae",
    "mse",
    "array_storage",
    "y_true_row_index",
    "y_pred_row_index",
]


def display_time() -> str:
    """函数功能：生成写入中文日志、metadata 和 status 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def now_token() -> str:
    """函数功能：生成默认输出目录使用的时间戳。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def append_log(output_dir: Path, message: str) -> None:
    """函数功能：追加写 smoke 主日志，便于长命令中途接手。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def write_status(output_dir: Path, payload: Mapping[str, object]) -> None:
    """函数功能：写出 status.json，记录当前 smoke 或索引构建阶段。"""
    status = dict(payload)
    status["updated_at"] = display_time()
    status["output_dir"] = str(output_dir)
    status["pid"] = int(os.getpid())
    (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_feature_columns(feature_cache_path: Path) -> List[str]:
    """
    函数功能：
        从 feature shard 表头推导数值特征列。

    说明：
        正式 TimeFuse feature cache 固定为 17 维 `FEATURE_COLUMNS`。这里仍从表头
        校验一次，避免误把 metadata 或未来新增非数值列当成 fusor 输入。
    """
    header = pd.read_csv(feature_cache_path, nrows=0)
    missing = sorted(set(FEATURE_CACHE_COLUMNS).difference(header.columns))
    if missing:
        raise ValueError(f"feature shard 缺少字段：{missing}")
    feature_cols = [col for col in FEATURE_COLUMNS if col in header.columns]
    if feature_cols != FEATURE_COLUMNS:
        raise ValueError(f"feature shard 特征列不完整：actual={feature_cols}")
    return list(feature_cols)


def collect_feature_sample_keys(feature_shard_path: Path, *, max_rows: Optional[int], chunk_rows: int) -> List[str]:
    """
    函数功能：
        只读取 feature shard 的 sample_key 列，收集当前 shard 或 smoke 子集所需 key。

    关键约束：
        这里的 key 集合规模至多为单个 feature shard；full-scale 正式 64 shard
        逐 shard 运行时，不会产生 23M 全量 key 常驻内存。
    """
    keys: List[str] = []
    rows_seen = 0
    for chunk_df in pd.read_csv(feature_shard_path, usecols=["sample_key"], chunksize=int(chunk_rows)):
        values = chunk_df["sample_key"].astype(str).tolist()
        if max_rows is not None:
            remaining = int(max_rows) - rows_seen
            if remaining <= 0:
                break
            values = values[:remaining]
        keys.extend(values)
        rows_seen += len(values)
        if max_rows is not None and rows_seen >= int(max_rows):
            break
    if not keys:
        raise ValueError("feature shard 中没有可读取的 sample_key")
    if len(keys) != len(set(keys)):
        raise ValueError("feature shard 子集中存在重复 sample_key")
    return keys


class OracleSQLiteIndex:
    """
    类功能：
        封装 shard-local oracle labels SQLite 查询。

    说明：
        oracle parquet 全量有 46M 行（MAE/MSE 两套标签）。reader 只为当前
        feature shard 的 sample_key 和指定 metric 建本地索引，batch 训练时按
        sample_key 查询，不做全量 labels DataFrame join。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # 预取线程会发起 batch 查询；SQLite 连接允许跨线程使用，但每次查询用锁
        # 串行化，避免同一连接被两个预取任务同时访问。
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def fetch_rows(self, sample_keys: Sequence[str]) -> Dict[str, Dict[str, object]]:
        """函数功能：批量查询当前 batch 的 oracle label 和五专家窗口误差。"""
        keys = [str(key) for key in sample_keys]
        if not keys:
            return {}
        placeholders = ",".join(["?"] * len(keys))
        with self._lock:
            rows = self.connection.execute(
                f"""
                SELECT sample_key, config_name, split, dataset_name, item_id, channel_id,
                       window_index, metric, oracle_model, oracle_value,
                       DLinear, PatchTST, CrossFormer, ES, NaiveForecaster
                FROM oracle_index
                WHERE sample_key IN ({placeholders})
                """,
                keys,
            ).fetchall()
        return {str(row["sample_key"]): dict(row) for row in rows}

    def close(self) -> None:
        """函数功能：关闭 SQLite 连接，释放文件句柄。"""
        self.connection.close()


class PredictionSQLiteIndex:
    """
    类功能：
        封装 shard-local prediction manifest SQLite 查询。

    说明：
        可从五个专家 shard manifest 或单个 merged manifest 构建。若输入为 merged
        116M manifest，也只把当前 shard 的 `(sample_key, model_name)` 写入 SQLite，
        不保留全量 Python lookup。
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # 同 oracle index；允许预取线程查询，并用锁保证连接级访问顺序清晰。
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def fetch_records(self, sample_keys: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, object]]:
        """函数功能：批量查询当前 batch 五专家 prediction record。"""
        keys = [str(key) for key in sample_keys]
        if not keys:
            return {}
        placeholders = ",".join(["?"] * len(keys))
        with self._lock:
            rows = self.connection.execute(
                f"""
                SELECT sample_key, model_name, manifest_dir, y_true_path, y_pred_path, mae, mse,
                       array_storage, y_true_row_index, y_pred_row_index
                FROM prediction_index
                WHERE sample_key IN ({placeholders})
                """,
                keys,
            ).fetchall()
        records: Dict[Tuple[str, str], Dict[str, object]] = {}
        for row in rows:
            record = dict(row)
            manifest_dir = Path(str(record.pop("manifest_dir")))
            sample_key = str(record["sample_key"])
            model_name = str(record["model_name"])
            record["y_true_path"] = resolve_cache_array_path(str(record["y_true_path"]), manifest_dir)
            record["y_pred_path"] = resolve_cache_array_path(str(record["y_pred_path"]), manifest_dir)
            records[(sample_key, model_name)] = record
        return records

    def close(self) -> None:
        """函数功能：关闭 SQLite 连接，释放文件句柄。"""
        self.connection.close()


def _prepare_sqlite_path(index_db_path: Path) -> Tuple[sqlite3.Connection, Path]:
    """函数功能：创建临时 SQLite 文件，并配置适合一次性批量写入的 pragma。"""
    index_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path = index_db_path.with_suffix(index_db_path.suffix + ".tmp")
    if tmp_db_path.exists():
        tmp_db_path.unlink()
    if index_db_path.exists():
        index_db_path.unlink()
    connection = sqlite3.connect(str(tmp_db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection, tmp_db_path


def build_oracle_sqlite_index(
    labels_path: Path,
    *,
    sample_keys: Sequence[str],
    metric: str,
    index_db_path: Path,
    parquet_batch_rows: int,
) -> OracleSQLiteIndex:
    """
    函数功能：
        从 full-scale oracle parquet 构建当前 feature shard 的 SQLite 索引。

    实现说明：
        使用 PyArrow `iter_batches` 分批扫描，只将 `metric` 和当前 shard
        `sample_key` 命中的行写入 SQLite。这样内存峰值由 parquet batch、
        shard key set 和 SQLite 写入缓冲决定，不随全量 parquet 行数线性增长。
    """
    if metric not in {"mae", "mse"}:
        raise ValueError(f"metric 必须为 mae/mse，实际为 {metric}")
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 oracle labels parquet：{labels_path}")
    key_set = {str(key) for key in sample_keys}
    if not key_set:
        raise ValueError("oracle index 至少需要一个 sample_key")

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    connection, tmp_db_path = _prepare_sqlite_path(index_db_path)
    connection.execute(
        """
        CREATE TABLE oracle_index (
            sample_key TEXT PRIMARY KEY,
            config_name TEXT NOT NULL,
            split TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            item_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            window_index INTEGER NOT NULL,
            metric TEXT NOT NULL,
            oracle_model TEXT NOT NULL,
            oracle_value REAL NOT NULL,
            DLinear REAL NOT NULL,
            PatchTST REAL NOT NULL,
            CrossFormer REAL NOT NULL,
            ES REAL NOT NULL,
            NaiveForecaster REAL NOT NULL
        )
        """
    )

    value_set = pa.array(list(key_set), type=pa.string())
    rows_seen = 0
    rows_matched = 0
    parquet_file = pq.ParquetFile(labels_path)
    for batch_idx, record_batch in enumerate(
        parquet_file.iter_batches(batch_size=int(parquet_batch_rows), columns=ORACLE_COLUMNS),
        start=1,
    ):
        rows_seen += int(record_batch.num_rows)
        table = pa.Table.from_batches([record_batch])
        mask = pc.and_(pc.equal(table["metric"], str(metric)), pc.is_in(table["sample_key"], value_set=value_set))
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            continue
        matched_df = filtered.to_pandas()
        rows_matched += int(len(matched_df))
        insert_rows = [
            (
                str(row.sample_key),
                str(row.config_name),
                str(row.split),
                str(row.dataset_name),
                int(row.item_id),
                int(row.channel_id),
                int(row.window_index),
                str(row.metric),
                str(row.oracle_model),
                float(row.oracle_value),
                float(row.DLinear),
                float(row.PatchTST),
                float(row.CrossFormer),
                float(row.ES),
                float(row.NaiveForecaster),
            )
            for row in matched_df.itertuples(index=False)
        ]
        try:
            connection.executemany(
                """
                INSERT INTO oracle_index (
                    sample_key, config_name, split, dataset_name, item_id, channel_id,
                    window_index, metric, oracle_model, oracle_value,
                    DLinear, PatchTST, CrossFormer, ES, NaiveForecaster
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("oracle labels 中当前 metric 的 sample_key 存在重复") from exc
        connection.commit()
        if batch_idx == 1 or batch_idx % 25 == 0:
            print(
                f"[oracle_index] batches={batch_idx} rows_seen={rows_seen} "
                f"matched_rows={rows_matched} target_sample_keys={len(key_set)}",
                flush=True,
            )
        if rows_matched == len(key_set):
            break

    actual_records = int(connection.execute("SELECT COUNT(*) FROM oracle_index").fetchone()[0])
    if actual_records != len(key_set):
        connection.close()
        missing_count = len(key_set) - actual_records
        raise ValueError(f"oracle labels 子集不完整：missing_count={missing_count} actual={actual_records}")
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    metadata = {
        "created_at": display_time(),
        "labels_path": str(labels_path),
        "metric": str(metric),
        "target_sample_keys": int(len(key_set)),
        "actual_records": int(actual_records),
        "parquet_batch_rows": int(parquet_batch_rows),
    }
    connection.executemany(
        "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
        [(str(key), json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
    )
    connection.commit()
    connection.close()
    tmp_db_path.replace(index_db_path)
    print(f"[oracle_index] sqlite_index_ready path={index_db_path} records={actual_records}", flush=True)
    return OracleSQLiteIndex(index_db_path)


def build_prediction_sqlite_index(
    prediction_manifest_paths: Sequence[Path],
    *,
    sample_keys: Sequence[str],
    index_db_path: Path,
    chunk_read_rows: int,
) -> PredictionSQLiteIndex:
    """
    函数功能：
        从 prediction manifest 构建当前 feature shard 的 SQLite 索引。

    输入可以是：
        - 五个专家的 `sample_shard_xxxx/manifest.csv`，推荐用于 shard-aware smoke
          和正式 shard 训练；
        - 一个 full merged `manifest.csv`，用于兼容已有 downstream 入口。
    """
    manifest_paths = [Path(path) for path in prediction_manifest_paths]
    if not manifest_paths:
        raise ValueError("至少需要一个 prediction manifest path")
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 prediction manifest：{manifest_path}")

    key_set = {str(key) for key in sample_keys}
    if not key_set:
        raise ValueError("prediction index 至少需要一个 sample_key")

    connection, tmp_db_path = _prepare_sqlite_path(index_db_path)
    connection.execute(
        """
        CREATE TABLE prediction_index (
            sample_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            manifest_dir TEXT NOT NULL,
            y_true_path TEXT NOT NULL,
            y_pred_path TEXT NOT NULL,
            mae REAL NOT NULL,
            mse REAL NOT NULL,
            array_storage TEXT,
            y_true_row_index INTEGER,
            y_pred_row_index INTEGER,
            PRIMARY KEY (sample_key, model_name)
        )
        """
    )

    rows_seen = 0
    matched_rows = 0
    for manifest_path in manifest_paths:
        manifest_dir = manifest_path.parent
        for chunk_idx, chunk_df in enumerate(
            pd.read_csv(manifest_path, usecols=PREDICTION_MANIFEST_COLUMNS, chunksize=int(chunk_read_rows)),
            start=1,
        ):
            rows_seen += int(len(chunk_df))
            matched_df = chunk_df[chunk_df["sample_key"].astype(str).isin(key_set)]
            if matched_df.empty:
                continue
            matched_rows += int(len(matched_df))
            insert_rows = [
                (
                    str(row.sample_key),
                    str(row.model_name),
                    str(manifest_dir),
                    str(row.y_true_path),
                    str(row.y_pred_path),
                    float(row.mae),
                    float(row.mse),
                    str(row.array_storage) if pd.notna(row.array_storage) else None,
                    None if pd.isna(row.y_true_row_index) else int(row.y_true_row_index),
                    None if pd.isna(row.y_pred_row_index) else int(row.y_pred_row_index),
                )
                for row in matched_df.itertuples(index=False)
            ]
            try:
                connection.executemany(
                    """
                    INSERT INTO prediction_index (
                        sample_key, model_name, manifest_dir, y_true_path, y_pred_path, mae, mse,
                        array_storage, y_true_row_index, y_pred_row_index
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("prediction manifest 中 sample_key + model_name 存在重复") from exc
            connection.commit()
            if chunk_idx == 1 or chunk_idx % 25 == 0:
                print(
                    f"[prediction_index] manifest={manifest_path.name} chunks={chunk_idx} rows_seen={rows_seen} "
                    f"matched_rows={matched_rows} target_sample_keys={len(key_set)}",
                    flush=True,
                )
        if matched_rows == len(key_set) * len(MODEL_COLUMNS):
            break

    expected_records = len(key_set) * len(MODEL_COLUMNS)
    actual_records = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
    if actual_records != expected_records:
        connection.close()
        raise ValueError(
            f"prediction manifest 子集不完整：expected_records={expected_records} actual={actual_records} "
            f"sample_keys={len(key_set)} manifest_count={len(manifest_paths)}"
        )
    connection.execute("CREATE INDEX idx_prediction_index_sample_key ON prediction_index(sample_key)")
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    metadata = {
        "created_at": display_time(),
        "prediction_manifest_paths": [str(path) for path in manifest_paths],
        "target_sample_keys": int(len(key_set)),
        "expected_records": int(expected_records),
        "actual_records": int(actual_records),
        "chunk_read_rows": int(chunk_read_rows),
    }
    connection.executemany(
        "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
        [(str(key), json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
    )
    connection.commit()
    connection.close()
    tmp_db_path.replace(index_db_path)
    print(f"[prediction_index] sqlite_index_ready path={index_db_path} records={actual_records}", flush=True)
    return PredictionSQLiteIndex(index_db_path)


def _optional_int(record: Mapping[str, object], key: str) -> Optional[int]:
    """函数功能：读取 SQLite record 中可能为空的 packed row index。"""
    value = record.get(key)
    if value is None:
        return None
    return int(value)


def _record_storage(record: Mapping[str, object]) -> str:
    """函数功能：读取 prediction record 的数组存储格式，空值按 legacy 小文件处理。"""
    return str(record.get("array_storage", PER_SAMPLE_NPY_STORAGE) or PER_SAMPLE_NPY_STORAGE)


def _load_array_grouped(records: Sequence[Mapping[str, object]], array_kind: str) -> np.ndarray:
    """
    函数功能：
        对同一个 batch 的 packed npy 按路径分组读取。

    优化原因：
        full-scale prediction cache 的 packed npy 一个文件包含一个 sample shard
        的多行窗口数组。旧实现对每个 sample 调一次 `np.load(mmap_mode="r")`，
        同一 batch 会重复打开同一个大文件数百次，训练首批次就可能耗时很久。
        这里按 `(array_path, row_index)` 批量切片，每个路径在当前 batch 只打开
        一次；legacy per-sample 小文件仍回退到统一读取接口。
    """
    if array_kind not in {"y_true", "y_pred"}:
        raise ValueError(f"array_kind 只能是 y_true/y_pred，实际为 {array_kind}")
    if not records:
        raise ValueError("records 不能为空")

    path_key = f"{array_kind}_path"
    row_key = f"{array_kind}_row_index"
    output: List[Optional[np.ndarray]] = [None] * len(records)
    grouped: Dict[Path, List[Tuple[int, int]]] = {}
    for position, record in enumerate(records):
        storage = _record_storage(record)
        row_index = _optional_int(record, row_key)
        if storage == PACKED_NPY_STORAGE or row_index is not None:
            if row_index is None:
                raise ValueError(f"packed 数组缺少 {row_key}：{record.get(path_key)}")
            grouped.setdefault(Path(str(record[path_key])), []).append((position, int(row_index)))
            continue
        output[position] = load_prediction_array(record, array_kind)

    for path, positions in grouped.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到 prediction cache 数组：{path}")
        array = np.load(path, mmap_mode="r")
        row_indices = np.asarray([row_index for _, row_index in positions], dtype=np.int64)
        if int(row_indices.min()) < 0 or int(row_indices.max()) >= int(array.shape[0]):
            raise IndexError(f"{array_kind}_row_index 越界：shape={array.shape} path={path}")
        values = np.asarray(array[row_indices], dtype=np.float32)
        for local_idx, (position, _) in enumerate(positions):
            output[position] = values[local_idx]

    if any(value is None for value in output):
        raise RuntimeError("batch 数组读取后仍存在空位")
    return np.stack([np.asarray(value, dtype=np.float32) for value in output], axis=0)


def load_prediction_tensors_from_index(
    sample_keys: Sequence[str],
    prediction_index: PredictionSQLiteIndex,
    *,
    error_metric: str,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    函数功能：
        按 batch 顺序读取五专家 y_pred、共享 y_true 和专家误差。

    并行策略：
        `num_workers > 1` 时用线程并行读取不同 sample 的 memmap row。线程数只
        作用于当前 batch，且 reader 最多预取一个 batch，因此不会增加超过
        `batch_size * prefetch_batches` 级别的数组内存峰值。
    """
    keys = [str(key) for key in sample_keys]
    batch_lookup = prediction_index.fetch_records(keys)
    missing = [
        (sample_key, model_name)
        for sample_key in keys
        for model_name in MODEL_COLUMNS
        if (sample_key, model_name) not in batch_lookup
    ]
    if missing:
        raise ValueError(f"prediction index 缺少当前 batch 专家记录，示例：{missing[:5]}")

    first_model_records = [batch_lookup[(sample_key, MODEL_COLUMNS[0])] for sample_key in keys]
    y_true = _load_array_grouped(first_model_records, "y_true").astype(np.float32)
    model_pred_arrays: List[np.ndarray] = []
    for model_name in MODEL_COLUMNS:
        model_records = [batch_lookup[(sample_key, model_name)] for sample_key in keys]
        model_pred_arrays.append(_load_array_grouped(model_records, "y_pred").astype(np.float32))
        for sample_idx, (sample_key, first_record, current_record) in enumerate(zip(keys, first_model_records, model_records)):
            first_identity = (str(first_record["y_true_path"]), _optional_int(first_record, "y_true_row_index"))
            current_identity = (str(current_record["y_true_path"]), _optional_int(current_record, "y_true_row_index"))
            if current_identity != first_identity:
                current_true = _load_array_grouped([current_record], "y_true")[0]
                expected_true = y_true[sample_idx]
                if not np.array_equal(expected_true, current_true):
                    raise ValueError(f"同一 sample_key 的 y_true 内容不一致：{sample_key}")
    y_preds = np.stack(model_pred_arrays, axis=1).astype(np.float32)
    if y_preds.shape[0] != y_true.shape[0] or y_preds.shape[2:] != y_true.shape[1:]:
        raise ValueError(f"y_pred/y_true shape 不一致：y_pred={y_preds.shape} y_true={y_true.shape}")
    diff = y_preds - y_true[:, None, ...]
    reduce_axes = tuple(range(2, diff.ndim))
    if error_metric == "mae":
        expert_errors = np.mean(np.abs(diff), axis=reduce_axes).astype(np.float32)
    elif error_metric == "mse":
        expert_errors = np.mean(diff ** 2, axis=reduce_axes).astype(np.float32)
    else:
        raise ValueError(f"未知 error_metric={error_metric}")
    if y_preds.ndim < 3:
        raise ValueError(f"专家预测张量维度异常：{y_preds.shape}")
    if not (np.isfinite(y_preds).all() and np.isfinite(y_true).all() and np.isfinite(expert_errors).all()):
        raise ValueError("prediction tensor 中存在非有限值")
    return y_preds, y_true, expert_errors


@dataclass(frozen=True)
class TimeFuseFusorBatch:
    """
    类功能：
        固定 full-scale TimeFuse-style fusor 的 batch 输入输出契约。

    字段说明：
        - `features`: `[B, F]` float32，F 当前固定为 17；
        - `labels`: 与 batch sample_key 对齐的 oracle rows；
        - `y_pred`: `[B, 5, pred_len, channels]` 或等价专家预测张量；
        - `y_true`: `[B, pred_len, channels]` 或等价真实未来张量；
        - `expert_errors`: `[B, 5]`，按指定 metric 从数组复算。
    """

    sample_keys: List[str]
    metadata_df: pd.DataFrame
    features: np.ndarray
    labels: List[Dict[str, object]]
    y_pred: np.ndarray
    y_true: np.ndarray
    expert_errors: np.ndarray


class Stage1TimeFuseFusorStreamingReader:
    """
    类功能：
        按 feature shard batch 流式产出 fusor 训练/验证所需张量。

    关键约束：
        该 reader 只持有当前 batch 和可选的一个预取 batch；oracle/prediction
        全量输入均通过 SQLite batch query 访问。
    """

    def __init__(
        self,
        *,
        feature_shard_path: Path,
        oracle_index: OracleSQLiteIndex,
        prediction_index: PredictionSQLiteIndex,
        feature_columns: Sequence[str],
        batch_size: int,
        metric: str,
        max_rows: Optional[int] = None,
        prediction_num_workers: int = 1,
        prefetch_batches: int = 1,
        split_filter: Optional[str] = None,
    ) -> None:
        self.feature_shard_path = Path(feature_shard_path)
        self.oracle_index = oracle_index
        self.prediction_index = prediction_index
        self.feature_columns = list(feature_columns)
        self.batch_size = int(batch_size)
        self.metric = str(metric)
        self.max_rows = None if max_rows is None else int(max_rows)
        self.prediction_num_workers = int(prediction_num_workers)
        self.prefetch_batches = int(prefetch_batches)
        self.split_filter = None if split_filter is None else str(split_filter)

    def _iter_feature_frames(self) -> Iterator[pd.DataFrame]:
        """函数功能：从 feature shard CSV 按 batch 流式读取 metadata 和 17 维特征。"""
        usecols = list(FEATURE_CACHE_COLUMNS)
        rows_seen = 0
        raw_chunk_rows = max(self.batch_size, min(200_000, self.batch_size * 1024))
        for chunk_df in pd.read_csv(self.feature_shard_path, usecols=usecols, chunksize=raw_chunk_rows):
            if self.split_filter is not None:
                # split 过滤必须在读取 oracle/prediction arrays 之前完成；否则训练
                # vali 时会先读入后续会被丢弃的 test prediction arrays。
                chunk_df = chunk_df[chunk_df["split"].astype(str) == self.split_filter].copy()
                if chunk_df.empty:
                    continue
            if self.max_rows is not None:
                remaining = self.max_rows - rows_seen
                if remaining <= 0:
                    break
                chunk_df = chunk_df.head(remaining).copy()
            rows_seen += int(len(chunk_df))
            if chunk_df.empty:
                continue
            # 原始 CSV 读取块可以较大；过滤 split 后再切成稳定 batch，避免
            # full-scale 训练出现几十个样本的小 batch，放大 SQLite/np.memmap 开销。
            for start in range(0, len(chunk_df), self.batch_size):
                yield chunk_df.iloc[start : start + self.batch_size].reset_index(drop=True)
            if self.max_rows is not None and rows_seen >= self.max_rows:
                break

    def _make_batch(self, feature_df: pd.DataFrame) -> TimeFuseFusorBatch:
        """函数功能：将一个 feature DataFrame batch 补齐 oracle 和 prediction 张量。"""
        sample_keys = feature_df["sample_key"].astype(str).tolist()
        oracle_rows = self.oracle_index.fetch_rows(sample_keys)
        missing_labels = [key for key in sample_keys if key not in oracle_rows]
        if missing_labels:
            raise ValueError(f"oracle index 缺少当前 batch labels，示例：{missing_labels[:5]}")
        labels = [oracle_rows[key] for key in sample_keys]

        feature_values = feature_df[self.feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(feature_values).all():
            raise ValueError("feature batch 中存在 NaN/Inf")

        y_pred, y_true, expert_errors = load_prediction_tensors_from_index(
            sample_keys,
            self.prediction_index,
            error_metric=self.metric,
            num_workers=self.prediction_num_workers,
        )
        metadata_df = feature_df[
            [
                "sample_key",
                "config_name",
                "split",
                "dataset_name",
                "item_id",
                "channel_id",
                "window_index",
                "history_length",
                "pred_length",
            ]
        ].copy()
        return TimeFuseFusorBatch(
            sample_keys=sample_keys,
            metadata_df=metadata_df,
            features=feature_values,
            labels=labels,
            y_pred=y_pred,
            y_true=y_true,
            expert_errors=expert_errors,
        )

    def __iter__(self) -> Iterator[TimeFuseFusorBatch]:
        """函数功能：按顺序产出 batch，并可用单线程预取下一批 I/O。"""
        if self.prefetch_batches <= 0:
            for feature_df in self._iter_feature_frames():
                yield self._make_batch(feature_df)
            return

        # 只允许一个后台任务预取下一批，避免 I/O 并行把数组内存峰值放大。
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = None
            for feature_df in self._iter_feature_frames():
                next_future = executor.submit(self._make_batch, feature_df)
                if pending is not None:
                    yield pending.result()
                pending = next_future
            if pending is not None:
                yield pending.result()


def discover_prediction_shard_manifests(prediction_shard_root: Path, feature_shard_path: Path) -> List[Path]:
    """
    函数功能：
        根据 feature shard 目录名自动发现五专家同编号 prediction shard manifest。
    """
    shard_name = feature_shard_path.parent.name
    paths = [prediction_shard_root / model_name / shard_name / "manifest.csv" for model_name in MODEL_COLUMNS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"无法自动发现五专家 prediction shard manifest：{missing}")
    return paths


def parse_args() -> argparse.Namespace:
    """函数功能：解析 streaming reader smoke test 参数。"""
    parser = argparse.ArgumentParser(description="Smoke test Stage 1 full-scale TimeFuse fusor streaming reader.")
    parser.add_argument("--feature-shard-path", type=Path, default=DEFAULT_FEATURE_SHARD_PATH, help="单个 feature_cache.csv shard。")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_ORACLE_LABELS_PATH, help="full-scale oracle labels parquet。")
    parser.add_argument("--prediction-manifest-path", type=Path, action="append", default=None, help="prediction manifest；可重复传入五专家 shard manifest，或传入一个 merged manifest。")
    parser.add_argument("--prediction-shard-root", type=Path, default=DEFAULT_PREDICTION_SHARD_ROOT, help="未显式传 manifest 时，用该根目录自动发现五专家同编号 shard manifest。")
    parser.add_argument("--output-dir", type=Path, default=None, help="smoke 输出目录；默认写 /data2 run_outputs。")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="oracle label 和专家误差口径。")
    parser.add_argument("--batch-size", type=int, default=8, help="reader batch size。")
    parser.add_argument("--max-rows", type=int, default=16, help="smoke 从 feature shard 前 N 行读取；只验证少量 batch。")
    parser.add_argument("--feature-key-chunk-rows", type=int, default=200000, help="收集 shard sample_key 的 CSV chunk 大小。")
    parser.add_argument("--prediction-chunk-rows", type=int, default=200000, help="prediction manifest CSV chunk 大小。")
    parser.add_argument("--oracle-parquet-batch-rows", type=int, default=200000, help="oracle parquet batch scan 行数。")
    parser.add_argument("--prediction-num-workers", type=int, default=2, help="当前 batch 内 prediction row 读取线程数。")
    parser.add_argument("--prefetch-batches", type=int, default=1, help="最多预取 batch 数；当前实现限制为 0 或 1。")
    parser.add_argument("--smoke-batches", type=int, default=2, help="实际消费多少个 batch 后停止。")
    return parser.parse_args()


def validate_batch_contract(batch: TimeFuseFusorBatch, *, feature_dim: int, metric: str) -> Dict[str, object]:
    """函数功能：校验并汇总单个 batch 的 fusor 输入输出契约。"""
    batch_size = len(batch.sample_keys)
    if batch.features.shape != (batch_size, int(feature_dim)):
        raise ValueError(f"feature shape 异常：{batch.features.shape}")
    if batch.y_pred.shape[0] != batch_size or batch.y_pred.shape[1] != len(MODEL_COLUMNS):
        raise ValueError(f"y_pred shape 异常：{batch.y_pred.shape}")
    if batch.y_true.shape[0] != batch_size:
        raise ValueError(f"y_true shape 异常：{batch.y_true.shape}")
    if batch.expert_errors.shape != (batch_size, len(MODEL_COLUMNS)):
        raise ValueError(f"expert_errors shape 异常：{batch.expert_errors.shape}")
    if [row["sample_key"] for row in batch.labels] != batch.sample_keys:
        raise ValueError("labels 顺序与 feature batch sample_key 不一致")

    diff = batch.y_pred[:, 0] - batch.y_true
    if metric == "mae":
        recomputed = np.mean(np.abs(diff), axis=tuple(range(1, batch.y_true.ndim)))
    elif metric == "mse":
        recomputed = np.mean(diff ** 2, axis=tuple(range(1, batch.y_true.ndim)))
    else:
        raise ValueError(f"未知 metric={metric}")
    max_abs_error_delta = float(np.max(np.abs(recomputed - batch.expert_errors[:, 0])))
    if max_abs_error_delta > 1e-5:
        raise ValueError(f"expert_errors 与数组复算不一致：max_delta={max_abs_error_delta}")
    return {
        "batch_size": int(batch_size),
        "feature_shape": list(batch.features.shape),
        "y_pred_shape": list(batch.y_pred.shape),
        "y_true_shape": list(batch.y_true.shape),
        "expert_errors_shape": list(batch.expert_errors.shape),
        "first_sample_key": batch.sample_keys[0],
        "first_oracle_model": str(batch.labels[0]["oracle_model"]),
        "first_oracle_value": float(batch.labels[0]["oracle_value"]),
        "max_abs_error_delta_dlinear": max_abs_error_delta,
    }


def main() -> None:
    """函数功能：执行 1-shard / 少量 batch reader smoke test。"""
    args = parse_args()
    if args.prefetch_batches not in {0, 1}:
        raise ValueError("--prefetch-batches 当前只支持 0 或 1，避免无意放大内存峰值")
    if int(args.max_rows) <= 0 or int(args.batch_size) <= 0 or int(args.smoke_batches) <= 0:
        raise ValueError("--max-rows、--batch-size、--smoke-batches 必须为正整数")

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"{now_token()}_stage1_timefuse_fusor_streaming_reader_smoke"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    append_log(output_dir, "启动 Stage 1 TimeFuse fusor streaming reader smoke test")
    write_status(output_dir, {"status": "running", "phase": "collect_feature_keys"})

    feature_cols = infer_feature_columns(args.feature_shard_path)
    sample_keys = collect_feature_sample_keys(
        args.feature_shard_path,
        max_rows=int(args.max_rows),
        chunk_rows=int(args.feature_key_chunk_rows),
    )
    append_log(output_dir, f"已收集 feature shard smoke sample_key={len(sample_keys)}")

    prediction_manifest_paths = args.prediction_manifest_path
    if not prediction_manifest_paths:
        prediction_manifest_paths = discover_prediction_shard_manifests(args.prediction_shard_root, args.feature_shard_path)
    append_log(output_dir, f"prediction manifest 数量={len(prediction_manifest_paths)}")

    write_status(output_dir, {"status": "running", "phase": "build_oracle_index", "sample_key_count": len(sample_keys)})
    oracle_index = build_oracle_sqlite_index(
        args.labels_path,
        sample_keys=sample_keys,
        metric=str(args.metric),
        index_db_path=output_dir / "oracle_labels_index.sqlite",
        parquet_batch_rows=int(args.oracle_parquet_batch_rows),
    )
    write_status(output_dir, {"status": "running", "phase": "build_prediction_index", "sample_key_count": len(sample_keys)})
    prediction_index = build_prediction_sqlite_index(
        prediction_manifest_paths,
        sample_keys=sample_keys,
        index_db_path=output_dir / "prediction_manifest_index.sqlite",
        chunk_read_rows=int(args.prediction_chunk_rows),
    )

    reader = Stage1TimeFuseFusorStreamingReader(
        feature_shard_path=args.feature_shard_path,
        oracle_index=oracle_index,
        prediction_index=prediction_index,
        feature_columns=feature_cols,
        batch_size=int(args.batch_size),
        metric=str(args.metric),
        max_rows=int(args.max_rows),
        prediction_num_workers=int(args.prediction_num_workers),
        prefetch_batches=int(args.prefetch_batches),
    )
    write_status(output_dir, {"status": "running", "phase": "consume_reader_batches", "sample_key_count": len(sample_keys)})

    batch_summaries: List[Dict[str, object]] = []
    start = time.perf_counter()
    try:
        for batch_idx, batch in enumerate(reader, start=1):
            summary = validate_batch_contract(batch, feature_dim=len(feature_cols), metric=str(args.metric))
            summary["batch_index"] = int(batch_idx)
            batch_summaries.append(summary)
            append_log(
                output_dir,
                f"batch={batch_idx} feature_shape={summary['feature_shape']} "
                f"y_pred_shape={summary['y_pred_shape']} first_key={summary['first_sample_key']}",
            )
            if batch_idx >= int(args.smoke_batches):
                break
    finally:
        oracle_index.close()
        prediction_index.close()

    elapsed = time.perf_counter() - start
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "feature_shard_path": str(args.feature_shard_path),
        "labels_path": str(args.labels_path),
        "prediction_manifest_paths": [str(path) for path in prediction_manifest_paths],
        "metric": str(args.metric),
        "max_rows": int(args.max_rows),
        "batch_size": int(args.batch_size),
        "smoke_batches": int(args.smoke_batches),
        "feature_version": FEATURE_VERSION,
        "feature_type": FEATURE_TYPE,
        "feature_columns": feature_cols,
        "feature_dim": len(feature_cols),
        "model_columns": MODEL_COLUMNS,
        "reader_contract": {
            "features": "[B, 17] float32",
            "labels": "batch-ordered oracle label rows from shard-local SQLite",
            "y_pred": "[B, 5, pred_len, channels] float32 from packed/per-sample arrays",
            "y_true": "[B, pred_len, channels] float32 shared per sample",
            "expert_errors": "[B, 5] float32 recomputed from arrays",
        },
        "memory_policy": [
            "feature CSV is streamed by batch",
            "oracle parquet is materialized only into shard-local SQLite for selected sample_key",
            "prediction manifest is materialized only into shard-local SQLite for selected sample_key",
            "reader holds current batch and at most one prefetched batch",
        ],
        "parallel_policy": {
            "prediction_num_workers": int(args.prediction_num_workers),
            "prefetch_batches": int(args.prefetch_batches),
            "reason": "只并行当前 batch 的数组 row 读取和至多一个 batch 预取，提升 I/O 重叠但不扩大到全量数据常驻内存。",
        },
        "elapsed_seconds_reader_only": float(elapsed),
        "batch_summaries": batch_summaries,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "batch_summaries.json").write_text(
        json.dumps(batch_summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_status(
        output_dir,
        {
            "status": "completed",
            "phase": "done",
            "batch_count": len(batch_summaries),
            "sample_key_count": len(sample_keys),
            "metadata_path": str(output_dir / "metadata.json"),
        },
    )
    append_log(output_dir, f"smoke test 完成，batch_count={len(batch_summaries)}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
