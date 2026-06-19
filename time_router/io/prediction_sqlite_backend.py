#!/usr/bin/env python3
"""
文件功能：
    提供 Stage 1 P10b 最小 shared prediction SQLite backend helper。

设计边界：
    该 helper 只负责 prediction manifest 的目标 sample_key 子集索引、
    batch record 查询和轻量 metadata。它不创建 run_dir，不读取 feature/oracle/loss，
    不写 status/metadata/checkpoint，也不接入 Visual Router 或 TimeFuse 正式入口。
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from time_router.io.prediction_array_io import resolve_cache_array_path


REQUIRED_SQLITE_MANIFEST_COLUMNS = [
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


@dataclass(frozen=True)
class PredictionSQLiteBackendMetadata:
    """
    类功能：
        描述 shared prediction SQLite backend 的构建结果。

    关键约束：
        `target_sample_keys` 保留调用方传入顺序；`missing_report` 只记录缺失情况，
        是否把缺失视为错误由 `build_prediction_sqlite_backend(..., allow_missing=...)`
        控制。
    """

    target_sample_keys: Tuple[str, ...]
    expected_records: int
    actual_records: int
    chunk_read_rows: int
    model_columns: Tuple[str, ...]
    manifest_path: str
    manifest_dir: str
    index_db_path: str
    created_at: str
    missing_report: Dict[str, object]


class PreparedPredictionSQLiteBackend:
    """
    类功能：
        封装已准备好的 prediction SQLite 子集索引。

    输入：
        index_db_path: `build_prediction_sqlite_backend(...)` 创建的 SQLite 文件。
        manifest_dir: manifest 所在目录，用于解析相对数组路径。
        model_columns: 调用方要求的专家顺序。
        metadata: 构建时写入的轻量 metadata。

    输出：
        `fetch_records(sample_keys)` 返回 `(sample_key, model_name) -> record` 字典。
        返回字典本身不隐含顺序，调用方可按输入 `sample_keys + model_columns`
        重建稳定顺序。
    """

    def __init__(
        self,
        *,
        index_db_path: Path,
        manifest_dir: Path,
        model_columns: Sequence[str],
        metadata: PredictionSQLiteBackendMetadata,
    ) -> None:
        self.index_db_path = Path(index_db_path)
        self.manifest_dir = Path(manifest_dir)
        self.model_columns = tuple(str(model_name) for model_name in model_columns)
        self.metadata = metadata
        self.connection = sqlite3.connect(str(self.index_db_path))
        self.connection.row_factory = sqlite3.Row

    def fetch_records(self, sample_keys: Sequence[str]) -> Dict[Tuple[str, str], Dict[str, object]]:
        """
        函数功能：
            查询当前 batch 的 prediction records。

        输入：
            sample_keys: 调用方显式传入的当前 batch sample_key 顺序。

        输出：
            `(sample_key, model_name) -> record` 字典，record 的数组路径已解析到
            可读取路径；不读取数组、不计算 loss、不触碰 oracle/feature。
        """
        keys = [str(key) for key in sample_keys]
        if not keys:
            return {}
        placeholders = ",".join(["?"] * len(keys))
        rows = self.connection.execute(
            f"""
            SELECT sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                   array_storage, y_true_row_index, y_pred_row_index
            FROM prediction_index
            WHERE sample_key IN ({placeholders})
            """,
            keys,
        ).fetchall()
        records: Dict[Tuple[str, str], Dict[str, object]] = {}
        for row in rows:
            record = dict(row)
            sample_key = str(record["sample_key"])
            model_name = str(record["model_name"])
            record["y_true_path"] = str(resolve_cache_array_path(str(record["y_true_path"]), self.manifest_dir))
            record["y_pred_path"] = str(resolve_cache_array_path(str(record["y_pred_path"]), self.manifest_dir))
            records[(sample_key, model_name)] = record
        return records

    def close(self) -> None:
        """函数功能：显式关闭 SQLite 连接，释放 smoke 或长任务中的文件句柄。"""
        self.connection.close()

    def __enter__(self) -> "PreparedPredictionSQLiteBackend":
        """函数功能：支持 with 语句，便于测试中稳定关闭连接。"""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """函数功能：退出 with 语句时关闭 SQLite 连接。"""
        self.close()


def build_prediction_sqlite_backend(
    *,
    manifest_path: Path,
    target_sample_keys: Sequence[str],
    index_db_path: Path,
    model_columns: Sequence[str],
    chunk_read_rows: int,
    allow_missing: bool = False,
) -> PreparedPredictionSQLiteBackend:
    """
    函数功能：
        从 prediction manifest 分块构建目标 sample_key 子集 SQLite 索引。

    输入：
        manifest_path: prediction cache manifest CSV。
        target_sample_keys: 调用方已经确定的 sample_key 集合与顺序。
        index_db_path: 成功构建后原子替换到该 SQLite 路径。
        model_columns: 调用方要求的专家顺序。
        chunk_read_rows: pandas 分块读取行数。
        allow_missing: 为 True 时允许缺失并只在 metadata 中记录 missing report；
            默认为 False，缺失 sample/model 会直接报错。

    输出：
        `PreparedPredictionSQLiteBackend`，可通过 `fetch_records(...)` 查询 batch records。

    关键约束：
        不加载全量 manifest 到内存；只写目标 sample_key 的 `(sample_key, model_name)`
        records。SQLite 构建使用临时文件，成功后 `os.replace` 原子替换，失败时删除
        临时文件，避免留下可误用的半成品。
    """
    manifest_path = Path(manifest_path)
    index_db_path = Path(index_db_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 prediction manifest：{manifest_path}")
    ordered_keys = _validate_unique_texts(target_sample_keys, "target_sample_keys")
    ordered_models = _validate_unique_texts(model_columns, "model_columns")
    if not ordered_keys:
        raise ValueError("target_sample_keys 不能为空")
    if not ordered_models:
        raise ValueError("model_columns 不能为空")
    chunk_read_rows = int(chunk_read_rows)
    if chunk_read_rows <= 0:
        raise ValueError("chunk_read_rows 必须为正整数")

    manifest_dir = manifest_path.parent
    index_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db_path = _temporary_sqlite_path(index_db_path)
    if tmp_db_path.exists():
        tmp_db_path.unlink()

    key_set = set(ordered_keys)
    model_set = set(ordered_models)
    expected_pairs = {(sample_key, model_name) for sample_key in ordered_keys for model_name in ordered_models}
    connection: Optional[sqlite3.Connection] = None
    actual_pairs: set[Tuple[str, str]] = set()
    try:
        connection = sqlite3.connect(str(tmp_db_path))
        _create_schema(connection)
        for chunk_df in pd.read_csv(
            manifest_path,
            usecols=REQUIRED_SQLITE_MANIFEST_COLUMNS,
            chunksize=chunk_read_rows,
        ):
            chunk_df = chunk_df.copy()
            chunk_df["sample_key"] = chunk_df["sample_key"].astype(str)
            chunk_df["model_name"] = chunk_df["model_name"].astype(str)
            matched_df = chunk_df[
                chunk_df["sample_key"].isin(key_set) & chunk_df["model_name"].isin(model_set)
            ]
            if matched_df.empty:
                continue
            insert_rows = [_row_to_sqlite_values(row) for row in matched_df.itertuples(index=False)]
            try:
                connection.executemany(
                    """
                    INSERT INTO prediction_index (
                        sample_key, model_name, y_true_path, y_pred_path, mae, mse,
                        array_storage, y_true_row_index, y_pred_row_index
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("prediction manifest 中 sample_key + model_name 存在重复") from exc
            actual_pairs.update((row[0], row[1]) for row in insert_rows)
            connection.commit()
            if expected_pairs.issubset(actual_pairs):
                break

        actual_records = int(connection.execute("SELECT COUNT(*) FROM prediction_index").fetchone()[0])
        expected_records = len(expected_pairs)
        missing_report = _build_missing_report(ordered_keys, ordered_models, actual_pairs)
        if missing_report["missing_records"] and not allow_missing:
            raise ValueError(
                "prediction SQLite 子集不完整："
                f"expected_records={expected_records} actual_records={actual_records} "
                f"missing_records={missing_report['missing_records']} "
                f"missing_samples={missing_report['missing_samples']} "
                f"missing_models_by_sample={missing_report['missing_models_by_sample']}"
            )

        metadata = PredictionSQLiteBackendMetadata(
            target_sample_keys=tuple(ordered_keys),
            expected_records=int(expected_records),
            actual_records=int(actual_records),
            chunk_read_rows=int(chunk_read_rows),
            model_columns=tuple(ordered_models),
            manifest_path=str(manifest_path),
            manifest_dir=str(manifest_dir),
            index_db_path=str(index_db_path),
            created_at=_created_at_cst(),
            missing_report=missing_report,
        )
        _write_metadata(connection, metadata)
        connection.commit()
        connection.close()
        connection = None
        os.replace(tmp_db_path, index_db_path)
        return PreparedPredictionSQLiteBackend(
            index_db_path=index_db_path,
            manifest_dir=manifest_dir,
            model_columns=ordered_models,
            metadata=metadata,
        )
    except Exception:
        if connection is not None:
            connection.close()
        if tmp_db_path.exists():
            tmp_db_path.unlink()
        raise


def load_prediction_sqlite_backend(index_db_path: Path) -> PreparedPredictionSQLiteBackend:
    """
    函数功能：
        从已存在的 SQLite backend 文件读取 metadata 并恢复 prepared backend 对象。

    关键约束：
        该函数只用于读取 P10b helper 写出的 SQLite index，不推断 run_dir 或正式入口
        状态；缺少 metadata 时直接报错，避免误用其他 SQLite 文件。
    """
    index_db_path = Path(index_db_path)
    if not index_db_path.exists():
        raise FileNotFoundError(f"找不到 prediction SQLite backend：{index_db_path}")
    connection = sqlite3.connect(str(index_db_path))
    try:
        rows = connection.execute("SELECT key, value FROM index_metadata").fetchall()
    except sqlite3.Error as exc:
        connection.close()
        raise ValueError(f"SQLite backend 缺少 index_metadata：{index_db_path}") from exc
    connection.close()
    raw = {str(key): json.loads(value) for key, value in rows}
    metadata = PredictionSQLiteBackendMetadata(
        target_sample_keys=tuple(str(key) for key in raw["target_sample_keys"]),
        expected_records=int(raw["expected_records"]),
        actual_records=int(raw["actual_records"]),
        chunk_read_rows=int(raw["chunk_read_rows"]),
        model_columns=tuple(str(model_name) for model_name in raw["model_columns"]),
        manifest_path=str(raw["manifest_path"]),
        manifest_dir=str(raw["manifest_dir"]),
        index_db_path=str(raw["index_db_path"]),
        created_at=str(raw["created_at"]),
        missing_report=dict(raw["missing_report"]),
    )
    return PreparedPredictionSQLiteBackend(
        index_db_path=index_db_path,
        manifest_dir=Path(metadata.manifest_dir),
        model_columns=metadata.model_columns,
        metadata=metadata,
    )


def records_to_ordered_rows(
    records: Mapping[Tuple[str, str], Mapping[str, object]],
    *,
    sample_keys: Sequence[str],
    model_columns: Sequence[str],
) -> List[Mapping[str, object]]:
    """
    函数功能：
        按 `sample_keys + model_columns` 将 fetch_records 返回的 dict 重排为稳定列表。

    输入：
        records: `PreparedPredictionSQLiteBackend.fetch_records(...)` 返回值。
        sample_keys: 调用方期望的 sample 维顺序。
        model_columns: 调用方期望的专家维顺序。

    输出：
        与嵌套顺序严格一致的 record 列表；缺失任一 pair 时直接报错。
    """
    ordered_rows: List[Mapping[str, object]] = []
    for sample_key in [str(key) for key in sample_keys]:
        for model_name in [str(model_name) for model_name in model_columns]:
            key = (sample_key, model_name)
            if key not in records:
                raise KeyError(f"fetch_records 结果缺少 record：sample_key={sample_key} model_name={model_name}")
            ordered_rows.append(records[key])
    return ordered_rows


def _validate_unique_texts(values: Sequence[str], name: str) -> List[str]:
    """函数功能：校验字符串序列非重复，并保留调用方顺序。"""
    texts = [str(value) for value in values]
    if len(texts) != len(set(texts)):
        raise ValueError(f"{name} 存在重复：{texts}")
    return texts


def _temporary_sqlite_path(index_db_path: Path) -> Path:
    """函数功能：生成同目录临时 SQLite 路径，保证后续 os.replace 可原子替换。"""
    suffix = index_db_path.suffix or ".sqlite"
    return index_db_path.with_name(f".{index_db_path.name}.{os.getpid()}.tmp{suffix}")


def _create_schema(connection: sqlite3.Connection) -> None:
    """函数功能：创建 prediction index 和查询索引。"""
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE prediction_index (
            sample_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
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
    connection.execute("CREATE INDEX idx_prediction_index_sample_key ON prediction_index(sample_key)")


def _row_to_sqlite_values(row: object) -> Tuple[object, ...]:
    """函数功能：把 pandas namedtuple 行转换为 SQLite 可写值。"""
    return (
        str(row.sample_key),
        str(row.model_name),
        str(row.y_true_path),
        str(row.y_pred_path),
        float(row.mae),
        float(row.mse),
        None if pd.isna(row.array_storage) else str(row.array_storage),
        None if pd.isna(row.y_true_row_index) else int(row.y_true_row_index),
        None if pd.isna(row.y_pred_row_index) else int(row.y_pred_row_index),
    )


def _build_missing_report(
    sample_keys: Sequence[str],
    model_columns: Sequence[str],
    actual_pairs: set[Tuple[str, str]],
) -> Dict[str, object]:
    """函数功能：生成 sample/model 维度的缺失报告，供 metadata 和报错复用。"""
    missing_by_sample: Dict[str, List[str]] = {}
    for sample_key in sample_keys:
        missing_models = [model_name for model_name in model_columns if (sample_key, model_name) not in actual_pairs]
        if missing_models:
            missing_by_sample[str(sample_key)] = missing_models
    missing_samples = [sample_key for sample_key in sample_keys if len(missing_by_sample.get(sample_key, [])) == len(model_columns)]
    return {
        "missing_records": int(sum(len(models) for models in missing_by_sample.values())),
        "missing_samples": missing_samples,
        "missing_models_by_sample": missing_by_sample,
    }


def _write_metadata(connection: sqlite3.Connection, metadata: PredictionSQLiteBackendMetadata) -> None:
    """函数功能：把 backend metadata 写入 SQLite，便于后续只读恢复和审计。"""
    payload = {
        "target_sample_keys": list(metadata.target_sample_keys),
        "expected_records": metadata.expected_records,
        "actual_records": metadata.actual_records,
        "chunk_read_rows": metadata.chunk_read_rows,
        "model_columns": list(metadata.model_columns),
        "manifest_path": metadata.manifest_path,
        "manifest_dir": metadata.manifest_dir,
        "index_db_path": metadata.index_db_path,
        "created_at": metadata.created_at,
        "missing_report": metadata.missing_report,
    }
    connection.execute("CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in payload.items()],
    )


def _created_at_cst() -> str:
    """函数功能：生成中文实验日志可读的本地 CST 时间戳。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
