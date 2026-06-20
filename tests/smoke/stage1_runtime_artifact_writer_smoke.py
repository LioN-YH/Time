#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P11c 最小 Runtime artifact writer smoke。

输入：
    无命令行输入；使用 tempfile 创建本地临时 output_root 和 run_dir。

输出：
    标准输出打印中文检查日志；若 canonical 目录、JSON schema 字段、split count、
    predictions/evaluation 分层或 Provider/run_dir 边界不符合预期则抛出异常。

关键约束：
    不访问 /data2，不启动训练，不修改正式入口，不新增 launcher/scripts。
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.runtime import (  # noqa: E402
    CANONICAL_RUN_SUBDIRS,
    create_run_dir,
    write_evaluation_summary,
    write_prediction_rows_csv,
    write_run_metadata,
    write_run_status,
    write_sample_manifest_ref,
    write_split_summary,
)


CREATED_AT = "2026-06-20T12:40:00+08:00"


class ProviderWithoutRunDir:
    """
    类功能：
        smoke-only Provider mock，用于证明 Provider 只接收 sample_keys，不知道 run_dir。

    输入：
        sample_keys: Runtime 从 manifest/split 解析出的显式样本顺序。

    输出：
        `fetch` 返回按输入顺序构造的内存 rows；不创建、不读取、不保存任何路径。
    """

    def __init__(self, sample_keys: Sequence[str]) -> None:
        self.sample_keys = tuple(sample_keys)
        self.received_run_dir = False

    def fetch(self) -> list[dict[str, object]]:
        """函数功能：返回最小 per-sample prediction rows。"""
        return [
            {
                "sample_key": sample_key,
                "selected_model": "DLinear",
                "y_true": 1.0 + index,
                "y_pred": 1.1 + index,
                "split": "vali" if index == 0 else "test",
            }
            for index, sample_key in enumerate(self.sample_keys)
        ]


def load_json(path: Path) -> dict[str, object]:
    """函数功能：读取 smoke 写出的 JSON 并返回 dict。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def run_smoke() -> None:
    """函数功能：执行 Runtime artifact writer 最小写出 smoke。"""
    print("开始 Stage 1 P11c Runtime artifact writer smoke")

    with tempfile.TemporaryDirectory(prefix="stage1_runtime_artifact_writer_") as temp_dir:
        output_root = Path(temp_dir)
        run_dir = create_run_dir(output_root, run_name="p11c_smoke")
        if run_dir != output_root / "p11c_smoke":
            raise AssertionError(f"run_dir 路径异常：{run_dir}")

        for subdir in CANONICAL_RUN_SUBDIRS:
            if not (run_dir / subdir).is_dir():
                raise AssertionError(f"canonical 子目录缺失：{subdir}")
        print("通过：canonical run_dir 和标准子目录已创建")

        manifest_ref = {
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "reference_type": "path",
            "path": "memory://p11c_smoke_manifest",
            "checksum": "dummy-sha256",
            "checksum_algorithm": "sha256",
            "row_count": 2,
            "ordered_sample_keys_policy": "manifest_row_order",
            "created_at": CREATED_AT,
        }
        split_summary = {
            "split_summary_schema_version": "stage1_split_summary_v1",
            "split_strategy_name": "p11c_tempfile_smoke",
            "config_name": "96_48_S",
            "split_names": ["vali", "test"],
            "sample_count_by_split": {"vali": 1, "test": 1},
            "unique_sample_key_count": 2,
            "duplicate_sample_key_count": 0,
            "split_overlap_check": {
                "default_policy": "mutually_exclusive",
                "allowed_overlap": False,
                "overlap_sample_key_count": 0,
                "overlap_examples": [],
            },
            "ordered_sample_keys_policy": "manifest_row_order",
            "source_manifest_reference": manifest_ref,
            "created_at": CREATED_AT,
        }
        metadata = {
            "run_artifact_schema_version": "stage1_run_artifact_v1",
            "protocol_version": "stage1_canonical_runtime_v1",
            "sample_manifest_schema_version": "stage1_sample_manifest_v1",
            "evaluation_schema_version": "stage1_evaluation_summary_v1",
            "config_name": "96_48_S",
            "branch_name": "runtime_artifact_writer_smoke",
            "created_at": CREATED_AT,
            "inputs": {
                "sample_manifest": "inputs/sample_manifest_ref.json",
                "split_summary": "inputs/split_summary.json",
            },
        }
        status = {
            "status": "completed",
            "current_stage": "finalizing",
            "updated_at": CREATED_AT,
            "failure_reason": None,
            "checkpoint_pointer": None,
        }
        evaluation_summary = {
            "evaluation_schema_version": "stage1_evaluation_summary_v1",
            "sample_count": 2,
            "metrics": {"mae": 0.1, "mse": 0.01},
        }

        write_run_metadata(run_dir, metadata)
        write_run_status(run_dir, status)
        write_sample_manifest_ref(run_dir, manifest_ref)
        write_split_summary(run_dir, split_summary)
        write_evaluation_summary(run_dir, evaluation_summary)

        provider = ProviderWithoutRunDir(("sample_vali_0001", "sample_test_0001"))
        rows = provider.fetch()
        write_prediction_rows_csv(run_dir, rows)
        if provider.received_run_dir:
            raise AssertionError("Provider mock 不应接收 run_dir")

        loaded_metadata = load_json(run_dir / "run_metadata.json")
        loaded_status = load_json(run_dir / "run_status.json")
        loaded_manifest_ref = load_json(run_dir / "inputs" / "sample_manifest_ref.json")
        loaded_split_summary = load_json(run_dir / "inputs" / "split_summary.json")
        loaded_evaluation = load_json(run_dir / "evaluation" / "evaluation_summary.json")
        print("通过：JSON artifact 均可读取")

        if loaded_metadata["run_artifact_schema_version"] != "stage1_run_artifact_v1":
            raise AssertionError("run_metadata schema version 异常")
        if loaded_metadata["sample_manifest_schema_version"] != "stage1_sample_manifest_v1":
            raise AssertionError("run_metadata 未记录 SampleManifest schema version")
        if loaded_status["status"] != "completed":
            raise AssertionError("run_status status 异常")
        if loaded_manifest_ref["sample_manifest_schema_version"] != "stage1_sample_manifest_v1":
            raise AssertionError("sample_manifest_ref schema version 异常")
        if loaded_split_summary["split_summary_schema_version"] != "stage1_split_summary_v1":
            raise AssertionError("split_summary schema version 异常")
        if loaded_evaluation["evaluation_schema_version"] != "stage1_evaluation_summary_v1":
            raise AssertionError("evaluation_summary schema version 异常")
        print("通过：关键 schema version 字段存在且符合预期")

        if loaded_split_summary["sample_count_by_split"] != {"vali": 1, "test": 1}:
            raise AssertionError(f"split_summary count 异常：{loaded_split_summary['sample_count_by_split']}")
        if loaded_split_summary["unique_sample_key_count"] != 2:
            raise AssertionError("split_summary unique_sample_key_count 异常")
        print("通过：split_summary count 正确")

        prediction_csv = run_dir / "predictions" / "prediction_rows.csv"
        if not prediction_csv.is_file():
            raise AssertionError("prediction_rows.csv 未写入 predictions/")
        if (run_dir / "evaluation" / "prediction_rows.csv").exists():
            raise AssertionError("prediction rows 不应写入 evaluation/")
        if (run_dir / "predictions" / "evaluation_summary.json").exists():
            raise AssertionError("evaluation summary 不应写入 predictions/")
        with prediction_csv.open("r", encoding="utf-8", newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        if len(prediction_rows) != 2:
            raise AssertionError(f"prediction rows 行数异常：{len(prediction_rows)}")
        if prediction_rows[0]["sample_key"] != "sample_vali_0001":
            raise AssertionError("prediction rows 未保持 Provider 返回顺序")
        print("通过：predictions/ 与 evaluation/ 分离，per-sample rows 写出正确")

        if "/data2" in str(run_dir):
            raise AssertionError("smoke 不应访问 /data2")
        if not provider.sample_keys == ("sample_vali_0001", "sample_test_0001"):
            raise AssertionError("Provider mock sample_keys 异常")
        print("通过：Provider mock 不知道 run_dir，smoke 未访问 /data2 或启动训练")

    print("Stage 1 P11c Runtime artifact writer smoke 通过")


if __name__ == "__main__":
    run_smoke()
