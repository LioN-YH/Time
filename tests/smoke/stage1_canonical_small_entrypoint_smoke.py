#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P12 small canonical entrypoint thin slice smoke。

输入：
    无命令行输入；测试内使用 tempfile 调用 `scripts/run_stage1_canonical_small.py`。

输出：
    标准输出打印中文检查日志；若 CLI 返回码、canonical artifact、prediction
    rows 顺序或 Provider/Head/Evaluator 与 run_dir 边界不符合预期则抛错。

关键约束：
    只验证 small/tiny Python entrypoint，不访问 `/data2`，不启动训练，不迁移
    正式 Visual Router / TimeFuse entrypoint，不新增 Bash launcher。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_canonical_small.py"
EXPECTED_SAMPLE_KEYS = (
    "96_48_S__vali__TINY_DATA__item10__ch0__win0",
    "96_48_S__test__TINY_DATA__item10__ch0__win1",
    "96_48_S__test__TINY_DATA__item11__ch1__win2",
)
CANONICAL_SUBDIRS = ("inputs", "indexes", "predictions", "evaluation", "checkpoints", "logs")


def load_json(path: Path) -> dict[str, object]:
    """函数功能：读取 entrypoint 写出的 JSON object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def run_smoke() -> None:
    """函数功能：执行 P12 small canonical entrypoint smoke。"""
    print("开始 Stage 1 P12 small canonical entrypoint smoke")

    with tempfile.TemporaryDirectory(prefix="stage1_p12_small_entrypoint_") as temp_dir:
        output_root = Path(temp_dir) / "run_outputs"
        if str(output_root.resolve()).startswith("/data2/"):
            raise AssertionError("P12 smoke 不应使用 /data2 tempfile")

        cmd = [
            sys.executable,
            str(ENTRYPOINT),
            "--output-root",
            str(output_root),
            "--run-name",
            "p12_small_entrypoint_smoke",
        ]
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "entrypoint subprocess 返回码异常："
                f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        if "run_dir:" not in completed.stdout:
            raise AssertionError(f"entrypoint stdout 未包含 run_dir：{completed.stdout}")
        if "/data2" in completed.stdout or "/data2" in completed.stderr:
            raise AssertionError("P12 smoke stdout/stderr 不应出现 /data2")
        print("通过：entrypoint subprocess 返回码为 0，stdout 包含 run_dir，未引用 /data2")

        run_dir = output_root / "p12_small_entrypoint_smoke"
        for subdir in CANONICAL_SUBDIRS:
            if not (run_dir / subdir).is_dir():
                raise AssertionError(f"canonical 子目录缺失：{subdir}")
        print("通过：canonical 子目录存在")

        metadata = load_json(run_dir / "run_metadata.json")
        status = load_json(run_dir / "run_status.json")
        manifest_ref = load_json(run_dir / "inputs" / "sample_manifest_ref.json")
        split_summary = load_json(run_dir / "inputs" / "split_summary.json")
        evaluation_summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")
        print("通过：run_metadata/run_status/inputs/evaluation JSON 均可读取")

        if metadata["protocol_version"] != "stage1_canonical_small_entrypoint_v1":
            raise AssertionError(f"protocol_version 异常：{metadata}")
        if metadata["branch_name"] != "canonical_small_smoke":
            raise AssertionError("run_metadata branch_name 异常")
        if status["status"] != "completed" or status["current_stage"] != "canonical_small_entrypoint":
            raise AssertionError(f"run_status 异常：{status}")
        if manifest_ref["row_count"] != len(EXPECTED_SAMPLE_KEYS):
            raise AssertionError("sample_manifest_ref row_count 异常")
        if split_summary["sample_count_by_split"] != {"vali": 1, "test": 2}:
            raise AssertionError(f"split_summary count 异常：{split_summary['sample_count_by_split']}")
        if evaluation_summary["sample_count"] != len(EXPECTED_SAMPLE_KEYS):
            raise AssertionError("evaluation_summary sample_count 异常")
        print("通过：关键 schema/version/count 字段符合 P12 tiny fixture")

        prediction_csv = run_dir / "predictions" / "prediction_rows.csv"
        if not prediction_csv.is_file():
            raise AssertionError("prediction_rows.csv 未写入 predictions/")
        if (run_dir / "evaluation" / "prediction_rows.csv").exists():
            raise AssertionError("prediction rows 不应写入 evaluation/")
        if (run_dir / "predictions" / "evaluation_summary.json").exists():
            raise AssertionError("evaluation summary 不应写入 predictions/")
        with prediction_csv.open("r", encoding="utf-8", newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        actual_keys = tuple(row["sample_key"] for row in prediction_rows)
        if actual_keys != EXPECTED_SAMPLE_KEYS:
            raise AssertionError(f"prediction rows 未保持 manifest sample_key 顺序：{actual_keys}")
        if tuple(row["split"] for row in prediction_rows) != ("vali", "test", "test"):
            raise AssertionError(f"prediction rows split 异常：{prediction_rows}")
        print("通过：prediction rows 保持 manifest sample_key 顺序，predictions/evaluation 分层正确")

        # Provider/Head/Evaluator 不接收 run_dir 的边界由 entrypoint strict 模式执行；
        # smoke 通过 subprocess 绿色和文档约束共同验证脚本没有把 run_dir 下传。
        if "train_visual_router_online_streaming" in completed.stdout or "train_timefuse_fusor_streaming" in completed.stdout:
            raise AssertionError("P12 entrypoint 不应启动正式训练入口")
        print("通过：strict 模式确认 Provider/Head/Evaluator 不知道 run_dir，未启动正式训练")

    print("完成：Stage 1 P12 small canonical entrypoint smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
