#!/usr/bin/env python3
"""
文件功能：
    执行 Stage 1 `96_48_S` full-scale 框架 dry-run。

设计约束：
    - dry-run 只抽取小样本，目的是验证 full-scale 模板的闭环和输出契约；
    - prediction cache 使用 `packed_npy_v1`，避免 per-sample 小文件口径；
    - streaming online router 不保存伪图像 tensor 或 ViT embedding；
    - 每个子步骤写 `main.log`、`status.json` 和 metadata，失败后可从已完成步骤继续。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
STAGE_DIR = WORKSPACE / "visual_router_experiments" / "stage1_vali_test_router"
PILOT_DIR = STAGE_DIR / "pilot"
DEFAULT_SAMPLE_MANIFEST = (
    RUN_OUTPUT_ROOT
    / "2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k"
    / "sample_manifest.csv"
)
MODELS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]


def now_token() -> str:
    """函数功能：生成 run 目录时间戳。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 status/metadata 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 dry-run 参数。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 full-scale framework dry-run.")
    parser.add_argument("--source-sample-manifest-path", type=Path, default=DEFAULT_SAMPLE_MANIFEST, help="已有 sample_manifest.csv 来源。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="dry-run 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式输出目录。")
    parser.add_argument("--samples-per-split", type=int, default=4, help="每个 split 取多少 sample_key。")
    parser.add_argument("--sample-shard-count", type=int, default=2, help="prediction cache dry-run sample shard 数。")
    parser.add_argument("--embedding-batch-size", type=int, default=4, help="streaming router ViT batch size。")
    parser.add_argument("--router-epochs", type=int, default=1, help="streaming router dry-run epoch 数。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="streaming router 设备。")
    parser.add_argument("--local-files-only", action="store_true", help="streaming router 只使用本地 HF cache。")
    parser.add_argument("--skip-existing", action="store_true", help="若步骤 status=completed 则跳过。")
    return parser.parse_args()


def shell_quote(value: object) -> str:
    """函数功能：shell 命令参数安全转义，用于 metadata 展示。"""
    return shlex.quote(str(value))


def write_status(path: Path, status: Dict[str, object]) -> None:
    """函数功能：写 status.json。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_completed(step_dir: Path) -> bool:
    """函数功能：判断某步骤是否已经完成。"""
    status_path = step_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        return json.loads(status_path.read_text(encoding="utf-8")).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def append_root_log(output_dir: Path, message: str) -> None:
    """函数功能：向 dry-run 根目录 main.log 追加一行总控日志。"""
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def run_step(name: str, cmd: Sequence[object], step_dir: Path, *, skip_existing: bool, output_dir: Path) -> None:
    """
    函数功能：
        执行一个 dry-run 步骤，并写 main.log/status.json。

    说明：
        失败时保留 main.log 和 failed status，便于后续定位并重跑单步。
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    status_path = step_dir / "status.json"
    main_log = step_dir / "main.log"
    if skip_existing and is_completed(step_dir):
        append_root_log(output_dir, f"skip completed step: {name}")
        return
    started_at = display_time()
    append_root_log(output_dir, f"start step: {name}")
    write_status(status_path, {"status": "running", "step": name, "started_at": started_at, "cmd": [str(part) for part in cmd]})
    with main_log.open("w", encoding="utf-8") as log_f:
        log_f.write(f"[{started_at}] running: {' '.join(shell_quote(part) for part in cmd)}\n")
        log_f.flush()
        try:
            subprocess.run([str(part) for part in cmd], cwd=str(WORKSPACE), stdout=log_f, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            write_status(
                status_path,
                {
                    "status": "failed",
                    "step": name,
                    "started_at": started_at,
                    "updated_at": display_time(),
                    "cmd": [str(part) for part in cmd],
                    "main_log": str(main_log),
                    "error": repr(exc),
                },
            )
            append_root_log(output_dir, f"failed step: {name}; see {main_log}")
            raise
    write_status(
        status_path,
        {
            "status": "completed",
            "step": name,
            "started_at": started_at,
            "updated_at": display_time(),
            "cmd": [str(part) for part in cmd],
            "main_log": str(main_log),
        },
    )
    append_root_log(output_dir, f"completed step: {name}")


def build_dry_manifest(source_path: Path, output_dir: Path, samples_per_split: int) -> Path:
    """函数功能：从已有 manifest 抽取小样本 dry-run manifest。"""
    if not source_path.exists():
        raise FileNotFoundError(f"找不到 source sample manifest：{source_path}")
    df = pd.read_csv(source_path)
    rows = []
    for split, split_df in df.groupby("split", sort=True):
        rows.append(split_df.sort_values(["dataset_name", "item_id", "channel_id", "window_index"]).head(int(samples_per_split)))
    dry_df = pd.concat(rows, ignore_index=True)
    manifest_dir = output_dir / "sample_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "sample_manifest.csv"
    dry_df.to_csv(manifest_path, index=False)
    metadata = {
        "status": "completed",
        "generated_at": display_time(),
        "source_sample_manifest_path": str(source_path),
        "sample_manifest_path": str(manifest_path),
        "samples_per_split": int(samples_per_split),
        "sample_count": int(len(dry_df)),
        "split_counts": {str(k): int(v) for k, v in dry_df["split"].value_counts().sort_index().items()},
    }
    write_status(manifest_dir / "status.json", metadata)
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def restore_completed_root_status(output_dir: Path, root_status: Path, args: argparse.Namespace) -> bool:
    """
    函数功能：
        当显式输出目录已完成且用户传入 `--skip-existing` 时，恢复根 status 并直接返回。

    说明：
        这解决手动重复执行某个子步骤后根 `status.json` 被失败状态覆盖的问题。只要
        metadata 证明完整 dry-run 已完成，就把根 status 恢复为 completed，避免后续
        监控误判。
    """
    metadata_path = output_dir / "metadata.json"
    if not bool(args.skip_existing) or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metadata.get("status") != "completed":
        return False
    metadata["status_restored_at"] = display_time()
    write_status(root_status, metadata)
    append_root_log(output_dir, "root status restored from completed metadata because --skip-existing was set")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return True


def main() -> None:
    """函数功能：执行 full-scale dry-run 闭环。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_full_scale_dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_status = output_dir / "status.json"
    if restore_completed_root_status(output_dir, root_status, args):
        return
    root_log = output_dir / "main.log"
    root_log.write_text("", encoding="utf-8")
    append_root_log(output_dir, "start full-scale dry-run orchestration")
    write_status(
        root_status,
        {"status": "running", "started_at": display_time(), "output_dir": str(output_dir), "main_log": str(root_log)},
    )

    try:
        dry_manifest = build_dry_manifest(args.source_sample_manifest_path, output_dir, int(args.samples_per_split))
        append_root_log(output_dir, f"built dry sample manifest: {dry_manifest}")

        shard_dirs: List[Path] = []
        for model_name in MODELS:
            for shard_index in range(int(args.sample_shard_count)):
                shard_dir = output_dir / "prediction_cache_shards" / model_name / f"sample_shard_{shard_index:04d}"
                shard_dirs.append(shard_dir)
                run_step(
                    f"prediction_cache_{model_name}_{shard_index}",
                    [
                        PYTHON,
                        STAGE_DIR / "build_prediction_cache_from_manifest.py",
                        "--sample-manifest-path",
                        dry_manifest,
                        "--models",
                        model_name,
                        "--shard-index",
                        shard_index,
                        "--shard-count",
                        int(args.sample_shard_count),
                        "--array-storage",
                        "packed_npy_v1",
                        "--batch-size",
                        4 if model_name in {"DLinear", "PatchTST", "CrossFormer"} else 2,
                        "--output-dir",
                        shard_dir,
                        "--device-note",
                        "dry_run_cpu_or_visible_device",
                    ],
                    shard_dir,
                    skip_existing=bool(args.skip_existing),
                    output_dir=output_dir,
                )

        merged_dir = output_dir / "merged_cache"
        run_step(
            "merge_prediction_cache",
            [
                PYTHON,
                STAGE_DIR / "merge_prediction_cache_shards.py",
                "--shard-dirs",
                *shard_dirs,
                "--output-dir",
                merged_dir,
            ],
            output_dir / "steps" / "merge_prediction_cache",
            skip_existing=bool(args.skip_existing),
            output_dir=output_dir,
        )
        run_step(
            "compute_oracle",
            [PYTHON, PILOT_DIR / "compute_window_oracle_from_cache.py", "--cache-dir", merged_dir],
            output_dir / "steps" / "compute_oracle",
            skip_existing=bool(args.skip_existing),
            output_dir=output_dir,
        )
        run_step(
            "enrich_tsf_cell",
            [PYTHON, PILOT_DIR / "enrich_cache_with_tsf_cell.py", "--cache-dir", merged_dir],
            output_dir / "steps" / "enrich_tsf_cell",
            skip_existing=bool(args.skip_existing),
            output_dir=output_dir,
        )
        run_step(
            "evaluate_baselines",
            [PYTHON, STAGE_DIR / "evaluate_router_baselines.py", "--labels-path", merged_dir / "window_oracle_labels_with_tsf_cell.csv"],
            output_dir / "steps" / "evaluate_baselines",
            skip_existing=bool(args.skip_existing),
            output_dir=output_dir,
        )

        router_dir = output_dir / "streaming_online_router"
        router_cmd = [
            PYTHON,
            STAGE_DIR / "train_visual_router_online_streaming.py",
            "--labels-path",
            merged_dir / "window_oracle_labels_with_tsf_cell.csv",
            "--prediction-manifest-path",
            merged_dir / "manifest.csv",
            "--output-dir",
            router_dir,
            "--epochs",
            int(args.router_epochs),
            "--embedding-batch-size",
            int(args.embedding_batch_size),
            "--batch-size",
            4,
            "--device",
            str(args.device),
            "--skip-soft-fusion",
        ]
        if args.local_files_only:
            router_cmd.append("--local-files-only")
        run_step("streaming_online_router", router_cmd, router_dir, skip_existing=bool(args.skip_existing), output_dir=output_dir)

        calibration_dir = output_dir / "soft_fusion_calibration"
        run_step(
            "soft_fusion_calibration",
            [
                PYTHON,
                STAGE_DIR / "evaluate_soft_fusion_calibration.py",
                "--router-predictions-path",
                router_dir / "visual_router_predictions.csv",
                "--prediction-manifest-path",
                merged_dir / "manifest.csv",
                "--labels-path",
                merged_dir / "window_oracle_labels_with_tsf_cell.csv",
                "--output-dir",
                calibration_dir,
                "--temperatures",
                "0.5,1.0",
                "--top-k-values",
                "all,1,2",
            ],
            calibration_dir,
            skip_existing=bool(args.skip_existing),
            output_dir=output_dir,
        )

        status = {
            "status": "completed",
            "updated_at": display_time(),
            "output_dir": str(output_dir),
            "main_log": str(root_log),
            "dry_sample_manifest_path": str(dry_manifest),
            "merged_cache_dir": str(merged_dir),
            "streaming_router_dir": str(router_dir),
            "calibration_dir": str(calibration_dir),
            "array_storage": "packed_npy_v1",
            "embedding_storage": "runtime_only_not_saved",
            "sample_shard_count": int(args.sample_shard_count),
        }
        write_status(root_status, status)
        (output_dir / "metadata.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        append_root_log(output_dir, "completed full-scale dry-run orchestration")
        print(json.dumps(status, indent=2, ensure_ascii=False))
    except Exception as exc:
        append_root_log(output_dir, f"failed full-scale dry-run orchestration: {repr(exc)}")
        write_status(
            root_status,
            {
                "status": "failed",
                "updated_at": display_time(),
                "output_dir": str(output_dir),
                "main_log": str(root_log),
                "error": repr(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
