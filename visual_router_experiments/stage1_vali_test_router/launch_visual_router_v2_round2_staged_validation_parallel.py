#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 staged full-scale validation / 1M gate launcher。

执行边界：
    - 本脚本编排 smoke / one_shard / one_million staged validation，不启动 116M fullscale 长跑；
    - layout 仅支持 `spatial_panel_3view` 与 `current_rgb_3view` 等显式传入集合；
    - 后端固定为 `film_mean_patch_aux`，训练和评估复用现有 Round2 fixed FiLM 入口；
    - prediction lookup 通过 subset SQLite 预构建，避免 worker 并行扫描全量 manifest。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "build_visual_router_v2_round2_staged_samples.py"
FEATURE_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "build_visual_router_v2_round2_layout_features.py"
TRAIN_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_v2_round2_layout_film.py"
SUMMARY_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "summarize_visual_router_v2_round2_staged_validation.py"
DEFAULT_PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_staged_fullscale_validation_thin_slice")
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2" / "staged_fullscale_validation"
DEFAULT_LAYOUTS = ("spatial_panel_3view", "current_rgb_3view")
STAGED_SAMPLE_SETS = ("staged_train", "staged_selection", "staged_diagnostic", "staged_test")
ARTIFACT_PREFIX = "round2_staged_fullscale"


def display_time() -> str:
    """函数功能：生成 launcher 日志时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔参数，并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seeds(text: str) -> List[int]:
    """函数功能：解析 seed 列表。"""
    return [int(value) for value in parse_csv(text)]


def parse_args() -> argparse.Namespace:
    """函数功能：解析 staged launcher 参数。"""
    parser = argparse.ArgumentParser(description="Launch Round2 staged full-scale validation thin slice.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--sample-manifest", type=Path, default=None, help="若省略，则先运行 staged sample builder。")
    parser.add_argument("--sample-scale", choices=["smoke", "one_shard", "one_million"], default="smoke")
    parser.add_argument("--layouts", default=",".join(DEFAULT_LAYOUTS))
    parser.add_argument("--backend", choices=["film_mean_patch_aux"], default="film_mean_patch_aux")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--seeds", default="16")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="本 thin slice 训练入口无单独 eval-only；当前作为 train-only 之后 aggregate 的兼容占位。")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-shard-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--feature-by-sample-set", action="store_true", help="按 layout×sample_set 启动 feature worker，提高 1M gate 多 GPU 利用率。")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="传给 feature/train 的额外 smoke 限制。")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-procs-per-device", type=int, default=1)
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def default_manifest_path(args: argparse.Namespace) -> Path:
    """函数功能：返回 sample builder 默认 manifest 路径。"""
    return Path(args.output_dir) / "inputs" / f"round2_staged_{args.sample_scale}_sample_manifest.csv"


def sample_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 staged sample builder 命令。"""
    cmd = [str(args.python), str(SAMPLE_SCRIPT), "--output-dir", str(args.output_dir), "--sample-scale", str(args.sample_scale)]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def feature_flags(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 feature builder 共用参数。"""
    return [
        "--sample-manifest",
        str(sample_manifest),
        "--output-dir",
        str(args.output_dir),
        "--layouts",
        str(args.layouts),
        "--sample-sets",
        ",".join(STAGED_SAMPLE_SETS),
        "--artifact-prefix",
        ARTIFACT_PREFIX,
    ]


def train_flags(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 fixed FiLM trainer 共用参数。"""
    return [
        "--sample-manifest",
        str(sample_manifest),
        "--output-dir",
        str(args.output_dir),
        "--feature-dir",
        str(args.output_dir),
        "--summary-copy-dir",
        str(args.summary_copy_dir),
        "--layouts",
        str(args.layouts),
        "--seeds",
        str(args.seeds),
        "--artifact-prefix",
        ARTIFACT_PREFIX,
        "--train-sample-set",
        "staged_train",
        "--selection-sample-set",
        "staged_selection",
        "--diagnostic-sample-set",
        "staged_diagnostic",
        "--test-sample-set",
        "staged_test",
        "--experiment-label",
        f"Round2 staged full-scale {args.sample_scale}",
        "--summary-title",
        f"Visual Router V2 Round2 Staged Full-Scale {args.sample_scale} Summary",
    ]


def feature_command(args: argparse.Namespace, sample_manifest: Path, layout: str, device: str, sample_set: str | None = None) -> List[str]:
    """函数功能：构造单 layout feature worker 命令。"""
    flags = feature_flags(args, sample_manifest)
    if sample_set is not None:
        # 1M gate 中按 sample_set worker 拆分，避免单 layout 串行处理四个集合。
        flags = [
            "--sample-manifest",
            str(sample_manifest),
            "--output-dir",
            str(args.output_dir),
            "--layouts",
            str(args.layouts),
            "--sample-sets",
            str(sample_set),
            "--artifact-prefix",
            ARTIFACT_PREFIX,
        ]
    cmd = [
        str(args.python),
        str(FEATURE_SCRIPT),
        *flags,
        "--layout",
        layout,
        "--device",
        device,
        "--shard-size",
        str(args.feature_shard_size),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
    ]
    if sample_set is not None:
        cmd.append("--sample-set-worker")
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def feature_aggregate_command(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 feature manifest aggregation 命令。"""
    cmd = [str(args.python), str(FEATURE_SCRIPT), *feature_flags(args, sample_manifest), "--aggregate-only"]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return cmd


def build_index_command(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 prediction subset SQLite 预构建命令。"""
    cmd = [str(args.python), str(TRAIN_SCRIPT), *train_flags(args, sample_manifest), "--build-index-only"]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def train_command(args: argparse.Namespace, sample_manifest: Path, layout: str, seed: int, device: str) -> List[str]:
    """函数功能：构造单 layout/seed train+eval worker 命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        *train_flags(args, sample_manifest),
        "--layout",
        layout,
        "--seed",
        str(seed),
        "--device",
        device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--devices-requested",
        str(args.devices),
        "--parallel-launcher-used",
        "--run-single",
    ]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def train_aggregate_command(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 fixed FiLM aggregation 命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        *train_flags(args, sample_manifest),
        "--devices-requested",
        str(args.devices),
        "--parallel-launcher-used",
        "--aggregate-only",
    ]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return cmd


def summary_command(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 staged report summary 命令。"""
    return [
        str(args.python),
        str(SUMMARY_SCRIPT),
        "--run-dir",
        str(args.output_dir),
        "--sample-manifest",
        str(sample_manifest),
        "--summary-copy-dir",
        str(args.summary_copy_dir),
        "--artifact-prefix",
        ARTIFACT_PREFIX,
        "--layouts",
        str(args.layouts),
        "--seeds",
        str(args.seeds),
        "--sample-scale",
        str(args.sample_scale),
    ]


def command_preview(args: argparse.Namespace, sample_manifest: Path) -> Dict[str, object]:
    """函数功能：生成 dry-run command preview，不启动任何重任务。"""
    layouts = parse_csv(args.layouts)
    seeds = parse_seeds(args.seeds)
    devices = parse_csv(args.devices)
    train_tasks = []
    task_idx = 0
    for layout in layouts:
        for seed in seeds:
            device = devices[task_idx % len(devices)]
            train_tasks.append(train_command(args, sample_manifest, layout, seed, device))
            task_idx += 1
    return {
        "sample": sample_command(args) if args.sample_manifest is None else ["reuse", str(sample_manifest)],
        "features": [feature_command(args, sample_manifest, layout, devices[idx % len(devices)]) for idx, layout in enumerate(layouts)],
        "feature_aggregate": feature_aggregate_command(args, sample_manifest),
        "prediction_index": build_index_command(args, sample_manifest),
        "train_tasks": train_tasks,
        "train_aggregate": train_aggregate_command(args, sample_manifest),
        "summary": summary_command(args, sample_manifest),
    }


def launch_process(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    """函数功能：启动子进程并写独立日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(".command.txt").write_text(" ".join(str(part) for part in cmd) + "\n", encoding="utf-8")
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(list(cmd), cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    process._codex_log_handle = handle  # type: ignore[attr-defined]
    return process


def close_log(process: subprocess.Popen) -> None:
    """函数功能：关闭挂在 Popen 上的日志句柄。"""
    handle = getattr(process, "_codex_log_handle", None)
    if handle is not None:
        handle.close()


def run_parallel(args: argparse.Namespace, tasks: Sequence[Tuple[str, str, List[str], Path]], *, phase: str) -> None:
    """函数功能：按 device slot 并行运行隔离任务。"""
    devices = parse_csv(args.devices)
    slots = [device for _round in range(int(args.max_procs_per_device)) for device in devices]
    pending = list(tasks)
    running: Dict[subprocess.Popen, Tuple[str, str, Path]] = {}
    failed: List[Dict[str, object]] = []
    while pending or running:
        while pending and len(running) < len(slots):
            task_name, device, cmd, log_path = pending.pop(0)
            process = launch_process(cmd, cwd=REPO_ROOT, log_path=log_path)
            running[process] = (task_name, device, log_path)
            print(f"[{display_time()}] {phase} launched task={task_name} device={device} pid={process.pid}", flush=True)
        for process, info in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            close_log(process)
            task_name, device, log_path = info
            del running[process]
            print(f"[{display_time()}] {phase} finished task={task_name} device={device} returncode={returncode}", flush=True)
            if returncode != 0:
                failed.append({"task": task_name, "device": device, "returncode": returncode, "log_path": str(log_path)})
        if pending or running:
            time.sleep(float(args.poll_seconds))
    if failed:
        raise SystemExit(f"{phase} 子任务失败：{failed}")


def run_command(cmd: Sequence[str], *, log_path: Path) -> None:
    """函数功能：运行单进程阶段命令并校验返回码。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(".command.txt").write_text(" ".join(str(part) for part in cmd) + "\n", encoding="utf-8")
    print(f"[{display_time()}] running {' '.join(str(part) for part in cmd[:3])} ...", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(cmd), cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"命令失败，详见 {log_path}")


def main() -> None:
    """函数功能：执行 dry-run、feature-only、train-only、aggregate-only 或完整 staged thin slice。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = Path(args.sample_manifest) if args.sample_manifest is not None else default_manifest_path(args)
    layouts = parse_csv(args.layouts)
    seeds = parse_seeds(args.seeds)
    devices = parse_csv(args.devices)
    launcher_meta = {
        "status": "dry_run" if args.dry_run else "started",
        "started_at": display_time(),
        "launcher": str(Path(__file__).resolve()),
        "sample_script": str(SAMPLE_SCRIPT),
        "feature_script": str(FEATURE_SCRIPT),
        "train_script": str(TRAIN_SCRIPT),
        "summary_script": str(SUMMARY_SCRIPT),
        "output_dir": str(output_dir),
        "summary_copy_dir": str(args.summary_copy_dir),
        "sample_manifest": str(sample_manifest),
        "sample_scale": str(args.sample_scale),
        "layouts": layouts,
        "backend": str(args.backend),
        "seeds": seeds,
        "devices_requested": devices,
        "sample_sets": list(STAGED_SAMPLE_SETS),
        "constraints": {
            "not_1m_run": str(args.sample_scale) != "one_million",
            "is_1m_staged_gate": str(args.sample_scale) == "one_million",
            "not_116m_full_scale_run": True,
            "loaded_116m_prediction_manifest_to_memory": False,
            "saved_pseudo_image_tensor": False,
            "router_head_logic_changed": False,
            "imageization_semantics_changed": False,
        },
    }
    if args.dry_run:
        launcher_meta["command_preview"] = command_preview(args, sample_manifest)
        write_json(output_dir / "staged_launcher_metadata.json", launcher_meta)
        print(json.dumps(launcher_meta, indent=2, ensure_ascii=False))
        return
    write_json(output_dir / "staged_launcher_metadata.json", launcher_meta)

    if args.aggregate_only:
        run_command(feature_aggregate_command(args, sample_manifest), log_path=output_dir / "feature_aggregation.log")
        run_command(train_aggregate_command(args, sample_manifest), log_path=output_dir / "training_aggregation.log")
        run_command(summary_command(args, sample_manifest), log_path=output_dir / "staged_summary.log")
    else:
        if args.sample_manifest is None and not args.train_only:
            run_command(sample_command(args), log_path=output_dir / "sample_builder.log")
        if not args.train_only:
            feature_tasks: List[Tuple[str, str, List[str], Path]] = []
            if args.feature_by_sample_set:
                task_idx = 0
                for layout in layouts:
                    for sample_set in STAGED_SAMPLE_SETS:
                        device = devices[task_idx % len(devices)]
                        task_name = f"{layout}_{sample_set}"
                        feature_tasks.append(
                            (
                                task_name,
                                device,
                                feature_command(args, sample_manifest, layout, device, sample_set=sample_set),
                                output_dir / "feature_logs" / f"{task_name}.log",
                            )
                        )
                        task_idx += 1
            else:
                for idx, layout in enumerate(layouts):
                    device = devices[idx % len(devices)]
                    feature_tasks.append((layout, device, feature_command(args, sample_manifest, layout, device), output_dir / "feature_logs" / f"{layout}.log"))
            run_parallel(args, feature_tasks, phase="feature")
            run_command(feature_aggregate_command(args, sample_manifest), log_path=output_dir / "feature_aggregation.log")
        if not args.feature_only:
            run_command(build_index_command(args, sample_manifest), log_path=output_dir / "prediction_index_build.log")
            train_tasks: List[Tuple[str, str, List[str], Path]] = []
            task_idx = 0
            for layout in layouts:
                for seed in seeds:
                    device = devices[task_idx % len(devices)]
                    train_tasks.append((f"{layout}_seed{seed}", device, train_command(args, sample_manifest, layout, seed, device), output_dir / "tasks" / f"{layout}_seed{seed}" / "task.log"))
                    task_idx += 1
            run_parallel(args, train_tasks, phase="training")
            run_command(train_aggregate_command(args, sample_manifest), log_path=output_dir / "training_aggregation.log")
            run_command(summary_command(args, sample_manifest), log_path=output_dir / "staged_summary.log")

    meta = json.loads((output_dir / "staged_launcher_metadata.json").read_text(encoding="utf-8"))
    meta.update({"status": "completed", "finished_at": display_time()})
    write_json(output_dir / "staged_launcher_metadata.json", meta)
    print(f"[{display_time()}] Round2 staged full-scale validation launcher completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
