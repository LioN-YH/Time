#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2c layout feature cache 与固定 FiLM screening 的多 GPU
    进程级 launcher。

设计边界：
    - feature 阶段按 layout 启动独立进程，每个 layout 只写自己的 feature 子目录；
    - training 阶段按 layout × seed 启动独立进程，每个任务只写自己的 task 子目录；
    - unified manifest、screening summary 和 status 只由单独 aggregation step 写出；
    - 不使用 DataParallel/DDP，设备通过 `--devices` 显式分配。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "build_visual_router_v2_round2_layout_features.py"
TRAIN_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_v2_round2_layout_film.py"
DEFAULT_PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening")
DEFAULT_LAYOUTS = (
    "current_rgb_3view",
    "spatial_panel_3view",
    "line_only",
    "line_difference_band",
    "fft_absolute_energy",
    "top3fold_period_layout",
)


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
    """函数功能：解析 seeds。"""
    return [int(value) for value in parse_csv(text)]


def parse_args() -> argparse.Namespace:
    """函数功能：解析 Round2c 并行 launcher 参数。"""
    parser = argparse.ArgumentParser(description="Launch Round2c layout feature cache and fixed FiLM screening.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--layouts", default=",".join(DEFAULT_LAYOUTS))
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-shard-size", type=int, default=2000)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true", help="feature 阶段只使用本地 Hugging Face cache。")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke。")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-procs-per-device", type=int, default=1)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """函数功能：稳定写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def common_flags(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 feature/train 子命令共享参数。"""
    flags = ["--output-dir", str(args.output_dir), "--layouts", str(args.layouts)]
    if args.max_samples_per_set is not None:
        flags.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.overwrite:
        flags.append("--overwrite")
    return flags


def feature_command(args: argparse.Namespace, layout: str, device: str) -> List[str]:
    """函数功能：构造单 layout feature worker 命令。"""
    return [
        str(args.python),
        str(FEATURE_SCRIPT),
        "--output-dir",
        str(args.output_dir),
        "--layouts",
        str(args.layouts),
        "--layout",
        layout,
        "--device",
        device,
        "--shard-size",
        str(args.feature_shard_size),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
        *(["--max-samples-per-set", str(args.max_samples_per_set)] if args.max_samples_per_set is not None else []),
        *(["--local-files-only"] if args.local_files_only else []),
        *(["--overwrite"] if args.overwrite else []),
    ]


def feature_aggregate_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 feature aggregation 命令。"""
    cmd = [str(args.python), str(FEATURE_SCRIPT), "--output-dir", str(args.output_dir), "--layouts", str(args.layouts), "--aggregate-only"]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return cmd


def build_index_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 Round2c prediction index 预构建命令。"""
    cmd = [str(args.python), str(TRAIN_SCRIPT), "--output-dir", str(args.output_dir), "--layouts", str(args.layouts), "--seeds", str(args.seeds), "--build-index-only"]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def train_command(args: argparse.Namespace, layout: str, seed: int, device: str) -> List[str]:
    """函数功能：构造单 layout/seed train worker 命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(args.output_dir),
        "--feature-dir",
        str(args.output_dir),
        "--layouts",
        str(args.layouts),
        "--layout",
        layout,
        "--seed",
        str(seed),
        "--seeds",
        str(args.seeds),
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


def train_aggregate_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 training aggregation 命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(args.output_dir),
        "--feature-dir",
        str(args.output_dir),
        "--layouts",
        str(args.layouts),
        "--seeds",
        str(args.seeds),
        "--devices-requested",
        str(args.devices),
        "--parallel-launcher-used",
        "--aggregate-only",
    ]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return cmd


def launch_process(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    """函数功能：启动子进程并把 stdout/stderr 写入指定日志。"""
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
    """函数功能：按 device slots 并行运行一组相互独立任务。"""
    devices = parse_csv(args.devices)
    if int(args.max_procs_per_device) < 1:
        raise ValueError("--max-procs-per-device 必须 >= 1")
    # 按轮次交错设备，避免 max_procs_per_device>1 时先把同一张 GPU 填满。
    # 例如 4 卡、每卡 2 进程时得到 cuda0,cuda1,cuda2,cuda3,cuda0,cuda1...
    device_slots = [device for _round in range(int(args.max_procs_per_device)) for device in devices]
    pending = list(tasks)
    running: Dict[subprocess.Popen, Tuple[str, str, Path]] = {}
    failed: List[Dict[str, object]] = []
    while pending or running:
        while pending and len(running) < len(device_slots):
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
    """函数功能：按请求执行 feature-only、train-only、aggregate-only 或完整流水线。"""
    args = parse_args()
    layouts = parse_csv(args.layouts)
    seeds = parse_seeds(args.seeds)
    devices = parse_csv(args.devices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launcher_meta = {
        "status": "started",
        "started_at": display_time(),
        "launcher": str(Path(__file__).resolve()),
        "feature_script": str(FEATURE_SCRIPT),
        "train_script": str(TRAIN_SCRIPT),
        "output_dir": str(args.output_dir),
        "layouts": layouts,
        "seeds": seeds,
        "devices_requested": devices,
        "feature_only": bool(args.feature_only),
        "train_only": bool(args.train_only),
        "aggregate_only": bool(args.aggregate_only),
        "local_files_only": bool(args.local_files_only),
    }
    write_json(args.output_dir / "launcher_metadata.json", launcher_meta)

    if args.aggregate_only:
        run_command(feature_aggregate_command(args), log_path=args.output_dir / "feature_aggregation.log")
        run_command(train_aggregate_command(args), log_path=args.output_dir / "training_aggregation.log")
    else:
        if not args.train_only:
            feature_tasks: List[Tuple[str, str, List[str], Path]] = []
            for idx, layout in enumerate(layouts):
                device = devices[idx % len(devices)]
                feature_tasks.append((layout, device, feature_command(args, layout, device), args.output_dir / "feature_logs" / f"{layout}.log"))
            run_parallel(args, feature_tasks, phase="feature")
            run_command(feature_aggregate_command(args), log_path=args.output_dir / "feature_aggregation.log")
        if not args.feature_only:
            run_command(build_index_command(args), log_path=args.output_dir / "prediction_index_build.log")
            train_tasks: List[Tuple[str, str, List[str], Path]] = []
            task_idx = 0
            for layout in layouts:
                for seed in seeds:
                    device = devices[task_idx % len(devices)]
                    train_tasks.append((f"{layout}_seed{seed}", device, train_command(args, layout, seed, device), args.output_dir / "tasks" / f"{layout}_seed{seed}" / "task.log"))
                    task_idx += 1
            run_parallel(args, train_tasks, phase="training")
            run_command(train_aggregate_command(args), log_path=args.output_dir / "training_aggregation.log")

    launcher_meta["status"] = "completed"
    launcher_meta["finished_at"] = display_time()
    write_json(args.output_dir / "launcher_metadata.json", launcher_meta)
    print(f"[{display_time()}] Round2c layout screening launcher completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
