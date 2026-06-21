#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round 1 P2e FiLM frozen pilot_test eval extension 的
    多 GPU 进程级 launcher。

设计边界：
    - 每个 variant/seed 是一个独立 Python 进程和独立输出子目录；
    - launcher 只做 frozen eval 调度和最终 aggregation，不训练新模型；
    - 默认优先使用 cuda:1,cuda:2,cuda:3，避免默认占用 GPU 0。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "evaluate_visual_router_v2_round1_film_final_test_extension.py"
DEFAULT_PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film_final_test_extension")
FILM_VARIANTS = ("film_cls_mean_concat_aux", "film_mean_patch_aux")


def display_time() -> str:
    """函数功能：生成 launcher metadata 和日志使用的本地时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 P2e FiLM frozen final test 并行 launcher 参数。"""
    parser = argparse.ArgumentParser(description="Launch P2e FiLM frozen pilot_test eval tasks across GPUs.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--devices", default="cuda:1,cuda:2,cuda:3", help="逗号分隔设备；显式包含 cuda:0 时才使用 GPU 0。")
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--variants", default=",".join(FILM_VARIANTS))
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--max-procs-per-device",
        type=int,
        default=1,
        help="每个 device 同时运行的 frozen eval 进程数；显存确认充足时可设为 2。",
    )
    return parser.parse_args()


def parse_csv(text: str) -> List[str]:
    """函数功能：解析逗号分隔字符串并去重保序。"""
    values: List[str] = []
    for part in str(text).split(","):
        part = part.strip()
        if part and part not in values:
            values.append(part)
    if not values:
        raise ValueError("逗号分隔参数不能为空")
    return values


def parse_seeds(text: str) -> List[int]:
    """函数功能：解析 seed 列表。"""
    return [int(value) for value in parse_csv(text)]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def command_for_task(args: argparse.Namespace, variant: str, seed: int, device: str) -> List[str]:
    """函数功能：构造单个 variant/seed 的 frozen eval 命令。"""
    cmd = [
        str(args.python),
        str(EVAL_SCRIPT),
        "--output-dir",
        str(args.output_dir),
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--device",
        device,
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--devices-requested",
        str(args.devices),
        "--parallel-eval-used",
        "--run-single",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def aggregation_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造所有单任务完成后的单进程汇总命令。"""
    cmd = [
        str(args.python),
        str(EVAL_SCRIPT),
        "--output-dir",
        str(args.output_dir),
        "--variants",
        str(args.variants),
        "--seeds",
        str(args.seeds),
        "--devices-requested",
        str(args.devices),
        "--parallel-eval-used",
        "--aggregate-only",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def launch_task(args: argparse.Namespace, task: Tuple[str, int], device: str) -> subprocess.Popen:
    """函数功能：启动一个单任务进程，并把 stdout/stderr 写入独立日志。"""
    variant, seed = task
    log_dir = args.output_dir / "tasks" / f"{variant}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "task.log"
    cmd = command_for_task(args, variant, seed, device)
    (log_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    # device 已作为 torch device 传入；这里不改 CUDA_VISIBLE_DEVICES，便于 metadata 保留真实 GPU 编号。
    process = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    process._codex_log_handle = handle  # type: ignore[attr-defined]
    return process


def close_process_log(process: subprocess.Popen) -> None:
    """函数功能：关闭附着在 Popen 上的日志文件句柄。"""
    handle = getattr(process, "_codex_log_handle", None)
    if handle is not None:
        handle.close()


def main() -> None:
    """函数功能：并行调度 2 variants × 3 seeds，并在全部成功后汇总。"""
    args = parse_args()
    devices = parse_csv(args.devices)
    variants = parse_csv(args.variants)
    if int(args.max_procs_per_device) < 1:
        raise ValueError("--max-procs-per-device 必须 >= 1")
    device_slots = [device for device in devices for _ in range(int(args.max_procs_per_device))]
    seeds = parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(variant, seed) for variant in variants for seed in seeds]
    launcher_meta = {
        "status": "started",
        "started_at": display_time(),
        "launcher": str(Path(__file__).resolve()),
        "eval_script": str(EVAL_SCRIPT),
        "parallel_backend": "process_per_variant_seed",
        "devices_requested": devices,
        "max_procs_per_device": int(args.max_procs_per_device),
        "variants": variants,
        "seeds": seeds,
        "tasks": [{"variant": variant, "seed": seed} for variant, seed in tasks],
        "output_dir": str(args.output_dir),
        "trained_new_model": False,
    }
    write_json(args.output_dir / "launcher_metadata.json", launcher_meta)

    pending = list(tasks)
    running: Dict[subprocess.Popen, Tuple[str, int, str]] = {}
    failed: List[Dict[str, object]] = []
    while pending or running:
        while pending and len(running) < len(device_slots):
            device = device_slots[len(running) % len(device_slots)]
            task = pending.pop(0)
            process = launch_task(args, task, device)
            running[process] = (task[0], task[1], device)
            print(f"[{display_time()}] launched variant={task[0]} seed={task[1]} device={device} pid={process.pid}", flush=True)
        for process, info in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            close_process_log(process)
            variant, seed, device = info
            del running[process]
            print(f"[{display_time()}] finished variant={variant} seed={seed} device={device} returncode={returncode}", flush=True)
            if returncode != 0:
                failed.append({"variant": variant, "seed": seed, "device": device, "returncode": returncode})
        if pending or running:
            time.sleep(float(args.poll_seconds))
    if failed:
        launcher_meta["status"] = "failed"
        launcher_meta["failed"] = failed
        launcher_meta["finished_at"] = display_time()
        write_json(args.output_dir / "launcher_metadata.json", launcher_meta)
        raise SystemExit(f"有 FiLM final eval 子任务失败：{failed}")

    agg_cmd = aggregation_command(args)
    agg_log = args.output_dir / "aggregation.log"
    (args.output_dir / "aggregation_command.txt").write_text(" ".join(agg_cmd) + "\n", encoding="utf-8")
    print(f"[{display_time()}] starting aggregation", flush=True)
    with agg_log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(agg_cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        launcher_meta["status"] = "aggregation_failed"
        launcher_meta["aggregation_returncode"] = result.returncode
        launcher_meta["finished_at"] = display_time()
        write_json(args.output_dir / "launcher_metadata.json", launcher_meta)
        raise SystemExit(f"aggregation 失败，详见 {agg_log}")
    launcher_meta["status"] = "completed"
    launcher_meta["finished_at"] = display_time()
    write_json(args.output_dir / "launcher_metadata.json", launcher_meta)
    print(f"[{display_time()}] P2e FiLM final test parallel launcher completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
