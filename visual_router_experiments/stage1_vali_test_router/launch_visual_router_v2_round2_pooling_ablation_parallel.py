#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2 65k pooling ablation 的进程级 launcher。

实验边界：
    - 固定 layout 为 spatial_panel_3view；
    - 复用 Round2e-b 65k expanded feature cache，不重新生成 samples，不重跑 ViT；
    - 只比较 cls / mean_patch / cls_mean_concat 三种 visual input；
    - 每个 variant × seed 写独立 task 目录，统一 summary 只在 aggregate step 单进程写出。
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
TRAIN_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_v2_round2_layout_film.py"
DEFAULT_PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
DEFAULT_SAMPLE_MANIFEST = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_samples/round2_expanded_sample_manifest.csv")
DEFAULT_FEATURE_DIR = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_expanded_layout_validation")
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-23_visual_router_v2_round2_65k_pooling_ablation")
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2" / "65k_pooling_ablation"
DEFAULT_VISUAL_MODES = "cls,mean_patch,cls_mean_concat"


def display_time() -> str:
    """函数功能：生成 launcher 日志时间戳。"""
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
    """函数功能：解析 65k pooling ablation launcher 参数。"""
    parser = argparse.ArgumentParser(description="Launch Round2 65k pooling ablation.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--layout", default="spatial_panel_3view")
    parser.add_argument("--visual-input-modes", default=DEFAULT_VISUAL_MODES)
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke。")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-procs-per-device", type=int, default=1)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def common_train_flags(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 pooling ablation 训练脚本共用参数。"""
    flags = [
        "--sample-manifest",
        str(args.sample_manifest),
        "--output-dir",
        str(args.output_dir),
        "--feature-dir",
        str(args.feature_dir),
        "--feature-artifact-prefix",
        "round2_expanded_layout",
        "--summary-copy-dir",
        str(args.summary_copy_dir),
        "--layout",
        str(args.layout),
        "--layouts",
        str(args.layout),
        "--visual-input-modes",
        str(args.visual_input_modes),
        "--artifact-prefix",
        "round2_65k_pooling",
        "--train-sample-set",
        "round2_train_expanded",
        "--selection-sample-set",
        "round2_selection_expanded",
        "--diagnostic-sample-set",
        "round2_diagnostic_balanced_expanded",
        "--test-sample-set",
        "round2_test_expanded",
        "--experiment-label",
        "Round2 65k pooling ablation",
        "--summary-title",
        "Visual Router V2 Round2 65k Pooling Ablation Summary",
        "--seeds",
        str(args.seeds),
    ]
    if args.max_samples_per_set is not None:
        flags.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return flags


def build_index_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造 prediction subset SQLite 预构建命令。"""
    cmd = [str(args.python), str(TRAIN_SCRIPT), *common_train_flags(args), "--build-index-only"]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def train_command(args: argparse.Namespace, mode: str, seed: int, device: str) -> List[str]:
    """函数功能：构造单 pooling mode/seed 训练评估命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        *common_train_flags(args),
        "--visual-input-mode",
        str(mode),
        "--seed",
        str(seed),
        "--device",
        str(device),
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
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def aggregate_command(args: argparse.Namespace) -> List[str]:
    """函数功能：构造单进程聚合命令。"""
    return [
        str(args.python),
        str(TRAIN_SCRIPT),
        *common_train_flags(args),
        "--devices-requested",
        str(args.devices),
        "--parallel-launcher-used",
        "--aggregate-only",
    ]


def launch_process(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> subprocess.Popen:
    """函数功能：启动子进程并把 stdout/stderr 写入独立日志。"""
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


def run_parallel(args: argparse.Namespace, tasks: Sequence[Tuple[str, str, List[str], Path]]) -> None:
    """函数功能：按 device slots 并行运行互相隔离的训练任务。"""
    devices = parse_csv(args.devices)
    if int(args.max_procs_per_device) < 1:
        raise ValueError("--max-procs-per-device 必须 >= 1")
    device_slots = [device for _round in range(int(args.max_procs_per_device)) for device in devices]
    pending = list(tasks)
    running: Dict[subprocess.Popen, Tuple[str, str, Path]] = {}
    failed: List[Dict[str, object]] = []
    while pending or running:
        while pending and len(running) < len(device_slots):
            task_name, device, cmd, log_path = pending.pop(0)
            print(f"[{display_time()}] launch {task_name} on {device}", flush=True)
            running[launch_process(cmd, cwd=REPO_ROOT, log_path=log_path)] = (task_name, device, log_path)
        time.sleep(float(args.poll_seconds))
        for process in list(running):
            rc = process.poll()
            if rc is None:
                continue
            task_name, device, log_path = running.pop(process)
            close_log(process)
            if rc != 0:
                failed.append({"task": task_name, "device": device, "returncode": rc, "log_path": str(log_path)})
                print(f"[{display_time()}] FAILED {task_name} rc={rc} log={log_path}", flush=True)
            else:
                print(f"[{display_time()}] done {task_name} log={log_path}", flush=True)
    if failed:
        write_json(Path(args.output_dir) / "pooling_launcher_failed_tasks.json", {"failed": failed, "updated_at": display_time()})
        raise RuntimeError(f"{len(failed)} pooling ablation task(s) failed")


def main() -> None:
    """函数功能：执行 index、并行训练和单进程聚合。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "pooling_launcher_metadata.json",
        {
            "status": "started",
            "updated_at": display_time(),
            "layout_fixed_to": str(args.layout),
            "visual_input_modes": parse_csv(args.visual_input_modes),
            "seeds": parse_seeds(args.seeds),
            "feature_dir": str(args.feature_dir),
            "sample_manifest": str(args.sample_manifest),
        },
    )
    if not args.train_only and not args.aggregate_only:
        print(f"[{display_time()}] build prediction subset index", flush=True)
        subprocess.run(build_index_command(args), cwd=str(REPO_ROOT), check=True)
    if not args.aggregate_only:
        modes = parse_csv(args.visual_input_modes)
        seeds = parse_seeds(args.seeds)
        devices = parse_csv(args.devices)
        tasks: List[Tuple[str, str, List[str], Path]] = []
        for idx, (mode, seed) in enumerate((mode, seed) for mode in modes for seed in seeds):
            device = devices[idx % len(devices)]
            task_name = f"{mode}_seed{seed}"
            tasks.append((task_name, device, train_command(args, mode, seed, device), output_dir / "pooling_train_logs" / f"{task_name}.log"))
        run_parallel(args, tasks)
    print(f"[{display_time()}] aggregate pooling ablation", flush=True)
    subprocess.run(aggregate_command(args), cwd=str(REPO_ROOT), check=True)
    write_json(output_dir / "pooling_launcher_metadata.json", {"status": "completed", "updated_at": display_time(), "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
