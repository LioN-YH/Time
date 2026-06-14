#!/usr/bin/env python3
"""
文件功能：
    生成 Stage 1 full-scale prediction cache 可恢复 launcher。

输入：
    - sample_manifest.csv；
    - sample shard 数量；
    - GPU 绑定列表。

输出：
    - launcher.sh：启动全部专家/sample shard 任务；
    - status.json：记录任务清单、进度命令和 merge 命令；
    - launch_plan.md：中文运行计划。

关键约束：
    - 默认仅生成 launcher，不自动启动长任务；
    - DLinear/PatchTST/CrossFormer 绑定 GPU 并行，ES/NaiveForecaster 独立 CPU 进程；
    - 每个任务写独立 `main.log` 和 shard 输出目录，失败后可精确重跑单个任务；
    - 默认使用 `packed_npy_v1`，避免 full-scale per-sample 小文件爆炸。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
STAGE_DIR = WORKSPACE / "visual_router_experiments" / "stage1_vali_test_router"
DEEP_MODELS = ["DLinear", "PatchTST", "CrossFormer"]
STAT_MODELS = ["ES", "NaiveForecaster"]


def now_token() -> str:
    """函数功能：生成输出目录时间戳。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 status/plan 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 launcher 生成参数。"""
    parser = argparse.ArgumentParser(description="Create full-scale Stage 1 prediction cache launcher.")
    parser.add_argument("--sample-manifest-path", type=Path, required=True, help="sample_manifest.csv 路径。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="launcher 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式 launcher 目录。")
    parser.add_argument("--sample-shard-count", type=int, default=1, help="每个专家拆成多少 sample shard。")
    parser.add_argument("--gpus", default="0,1,2", help="深度专家 GPU 绑定列表，逗号分隔。")
    parser.add_argument("--deep-batch-size", type=int, default=512, help="深度模型 batch size。")
    parser.add_argument("--stat-batch-size", type=int, default=64, help="统计模型 batch size。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers。")
    parser.add_argument("--array-storage", choices=["packed_npy_v1", "per_sample_npy"], default="packed_npy_v1", help="数组落盘模式。")
    parser.add_argument("--auto-start", action="store_true", help="生成后立即执行 launcher.sh。")
    return parser.parse_args()


def shell_quote(value: object) -> str:
    """函数功能：shell 命令参数安全转义。"""
    return shlex.quote(str(value))


def build_task_commands(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, object]]:
    """函数功能：构造专家 × sample shard 任务命令。"""
    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if len(gpu_ids) < len(DEEP_MODELS):
        raise ValueError(f"深度专家需要至少 {len(DEEP_MODELS)} 个 GPU 绑定，实际为 {gpu_ids}")
    if int(args.sample_shard_count) <= 0:
        raise ValueError("--sample-shard-count 必须为正整数")

    tasks: List[Dict[str, object]] = []
    builder = STAGE_DIR / "build_prediction_cache_from_manifest.py"
    for model_name in [*DEEP_MODELS, *STAT_MODELS]:
        for shard_index in range(int(args.sample_shard_count)):
            shard_dir = output_dir / "shards" / model_name / f"sample_shard_{shard_index:04d}_of_{int(args.sample_shard_count):04d}"
            main_log = shard_dir / "main.log"
            is_deep = model_name in DEEP_MODELS
            gpu_id = gpu_ids[DEEP_MODELS.index(model_name)] if is_deep else None
            batch_size = int(args.deep_batch_size if is_deep else args.stat_batch_size)
            cmd = [
                str(PYTHON),
                str(builder),
                "--sample-manifest-path",
                str(args.sample_manifest_path),
                "--models",
                model_name,
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(args.num_workers),
                "--local-rank",
                "0" if is_deep else "-1",
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(args.sample_shard_count),
                "--array-storage",
                str(args.array_storage),
                "--output-dir",
                str(shard_dir),
                "--device-note",
                f"CUDA_VISIBLE_DEVICES={gpu_id}" if is_deep else "cpu_statistical_model",
            ]
            tasks.append(
                {
                    "model_name": model_name,
                    "sample_shard_index": int(shard_index),
                    "sample_shard_count": int(args.sample_shard_count),
                    "kind": "deep_gpu" if is_deep else "stat_cpu",
                    "gpu": gpu_id,
                    "output_dir": str(shard_dir),
                    "main_log": str(main_log),
                    "cmd": cmd,
                }
            )
    return tasks


def write_launcher(output_dir: Path, tasks: List[Dict[str, object]]) -> Path:
    """函数功能：写出 bash launcher。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = output_dir / "launcher.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shell_quote(WORKSPACE)}",
        f"mkdir -p {shell_quote(output_dir / 'pids')}",
        "",
    ]
    for task in tasks:
        shard_dir = Path(str(task["output_dir"]))
        main_log = Path(str(task["main_log"]))
        env_prefix = f"CUDA_VISIBLE_DEVICES={task['gpu']} " if task["gpu"] is not None else ""
        lines.extend(
            [
                f"mkdir -p {shell_quote(shard_dir)}",
                f"echo '[launch] {task['model_name']} shard {task['sample_shard_index']}/{task['sample_shard_count']} -> {shard_dir}'",
                "(",
                "  set -euo pipefail",
                "  " + env_prefix + " ".join(shell_quote(part) for part in task["cmd"]),
                f") > {shell_quote(main_log)} 2>&1 &",
                f"echo $! > {shell_quote(output_dir / 'pids' / (task['model_name'] + '_' + str(task['sample_shard_index']) + '.pid'))}",
                "",
            ]
        )
    lines.extend(["echo '[launch] started tasks:'", f"ls -1 {shell_quote(output_dir / 'pids')}"])
    launcher_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launcher_path.chmod(0o755)
    return launcher_path


def write_plan(output_dir: Path, tasks: List[Dict[str, object]], args: argparse.Namespace, launcher_path: Path) -> None:
    """函数功能：写出中文运行计划和 status。"""
    merge_cmd = [
        str(PYTHON),
        str(STAGE_DIR / "merge_prediction_cache_shards.py"),
        "--shard-dirs",
        *[str(task["output_dir"]) for task in tasks],
        "--output-dir",
        str(output_dir / "merged_cache"),
    ]
    plan_lines = [
        "# Stage 1 Full-Scale Prediction Cache Launcher Plan",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 输入与分片",
        "",
        f"- sample_manifest: `{args.sample_manifest_path}`",
        f"- sample_shard_count: `{args.sample_shard_count}`",
        f"- array_storage: `{args.array_storage}`",
        "",
        "## GPU/CPU 策略",
        "",
        "- DLinear、PatchTST、CrossFormer 分别绑定不同 GPU；每个任务内部 `--local-rank 0`。",
        "- ES、NaiveForecaster 使用 CPU 独立进程。",
        "- 每个专家/sample shard 写独立 `main.log`、`status.json` 和 manifest；失败后只重跑对应 shard。",
        "",
        "## 启动命令",
        "",
        f"```bash\nbash {launcher_path}\n```",
        "",
        "## 合并命令",
        "",
        "所有 shard `status=completed` 后执行：",
        "",
        "```bash",
        " ".join(shell_quote(part) for part in merge_cmd),
        "```",
        "",
        "## 任务清单",
        "",
    ]
    for task in tasks:
        plan_lines.append(f"- `{task['model_name']}` shard `{task['sample_shard_index']}`: `{task['output_dir']}`，日志 `{task['main_log']}`")
    (output_dir / "launch_plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    status = {
        "status": "launcher_created",
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "main_log": str(output_dir / "main.log"),
        "sample_manifest_path": str(args.sample_manifest_path),
        "launcher_path": str(launcher_path),
        "auto_start": bool(args.auto_start),
        "sample_shard_count": int(args.sample_shard_count),
        "array_storage": str(args.array_storage),
        "tasks": tasks,
        "merge_command": merge_cmd,
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_root_log(output_dir: Path, message: str) -> None:
    """函数功能：向 launcher 根目录追加执行日志。"""
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def main() -> None:
    """函数功能：生成 launcher，并按需启动。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_prediction_cache_full_scale_launcher"
    output_dir.mkdir(parents=True, exist_ok=True)
    root_log = output_dir / "main.log"
    root_log.write_text("", encoding="utf-8")
    append_root_log(output_dir, "start full-scale prediction cache launcher generation")
    tasks = build_task_commands(args, output_dir)
    launcher_path = write_launcher(output_dir, tasks)
    write_plan(output_dir, tasks, args, launcher_path)
    append_root_log(output_dir, f"wrote launcher to {launcher_path}")
    append_root_log(output_dir, "completed launcher generation")
    print(f"wrote launcher to {launcher_path}")
    print(f"status: {output_dir / 'status.json'}")
    print(f"plan: {output_dir / 'launch_plan.md'}")
    if args.auto_start:
        subprocess.run(["bash", str(launcher_path)], cwd=str(WORKSPACE), check=True)


if __name__ == "__main__":
    main()
