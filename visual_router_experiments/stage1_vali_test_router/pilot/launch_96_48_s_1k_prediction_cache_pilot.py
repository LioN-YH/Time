#!/usr/bin/env python3
"""
文件功能：
    生成 96_48_S 1k prediction cache 后台运行 launcher。

输入：
    - `build_stage1_sample_manifest.py` 生成的 sample_manifest.csv；
    - 输出根目录；
    - GPU 绑定列表。

输出：
    - launcher.sh：可直接 `bash launcher.sh` 启动后台 shard；
    - status.json：记录 launcher 生成状态、计划任务和查看进度命令；
    - launch_plan.md：中文运行计划。

关键约束：
    - 本脚本默认只生成 launcher，不自动启动长任务；
    - 三个深度专家分别绑定不同 GPU；两个统计专家默认 CPU 串行/并发独立进程；
    - 每个任务写入独立 shard 目录和 main.log，避免并发写同一 manifest；
    - 合并前必须运行 `merge_prediction_cache_shards.py` 并校验五专家完整性。
"""

from __future__ import annotations

import argparse
import json
import shlex
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
    """函数功能：生成输出目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata/status/plan 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 launcher 生成参数。"""
    parser = argparse.ArgumentParser(description="Create launcher for 96_48_S 1k prediction cache shards.")
    parser.add_argument("--sample-manifest-path", type=Path, required=True, help="1k sample_manifest.csv 路径。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="launcher 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式 launcher 目录。")
    parser.add_argument("--gpus", default="0,1,2", help="深度专家 GPU 绑定列表，逗号分隔；默认使用 0,1,2。")
    parser.add_argument("--deep-batch-size", type=int, default=512, help="深度模型 prediction cache batch size。")
    parser.add_argument("--stat-batch-size", type=int, default=64, help="统计模型 prediction cache batch size。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers。")
    parser.add_argument("--auto-start", action="store_true", help="生成后立即执行 launcher.sh；默认不启动。")
    return parser.parse_args()


def shell_quote(value: object) -> str:
    """函数功能：shell 命令参数安全转义。"""
    return shlex.quote(str(value))


def build_task_commands(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, object]]:
    """函数功能：构造五专家 shard 任务命令。"""
    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if len(gpu_ids) < len(DEEP_MODELS):
        raise ValueError(f"深度专家需要至少 {len(DEEP_MODELS)} 个 GPU 绑定，实际为 {gpu_ids}")

    tasks: List[Dict[str, object]] = []
    builder = STAGE_DIR / "build_prediction_cache_from_manifest.py"
    for model_name, gpu_id in zip(DEEP_MODELS, gpu_ids):
        shard_dir = output_dir / "shards" / model_name
        main_log = shard_dir / "main.log"
        cmd = [
            "CUDA_VISIBLE_DEVICES=" + str(gpu_id),
            str(PYTHON),
            str(builder),
            "--sample-manifest-path",
            str(args.sample_manifest_path),
            "--models",
            model_name,
            "--batch-size",
            str(args.deep_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--local-rank",
            "0",
            "--output-dir",
            str(shard_dir),
            "--device-note",
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
        ]
        tasks.append({"model_name": model_name, "kind": "deep_gpu", "gpu": gpu_id, "output_dir": str(shard_dir), "main_log": str(main_log), "cmd": cmd})

    for model_name in STAT_MODELS:
        shard_dir = output_dir / "shards" / model_name
        main_log = shard_dir / "main.log"
        cmd = [
            str(PYTHON),
            str(builder),
            "--sample-manifest-path",
            str(args.sample_manifest_path),
            "--models",
            model_name,
            "--batch-size",
            str(args.stat_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--local-rank",
            "-1",
            "--output-dir",
            str(shard_dir),
            "--device-note",
            "cpu_statistical_model",
        ]
        tasks.append({"model_name": model_name, "kind": "stat_cpu", "gpu": None, "output_dir": str(shard_dir), "main_log": str(main_log), "cmd": cmd})
    return tasks


def write_launcher(output_dir: Path, tasks: List[Dict[str, object]]) -> Path:
    """函数功能：写出 bash launcher。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = output_dir / "launcher.sh"
    shard_dirs = [Path(str(task["output_dir"])) for task in tasks]
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
        lines.extend(
            [
                f"mkdir -p {shell_quote(shard_dir)}",
                f"echo '[launch] {task['model_name']} -> {shard_dir}'",
                "(",
                "  set -euo pipefail",
                "  " + " ".join(shell_quote(part) for part in task["cmd"]),
                f") > {shell_quote(main_log)} 2>&1 &",
                f"echo $! > {shell_quote(output_dir / 'pids' / (str(task['model_name']) + '.pid'))}",
                "",
            ]
        )
    lines.extend(
        [
            "echo '[launch] started tasks:'",
            f"ls -1 {shell_quote(output_dir / 'pids')}",
            "echo '[launch] progress commands:'",
        ]
    )
    for shard_dir in shard_dirs:
        lines.append(f"echo '  tail -f {shard_dir / 'main.log'}'")
        lines.append(f"echo '  cat {shard_dir / 'status.json'}'")
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
        "# 96_48_S 1k Prediction Cache Launcher Plan",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 输入",
        "",
        f"- sample_manifest: `{args.sample_manifest_path}`",
        "",
        "## GPU/CPU 策略",
        "",
        "- DLinear、PatchTST、CrossFormer 分别绑定不同 GPU；脚本内部使用 `--local-rank 0`，因此每个进程只看到自己的 `cuda:0`。",
        "- ES、NaiveForecaster 为统计模型，默认 CPU 进程；它们不占用 GPU，但可能消耗 CPU 时间。",
        "- 每个专家写独立 shard 目录和 `main.log`，避免并发写同一个 manifest。",
        "",
        "## 启动命令",
        "",
        f"```bash\nbash {launcher_path}\n```",
        "",
        "## 查看进度",
        "",
        "```bash",
        f"tail -f {output_dir}/shards/DLinear/main.log",
        f"cat {output_dir}/shards/DLinear/status.json",
        "nvidia-smi",
        "```",
        "",
        "## 合并命令",
        "",
        "五个 shard 全部 `status=completed` 后执行：",
        "",
        "```bash",
        " ".join(shell_quote(part) for part in merge_cmd),
        "```",
        "",
    ]
    for task in tasks:
        plan_lines.append(f"- `{task['model_name']}`: `{task['output_dir']}`，日志 `{task['main_log']}`")
    (output_dir / "launch_plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    status = {
        "status": "launcher_created",
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "sample_manifest_path": str(args.sample_manifest_path),
        "launcher_path": str(launcher_path),
        "auto_start": bool(args.auto_start),
        "tasks": tasks,
        "merge_command": merge_cmd,
        "progress_commands": [
            f"tail -f {output_dir}/shards/DLinear/main.log",
            f"cat {output_dir}/shards/DLinear/status.json",
            "nvidia-smi",
        ],
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    """函数功能：生成 launcher，并按需自动启动。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_prediction_cache_96_48_s_1k_launcher"
    tasks = build_task_commands(args, output_dir)
    launcher_path = write_launcher(output_dir, tasks)
    write_plan(output_dir, tasks, args, launcher_path)
    print(f"wrote launcher to {launcher_path}")
    print(f"status: {output_dir / 'status.json'}")
    print(f"plan: {output_dir / 'launch_plan.md'}")
    if args.auto_start:
        import subprocess

        subprocess.run(["bash", str(launcher_path)], cwd=str(WORKSPACE), check=True)


if __name__ == "__main__":
    main()
