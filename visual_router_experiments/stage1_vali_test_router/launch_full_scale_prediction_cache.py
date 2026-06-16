#!/usr/bin/env python3
"""
文件功能：
    生成 Stage 1 full-scale prediction cache 可恢复 launcher。

输入：
    - sample_manifest.csv，或 full-scale sample_manifest_shard_index.csv；
    - sample shard 数量；
    - GPU 绑定列表。

输出：
    - launcher.sh：启动全部专家/sample shard 任务；
    - status.json：记录任务清单、进度命令和 merge 命令；
    - launch_plan.md：中文运行计划。

关键约束：
    - 默认仅生成 launcher，不自动启动长任务；
    - DLinear/PatchTST/CrossFormer 绑定 GPU 并行，ES/NaiveForecaster 独立 CPU worker；
    - 每个任务写独立 `main.log` 和 shard 输出目录，失败后可精确重跑单个任务；
    - 默认使用 `packed_npy_v1`，避免 full-scale per-sample 小文件爆炸。
    - 若提供 shard index，launcher 会为每个专家启动一个后台 worker，worker 内顺序
      执行所有 sample shard，避免一次性启动几百个进程。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
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
    parser.add_argument("--sample-manifest-path", type=Path, default=None, help="未预分片 sample_manifest.csv 路径。")
    parser.add_argument(
        "--sample-manifest-shard-index-path",
        type=Path,
        default=None,
        help="build_full_scale_sample_manifest.py 生成的 sample_manifest_shard_index.csv；正式 full-scale 推荐使用。",
    )
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


def load_manifest_shards(args: argparse.Namespace) -> List[Dict[str, object]]:
    """
    函数功能：解析 launcher 应消费的 sample manifest shard。

    说明：
        正式 full-scale 使用 `sample_manifest_shard_index.csv`，让每个 builder 进程只读
        自己的分片 manifest，避免每个任务重复读取 2000 万行总表。兼容路径仍保留给
        dry-run 或小规模单 manifest 使用。
    """
    if args.sample_manifest_shard_index_path is not None and args.sample_manifest_path is not None:
        raise ValueError("--sample-manifest-path 与 --sample-manifest-shard-index-path 只能二选一")
    if args.sample_manifest_shard_index_path is None and args.sample_manifest_path is None:
        raise ValueError("必须提供 --sample-manifest-path 或 --sample-manifest-shard-index-path")

    if args.sample_manifest_shard_index_path is not None:
        index_path = Path(args.sample_manifest_shard_index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"找不到 sample manifest shard index：{index_path}")
        index_df = pd.read_csv(index_path)
        required = {"shard_index", "shard_count", "sample_count", "sample_manifest_path"}
        missing = sorted(required.difference(index_df.columns))
        if missing:
            raise ValueError(f"sample shard index 缺少字段：{missing}")
        rows = []
        for row in index_df.sort_values("shard_index").itertuples(index=False):
            sample_path = Path(str(row.sample_manifest_path))
            if not sample_path.is_absolute():
                sample_path = index_path.parent / sample_path
            rows.append(
                {
                    "source_shard_index": int(row.shard_index),
                    "source_shard_count": int(row.shard_count),
                    "sample_count": int(row.sample_count),
                    "sample_manifest_path": str(sample_path),
                    "builder_shard_index": 0,
                    "builder_shard_count": 1,
                }
            )
        return rows

    if int(args.sample_shard_count) <= 0:
        raise ValueError("--sample-shard-count 必须为正整数")
    rows = []
    for shard_index in range(int(args.sample_shard_count)):
        rows.append(
            {
                "source_shard_index": int(shard_index),
                "source_shard_count": int(args.sample_shard_count),
                "sample_count": None,
                "sample_manifest_path": str(args.sample_manifest_path),
                "builder_shard_index": int(shard_index),
                "builder_shard_count": int(args.sample_shard_count),
            }
        )
    return rows


def build_task_commands(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, object]]:
    """函数功能：构造专家 × sample shard 任务命令。"""
    gpu_ids = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if len(gpu_ids) < len(DEEP_MODELS):
        raise ValueError(f"深度专家需要至少 {len(DEEP_MODELS)} 个 GPU 绑定，实际为 {gpu_ids}")

    tasks: List[Dict[str, object]] = []
    builder = STAGE_DIR / "build_prediction_cache_from_manifest.py"
    manifest_shards = load_manifest_shards(args)
    for model_name in [*DEEP_MODELS, *STAT_MODELS]:
        for shard_info in manifest_shards:
            shard_index = int(shard_info["source_shard_index"])
            shard_count = int(shard_info["source_shard_count"])
            shard_dir = output_dir / "shards" / model_name / f"sample_shard_{shard_index:04d}_of_{shard_count:04d}"
            main_log = shard_dir / "main.log"
            is_deep = model_name in DEEP_MODELS
            gpu_id = gpu_ids[DEEP_MODELS.index(model_name)] if is_deep else None
            batch_size = int(args.deep_batch_size if is_deep else args.stat_batch_size)
            cmd = [
                str(PYTHON),
                str(builder),
                "--sample-manifest-path",
                str(shard_info["sample_manifest_path"]),
                "--models",
                model_name,
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(args.num_workers),
                "--local-rank",
                "0" if is_deep else "-1",
                "--shard-index",
                str(shard_info["builder_shard_index"]),
                "--shard-count",
                str(shard_info["builder_shard_count"]),
                "--array-storage",
                str(args.array_storage),
                "--resume",
                "--output-dir",
                str(shard_dir),
                "--device-note",
                f"CUDA_VISIBLE_DEVICES={gpu_id}" if is_deep else "cpu_statistical_model",
            ]
            tasks.append(
                {
                    "model_name": model_name,
                    "sample_shard_index": int(shard_index),
                    "sample_shard_count": int(shard_count),
                    "source_sample_count": shard_info["sample_count"],
                    "sample_manifest_path": str(shard_info["sample_manifest_path"]),
                    "kind": "deep_gpu" if is_deep else "stat_cpu",
                    "gpu": gpu_id,
                    "output_dir": str(shard_dir),
                    "main_log": str(main_log),
                    "cmd": cmd,
                }
            )
    return tasks


def model_task_groups(tasks: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    """函数功能：按模型名分组任务，供 launcher 生成每专家 worker。"""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for task in tasks:
        grouped.setdefault(str(task["model_name"]), []).append(task)
    for model_name in grouped:
        grouped[model_name] = sorted(grouped[model_name], key=lambda item: int(item["sample_shard_index"]))
    return grouped


def unique_sample_shard_count(tasks: List[Dict[str, object]]) -> int:
    """函数功能：统计唯一 sample shard 数，避免按五专家任务数重复计数。"""
    return int(len({int(task["sample_shard_index"]) for task in tasks}))


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
    grouped = model_task_groups(tasks)
    for model_name in [*DEEP_MODELS, *STAT_MODELS]:
        model_tasks = grouped.get(model_name, [])
        if not model_tasks:
            continue
        worker_log = output_dir / "shards" / model_name / "worker.log"
        gpu = model_tasks[0]["gpu"]
        env_prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        lines.extend(
            [
                f"mkdir -p {shell_quote(worker_log.parent)}",
                f"echo '[launch] worker {model_name} -> {worker_log}'",
                "(",
                "  set -euo pipefail",
                f"  echo '[worker] {model_name} start at '$(date '+%Y-%m-%d %H:%M:%S %Z')",
            ]
        )
        for task in model_tasks:
            shard_dir = Path(str(task["output_dir"]))
            main_log = Path(str(task["main_log"]))
            status_path = shard_dir / "status.json"
            cmd_text = " ".join(shell_quote(part) for part in task["cmd"])
            pid_name = task["model_name"] + "_" + str(task["sample_shard_index"]) + ".pid"
            del pid_name
            lines.extend(
                [
                    f"  mkdir -p {shell_quote(shard_dir)}",
                    f"  if [[ -f {shell_quote(status_path)} ]] && grep -q '\"status\": \"completed\"' {shell_quote(status_path)}; then",
                    f"    echo '[worker] skip completed {task['model_name']} shard {task['sample_shard_index']}/{task['sample_shard_count']}'",
                    "  else",
                    f"    echo '[worker] run {task['model_name']} shard {task['sample_shard_index']}/{task['sample_shard_count']} -> {shard_dir}'",
                    f"    {env_prefix}{cmd_text} > {shell_quote(main_log)} 2>&1",
                    "  fi",
                    "",
                ]
            )
        lines.extend(
            [
                f"  echo '[worker] {model_name} completed at '$(date '+%Y-%m-%d %H:%M:%S %Z')",
                f") > {shell_quote(worker_log)} 2>&1 &",
                f"echo $! > {shell_quote(output_dir / 'pids' / (model_name + '.pid'))}",
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
        f"- sample_manifest_shard_index: `{args.sample_manifest_shard_index_path}`",
        f"- sample_shard_count: `{unique_sample_shard_count(tasks)}`",
        f"- array_storage: `{args.array_storage}`",
        "",
        "## GPU/CPU 策略",
        "",
        "- DLinear、PatchTST、CrossFormer 分别绑定不同 GPU；每个任务内部 `--local-rank 0`。",
        "- 每个专家启动一个后台 worker；worker 内顺序执行 sample shard，避免一次性拉起过多进程。",
        "- ES、NaiveForecaster 使用 CPU 独立 worker。",
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
        "sample_manifest_path": str(args.sample_manifest_path) if args.sample_manifest_path is not None else None,
        "sample_manifest_shard_index_path": str(args.sample_manifest_shard_index_path) if args.sample_manifest_shard_index_path is not None else None,
        "launcher_path": str(launcher_path),
        "auto_start": bool(args.auto_start),
        "sample_shard_count": unique_sample_shard_count(tasks),
        "array_storage": str(args.array_storage),
        "worker_mode": "one_background_worker_per_model_sequential_sample_shards",
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
