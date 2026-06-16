#!/usr/bin/env python3
"""
文件功能：
    生成并可启动 Stage 1 `96_48_S` full-scale TimeFuse-derived feature cache
    多 lane CPU launcher。

输入：
    - full-scale `sample_manifest_shard_index.csv`；
    - `build_timefuse_feature_cache_from_manifest.py` 正式 builder。

输出：
    - `launcher.sh`：按 lane 后台执行所有 sample shard；
    - `status.json` / `metadata.json`：记录任务清单、PID、监控命令和恢复命令；
    - `launch_plan.md`：中文运行计划；
    - `shards/sample_shard_XXXX_of_NNNN/`：每个 shard 的 feature cache 输出目录。

关键约束：
    - 本任务只做 CPU 数值元特征提取，不使用 GPU；
    - 每个 shard 独立写 `feature_cache.csv`、`metadata.json`、`status.json`、`main.log`；
    - launcher 和 builder 都支持 `--resume`，已完成 shard 会被跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


WORKSPACE = Path("/home/shiyuhong/Time")
PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
STAGE_DIR = WORKSPACE / "visual_router_experiments" / "stage1_vali_test_router"
FULL_SCALE_ROOT = Path("/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale")
DEFAULT_SHARD_INDEX = FULL_SCALE_ROOT / "sample_manifest_full_scale" / "sample_manifest_shard_index.csv"
DEFAULT_OUTPUT_DIR = FULL_SCALE_ROOT / "timefuse_feature_cache_full_scale_launcher"
DEFAULT_CONFIG = (
    WORKSPACE
    / "quito"
    / "outputs"
    / "default_baseline"
    / "dlinear"
    / "96_48_S"
    / "seed_16"
    / "EVALUATE"
    / "ver_0"
    / "config.yaml"
)


def display_time() -> str:
    """函数功能：生成写入 status/plan 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def parse_args() -> argparse.Namespace:
    """函数功能：解析 TimeFuse feature cache launcher 参数。"""
    parser = argparse.ArgumentParser(description="Create full-scale TimeFuse feature cache launcher.")
    parser.add_argument(
        "--sample-manifest-shard-index-path",
        type=Path,
        default=DEFAULT_SHARD_INDEX,
        help="full-scale sample_manifest_shard_index.csv。",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Quito evaluate config。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="launcher 输出目录。")
    parser.add_argument("--lane-count", type=int, default=8, help="CPU lane 数量；每个 lane 顺序处理若干 sample shard。")
    parser.add_argument("--batch-size", type=int, default=512, help="传给 builder 的 DataLoader batch size。")
    parser.add_argument("--num-workers", type=int, default=0, help="传给 builder 的 DataLoader num_workers。")
    parser.add_argument("--resume", action="store_true", help="launcher 和 builder 均启用断点续跑。")
    parser.add_argument("--auto-start", action="store_true", help="生成 launcher 后立即后台启动。")
    parser.add_argument("--max-shards", type=int, default=None, help="只生成前 N 个 shard 任务，用于 smoke/dry-run。")
    parser.add_argument("--shard-indices", type=int, nargs="+", default=None, help="只生成指定 shard_index 任务。")
    parser.add_argument("--builder-max-samples", type=int, default=None, help="传给 builder 的 --max-samples，用于 smoke。")
    return parser.parse_args()


def shell_quote(value: object) -> str:
    """函数功能：shell 命令参数安全转义。"""
    return shlex.quote(str(value))


def append_root_log(output_dir: Path, message: str) -> None:
    """函数功能：向 launcher 根目录追加主日志。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "main.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def load_manifest_shards(index_path: Path, args: argparse.Namespace) -> List[Dict[str, object]]:
    """
    函数功能：
        读取 full-scale sample shard index，并按命令行选项筛选任务。
    """
    if not index_path.exists():
        raise FileNotFoundError(f"找不到 sample manifest shard index：{index_path}")
    index_df = pd.read_csv(index_path)
    required = {"shard_index", "shard_count", "sample_count", "sample_manifest_path"}
    missing = sorted(required.difference(index_df.columns))
    if missing:
        raise ValueError(f"sample shard index 缺少字段：{missing}")
    index_df = index_df.sort_values("shard_index").reset_index(drop=True)
    if args.shard_indices is not None:
        wanted = set(int(value) for value in args.shard_indices)
        index_df = index_df[index_df["shard_index"].astype(int).isin(wanted)].copy()
    if args.max_shards is not None:
        if int(args.max_shards) <= 0:
            raise ValueError("--max-shards 必须为正整数")
        index_df = index_df.head(int(args.max_shards)).copy()
    if index_df.empty:
        raise ValueError("筛选后没有任何 sample shard 任务")

    rows: List[Dict[str, object]] = []
    for row in index_df.itertuples(index=False):
        sample_path = Path(str(row.sample_manifest_path))
        if not sample_path.is_absolute():
            sample_path = index_path.parent / sample_path
        rows.append(
            {
                "sample_shard_index": int(row.shard_index),
                "sample_shard_count": int(row.shard_count),
                "sample_count": int(row.sample_count),
                "sample_manifest_path": str(sample_path),
            }
        )
    return rows


def build_tasks(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, object]]:
    """函数功能：构造每个 sample shard 的 builder 命令。"""
    if int(args.lane_count) <= 0:
        raise ValueError("--lane-count 必须为正整数")
    builder = STAGE_DIR / "build_timefuse_feature_cache_from_manifest.py"
    shard_rows = load_manifest_shards(Path(args.sample_manifest_shard_index_path), args)
    tasks: List[Dict[str, object]] = []
    for shard_info in shard_rows:
        shard_index = int(shard_info["sample_shard_index"])
        shard_count = int(shard_info["sample_shard_count"])
        shard_dir = output_dir / "shards" / f"sample_shard_{shard_index:04d}_of_{shard_count:04d}"
        main_log = shard_dir / "main.log"
        cmd = [
            str(PYTHON),
            str(builder),
            "--sample-manifest-path",
            str(shard_info["sample_manifest_path"]),
            "--config-path",
            str(args.config_path),
            "--output-dir",
            str(shard_dir),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.builder_max_samples is not None:
            cmd.extend(["--max-samples", str(args.builder_max_samples)])
        tasks.append(
            {
                "sample_shard_index": shard_index,
                "sample_shard_count": shard_count,
                "source_sample_count": int(shard_info["sample_count"]),
                "sample_manifest_path": str(shard_info["sample_manifest_path"]),
                "lane_index": int(shard_index % int(args.lane_count)),
                "output_dir": str(shard_dir),
                "main_log": str(main_log),
                "cmd": cmd,
            }
        )
    return tasks


def tasks_by_lane(tasks: Sequence[Dict[str, object]], lane_count: int) -> Dict[int, List[Dict[str, object]]]:
    """函数功能：按 lane 分配任务，并保持 shard_index 升序。"""
    grouped: Dict[int, List[Dict[str, object]]] = {lane_idx: [] for lane_idx in range(int(lane_count))}
    for task in tasks:
        grouped[int(task["lane_index"])].append(task)
    for lane_idx in grouped:
        grouped[lane_idx] = sorted(grouped[lane_idx], key=lambda item: int(item["sample_shard_index"]))
    return grouped


def write_launcher(output_dir: Path, tasks: Sequence[Dict[str, object]], args: argparse.Namespace) -> Path:
    """函数功能：写出 bash launcher，使用后台 lane 顺序处理 shard。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    lane_script_dir = output_dir / "lane_scripts"
    lane_script_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = output_dir / "launcher.sh"
    grouped = tasks_by_lane(tasks, int(args.lane_count))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shell_quote(WORKSPACE)}",
        f"mkdir -p {shell_quote(output_dir / 'pids')} {shell_quote(output_dir / 'logs')}",
        "",
        f"echo '[launch] TimeFuse feature cache lanes={int(args.lane_count)} at '$(date '+%Y-%m-%d %H:%M:%S %Z')",
        "",
    ]
    for lane_idx in range(int(args.lane_count)):
        lane_tasks = grouped.get(lane_idx, [])
        if not lane_tasks:
            continue
        lane_log = output_dir / "logs" / f"lane_{lane_idx:02d}.log"
        lane_script = lane_script_dir / f"lane_{lane_idx:02d}.sh"
        lane_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shell_quote(WORKSPACE)}",
            f"echo '[lane {lane_idx:02d}] start at '$(date '+%Y-%m-%d %H:%M:%S %Z')",
        ]
        for task in lane_tasks:
            shard_dir = Path(str(task["output_dir"]))
            status_path = shard_dir / "status.json"
            main_log = Path(str(task["main_log"]))
            cmd_text = " ".join(shell_quote(part) for part in task["cmd"])
            lane_lines.extend(
                [
                    f"mkdir -p {shell_quote(shard_dir)}",
                    f"if [[ -f {shell_quote(status_path)} ]] && grep -q '\"status\": \"completed\"' {shell_quote(status_path)}; then",
                    f"  echo '[lane {lane_idx:02d}] skip completed shard {task['sample_shard_index']}/{task['sample_shard_count']}'",
                    "  else",
                    f"  echo '[lane {lane_idx:02d}] run shard {task['sample_shard_index']}/{task['sample_shard_count']} -> {shard_dir}'",
                    f"  {cmd_text} > {shell_quote(main_log)} 2>&1",
                    "fi",
                    "",
                ]
            )
        lane_lines.append(f"echo '[lane {lane_idx:02d}] completed at '$(date '+%Y-%m-%d %H:%M:%S %Z')")
        lane_script.write_text("\n".join(lane_lines) + "\n", encoding="utf-8")
        lane_script.chmod(0o755)
        lines.extend(
            [
                f"setsid bash {shell_quote(lane_script)} > {shell_quote(lane_log)} 2>&1 < /dev/null &",
                f"echo $! > {shell_quote(output_dir / 'pids' / f'lane_{lane_idx:02d}.pid')}",
                "",
            ]
        )
    lines.extend(["echo '[launch] started lane pid files:'", f"ls -1 {shell_quote(output_dir / 'pids')}"])
    launcher_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launcher_path.chmod(0o755)
    return launcher_path


def build_monitor_command(output_dir: Path) -> List[str]:
    """函数功能：生成轻量状态审计命令，写入 status 便于 handoff。"""
    return [
        str(PYTHON),
        "-c",
        (
            "import collections,json,pathlib;"
            f"root=pathlib.Path({str(output_dir / 'shards')!r});"
            "files=list(root.glob('sample_shard_*/status.json'));"
            "c=collections.Counter();"
            "\nfor p in files:\n"
            "    try: c[json.loads(p.read_text()).get('status','unknown')]+=1\n"
            "    except Exception: c['bad_json']+=1\n"
            "print({'status_files':len(files), **dict(c)})"
        ),
    ]


def write_plan_and_status(
    output_dir: Path,
    tasks: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    launcher_path: Path,
    *,
    status: str,
    pids: Optional[Dict[str, int]] = None,
) -> None:
    """函数功能：写中文运行计划、status.json 和 metadata.json。"""
    monitor_command = build_monitor_command(output_dir)
    stop_command = f"for p in {shell_quote(output_dir / 'pids')}/*.pid; do kill -TERM -- -$(cat \"$p\") 2>/dev/null || kill -TERM $(cat \"$p\") 2>/dev/null || true; done"
    resume_command = f"bash {shell_quote(launcher_path)}"
    plan_lines = [
        "# Stage 1 Full-Scale TimeFuse Feature Cache Launcher Plan",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 输入",
        "",
        f"- sample_manifest_shard_index: `{args.sample_manifest_shard_index_path}`",
        f"- config_path: `{args.config_path}`",
        f"- output_dir: `{output_dir}`",
        f"- task_count: `{len(tasks)}`",
        f"- lane_count: `{args.lane_count}`",
        "",
        "## 并行策略",
        "",
        "- 本任务的 17 维 TimeFuse-derived 元特征由 `numpy/scipy/statsmodels` 在 CPU 上计算。",
        "- GPU 对 ADF、ACF、AutoReg、periodogram 这些小窗口统计调用没有直接收益，强行搬到 GPU 反而会增加数据搬运和调度开销。",
        "- 因此采用多 lane CPU 并行；每个 lane 顺序处理若干 sample shard，每个 shard 独立可恢复。",
        "",
        "## 启动命令",
        "",
        "```bash",
        f"bash {launcher_path}",
        "```",
        "",
        "## 监控命令",
        "",
        "```bash",
        " ".join(shell_quote(part) for part in monitor_command),
        f"tail -n 80 {shell_quote(output_dir / 'logs' / 'lane_00.log')}",
        "```",
        "",
        "## 停止命令",
        "",
        "```bash",
        stop_command,
        "```",
        "",
        "## 恢复方式",
        "",
        "- 保留已完成 shard 输出，重新执行启动命令即可；launcher 会跳过 `status=completed` 的 shard。",
        "- 单个失败 shard 可直接重跑对应 `cmd`，builder 的 `--resume` 会保留完整 item 组。",
        "",
        "## 任务清单",
        "",
    ]
    for task in tasks:
        plan_lines.append(
            f"- shard `{task['sample_shard_index']}` lane `{task['lane_index']}`: "
            f"`{task['output_dir']}`，日志 `{task['main_log']}`"
        )
    (output_dir / "launch_plan.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")

    payload = {
        "status": status,
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "sample_manifest_shard_index_path": str(args.sample_manifest_shard_index_path),
        "config_path": str(args.config_path),
        "launcher_path": str(launcher_path),
        "auto_start": bool(args.auto_start),
        "resume": bool(args.resume),
        "task_count": int(len(tasks)),
        "lane_count": int(args.lane_count),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "builder_max_samples": args.builder_max_samples,
        "max_shards": args.max_shards,
        "shard_indices": args.shard_indices,
        "tasks": list(tasks),
        "pids": pids or {},
        "monitor_command": monitor_command,
        "stop_command": stop_command,
        "resume_command": resume_command,
        "gpu_strategy": "not_used_cpu_only; TimeFuse-derived features are CPU-bound statsmodels/scipy/numpy operations",
        "expected_output_files_per_shard": ["feature_cache.csv", "metadata.json", "status.json", "main.log"],
    }
    (output_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_lane_pids(output_dir: Path) -> Dict[str, int]:
    """函数功能：读取 launcher 写出的 lane PID 文件。"""
    pids: Dict[str, int] = {}
    for pid_path in sorted((output_dir / "pids").glob("lane_*.pid")):
        try:
            pids[pid_path.stem] = int(pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            continue
    return pids


def main() -> None:
    """函数功能：生成 launcher，并按需启动 full-scale feature cache 任务。"""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "main.log").exists():
        (output_dir / "main.log").write_text("", encoding="utf-8")

    append_root_log(output_dir, "start TimeFuse feature cache launcher generation")
    tasks = build_tasks(args, output_dir)
    launcher_path = write_launcher(output_dir, tasks, args)
    write_plan_and_status(output_dir, tasks, args, launcher_path, status="launcher_created")
    append_root_log(output_dir, f"wrote launcher to {launcher_path}")

    if args.auto_start:
        append_root_log(output_dir, "auto-start launcher")
        subprocess.run(["bash", str(launcher_path)], cwd=str(WORKSPACE), check=True)
        pids = read_lane_pids(output_dir)
        write_plan_and_status(
            output_dir,
            tasks,
            args,
            launcher_path,
            status="running",
            pids=pids,
        )
        append_root_log(output_dir, f"started lanes pids={pids}")

    print(f"wrote TimeFuse feature cache launcher to {launcher_path}")
    print(f"status: {output_dir / 'status.json'}")
    print(f"plan: {output_dir / 'launch_plan.md'}")


if __name__ == "__main__":
    main()
