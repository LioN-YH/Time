#!/usr/bin/env python3
"""
文件功能：
    为 Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline 生成并启动
    正式后台 launcher。

设计约束：
    - 只做编排、preflight 和接手信息写入，训练/eval 仍复用
      `train_timefuse_fusor_streaming.py`；
    - 输入固定覆盖 64 个 feature shard 和同编号五专家 prediction shard；
    - 不改成全量 manifest lookup 或全量 DataFrame join；
    - 后台启动写出 PID/PGID、主日志、状态、metadata、停止命令和恢复命令；
    - GPU 模式只允许物理 GPU 2/3；默认使用 CPU，避免抢占正在运行的
      full-scale visual router eval-only 任务。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


WORKSPACE = Path("/home/shiyuhong/Time")
PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
DATA2_RUN_OUTPUT_ROOT = Path("/data2/syh/Time/run_outputs")
FULL_SCALE_ROOT = DATA2_RUN_OUTPUT_ROOT / "2026-06-15_stage1_96_48_s_full_scale"
FEATURE_SHARD_ROOT = FULL_SCALE_ROOT / "timefuse_feature_cache_full_scale_launcher" / "shards"
PREDICTION_SHARD_ROOT = FULL_SCALE_ROOT / "prediction_cache_full_scale_launcher" / "shards"
ORACLE_LABELS_PATH = (
    FULL_SCALE_ROOT
    / "prediction_cache_full_scale_launcher"
    / "oracle_labels_full_scale_2026-06-16"
    / "window_oracle_labels.parquet"
)
MERGED_CACHE_STATUS = FULL_SCALE_ROOT / "prediction_cache_full_scale_launcher" / "merged_cache" / "status.json"
TRAIN_SCRIPT = WORKSPACE / "visual_router_experiments" / "stage1_vali_test_router" / "train_timefuse_fusor_streaming.py"
DEFAULT_OUTPUT_DIR = DATA2_RUN_OUTPUT_ROOT / "2026-06-18_stage1_timefuse_fusor_full_scale_cpu"
MODEL_COLUMNS = ["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]
EXPECTED_SAMPLE_COUNT = 23_275_170
EXPECTED_PREDICTION_RECORDS = EXPECTED_SAMPLE_COUNT * len(MODEL_COLUMNS)


def display_time() -> str:
    """函数功能：生成本地 CST 时间字符串，供 status/metadata/main.log 使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def shell_quote(value: object) -> str:
    """函数功能：对 shell 参数做安全引用。"""
    return shlex.quote(str(value))


def to_jsonable(value: object) -> object:
    """函数功能：把 Path 等对象转为 JSON 可写对象。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：原子写 JSON，避免监控时读到半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def append_log(output_dir: Path, message: str) -> None:
    """函数功能：向 launcher 根级 main.log 追加中文事件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "launcher.log").open("a", encoding="utf-8") as log_f:
        log_f.write(f"[{display_time()}] {message}\n")


def read_json(path: Path) -> Dict[str, object]:
    """函数功能：读取 JSON 文件并返回 dict。"""
    return json.loads(path.read_text(encoding="utf-8"))


def path_size(path: Path) -> int:
    """函数功能：返回文件大小；缺失时返回 0 便于 preflight 报错聚合。"""
    try:
        return int(path.stat().st_size)
    except FileNotFoundError:
        return 0


def discover_feature_shards(root: Path) -> List[Path]:
    """函数功能：发现 64 个正式 TimeFuse feature shard。"""
    return sorted(Path(root).glob("sample_shard_*_of_0064/feature_cache.csv"))


def expected_prediction_manifests(shard_name: str, root: Path) -> List[Path]:
    """函数功能：生成当前 sample shard 对应的五专家 prediction manifest 路径。"""
    return [Path(root) / model_name / shard_name / "manifest.csv" for model_name in MODEL_COLUMNS]


def read_csv_header(path: Path) -> List[str]:
    """函数功能：只读取 CSV 表头，用于轻量 schema preflight。"""
    with path.open("r", encoding="utf-8", newline="") as csv_f:
        return next(csv.reader(csv_f))


def process_is_alive(pid: int) -> bool:
    """函数功能：检查 PID 是否仍存在。"""
    return Path(f"/proc/{int(pid)}").exists()


def current_process_matches(pattern: str) -> List[str]:
    """函数功能：用 pgrep 查找当前机器上匹配的长跑进程。"""
    result = subprocess.run(["pgrep", "-af", pattern], text=True, capture_output=True, check=False)
    lines = []
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if parts and parts[0].isdigit() and int(parts[0]) == os.getpid():
            continue
        if "pgrep -af" in line:
            continue
        lines.append(line)
    return lines


def output_dir_has_only_launcher_files(output_dir: Path) -> bool:
    """
    函数功能：
        判断已有输出目录是否只包含 launcher 自己生成的元文件。

    说明：
        preflight 需要先创建目录写报告；这些文件不应被误判为“已有正式任务”。
        一旦出现 indexes/checkpoints/predictions 等训练产物，仍会触发冲突保护。
    """
    allowed_names = {
        "preflight_report.json",
        "metadata.json",
        "status.json",
        "launcher.log",
        "command.sh",
        "command_resume.sh",
        "launcher.sh",
        "stop.sh",
        "resume.sh",
    }
    if not output_dir.exists():
        return True
    return all(path.name in allowed_names for path in output_dir.iterdir())


def disk_snapshot(paths: Sequence[Path]) -> Dict[str, Dict[str, object]]:
    """函数功能：采集目标挂载点磁盘容量，preflight 判断输出空间是否足够。"""
    snapshot: Dict[str, Dict[str, object]] = {}
    for path in paths:
        usage = shutil.disk_usage(path)
        snapshot[str(path)] = {
            "total_gb": round(usage.total / 1024**3, 3),
            "used_gb": round(usage.used / 1024**3, 3),
            "free_gb": round(usage.free / 1024**3, 3),
        }
    return snapshot


def nvidia_smi_snapshot() -> Dict[str, object]:
    """函数功能：采集 GPU 轻量状态；无 nvidia-smi 时只记录错误。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        return {"rows": result.stdout.strip().splitlines()}
    except Exception as exc:  # pragma: no cover - 只影响机器观测
        return {"error": str(exc)}


def build_preflight(args: argparse.Namespace) -> Dict[str, object]:
    """
    函数功能：
        检查正式 full-scale fusor 启动前置条件。

    检查范围：
        oracle labels、feature shards、prediction manifests、磁盘、已有进程、
        输出目录和 GPU 约束。这里不扫描大 manifest 正文。
    """
    output_dir = Path(args.output_dir)
    feature_shards = discover_feature_shards(args.feature_shard_root)
    feature_status_counts: Dict[str, int] = {}
    feature_rows = 0
    feature_schema_errors: List[str] = []
    required_feature_cols = {
        "sample_key",
        "split",
        "config_name",
        "mean",
        "std",
        "spectral_kurtosis",
    }
    for feature_path in feature_shards:
        status_path = feature_path.parent / "status.json"
        if status_path.exists():
            status = read_json(status_path)
            status_value = str(status.get("status"))
            feature_status_counts[status_value] = feature_status_counts.get(status_value, 0) + 1
            feature_rows += int(status.get("sample_count") or status.get("rows_written") or 0)
        else:
            feature_status_counts["missing_status"] = feature_status_counts.get("missing_status", 0) + 1
        try:
            header = set(read_csv_header(feature_path))
            missing = sorted(required_feature_cols.difference(header))
            if missing:
                feature_schema_errors.append(f"{feature_path}: missing={missing}")
        except Exception as exc:
            feature_schema_errors.append(f"{feature_path}: {exc}")

    missing_prediction_manifests: List[str] = []
    prediction_manifest_count = 0
    prediction_schema_errors: List[str] = []
    required_prediction_cols = {
        "sample_key",
        "model_name",
        "y_true_path",
        "y_pred_path",
        "mae",
        "mse",
        "array_storage",
        "y_true_row_index",
        "y_pred_row_index",
    }
    for feature_path in feature_shards:
        shard_name = feature_path.parent.name
        for manifest_path in expected_prediction_manifests(shard_name, args.prediction_shard_root):
            if not manifest_path.exists():
                missing_prediction_manifests.append(str(manifest_path))
                continue
            prediction_manifest_count += 1
            try:
                header = set(read_csv_header(manifest_path))
                missing = sorted(required_prediction_cols.difference(header))
                if missing:
                    prediction_schema_errors.append(f"{manifest_path}: missing={missing}")
            except Exception as exc:
                prediction_schema_errors.append(f"{manifest_path}: {exc}")

    merged_status = read_json(args.merged_cache_status) if Path(args.merged_cache_status).exists() else {}
    oracle_status_path = args.labels_path.parent / "status.json"
    oracle_status = read_json(oracle_status_path) if oracle_status_path.exists() else {}
    existing_fusor_processes = current_process_matches("train_timefuse_fusor_streaming.py")
    existing_launcher_processes = current_process_matches("launch_timefuse_fusor_full_scale.py")
    visual_router_processes = current_process_matches("train_visual_router_online_streaming.py")

    output_pid_path = output_dir / "pid.txt"
    output_pid_alive = False
    output_pid: Optional[int] = None
    if output_pid_path.exists():
        try:
            output_pid = int(output_pid_path.read_text(encoding="utf-8").strip())
            output_pid_alive = process_is_alive(output_pid)
        except ValueError:
            output_pid_alive = False

    failures: List[str] = []
    warnings: List[str] = []
    if len(feature_shards) != 64:
        failures.append(f"feature shard 数量不是 64：actual={len(feature_shards)}")
    if feature_status_counts.get("completed") != 64:
        failures.append(f"feature shard 未全部 completed：{feature_status_counts}")
    if feature_rows != EXPECTED_SAMPLE_COUNT:
        failures.append(f"feature status 行数合计不等于预期：actual={feature_rows} expected={EXPECTED_SAMPLE_COUNT}")
    if feature_schema_errors:
        failures.append(f"feature schema 检查失败 {len(feature_schema_errors)} 个")
    if not Path(args.labels_path).exists() or path_size(args.labels_path) == 0:
        failures.append(f"oracle labels parquet 缺失或为空：{args.labels_path}")
    if oracle_status.get("status") != "completed":
        failures.append(f"oracle labels status 不是 completed：{oracle_status}")
    if prediction_manifest_count != 64 * len(MODEL_COLUMNS):
        failures.append(f"prediction manifest 数量不是 320：actual={prediction_manifest_count}")
    if missing_prediction_manifests:
        failures.append(f"prediction manifest 缺失 {len(missing_prediction_manifests)} 个")
    if prediction_schema_errors:
        failures.append(f"prediction schema 检查失败 {len(prediction_schema_errors)} 个")
    if merged_status.get("status") != "completed":
        failures.append(f"merged cache status 不是 completed：{merged_status}")
    if int(merged_status.get("record_count", 0)) != EXPECTED_PREDICTION_RECORDS:
        failures.append(f"merged cache record_count 不等于预期：{merged_status.get('record_count')}")
    if int(merged_status.get("sample_count", 0)) != EXPECTED_SAMPLE_COUNT:
        failures.append(f"merged cache sample_count 不等于预期：{merged_status.get('sample_count')}")
    if existing_fusor_processes:
        failures.append(f"已有 TimeFuse fusor 训练进程：{existing_fusor_processes}")
    if output_pid_alive:
        failures.append(f"输出目录已有存活 PID：pid={output_pid}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output_dir and not output_dir_has_only_launcher_files(output_dir):
        failures.append(f"输出目录已存在且非空：{output_dir}")
    if shutil.disk_usage(output_dir.parent).free < int(args.min_free_gb) * 1024**3:
        failures.append(f"输出根目录剩余空间小于 {args.min_free_gb}GB：{output_dir.parent}")
    if args.device == "cuda":
        visible = str(args.cuda_visible_devices).replace(" ", "")
        if visible not in {"2", "3", "2,3"}:
            failures.append(f"GPU 模式只允许 CUDA_VISIBLE_DEVICES=2,3 或单卡子集，实际={args.cuda_visible_devices}")
    if visual_router_processes and args.device == "cuda":
        warnings.append("GPU2/3 上仍有 visual router 任务；可继续但可能争抢资源")
    if visual_router_processes and args.device == "cpu":
        warnings.append("检测到 visual router eval-only 正在运行；本 launcher 使用 CPU 避免抢 GPU2/3")
    if existing_launcher_processes:
        warnings.append(f"检测到 launcher 相关进程：{existing_launcher_processes}")

    return {
        "checked_at": display_time(),
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "output_dir": str(output_dir),
        "device": str(args.device),
        "cuda_visible_devices": str(args.cuda_visible_devices) if args.device == "cuda" else "",
        "feature_shard_count": len(feature_shards),
        "feature_status_counts": feature_status_counts,
        "feature_rows_from_status": feature_rows,
        "feature_schema_error_count": len(feature_schema_errors),
        "feature_schema_errors_sample": feature_schema_errors[:5],
        "prediction_manifest_count": prediction_manifest_count,
        "missing_prediction_manifests_sample": missing_prediction_manifests[:5],
        "prediction_schema_error_count": len(prediction_schema_errors),
        "prediction_schema_errors_sample": prediction_schema_errors[:5],
        "oracle_labels_path": str(args.labels_path),
        "oracle_labels_size_bytes": path_size(args.labels_path),
        "oracle_status": oracle_status,
        "merged_cache_status": merged_status,
        "disk": disk_snapshot([output_dir.parent, WORKSPACE]),
        "nvidia_smi": nvidia_smi_snapshot(),
        "existing_fusor_processes": existing_fusor_processes,
        "visual_router_processes": visual_router_processes,
        "output_pid": output_pid,
        "output_pid_alive": output_pid_alive,
    }


def train_command(args: argparse.Namespace, *, resume: bool) -> List[str]:
    """函数功能：组装调用 streaming train/eval 入口的命令。"""
    feature_shards = discover_feature_shards(args.feature_shard_root)
    cmd = [
        str(PYTHON),
        str(TRAIN_SCRIPT),
        "--labels-path",
        str(args.labels_path),
        "--prediction-shard-root",
        str(args.prediction_shard_root),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--huber-beta",
        str(args.huber_beta),
        "--seed",
        str(args.seed),
        "--max-feature-shards",
        "64",
        "--feature-read-chunk-rows",
        str(args.feature_read_chunk_rows),
        "--prediction-chunk-rows",
        str(args.prediction_chunk_rows),
        "--oracle-parquet-batch-rows",
        str(args.oracle_parquet_batch_rows),
        "--prediction-num-workers",
        str(args.prediction_num_workers),
        "--prefetch-batches",
        str(args.prefetch_batches),
        "--status-update-interval",
        str(args.status_update_interval),
        "--sample-prediction-limit",
        str(args.sample_prediction_limit),
    ]
    if args.train_only:
        cmd.append("--train-only")
    if resume:
        checkpoint = Path(args.output_dir) / "checkpoints" / "latest_timefuse_fusor.pt"
        cmd.extend(["--resume-checkpoint", str(checkpoint)])
    for feature_path in feature_shards:
        cmd.extend(["--feature-shard-path", str(feature_path)])
    return cmd


def write_shell_scripts(args: argparse.Namespace) -> Dict[str, str]:
    """函数功能：写出 command/launcher/stop/resume shell 脚本。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_lines = [
        "export PYTHONUNBUFFERED=1",
        f"export OMP_NUM_THREADS={int(args.omp_num_threads)}",
    ]
    if args.device == "cuda":
        env_lines.append(f"export CUDA_VISIBLE_DEVICES={shell_quote(args.cuda_visible_devices)}")
    else:
        # CPU 正式跑时显式隐藏 CUDA，避免 torch 初始化或误占 GPU2/3。
        env_lines.append("export CUDA_VISIBLE_DEVICES=")

    initial_cmd = " ".join(shell_quote(part) for part in train_command(args, resume=False))
    resume_cmd = " ".join(shell_quote(part) for part in train_command(args, resume=True))
    command_path = output_dir / "command.sh"
    command_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shell_quote(WORKSPACE)}",
                *env_lines,
                initial_cmd,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)

    resume_command_path = output_dir / "command_resume.sh"
    resume_command_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shell_quote(WORKSPACE)}",
                *env_lines,
                f"if [[ -f {shell_quote(output_dir / 'checkpoints' / 'latest_timefuse_fusor.pt')} ]]; then",
                f"  {resume_cmd}",
                "else",
                f"  {initial_cmd}",
                "fi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resume_command_path.chmod(0o755)

    launcher_path = output_dir / "launcher.sh"
    launcher_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"mkdir -p {shell_quote(output_dir)}",
                f"if [[ -f {shell_quote(output_dir / 'pid.txt')} ]] && kill -0 \"$(cat {shell_quote(output_dir / 'pid.txt')})\" 2>/dev/null; then",
                "  echo 'existing process is still alive' >&2",
                "  exit 1",
                "fi",
                f"setsid bash {shell_quote(command_path)} > {shell_quote(output_dir / 'main.log')} 2>&1 < /dev/null &",
                "pid=$!",
                f"echo \"$pid\" > {shell_quote(output_dir / 'pid.txt')}",
                "sleep 1",
                "pgid=$(ps -o pgid= -p \"$pid\" | tr -d ' ' || true)",
                "if [[ -z \"$pgid\" ]]; then pgid=\"$pid\"; fi",
                f"echo \"$pgid\" > {shell_quote(output_dir / 'pgid.txt')}",
                "echo \"started pid=$pid pgid=$pgid\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)

    stop_path = output_dir / "stop.sh"
    stop_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"if [[ -f {shell_quote(output_dir / 'pgid.txt')} ]]; then",
                f"  kill -TERM -- -$(cat {shell_quote(output_dir / 'pgid.txt')}) 2>/dev/null || true",
                f"elif [[ -f {shell_quote(output_dir / 'pid.txt')} ]]; then",
                f"  kill -TERM $(cat {shell_quote(output_dir / 'pid.txt')}) 2>/dev/null || true",
                "fi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stop_path.chmod(0o755)

    resume_path = output_dir / "resume.sh"
    resume_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"if [[ -f {shell_quote(output_dir / 'pid.txt')} ]] && kill -0 \"$(cat {shell_quote(output_dir / 'pid.txt')})\" 2>/dev/null; then",
                "  echo 'existing process is still alive' >&2",
                "  exit 1",
                "fi",
                f"setsid bash {shell_quote(resume_command_path)} > {shell_quote(output_dir / 'main.log')} 2>&1 < /dev/null &",
                "pid=$!",
                f"echo \"$pid\" > {shell_quote(output_dir / 'pid.txt')}",
                "sleep 1",
                "pgid=$(ps -o pgid= -p \"$pid\" | tr -d ' ' || true)",
                "if [[ -z \"$pgid\" ]]; then pgid=\"$pid\"; fi",
                f"echo \"$pgid\" > {shell_quote(output_dir / 'pgid.txt')}",
                "echo \"resumed pid=$pid pgid=$pgid\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resume_path.chmod(0o755)

    return {
        "command": str(command_path),
        "command_resume": str(resume_command_path),
        "launcher": str(launcher_path),
        "stop": str(stop_path),
        "resume": str(resume_path),
    }


def write_metadata_and_status(
    args: argparse.Namespace,
    *,
    preflight: Mapping[str, object],
    scripts: Mapping[str, str],
    status: str,
    pid: Optional[int] = None,
    pgid: Optional[int] = None,
) -> None:
    """函数功能：写出 launcher 级 metadata/status，记录接手方式。"""
    output_dir = Path(args.output_dir)
    stop_command = f"bash {scripts['stop']}"
    resume_command = f"bash {scripts['resume']}"
    metadata = {
        "generated_at": display_time(),
        "launcher_version": "stage1_timefuse_fusor_full_scale_launcher_v1",
        "purpose": "Stage 1 96_48_S full-scale TimeFuse-style fusor baseline",
        "output_dir": str(output_dir),
        "train_script": str(TRAIN_SCRIPT),
        "device": str(args.device),
        "gpu_policy": "CUDA_VISIBLE_DEVICES=2,3 only when --device cuda; current launcher may use CPU to avoid visual router GPU contention",
        "cuda_visible_devices": str(args.cuda_visible_devices) if args.device == "cuda" else "",
        "cpu_reason": args.cpu_reason if args.device == "cpu" else "",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "feature_shard_count": int(preflight.get("feature_shard_count", 0)),
        "prediction_manifest_count": int(preflight.get("prediction_manifest_count", 0)),
        "expected_sample_count": EXPECTED_SAMPLE_COUNT,
        "expected_prediction_records": EXPECTED_PREDICTION_RECORDS,
        "preflight": preflight,
        "scripts": scripts,
        "pid": pid,
        "pgid": pgid,
        "monitor_commands": [
            f"ps -p {pid if pid else '$(cat ' + str(output_dir / 'pid.txt') + ')'} -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd",
            f"tail -n 120 {output_dir / 'main.log'}",
            f"cat {output_dir / 'status.json'}",
            f"find {output_dir / 'indexes'} -maxdepth 2 -name '*.sqlite' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\\n' 2>/dev/null | sort | tail -n 20",
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
            "df -h /data2 /home",
        ],
        "stop_command": stop_command,
        "resume_command": resume_command,
        "resume_policy": "若 latest_timefuse_fusor.pt 已存在，resume.sh 使用 --resume-checkpoint 跳过已完成 epoch 并继续 eval；否则重新从头构建 index 和训练。",
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(
        output_dir / "status.json",
        {
            "status": status,
            "phase": "launcher_started" if status == "running" else "launcher_created",
            "updated_at": display_time(),
            "output_dir": str(output_dir),
            "pid": pid,
            "pgid": pgid,
            "metadata_path": str(output_dir / "metadata.json"),
            "main_log": str(output_dir / "main.log"),
            "launcher_log": str(output_dir / "launcher.log"),
            "stop_command": stop_command,
            "resume_command": resume_command,
            "preflight_ok": bool(preflight.get("ok")),
            "preflight_warnings": preflight.get("warnings", []),
        },
    )


def start_launcher(args: argparse.Namespace, scripts: Mapping[str, str]) -> Dict[str, Optional[int]]:
    """函数功能：执行 launcher.sh 并读取 PID/PGID。"""
    result = subprocess.run(["bash", scripts["launcher"]], cwd=str(WORKSPACE), text=True, capture_output=True, check=True)
    append_log(Path(args.output_dir), result.stdout.strip())
    time.sleep(1)
    pid = int((Path(args.output_dir) / "pid.txt").read_text(encoding="utf-8").strip())
    pgid_text = (Path(args.output_dir) / "pgid.txt").read_text(encoding="utf-8").strip()
    pgid = int(pgid_text) if pgid_text else pid
    return {"pid": pid, "pgid": pgid}


def parse_args() -> argparse.Namespace:
    """函数功能：解析 full-scale fusor launcher 参数。"""
    parser = argparse.ArgumentParser(description="Launch Stage 1 full-scale TimeFuse-style fusor baseline.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="正式 full-scale fusor 输出目录。")
    parser.add_argument("--feature-shard-root", type=Path, default=FEATURE_SHARD_ROOT, help="64 个 feature shard 根目录。")
    parser.add_argument("--prediction-shard-root", type=Path, default=PREDICTION_SHARD_ROOT, help="五专家 prediction shard 根目录。")
    parser.add_argument("--labels-path", type=Path, default=ORACLE_LABELS_PATH, help="oracle labels parquet。")
    parser.add_argument("--merged-cache-status", type=Path, default=MERGED_CACHE_STATUS, help="merged cache status.json，用于 preflight。")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="正式任务设备；默认 CPU。")
    parser.add_argument("--cuda-visible-devices", default="2,3", help="GPU 模式可见设备，只允许 2/3 或单卡子集。")
    parser.add_argument("--cpu-reason", default="TimeFuse fusor 为 17 维线性权重模型，主要瓶颈在 shard-local SQLite/packed array I/O；同时 GPU2/3 已有 full-scale visual router eval-only，故正式 baseline 先用 CPU 避免资源争抢。")
    parser.add_argument("--epochs", type=int, default=1, help="full-scale fusor 训练 epoch。")
    parser.add_argument("--batch-size", type=int, default=256, help="streaming batch size。")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate。")
    parser.add_argument("--huber-beta", type=float, default=0.01, help="SmoothL1Loss beta。")
    parser.add_argument("--seed", type=int, default=16, help="随机种子。")
    parser.add_argument("--feature-read-chunk-rows", type=int, default=200000)
    parser.add_argument("--prediction-chunk-rows", type=int, default=200000)
    parser.add_argument("--oracle-parquet-batch-rows", type=int, default=200000)
    parser.add_argument("--prediction-num-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=1)
    parser.add_argument("--status-update-interval", type=int, default=200)
    parser.add_argument("--sample-prediction-limit", type=int, default=500)
    parser.add_argument("--omp-num-threads", type=int, default=4)
    parser.add_argument("--min-free-gb", type=int, default=300, help="输出挂载点最低剩余空间。")
    parser.add_argument("--train-only", action="store_true", help="只训练 checkpoint，不执行 test eval。默认执行 train+eval。")
    parser.add_argument("--preflight-only", action="store_true", help="只执行 preflight 和写脚本，不启动后台任务。")
    parser.add_argument("--allow-existing-output-dir", action="store_true", help="允许复用已有 launcher 输出目录。")
    parser.add_argument("--auto-start", action="store_true", help="preflight 通过后立即后台启动。")
    return parser.parse_args()


def main() -> None:
    """函数功能：执行 preflight、生成 launcher，并按需后台启动正式任务。"""
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("正式 full-scale fusor launcher 的 --epochs 至少为 1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    append_log(output_dir, "开始 full-scale TimeFuse-style fusor launcher preflight")
    preflight = build_preflight(args)
    write_json(output_dir / "preflight_report.json", preflight)
    scripts = write_shell_scripts(args)
    if not preflight["ok"]:
        write_metadata_and_status(args, preflight=preflight, scripts=scripts, status="preflight_failed")
        print(json.dumps(to_jsonable(preflight), indent=2, ensure_ascii=False), flush=True)
        sys.exit(2)
    write_metadata_and_status(args, preflight=preflight, scripts=scripts, status="launcher_created")
    append_log(output_dir, "preflight 通过，launcher 脚本已生成")
    if args.preflight_only or not args.auto_start:
        print(json.dumps(to_jsonable({"status": "launcher_created", "output_dir": output_dir, "scripts": scripts}), indent=2, ensure_ascii=False), flush=True)
        return
    pid_info = start_launcher(args, scripts)
    write_metadata_and_status(
        args,
        preflight=preflight,
        scripts=scripts,
        status="running",
        pid=pid_info["pid"],
        pgid=pid_info["pgid"],
    )
    append_log(output_dir, f"正式后台任务已启动 pid={pid_info['pid']} pgid={pid_info['pgid']}")
    print(json.dumps(to_jsonable({"status": "running", "output_dir": output_dir, **pid_info, "scripts": scripts}), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
