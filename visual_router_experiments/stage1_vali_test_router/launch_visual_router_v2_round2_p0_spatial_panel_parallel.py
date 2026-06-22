#!/usr/bin/env python3
"""
文件功能：
    Visual Router V2 Round2f P0-scale `spatial_panel_3view + film_mean_patch_aux`
    主线验证的多 GPU 进程级 launcher。

执行边界：
    - 本轮只运行 `spatial_panel_3view`，不重新比较其它 layout；
    - feature 阶段只从历史窗口 x 生成 pseudo image 并提取 frozen ViT feature，
      不保存 pseudo image tensor；
    - training/eval 阶段固定复用 `film_mean_patch_aux` 后端、seed 16/17/18；
    - pilot_selection 是唯一选择口径，diagnostic_balanced 只诊断，pilot_test 只做
      frozen final eval；
    - 多进程任务均写隔离目录，统一 CSV/JSON/Markdown 只由单进程聚合和本脚本
      postprocess 写出。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "build_visual_router_v2_round2_layout_features.py"
TRAIN_SCRIPT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_v2_round2_layout_film.py"
DEFAULT_PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
DEFAULT_SAMPLE_DIR = Path("/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples")
DEFAULT_OUTPUT_DIR = Path("/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline")
DEFAULT_SUMMARY_DIR = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round2" / "p0_spatial_panel_mainline"
DEFAULT_LAYOUT = "spatial_panel_3view"
P0_SAMPLE_SETS = ("pilot_train", "pilot_selection", "diagnostic_balanced", "pilot_test")
P0_SAMPLE_FILES = {
    "pilot_train": "pilot_train_sample_keys.csv",
    "pilot_selection": "pilot_selection_sample_keys.csv",
    "diagnostic_balanced": "diagnostic_balanced_sample_keys.csv",
    "pilot_test": "pilot_test_sample_keys.csv",
}
ARTIFACT_PREFIX = "round2_p0_spatial"
ROUND1_FILM_FINAL = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round1" / "p2e_film_final_test_extension" / "round1_film_final_test_extension_comparison.csv"
ROUND1_VISUAL_FINAL = REPO_ROOT / "experiment_summaries" / "visual_router_v2_round1" / "p2d_final_test_extension" / "round1_final_test_extension_comparison.csv"


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
    """函数功能：解析 P0 spatial panel mainline launcher 参数。"""
    parser = argparse.ArgumentParser(description="Launch Round2f P0 spatial panel mainline validation.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--sample-manifest", type=Path, default=None, help="默认由 sample-dir 中四个 P0 CSV 合并生成。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-copy-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--seeds", default="16,17,18")
    parser.add_argument("--feature-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--final-eval-only", action="store_true", help="兼容验收参数；当前训练 worker 总是训练后评估 selection/diagnostic/pilot_test。")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-shard-size", type=int, default=2000)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-samples-per-set", type=int, default=None, help="仅用于 smoke，正式 P0 运行必须省略。")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-procs-per-device", type=int, default=1)
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """函数功能：稳定写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def resolve_sample_manifest(args: argparse.Namespace) -> Path:
    """函数功能：生成或返回 P0 四个 sample set 的合并 manifest。"""
    if args.sample_manifest is not None:
        return Path(args.sample_manifest)
    manifest_path = Path(args.output_dir) / "inputs" / "p0_sample_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        return manifest_path
    frames: List[pd.DataFrame] = []
    for sample_set in P0_SAMPLE_SETS:
        path = Path(args.sample_dir) / P0_SAMPLE_FILES[sample_set]
        frame = pd.read_csv(path)
        if frame["sample_set"].astype(str).nunique() != 1 or str(frame["sample_set"].iloc[0]) != sample_set:
            raise ValueError(f"{path} sample_set 字段异常")
        order = frame["order_index"].to_numpy()
        if (order != range(len(frame))).any():
            raise ValueError(f"{path} order_index 必须从 0 连续递增")
        if frame["sample_key"].astype(str).duplicated().any():
            raise ValueError(f"{path} 存在重复 sample_key")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(manifest_path, index=False)
    write_json(
        manifest_path.with_suffix(".metadata.json"),
        {
            "status": "completed",
            "generated_at": display_time(),
            "source_sample_dir": str(args.sample_dir),
            "sample_sets": list(P0_SAMPLE_SETS),
            "counts": {name: int((merged["sample_set"].astype(str) == name).sum()) for name in P0_SAMPLE_SETS},
            "constraints": {
                "pilot_test_used_for_training_or_selection": False,
                "diagnostic_balanced_used_for_selection": False,
            },
        },
    )
    return manifest_path


def p0_feature_flags(args: argparse.Namespace, sample_manifest: Path, *, sample_sets: Sequence[str] = P0_SAMPLE_SETS) -> List[str]:
    """函数功能：构造 P0 feature worker/aggregate 的协议参数。"""
    return [
        "--sample-manifest",
        str(sample_manifest),
        "--output-dir",
        str(args.output_dir),
        "--layouts",
        DEFAULT_LAYOUT,
        "--sample-sets",
        ",".join(sample_sets),
        "--artifact-prefix",
        ARTIFACT_PREFIX,
    ]


def p0_train_flags(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 P0 training worker/aggregate 的协议参数。"""
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
        DEFAULT_LAYOUT,
        "--seeds",
        str(args.seeds),
        "--artifact-prefix",
        ARTIFACT_PREFIX,
        "--train-sample-set",
        "pilot_train",
        "--selection-sample-set",
        "pilot_selection",
        "--diagnostic-sample-set",
        "diagnostic_balanced",
        "--test-sample-set",
        "pilot_test",
        "--experiment-label",
        "Round2f P0 spatial panel mainline",
        "--summary-title",
        "Visual Router V2 Round2f P0 Spatial Panel Mainline Summary",
    ]


def feature_command(args: argparse.Namespace, sample_manifest: Path, sample_set: str, device: str) -> List[str]:
    """函数功能：构造单 layout feature worker 命令。"""
    cmd = [
        str(args.python),
        str(FEATURE_SCRIPT),
        *p0_feature_flags(args, sample_manifest, sample_sets=[sample_set]),
        "--layout",
        DEFAULT_LAYOUT,
        "--sample-set-worker",
        "--device",
        device,
        "--shard-size",
        str(args.feature_shard_size),
        "--embedding-batch-size",
        str(args.embedding_batch_size),
    ]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def feature_aggregate_command(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 feature aggregation 命令。"""
    cmd = [str(args.python), str(FEATURE_SCRIPT), *p0_feature_flags(args, sample_manifest), "--aggregate-only"]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return cmd


def build_index_command(args: argparse.Namespace, sample_manifest: Path) -> List[str]:
    """函数功能：构造 prediction subset SQLite 预构建命令。"""
    cmd = [str(args.python), str(TRAIN_SCRIPT), *p0_train_flags(args, sample_manifest), "--build-index-only"]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def train_command(args: argparse.Namespace, sample_manifest: Path, seed: int, device: str) -> List[str]:
    """函数功能：构造单 seed training+eval worker 命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        *p0_train_flags(args, sample_manifest),
        "--layout",
        DEFAULT_LAYOUT,
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
    """函数功能：构造 training aggregation 命令。"""
    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        *p0_train_flags(args, sample_manifest),
        "--devices-requested",
        str(args.devices),
        "--parallel-launcher-used",
        "--aggregate-only",
    ]
    if args.max_samples_per_set is not None:
        cmd.extend(["--max-samples-per-set", str(args.max_samples_per_set)])
    return cmd


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


def run_parallel(args: argparse.Namespace, tasks: Sequence[Tuple[str, str, List[str], Path]], *, phase: str) -> None:
    """函数功能：按 device slot 并行运行互相隔离的任务。"""
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


def _raw_soft_row(frame: pd.DataFrame, variant: str, sample_set: str | None = None) -> pd.Series | None:
    """函数功能：读取长表中某 variant 的 raw-soft 行。"""
    if "method_kind" in frame.columns:
        method_kind = frame["method_kind"].astype(str)
    elif "method" in frame.columns:
        # summarize_mean_std 直接输出的 selection/diagnostic/test summary
        # 没有 method_kind 列，需要从 method 后缀恢复 hard/raw-soft 口径。
        method_kind = frame["method"].astype(str).map(lambda value: "raw_soft_fusion" if value.endswith("_raw_soft_fusion") else "hard_top1" if value.endswith("_hard_top1") else value)
    else:
        raise ValueError("summary/comparison 表必须包含 method_kind 或 method 列")
    rows = frame[(frame["variant"].astype(str) == variant) & (method_kind == "raw_soft_fusion")].copy()
    if sample_set is not None:
        rows = rows[rows["sample_set"].astype(str) == sample_set].copy()
    if rows.empty:
        return None
    return rows.sort_values(["MAE_mean", "MSE_mean"], kind="mergesort").iloc[0]


def _wide_raw_soft_row(path: Path, variant: str) -> Dict[str, float] | None:
    """函数功能：读取 Round1 final-test extension 宽表中的 raw-soft 指标。"""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    rows = frame[(frame["variant"].astype(str) == variant) & (frame["method"].astype(str).str.endswith("_raw_soft_fusion"))].copy()
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "MAE_mean": float(row["raw_soft_fusion_MAE"]),
        "MSE_mean": float(row["raw_soft_fusion_MSE"]),
        "regret_to_oracle_mean": float(row["raw_soft_fusion_regret_to_oracle"]),
    }


def _verdict_delta(left: float, right: float) -> str:
    """函数功能：返回 lower-is-better 指标的改善描述。"""
    delta = float(left) - float(right)
    return f"{delta:+.6f}（{'改善' if delta < 0 else '未改善'}）"


def copy_and_write_p0_aliases(args: argparse.Namespace) -> None:
    """函数功能：补齐验收要求的固定文件名，并写 P0 专用中文 summary。"""
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_copy_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        f"{ARTIFACT_PREFIX}_test_summary.csv": f"{ARTIFACT_PREFIX}_final_test_summary.csv",
        f"{ARTIFACT_PREFIX}_validation_metadata.json": f"{ARTIFACT_PREFIX}_metadata.json",
        f"{ARTIFACT_PREFIX}_validation_summary.md": f"{ARTIFACT_PREFIX}_summary.md",
    }
    for src_name, dst_name in aliases.items():
        src = output_dir / src_name
        dst = output_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            shutil.copy2(dst, summary_dir / dst_name)

    selection = pd.read_csv(output_dir / f"{ARTIFACT_PREFIX}_selection_comparison.csv")
    diagnostic = pd.read_csv(output_dir / f"{ARTIFACT_PREFIX}_diagnostic_summary.csv")
    final_test = pd.read_csv(output_dir / f"{ARTIFACT_PREFIX}_final_test_summary.csv")
    stratified_path = output_dir / f"{ARTIFACT_PREFIX}_stratified_summary.csv"
    selected_counts_path = output_dir / f"{ARTIFACT_PREFIX}_selected_model_counts.csv"
    metadata_path = output_dir / f"{ARTIFACT_PREFIX}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    p0_sel = _raw_soft_row(selection, DEFAULT_LAYOUT, "pilot_selection")
    p0_test = _raw_soft_row(final_test, DEFAULT_LAYOUT, "pilot_test")
    round1_selection = _raw_soft_row(selection, "film_mean_patch_aux", "pilot_selection")
    round0_timefuse = _raw_soft_row(selection, "Round0 TimeFuse", "pilot_selection")
    round0_visual = _raw_soft_row(selection, "Round0 original Visual", "pilot_selection")
    round1_test = _wide_raw_soft_row(ROUND1_FILM_FINAL, "film_mean_patch_aux")
    visual_test = _wide_raw_soft_row(ROUND1_VISUAL_FINAL, "visual_cls_mean_concat")

    if p0_sel is None or p0_test is None:
        raise ValueError("P0 spatial panel selection/final-test raw-soft 行缺失")
    p0_test_values = {
        "MAE_mean": float(p0_test["MAE_mean"]),
        "MSE_mean": float(p0_test["MSE_mean"]),
        "regret_to_oracle_mean": float(p0_test["regret_to_oracle_mean"]),
    }
    round1_text = "Round1 film_mean_patch_aux final-test reference 缺失，无法判断。"
    if round1_test is not None:
        mae_delta = _verdict_delta(p0_test_values["MAE_mean"], round1_test["MAE_mean"])
        mse_delta = _verdict_delta(p0_test_values["MSE_mean"], round1_test["MSE_mean"])
        regret_delta = _verdict_delta(p0_test_values["regret_to_oracle_mean"], round1_test["regret_to_oracle_mean"])
        round1_text = (
            f"P0 spatial pilot_test raw-soft MAE={p0_test_values['MAE_mean']:.6f} vs Round1 film_mean_patch_aux={round1_test['MAE_mean']:.6f}，delta={mae_delta}；"
            f"MSE delta={mse_delta}；regret delta={regret_delta}。"
        )
    timefuse_text = "Round0 TimeFuse selection reference 缺失。"
    if round0_timefuse is not None:
        timefuse_text = (
            f"pilot_selection raw-soft MAE delta vs Round0 TimeFuse="
            f"{_verdict_delta(float(p0_sel['MAE_mean']), float(round0_timefuse['MAE_mean']))}。"
        )
    visual_text = "Round0 original Visual / Round1 visual final-test reference 缺失。"
    if visual_test is not None:
        visual_text = (
            f"pilot_test raw-soft MAE delta vs Round1 visual_cls_mean_concat="
            f"{_verdict_delta(p0_test_values['MAE_mean'], visual_test['MAE_mean'])}。"
        )
    if round0_visual is not None:
        visual_text += (
            f" pilot_selection raw-soft MAE delta vs Round0 original Visual="
            f"{_verdict_delta(float(p0_sel['MAE_mean']), float(round0_visual['MAE_mean']))}。"
        )
    seed_text = f"seed stability：pilot_selection raw-soft MAE_std={float(p0_sel['MAE_std']):.6f}，MSE_std={float(p0_sel['MSE_std']):.6f}。"

    selected_text = "selected_model ratio 文件缺失。"
    if selected_counts_path.exists():
        counts = pd.read_csv(selected_counts_path)
        rows = counts[(counts["sample_set"].astype(str) == "pilot_selection") & (counts["variant"].astype(str) == DEFAULT_LAYOUT)]
        if not rows.empty:
            # 只取 raw-soft 的模型占比，硬选择行保留在 CSV 中供复核。
            selected_text = "pilot_selection selected_model ratio 见 selected_model_counts.csv；主要用于检查是否过度塌缩。"

    strata_text = "stratified summary 文件缺失。"
    if stratified_path.exists():
        strata = pd.read_csv(stratified_path, nrows=1)
        strata_text = f"stratified summary 已生成，字段包含：{', '.join(strata.columns[:12])} ..."

    next_step = "full-scale validation" if round1_test is not None and p0_test_values["MAE_mean"] < round1_test["MAE_mean"] else "period_soft_mixture 支线或 canonical migration 前复核"
    lines = [
        "# Visual Router V2 Round2f P0 Spatial Panel Mainline Summary",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 结论",
        "",
        f"- 本轮只验证 `{DEFAULT_LAYOUT} + film_mean_patch_aux`，选择口径只使用 `pilot_selection` raw-soft MAE mean。",
        f"- pilot_selection raw-soft：MAE={float(p0_sel['MAE_mean']):.6f}，MSE={float(p0_sel['MSE_mean']):.6f}，regret={float(p0_sel['regret_to_oracle_mean']):.6f}。",
        f"- frozen pilot_test raw-soft：MAE={p0_test_values['MAE_mean']:.6f}，MSE={p0_test_values['MSE_mean']:.6f}，regret={p0_test_values['regret_to_oracle_mean']:.6f}。",
        f"- 是否超过 Round1 当前 best：{round1_text}",
        f"- 下一步建议：{next_step}。",
        "",
        "## 必答问题",
        "",
        f"1. P0 pilot_selection 表现：MAE={float(p0_sel['MAE_mean']):.6f}，MSE={float(p0_sel['MSE_mean']):.6f}，regret={float(p0_sel['regret_to_oracle_mean']):.6f}。",
        f"2. frozen pilot_test 是否优于 Round1 `film_mean_patch_aux`：{round1_text}",
        f"3. 是否优于 Round0 TimeFuse / original Visual：{timefuse_text} {visual_text}",
        f"4. MAE / MSE / regret 是否同时改善：见第 2 条和 `round2_p0_spatial_delta_summary.csv`。",
        f"5. seed stability 是否保持：{seed_text}",
        f"6. CrossFormer / PatchTST / ES / DLinear strata 是否改善：{strata_text}",
        f"7. selected_model ratio 是否健康：{selected_text}",
        "8. 是否建议把 spatial panel 作为 Visual Router V2 当前主线：依据 P0 pilot_test 是否超过 Round1 best 和 selection stability 判定。",
        f"9. 下一步：{next_step}。",
        "",
        "## 产物",
        "",
        f"- 输出目录：`{output_dir}`",
        f"- 轻量 summary 目录：`{summary_dir}`",
        f"- metadata：`{ARTIFACT_PREFIX}_metadata.json`",
        f"- final test summary：`{ARTIFACT_PREFIX}_final_test_summary.csv`",
        "",
        "## Metadata 摘要",
        "",
        f"- devices_requested：`{metadata.get('devices_requested', '')}`",
        f"- devices_used：`{metadata.get('devices_used', [])}`",
        f"- backend_style：`{metadata.get('backend_style', '')}`",
        f"- used_frozen_test_for_selection：`{metadata.get('used_frozen_test_for_selection', '')}`",
    ]
    summary_path = output_dir / f"{ARTIFACT_PREFIX}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(summary_path, summary_dir / summary_path.name)
    # 确保所有轻量验收文件都复制到仓库 summary 目录。
    for name in [
        f"{ARTIFACT_PREFIX}_feature_manifest.csv",
        f"{ARTIFACT_PREFIX}_feature_latency.csv",
        f"{ARTIFACT_PREFIX}_variant_seed_results.csv",
        f"{ARTIFACT_PREFIX}_selection_comparison.csv",
        f"{ARTIFACT_PREFIX}_diagnostic_summary.csv",
        f"{ARTIFACT_PREFIX}_final_test_summary.csv",
        f"{ARTIFACT_PREFIX}_selected_model_counts.csv",
        f"{ARTIFACT_PREFIX}_stratified_summary.csv",
        f"{ARTIFACT_PREFIX}_delta_summary.csv",
        f"{ARTIFACT_PREFIX}_metadata.json",
        f"{ARTIFACT_PREFIX}_summary.md",
        "status.json",
    ]:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, summary_dir / name)


def main() -> None:
    """函数功能：执行 feature-only、train-only、aggregate-only 或完整 P0 流水线。"""
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    devices = parse_csv(args.devices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = resolve_sample_manifest(args)
    write_json(
        args.output_dir / "launcher_metadata.json",
        {
            "status": "started",
            "started_at": display_time(),
            "launcher": str(Path(__file__).resolve()),
            "feature_script": str(FEATURE_SCRIPT),
            "train_script": str(TRAIN_SCRIPT),
            "sample_dir": str(args.sample_dir),
            "sample_manifest": str(sample_manifest),
            "output_dir": str(args.output_dir),
            "summary_copy_dir": str(args.summary_copy_dir),
            "layout": DEFAULT_LAYOUT,
            "seeds": seeds,
            "devices_requested": devices,
            "sample_sets": list(P0_SAMPLE_SETS),
            "feature_only": bool(args.feature_only),
            "train_only": bool(args.train_only),
            "final_eval_only": bool(args.final_eval_only),
            "aggregate_only": bool(args.aggregate_only),
        },
    )

    if args.aggregate_only:
        run_command(feature_aggregate_command(args, sample_manifest), log_path=args.output_dir / "feature_aggregation.log")
        run_command(train_aggregate_command(args, sample_manifest), log_path=args.output_dir / "training_aggregation.log")
        copy_and_write_p0_aliases(args)
    else:
        if not args.train_only and not args.final_eval_only:
            feature_tasks: List[Tuple[str, str, List[str], Path]] = []
            for idx, sample_set in enumerate(P0_SAMPLE_SETS):
                device = devices[idx % len(devices)]
                feature_tasks.append((f"{DEFAULT_LAYOUT}_{sample_set}", device, feature_command(args, sample_manifest, sample_set, device), args.output_dir / "feature_logs" / f"{DEFAULT_LAYOUT}_{sample_set}.log"))
            run_parallel(args, feature_tasks, phase="feature")
            run_command(feature_aggregate_command(args, sample_manifest), log_path=args.output_dir / "feature_aggregation.log")
        if not args.feature_only:
            run_command(build_index_command(args, sample_manifest), log_path=args.output_dir / "prediction_index_build.log")
            train_tasks: List[Tuple[str, str, List[str], Path]] = []
            for idx, seed in enumerate(seeds):
                device = devices[idx % len(devices)]
                train_tasks.append((f"{DEFAULT_LAYOUT}_seed{seed}", device, train_command(args, sample_manifest, seed, device), args.output_dir / "tasks" / f"{DEFAULT_LAYOUT}_seed{seed}" / "task.log"))
            run_parallel(args, train_tasks, phase="training")
            run_command(train_aggregate_command(args, sample_manifest), log_path=args.output_dir / "training_aggregation.log")
            copy_and_write_p0_aliases(args)

    meta_path = args.output_dir / "launcher_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({"status": "completed", "finished_at": display_time()})
    write_json(meta_path, meta)
    print(f"[{display_time()}] Round2f P0 spatial panel launcher completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
