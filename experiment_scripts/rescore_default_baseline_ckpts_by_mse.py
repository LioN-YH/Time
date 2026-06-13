#!/usr/bin/env python3
"""
文件功能：
    从既有 default_baseline 训练输出中恢复每轮 validation MSE/MAE，
    找出按 validation MSE 选择的 best checkpoint，并可选地补跑 test evaluate。

设计说明：
    训练时 TensorBoard event 已记录 valid/MSE_epoch 和 valid/MAE_epoch，因此本脚本
    优先读取 event，避免对每个 checkpoint 重新扫描 validation split。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from run_default_baseline_finetune_eval import (
    BaselineJob,
    ENV_BIN,
    REPO_DIR,
    RUN_OUTPUT_ROOT,
    apply_evaluate_overrides,
    build_env,
    build_evaluate_command,
    find_eval_results,
    load_yaml,
    summarize_eval_results,
    write_json,
    write_yaml,
)


WORKSPACE = Path("/home/shiyuhong/Time")
DEFAULT_BASELINE_ROOT = REPO_DIR / "outputs" / "default_baseline"
MSE_BEST_OUTPUT_ROOT = "outputs/default_baseline_mse_best"


@dataclass
class RescoreRow:
    """函数功能：保存单个 checkpoint 复盘结果，便于 JSON/CSV 双格式输出。"""

    model: str
    config_name: str
    seed: int
    status: str
    epoch: Optional[int]
    checkpoint_path: Optional[str]
    validation_mse: Optional[float]
    validation_mae: Optional[float]
    is_mse_best: bool = False
    is_mae_best_checkpoint: bool = False
    mse_best_checkpoint_path: str = ""
    mae_best_checkpoint_path: str = ""
    mse_best_epoch: Optional[int] = None
    mae_best_epoch: Optional[int] = None
    mse_best_differs_from_mae_best: bool = False
    test_eval_results_path: Optional[str] = None
    test_mse: Optional[float] = None
    test_mae: Optional[float] = None
    message: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """函数功能：解析 MSE-best 复盘参数。"""
    parser = argparse.ArgumentParser(description="Rescore existing default baseline checkpoints by validation MSE.")
    parser.add_argument("--seed", type=int, default=16, help="要复盘的 seed。")
    parser.add_argument("--only", type=str, default="", help="只复盘指定任务，格式如 patchtst:576_288_S。")
    parser.add_argument("--evaluate-different", action="store_true", help="仅对 MSE-best 与 MAE-best 不同的任务补跑 test evaluate。")
    parser.add_argument("--num-processes", type=int, default=1, help="补 evaluate 使用的 Ray evaluator 数。")
    parser.add_argument("--gpu-ids", type=str, default="0", help="补 evaluate 暴露的 GPU id。")
    parser.add_argument("--use-gpu", type=int, default=1, choices=[0, 1], help="补 evaluate 是否使用 GPU。")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="补 evaluate batch size。")
    parser.add_argument("--num-workers", type=int, default=6, help="补 evaluate DataLoader worker 数。")
    parser.add_argument("--dry-run", action="store_true", help="只生成复盘表和 evaluate 配置，不执行补 evaluate。")
    return parser.parse_args()


def parse_scalar_events(finetune_dir: Path) -> Dict[str, Dict[int, float]]:
    """函数功能：从 TensorBoard event 读取 epoch 粒度验证指标。"""
    event_files = list(finetune_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"没有找到 TensorBoard event：{finetune_dir}")

    accumulator = EventAccumulator(str(finetune_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    required_tags = ["valid/MSE_epoch", "valid/MAE_epoch"]
    missing_tags = [tag for tag in required_tags if tag not in scalar_tags]
    if missing_tags:
        raise ValueError(f"{finetune_dir} 缺少验证指标：{missing_tags}")

    return {
        "mse": {event.step: float(event.value) for event in accumulator.Scalars("valid/MSE_epoch")},
        "mae": {event.step: float(event.value) for event in accumulator.Scalars("valid/MAE_epoch")},
    }


def load_running_output_dirs() -> set[str]:
    """
    函数功能：
        从当前 finetune 进程的 --config_path 反查 logging.output_dir，用于标注 running。

    设计说明：
        baseline 并发 lane 使用独立临时 YAML；进程命令行里没有最终输出目录，所以这里读取
        仍在运行的 YAML，避免把正在写 event/checkpoint 的任务误判成坏目录。
    """
    running_output_dirs: set[str] = set()
    proc_root = Path("/proc")
    for cmdline_path in proc_root.glob("[0-9]*/cmdline"):
        try:
            parts = cmdline_path.read_bytes().decode("utf-8", errors="ignore").split("\0")
        except OSError:
            continue
        if not any(part.endswith("finetune.py") for part in parts):
            continue
        try:
            config_path = Path(parts[parts.index("--config_path") + 1])
        except (ValueError, IndexError):
            continue
        if not config_path.exists():
            continue
        try:
            config = load_yaml(config_path)
        except Exception:
            continue
        output_dir = config.get("logging", {}).get("output_dir")
        if output_dir:
            running_output_dirs.add(str(output_dir).rstrip("/"))
    return running_output_dirs


def classify_skip_status(finetune_dir: Path, running_output_dirs: set[str], message: str) -> str:
    """函数功能：根据 output_dir 和异常类型，把不可复盘目录归类为 running/no_events 等状态。"""
    output_dir = finetune_output_dir_key(finetune_dir)
    if output_dir in running_output_dirs:
        return "running"
    if "TensorBoard event" in message:
        return "no_events"
    if "缺少验证指标" in message:
        return "no_validation_yet"
    if "checkpoint" in message:
        return "no_checkpoints"
    if "没有可匹配" in message:
        return "no_matching_epochs"
    return "skipped"


def finetune_output_dir_key(finetune_dir: Path) -> str:
    """函数功能：把 FINE_TUNE/ver_0 路径还原成配置中的 logging.output_dir。"""
    rel_parts = finetune_dir.relative_to(DEFAULT_BASELINE_ROOT).parts
    model, config_name, seed_part = rel_parts[0], rel_parts[1], rel_parts[2]
    return f"outputs/default_baseline/{model}/{config_name}/{seed_part}"


def append_skip_row(
    rows: List[RescoreRow],
    *,
    model: str,
    config_name: str,
    seed: int,
    status: str,
    message: str,
) -> None:
    """函数功能：把不可复盘任务也写进输出表，避免 silent skip。"""
    rows.append(
        RescoreRow(
            model=model,
            config_name=config_name,
            seed=seed,
            status=status,
            epoch=None,
            checkpoint_path=None,
            validation_mse=None,
            validation_mae=None,
            mae_best_epoch=None,
            message=message,
        )
    )


def parse_epoch_from_name(path: Path, prefix: str) -> Optional[int]:
    """函数功能：从 Quito checkpoint 文件名中解析 epoch。"""
    pattern = rf"{prefix}_epoch=(\d+)_step=\d+_.*\.ckpt$"
    match = re.match(pattern, path.name)
    if not match:
        return None
    return int(match.group(1))


def find_epoch_checkpoints(finetune_dir: Path) -> Dict[int, Path]:
    """函数功能：建立 epoch 到 ckpt_epoch checkpoint 的映射。"""
    checkpoint_dir = finetune_dir / "checkpoints"
    mapping: Dict[int, Path] = {}
    for checkpoint_path in checkpoint_dir.glob("ckpt_epoch=*_step=*.ckpt"):
        epoch = parse_epoch_from_name(checkpoint_path, "ckpt")
        if epoch is not None:
            mapping[epoch] = checkpoint_path.resolve()
    if not mapping:
        raise FileNotFoundError(f"没有找到 ckpt_epoch checkpoint：{checkpoint_dir}")
    return mapping


def find_mae_best_checkpoint(finetune_dir: Path) -> Optional[Path]:
    """函数功能：找到当前训练产物中的 MAE-best checkpoint。"""
    candidates = sorted((finetune_dir / "checkpoints").glob("best_epoch=*_step=*.ckpt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1].resolve() if candidates else None


def parse_eval_means(eval_results_path: Path) -> Dict[str, float]:
    """函数功能：读取 Quito test evaluation JSON 并计算整体 MSE/MAE 均值。"""
    with eval_results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("final_results", [])
    output: Dict[str, float] = {}
    for metric in ["MSE", "MAE"]:
        values = [float(row["metrics"][metric]) for row in rows if metric in row.get("metrics", {})]
        if values:
            output[metric] = sum(values) / len(values)
    return output


def iter_target_dirs(seed: int, only: str) -> List[Path]:
    """函数功能：枚举需要复盘的 FINE_TUNE/ver_0 目录。"""
    dirs = sorted(DEFAULT_BASELINE_ROOT.glob(f"*/*/seed_{seed}/FINE_TUNE/ver_0"))
    if not only:
        return dirs

    wanted = {item.strip() for item in only.split(",") if item.strip()}
    selected = []
    for finetune_dir in dirs:
        parts = finetune_dir.relative_to(DEFAULT_BASELINE_ROOT).parts
        model, config_name = parts[0], parts[1]
        if f"{model}:{config_name}" in wanted:
            selected.append(finetune_dir)
    missing = wanted - {f"{path.relative_to(DEFAULT_BASELINE_ROOT).parts[0]}:{path.relative_to(DEFAULT_BASELINE_ROOT).parts[1]}" for path in selected}
    if missing:
        raise ValueError(f"--only 包含未找到训练输出的任务：{sorted(missing)}")
    return selected


def prepare_mse_best_evaluate_config(
    *,
    run_dir: Path,
    model: str,
    config_name: str,
    seed: int,
    checkpoint_path: str,
    args: argparse.Namespace,
) -> Path:
    """函数功能：为 MSE-best checkpoint 生成独立 evaluate 配置。"""
    source_config_path = REPO_DIR / "configs" / "evaluate" / model / f"{config_name}.yaml"
    config = load_yaml(source_config_path)
    output_dir = f"{MSE_BEST_OUTPUT_ROOT}/{model}/{config_name}/seed_{seed}"
    config = apply_evaluate_overrides(
        config,
        output_dir=output_dir,
        seed=seed,
        checkpoint_path=checkpoint_path,
        batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    target_path = run_dir / "generated_configs" / model / config_name / f"mse_best_evaluate_seed_{seed}.yaml"
    write_yaml(target_path, config)
    return target_path


def run_evaluate(command: List[str], *, env: dict, log_path: Path, dry_run: bool) -> int:
    """函数功能：运行补 evaluate 命令并保存独立日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# command: {' '.join(command)}\n")
        log_file.write(f"# dry_run: {dry_run}\n\n")
        log_file.flush()
        if dry_run:
            log_file.write("# return_code: 0\n")
            return 0

        process = subprocess.Popen(
            command,
            cwd=str(REPO_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
        log_file.write(f"\n# return_code: {return_code}\n")
        return return_code


def main() -> int:
    """函数功能：执行 MSE-best 复盘，并按需补 evaluate。"""
    args = parse_args()
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = RUN_OUTPUT_ROOT / f"{run_stamp}_default_baseline_mse_best_rescore"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: List[RescoreRow] = []
    env = build_env(args.gpu_ids)
    running_output_dirs = load_running_output_dirs()

    for finetune_dir in iter_target_dirs(args.seed, args.only):
        rel_parts = finetune_dir.relative_to(DEFAULT_BASELINE_ROOT).parts
        model, config_name = rel_parts[0], rel_parts[1]
        task_status = "running" if finetune_output_dir_key(finetune_dir) in running_output_dirs else "completed"
        try:
            scalars = parse_scalar_events(finetune_dir)
            epoch_checkpoints = find_epoch_checkpoints(finetune_dir)
            mae_best_checkpoint = find_mae_best_checkpoint(finetune_dir)
            mae_best_epoch = parse_epoch_from_name(mae_best_checkpoint, "best") if mae_best_checkpoint else None

            available_epochs = sorted(set(scalars["mse"]) & set(epoch_checkpoints))
            if not available_epochs:
                raise ValueError(f"{finetune_dir} 没有可匹配的 validation MSE 与 checkpoint epoch。")
        except (FileNotFoundError, ValueError) as exc:
            message = str(exc)
            status = classify_skip_status(finetune_dir, running_output_dirs, message)
            append_skip_row(
                rows,
                model=model,
                config_name=config_name,
                seed=args.seed,
                status=status,
                message=message,
            )
            print(f"[{status}] 跳过 {model}:{config_name}：{message}", flush=True)
            continue

        mse_best_epoch = min(available_epochs, key=lambda epoch: scalars["mse"][epoch])
        mse_best_checkpoint = epoch_checkpoints[mse_best_epoch]
        # 同一轮会同时保存 ckpt_epoch 与 best_epoch 两个文件；判断 protocol 差异时按 epoch 对齐。
        differs = mae_best_epoch is None or mse_best_epoch != mae_best_epoch

        test_results_path: Optional[Path] = None
        test_means: Dict[str, float] = {}
        if args.evaluate_different and differs and task_status == "completed":
            evaluate_config_path = prepare_mse_best_evaluate_config(
                run_dir=run_dir,
                model=model,
                config_name=config_name,
                seed=args.seed,
                checkpoint_path=str(mse_best_checkpoint),
                args=args,
            )
            command = build_evaluate_command(evaluate_config_path, args)
            log_path = run_dir / "logs" / f"{model}_{config_name}_seed{args.seed}_mse_best_evaluate.log"
            return_code = run_evaluate(command, env=env, log_path=log_path, dry_run=args.dry_run)
            if return_code != 0:
                raise RuntimeError(f"{model}:{config_name} MSE-best evaluate 失败，return_code={return_code}")

            if not args.dry_run:
                output_base = REPO_DIR / MSE_BEST_OUTPUT_ROOT / model / config_name / f"seed_{args.seed}" / "EVALUATE"
                evaluate_output_dir = max([path for path in output_base.glob("ver_*") if path.is_dir()], key=lambda path: path.stat().st_mtime)
                test_results_path = find_eval_results(evaluate_output_dir)
                test_means = parse_eval_means(test_results_path)

                job = BaselineJob(model=model, config_name=config_name, seed=args.seed)
                job.best_checkpoint_path = str(mse_best_checkpoint)
                job.eval_results_path = str(test_results_path)
                summarize_eval_results(job, run_dir / "cluster_analysis")

        for epoch in available_epochs:
            rows.append(
                RescoreRow(
                    model=model,
                    config_name=config_name,
                    seed=args.seed,
                    status=task_status,
                    epoch=epoch,
                    checkpoint_path=str(epoch_checkpoints[epoch]),
                    validation_mse=scalars["mse"][epoch],
                    validation_mae=scalars["mae"].get(epoch),
                    is_mse_best=(epoch == mse_best_epoch),
                    is_mae_best_checkpoint=(epoch == mae_best_epoch),
                    mse_best_checkpoint_path=str(mse_best_checkpoint),
                    mae_best_checkpoint_path=str(mae_best_checkpoint) if mae_best_checkpoint else "",
                    mse_best_epoch=mse_best_epoch,
                    mae_best_epoch=mae_best_epoch,
                    mse_best_differs_from_mae_best=differs,
                    test_eval_results_path=str(test_results_path) if test_results_path else None,
                    test_mse=test_means.get("MSE"),
                    test_mae=test_means.get("MAE"),
                    message=None,
                )
            )

    payload = {
        "run_dir": str(run_dir),
        "seed": args.seed,
        "only": args.only,
        "evaluate_different": args.evaluate_different,
        "rows": [asdict(row) for row in rows],
    }
    write_json(run_dir / "mse_best_rescore.json", payload)
    pd.DataFrame([asdict(row) for row in rows]).to_csv(run_dir / "mse_best_rescore.csv", index=False)

    compact_rows = []
    for row in rows:
        if row.is_mse_best:
            compact_rows.append(
                {
                    "model": row.model,
                    "config_name": row.config_name,
                    "seed": row.seed,
                    "status": row.status,
                    "mse_best_epoch": row.mse_best_epoch,
                    "mae_best_epoch": row.mae_best_epoch,
                    "validation_mse": row.validation_mse,
                    "validation_mae": row.validation_mae,
                    "mse_best_differs_from_mae_best": row.mse_best_differs_from_mae_best,
                    "mse_best_checkpoint_path": row.mse_best_checkpoint_path,
                    "mae_best_checkpoint_path": row.mae_best_checkpoint_path,
                    "test_eval_results_path": row.test_eval_results_path,
                    "test_mse": row.test_mse,
                    "test_mae": row.test_mae,
                    "message": row.message,
                }
            )
        elif row.epoch is None:
            compact_rows.append(
                {
                    "model": row.model,
                    "config_name": row.config_name,
                    "seed": row.seed,
                    "status": row.status,
                    "mse_best_epoch": None,
                    "mae_best_epoch": None,
                    "validation_mse": None,
                    "validation_mae": None,
                    "mse_best_differs_from_mae_best": False,
                    "mse_best_checkpoint_path": "",
                    "mae_best_checkpoint_path": "",
                    "test_eval_results_path": None,
                    "test_mse": None,
                    "test_mae": None,
                    "message": row.message,
                }
            )
    pd.DataFrame(compact_rows).to_csv(run_dir / "mse_best_summary.csv", index=False)

    print(f"复盘输出目录：{run_dir}", flush=True)
    print(pd.DataFrame(compact_rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
