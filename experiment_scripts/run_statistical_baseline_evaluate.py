#!/usr/bin/env python3
"""
文件功能：
    编排 QuitoBench 统计基线 ES 与 SNaive 的 evaluate，并生成整体与 cluster 汇总。

实验范围：
    - 模型：ES、SNaive
    - 配置：96_48_S、576_288_S、1024_512_S

设计说明：
    ES/SNaive 不需要 finetune；官方 evaluate 配置中的 checkpoint_path 为 [null]，
    只用于满足 evaluate.py 的 checkpoint 列表接口。本脚本不修改官方 configs，
    每次运行都会生成临时 evaluate YAML，并把结果写到 outputs/statistical_baseline。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from run_default_baseline_finetune_eval import (
    BaselineJob,
    ENV_BIN,
    REPO_DIR,
    RUN_OUTPUT_ROOT,
    find_eval_results,
    load_yaml,
    summarize_eval_results,
    write_json,
    write_yaml,
)


JOB_ORDER = (
    ("es", "96_48_S"),
    ("snaive", "96_48_S"),
    ("es", "576_288_S"),
    ("snaive", "576_288_S"),
    ("es", "1024_512_S"),
    ("snaive", "1024_512_S"),
)


@dataclass
class StatisticalEvalJob:
    """函数功能：保存一个统计基线 evaluate 任务的状态与输出路径。"""

    model: str
    config_name: str
    seed: int
    status: str = "pending"
    return_code: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    evaluate_config_path: Optional[str] = None
    evaluate_log_path: Optional[str] = None
    evaluate_command: List[str] = field(default_factory=list)
    evaluate_output_dir: Optional[str] = None
    eval_results_path: Optional[str] = None
    per_item_results_path: Optional[str] = None
    cluster_metrics_path: Optional[str] = None
    summary_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def job_name(self) -> str:
        """函数功能：生成稳定任务名。"""
        return f"{self.model}_{self.config_name}_seed{self.seed}"

    @property
    def source_config_path(self) -> Path:
        """函数功能：返回官方 evaluate 源配置。"""
        return REPO_DIR / "configs" / "evaluate" / self.model / f"{self.config_name}.yaml"


def now_str() -> str:
    """函数功能：返回秒级本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_jobs(seed: int, only: str) -> List[StatisticalEvalJob]:
    """函数功能：生成 ES/SNaive evaluate 任务列表，并支持按 model:config 过滤。"""
    jobs = [StatisticalEvalJob(model=model, config_name=config_name, seed=seed) for model, config_name in JOB_ORDER]
    if not only:
        return jobs

    wanted = {item.strip() for item in only.split(",") if item.strip()}
    valid = {f"{job.model}:{job.config_name}" for job in jobs}
    unknown = wanted - valid
    if unknown:
        raise ValueError(f"--only 包含未知任务：{sorted(unknown)}；可选任务：{sorted(valid)}")
    return [job for job in jobs if f"{job.model}:{job.config_name}" in wanted]


def validate_inputs(jobs: Iterable[StatisticalEvalJob]) -> None:
    """函数功能：检查 Quito 仓库、环境和官方配置是否存在。"""
    if not REPO_DIR.exists():
        raise FileNotFoundError(f"Quito 仓库不存在：{REPO_DIR}")
    if not (ENV_BIN / "quito-cli").exists():
        raise FileNotFoundError(f"找不到 quito-cli：{ENV_BIN / 'quito-cli'}")

    missing = [str(job.source_config_path) for job in jobs if not job.source_config_path.exists()]
    if missing:
        raise FileNotFoundError("以下 evaluate 配置不存在：\n" + "\n".join(missing))


def apply_evaluate_overrides(config: dict, *, output_dir: str, seed: int, eval_batch_size: int, num_workers: int) -> dict:
    """
    函数功能：
        在保留 ES/SNaive 模型参数的前提下，统一 evaluate 输出目录和资源参数。
    """
    config.setdefault("training", {})
    config["training"]["seed"] = seed
    config["training"]["batch_size"] = eval_batch_size
    config["training"]["eval_batch_size"] = eval_batch_size
    config["training"]["num_workers"] = num_workers
    config["training"]["shuffle"] = False

    config.setdefault("logging", {})
    config["logging"]["output_dir"] = output_dir

    # 统计模型没有训练 checkpoint；保留 [None] 以满足 evaluate.py 的 checkpoint 列表接口。
    config.setdefault("resume", {})
    config["resume"]["checkpoint_path"] = [None]
    return config


def prepare_evaluate_config(job: StatisticalEvalJob, generated_root: Path, args: argparse.Namespace) -> Path:
    """函数功能：生成单个统计基线任务的临时 evaluate 配置。"""
    config = load_yaml(job.source_config_path)
    output_dir = f"outputs/statistical_baseline/{job.model}/{job.config_name}/seed_{job.seed}"
    config = apply_evaluate_overrides(
        config,
        output_dir=output_dir,
        seed=job.seed,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    target_path = generated_root / job.model / job.config_name / f"evaluate_seed_{job.seed}.yaml"
    write_yaml(target_path, config)
    job.evaluate_config_path = str(target_path)
    return target_path


def build_env() -> dict:
    """函数功能：构造 evaluate 子进程环境。"""
    env = os.environ.copy()
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("RAY_DEDUP_LOGS", "1")
    return env


def build_evaluate_command(config_path: Path, args: argparse.Namespace) -> List[str]:
    """函数功能：生成 quito-cli evaluate 命令。"""
    return [
        str(ENV_BIN / "quito-cli"),
        "evaluate",
        "--config_path",
        str(config_path),
        "--num_processes",
        str(args.num_processes),
        "--use_gpu",
        str(args.use_gpu),
    ]


def latest_version_dir(base_dir: Path) -> Path:
    """函数功能：返回最新 ver_* 输出目录。"""
    candidates = [path for path in base_dir.glob("ver_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"找不到版本目录：{base_dir}/ver_*")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_command(command: List[str], *, log_path: Path, env: dict, dry_run: bool) -> int:
    """函数功能：运行 evaluate 子命令并写日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# start_time: {now_str()}\n")
        log_file.write(f"# command: {' '.join(command)}\n")
        log_file.write(f"# dry_run: {dry_run}\n\n")
        log_file.flush()
        if dry_run:
            log_file.write(f"\n# end_time: {now_str()}\n# return_code: 0\n")
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
        log_file.write(f"\n# end_time: {now_str()}\n# return_code: {return_code}\n")
        return return_code


def summarize_statistical_eval(job: StatisticalEvalJob, result_root: Path) -> None:
    """函数功能：复用 baseline 汇总逻辑，生成 per-item、cluster 和 summary 输出。"""
    if not job.eval_results_path:
        raise ValueError(f"{job.job_name} 尚未设置 eval_results_path。")

    baseline_job = BaselineJob(model=job.model, config_name=job.config_name, seed=job.seed)
    baseline_job.best_checkpoint_path = "statistical_model_no_checkpoint"
    baseline_job.eval_results_path = job.eval_results_path
    summarize_eval_results(baseline_job, result_root)

    job.per_item_results_path = baseline_job.per_item_results_path
    job.cluster_metrics_path = baseline_job.cluster_metrics_path
    job.summary_path = baseline_job.summary_path
    job.metrics = baseline_job.metrics


def write_status(status_path: Path, jobs: List[StatisticalEvalJob], *, run_dir: Path, extra: Optional[dict] = None) -> None:
    """函数功能：写出运行状态 JSON。"""
    payload = {
        "updated_at": now_str(),
        "run_dir": str(run_dir),
        "jobs": [asdict(job) for job in jobs],
    }
    if extra:
        payload.update(extra)
    write_json(status_path, payload)


def write_final_summary(run_dir: Path, jobs: List[StatisticalEvalJob]) -> None:
    """函数功能：写出 6 个统计基线任务的整体指标表。"""
    rows = []
    for job in jobs:
        row = {
            "model": job.model,
            "config_name": job.config_name,
            "seed": job.seed,
            "status": job.status,
            "evaluate_output_dir": job.evaluate_output_dir,
            "eval_results_path": job.eval_results_path,
            "per_item_results_path": job.per_item_results_path,
            "cluster_metrics_path": job.cluster_metrics_path,
        }
        row.update(job.metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "statistical_baseline_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    """函数功能：解析统计基线 evaluate 参数。"""
    parser = argparse.ArgumentParser(description="Run ES/SNaive QuitoBench statistical baseline evaluation.")
    parser.add_argument("--seed", type=int, default=16, help="evaluate 随机种子。")
    parser.add_argument("--num-processes", type=int, default=8, help="Ray evaluator actor 数。")
    parser.add_argument("--use-gpu", type=int, default=0, choices=[0, 1], help="是否使用 GPU；统计基线默认 CPU。")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="每个 evaluator 的 eval batch size。")
    parser.add_argument("--num-workers", type=int, default=2, help="每个 DataLoader 的 worker 数。")
    parser.add_argument("--only", type=str, default="", help="只运行指定任务，例如 es:96_48_S,snaive:96_48_S。")
    parser.add_argument("--dry-run", action="store_true", help="只生成配置和命令，不执行 evaluate。")
    parser.add_argument("--continue-on-failure", action="store_true", help="单个任务失败后继续后续任务。")
    return parser.parse_args()


def main() -> int:
    """函数功能：顺序执行 ES/SNaive evaluate 并生成汇总。"""
    args = parse_args()
    jobs = build_jobs(seed=args.seed, only=args.only)
    validate_inputs(jobs)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = RUN_OUTPUT_ROOT / f"{run_stamp}_statistical_baseline_evaluate"
    generated_root = run_dir / "generated_configs"
    result_root = run_dir / "cluster_analysis"
    log_root = run_dir / "logs"
    status_path = run_dir / "status.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    extra_status = {
        "seed": args.seed,
        "num_processes": args.num_processes,
        "use_gpu": args.use_gpu,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "dry_run": args.dry_run,
    }
    write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

    env = build_env()
    for job in jobs:
        job.status = "running"
        job.start_time = now_str()
        evaluate_config_path = prepare_evaluate_config(job, generated_root, args)
        command = build_evaluate_command(evaluate_config_path, args)
        log_path = log_root / f"{job.job_name}_evaluate.log"
        job.evaluate_command = command
        job.evaluate_log_path = str(log_path)
        write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

        return_code = run_command(command, log_path=log_path, env=env, dry_run=args.dry_run)
        job.return_code = return_code
        job.end_time = now_str()
        if return_code != 0:
            job.status = "failed"
            write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)
            if not args.continue_on_failure:
                write_final_summary(run_dir, jobs)
                return return_code
            continue

        if args.dry_run:
            job.status = "dry_run_completed"
            write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)
            continue

        output_base = REPO_DIR / "outputs" / "statistical_baseline" / job.model / job.config_name / f"seed_{job.seed}" / "EVALUATE"
        evaluate_output_dir = latest_version_dir(output_base)
        job.evaluate_output_dir = str(evaluate_output_dir.resolve())
        job.eval_results_path = str(find_eval_results(evaluate_output_dir))
        summarize_statistical_eval(job, result_root)
        job.status = "completed"
        write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

    write_final_summary(run_dir, jobs)
    print(f"统计基线输出目录：{run_dir}", flush=True)
    print(pd.read_csv(run_dir / "statistical_baseline_summary.csv").to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
