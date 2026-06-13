#!/usr/bin/env python3
"""
文件功能：
    编排 QuitoBench 默认架构参数的单 seed baseline 训练与评估。

实验范围：
    - 模型：DLinear、PatchTST、CrossFormer
    - 设置：96_48_S、576_288_S、1024_512_S
    - seed：默认 16
    - 训练/评估资源：每个实验使用 4 张 GPU；由于机器总共 4 张 GPU，任务按顺序执行。

设计说明：
    本脚本不修改官方 configs。每次运行都会在 experiment_logs/run_outputs 下生成临时
    finetune/evaluate 配置，并把 Quito 输出写入 outputs/default_baseline/...，避免污染
    官方默认输出目录，也方便后续与 tuned 结果并排比较。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml


WORKSPACE = Path("/home/shiyuhong/Time")
REPO_DIR = WORKSPACE / "quito"
ENV_BIN = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
ITEM_CLUSTER_PATH = REPO_DIR / "examples" / "datasets" / "cluster_data" / "item_clusters.csv"

# 关键约束：每个任务都会占满 4 张 GPU，因此这里的顺序就是实际执行顺序。
DEFAULT_JOB_ORDER = (
    ("dlinear", "96_48_S"),
    ("patchtst", "96_48_S"),
    ("crossformer", "96_48_S"),
    ("dlinear", "576_288_S"),
    ("patchtst", "576_288_S"),
    ("crossformer", "576_288_S"),
    ("dlinear", "1024_512_S"),
    ("patchtst", "1024_512_S"),
    ("crossformer", "1024_512_S"),
)


@dataclass
class BaselineJob:
    """函数功能：保存单个 baseline 任务的配置、命令、输出路径和运行状态。"""

    model: str
    config_name: str
    seed: int
    status: str = "pending"
    return_code: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    finetune_config_path: Optional[str] = None
    evaluate_config_path: Optional[str] = None
    finetune_log_path: Optional[str] = None
    evaluate_log_path: Optional[str] = None
    finetune_command: List[str] = field(default_factory=list)
    evaluate_command: List[str] = field(default_factory=list)
    finetune_output_dir: Optional[str] = None
    evaluate_output_dir: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    eval_results_path: Optional[str] = None
    per_item_results_path: Optional[str] = None
    cluster_metrics_path: Optional[str] = None
    summary_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def job_name(self) -> str:
        """函数功能：生成稳定任务名，用于日志、状态文件和结果汇总。"""
        return f"{self.model}_{self.config_name}_seed{self.seed}"

    @property
    def finetune_source_config_path(self) -> Path:
        """函数功能：返回官方 finetune 源配置路径。"""
        return REPO_DIR / "configs" / "finetune" / self.model / f"{self.config_name}.yaml"

    @property
    def evaluate_source_config_path(self) -> Path:
        """函数功能：返回官方 evaluate 源配置路径。"""
        return REPO_DIR / "configs" / "evaluate" / self.model / f"{self.config_name}.yaml"


def now_str() -> str:
    """函数功能：返回秒级本地时间字符串，便于状态追踪。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> dict:
    """函数功能：用结构化方式读取 YAML，避免字符串替换误改配置。"""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, payload: dict) -> None:
    """函数功能：写出 YAML，并保留中文注释可读性所需的 Unicode 字符。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_json(path: Path, payload: dict) -> None:
    """函数功能：写出带中文字段的 JSON 状态或汇总文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_jobs(seed: int, only: str = "") -> List[BaselineJob]:
    """
    函数功能：
        生成 baseline 任务列表，并支持按 model:config 筛选。

    输入：
        only: 逗号分隔的任务名，例如 dlinear:96_48_S,patchtst:96_48_S。
    """
    jobs = [BaselineJob(model=model, config_name=config_name, seed=seed) for model, config_name in DEFAULT_JOB_ORDER]
    if not only:
        return jobs

    wanted = {item.strip() for item in only.split(",") if item.strip()}
    valid_names = {f"{job.model}:{job.config_name}" for job in jobs}
    unknown = wanted - valid_names
    if unknown:
        raise ValueError(f"--only 包含未知任务：{sorted(unknown)}；可选任务：{sorted(valid_names)}")
    return [job for job in jobs if f"{job.model}:{job.config_name}" in wanted]


def validate_inputs(jobs: Iterable[BaselineJob]) -> None:
    """函数功能：启动前检查仓库、环境、配置和 cluster 映射文件是否存在。"""
    if not REPO_DIR.exists():
        raise FileNotFoundError(f"Quito 仓库不存在：{REPO_DIR}")
    if not (ENV_BIN / "quito-cli").exists():
        raise FileNotFoundError(f"找不到 quito-cli：{ENV_BIN / 'quito-cli'}")
    if not ITEM_CLUSTER_PATH.exists():
        raise FileNotFoundError(f"找不到 cluster 映射文件：{ITEM_CLUSTER_PATH}")

    missing_paths = []
    for job in jobs:
        if not job.finetune_source_config_path.exists():
            missing_paths.append(str(job.finetune_source_config_path))
        if not job.evaluate_source_config_path.exists():
            missing_paths.append(str(job.evaluate_source_config_path))
    if missing_paths:
        raise FileNotFoundError("以下配置文件不存在：\n" + "\n".join(missing_paths))


def apply_training_overrides(
    config: dict,
    *,
    output_dir: str,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    num_epochs: int,
    num_workers: int,
    selection_metric: str,
    resume_checkpoint: str,
    resume_mode: str,
) -> dict:
    """
    函数功能：
        在保留模型架构参数的前提下，统一微调训练侧参数。

    关键假设：
        - batch_size 是每张 GPU / 每个 DDP 进程的 batch，4 卡全局 batch 为 batch_size * 4。
        - 这里使用官方 tune 配置中较常见的 batch=256、lr=1e-3 作为 default baseline
          的保守训练设置；后续 tuned 结果会单独覆盖这些参数。
    """
    config.setdefault("training", {})
    config["training"]["seed"] = seed
    config["training"]["num_epochs"] = num_epochs
    config["training"]["batch_size"] = batch_size
    config["training"]["eval_batch_size"] = eval_batch_size
    config["training"]["learning_rate"] = learning_rate
    config["training"]["num_workers"] = num_workers
    config["training"]["drop_last"] = True

    config.setdefault("optimization", {})
    config["optimization"]["scheduler"] = "cosine"
    config["optimization"]["scheduler_kwargs"] = {
        "T_max": num_epochs,
        "eta_min": max(learning_rate * 0.1, 1e-6),
    }
    config["optimization"]["optimizer"] = "adam"
    config["optimization"].setdefault("optimizer_kwargs", {})
    config["optimization"]["fp16"] = False

    config.setdefault("checkpointing", {})
    config["checkpointing"]["enable_checkpoints"] = True
    config["checkpointing"]["save_last_k"] = max(1, num_epochs)
    config["checkpointing"]["save_steps"] = None
    config["checkpointing"]["save_epochs"] = 1

    config.setdefault("logging", {})
    config["logging"]["output_dir"] = output_dir
    config["logging"]["logging_steps"] = 100
    config["logging"]["logging_epochs"] = 1

    config.setdefault("early_stopping", {})
    config["early_stopping"]["enable_early_stopping"] = False
    config["early_stopping"]["es_metric"] = selection_metric
    config["early_stopping"]["greater_is_better"] = False

    config.setdefault("evaluation", {})
    config["evaluation"]["eval_metrics"] = ["mse", "mae", "mase"]
    config["evaluation"]["eval_steps"] = None
    config["evaluation"]["eval_epochs"] = 1
    config["evaluation"]["save_eval_results_top_k"] = 0

    config.setdefault("resume", {})
    config.setdefault("model", {})
    # strict：恢复完整训练状态；model_only：只加载模型权重，重置优化器和调度器。
    if resume_checkpoint and resume_mode == "strict":
        config["resume"]["checkpoint_path"] = resume_checkpoint
        config["model"]["checkpoint_path"] = None
    elif resume_checkpoint and resume_mode == "model_only":
        config["resume"]["checkpoint_path"] = None
        config["model"]["checkpoint_path"] = resume_checkpoint
    else:
        config["resume"]["checkpoint_path"] = None
        config["model"]["checkpoint_path"] = None
    return config


def apply_evaluate_overrides(
    config: dict,
    *,
    output_dir: str,
    seed: int,
    checkpoint_path: str,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> dict:
    """
    函数功能：
        生成评估配置，指向刚训练得到的 best checkpoint，并保留官方 evaluate 的完整指标集合。
    """
    config.setdefault("training", {})
    config["training"]["seed"] = seed
    config["training"]["batch_size"] = batch_size
    config["training"]["eval_batch_size"] = eval_batch_size
    config["training"]["num_workers"] = num_workers

    config.setdefault("logging", {})
    config["logging"]["output_dir"] = output_dir

    config.setdefault("resume", {})
    config["resume"]["checkpoint_path"] = checkpoint_path
    return config


def prepare_finetune_config(job: BaselineJob, generated_root: Path, args: argparse.Namespace) -> Path:
    """函数功能：从官方 finetune 配置生成本次 baseline 的临时训练配置。"""
    output_dir = f"outputs/default_baseline/{job.model}/{job.config_name}/seed_{job.seed}"
    config = load_yaml(job.finetune_source_config_path)
    config = apply_training_overrides(
        config,
        output_dir=output_dir,
        seed=job.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_workers=args.num_workers,
        selection_metric=args.selection_metric,
        resume_checkpoint=args.resume_checkpoint,
        resume_mode=args.resume_mode,
    )
    target_path = generated_root / job.model / job.config_name / f"finetune_seed_{job.seed}.yaml"
    write_yaml(target_path, config)
    job.finetune_config_path = str(target_path)
    return target_path


def prepare_evaluate_config(job: BaselineJob, generated_root: Path, args: argparse.Namespace) -> Path:
    """函数功能：从官方 evaluate 配置生成指向 best checkpoint 的临时评估配置。"""
    if not job.best_checkpoint_path:
        raise ValueError(f"{job.job_name} 尚未设置 best checkpoint，不能生成 evaluate 配置。")

    output_dir = f"outputs/default_baseline/{job.model}/{job.config_name}/seed_{job.seed}"
    config = load_yaml(job.evaluate_source_config_path)
    config = apply_evaluate_overrides(
        config,
        output_dir=output_dir,
        seed=job.seed,
        checkpoint_path=job.best_checkpoint_path,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    target_path = generated_root / job.model / job.config_name / f"evaluate_seed_{job.seed}.yaml"
    write_yaml(target_path, config)
    job.evaluate_config_path = str(target_path)
    return target_path


def build_env(gpu_ids: str) -> dict:
    """函数功能：构造训练/评估子进程环境，显式暴露本轮要使用的 GPU。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("RAY_DEDUP_LOGS", "1")
    return env


def build_finetune_command(config_path: Path, args: argparse.Namespace) -> List[str]:
    """
    函数功能：生成 finetune 命令。

    关键说明：
        Quito CLI 在 finetune 时总是通过 torchrun 启动。多个单 GPU lane 并发时，
        即使 num_processes=1，多个 torchrun 仍会抢默认 rendezvous 端口，导致任务
        立即失败。因此单进程场景直接调用 finetune.py，让脚本按普通单进程 GPU 模式运行。
    """
    if args.num_processes == 1:
        return [
            str(ENV_BIN / "python"),
            str(REPO_DIR / "quito" / "scripts" / "finetune.py"),
            "--use_gpu",
            str(args.use_gpu),
            "--config_path",
            str(config_path),
            "--seed",
            str(args.seed),
        ]

    return [
        str(ENV_BIN / "quito-cli"),
        "finetune",
        "--config_path",
        str(config_path),
        "--num_processes",
        str(args.num_processes),
        "--use_gpu",
        str(args.use_gpu),
        "--seed",
        str(args.seed),
    ]


def build_evaluate_command(config_path: Path, args: argparse.Namespace) -> List[str]:
    """函数功能：生成 4 卡 Ray evaluate 命令。"""
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


def run_command(command: List[str], *, log_path: Path, env: dict, dry_run: bool) -> int:
    """
    函数功能：
        运行一个子命令，并将 stdout/stderr 写入独立日志。

    返回：
        子进程返回码；dry-run 时固定返回 0。
    """
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


def latest_version_dir(base_dir: Path) -> Path:
    """函数功能：返回 Quito 自动创建的最新 ver_* 目录。"""
    candidates = [path for path in base_dir.glob("ver_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"找不到版本目录：{base_dir}/ver_*")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_best_checkpoint(finetune_output_dir: Path) -> Path:
    """函数功能：在训练输出目录中找到 best checkpoint；若不存在则回退到 last checkpoint。"""
    checkpoint_dir = finetune_output_dir / "checkpoints"
    best_candidates = sorted(checkpoint_dir.glob("best_*.ckpt"), key=lambda path: path.stat().st_mtime)
    if best_candidates:
        return best_candidates[-1].resolve()

    last_candidates = sorted(checkpoint_dir.glob("last_*.ckpt"), key=lambda path: path.stat().st_mtime)
    if last_candidates:
        return last_candidates[-1].resolve()

    raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_dir}")


def find_eval_results(evaluate_output_dir: Path) -> Path:
    """函数功能：找到 evaluate 阶段输出的 eval_results_*.json 文件。"""
    candidates = sorted(evaluate_output_dir.glob("eval_results_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"找不到评估结果 JSON：{evaluate_output_dir}/eval_results_*.json")
    return candidates[-1].resolve()


def summarize_eval_results(job: BaselineJob, result_root: Path) -> None:
    """
    函数功能：
        读取 Quito 原始评估 JSON，额外生成 per-item、per-cluster 和 overall summary。

    关键说明：
        Quito 原始 JSON 已经按 user/item 输出指标。本函数只做结构化展开和 cluster 映射，
        不重新计算模型预测，避免改变官方 evaluator 的行为。
    """
    if not job.eval_results_path:
        raise ValueError(f"{job.job_name} 尚未设置 eval_results_path。")

    with Path(job.eval_results_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = []
    for item in payload.get("final_results", []):
        row = {
            "model": job.model,
            "config_name": job.config_name,
            "seed": job.seed,
            "item_id": int(item["user_id"]),
            "n_samples": item.get("n_samples", 0),
            "eval_time": item.get("eval_time", 0.0),
        }
        row.update(item.get("metrics", {}))
        rows.append(row)

    if not rows:
        raise ValueError(f"{job.eval_results_path} 中没有 final_results。")

    result_df = pd.DataFrame(rows)
    cluster_df = pd.read_csv(ITEM_CLUSTER_PATH)
    keep_cols = [
        "item_id",
        "cluster",
        "group_name",
        "forecastability_cat",
        "season_strength_cat",
        "trend_strength_cat",
        "cv_cat",
        "missing_ratio_cat",
    ]
    keep_cols = [col for col in keep_cols if col in cluster_df.columns]
    result_df = result_df.merge(cluster_df[keep_cols], on="item_id", how="left")

    metric_cols = [
        col
        for col in result_df.columns
        if col not in {"model", "config_name", "seed", "item_id", "n_samples", "eval_time", "cluster", "group_name"}
        and pd.api.types.is_numeric_dtype(result_df[col])
    ]

    output_dir = result_root / job.model / job.config_name / f"seed_{job.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_item_path = output_dir / "per_item_results.csv"
    cluster_metrics_path = output_dir / "cluster_metrics.csv"
    summary_path = output_dir / "summary.json"

    result_df.to_csv(per_item_path, index=False)

    # cluster 层面同时保留均值、标准差、item 数和样本数，后续画图或统计检验都更方便。
    agg_parts = []
    group_cols = ["cluster", "group_name"]
    for keys, group in result_df.groupby(group_cols, dropna=False):
        row = {
            "cluster": keys[0],
            "group_name": keys[1],
            "item_count": int(group["item_id"].nunique()),
            "sample_count": int(group["n_samples"].sum()),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        agg_parts.append(row)
    cluster_metrics_df = pd.DataFrame(agg_parts).sort_values(["cluster", "group_name"])
    cluster_metrics_df.to_csv(cluster_metrics_path, index=False)

    metrics = {metric: float(result_df[metric].mean()) for metric in metric_cols}
    summary = {
        "model": job.model,
        "model_name": payload.get("model_name"),
        "config_name": job.config_name,
        "seed": job.seed,
        "item_count": int(result_df["item_id"].nunique()),
        "sample_count": int(result_df["n_samples"].sum()),
        "metrics_mean_over_items": metrics,
        "eval_results_path": job.eval_results_path,
        "per_item_results_path": str(per_item_path),
        "cluster_metrics_path": str(cluster_metrics_path),
    }
    write_json(summary_path, summary)

    job.per_item_results_path = str(per_item_path)
    job.cluster_metrics_path = str(cluster_metrics_path)
    job.summary_path = str(summary_path)
    job.metrics = metrics


def write_status(status_path: Path, jobs: List[BaselineJob], *, run_dir: Path, extra: Optional[dict] = None) -> None:
    """函数功能：写出全局状态文件，便于中途查看和恢复。"""
    payload = {
        "updated_at": now_str(),
        "run_dir": str(run_dir),
        "jobs": [asdict(job) for job in jobs],
    }
    if extra:
        payload.update(extra)
    write_json(status_path, payload)


def write_final_summary(run_dir: Path, jobs: List[BaselineJob]) -> None:
    """函数功能：汇总全部已完成任务的核心指标，方便快速查看 baseline 表。"""
    rows = []
    for job in jobs:
        if job.status != "completed":
            continue
        row = {
            "model": job.model,
            "config_name": job.config_name,
            "seed": job.seed,
            "finetune_output_dir": job.finetune_output_dir,
            "evaluate_output_dir": job.evaluate_output_dir,
            "best_checkpoint_path": job.best_checkpoint_path,
            "eval_results_path": job.eval_results_path,
            "per_item_results_path": job.per_item_results_path,
            "cluster_metrics_path": job.cluster_metrics_path,
        }
        row.update(job.metrics)
        rows.append(row)

    summary_path = run_dir / "baseline_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def parse_args() -> argparse.Namespace:
    """函数功能：解析 baseline 编排参数，并设置单 seed 4 卡默认值。"""
    parser = argparse.ArgumentParser(description="Run default-config QuitoBench baseline finetune/evaluate jobs.")
    parser.add_argument("--seed", type=int, default=16, help="本轮 baseline 使用的单个随机种子。")
    parser.add_argument("--num-processes", type=int, default=4, help="每个训练/评估任务使用的进程/GPU 数。")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3", help="暴露给子进程的 GPU id 列表。")
    parser.add_argument("--use-gpu", type=int, default=1, choices=[0, 1], help="是否使用 GPU。")
    parser.add_argument("--batch-size", type=int, default=256, help="每张 GPU 的训练 batch size。")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="每张 GPU 的评估 batch size。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="baseline 训练学习率。")
    parser.add_argument("--num-epochs", type=int, default=5, help="baseline 训练 epoch 数。")
    parser.add_argument("--num-workers", type=int, default=6, help="每个进程的 DataLoader worker 数。")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="mse",
        choices=["mse", "mae"],
        help="用于选择 best checkpoint 的验证集指标；论文 protocol 使用 mse。",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="续训 checkpoint 路径；需要配合 --only 指定单个任务。",
    )
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="strict",
        choices=["strict", "model_only"],
        help="strict 加载 optimizer/scheduler/epoch；model_only 只加载模型权重并重置优化器和调度器。",
    )
    parser.add_argument("--only", type=str, default="", help="只运行指定任务，格式如 dlinear:96_48_S。")
    parser.add_argument("--dry-run", action="store_true", help="只生成配置和命令，不启动训练/评估。")
    parser.add_argument("--continue-on-failure", action="store_true", help="某个任务失败后继续后续任务。")
    return parser.parse_args()


def main() -> int:
    """函数功能：创建临时配置，顺序执行 4 卡 baseline 训练与评估，并生成 cluster 汇总。"""
    args = parse_args()
    jobs = build_jobs(seed=args.seed, only=args.only)
    if args.resume_checkpoint and len(jobs) != 1:
        raise ValueError("--resume-checkpoint 只能和 --only 指定的单个任务一起使用，避免 checkpoint 套错任务。")
    validate_inputs(jobs)

    # 多个并发 lane 可能在同一秒启动；使用微秒级时间戳避免 run_dir/status.json 冲突。
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = RUN_OUTPUT_ROOT / f"{run_stamp}_default_baseline_finetune_eval"
    generated_root = run_dir / "generated_configs"
    result_root = run_dir / "cluster_analysis"
    log_root = run_dir / "logs"
    status_path = run_dir / "status.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    extra_status = {
        "seed": args.seed,
        "num_processes": args.num_processes,
        "gpu_ids": args.gpu_ids,
        "batch_size_per_gpu": args.batch_size,
        "eval_batch_size_per_gpu": args.eval_batch_size,
        "global_batch_size": args.batch_size * args.num_processes,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "selection_metric": args.selection_metric,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_mode": args.resume_mode,
        "dry_run": args.dry_run,
    }

    print(f"运行目录：{run_dir}", flush=True)
    print(f"状态文件：{status_path}", flush=True)
    print(f"任务数量：{len(jobs)}", flush=True)
    print(f"任务顺序：{[job.job_name for job in jobs]}", flush=True)
    print(f"每个任务使用 GPU：{args.gpu_ids}", flush=True)
    print(f"每个任务进程数：{args.num_processes}", flush=True)
    print(f"每卡 batch_size：{args.batch_size}", flush=True)
    print(f"全局 batch_size：{args.batch_size * args.num_processes}", flush=True)
    print(f"learning_rate：{args.learning_rate}", flush=True)
    print(f"selection_metric：{args.selection_metric}", flush=True)
    if args.resume_checkpoint:
        print(f"resume_checkpoint：{args.resume_checkpoint}", flush=True)
        print(f"resume_mode：{args.resume_mode}", flush=True)
    print(f"dry_run：{args.dry_run}", flush=True)

    env = build_env(args.gpu_ids)
    failed_jobs = []
    write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

    for job in jobs:
        job.start_time = now_str()
        job.status = "running"
        write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

        try:
            finetune_config_path = prepare_finetune_config(job, generated_root, args)
            finetune_command = build_finetune_command(finetune_config_path, args)
            job.finetune_command = finetune_command
            job.finetune_log_path = str(log_root / f"{job.job_name}_finetune.log")

            print(f"[{now_str()}] 开始训练 {job.job_name}", flush=True)
            return_code = run_command(
                finetune_command,
                log_path=Path(job.finetune_log_path),
                env=env,
                dry_run=args.dry_run,
            )
            if return_code != 0:
                raise RuntimeError(f"finetune 失败，return_code={return_code}")

            base_output_dir = REPO_DIR / "outputs" / "default_baseline" / job.model / job.config_name / f"seed_{job.seed}"
            if args.dry_run:
                job.finetune_output_dir = str(base_output_dir / "FINE_TUNE" / "ver_dry_run")
                job.best_checkpoint_path = str(base_output_dir / "FINE_TUNE" / "ver_dry_run" / "checkpoints" / "best_dry_run.ckpt")
            else:
                finetune_output_dir = latest_version_dir(base_output_dir / "FINE_TUNE")
                job.finetune_output_dir = str(finetune_output_dir.resolve())
                job.best_checkpoint_path = str(find_best_checkpoint(finetune_output_dir))

            evaluate_config_path = prepare_evaluate_config(job, generated_root, args)
            evaluate_command = build_evaluate_command(evaluate_config_path, args)
            job.evaluate_command = evaluate_command
            job.evaluate_log_path = str(log_root / f"{job.job_name}_evaluate.log")

            print(f"[{now_str()}] 开始评估 {job.job_name}", flush=True)
            return_code = run_command(
                evaluate_command,
                log_path=Path(job.evaluate_log_path),
                env=env,
                dry_run=args.dry_run,
            )
            if return_code != 0:
                raise RuntimeError(f"evaluate 失败，return_code={return_code}")

            if args.dry_run:
                job.evaluate_output_dir = str(base_output_dir / "EVALUATE" / "ver_dry_run")
                job.eval_results_path = str(base_output_dir / "EVALUATE" / "ver_dry_run" / "eval_results_dry_run.json")
                job.metrics = {}
            else:
                evaluate_output_dir = latest_version_dir(base_output_dir / "EVALUATE")
                job.evaluate_output_dir = str(evaluate_output_dir.resolve())
                job.eval_results_path = str(find_eval_results(evaluate_output_dir))
                summarize_eval_results(job, result_root)

            job.status = "completed"
            job.return_code = 0
            job.end_time = now_str()
            print(f"[{now_str()}] 完成 {job.job_name}", flush=True)

        except Exception as exc:
            job.status = "failed"
            job.return_code = 1
            job.end_time = now_str()
            failed_jobs.append(job.job_name)
            print(f"[{now_str()}] 失败 {job.job_name}: {exc}", flush=True)
            if not args.continue_on_failure:
                write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)
                break

        write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)
        write_final_summary(run_dir, jobs)
        time.sleep(3)

    write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)
    write_final_summary(run_dir, jobs)

    if failed_jobs:
        print(f"存在失败任务：{failed_jobs}", flush=True)
        return 1

    print("全部 baseline 任务结束。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
