#!/usr/bin/env python3
"""
文件功能：
    编排 PatchTST / CrossFormer 的单卡并发架构粗搜任务。

当前实验方案：
    - 每个 tuning job 只暴露 1 张 GPU，多个 job 并发运行。
    - 为了尽量贴近后续 4 卡 finetune 的全局 batch，本脚本会为每个任务生成临时
      base config，并将 training.batch_size 改为 1024。
    - 任务启动顺序按用户指定的优先级排列：
        1. patchtst_96_48_S、crossformer_96_48_S
        2. patchtst_576_288_S、crossformer_576_288_S
        3. patchtst_1024_512_S、crossformer_1024_512_S

重要说明：
    这是“单卡并发粗搜”脚本，不是最终 4 卡验证脚本。粗搜完成后，应取每组 top-1
    或接近的 top-2 参数，再用 4 卡配置做固定架构验证。
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
from typing import Dict, List, Optional

import yaml


WORKSPACE = Path("/home/shiyuhong/Time")
REPO_DIR = WORKSPACE / "quito"
ENV_BIN = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"

# 关键假设：这里的顺序就是粗搜优先级。并发槽会从前往后取任务，因此 96_48 会最先启动。
DEFAULT_JOB_ORDER = (
    ("patchtst", "96_48_S"),
    ("crossformer", "96_48_S"),
    ("patchtst", "576_288_S"),
    ("crossformer", "576_288_S"),
    ("patchtst", "1024_512_S"),
    ("crossformer", "1024_512_S"),
)


@dataclass
class TuningJob:
    """函数功能：记录单个 tuning job 的配置、运行状态和可复核路径。"""

    model: str
    config_name: str
    status: str = "pending"
    return_code: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    gpu_id: Optional[int] = None
    pid: Optional[int] = None
    log_path: Optional[str] = None
    generated_config_path: Optional[str] = None
    command: List[str] = field(default_factory=list)
    expected_output_root: Optional[str] = None

    @property
    def job_name(self) -> str:
        """函数功能：生成稳定任务名，用于日志文件和状态文件。"""
        return f"{self.model}_{self.config_name}"

    @property
    def source_config_path(self) -> Path:
        """函数功能：返回 Quito 原始 tune base config 路径。"""
        return REPO_DIR / "configs" / "tune" / self.model / f"{self.config_name}.yaml"

    @property
    def tuning_config_path(self) -> Path:
        """函数功能：返回该模型的架构搜索空间配置路径。"""
        return REPO_DIR / "configs" / "tune" / self.model / "tuning_config.yaml"


def now_str() -> str:
    """函数功能：返回秒级本地时间字符串，用于状态和日志。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_default_jobs() -> List[TuningJob]:
    """函数功能：按用户要求的优先级构造默认 6 组粗搜任务。"""
    return [TuningJob(model=model, config_name=config_name) for model, config_name in DEFAULT_JOB_ORDER]


def filter_jobs(jobs: List[TuningJob], only: str) -> List[TuningJob]:
    """
    函数功能：
        根据 --only 筛选任务，便于从失败点续跑。

    输入格式：
        多个任务用逗号分隔，单个任务格式为 model:config，例如：
        patchtst:96_48_S,crossformer:96_48_S
    """
    if not only:
        return jobs

    wanted = {item.strip() for item in only.split(",") if item.strip()}
    valid_names = {f"{job.model}:{job.config_name}" for job in jobs}
    unknown = wanted - valid_names
    if unknown:
        raise ValueError(f"--only 包含未知任务：{sorted(unknown)}；可选任务：{sorted(valid_names)}")
    return [job for job in jobs if f"{job.model}:{job.config_name}" in wanted]


def validate_inputs(jobs: List[TuningJob]) -> None:
    """函数功能：在启动前检查 Quito 仓库、conda 环境和配置文件是否存在。"""
    if not REPO_DIR.exists():
        raise FileNotFoundError(f"Quito 仓库不存在：{REPO_DIR}")
    if not (ENV_BIN / "quito-cli").exists():
        raise FileNotFoundError(f"找不到 quito-cli：{ENV_BIN / 'quito-cli'}")

    missing_paths = []
    for job in jobs:
        if not job.source_config_path.exists():
            missing_paths.append(str(job.source_config_path))
        if not job.tuning_config_path.exists():
            missing_paths.append(str(job.tuning_config_path))
    if missing_paths:
        raise FileNotFoundError("以下配置文件不存在：\n" + "\n".join(missing_paths))


def prepare_generated_config(
    job: TuningJob,
    *,
    generated_root: Path,
    batch_size: int,
    eval_batch_size: Optional[int],
) -> Path:
    """
    函数功能：
        读取 Quito 原始 base config，生成单卡粗搜用的临时 config。

    关键实现说明：
        - 使用 YAML 结构化读写，避免字符串替换误改无关字段。
        - training.batch_size 改为 1024，是为了让单卡粗搜的全局 batch 接近原 4 卡
          finetune/tune 的 256*4。
        - logging.output_dir 单独放到 outputs/single_gpu_screen，避免污染正式 4 卡输出目录。
    """
    with job.source_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.setdefault("training", {})
    config["training"]["batch_size"] = batch_size
    if eval_batch_size is not None:
        config["training"]["eval_batch_size"] = eval_batch_size

    config.setdefault("logging", {})
    config["logging"]["output_dir"] = f"outputs/single_gpu_screen/{job.model}/{job.config_name}"

    target_dir = generated_root / job.model
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{job.config_name}.yaml"
    with target_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    job.generated_config_path = str(target_path)
    job.expected_output_root = str(REPO_DIR / "outputs" / "single_gpu_screen" / job.model / job.config_name / "TUNE")
    return target_path


def build_command(job: TuningJob, *, num_processes: int, num_samples: int, use_gpu: int) -> List[str]:
    """
    函数功能：
        为单个单卡 tuning job 生成 quito-cli 命令。

    关键约束：
        generated_config_path 使用绝对路径，确保并发子进程都读取本次 run_dir 下的临时配置。
    """
    if job.generated_config_path is None:
        raise ValueError(f"{job.job_name} 尚未生成临时 config。")

    return [
        str(ENV_BIN / "quito-cli"),
        "tune",
        "--config_path",
        job.generated_config_path,
        "--tuning_config_path",
        str(job.tuning_config_path),
        "--num_processes",
        str(num_processes),
        "--num_samples",
        str(num_samples),
        "--use_gpu",
        str(use_gpu),
    ]


def build_env(gpu_id: int) -> dict:
    """
    函数功能：
        构造单 job 的子进程环境，只暴露一张 GPU。

    关键假设：
        独立 Ray 进程只看到一张 GPU 后，ScalingConfig(num_workers=1) 会稳定落在这张卡上。
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PATH"] = f"{ENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("RAY_DEDUP_LOGS", "1")
    return env


def write_status(status_path: Path, jobs: List[TuningJob], *, run_dir: Path, extra: Optional[dict] = None) -> None:
    """函数功能：将全部任务状态写入 JSON，便于中途查看和实验复盘。"""
    payload = {
        "updated_at": now_str(),
        "run_dir": str(run_dir),
        "jobs": [asdict(job) for job in jobs],
    }
    if extra:
        payload.update(extra)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def launch_job(
    *,
    job: TuningJob,
    gpu_id: int,
    run_dir: Path,
    status_path: Path,
    jobs: List[TuningJob],
    num_processes: int,
    num_samples: int,
    use_gpu: int,
    dry_run: bool,
) -> Optional[subprocess.Popen]:
    """
    函数功能：
        启动一个单卡 tuning 子进程，并把 stdout/stderr 写入独立日志。

    返回：
        dry-run 时返回 None；正式运行时返回 Popen 对象，由主循环轮询。
    """
    job.gpu_id = gpu_id
    job.command = build_command(job, num_processes=num_processes, num_samples=num_samples, use_gpu=use_gpu)
    job.start_time = now_str()
    job.status = "planned" if dry_run else "running"
    job.log_path = str(run_dir / f"{job.job_name}.log")

    log_file = Path(job.log_path).open("w", encoding="utf-8")
    log_file.write(f"# job: {job.job_name}\n")
    log_file.write(f"# start_time: {job.start_time}\n")
    log_file.write(f"# gpu_id: {gpu_id}\n")
    log_file.write(f"# generated_config_path: {job.generated_config_path}\n")
    log_file.write(f"# expected_output_root: {job.expected_output_root}\n")
    log_file.write(f"# command: {' '.join(job.command)}\n\n")
    log_file.flush()

    if dry_run:
        job.return_code = 0
        job.end_time = now_str()
        log_file.write(f"# end_time: {job.end_time}\n")
        log_file.write("# return_code: 0\n")
        log_file.close()
        write_status(status_path, jobs, run_dir=run_dir)
        return None

    process = subprocess.Popen(
        job.command,
        cwd=str(REPO_DIR),
        env=build_env(gpu_id),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job.pid = process.pid
    # 将日志文件句柄挂到进程对象，主循环收尾时关闭，避免并发日志被提前关闭。
    process._quito_log_file = log_file  # type: ignore[attr-defined]
    write_status(status_path, jobs, run_dir=run_dir)
    return process


def parse_args() -> argparse.Namespace:
    """函数功能：解析脚本参数，并设置单卡并发粗搜默认值。"""
    parser = argparse.ArgumentParser(description="Run single-GPU concurrent PatchTST/CrossFormer tuning screen.")
    parser.add_argument("--max-parallel", type=int, default=4, help="最多同时运行的单卡 tuning job 数。")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3", help="可用 GPU id 列表，例如 0,1,2,3。")
    parser.add_argument("--num-processes", type=int, default=1, help="每个 job 的 Ray Train worker 数；单卡粗搜应为 1。")
    parser.add_argument("--num-samples", type=int, default=10, help="每组 Ray Tune 采样次数。")
    parser.add_argument("--batch-size", type=int, default=1024, help="临时 base config 中写入的训练 batch_size。")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="临时 base config 中写入的 eval_batch_size。")
    parser.add_argument("--use-gpu", type=int, default=1, choices=[0, 1], help="是否使用 GPU。")
    parser.add_argument("--only", type=str, default="", help="只运行指定任务，格式如 patchtst:96_48_S。")
    parser.add_argument("--dry-run", action="store_true", help="只生成配置、命令和状态文件，不启动训练。")
    parser.add_argument("--continue-on-failure", action="store_true", help="某个 job 失败后继续运行队列中的后续任务。")
    return parser.parse_args()


def main() -> int:
    """
    函数功能：
        创建临时配置，按优先级队列调度单卡并发 tuning job，并记录完整状态。
    """
    args = parse_args()
    jobs = filter_jobs(build_default_jobs(), args.only)
    validate_inputs(jobs)

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if args.use_gpu and not gpu_ids:
        raise ValueError("use_gpu=1 时必须提供至少一个 GPU id。")
    if args.use_gpu and args.num_processes != 1:
        raise ValueError("当前脚本用于单卡粗搜，use_gpu=1 时 --num-processes 应保持为 1。")

    max_parallel = min(args.max_parallel, len(gpu_ids), len(jobs)) if args.use_gpu else min(args.max_parallel, len(jobs))

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUN_OUTPUT_ROOT / f"{run_stamp}_patchtst_crossformer_single_gpu_screen"
    generated_root = run_dir / "generated_configs"
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"

    for job in jobs:
        prepare_generated_config(
            job,
            generated_root=generated_root,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
        )

    print(f"运行目录：{run_dir}", flush=True)
    print(f"状态文件：{status_path}", flush=True)
    print(f"任务数量：{len(jobs)}", flush=True)
    print(f"任务顺序：{[job.job_name for job in jobs]}", flush=True)
    print(f"GPU 列表：{gpu_ids}", flush=True)
    print(f"最大并发：{max_parallel}", flush=True)
    print(f"每组 num_processes：{args.num_processes}", flush=True)
    print(f"每组 num_samples：{args.num_samples}", flush=True)
    print(f"临时 batch_size：{args.batch_size}", flush=True)
    print(f"临时 eval_batch_size：{args.eval_batch_size}", flush=True)
    print(f"dry_run：{args.dry_run}", flush=True)

    extra_status = {
        "max_parallel": max_parallel,
        "gpu_ids": gpu_ids,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_processes": args.num_processes,
        "num_samples": args.num_samples,
    }
    write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

    if args.dry_run:
        # dry-run 只验证配置和命令，不模拟真实并发等待；GPU 采用 round-robin 展示预期分配。
        for index, job in enumerate(jobs):
            gpu_id = gpu_ids[index % len(gpu_ids)] if args.use_gpu else -1
            print(f"[{now_str()}] 计划 {job.job_name}，GPU={gpu_id}", flush=True)
            launch_job(
                job=job,
                gpu_id=gpu_id,
                run_dir=run_dir,
                status_path=status_path,
                jobs=jobs,
                num_processes=args.num_processes,
                num_samples=args.num_samples,
                use_gpu=args.use_gpu,
                dry_run=True,
            )
        print("全部 dry-run 任务已生成。", flush=True)
        return 0

    pending = jobs[:]
    running: Dict[subprocess.Popen, TuningJob] = {}
    free_gpus = gpu_ids[:]
    failed_jobs: List[str] = []

    while pending or running:
        while pending and len(running) < max_parallel and (free_gpus or not args.use_gpu):
            job = pending.pop(0)
            gpu_id = free_gpus.pop(0) if args.use_gpu else -1
            print(f"[{now_str()}] 启动 {job.job_name}，GPU={gpu_id}", flush=True)
            process = launch_job(
                job=job,
                gpu_id=gpu_id,
                run_dir=run_dir,
                status_path=status_path,
                jobs=jobs,
                num_processes=args.num_processes,
                num_samples=args.num_samples,
                use_gpu=args.use_gpu,
                dry_run=args.dry_run,
            )
            if process is not None:
                running[process] = job

        time.sleep(10)

        for process, job in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue

            job.return_code = return_code
            job.end_time = now_str()
            job.status = "completed" if return_code == 0 else "failed"
            if return_code != 0:
                failed_jobs.append(job.job_name)

            log_file = getattr(process, "_quito_log_file", None)
            if log_file is not None:
                log_file.write(f"\n# end_time: {job.end_time}\n")
                log_file.write(f"# return_code: {return_code}\n")
                log_file.close()

            running.pop(process)
            if args.use_gpu and job.gpu_id is not None:
                free_gpus.append(job.gpu_id)
                free_gpus.sort()

            print(f"[{now_str()}] 结束 {job.job_name}，状态={job.status}，return_code={return_code}", flush=True)
            write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)

            if return_code != 0 and not args.continue_on_failure:
                print("检测到失败任务，已按默认策略停止调度新任务；等待已启动任务结束。", flush=True)
                pending.clear()

    write_status(status_path, jobs, run_dir=run_dir, extra=extra_status)
    if failed_jobs:
        print(f"存在失败任务：{failed_jobs}", flush=True)
        return 1

    print("全部任务结束。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
