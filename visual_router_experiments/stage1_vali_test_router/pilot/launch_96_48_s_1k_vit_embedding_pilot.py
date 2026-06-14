#!/usr/bin/env python3
"""
文件功能：
    生成 96_48_S 1k ViT embedding smoke 后台运行 launcher。

输入：
    - sample_manifest.csv；
    - 输出目录和外部 embedding cache 目录。

输出：
    - launcher.sh：可直接启动单 GPU embedding smoke；
    - status.json：记录 launcher 生成状态和进度查看命令；
    - launch_plan.md：中文计划。

关键约束：
    - 默认只生成 launcher，不自动启动；
    - 不保存伪图像 tensor；
    - embedding `.npy` 若需要 smoke cache，默认写入 `/data2/syh/Time/cache_shards/`；
    - 后续 online embedding 正式路线应优先运行内存缓存，而不是长期依赖 `.npy`。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


WORKSPACE = Path("/home/shiyuhong/Time")
RUN_OUTPUT_ROOT = WORKSPACE / "experiment_logs" / "run_outputs"
CACHE_ROOT = Path("/data2/syh/Time/cache_shards")
PYTHON = Path("/home/shiyuhong/application/miniconda3/envs/quito/bin/python")
STAGE_DIR = WORKSPACE / "visual_router_experiments" / "stage1_vali_test_router"
PILOT_DIR = STAGE_DIR / "pilot"


def now_token() -> str:
    """函数功能：生成输出目录时间戳，精确到微秒避免重名。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def display_time() -> str:
    """函数功能：生成写入 metadata/status/plan 的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def shell_quote(value: object) -> str:
    """函数功能：shell 命令参数安全转义。"""
    return shlex.quote(str(value))


def parse_args() -> argparse.Namespace:
    """函数功能：解析 launcher 参数。"""
    parser = argparse.ArgumentParser(description="Create launcher for 96_48_S 1k ViT embedding smoke.")
    parser.add_argument("--sample-manifest-path", type=Path, required=True, help="1k sample_manifest.csv 路径。")
    parser.add_argument("--output-root", type=Path, default=RUN_OUTPUT_ROOT, help="launcher 输出根目录。")
    parser.add_argument("--output-dir", type=Path, default=None, help="显式 launcher 目录。")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT, help="embedding npy 外部缓存根目录。")
    parser.add_argument("--gpu", default="3", help="ViT embedding 使用的物理 GPU id。")
    parser.add_argument("--batch-size", type=int, default=32, help="ViT 前向 batch size。")
    parser.add_argument("--auto-start", action="store_true", help="生成后立即启动；默认不启动。")
    return parser.parse_args()


def main() -> None:
    """函数功能：生成 embedding launcher，并按需自动启动。"""
    args = parse_args()
    output_dir = args.output_dir or args.output_root / f"{now_token()}_visual_router_stage1_vit_embedding_96_48_s_1k_launcher"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_output_dir = output_dir / "embedding_run"
    embedding_cache_dir = args.cache_root / f"{output_dir.name}_embeddings"
    main_log = embedding_output_dir / "main.log"
    launcher_path = output_dir / "launcher.sh"
    builder = PILOT_DIR / "build_vit_embeddings_pilot.py"
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + str(args.gpu),
        str(PYTHON),
        str(builder),
        "--sample-manifest-path",
        str(args.sample_manifest_path),
        "--local-files-only",
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cuda",
        "--output-dir",
        str(embedding_output_dir),
        "--cache-root",
        str(embedding_cache_dir),
        "--period-selection",
        "fixed_candidates",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shell_quote(WORKSPACE)}",
        f"mkdir -p {shell_quote(embedding_output_dir)}",
        "(",
        "  set -euo pipefail",
        "  " + " ".join(shell_quote(part) for part in cmd),
        f") > {shell_quote(main_log)} 2>&1 &",
        f"echo $! > {shell_quote(output_dir / 'embedding.pid')}",
        f"echo 'tail -f {main_log}'",
        f"echo 'cat {embedding_output_dir / 'embedding_metadata.json'}'",
        "echo 'nvidia-smi'",
    ]
    launcher_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launcher_path.chmod(0o755)

    plan = [
        "# 96_48_S 1k ViT Embedding Launcher Plan",
        "",
        f"生成时间：{display_time()}",
        "",
        "## 输入与输出",
        "",
        f"- sample_manifest: `{args.sample_manifest_path}`",
        f"- embedding output: `{embedding_output_dir}`",
        f"- embedding cache: `{embedding_cache_dir}`",
        f"- main.log: `{main_log}`",
        "",
        "## 启动命令",
        "",
        f"```bash\nbash {launcher_path}\n```",
        "",
        "## 查看进度",
        "",
        f"```bash\ntail -f {main_log}\ncat {embedding_output_dir / 'embedding_metadata.json'}\nnvidia-smi\n```",
        "",
        "## 约束",
        "",
        "- 不保存伪图像 tensor。",
        "- 本 launcher 是 1k embedding cache smoke / 历史对照；当前正式 online 方案仍使用运行内存缓存。",
        "",
    ]
    (output_dir / "launch_plan.md").write_text("\n".join(plan), encoding="utf-8")
    status = {
        "status": "launcher_created",
        "generated_at": display_time(),
        "output_dir": str(output_dir),
        "sample_manifest_path": str(args.sample_manifest_path),
        "launcher_path": str(launcher_path),
        "auto_start": bool(args.auto_start),
        "gpu": str(args.gpu),
        "embedding_output_dir": str(embedding_output_dir),
        "embedding_cache_dir": str(embedding_cache_dir),
        "main_log": str(main_log),
        "command": cmd,
        "progress_commands": [
            f"tail -f {main_log}",
            f"cat {embedding_output_dir / 'embedding_metadata.json'}",
            "nvidia-smi",
        ],
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote launcher to {launcher_path}")
    print(f"status: {output_dir / 'status.json'}")
    print(f"plan: {output_dir / 'launch_plan.md'}")
    if args.auto_start:
        subprocess.run(["bash", str(launcher_path)], cwd=str(WORKSPACE), check=True)


if __name__ == "__main__":
    main()
