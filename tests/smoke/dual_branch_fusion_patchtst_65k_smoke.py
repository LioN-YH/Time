#!/usr/bin/env python3
"""
文件功能：
    PatchTST + fixed visual embedding 双分支融合实验 smoke。

输入：
    测试内构造的小规模 synthetic npz cache。

输出：
    临时目录中的 config/metrics/predictions/training_log/summary，并在标准输出打印中文检查日志。

关键约束：
    smoke 不访问 /data2，不生成图像，不运行 ViT，不训练 PatchTST；验证四个第一批
    轻量融合变体和 residual-safe mode 可以 forward、跑 1-2 个 mini train step 并写出验收产物。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FUSION_MODES = ("feature_concat", "film", "residual_feature", "visual_residual", "patchtst_residual_visual")


def build_synthetic_cache(cache_dir: Path) -> tuple[Path, Path]:
    """函数功能：构造同一 sample_key/split 的 PatchTST 与 visual synthetic cache。"""
    rng = np.random.default_rng(20260622)
    sample_count = 36
    horizon = 4
    channel = 2
    ts_dim = 6
    visual_dim = 5
    sample_key = np.asarray([f"sample_{idx:04d}" for idx in range(sample_count)])
    split = np.asarray(["train"] * 20 + ["val"] * 8 + ["test"] * 8)

    h_ts = rng.normal(size=(sample_count, ts_dim)).astype(np.float32)
    h_vis = rng.normal(size=(sample_count, visual_dim)).astype(np.float32)
    base = rng.normal(size=(sample_count, horizon, channel)).astype(np.float32)
    visual_signal = h_vis[:, :1, None] * 0.05
    y_true = base + visual_signal.astype(np.float32)
    y_patchtst = y_true + rng.normal(scale=0.08, size=y_true.shape).astype(np.float32)

    patchtst_cache = cache_dir / "patchtst_cache.npz"
    visual_cache = cache_dir / "visual_cache.npz"
    np.savez_compressed(
        patchtst_cache,
        sample_key=sample_key,
        split=split,
        h_ts=h_ts,
        y_patchtst=y_patchtst,
        y_true=y_true,
    )
    np.savez_compressed(
        visual_cache,
        sample_key=sample_key,
        h_vis=h_vis,
    )
    return patchtst_cache, visual_cache


def run_one_mode(tmp_dir: Path, patchtst_cache: Path, visual_cache: Path, mode: str) -> None:
    """函数功能：运行单个 fusion mode 的 mini train/eval，并验证产物。"""
    output_dir = tmp_dir / f"run_{mode}"
    cmd = [
        sys.executable,
        "-m",
        "visual_router_experiments.dual_branch_fusion.train_patchtst_visual_65k",
        "--data_subset",
        "synthetic_smoke",
        "--ts_model",
        "patchtst",
        "--visual_embedding_cache",
        str(visual_cache),
        "--patchtst_cache",
        str(patchtst_cache),
        "--fusion_mode",
        mode,
        "--epochs",
        "2",
        "--batch_size",
        "8",
        "--lr",
        "0.01",
        "--hidden_dim",
        "16",
        "--dropout",
        "0.0",
        "--residual_scale",
        "0.1",
        "--seed",
        "7",
        "--device",
        "cpu",
        "--output_dir",
        str(output_dir),
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    required_files = ("config.json", "metrics.json", "predictions.npz", "training_log.txt", "summary.md")
    for filename in required_files:
        path = output_dir / filename
        if not path.exists():
            raise AssertionError(f"{mode} 未写出 {filename}")

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "patchtst_mae",
        "patchtst_mse",
        "dual_branch_mae",
        "dual_branch_mse",
        "delta_mae_vs_patchtst",
        "delta_mse_vs_patchtst",
        "beats_patchtst_mae",
        "beats_patchtst_mse",
        "best_val_epoch",
        "best_val_loss",
        "test_checkpoint",
    ):
        if key not in metrics:
            raise AssertionError(f"{mode} metrics.json 缺少 {key}")
    if metrics["test_checkpoint"] != "best_validation_checkpoint":
        raise AssertionError(f"{mode} 未使用 best validation checkpoint")

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("feature_standardization", {}).get("enabled") is not True:
        raise AssertionError(f"{mode} config.json 未记录默认启用 train-only feature standardization")

    with np.load(output_dir / "predictions.npz", allow_pickle=False) as data:
        y_patchtst = data["y_patchtst"]
        y_fusion = data["y_fusion"]
        y_true = data["y_true"]
        sample_key = data["sample_key"]
    if y_patchtst.shape != y_true.shape or y_fusion.shape != y_true.shape:
        raise AssertionError(f"{mode} 预测 shape 不一致：patch={y_patchtst.shape} fusion={y_fusion.shape} true={y_true.shape}")
    if tuple(y_true.shape) != (8, 4, 2):
        raise AssertionError(f"{mode} test shape 漂移：{y_true.shape}")
    if len(sample_key) != 8:
        raise AssertionError(f"{mode} test sample_key 数量错误：{len(sample_key)}")

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    if "PatchTST MAE" not in summary or "Dual-branch MAE" not in summary or "delta_mae" not in summary:
        raise AssertionError(f"{mode} summary.md 未包含核心对比指标")
    if "best validation checkpoint" not in summary:
        raise AssertionError(f"{mode} summary.md 未明确 best-val checkpoint")
    print(f"通过：{mode} forward/mini-train/metrics/summary")


def main() -> None:
    """函数功能：执行四个原始 fusion mode 和新增 residual-safe mode 的 smoke。"""
    print("开始 PatchTST + Visual dual-branch 65k synthetic smoke")
    tmp_dir = Path(tempfile.mkdtemp(prefix="dual_branch_fusion_smoke_"))
    try:
        patchtst_cache, visual_cache = build_synthetic_cache(tmp_dir)
        for mode in FUSION_MODES:
            run_one_mode(tmp_dir, patchtst_cache, visual_cache, mode)
        print("完成：四个第一批 fusion mode 与 residual-safe mode 均通过 synthetic smoke")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
