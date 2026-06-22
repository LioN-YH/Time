#!/usr/bin/env python3
"""
文件功能：
    65k PatchTST baseline + fixed visual embedding 双分支融合训练/评估入口。

输入：
    已有 PatchTST frozen cache 与视觉 embedding cache，均为 npz 或 npz shard 目录。

输出：
    `config.json`、`metrics.json`、`predictions.npz`、`training_log.txt` 和 `summary.md`。

关键约束：
    本入口不训练 PatchTST、不生成图像、不运行 ViT；只在同一批对齐 sample_key 和同一
    train/val/test split 上评估 PatchTST baseline 并训练轻量融合头。
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from visual_router_experiments.dual_branch_fusion.cache_dataset import (
    DualBranchTensorDataset,
    align_patchtst_and_visual_cache,
    split_indices,
)
from visual_router_experiments.dual_branch_fusion.fusion_heads import build_fusion_head
from visual_router_experiments.dual_branch_fusion.metrics import build_comparison_metrics, compute_mae, compute_mse


def display_time() -> str:
    """函数功能：生成输出文件中的本地时间字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def set_seed(seed: int) -> None:
    """函数功能：固定 Python、numpy 和 torch 随机种子，提升 smoke 与小实验可复现性。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """函数功能：以 UTF-8 JSON 写出轻量配置和指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_loader(
    dataset: DualBranchTensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    """函数功能：构造带固定 generator 的 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, generator=generator)


def run_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """函数功能：执行一个 train 或 eval epoch，并返回按样本加权的平均 loss。"""
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    total_count = 0
    for h_ts, h_vis, y_patchtst, y_true in loader:
        h_ts = h_ts.to(device)
        h_vis = h_vis.to(device)
        y_patchtst = y_patchtst.to(device)
        y_true = y_true.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        y_fusion = model(h_ts, h_vis, y_patchtst)
        loss = criterion(y_fusion, y_true)
        if is_train:
            loss.backward()
            optimizer.step()
        batch_size = int(h_ts.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size
    if total_count <= 0:
        raise ValueError("DataLoader 没有样本")
    return total_loss / total_count


@torch.no_grad()
def predict(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """函数功能：对一个 split 生成 dual-branch 预测。"""
    model.eval()
    preds: List[np.ndarray] = []
    for h_ts, h_vis, y_patchtst, _ in loader:
        y_fusion = model(h_ts.to(device), h_vis.to(device), y_patchtst.to(device))
        preds.append(y_fusion.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def format_summary(metrics: Dict[str, object], config: Dict[str, object], history: List[Dict[str, float]]) -> str:
    """函数功能：生成验收要求的 Markdown summary。"""
    rows = [
        ("PatchTST MAE", metrics["patchtst_mae"]),
        ("PatchTST MSE", metrics["patchtst_mse"]),
        ("Dual-branch MAE", metrics["dual_branch_mae"]),
        ("Dual-branch MSE", metrics["dual_branch_mse"]),
        ("delta_mae", metrics["delta_mae_vs_patchtst"]),
        ("delta_mse", metrics["delta_mse_vs_patchtst"]),
    ]
    table = ["| 指标 | 数值 |", "| --- | --- |"]
    for name, value in rows:
        table.append(f"| {name} | {float(value):.8f} |")
    table.append(f"| beats_patchtst_mae | {metrics['beats_patchtst_mae']} |")
    table.append(f"| beats_patchtst_mse | {metrics['beats_patchtst_mse']} |")

    history_lines = ["| epoch | train_loss | val_loss |", "| --- | --- | --- |"]
    for item in history:
        history_lines.append(f"| {int(item['epoch'])} | {item['train_loss']:.8f} | {item['val_loss']:.8f} |")

    return "\n".join(
        [
            "# PatchTST + Visual Dual-Branch 65k Summary",
            "",
            f"- 生成时间：{display_time()}",
            f"- fusion_mode：{config['fusion_mode']}",
            f"- data_subset：{config['data_subset']}",
            f"- train/val/test split：{config['train_split']} / {config['val_split']} / {config['test_split']}",
            f"- 视觉编码：fixed cache，不生成图像，不运行 ViT",
            f"- PatchTST：frozen cache baseline，不在本入口重新训练",
            "",
            "## 对比指标",
            "",
            *table,
            "",
            "## 训练损失",
            "",
            *history_lines,
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    """函数功能：解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train/evaluate PatchTST + fixed visual dual-branch fusion on 65k cache.")
    parser.add_argument("--data_subset", default="65k")
    parser.add_argument("--ts_model", default="patchtst")
    parser.add_argument("--visual_embedding_cache", type=Path, required=True)
    parser.add_argument("--patchtst_cache", type=Path, required=True)
    parser.add_argument(
        "--fusion_mode",
        choices=["feature_concat", "film", "residual_feature", "visual_residual", "pred_gate"],
        required=True,
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """函数功能：执行 cache 对齐、baseline 评估、融合头训练、test 评估与产物写出。"""
    args = parse_args()
    if args.ts_model.lower() != "patchtst":
        raise ValueError(f"本入口当前只支持 PatchTST，actual={args.ts_model}")
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空；如需覆盖请传 --overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    cache = align_patchtst_and_visual_cache(
        patchtst_cache_path=args.patchtst_cache,
        visual_embedding_cache_path=args.visual_embedding_cache,
        required_splits=[args.train_split, args.val_split, args.test_split],
    )
    train_idx = split_indices(cache, args.train_split)
    val_idx = split_indices(cache, args.val_split)
    test_idx = split_indices(cache, args.test_split)

    train_loader = make_loader(
        DualBranchTensorDataset(cache, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        DualBranchTensorDataset(cache, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = make_loader(
        DualBranchTensorDataset(cache, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    y_patchtst_test = cache.y_patchtst[test_idx]
    y_true_test = cache.y_true[test_idx]
    patchtst_baseline = {
        "patchtst_mae": compute_mae(y_patchtst_test, y_true_test),
        "patchtst_mse": compute_mse(y_patchtst_test, y_true_test),
    }

    output_dim = int(np.prod(cache.y_true.shape[1:]))
    model = build_fusion_head(
        mode=args.fusion_mode,
        ts_dim=int(cache.h_ts.shape[1]),
        visual_dim=int(cache.h_vis.shape[1]),
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.MSELoss()

    history: List[Dict[str, float]] = []
    log_lines = [
        f"start_time={display_time()}",
        f"train_samples={len(train_idx)} val_samples={len(val_idx)} test_samples={len(test_idx)}",
        f"patchtst_mae={patchtst_baseline['patchtst_mae']:.8f} patchtst_mse={patchtst_baseline['patchtst_mse']:.8f}",
    ]
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model=model, loader=train_loader, criterion=criterion, device=device, optimizer=optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model=model, loader=val_loader, criterion=criterion, device=device, optimizer=None)
        row = {"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)}
        history.append(row)
        log_lines.append(f"epoch={epoch} train_loss={train_loss:.8f} val_loss={val_loss:.8f}")

    y_dual_test = predict(model=model, loader=test_loader, device=device)
    y_dual_test = y_dual_test.reshape(y_true_test.shape)
    metrics = build_comparison_metrics(y_patchtst=y_patchtst_test, y_dual_branch=y_dual_test, y_true=y_true_test)
    metrics.update(
        {
            "fusion_mode": args.fusion_mode,
            "seed": int(args.seed),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        }
    )

    config = {
        "created_at": display_time(),
        "data_subset": args.data_subset,
        "ts_model": args.ts_model,
        "visual_embedding_cache": str(args.visual_embedding_cache),
        "patchtst_cache": str(args.patchtst_cache),
        "fusion_mode": args.fusion_mode,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "device": str(device),
        "ts_dim": int(cache.h_ts.shape[1]),
        "visual_dim": int(cache.h_vis.shape[1]),
        "target_shape": list(cache.y_true.shape[1:]),
        "aligned_sample_count": int(len(cache.sample_key)),
    }

    write_json(output_dir / "config.json", config)
    write_json(output_dir / "metrics.json", metrics)
    np.savez_compressed(
        output_dir / "predictions.npz",
        sample_key=cache.sample_key[test_idx],
        split=cache.split[test_idx],
        y_patchtst=y_patchtst_test.astype(np.float32, copy=False),
        y_fusion=y_dual_test.astype(np.float32, copy=False),
        y_true=y_true_test.astype(np.float32, copy=False),
    )
    (output_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(format_summary(metrics, config, history), encoding="utf-8")
    print(f"完成：输出已写入 {output_dir}")
    print(f"PatchTST MAE={metrics['patchtst_mae']:.8f} Dual MAE={metrics['dual_branch_mae']:.8f}")


if __name__ == "__main__":
    main()
