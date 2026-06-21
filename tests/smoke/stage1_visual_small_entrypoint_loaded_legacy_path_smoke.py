#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P16j Visual small entrypoint loaded legacy path smoke。

输入：
    在 tempfile 内构造 tiny legacy VisualMLPRouter checkpoint payload，使用
    P16c precomputed visual feature fixture 和 P16d scaler JSON fixture，调用
    `scripts/run_stage1_visual_small.py` 的显式 loaded legacy path。

输出：
    标准输出打印中文检查日志；若 canonical run_dir、metadata、P16i helper、
    P16a adapter、sample_key 保序、softmax 权重或边界约束漂移则抛错。

关键约束：
    本 smoke 只使用 small fixture / tempfile tiny checkpoint，不读取真实
    checkpoint，不访问 `/data2`，不启动 ViT/transformers，不调用正式 streaming
    训练入口，也不改变 TimeFuse small entrypoint。
"""

from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage1_visual_small import DEFAULT_EXPERT_PREDICTIONS_JSON, DEFAULT_SAMPLE_MANIFEST_CSV  # noqa: E402


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
STREAMING_ENTRYPOINT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
VISUAL_FEATURES_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_scaler_small" / "raw_visual_features.csv"
SCALER_STATE_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_scaler_small" / "scaler_state.json"
EXPECTED_TEST_SAMPLE_KEYS = (
    "96_48_S::ETTh1::item0::ch0::win1",
    "96_48_S::weather::item8::ch0::win2",
)
EXPECTED_MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
EXPECTED_FEATURE_DIM = 8
HIDDEN_DIM = 11


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON artifact 并确认其为 object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    函数功能：
        基于 legacy module 形状生成 deterministic tiny state_dict。

    关键约束：
        这里只写 tempfile tiny checkpoint，不读取真实 checkpoint。
    """
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        tensor = torch.linspace(-0.05 + index * 0.01, 0.05 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_tiny_checkpoint(path: Path) -> None:
    """函数功能：创建 P16j CLI 使用的 tiny checkpoint payload。"""
    if str(path.resolve()).startswith("/data2/"):
        raise AssertionError(f"P16j smoke 不应向 /data2 写 checkpoint：{path}")
    module = importlib.import_module(LEGACY_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    model = router_cls(input_dim=EXPECTED_FEATURE_DIM, hidden_dim=HIDDEN_DIM, output_dim=len(EXPECTED_MODEL_COLUMNS), dropout=0.0)
    payload: Mapping[str, Any] = {
        "router_state_dict": build_fake_state_dict(model),
        "scaler_state": {
            "kind": "metadata_only",
            "mean": [0.0] * EXPECTED_FEATURE_DIM,
            "scale": [1.0] * EXPECTED_FEATURE_DIM,
        },
        "config": {
            "input_dim": EXPECTED_FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": len(EXPECTED_MODEL_COLUMNS),
            "dropout": 0.0,
            "payload_name": "p16j_tempfile_tiny_checkpoint",
        },
        "metadata": {
            "stage": "P16j",
            "source": "tempfile checkpoint payload smoke",
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }
    torch.save(dict(payload), path)


def assert_prediction_rows(path: Path) -> None:
    """函数功能：验证 prediction_rows.csv 存在且保持 manifest/test sample_key 顺序。"""
    if not path.is_file():
        raise AssertionError("prediction_rows.csv 未写入")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(row["sample_key"] for row in rows) != EXPECTED_TEST_SAMPLE_KEYS:
        raise AssertionError(f"prediction rows sample_key 顺序漂移：{rows}")
    for row in rows:
        for field in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse", "max_weight", "weight_entropy"):
            if not np.isfinite(float(row[field])):
                raise AssertionError(f"prediction row {field} 不是有限值：{row}")


def run_smoke() -> None:
    """函数功能：执行 P16j loaded legacy path smoke。"""
    print("开始 Stage 1 P16j Visual small loaded legacy path smoke")
    if not VISUAL_FEATURES_CSV.is_file() or not SCALER_STATE_JSON.is_file():
        raise AssertionError("P16j smoke 需要 P16d raw feature/scaler fixture")
    streaming_before = STREAMING_ENTRYPOINT.read_bytes()

    with tempfile.TemporaryDirectory(prefix="stage1_p16j_visual_loaded_legacy_") as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint_path = temp_root / "tiny_legacy_visual_mlp_payload.pt"
        output_dir = temp_root / "run_outputs"
        save_tiny_checkpoint(checkpoint_path)
        cmd = [
            sys.executable,
            str(ENTRYPOINT),
            "--sample-manifest-csv",
            str(DEFAULT_SAMPLE_MANIFEST_CSV),
            "--expert-predictions-json",
            str(DEFAULT_EXPERT_PREDICTIONS_JSON),
            "--output-dir",
            str(output_dir),
            "--split-name",
            "test",
            "--run-id",
            "p16j_visual_loaded_legacy_path_smoke",
            "--feature-source",
            "precomputed",
            "--visual-features-csv",
            str(VISUAL_FEATURES_CSV),
            "--scaler-state-json",
            str(SCALER_STATE_JSON),
            "--use-loaded-legacy-mlp",
            "--router-checkpoint-payload",
            str(checkpoint_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "P16j loaded legacy path entrypoint 返回码异常："
                f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        stdout_stderr = completed.stdout + completed.stderr
        for token in ("/data2", "ViTModel", "AutoImageProcessor", "transformers", "train_visual_router_online_streaming.py"):
            if token in stdout_stderr:
                raise AssertionError(f"P16j stdout/stderr 不应出现禁止 token：{token}")
        if "loaded_legacy_mlp" not in completed.stdout:
            raise AssertionError(f"P16j stdout 未记录 loaded_legacy_mlp 摘要：{completed.stdout}")
        print("通过：CLI loaded legacy path 完成，stdout/stderr 未触碰真实数据、ViT 或 streaming 入口")

        run_dir = output_dir / "p16j_visual_loaded_legacy_path_smoke"
        status = load_json(run_dir / "run_status.json")
        metadata = load_json(run_dir / "run_metadata.json")
        evaluation_summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")
        if status["status"] != "completed":
            raise AssertionError(f"run_status 未 completed：{status}")
        visual_metadata = metadata["visual_router"]
        expected_metadata = {
            "feature_source": "precomputed",
            "loaded_legacy_mlp": True,
            "scaler_enabled": True,
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
            "formal_visual_router_migration": False,
            "p16i_helper_used": True,
            "p16a_adapter_used": True,
        }
        for key, expected_value in expected_metadata.items():
            if visual_metadata.get(key) != expected_value:
                raise AssertionError(f"visual metadata {key} 异常：actual={visual_metadata.get(key)!r} metadata={visual_metadata}")
        if visual_metadata["checkpoint_payload_source"] != "explicit_small_fixture":
            raise AssertionError(f"checkpoint_payload_source 异常：{visual_metadata}")
        if visual_metadata["head_lineage"]["checkpoint_payload_path"] != str(checkpoint_path):
            raise AssertionError("checkpoint path 只应记录在 Runtime metadata/head_lineage 中")
        if visual_metadata["feature_lineage"]["visual_features_csv"]["path"] != str(VISUAL_FEATURES_CSV):
            raise AssertionError("feature_lineage 未记录 precomputed fixture")
        if visual_metadata["feature_schema"].get("transformed_by") != "LoadedFeatureScaler":
            raise AssertionError(f"feature_schema 未记录 LoadedFeatureScaler：{visual_metadata['feature_schema']}")
        print("通过：run_metadata 记录 loaded legacy、precomputed feature、scaler 和非正式迁移边界")

        metrics = evaluation_summary["metrics"]
        for field in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse"):
            if field not in metrics or not np.isfinite(float(metrics[field])):
                raise AssertionError(f"evaluation summary metric 异常：{field} -> {metrics}")
        if tuple(evaluation_summary["model_columns"]) != EXPECTED_MODEL_COLUMNS:
            raise AssertionError(f"model_columns 未与 ExpertBatch 对齐：{evaluation_summary['model_columns']}")
        assert_prediction_rows(run_dir / "predictions" / "prediction_rows.csv")
        print("通过：evaluation summary 和 prediction rows 完整，sample_key/model_columns 对齐")

    if STREAMING_ENTRYPOINT.read_bytes() != streaming_before:
        raise AssertionError("P16j smoke 不应修改 train_visual_router_online_streaming.py")
    print("完成：Stage 1 P16j Visual small loaded legacy path smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
