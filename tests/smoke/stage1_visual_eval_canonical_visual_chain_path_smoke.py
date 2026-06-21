#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P20a Visual canonical eval visual-chain dry-run path smoke。

输入：
    复用 P13b small manifest / expert JSON、P19a raw window JSON fixture，并在
    tempfile 内构造与 dry-run pooled feature 维度匹配的 tiny VisualMLPRouter
    checkpoint payload。

输出：
    标准输出打印中文检查日志；若 canonical run_dir、metadata、prediction rows
    保序、import 边界或 fail-fast 负例漂移，则抛出 AssertionError。

关键约束：
    默认路径不导入 transformers，不加载真实 ViT，不访问 `/data2`，不启动训练，
    不调用或修改 `train_visual_router_online_streaming.py`。
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


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py"
P17A_SMOKE = REPO_ROOT / "tests" / "smoke" / "stage1_visual_eval_canonical_thin_slice_smoke.py"
STREAMING_ENTRYPOINT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
SAMPLE_MANIFEST_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "sample_manifest.csv"
EXPERT_PREDICTIONS_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "expert_predictions.json"
RAW_WINDOWS_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_feature_chain_dryrun" / "raw_windows.json"
EXPECTED_TEST_SAMPLE_KEYS = (
    "96_48_S::ETTh1::item0::ch0::win1",
    "96_48_S::weather::item8::ch0::win2",
)
EXPECTED_MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
FEATURE_DIM = 4
HIDDEN_DIM = 7


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON artifact 并确认其为 object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """函数功能：基于 legacy VisualMLPRouter shape 构造 deterministic tiny state_dict。"""
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        tensor = torch.linspace(-0.02 + index * 0.01, 0.04 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_tiny_checkpoint(path: Path) -> None:
    """函数功能：创建 visual-chain dry-run CLI 使用的 tiny checkpoint payload。"""
    module = importlib.import_module(LEGACY_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    model = router_cls(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, output_dim=len(EXPECTED_MODEL_COLUMNS), dropout=0.0)
    payload: Mapping[str, Any] = {
        "router_state_dict": build_fake_state_dict(model),
        "config": {
            "input_dim": FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": len(EXPECTED_MODEL_COLUMNS),
            "dropout": 0.0,
            "payload_name": "p20a_tempfile_tiny_visual_chain_checkpoint",
        },
        "metadata": {
            "stage": "P20a",
            "source": "tempfile checkpoint payload smoke",
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }
    torch.save(dict(payload), path)


def base_visual_chain_cmd(*, checkpoint_path: Path, output_dir: Path, run_id: str) -> list[str]:
    """函数功能：生成 P20a visual-chain dry-run entrypoint 基础命令。"""
    return [
        sys.executable,
        str(ENTRYPOINT),
        "--sample-manifest-csv",
        str(SAMPLE_MANIFEST_CSV),
        "--expert-predictions-json",
        str(EXPERT_PREDICTIONS_JSON),
        "--router-checkpoint-payload",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--config-name",
        "96_48_S",
        "--split-name",
        "test",
        "--feature-source",
        "visual-chain-dryrun",
        "--raw-window-json",
        str(RAW_WINDOWS_JSON),
        "--strict-checkpoint-load",
    ]


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """函数功能：运行子命令并捕获 stdout/stderr 供边界断言。"""
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def assert_p17a_precomputed_regression() -> None:
    """函数功能：确认默认 precomputed path 的 P17a smoke 仍通过。"""
    completed = run_command([sys.executable, str(P17A_SMOKE)])
    if completed.returncode != 0:
        raise AssertionError(
            "P17a precomputed path regression smoke 失败："
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    print("通过：P17a 默认 precomputed canonical eval smoke 仍通过")


def assert_no_forbidden_runtime_tokens(completed: subprocess.CompletedProcess[str]) -> None:
    """函数功能：确认默认 visual-chain smoke 未触碰真实 ViT、训练入口或 `/data2`。"""
    combined = completed.stdout + completed.stderr
    for token in ("ViTModel", "AutoImageProcessor", "train_visual_router_online_streaming.py", "/data2"):
        if token in combined:
            raise AssertionError(f"P20a stdout/stderr 不应出现禁止 token：{token}\n{combined}")
    if "transformers" in sys.modules:
        raise AssertionError("P20a smoke 默认路径不应导入 transformers")


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


def assert_visual_chain_run(run_dir: Path, checkpoint_path: Path) -> None:
    """函数功能：检查 P20a canonical run_dir artifact 和 visual_router metadata。"""
    status = load_json(run_dir / "run_status.json")
    metadata = load_json(run_dir / "run_metadata.json")
    evaluation_summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")
    if status["status"] != "completed" or status["current_stage"] != "visual_eval_canonical":
        raise AssertionError(f"run_status 异常：{status}")
    visual_metadata = metadata["visual_router"]
    expected_metadata = {
        "feature_source": "visual-chain-dryrun",
        "visual_chain_enabled": True,
        "vit_provider_mode": "injected-fake",
        "loads_real_vit": False,
        "visual_chain_runner": "VisualFeatureChainRunner",
        "encoder_provider": "VisualVitEncoderProvider",
        "training_started": False,
        "formal_training_migration": False,
        "full_scale_run": False,
        "loaded_legacy_mlp": True,
    }
    for key, expected_value in expected_metadata.items():
        if visual_metadata.get(key) != expected_value:
            raise AssertionError(f"visual metadata {key} 异常：actual={visual_metadata.get(key)!r} metadata={visual_metadata}")
    if metadata["inputs"]["visual_features_csv"] is not None:
        raise AssertionError("visual-chain dry-run 不应记录 precomputed visual_features_csv 输入")
    if metadata["inputs"]["raw_window_json"]["path"] != str(RAW_WINDOWS_JSON):
        raise AssertionError("raw_window_json 输入引用异常")
    if visual_metadata["head_lineage"]["checkpoint_payload_path"] != str(checkpoint_path):
        raise AssertionError("checkpoint path 只应记录在 Runtime metadata/head_lineage 中")
    if "run_dir" in json.dumps(visual_metadata["feature_lineage"], ensure_ascii=False):
        raise AssertionError("feature provider / runner lineage 不应接收 run_dir")
    if tuple(evaluation_summary["model_columns"]) != EXPECTED_MODEL_COLUMNS:
        raise AssertionError(f"model_columns 未与 ExpertBatch 对齐：{evaluation_summary['model_columns']}")
    assert_prediction_rows(run_dir / "predictions" / "prediction_rows.csv")


def assert_visual_chain_success(temp_root: Path, checkpoint_path: Path) -> None:
    """函数功能：运行 visual-chain-dryrun 正向 CLI 并检查 artifact。"""
    output_dir = temp_root / "run_outputs"
    completed = run_command(
        base_visual_chain_cmd(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            run_id="p20a_visual_chain_dryrun_smoke",
        )
    )
    if completed.returncode != 0:
        raise AssertionError(f"P20a visual-chain dry-run 应成功：stdout={completed.stdout}\nstderr={completed.stderr}")
    assert_no_forbidden_runtime_tokens(completed)
    assert_visual_chain_run(output_dir / "p20a_visual_chain_dryrun_smoke", checkpoint_path)
    print("通过：visual-chain-dryrun canonical run_dir、metadata、summary 和 prediction rows 均成立")


def assert_missing_raw_window_json_fails(temp_root: Path, checkpoint_path: Path) -> None:
    """函数功能：验证 visual-chain-dryrun 缺少 --raw-window-json 时 fail-fast。"""
    cmd = base_visual_chain_cmd(checkpoint_path=checkpoint_path, output_dir=temp_root / "missing_raw", run_id="missing_raw")
    index = cmd.index("--raw-window-json")
    del cmd[index : index + 2]
    completed = run_command(cmd)
    if completed.returncode == 0:
        raise AssertionError("缺少 --raw-window-json 时应失败")
    combined = completed.stdout + completed.stderr
    if "--raw-window-json" not in combined:
        raise AssertionError(f"缺少 raw-window-json 的失败信息应包含参数名：{combined}")
    print("通过：raw-window-json 缺失时 fail-fast")


def assert_missing_manifest_key_fails(temp_root: Path, checkpoint_path: Path) -> None:
    """函数功能：验证 raw-window fixture 缺少 manifest sample_key 时 fail-fast。"""
    with RAW_WINDOWS_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    del payload["windows"][EXPECTED_TEST_SAMPLE_KEYS[-1]]
    broken_raw = temp_root / "raw_windows_missing_key.json"
    broken_raw.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = base_visual_chain_cmd(checkpoint_path=checkpoint_path, output_dir=temp_root / "missing_key", run_id="missing_key")
    cmd[cmd.index(str(RAW_WINDOWS_JSON))] = str(broken_raw)
    completed = run_command(cmd)
    if completed.returncode == 0:
        raise AssertionError("raw-window-json 缺少 manifest sample_key 时应失败")
    combined = completed.stdout + completed.stderr
    if "raw_window_json 缺少 manifest sample_key" not in combined:
        raise AssertionError(f"缺少 manifest key 的失败信息异常：{combined}")
    print("通过：raw-window-json 缺少 manifest sample_key 时 fail-fast")


def run_smoke() -> None:
    """函数功能：执行 P20a visual-chain canonical eval path smoke。"""
    print("开始 Stage 1 P20a Visual canonical eval visual-chain path smoke")
    before_streaming_bytes = STREAMING_ENTRYPOINT.read_bytes()
    assert_p17a_precomputed_regression()
    with tempfile.TemporaryDirectory(prefix="stage1_p20a_visual_chain_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint_path = temp_root / "tiny_visual_chain_mlp_payload.pt"
        save_tiny_checkpoint(checkpoint_path)
        assert_visual_chain_success(temp_root, checkpoint_path)
        assert_missing_raw_window_json_fails(temp_root, checkpoint_path)
        assert_missing_manifest_key_fails(temp_root, checkpoint_path)
    if STREAMING_ENTRYPOINT.read_bytes() != before_streaming_bytes:
        raise AssertionError("P20a smoke 不应修改 train_visual_router_online_streaming.py")
    print("完成：Stage 1 P20a Visual canonical eval visual-chain path smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
