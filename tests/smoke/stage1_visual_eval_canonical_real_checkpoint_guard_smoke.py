#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P17b Visual canonical eval real-checkpoint guard smoke。

输入：
    使用 P17a canonical eval fixture 链路，并在临时目录构造 tiny checkpoint payload；
    另用 runtime guard helper 直接测试 `/data2` path policy。

输出：
    标准输出打印中文检查日志；若默认 tiny path、非授权真实 path 或 `/data2`
    双重授权策略不符合 P17b 要求，则抛出 AssertionError。

关键约束：
    默认 smoke 不读取真实 checkpoint，不创建或读取 `/data2` 文件，不启动 ViT、训练、
    full-scale 或 streaming 训练入口。
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_router.runtime import authorize_visual_eval_checkpoint_path  # noqa: E402


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
SAMPLE_MANIFEST_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "sample_manifest.csv"
EXPERT_PREDICTIONS_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "expert_predictions.json"
VISUAL_FEATURES_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_precomputed_small" / "visual_embeddings.csv"
EXPECTED_MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
EXPECTED_FEATURE_DIM = 8
HIDDEN_DIM = 11


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON object artifact，便于检查 metadata。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """函数功能：构造 deterministic tiny checkpoint state_dict，不读取真实 checkpoint。"""
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        tensor = torch.linspace(-0.03 + index * 0.01, 0.05 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_tiny_checkpoint(path: Path) -> None:
    """函数功能：创建 entrypoint 可 strict load 的 tiny legacy VisualMLPRouter payload。"""
    if str(path.resolve()).startswith("/data2/"):
        raise AssertionError("P17b guard smoke 不应向 /data2 写 checkpoint")
    module = importlib.import_module(LEGACY_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    model = router_cls(input_dim=EXPECTED_FEATURE_DIM, hidden_dim=HIDDEN_DIM, output_dim=len(EXPECTED_MODEL_COLUMNS), dropout=0.0)
    payload: Mapping[str, Any] = {
        "router_state_dict": build_fake_state_dict(model),
        "config": {
            "input_dim": EXPECTED_FEATURE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": len(EXPECTED_MODEL_COLUMNS),
            "dropout": 0.0,
            "payload_name": "p17b_tempfile_tiny_checkpoint",
        },
        "metadata": {
            "stage": "P17b",
            "source": "tempfile tiny checkpoint guard smoke",
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }
    torch.save(dict(payload), path)


def base_cmd(*, checkpoint_path: Path, output_dir: Path, run_id: str) -> list[str]:
    """函数功能：生成 P17b canonical eval CLI 基础命令。"""
    return [
        sys.executable,
        str(ENTRYPOINT),
        "--sample-manifest-csv",
        str(SAMPLE_MANIFEST_CSV),
        "--expert-predictions-json",
        str(EXPERT_PREDICTIONS_JSON),
        "--visual-features-csv",
        str(VISUAL_FEATURES_CSV),
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
        "--strict-checkpoint-load",
    ]


def run_entrypoint(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """函数功能：运行 canonical eval entrypoint 并捕获输出供断言。"""
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def assert_default_tmp_checkpoint_success() -> None:
    """
    函数功能：
        验证默认 tempfile tiny checkpoint 不传 allow-real-checkpoint 也能成功。
    """
    with tempfile.TemporaryDirectory(prefix="stage1_p17b_tmp_checkpoint_") as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint_path = temp_root / "tiny_legacy_visual_mlp_payload.pt"
        output_dir = temp_root / "run_outputs"
        save_tiny_checkpoint(checkpoint_path)

        completed = run_entrypoint(
            base_cmd(checkpoint_path=checkpoint_path, output_dir=output_dir, run_id="p17b_default_tmp_checkpoint")
        )
        if completed.returncode != 0:
            raise AssertionError(f"默认 tempfile tiny checkpoint 应成功：stdout={completed.stdout}\nstderr={completed.stderr}")
        run_dir = output_dir / "p17b_default_tmp_checkpoint"
        metadata = load_json(run_dir / "run_metadata.json")
        visual_metadata = metadata["visual_router"]
        if visual_metadata["loads_real_checkpoint"] is not False:
            raise AssertionError(f"tempfile tiny checkpoint 不应标记真实 checkpoint：{visual_metadata}")
        if visual_metadata["allow_real_checkpoint"] is not False:
            raise AssertionError(f"默认路径不应开启 allow_real_checkpoint：{visual_metadata}")
        if visual_metadata["allow_external_checkpoint_path"] is not False:
            raise AssertionError(f"默认路径不应开启 allow_external_checkpoint_path：{visual_metadata}")
        if visual_metadata["checkpoint_path_policy"] != "default_fixture_or_tmp_checkpoint":
            raise AssertionError(f"默认 checkpoint policy 异常：{visual_metadata}")
        if visual_metadata["checkpoint_payload_source"] != "explicit_fixture_or_tmp_path":
            raise AssertionError(f"默认 checkpoint payload source 异常：{visual_metadata}")
        for field, expected in (
            ("loads_real_vit", False),
            ("training_started", False),
            ("formal_training_migration", False),
        ):
            if visual_metadata[field] != expected:
                raise AssertionError(f"{field} 边界字段异常：{visual_metadata}")
    print("通过：默认 tempfile tiny checkpoint 成功，metadata 标记 loads_real_checkpoint=false")


def assert_unallowed_non_fixture_checkpoint_fails() -> None:
    """
    函数功能：
        验证非 fixture/tmp checkpoint 未显式授权时在 torch.load 前失败。
    """
    with tempfile.TemporaryDirectory(prefix=".stage1_p17b_non_fixture_", dir=REPO_ROOT) as temp_dir:
        non_fixture_root = Path(temp_dir)
        checkpoint_path = non_fixture_root / "synthetic_real_checkpoint_payload.pt"
        save_tiny_checkpoint(checkpoint_path)
        output_dir = non_fixture_root / "run_outputs"
        completed = run_entrypoint(
            base_cmd(checkpoint_path=checkpoint_path, output_dir=output_dir, run_id="p17b_unallowed_checkpoint")
        )
        if completed.returncode == 0:
            raise AssertionError("非 fixture/tmp checkpoint 未开启 allow-real-checkpoint 时应失败")
        combined = completed.stdout + completed.stderr
        if "--allow-real-checkpoint" not in combined:
            raise AssertionError(f"失败信息应提示必须显式 allow-real-checkpoint：{combined}")
    print("通过：非 fixture/tmp checkpoint 未授权时 fail-fast，错误信息提示 allow-real-checkpoint")


def assert_data2_guard_policy() -> None:
    """
    函数功能：
        直接测试 `/data2` checkpoint path guard，不创建文件、不调用 torch.load。
    """
    data2_checkpoint = Path("/data2/syh/Time/not_read_by_p17b_guard/real_visual_mlp_checkpoint.pt")
    try:
        authorize_visual_eval_checkpoint_path(
            data2_checkpoint,
            repo_root=REPO_ROOT,
            allow_real_checkpoint=True,
            allow_external_checkpoint_path=False,
        )
    except ValueError as exc:
        if "--allow-external-checkpoint-path" not in str(exc):
            raise AssertionError(f"/data2 未授权错误信息应提示 allow-external-checkpoint-path：{exc}") from exc
    else:
        raise AssertionError("/data2 checkpoint 缺少 allow-external-checkpoint-path 时应失败")

    policy = authorize_visual_eval_checkpoint_path(
        data2_checkpoint,
        repo_root=REPO_ROOT,
        allow_real_checkpoint=True,
        allow_external_checkpoint_path=True,
    )
    if policy.loads_real_checkpoint is not True:
        raise AssertionError(f"/data2 双重授权后应标记真实 checkpoint：{policy}")
    if policy.is_external_data2_path is not True:
        raise AssertionError(f"/data2 path 分类异常：{policy}")
    if policy.checkpoint_path_policy != "explicit_real_checkpoint_external_data2_authorized":
        raise AssertionError(f"/data2 path policy 异常：{policy}")
    print("通过：/data2 checkpoint guard 需要双重授权，授权后仅 path policy 通过且未读取文件")


def maybe_manual_real_checkpoint_dry_run() -> None:
    """
    函数功能：
        可选环境变量驱动的 manual real-checkpoint dry-run，默认跳过。

    关键约束：
        只有用户显式提供真实 checkpoint、feature CSV 和可选 scaler state 时才运行；仍不
        启动 ViT、训练或 full-scale。
    """
    checkpoint_payload = os.environ.get("STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD")
    feature_csv = os.environ.get("STAGE1_VISUAL_REAL_FEATURE_CSV")
    scaler_state_json = os.environ.get("STAGE1_VISUAL_REAL_SCALER_STATE_JSON")
    if not checkpoint_payload or not feature_csv:
        print("跳过：未设置 STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD/STAGE1_VISUAL_REAL_FEATURE_CSV manual dry-run 环境变量")
        return

    with tempfile.TemporaryDirectory(prefix="stage1_p17b_manual_real_checkpoint_") as temp_dir:
        output_dir = Path(temp_dir) / "run_outputs"
        cmd = base_cmd(
            checkpoint_path=Path(checkpoint_payload),
            output_dir=output_dir,
            run_id="p17b_manual_real_checkpoint_dry_run",
        )
        cmd[cmd.index(str(VISUAL_FEATURES_CSV))] = feature_csv
        cmd.extend(["--allow-real-checkpoint", "--checkpoint-path-label", "env:STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD"])
        if str(Path(checkpoint_payload).resolve()).startswith("/data2/"):
            cmd.append("--allow-external-checkpoint-path")
        if scaler_state_json:
            cmd.extend(["--scaler-state-json", scaler_state_json])
        completed = run_entrypoint(cmd)
        if completed.returncode != 0:
            raise AssertionError(f"manual real-checkpoint dry-run 失败：stdout={completed.stdout}\nstderr={completed.stderr}")
        print(f"通过：manual real-checkpoint dry-run 输出 {output_dir / 'p17b_manual_real_checkpoint_dry_run'}")


def run_smoke() -> None:
    """函数功能：执行 P17b real-checkpoint guard smoke。"""
    print("开始 Stage 1 P17b Visual eval real-checkpoint guard smoke")
    assert_default_tmp_checkpoint_success()
    assert_unallowed_non_fixture_checkpoint_fails()
    assert_data2_guard_policy()
    maybe_manual_real_checkpoint_dry_run()
    print("完成：Stage 1 P17b Visual eval real-checkpoint guard smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
