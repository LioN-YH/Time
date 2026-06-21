#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P17d Visual canonical eval real-artifact manual dry-run contract smoke。

输入：
    默认不读取真实 `/data2` 内容；未设置环境变量时只验证 manual dry-run 会按预期
    skip。测试内部会在仓库受控临时目录构造 synthetic real-artifact checkpoint、
    precomputed visual feature CSV 和可选 scaler state JSON。

输出：
    标准输出打印中文检查日志；若 manual real-artifact dry-run 标记、路径授权、
    canonical run_dir、contract fail-fast 或 eval-only 边界不符合 P17d 要求，则抛出
    AssertionError。

关键约束：
    本 smoke 不启动训练、不启动 ViT/transformers、不启动 full-scale、不修改正式
    streaming 训练入口，也不自动搜索 `/data2`。
"""

from __future__ import annotations

import csv
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


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
SAMPLE_MANIFEST_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "sample_manifest.csv"
EXPERT_PREDICTIONS_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "expert_predictions.json"
FIXTURE_FEATURES_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_precomputed_small" / "visual_embeddings.csv"
EXPECTED_MODEL_COLUMNS = ("DLinear", "PatchTST", "CrossFormer")
EXPECTED_FEATURE_DIM = 8
HIDDEN_DIM = 11


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON object artifact，供 metadata 与 summary 断言使用。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """函数功能：构造 deterministic synthetic checkpoint state_dict，不读取真实 checkpoint。"""
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        tensor = torch.linspace(-0.04 + index * 0.01, 0.04 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_synthetic_checkpoint(path: Path, *, input_dim: int = EXPECTED_FEATURE_DIM) -> None:
    """
    函数功能：
        在非 fixture/tmp 路径创建 legacy VisualMLPRouter checkpoint payload。

    输入：
        input_dim 可用于 negative case 构造 checkpoint config 与 feature dim 不一致。
    """
    if str(path.resolve()).startswith("/data2/"):
        raise AssertionError("P17d smoke 不应向 /data2 写 checkpoint")
    module = importlib.import_module(LEGACY_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    model = router_cls(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=len(EXPECTED_MODEL_COLUMNS), dropout=0.0)
    payload: Mapping[str, Any] = {
        "router_state_dict": build_fake_state_dict(model),
        "config": {
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": len(EXPECTED_MODEL_COLUMNS),
            "dropout": 0.0,
            "payload_name": "p17d_synthetic_real_artifact_checkpoint",
        },
        "metadata": {
            "stage": "P17d",
            "source": "synthetic repo-temp real-artifact manual dry-run smoke",
            "loads_real_checkpoint": True,
            "loads_real_vit": False,
        },
    }
    torch.save(dict(payload), path)


def copy_feature_fixture(path: Path, *, drop_last_sample: bool = False) -> None:
    """
    函数功能：
        复制 fixture feature CSV 到非 fixture/tmp 路径，必要时删除最后一个样本触发
        sample_key coverage fail-fast。
    """
    with FIXTURE_FEATURES_CSV.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise AssertionError("fixture feature CSV 缺少表头")
    if drop_last_sample:
        rows = rows[:-1]
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_scaler_json(path: Path) -> None:
    """函数功能：创建 synthetic LoadedFeatureScaler state，验证只 transform 不 fit。"""
    payload = {
        "scaler_schema_version": "stage1_visual_feature_scaler_v1",
        "feature_columns": [f"feature_{index}" for index in range(EXPECTED_FEATURE_DIM)],
        "mean": [0.05 * index for index in range(EXPECTED_FEATURE_DIM)],
        "scale": [1.0 + 0.05 * index for index in range(EXPECTED_FEATURE_DIM)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_test_sample_keys() -> tuple[str, ...]:
    """函数功能：从 SampleManifest fixture 读取 test split 的 canonical ordered sample_keys。"""
    with SAMPLE_MANIFEST_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(str(row["sample_key"]) for row in rows if row["split"] == "test")


def base_cmd(
    *,
    checkpoint_path: Path,
    feature_csv: Path,
    output_dir: Path,
    run_id: str,
    scaler_state_json: Path | None = None,
) -> list[str]:
    """函数功能：生成 P17d canonical eval manual real-artifact CLI 命令。"""
    cmd = [
        sys.executable,
        str(ENTRYPOINT),
        "--sample-manifest-csv",
        str(SAMPLE_MANIFEST_CSV),
        "--expert-predictions-json",
        str(EXPERT_PREDICTIONS_JSON),
        "--visual-features-csv",
        str(feature_csv),
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
        "--allow-real-checkpoint",
        "--allow-external-feature-path",
        "--checkpoint-path-label",
        "synthetic:repo-root-temp",
        "--feature-path-label",
        "synthetic:repo-root-temp",
        "--manual-real-artifact-dryrun",
    ]
    if scaler_state_json is not None:
        cmd.extend(
            [
                "--scaler-state-json",
                str(scaler_state_json),
                "--allow-external-scaler-path",
                "--scaler-path-label",
                "synthetic:repo-root-temp",
            ]
        )
    return cmd


def run_entrypoint(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """函数功能：运行 canonical eval entrypoint 并捕获 stdout/stderr。"""
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def append_data2_allow_flags(cmd: list[str], *, checkpoint_path: Path, feature_csv: Path, scaler_state_json: Path | None) -> None:
    """
    函数功能：
        对用户显式传入的 `/data2` artifact 自动追加对应 allow flag。

    关键约束：
        这里只根据已给路径授权，不搜索 `/data2`，metadata/stdout 仍会记录 allow 状态。
    """
    if str(checkpoint_path.resolve()).startswith("/data2/"):
        cmd.append("--allow-external-checkpoint-path")
    if str(feature_csv.resolve()).startswith("/data2/") and "--allow-external-feature-path" not in cmd:
        cmd.append("--allow-external-feature-path")
    if scaler_state_json is not None and str(scaler_state_json.resolve()).startswith("/data2/") and "--allow-external-scaler-path" not in cmd:
        cmd.append("--allow-external-scaler-path")


def maybe_manual_env_dry_run() -> None:
    """
    函数功能：
        使用用户显式环境变量执行真实 artifact manual dry-run；环境变量不完整时 skip。
    """
    checkpoint_payload = os.environ.get("STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD")
    feature_csv = os.environ.get("STAGE1_VISUAL_REAL_FEATURE_CSV")
    scaler_state_json = os.environ.get("STAGE1_VISUAL_REAL_SCALER_STATE_JSON")
    sample_manifest_csv = os.environ.get("STAGE1_VISUAL_REAL_SAMPLE_MANIFEST_CSV", str(SAMPLE_MANIFEST_CSV))
    expert_predictions_json = os.environ.get("STAGE1_VISUAL_REAL_EXPERT_PREDICTIONS_JSON", str(EXPERT_PREDICTIONS_JSON))
    output_dir_env = os.environ.get("STAGE1_VISUAL_REAL_OUTPUT_DIR")
    if not checkpoint_payload or not feature_csv:
        print("跳过：未设置完整 STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD/STAGE1_VISUAL_REAL_FEATURE_CSV manual dry-run 环境变量")
        return

    output_root = Path(output_dir_env) if output_dir_env else Path(tempfile.mkdtemp(prefix="stage1_p17d_manual_env_output_"))
    cmd = [
        sys.executable,
        str(ENTRYPOINT),
        "--sample-manifest-csv",
        sample_manifest_csv,
        "--expert-predictions-json",
        expert_predictions_json,
        "--visual-features-csv",
        feature_csv,
        "--router-checkpoint-payload",
        checkpoint_payload,
        "--output-dir",
        str(output_root),
        "--run-id",
        "p17d_manual_real_artifact_env_dryrun",
        "--config-name",
        "96_48_S",
        "--split-name",
        "test",
        "--strict-checkpoint-load",
        "--allow-real-checkpoint",
        "--allow-external-feature-path",
        "--checkpoint-path-label",
        "env:STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD",
        "--feature-path-label",
        "env:STAGE1_VISUAL_REAL_FEATURE_CSV",
        "--manual-real-artifact-dryrun",
    ]
    scaler_path = Path(scaler_state_json) if scaler_state_json else None
    if scaler_state_json:
        cmd.extend(
            [
                "--scaler-state-json",
                scaler_state_json,
                "--allow-external-scaler-path",
                "--scaler-path-label",
                "env:STAGE1_VISUAL_REAL_SCALER_STATE_JSON",
            ]
        )
    append_data2_allow_flags(cmd, checkpoint_path=Path(checkpoint_payload), feature_csv=Path(feature_csv), scaler_state_json=scaler_path)
    completed = run_entrypoint(cmd)
    if completed.returncode != 0:
        raise AssertionError(f"环境变量 manual real-artifact dry-run 失败：stdout={completed.stdout}\nstderr={completed.stderr}")
    run_dir = output_root / "p17d_manual_real_artifact_env_dryrun"
    metadata = load_json(run_dir / "run_metadata.json")["visual_router"]
    if metadata["manual_real_artifact_dryrun"] is not True:
        raise AssertionError(f"manual env dry-run 未记录 manual 标记：{metadata}")
    print(f"通过：环境变量 manual real-artifact dry-run 输出 {run_dir}")


def assert_synthetic_real_artifact_success() -> None:
    """函数功能：验证 synthetic real checkpoint + external feature + external scaler 成功写 canonical run_dir。"""
    with tempfile.TemporaryDirectory(prefix=".stage1_p17d_real_artifact_", dir=REPO_ROOT) as external_dir:
        external_root = Path(external_dir)
        checkpoint_path = external_root / "synthetic_real_visual_mlp_payload.pt"
        feature_csv = external_root / "synthetic_real_visual_embeddings.csv"
        scaler_json = external_root / "synthetic_real_scaler_state.json"
        output_dir = external_root / "run_outputs"
        save_synthetic_checkpoint(checkpoint_path)
        copy_feature_fixture(feature_csv)
        write_scaler_json(scaler_json)

        completed = run_entrypoint(
            base_cmd(
                checkpoint_path=checkpoint_path,
                feature_csv=feature_csv,
                scaler_state_json=scaler_json,
                output_dir=output_dir,
                run_id="p17d_synthetic_real_artifact_dryrun",
            )
        )
        if completed.returncode != 0:
            raise AssertionError(f"synthetic real-artifact manual dry-run 应成功：stdout={completed.stdout}\nstderr={completed.stderr}")
        run_dir = output_dir / "p17d_synthetic_real_artifact_dryrun"
        metadata = load_json(run_dir / "run_metadata.json")
        visual_metadata = metadata["visual_router"]
        expected_fields = {
            "entrypoint": "visual_eval_canonical",
            "manual_real_artifact_dryrun": True,
            "loads_real_checkpoint": True,
            "feature_source": "precomputed",
            "allow_external_feature_path": True,
            "scaler_enabled": True,
            "allow_external_scaler_path": True,
            "loads_real_vit": False,
            "training_started": False,
            "formal_training_migration": False,
        }
        for field, expected in expected_fields.items():
            if visual_metadata[field] != expected:
                raise AssertionError(f"{field} metadata 异常：expected={expected} actual={visual_metadata.get(field)} metadata={visual_metadata}")
        if visual_metadata["checkpoint_path_policy"] != "explicit_real_checkpoint_authorized":
            raise AssertionError(f"checkpoint policy 异常：{visual_metadata}")
        if visual_metadata["feature_path_policy"] != "explicit_external_feature_authorized":
            raise AssertionError(f"feature policy 异常：{visual_metadata}")
        if visual_metadata["scaler_path_policy"] != "explicit_external_scaler_authorized":
            raise AssertionError(f"scaler policy 异常：{visual_metadata}")

        summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")
        for metric_name, metric_value in summary["metrics"].items():
            if not isinstance(metric_value, (int, float)) or not torch.isfinite(torch.tensor(float(metric_value))):
                raise AssertionError(f"metrics 必须 finite：{metric_name}={metric_value}")

        with (run_dir / "predictions" / "prediction_rows.csv").open("r", encoding="utf-8", newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        prediction_keys = tuple(row["sample_key"] for row in prediction_rows)
        expected_keys = expected_test_sample_keys()
        if prediction_keys != expected_keys:
            raise AssertionError(f"prediction_rows sample_key 未保序：{prediction_keys}")
    print("通过：synthetic real-artifact manual dry-run 成功，canonical run_dir 与 metadata contract 正确")


def assert_checkpoint_input_dim_mismatch_fails() -> None:
    """函数功能：验证 checkpoint config input_dim 与 FeatureBatch feature dim 不一致时 fail-fast。"""
    with tempfile.TemporaryDirectory(prefix=".stage1_p17d_dim_mismatch_", dir=REPO_ROOT) as external_dir:
        external_root = Path(external_dir)
        checkpoint_path = external_root / "bad_input_dim_visual_mlp_payload.pt"
        feature_csv = external_root / "synthetic_real_visual_embeddings.csv"
        output_dir = external_root / "run_outputs"
        save_synthetic_checkpoint(checkpoint_path, input_dim=EXPECTED_FEATURE_DIM + 1)
        copy_feature_fixture(feature_csv)
        completed = run_entrypoint(
            base_cmd(
                checkpoint_path=checkpoint_path,
                feature_csv=feature_csv,
                output_dir=output_dir,
                run_id="p17d_checkpoint_input_dim_mismatch",
            )
        )
        if completed.returncode == 0:
            raise AssertionError("checkpoint input_dim 与 feature dim 不一致时应失败")
        combined = completed.stdout + completed.stderr
        if "checkpoint config input_dim" not in combined:
            raise AssertionError(f"失败信息应指出 checkpoint input_dim contract：{combined}")
    print("通过：checkpoint input_dim mismatch 触发 fail-fast")


def assert_feature_missing_sample_key_fails() -> None:
    """函数功能：验证 feature CSV 缺少 manifest sample_key 时 fail-fast。"""
    with tempfile.TemporaryDirectory(prefix=".stage1_p17d_missing_feature_", dir=REPO_ROOT) as external_dir:
        external_root = Path(external_dir)
        checkpoint_path = external_root / "synthetic_real_visual_mlp_payload.pt"
        feature_csv = external_root / "missing_sample_visual_embeddings.csv"
        output_dir = external_root / "run_outputs"
        save_synthetic_checkpoint(checkpoint_path)
        copy_feature_fixture(feature_csv, drop_last_sample=True)
        completed = run_entrypoint(
            base_cmd(
                checkpoint_path=checkpoint_path,
                feature_csv=feature_csv,
                output_dir=output_dir,
                run_id="p17d_feature_missing_sample_key",
            )
        )
        if completed.returncode == 0:
            raise AssertionError("feature CSV 缺少 manifest sample_key 时应失败")
        combined = completed.stdout + completed.stderr
        if "sample_key" not in combined:
            raise AssertionError(f"失败信息应指出 sample_key coverage contract：{combined}")
    print("通过：feature CSV 缺少 sample_key 触发 fail-fast")


def run_smoke() -> None:
    """函数功能：执行 P17d manual real-artifact contract smoke。"""
    print("开始 Stage 1 P17d Visual eval real-artifact manual dry-run contract smoke")
    maybe_manual_env_dry_run()
    assert_synthetic_real_artifact_success()
    assert_checkpoint_input_dim_mismatch_fails()
    assert_feature_missing_sample_key_fails()
    print("完成：Stage 1 P17d Visual eval real-artifact manual dry-run contract smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
