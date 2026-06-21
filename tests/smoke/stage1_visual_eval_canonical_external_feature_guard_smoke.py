#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P17c Visual canonical eval external feature/scaler guard smoke。

输入：
    使用 P17a canonical eval fixture 链路，在临时目录构造 tiny checkpoint payload，
    并在仓库根目录临时目录构造 synthetic external feature CSV / scaler JSON。

输出：
    标准输出打印中文检查日志；若默认 fixture path、非授权 external path、授权
    external path 或 `/data2` helper policy 不符合 P17c 要求，则抛出 AssertionError。

关键约束：
    默认 smoke 不读取真实 `/data2` feature/scaler/checkpoint，不启动 ViT、训练、
    full-scale 或 streaming 训练入口。
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

from time_router.runtime import authorize_visual_eval_feature_path, authorize_visual_eval_scaler_path  # noqa: E402


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
SAMPLE_MANIFEST_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "sample_manifest.csv"
EXPERT_PREDICTIONS_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_real_derived_small" / "expert_predictions.json"
FIXTURE_FEATURES_CSV = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_precomputed_small" / "visual_embeddings.csv"
FIXTURE_SCALER_JSON = REPO_ROOT / "tests" / "fixtures" / "stage1_visual_scaler_small" / "scaler_state.json"
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
        tensor = torch.linspace(-0.02 + index * 0.01, 0.06 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_tiny_checkpoint(path: Path) -> None:
    """函数功能：创建 entrypoint 可 strict load 的 tiny legacy VisualMLPRouter payload。"""
    if str(path.resolve()).startswith("/data2/"):
        raise AssertionError("P17c guard smoke 不应向 /data2 写 checkpoint")
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
            "payload_name": "p17c_tempfile_tiny_checkpoint",
        },
        "metadata": {
            "stage": "P17c",
            "source": "tempfile tiny checkpoint for external feature guard smoke",
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
        },
    }
    torch.save(dict(payload), path)


def copy_feature_fixture_to_external(path: Path) -> None:
    """
    函数功能：
        在非 fixture/tmp 路径创建 synthetic feature CSV，内容覆盖 manifest 样本。
    """
    with FIXTURE_FEATURES_CSV.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise AssertionError("fixture feature CSV 缺少表头")
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_external_scaler_json(path: Path) -> None:
    """函数功能：在非 fixture/tmp 路径创建 synthetic scaler state JSON。"""
    payload = {
        "scaler_schema_version": "stage1_visual_feature_scaler_v1",
        "feature_columns": [f"feature_{index}" for index in range(EXPECTED_FEATURE_DIM)],
        "mean": [0.1 * index for index in range(EXPECTED_FEATURE_DIM)],
        "scale": [1.0 + 0.1 * index for index in range(EXPECTED_FEATURE_DIM)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_cmd(*, checkpoint_path: Path, feature_csv: Path, output_dir: Path, run_id: str) -> list[str]:
    """函数功能：生成 P17c canonical eval CLI 基础命令。"""
    return [
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


def assert_visual_metadata_common(visual_metadata: Mapping[str, Any]) -> None:
    """函数功能：检查 P17c 仍保持 eval-only dry-run 边界字段。"""
    expected = {
        "feature_source": "precomputed",
        "loads_real_vit": False,
        "training_started": False,
        "formal_training_migration": False,
    }
    for field, value in expected.items():
        if visual_metadata[field] != value:
            raise AssertionError(f"{field} metadata 异常：{visual_metadata}")
    if visual_metadata["scaler_fit_performed"] is not False:
        raise AssertionError(f"P17c 不应执行 scaler fit：{visual_metadata}")


def assert_fixture_feature_no_scaler_success() -> None:
    """
    函数功能：
        验证默认 fixture feature CSV + no scaler 不传 external allow 仍成功。
    """
    with tempfile.TemporaryDirectory(prefix="stage1_p17c_fixture_feature_") as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint_path = temp_root / "tiny_legacy_visual_mlp_payload.pt"
        output_dir = temp_root / "run_outputs"
        save_tiny_checkpoint(checkpoint_path)

        completed = run_entrypoint(
            base_cmd(
                checkpoint_path=checkpoint_path,
                feature_csv=FIXTURE_FEATURES_CSV,
                output_dir=output_dir,
                run_id="p17c_fixture_feature_no_scaler",
            )
        )
        if completed.returncode != 0:
            raise AssertionError(f"fixture feature/no scaler 应成功：stdout={completed.stdout}\nstderr={completed.stderr}")
        visual_metadata = load_json(output_dir / "p17c_fixture_feature_no_scaler" / "run_metadata.json")["visual_router"]
        assert_visual_metadata_common(visual_metadata)
        if visual_metadata["allow_external_feature_path"] is not False:
            raise AssertionError(f"fixture feature 不应开启 external allow：{visual_metadata}")
        if visual_metadata["feature_path_policy"] != "default_fixture_or_tmp_feature":
            raise AssertionError(f"fixture feature policy 异常：{visual_metadata}")
        if visual_metadata["scaler_enabled"] is not False:
            raise AssertionError(f"no scaler path 应标记 scaler_enabled=false：{visual_metadata}")
    print("通过：默认 fixture feature CSV + no scaler 成功，metadata 记录 external allow=false")


def assert_unallowed_external_feature_fails() -> None:
    """
    函数功能：
        验证非 fixture/tmp feature CSV 未显式授权时在 provider 读取前失败。
    """
    with tempfile.TemporaryDirectory(prefix="stage1_p17c_checkpoint_") as checkpoint_dir, tempfile.TemporaryDirectory(
        prefix=".stage1_p17c_external_feature_",
        dir=REPO_ROOT,
    ) as external_dir:
        checkpoint_path = Path(checkpoint_dir) / "tiny_legacy_visual_mlp_payload.pt"
        output_dir = Path(checkpoint_dir) / "run_outputs"
        feature_csv = Path(external_dir) / "synthetic_visual_embeddings.csv"
        save_tiny_checkpoint(checkpoint_path)
        copy_feature_fixture_to_external(feature_csv)

        completed = run_entrypoint(
            base_cmd(
                checkpoint_path=checkpoint_path,
                feature_csv=feature_csv,
                output_dir=output_dir,
                run_id="p17c_unallowed_external_feature",
            )
        )
        if completed.returncode == 0:
            raise AssertionError("非 fixture/tmp feature CSV 未开启 allow-external-feature-path 时应失败")
        combined = completed.stdout + completed.stderr
        if "--allow-external-feature-path" not in combined:
            raise AssertionError(f"失败信息应提示 external feature 授权：{combined}")

        missing_feature_csv = Path(external_dir) / "missing_visual_embeddings.csv"
        missing_completed = run_entrypoint(
            base_cmd(
                checkpoint_path=checkpoint_path,
                feature_csv=missing_feature_csv,
                output_dir=output_dir,
                run_id="p17c_unallowed_missing_external_feature",
            )
        )
        if missing_completed.returncode == 0:
            raise AssertionError("不存在的 external feature CSV 未授权时也应先 fail-fast")
        missing_combined = missing_completed.stdout + missing_completed.stderr
        if "--allow-external-feature-path" not in missing_combined:
            raise AssertionError(f"未授权 missing external feature 应先提示授权而不是检查文件存在：{missing_combined}")
    print("通过：非 fixture/tmp feature CSV 未授权时 fail-fast，错误信息提示 allow-external-feature-path")


def assert_allowed_external_feature_success() -> None:
    """函数功能：验证非 fixture/tmp feature CSV 显式授权后成功并写入 policy metadata。"""
    with tempfile.TemporaryDirectory(prefix="stage1_p17c_checkpoint_") as checkpoint_dir, tempfile.TemporaryDirectory(
        prefix=".stage1_p17c_external_feature_",
        dir=REPO_ROOT,
    ) as external_dir:
        checkpoint_path = Path(checkpoint_dir) / "tiny_legacy_visual_mlp_payload.pt"
        output_dir = Path(checkpoint_dir) / "run_outputs"
        feature_csv = Path(external_dir) / "synthetic_visual_embeddings.csv"
        save_tiny_checkpoint(checkpoint_path)
        copy_feature_fixture_to_external(feature_csv)

        cmd = base_cmd(
            checkpoint_path=checkpoint_path,
            feature_csv=feature_csv,
            output_dir=output_dir,
            run_id="p17c_allowed_external_feature",
        )
        cmd.extend(["--allow-external-feature-path", "--feature-path-label", "synthetic:repo-root-temp"])
        completed = run_entrypoint(cmd)
        if completed.returncode != 0:
            raise AssertionError(f"授权 external feature 应成功：stdout={completed.stdout}\nstderr={completed.stderr}")
        visual_metadata = load_json(output_dir / "p17c_allowed_external_feature" / "run_metadata.json")["visual_router"]
        assert_visual_metadata_common(visual_metadata)
        if visual_metadata["feature_path_policy"] != "explicit_external_feature_authorized":
            raise AssertionError(f"external feature policy 异常：{visual_metadata}")
        if visual_metadata["allow_external_feature_path"] is not True:
            raise AssertionError(f"external feature 应记录 allow=true：{visual_metadata}")
        if visual_metadata["feature_path_label"] != "synthetic:repo-root-temp":
            raise AssertionError(f"feature path label 未写入 metadata：{visual_metadata}")
    print("通过：非 fixture/tmp feature CSV 授权后成功，metadata 记录 explicit_external_feature_authorized")


def assert_external_scaler_guard() -> None:
    """函数功能：验证非 fixture/tmp scaler JSON 未授权失败、授权后成功且不 fit。"""
    with tempfile.TemporaryDirectory(prefix="stage1_p17c_checkpoint_") as checkpoint_dir, tempfile.TemporaryDirectory(
        prefix=".stage1_p17c_external_scaler_",
        dir=REPO_ROOT,
    ) as external_dir:
        checkpoint_path = Path(checkpoint_dir) / "tiny_legacy_visual_mlp_payload.pt"
        output_dir = Path(checkpoint_dir) / "run_outputs"
        scaler_json = Path(external_dir) / "synthetic_scaler_state.json"
        save_tiny_checkpoint(checkpoint_path)
        write_external_scaler_json(scaler_json)

        unallowed_cmd = base_cmd(
            checkpoint_path=checkpoint_path,
            feature_csv=FIXTURE_FEATURES_CSV,
            output_dir=output_dir,
            run_id="p17c_unallowed_external_scaler",
        )
        unallowed_cmd.extend(["--scaler-state-json", str(scaler_json)])
        unallowed = run_entrypoint(unallowed_cmd)
        if unallowed.returncode == 0:
            raise AssertionError("非 fixture/tmp scaler JSON 未开启 allow-external-scaler-path 时应失败")
        if "--allow-external-scaler-path" not in (unallowed.stdout + unallowed.stderr):
            raise AssertionError(f"失败信息应提示 external scaler 授权：{unallowed.stdout}{unallowed.stderr}")

        allowed_cmd = base_cmd(
            checkpoint_path=checkpoint_path,
            feature_csv=FIXTURE_FEATURES_CSV,
            output_dir=output_dir,
            run_id="p17c_allowed_external_scaler",
        )
        allowed_cmd.extend(
            [
                "--scaler-state-json",
                str(scaler_json),
                "--allow-external-scaler-path",
                "--scaler-path-label",
                "synthetic:repo-root-temp",
            ]
        )
        allowed = run_entrypoint(allowed_cmd)
        if allowed.returncode != 0:
            raise AssertionError(f"授权 external scaler 应成功：stdout={allowed.stdout}\nstderr={allowed.stderr}")
        visual_metadata = load_json(output_dir / "p17c_allowed_external_scaler" / "run_metadata.json")["visual_router"]
        assert_visual_metadata_common(visual_metadata)
        if visual_metadata["scaler_enabled"] is not True:
            raise AssertionError(f"授权 scaler path 应标记 scaler_enabled=true：{visual_metadata}")
        if visual_metadata["scaler_fit_performed"] is not False:
            raise AssertionError(f"授权 scaler path 不应 fit：{visual_metadata}")
        if visual_metadata["scaler_path_policy"] != "explicit_external_scaler_authorized":
            raise AssertionError(f"external scaler policy 异常：{visual_metadata}")
        if visual_metadata["allow_external_scaler_path"] is not True:
            raise AssertionError(f"external scaler 应记录 allow=true：{visual_metadata}")
    print("通过：非 fixture/tmp scaler JSON 未授权失败、授权后成功且 metadata 记录 fit_performed=false")


def assert_data2_guard_policy() -> None:
    """
    函数功能：
        直接测试 `/data2` feature/scaler path guard，不创建文件、不读取内容。
    """
    data2_feature = Path("/data2/syh/Time/not_read_by_p17c_guard/precomputed_visual_features.csv")
    data2_scaler = Path("/data2/syh/Time/not_read_by_p17c_guard/scaler_state.json")

    for path, helper, flag_name in (
        (data2_feature, authorize_visual_eval_feature_path, "--allow-external-feature-path"),
        (data2_scaler, authorize_visual_eval_scaler_path, "--allow-external-scaler-path"),
    ):
        try:
            if helper is authorize_visual_eval_feature_path:
                helper(path, repo_root=REPO_ROOT, allow_external_feature_path=False)
            else:
                helper(path, repo_root=REPO_ROOT, allow_external_scaler_path=False)
        except ValueError as exc:
            if flag_name not in str(exc):
                raise AssertionError(f"/data2 未授权错误信息应提示 {flag_name}：{exc}") from exc
        else:
            raise AssertionError(f"/data2 path 缺少 {flag_name} 时应失败：{path}")

    feature_policy = authorize_visual_eval_feature_path(
        data2_feature,
        repo_root=REPO_ROOT,
        allow_external_feature_path=True,
        path_label="manual:data2-feature",
    )
    scaler_policy = authorize_visual_eval_scaler_path(
        data2_scaler,
        repo_root=REPO_ROOT,
        allow_external_scaler_path=True,
        path_label="manual:data2-scaler",
    )
    if feature_policy.path_policy != "explicit_external_feature_data2_authorized" or not feature_policy.is_external_data2_path:
        raise AssertionError(f"/data2 feature policy 异常：{feature_policy}")
    if scaler_policy.path_policy != "explicit_external_scaler_data2_authorized" or not scaler_policy.is_external_data2_path:
        raise AssertionError(f"/data2 scaler policy 异常：{scaler_policy}")
    print("通过：/data2 feature/scaler guard 需要显式授权，授权后仅 path policy 通过且未读取文件")


def maybe_manual_external_feature_dry_run() -> None:
    """
    函数功能：
        可选环境变量驱动的 manual evaluation-only dry-run，默认跳过。

    关键约束：
        只有用户显式提供真实 checkpoint + feature CSV 时才运行；若路径位于 `/data2`，
        自动追加对应 allow flag，但仍只做 evaluation-only dry-run。
    """
    checkpoint_payload = os.environ.get("STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD")
    feature_csv = os.environ.get("STAGE1_VISUAL_REAL_FEATURE_CSV")
    scaler_state_json = os.environ.get("STAGE1_VISUAL_REAL_SCALER_STATE_JSON")
    if not checkpoint_payload or not feature_csv:
        print("跳过：未设置 STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD/STAGE1_VISUAL_REAL_FEATURE_CSV manual dry-run 环境变量")
        return

    with tempfile.TemporaryDirectory(prefix="stage1_p17c_manual_external_feature_") as temp_dir:
        output_dir = Path(temp_dir) / "run_outputs"
        cmd = base_cmd(
            checkpoint_path=Path(checkpoint_payload),
            feature_csv=Path(feature_csv),
            output_dir=output_dir,
            run_id="p17c_manual_external_feature_dry_run",
        )
        cmd.extend(
            [
                "--allow-real-checkpoint",
                "--allow-external-feature-path",
                "--checkpoint-path-label",
                "env:STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD",
                "--feature-path-label",
                "env:STAGE1_VISUAL_REAL_FEATURE_CSV",
            ]
        )
        if str(Path(checkpoint_payload).resolve()).startswith("/data2/"):
            cmd.append("--allow-external-checkpoint-path")
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
        completed = run_entrypoint(cmd)
        if completed.returncode != 0:
            raise AssertionError(f"manual external feature dry-run 失败：stdout={completed.stdout}\nstderr={completed.stderr}")
        print(f"通过：manual external feature dry-run 输出 {output_dir / 'p17c_manual_external_feature_dry_run'}")


def run_smoke() -> None:
    """函数功能：执行 P17c external feature/scaler guard smoke。"""
    print("开始 Stage 1 P17c Visual eval external feature/scaler guard smoke")
    assert_fixture_feature_no_scaler_success()
    assert_unallowed_external_feature_fails()
    assert_allowed_external_feature_success()
    assert_external_scaler_guard()
    assert_data2_guard_policy()
    maybe_manual_external_feature_dry_run()
    print("完成：Stage 1 P17c Visual eval external feature/scaler guard smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
