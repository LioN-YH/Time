#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P21a Visual canonical eval small launcher/config pack smoke。

输入：
    使用 P21a 两份 JSON config，通过 Python launcher 间接调用 canonical eval
    entrypoint，并在 tempfile 中生成运行输出。

输出：
    标准输出打印中文检查日志；若 run_dir、metadata、dry-print 或 safety guard
    漂移，则抛出 AssertionError。

关键约束：
    本 smoke 不访问 `/data2`，不启动训练或 full-scale，不默认加载真实 ViT，不默认导入
    transformers，不修改 streaming 训练入口。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAUNCHER = REPO_ROOT / "scripts" / "run_stage1_visual_eval_small.py"
PRECOMPUTED_CONFIG = REPO_ROOT / "configs" / "stage1" / "visual_eval_small_precomputed.json"
VISUAL_CHAIN_CONFIG = REPO_ROOT / "configs" / "stage1" / "visual_eval_small_visual_chain.json"
STREAMING_ENTRYPOINT = REPO_ROOT / "visual_router_experiments" / "stage1_vali_test_router" / "train_visual_router_online_streaming.py"


def load_json(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON artifact 并确认其为 object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """函数功能：运行子命令并捕获 stdout/stderr 供断言。"""
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def assert_success(completed: subprocess.CompletedProcess[str], *, label: str) -> None:
    """函数功能：确认 launcher 子命令成功且未打印禁止 token。"""
    if completed.returncode != 0:
        raise AssertionError(f"{label} 失败：stdout={completed.stdout}\nstderr={completed.stderr}")
    combined = completed.stdout + completed.stderr
    for token in ("/data2", "ViTModel", "AutoImageProcessor", "train_visual_router_online_streaming.py"):
        if token in combined:
            raise AssertionError(f"{label} stdout/stderr 不应出现禁止 token：{token}\n{combined}")
    if "transformers" in sys.modules:
        raise AssertionError(f"{label} 默认路径不应导入 transformers")


def assert_run_dir(
    run_dir: Path,
    *,
    feature_source: str,
    visual_chain_enabled: bool,
    loads_real_vit: bool,
) -> None:
    """函数功能：检查 canonical run_dir status 和 visual metadata。"""
    status = load_json(run_dir / "run_status.json")
    metadata = load_json(run_dir / "run_metadata.json")
    if status["status"] != "completed" or status["current_stage"] != "visual_eval_canonical":
        raise AssertionError(f"run_status 异常：{status}")
    visual_metadata = metadata["visual_router"]
    expected = {
        "entrypoint": "visual_eval_canonical",
        "feature_source": feature_source,
        "visual_chain_enabled": visual_chain_enabled,
        "loads_real_vit": loads_real_vit,
        "loads_real_checkpoint": False,
        "training_started": False,
        "formal_training_migration": False,
        "full_scale_run": False,
    }
    for key, expected_value in expected.items():
        if visual_metadata.get(key) != expected_value:
            raise AssertionError(f"visual metadata {key} 异常：actual={visual_metadata.get(key)!r} metadata={visual_metadata}")
    if feature_source == "visual-chain-dryrun" and visual_metadata.get("vit_provider_mode") != "injected-fake":
        raise AssertionError(f"visual-chain 默认应使用 injected-fake ViT：{visual_metadata}")


def assert_precomputed_mode(temp_root: Path) -> None:
    """函数功能：验证 --mode precomputed 正向运行。"""
    output_dir = temp_root / "precomputed_outputs"
    run_id = "p21a_precomputed"
    completed = run_command(
        [
            sys.executable,
            str(LAUNCHER),
            "--config-json",
            str(PRECOMPUTED_CONFIG),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--mode",
            "precomputed",
        ]
    )
    assert_success(completed, label="P21a precomputed mode")
    assert_run_dir(output_dir / run_id, feature_source="precomputed", visual_chain_enabled=False, loads_real_vit=False)
    if "feature_source=precomputed run_dir=" not in completed.stdout:
        raise AssertionError(f"launcher stdout 未打印 precomputed run_dir：{completed.stdout}")
    print("通过：--mode precomputed small canonical eval 成功")


def assert_visual_chain_mode(temp_root: Path) -> None:
    """函数功能：验证 --mode visual-chain 正向运行。"""
    output_dir = temp_root / "visual_chain_outputs"
    run_id = "p21a_visual_chain"
    completed = run_command(
        [
            sys.executable,
            str(LAUNCHER),
            "--config-json",
            str(VISUAL_CHAIN_CONFIG),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--mode",
            "visual-chain",
        ]
    )
    assert_success(completed, label="P21a visual-chain mode")
    assert_run_dir(
        output_dir / run_id,
        feature_source="visual-chain-dryrun",
        visual_chain_enabled=True,
        loads_real_vit=False,
    )
    if "feature_source=visual-chain-dryrun run_dir=" not in completed.stdout:
        raise AssertionError(f"launcher stdout 未打印 visual-chain run_dir：{completed.stdout}")
    print("通过：--mode visual-chain small canonical eval 成功")


def assert_both_mode(temp_root: Path) -> None:
    """函数功能：验证 --mode both 连续生成两个互不冲突的 run_dir。"""
    output_dir = temp_root / "both_outputs"
    completed = run_command(
        [
            sys.executable,
            str(LAUNCHER),
            "--output-dir",
            str(output_dir),
            "--run-id",
            "p21a_both",
            "--mode",
            "both",
        ]
    )
    assert_success(completed, label="P21a both mode")
    precomputed_run = output_dir / "p21a_both_precomputed"
    visual_chain_run = output_dir / "p21a_both_visual_chain"
    if precomputed_run == visual_chain_run or not precomputed_run.is_dir() or not visual_chain_run.is_dir():
        raise AssertionError(f"both mode 未生成两个不冲突 run_dir：{completed.stdout}")
    assert_run_dir(precomputed_run, feature_source="precomputed", visual_chain_enabled=False, loads_real_vit=False)
    assert_run_dir(visual_chain_run, feature_source="visual-chain-dryrun", visual_chain_enabled=True, loads_real_vit=False)
    print("通过：--mode both 连续生成两个 canonical run_dir 且 run_id 不冲突")


def assert_dry_print_command(temp_root: Path) -> None:
    """函数功能：验证 --dry-print-command 只打印 canonical command，不创建 run_dir。"""
    output_dir = temp_root / "dry_print_outputs"
    run_id = "p21a_dry_print"
    completed = run_command(
        [
            sys.executable,
            str(LAUNCHER),
            "--config-json",
            str(PRECOMPUTED_CONFIG),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--mode",
            "precomputed",
            "--dry-print-command",
        ]
    )
    assert_success(completed, label="P21a dry-print")
    if "canonical_command=" not in completed.stdout or str(REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py") not in completed.stdout:
        raise AssertionError(f"dry-print 未打印 canonical command：{completed.stdout}")
    if (output_dir / run_id).exists():
        raise AssertionError("--dry-print-command 不应创建 canonical run_dir")
    print("通过：--dry-print-command 只打印 canonical command 且不创建 run_dir")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """函数功能：写入安全负例临时 config。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_safety_negative(temp_root: Path) -> None:
    """函数功能：验证 /data2、training_started 和 full_scale_run 负例 fail-fast。"""
    base_config = load_json(PRECOMPUTED_CONFIG)
    cases = [
        ("data2_path", {"sample_manifest_csv": "/data2/not_allowed/sample_manifest.csv"}, "/data2"),
        ("training_started", {"safety": {**base_config["safety"], "training_started": True}}, "training_started"),
        ("full_scale_run", {"safety": {**base_config["safety"], "full_scale_run": True}}, "full_scale_run"),
    ]
    for case_name, patch, expected_token in cases:
        bad_config = {**base_config, **patch}
        bad_path = temp_root / f"{case_name}.json"
        write_json(bad_path, bad_config)
        completed = run_command(
            [
                sys.executable,
                str(LAUNCHER),
                "--config-json",
                str(bad_path),
                "--output-dir",
                str(temp_root / f"{case_name}_outputs"),
                "--run-id",
                case_name,
                "--mode",
                "precomputed",
            ]
        )
        if completed.returncode == 0:
            raise AssertionError(f"{case_name} 负例应 fail-fast")
        combined = completed.stdout + completed.stderr
        if expected_token not in combined:
            raise AssertionError(f"{case_name} 负例错误信息应包含 {expected_token!r}：{combined}")
    print("通过：/data2、training_started、full_scale_run safety 负例均 fail-fast")


def run_smoke() -> None:
    """函数功能：执行 P21a launcher/config pack smoke。"""
    print("开始 Stage 1 P21a Visual eval small launcher smoke")
    before_streaming_bytes = STREAMING_ENTRYPOINT.read_bytes()
    with tempfile.TemporaryDirectory(prefix="stage1_p21a_visual_eval_small_") as temp_dir:
        temp_root = Path(temp_dir)
        assert_precomputed_mode(temp_root)
        assert_visual_chain_mode(temp_root)
        assert_both_mode(temp_root)
        assert_dry_print_command(temp_root)
        assert_safety_negative(temp_root)
    if STREAMING_ENTRYPOINT.read_bytes() != before_streaming_bytes:
        raise AssertionError("P21a smoke 不应修改 train_visual_router_online_streaming.py")
    print("完成：Stage 1 P21a Visual eval small launcher smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
