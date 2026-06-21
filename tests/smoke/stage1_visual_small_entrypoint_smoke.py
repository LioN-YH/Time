#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P15c Visual-specific small canonical entrypoint smoke。

输入：
    使用仓库内 P13b real-derived small manifest/expert JSON 和 P14b Visual mock
    history window fixture，在 tempfile 内调用 `scripts/run_stage1_visual_small.py`。

输出：
    标准输出打印中文检查日志；若 CLI、canonical run_dir、evaluation summary、
    prediction rows、VisualMockFeatureProvider 或 smoke-only MLP adapter contract 漂移则抛错。

关键约束：
    不访问 `/data2`，不启动训练/pressure/full-scale，不调用正式 Visual/TimeFuse
    entrypoint，不读取真实 checkpoint，不启动 ViT，不修改 generic 或 TimeFuse small CLI。
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_stage1_visual_small import (  # noqa: E402
    DEFAULT_EXPERT_PREDICTIONS_JSON,
    DEFAULT_HISTORY_WINDOWS_JSON,
    DEFAULT_SAMPLE_MANIFEST_CSV,
    JsonExpertSmallProvider,
    SmokeOnlyVisualMLPAdapter,
    build_loaded_smoke_mlp,
    load_history_windows_json,
    load_sample_manifest_csv,
)
from time_router.evaluation import EvaluationInputAdapter  # noqa: E402
from time_router.features import VisualMockFeatureProvider  # noqa: E402


ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_small.py"
GENERIC_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_canonical_small.py"
TIMEFUSE_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_timefuse_small.py"
EXPECTED_TEST_SAMPLE_KEYS = (
    "96_48_S::ETTh1::item0::ch0::win1",
    "96_48_S::weather::item8::ch0::win2",
)
CANONICAL_SUBDIRS = ("inputs", "indexes", "predictions", "evaluation", "checkpoints", "logs")


def load_json(path: Path) -> dict[str, object]:
    """函数功能：读取 JSON artifact 并确认其为 object。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} 不是 JSON object")
    return payload


def fail_torch_load(*args: object, **kwargs: object) -> object:
    """函数功能：阻断真实 checkpoint 读取。"""
    raise AssertionError(f"P15c smoke 不应调用 torch.load 读取 checkpoint：args={args} kwargs={kwargs}")


def assert_no_forbidden_static_references() -> None:
    """
    函数功能：
        静态检查 P15c entrypoint 未引用禁止的正式入口、Bash launcher、真实 ViT 或 `/data2`。
    """
    source = ENTRYPOINT.read_text(encoding="utf-8")
    forbidden_tokens = (
        "train_visual_router_online_streaming",
        "train_timefuse_fusor_streaming",
        "launch_timefuse_fusor_full_scale",
        "ViTModel",
        "AutoImageProcessor",
        "subprocess",
        "bash",
        "nohup",
        "tmux",
    )
    for token in forbidden_tokens:
        if token in source:
            raise AssertionError(f"P15c entrypoint 不应包含禁止 token：{token}")


def assert_protocol_objects_match_entrypoint() -> None:
    """
    函数功能：
        复用 entrypoint 的 provider/head 组合，在内存中验证 Visual FeatureBatch、
        ExpertBatch、RouterOutput 形状和对齐关系。
    """
    manifest = load_sample_manifest_csv(DEFAULT_SAMPLE_MANIFEST_CSV, split_name="test")
    ordered_sample_keys = manifest.sample_keys()
    if ordered_sample_keys != EXPECTED_TEST_SAMPLE_KEYS:
        raise AssertionError(f"test split sample_key 顺序漂移：{ordered_sample_keys}")

    history_windows = load_history_windows_json(DEFAULT_HISTORY_WINDOWS_JSON)
    feature_provider = VisualMockFeatureProvider(
        history_windows=history_windows,
        history_source_name="stage1_visual_feature_mock_history_window_x",
        source="tests/fixtures/stage1_visual_feature_mock/history_windows.json:in_memory",
    )
    feature_batch = feature_provider.load_batch(ordered_sample_keys)
    expert_provider = JsonExpertSmallProvider(DEFAULT_EXPERT_PREDICTIONS_JSON)
    expert_batch = expert_provider.load_batch(feature_batch.sample_keys)

    if feature_batch.sample_keys != expert_batch.sample_keys:
        raise AssertionError("FeatureBatch 未与 ExpertBatch sample_keys 对齐")
    if tuple(feature_batch.features.shape) != (len(EXPECTED_TEST_SAMPLE_KEYS), 8):
        raise AssertionError(f"FeatureBatch shape 异常：{feature_batch.features.shape}")
    if feature_batch.features.dtype != np.float32:
        raise AssertionError(f"FeatureBatch dtype 应为 float32：{feature_batch.features.dtype}")
    if feature_batch.feature_schema.get("feature_schema_name") != "visual_mock_history_encoder_v1":
        raise AssertionError(f"Visual feature_schema 漂移：{feature_batch.feature_schema}")
    if feature_batch.feature_schema.get("encoder_stub", {}).get("loads_real_vit") is not False:
        raise AssertionError("VisualMockFeatureProvider 不应加载真实 ViT")

    mlp = build_loaded_smoke_mlp(input_dim=int(feature_batch.features.shape[1]), output_dim=len(expert_batch.model_columns))
    adapter = SmokeOnlyVisualMLPAdapter(mlp=mlp, device=torch.device("cpu"))
    with patch.object(torch, "load", side_effect=fail_torch_load):
        router_output = adapter.predict(feature_batch, expert_batch.model_columns)
        result = EvaluationInputAdapter().evaluate(expert_batch=expert_batch, router_output=router_output)

    if router_output.model_columns != expert_batch.model_columns:
        raise AssertionError("RouterOutput model_columns 未与 ExpertBatch 对齐")
    if tuple(router_output.weights.shape) != (len(EXPECTED_TEST_SAMPLE_KEYS), len(expert_batch.model_columns)):
        raise AssertionError(f"weights shape 异常：{router_output.weights.shape}")
    if not np.isfinite(router_output.weights).all() or not np.isfinite(router_output.logits).all():
        raise AssertionError("RouterOutput logits/weights 必须全为有限值")
    np.testing.assert_allclose(np.sum(router_output.weights, axis=1), np.ones(len(EXPECTED_TEST_SAMPLE_KEYS)), rtol=0.0, atol=1e-6)

    if [row["sample_key"] for row in result.per_sample_rows] != list(EXPECTED_TEST_SAMPLE_KEYS):
        raise AssertionError("evaluation rows 未保持 sample_key 顺序")
    summary = result.summary
    for field in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse", "selected_counts"):
        if field not in summary:
            raise AssertionError(f"evaluation summary 缺少字段：{field}")
    print("通过：内存协议对象验证覆盖 Visual FeatureBatch、ExpertBatch/RouterOutput 对齐和 softmax 权重")


def run_smoke() -> None:
    """函数功能：执行 P15c Visual-specific small entrypoint smoke。"""
    print("开始 Stage 1 P15c Visual-specific small entrypoint smoke")
    assert_no_forbidden_static_references()
    assert_protocol_objects_match_entrypoint()

    generic_before = GENERIC_ENTRYPOINT.read_bytes()
    timefuse_before = TIMEFUSE_ENTRYPOINT.read_bytes()
    with tempfile.TemporaryDirectory(prefix="stage1_p15c_visual_small_") as temp_dir:
        output_dir = Path(temp_dir) / "run_outputs"
        if str(output_dir.resolve()).startswith("/data2/"):
            raise AssertionError("P15c smoke 不应使用 /data2 tempfile")
        cmd = [
            sys.executable,
            str(ENTRYPOINT),
            "--sample-manifest-csv",
            str(DEFAULT_SAMPLE_MANIFEST_CSV),
            "--history-windows-json",
            str(DEFAULT_HISTORY_WINDOWS_JSON),
            "--expert-predictions-json",
            str(DEFAULT_EXPERT_PREDICTIONS_JSON),
            "--output-dir",
            str(output_dir),
            "--split-name",
            "test",
            "--run-id",
            "p15c_visual_small_entrypoint_smoke",
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
                "Visual small entrypoint 返回码异常："
                f"returncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
            )
        if "run_dir:" not in completed.stdout:
            raise AssertionError(f"entrypoint stdout 未包含 run_dir：{completed.stdout}")
        stdout_stderr = completed.stdout + completed.stderr
        forbidden_runtime_tokens = (
            "/data2",
            "train_visual_router_online_streaming",
            "train_timefuse_fusor_streaming",
            "VisualMLPRouter",
            "ViTModel",
            "torch.load",
        )
        for token in forbidden_runtime_tokens:
            if token in stdout_stderr:
                raise AssertionError(f"P15c stdout/stderr 不应出现禁止 token：{token}")
        print("通过：entrypoint subprocess 完成，未引用 /data2、正式训练入口、真实 checkpoint 或 ViT")

        run_dir = output_dir / "p15c_visual_small_entrypoint_smoke"
        for subdir in CANONICAL_SUBDIRS:
            if not (run_dir / subdir).is_dir():
                raise AssertionError(f"canonical 子目录缺失：{subdir}")
        metadata = load_json(run_dir / "run_metadata.json")
        status = load_json(run_dir / "run_status.json")
        manifest_ref = load_json(run_dir / "inputs" / "sample_manifest_ref.json")
        split_summary = load_json(run_dir / "inputs" / "split_summary.json")
        evaluation_summary = load_json(run_dir / "evaluation" / "evaluation_summary.json")
        print("通过：canonical run_dir、metadata/status、inputs 和 evaluation JSON 均存在")

        if metadata["protocol_version"] != "stage1_visual_small_entrypoint_v1":
            raise AssertionError(f"protocol_version 异常：{metadata}")
        if metadata["branch_name"] != "visual_router_small":
            raise AssertionError("run_metadata branch_name 异常")
        visual_metadata = metadata["visual_router"]
        if visual_metadata["feature_provider"] != "VisualMockFeatureProvider":
            raise AssertionError(f"run_metadata 未记录 VisualMockFeatureProvider：{visual_metadata}")
        if visual_metadata["formal_visual_router_migration"] is not False:
            raise AssertionError("P15c 不应标记为正式 Visual Router 迁移")
        if visual_metadata["loads_real_checkpoint"] is not False or visual_metadata["loads_real_vit"] is not False:
            raise AssertionError("P15c metadata 必须记录未加载真实 checkpoint/ViT")
        if status["status"] != "completed" or status["current_stage"] != "visual_small_entrypoint":
            raise AssertionError(f"run_status 异常：{status}")
        if manifest_ref["row_count"] != len(EXPECTED_TEST_SAMPLE_KEYS):
            raise AssertionError("sample_manifest_ref row_count 异常")
        if split_summary["sample_count_by_split"] != {"test": len(EXPECTED_TEST_SAMPLE_KEYS)}:
            raise AssertionError(f"split_summary count 异常：{split_summary['sample_count_by_split']}")
        metrics = evaluation_summary["metrics"]
        for field in ("hard_mae", "hard_mse", "raw_soft_mae", "raw_soft_mse"):
            if field not in metrics:
                raise AssertionError(f"evaluation metrics 缺少字段：{field}")
        if "selected_counts" not in evaluation_summary:
            raise AssertionError("evaluation summary 缺少 selected_counts")
        print("通过：summary 包含 hard/raw-soft MAE/MSE 与 selected counts")

        prediction_csv = run_dir / "predictions" / "prediction_rows.csv"
        if not prediction_csv.is_file():
            raise AssertionError("prediction_rows.csv 未写入 predictions/")
        with prediction_csv.open("r", encoding="utf-8", newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
        if tuple(row["sample_key"] for row in prediction_rows) != EXPECTED_TEST_SAMPLE_KEYS:
            raise AssertionError(f"prediction rows sample_key 顺序漂移：{prediction_rows}")
        if tuple(row["split"] for row in prediction_rows) != ("test", "test"):
            raise AssertionError(f"prediction rows split 异常：{prediction_rows}")
        if not (run_dir / "logs" / "visual_small_entrypoint.log").is_file():
            raise AssertionError("logs/visual_small_entrypoint.log 未写出")
        print("通过：prediction rows 保持 sample_key 顺序，最小日志文件已写出")

    if GENERIC_ENTRYPOINT.read_bytes() != generic_before:
        raise AssertionError("P15c smoke 运行不应修改 generic scripts/run_stage1_canonical_small.py")
    if TIMEFUSE_ENTRYPOINT.read_bytes() != timefuse_before:
        raise AssertionError("P15c smoke 运行不应修改 scripts/run_stage1_timefuse_small.py")
    print("完成：Stage 1 P15c Visual-specific small entrypoint smoke 全部通过")


if __name__ == "__main__":
    run_smoke()
