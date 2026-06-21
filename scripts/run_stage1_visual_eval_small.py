#!/usr/bin/env python3
"""
文件功能：
    Stage 1 P21a Visual canonical eval small launcher。

输入：
    读取 small JSON config，生成 canonical eval entrypoint 命令；默认使用 repo 内
    tests/fixtures 和 `/tmp` tiny checkpoint payload。

输出：
    调用 `scripts/run_stage1_visual_eval_canonical.py` 写出 canonical run_dir，并在
    stdout 打印每个 run_dir 与 feature_source。

关键约束：
    本 launcher 只是 thin wrapper，不复制 evaluation 逻辑，不访问 `/data2`，不启动
    训练或 full-scale，不默认加载真实 ViT，不默认导入 transformers，不修改 streaming
    训练入口。
"""

from __future__ import annotations

import argparse
import importlib
import json
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CANONICAL_ENTRYPOINT = REPO_ROOT / "scripts" / "run_stage1_visual_eval_canonical.py"
DEFAULT_PRECOMPUTED_CONFIG = REPO_ROOT / "configs" / "stage1" / "visual_eval_small_precomputed.json"
DEFAULT_VISUAL_CHAIN_CONFIG = REPO_ROOT / "configs" / "stage1" / "visual_eval_small_visual_chain.json"
LEGACY_IMPORT_PATH = "visual_router_experiments.stage1_vali_test_router.train_visual_router"
ALLOWED_FEATURE_SOURCES = {"precomputed", "visual-chain-dryrun"}


def load_json_object(path: Path) -> dict[str, Any]:
    """函数功能：读取 JSON object config，并在类型异常时 fail-fast。"""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config-json 必须是 JSON object：{path}")
    return payload


def _is_under_or_equal(path: Path, root: Path) -> bool:
    """函数功能：判断 resolved path 是否等于 root 或位于 root 下。"""
    return path == root or str(path).startswith(f"{root}/")


def is_data2_path(path: str | Path) -> bool:
    """函数功能：判断 path 是否指向 `/data2` 或其子路径。"""
    resolved = Path(path).expanduser().resolve()
    return _is_under_or_equal(resolved, Path("/data2").resolve())


def resolve_repo_path(value: str | None, *, field_name: str, required: bool = False) -> Path | None:
    """
    函数功能：
        把 config 中的相对路径解析到 repo root；空值按 required 规则处理。
    """
    if value in (None, ""):
        if required:
            raise ValueError(f"config 缺少必需路径字段：{field_name}")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def iter_config_path_strings(value: object) -> list[str]:
    """函数功能：递归收集 config 中看起来像文件系统路径的字符串，供 `/data2` guard 使用。"""
    if isinstance(value, str):
        if value.startswith("/") or "/" in value or value.startswith("~"):
            return [value]
        return []
    if isinstance(value, Mapping):
        paths: list[str] = []
        for nested in value.values():
            paths.extend(iter_config_path_strings(nested))
        return paths
    if isinstance(value, list):
        paths = []
        for nested in value:
            paths.extend(iter_config_path_strings(nested))
        return paths
    return []


def validate_safety(config: Mapping[str, Any], *, allow_real_vit: bool) -> None:
    """
    函数功能：
        对 small config 做轻量 safety gate，阻止 P21a 默认路径滑向真实数据或训练。
    """
    safety = config.get("safety", {})
    if not isinstance(safety, Mapping):
        raise ValueError("config.safety 必须是 object")
    allow_data2 = bool(safety.get("allow_data2", False))
    if not allow_data2:
        for raw_path in iter_config_path_strings(config):
            if is_data2_path(raw_path):
                raise ValueError(f"P21a small launcher 默认禁止 /data2 path：{raw_path}")
    if bool(safety.get("training_started", False)):
        raise ValueError("P21a small launcher 不允许 training_started=true")
    if bool(safety.get("full_scale_run", False)):
        raise ValueError("P21a small launcher 不允许 full_scale_run=true")
    if bool(config.get("training_started", False)):
        raise ValueError("P21a small launcher 不允许顶层 training_started=true")
    if bool(config.get("full_scale_run", False)):
        raise ValueError("P21a small launcher 不允许顶层 full_scale_run=true")
    if bool(config.get("manual_real_vit", False)) and not allow_real_vit:
        raise ValueError("manual_real_vit=true 必须在 CLI 显式传入 --allow-real-vit")
    if bool(safety.get("loads_real_vit", False)) and not allow_real_vit:
        raise ValueError("loads_real_vit=true 必须在 CLI 显式传入 --allow-real-vit")


def load_model_columns(expert_predictions_json: Path) -> tuple[str, ...]:
    """函数功能：从 expert JSON 读取模型列顺序，用于 tiny checkpoint output_dim。"""
    payload = load_json_object(expert_predictions_json)
    model_columns = payload.get("model_columns")
    if not isinstance(model_columns, list) or not model_columns:
        raise ValueError(f"expert_predictions_json 缺少非空 model_columns：{expert_predictions_json}")
    return tuple(str(model_name) for model_name in model_columns)


def build_fake_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """函数功能：基于 legacy VisualMLPRouter shape 构造 deterministic tiny state_dict。"""
    fake_state: dict[str, torch.Tensor] = {}
    for index, (key, value) in enumerate(model.state_dict().items()):
        tensor = torch.linspace(-0.03 + index * 0.01, 0.05 + index * 0.01, steps=value.numel(), dtype=value.dtype)
        fake_state[f"module.{key}"] = tensor.reshape_as(value).clone()
    return fake_state


def save_auto_tempfile_checkpoint(path: Path, *, spec: Mapping[str, Any], output_dim: int) -> None:
    """
    函数功能：
        在 `/tmp` 中创建 small canonical eval 使用的 tiny legacy MLP checkpoint payload。
    """
    if not _is_under_or_equal(path.resolve(), Path("/tmp").resolve()):
        raise ValueError(f"auto-tempfile checkpoint 必须位于 /tmp：{path}")
    module = importlib.import_module(LEGACY_IMPORT_PATH)
    router_cls = getattr(module, "VisualMLPRouter")
    input_dim = int(spec["input_dim"])
    hidden_dim = int(spec.get("hidden_dim", max(4, input_dim + 1)))
    dropout = float(spec.get("dropout", 0.0))
    model = router_cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    payload: Mapping[str, Any] = {
        "router_state_dict": build_fake_state_dict(model),
        "config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "payload_name": str(spec.get("payload_name", "p21a_auto_tempfile_tiny_checkpoint")),
        },
        "metadata": {
            "stage": "P21a",
            "source": "auto-tempfile checkpoint payload from visual eval small launcher",
            "loads_real_checkpoint": False,
            "loads_real_vit": False,
            "training_started": False,
            "full_scale_run": False,
        },
    }
    torch.save(dict(payload), path)


def materialize_checkpoint(config: Mapping[str, Any], *, temp_root: Path, expert_predictions_json: Path) -> Path:
    """
    函数功能：
        把 config 的 router_checkpoint_payload 解析为 canonical entrypoint 可读取的 path。
    """
    spec = config.get("router_checkpoint_payload")
    if isinstance(spec, str):
        if spec != "auto-tempfile":
            checkpoint_path = resolve_repo_path(spec, field_name="router_checkpoint_payload", required=True)
            assert checkpoint_path is not None
            return checkpoint_path
        spec = {"mode": "auto-tempfile"}
    if not isinstance(spec, Mapping):
        raise ValueError("router_checkpoint_payload 必须是 path string 或 object")
    mode = str(spec.get("mode", ""))
    if mode != "auto-tempfile":
        path_value = spec.get("path")
        if not isinstance(path_value, str):
            raise ValueError("非 auto-tempfile router_checkpoint_payload 必须提供 path")
        checkpoint_path = resolve_repo_path(path_value, field_name="router_checkpoint_payload.path", required=True)
        assert checkpoint_path is not None
        return checkpoint_path
    checkpoint_path = temp_root / f"{str(config['feature_source']).replace('-', '_')}_tiny_visual_mlp_payload.pt"
    model_columns = load_model_columns(expert_predictions_json)
    save_auto_tempfile_checkpoint(checkpoint_path, spec=spec, output_dim=len(model_columns))
    return checkpoint_path


def build_run_id(config: Mapping[str, Any], *, explicit_run_id: str | None, suffix: str | None) -> str:
    """函数功能：生成单层 run_id；both 模式显式 run_id 会追加分支后缀避免冲突。"""
    if explicit_run_id:
        return f"{explicit_run_id}_{suffix}" if suffix else explicit_run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    feature_source = str(config["feature_source"]).replace("-", "_")
    config_name = str(config["config_name"]).replace("/", "_")
    return f"{config_name}_{feature_source}_{timestamp}"


def build_canonical_command(
    *,
    python_executable: str,
    config: Mapping[str, Any],
    output_dir: Path,
    run_id: str,
    checkpoint_path: Path,
    allow_real_checkpoint: bool,
    allow_real_vit: bool,
) -> list[str]:
    """
    函数功能：
        将 small config 映射为 canonical eval entrypoint CLI，不复制 evaluation 逻辑。
    """
    feature_source = str(config.get("feature_source", ""))
    if feature_source not in ALLOWED_FEATURE_SOURCES:
        raise ValueError(f"未知 feature_source：{feature_source}")
    sample_manifest_csv = resolve_repo_path(str(config.get("sample_manifest_csv", "")), field_name="sample_manifest_csv", required=True)
    expert_predictions_json = resolve_repo_path(
        str(config.get("expert_predictions_json", "")),
        field_name="expert_predictions_json",
        required=True,
    )
    assert sample_manifest_csv is not None and expert_predictions_json is not None
    cmd = [
        python_executable,
        str(CANONICAL_ENTRYPOINT),
        "--sample-manifest-csv",
        str(sample_manifest_csv),
        "--expert-predictions-json",
        str(expert_predictions_json),
        "--router-checkpoint-payload",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--config-name",
        str(config["config_name"]),
        "--split-name",
        str(config["split_name"]),
        "--feature-source",
        feature_source,
        "--visual-chain-mode",
        str(config.get("visual_chain_mode", "dryrun")),
        "--vit-provider-mode",
        str(config.get("vit_provider_mode", "injected-fake")),
        "--checkpoint-path-label",
        "p21a_small_launcher_auto_tempfile",
    ]
    if bool(config.get("strict_checkpoint_load", False)):
        cmd.append("--strict-checkpoint-load")
    if allow_real_checkpoint:
        cmd.append("--allow-real-checkpoint")
    scaler_state_json = resolve_repo_path(config.get("scaler_state_json"), field_name="scaler_state_json")
    if scaler_state_json is not None:
        cmd.extend(["--scaler-state-json", str(scaler_state_json)])
    if feature_source == "precomputed":
        visual_features_csv = resolve_repo_path(config.get("visual_features_csv"), field_name="visual_features_csv", required=True)
        assert visual_features_csv is not None
        cmd.extend(["--visual-features-csv", str(visual_features_csv)])
    if feature_source == "visual-chain-dryrun":
        raw_window_json = resolve_repo_path(config.get("raw_window_json"), field_name="raw_window_json", required=True)
        assert raw_window_json is not None
        cmd.extend(["--raw-window-json", str(raw_window_json)])
    if bool(config.get("manual_real_vit", False)):
        cmd.append("--manual-real-vit")
        if allow_real_vit:
            cmd.append("--allow-real-vit")
    return cmd


def run_one(
    *,
    config_path: Path,
    output_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    temp_root: Path,
) -> Path | None:
    """函数功能：执行或打印单个 config 对应的 canonical run。"""
    config = load_json_object(config_path)
    validate_safety(config, allow_real_vit=bool(args.allow_real_vit))
    expert_predictions_json = resolve_repo_path(
        str(config.get("expert_predictions_json", "")),
        field_name="expert_predictions_json",
        required=True,
    )
    assert expert_predictions_json is not None
    checkpoint_path = materialize_checkpoint(config, temp_root=temp_root, expert_predictions_json=expert_predictions_json)
    cmd = build_canonical_command(
        python_executable=str(args.python_executable),
        config=config,
        output_dir=output_dir,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        allow_real_checkpoint=bool(args.allow_real_checkpoint),
        allow_real_vit=bool(args.allow_real_vit),
    )
    feature_source = str(config["feature_source"])
    if bool(args.dry_print_command):
        print(f"feature_source={feature_source} canonical_command={shlex.join(cmd)}")
        return None
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "canonical eval command failed: "
            f"feature_source={feature_source} returncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    run_dir = output_dir / run_id
    print(f"feature_source={feature_source} run_dir={run_dir}")
    return run_dir


def select_config_runs(args: argparse.Namespace) -> list[tuple[Path, str | None]]:
    """函数功能：根据 --mode/--config-json 选择待运行 config，both 默认跑两份内置 config。"""
    mode = str(args.mode)
    if args.config_json and mode == "both":
        raise ValueError("--mode both 使用内置 precomputed/visual-chain config；不要同时传 --config-json")
    if args.config_json:
        return [(Path(args.config_json), None)]
    if mode == "precomputed":
        return [(DEFAULT_PRECOMPUTED_CONFIG, None)]
    if mode == "visual-chain":
        return [(DEFAULT_VISUAL_CHAIN_CONFIG, None)]
    if mode == "both":
        return [(DEFAULT_PRECOMPUTED_CONFIG, "precomputed"), (DEFAULT_VISUAL_CHAIN_CONFIG, "visual_chain")]
    raise ValueError(f"未知 mode：{mode}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """函数功能：解析 P21a small launcher CLI。"""
    parser = argparse.ArgumentParser(description="Run Stage 1 Visual canonical eval small config pack.")
    parser.add_argument("--config-json", default=None, help="small eval JSON config；省略时按 --mode 使用内置 config。")
    parser.add_argument("--output-dir", required=True, help="canonical run_dir 父目录。")
    parser.add_argument("--run-id", default=None, help="可选 run id；both 模式会追加分支后缀。")
    parser.add_argument("--mode", default="precomputed", choices=("precomputed", "visual-chain", "both"), help="small eval 运行模式。")
    parser.add_argument("--python-executable", default=sys.executable, help="执行 canonical entrypoint 的 Python。")
    parser.add_argument("--dry-print-command", action="store_true", help="只打印 canonical command，不执行。")
    parser.add_argument("--allow-real-checkpoint", action="store_true", help="透传给 canonical entrypoint；默认 false。")
    parser.add_argument("--allow-real-vit", action="store_true", help="仅 config 显式 manual real ViT 时允许；默认 false。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """函数功能：命令行入口，连续执行所选 small config。"""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if is_data2_path(output_dir):
        raise ValueError(f"P21a small launcher 默认禁止 /data2 output_dir：{output_dir}")
    selected = select_config_runs(args)
    if bool(args.dry_print_command):
        # dry-print 输出的是可复用的 canonical command，因此 tiny checkpoint 不能放在
        # with TemporaryDirectory 自动清理目录里。
        temp_root = Path(tempfile.mkdtemp(prefix="stage1_p21a_visual_eval_small_dryprint_"))
        for config_path, suffix in selected:
            config = load_json_object(config_path)
            run_id = build_run_id(config, explicit_run_id=args.run_id, suffix=suffix)
            run_one(config_path=config_path, output_dir=output_dir, run_id=run_id, args=args, temp_root=temp_root)
        return
    with tempfile.TemporaryDirectory(prefix="stage1_p21a_visual_eval_small_") as temp_dir:
        temp_root = Path(temp_dir)
        for config_path, suffix in selected:
            config = load_json_object(config_path)
            run_id = build_run_id(config, explicit_run_id=args.run_id, suffix=suffix)
            run_one(config_path=config_path, output_dir=output_dir, run_id=run_id, args=args, temp_root=temp_root)


if __name__ == "__main__":
    main()
