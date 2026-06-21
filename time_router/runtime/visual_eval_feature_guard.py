"""
文件功能：
    Stage 1 P17c Visual canonical eval feature/scaler path policy helper。

设计边界：
    本 helper 只判断显式传入的 precomputed visual feature CSV 或 scaler state
    JSON 路径是否被授权读取，并返回可写入 Runtime metadata 的 policy 结果。
    它不读取文件内容，不自动搜索 artifact，不从旧 run_dir 推断路径，也不接触
    FeatureProvider、FeatureScaler 或 RouterHead adapter interface。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from time_router.runtime.visual_eval_checkpoint_guard import is_data2_path


@dataclass(frozen=True)
class VisualEvalPathPolicy:
    """
    类功能：
        描述 Visual eval 外部 feature/scaler path guard 的判定结果。

    字段说明：
        policy_name: 当前 guard 版本名；
        artifact_role: artifact 类型，当前为 feature 或 scaler；
        resolved_path: 解析后的路径，仅供 Runtime/entrypoint 内部继续读取；
        is_fixture_or_tempfile: 是否属于默认 fixture/tmp 白名单；
        is_external_data2_path: 是否属于 `/data2` 外部数据路径；
        path_policy: 可写入 metadata 的策略摘要；
        path_label: 用户提供的脱敏来源说明，不参与读取；
        allow_external_path: 本次 CLI 是否显式允许外部路径。
    """

    policy_name: str
    artifact_role: str
    resolved_path: Path
    is_fixture_or_tempfile: bool
    is_external_data2_path: bool
    path_policy: str
    path_label: str
    allow_external_path: bool


def _is_under_or_equal(path: Path, root: Path) -> bool:
    """函数功能：判断 resolved path 是否等于 root 或位于 root 下。"""
    return path == root or str(path).startswith(f"{root}/")


def is_fixture_or_tempfile_visual_eval_artifact(path: str | Path, *, repo_root: str | Path) -> bool:
    """
    函数功能：
        判断 feature/scaler path 是否属于默认允许的 fixture/tempfile 范围。

    关键约束：
        这里只做路径归类，不检查文件存在，也不读取 artifact 内容。
    """
    resolved = Path(path).resolve()
    repo_root_path = Path(repo_root).resolve()
    allowed_roots = (
        (repo_root_path / "tests" / "fixtures").resolve(),
        Path("/tmp").resolve(),
    )
    return any(_is_under_or_equal(resolved, root) for root in allowed_roots)


def authorize_visual_eval_feature_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    allow_external_feature_path: bool,
    path_label: str = "",
) -> VisualEvalPathPolicy:
    """
    函数功能：
        执行 P17c Visual canonical eval precomputed feature CSV path 授权检查。

    关键约束：
        - 默认只允许 `tests/fixtures` 或 `/tmp` feature CSV。
        - 非 fixture/tmp feature CSV 必须显式开启 --allow-external-feature-path。
        - `/data2` feature CSV 也只做显式授权检查，不搜索、不读取未授权路径。
    """
    resolved = Path(path).resolve()
    fixture_or_tempfile = is_fixture_or_tempfile_visual_eval_artifact(resolved, repo_root=repo_root)
    data2_path = is_data2_path(resolved)
    if not fixture_or_tempfile and not allow_external_feature_path:
        raise ValueError(
            "visual feature CSV 不在 tests/fixtures 或 /tmp；必须显式开启 "
            f"--allow-external-feature-path 后才允许 evaluation-only dry-run 读取：{resolved}"
        )

    if fixture_or_tempfile:
        path_policy = "default_fixture_or_tmp_feature"
    elif data2_path:
        path_policy = "explicit_external_feature_data2_authorized"
    else:
        path_policy = "explicit_external_feature_authorized"

    return VisualEvalPathPolicy(
        policy_name="stage1_p17c_visual_eval_feature_path_guard_v1",
        artifact_role="feature",
        resolved_path=resolved,
        is_fixture_or_tempfile=fixture_or_tempfile,
        is_external_data2_path=data2_path,
        path_policy=path_policy,
        path_label=str(path_label or ""),
        allow_external_path=bool(allow_external_feature_path),
    )


def authorize_visual_eval_scaler_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    allow_external_scaler_path: bool,
    path_label: str = "",
) -> VisualEvalPathPolicy:
    """
    函数功能：
        执行 P17c Visual canonical eval scaler state JSON path 授权检查。

    关键约束：
        scaler state 只有调用方显式传入时才会读取和 transform；本 helper 不执行
        fit、partial_fit 或文件内容检查。
    """
    resolved = Path(path).resolve()
    fixture_or_tempfile = is_fixture_or_tempfile_visual_eval_artifact(resolved, repo_root=repo_root)
    data2_path = is_data2_path(resolved)
    if not fixture_or_tempfile and not allow_external_scaler_path:
        raise ValueError(
            "scaler state JSON 不在 tests/fixtures 或 /tmp；必须显式开启 "
            f"--allow-external-scaler-path 后才允许 evaluation-only dry-run 读取：{resolved}"
        )

    if fixture_or_tempfile:
        path_policy = "default_fixture_or_tmp_scaler"
    elif data2_path:
        path_policy = "explicit_external_scaler_data2_authorized"
    else:
        path_policy = "explicit_external_scaler_authorized"

    return VisualEvalPathPolicy(
        policy_name="stage1_p17c_visual_eval_scaler_path_guard_v1",
        artifact_role="scaler",
        resolved_path=resolved,
        is_fixture_or_tempfile=fixture_or_tempfile,
        is_external_data2_path=data2_path,
        path_policy=path_policy,
        path_label=str(path_label or ""),
        allow_external_path=bool(allow_external_scaler_path),
    )
