"""
文件功能：
    Stage 1 P17b Visual canonical eval checkpoint path policy helper。

设计边界：
    本 helper 只判断显式传入的 legacy VisualMLPRouter checkpoint payload 路径是否
    被授权读取，并返回可写入 Runtime metadata 的脱敏 policy 结果。它不调用
    `torch.load`，不检查文件是否存在，不搜索 checkpoint，不从 run_dir 推断路径，也不
    接触 FeatureProvider 或 RouterHead adapter interface。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointPathPolicy:
    """
    类功能：
        描述 Visual eval checkpoint path guard 的判定结果。

    字段说明：
        policy_name: 当前 guard 版本名；
        resolved_path: 解析后的 checkpoint 路径，仅供 Runtime 内部继续读取；
        is_fixture_or_tempfile: 是否属于默认 tiny fixture/tempfile 白名单；
        is_external_data2_path: 是否属于 `/data2` 外部数据路径；
        loads_real_checkpoint: metadata 使用的真实 checkpoint 标记；
        checkpoint_path_policy: 可写入 metadata 的策略摘要；
        checkpoint_payload_source: 可写入 metadata 的 payload 来源分类。
    """

    policy_name: str
    resolved_path: Path
    is_fixture_or_tempfile: bool
    is_external_data2_path: bool
    loads_real_checkpoint: bool
    checkpoint_path_policy: str
    checkpoint_payload_source: str


def _is_under_or_equal(path: Path, root: Path) -> bool:
    """函数功能：判断 resolved path 是否等于 root 或位于 root 下。"""
    return path == root or str(path).startswith(f"{root}/")


def is_fixture_or_tempfile_checkpoint(path: str | Path, *, repo_root: str | Path) -> bool:
    """
    函数功能：
        判断 checkpoint path 是否属于默认允许的 tiny fixture/tempfile 范围。

    关键约束：
        这里只做路径归类，不检查文件存在，也不读取 checkpoint 内容。
    """
    resolved = Path(path).resolve()
    repo_root_path = Path(repo_root).resolve()
    allowed_roots = (
        (repo_root_path / "tests" / "fixtures").resolve(),
        Path("/tmp").resolve(),
    )
    return any(_is_under_or_equal(resolved, root) for root in allowed_roots)


def is_data2_path(path: str | Path) -> bool:
    """函数功能：判断 path 是否指向 `/data2` 或其子路径。"""
    resolved = Path(path).resolve()
    data2_root = Path("/data2").resolve()
    return _is_under_or_equal(resolved, data2_root)


def authorize_visual_eval_checkpoint_path(
    path: str | Path,
    *,
    repo_root: str | Path,
    allow_real_checkpoint: bool,
    allow_external_checkpoint_path: bool,
) -> CheckpointPathPolicy:
    """
    函数功能：
        执行 P17b Visual canonical eval checkpoint path 授权检查。

    输入：
        path: 用户显式提供的 checkpoint payload 路径；
        repo_root: 当前仓库根目录，用于识别 `tests/fixtures`；
        allow_real_checkpoint: 是否允许非 fixture/tempfile checkpoint；
        allow_external_checkpoint_path: 是否额外允许 `/data2` checkpoint path。

    输出：
        `CheckpointPathPolicy`，供入口继续读取文件并写入 metadata。

    关键约束：
        - 默认只允许 `tests/fixtures` 或 `/tmp` tiny payload。
        - 非 fixture/tmp 路径必须显式开启 allow_real_checkpoint。
        - `/data2` 路径还必须额外开启 allow_external_checkpoint_path。
        - 本函数不检查文件存在、不调用 `torch.load`，因此 smoke 可安全测试 `/data2`
          policy 而不创建或读取 `/data2` 文件。
    """
    resolved = Path(path).resolve()
    fixture_or_tempfile = is_fixture_or_tempfile_checkpoint(resolved, repo_root=repo_root)
    data2_path = is_data2_path(resolved)

    if not fixture_or_tempfile and not allow_real_checkpoint:
        raise ValueError(
            "checkpoint path 不在 tests/fixtures 或 /tmp；必须显式开启 --allow-real-checkpoint "
            f"后才允许 evaluation-only dry-run 读取：{resolved}"
        )
    if data2_path and not allow_external_checkpoint_path:
        raise ValueError(
            "/data2 checkpoint path 需要额外显式开启 --allow-external-checkpoint-path；"
            f"当前只完成 path guard，不会自动搜索或读取未授权外部路径：{resolved}"
        )

    loads_real_checkpoint = (not fixture_or_tempfile) and bool(allow_real_checkpoint)
    if fixture_or_tempfile:
        checkpoint_path_policy = "default_fixture_or_tmp_checkpoint"
        checkpoint_payload_source = "explicit_fixture_or_tmp_path"
    elif data2_path:
        checkpoint_path_policy = "explicit_real_checkpoint_external_data2_authorized"
        checkpoint_payload_source = "explicit_real_checkpoint_path"
    else:
        checkpoint_path_policy = "explicit_real_checkpoint_authorized"
        checkpoint_payload_source = "explicit_real_checkpoint_path"

    return CheckpointPathPolicy(
        policy_name="stage1_p17b_visual_eval_checkpoint_path_guard_v1",
        resolved_path=resolved,
        is_fixture_or_tempfile=fixture_or_tempfile,
        is_external_data2_path=data2_path,
        loads_real_checkpoint=loads_real_checkpoint,
        checkpoint_path_policy=checkpoint_path_policy,
        checkpoint_payload_source=checkpoint_payload_source,
    )
