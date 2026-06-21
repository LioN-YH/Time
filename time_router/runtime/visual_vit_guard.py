"""
文件功能：
    Stage 1 P19b guarded real ViT model / processor path policy helper。

设计边界：
    本 helper 只对调用方显式传入的 ViT model / processor 路径做授权判定，
    返回可写入 provider metadata 的 policy 摘要。它不检查文件是否存在，
    不读取模型文件，不导入 transformers，不自动搜索 `/data2`，也不从
    checkpoint 或 run_dir 推断 ViT 路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from time_router.runtime.visual_eval_checkpoint_guard import is_data2_path


@dataclass(frozen=True)
class VisualVitModelPathPolicy:
    """
    类功能：
        描述 P19b ViT model / processor 路径授权结果。

    字段说明：
        model_path 和 processor_path 是解析后的显式路径；processor_path 为空时
        表示调用方选择与 model_path 共用同一 HuggingFace artifact 目录。
    """

    policy_name: str
    model_path: Path
    processor_path: Path | None
    is_model_fixture_or_tempfile: bool
    is_processor_fixture_or_tempfile: bool | None
    is_model_external_data2_path: bool
    is_processor_external_data2_path: bool | None
    loads_real_vit: bool
    model_path_policy: str
    processor_path_policy: str
    allow_real_vit: bool
    allow_external_vit_path: bool


def _is_under_or_equal(path: Path, root: Path) -> bool:
    """函数功能：判断 resolved path 是否等于 root 或位于 root 下。"""
    return path == root or str(path).startswith(f"{root}/")


def is_fixture_or_tempfile_visual_vit_artifact(path: str | Path, *, repo_root: str | Path) -> bool:
    """
    函数功能：
        判断 ViT model / processor path 是否属于默认允许的 fixture/tmp 范围。

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


def _classify_path(
    *,
    path: Path,
    repo_root: str | Path,
    allow_real_vit: bool,
    allow_external_vit_path: bool,
    role: str,
) -> tuple[bool, bool, str]:
    """
    函数功能：
        对单个 model/processor 路径做授权判定并返回 metadata policy。
    """
    fixture_or_tempfile = is_fixture_or_tempfile_visual_vit_artifact(path, repo_root=repo_root)
    data2_path = is_data2_path(path)

    if not fixture_or_tempfile and not allow_real_vit:
        raise ValueError(
            f"visual ViT {role} path 不在 tests/fixtures 或 /tmp；必须显式开启 "
            f"--allow-real-vit 后才允许构造 real ViT encoder provider：{path}"
        )
    if data2_path and not allow_external_vit_path:
        raise ValueError(
            f"/data2 visual ViT {role} path 需要额外显式开启 --allow-external-vit-path；"
            f"当前只完成 path guard，不会自动搜索或读取未授权外部路径：{path}"
        )

    if fixture_or_tempfile:
        path_policy = f"default_fixture_or_tmp_vit_{role}"
    elif data2_path:
        path_policy = f"explicit_real_vit_{role}_external_data2_authorized"
    else:
        path_policy = f"explicit_real_vit_{role}_authorized"
    return fixture_or_tempfile, data2_path, path_policy


def authorize_visual_vit_model_paths(
    *,
    model_path: str | Path,
    processor_path: str | Path | None = None,
    repo_root: str | Path,
    allow_real_vit: bool,
    allow_external_vit_path: bool,
) -> VisualVitModelPathPolicy:
    """
    函数功能：
        执行 P19b guarded real ViT model / processor path 授权检查。

    输入：
        model_path: 用户显式提供的 ViT model artifact path；
        processor_path: 用户显式提供的 processor path；为空时复用 model_path；
        repo_root: 当前仓库根目录，用于识别 `tests/fixtures`；
        allow_real_vit: 是否允许非 fixture/tmp ViT artifact；
        allow_external_vit_path: 是否额外允许 `/data2` ViT artifact path。

    输出：
        `VisualVitModelPathPolicy`，供 Runtime / 显式构造函数继续加载真实 provider。

    关键约束：
        - 默认只允许 `tests/fixtures` 或 `/tmp` tiny/local dry-run artifact。
        - 非 fixture/tmp 路径必须显式开启 allow_real_vit。
        - `/data2` 路径还必须额外开启 allow_external_vit_path。
        - 本函数不检查文件存在、不导入 transformers、不读取模型文件。
    """
    resolved_model_path = Path(model_path).resolve()
    resolved_processor_path = Path(processor_path).resolve() if processor_path is not None else None

    model_fixture, model_data2, model_policy = _classify_path(
        path=resolved_model_path,
        repo_root=repo_root,
        allow_real_vit=allow_real_vit,
        allow_external_vit_path=allow_external_vit_path,
        role="model",
    )

    if resolved_processor_path is None:
        processor_fixture = None
        processor_data2 = None
        processor_policy = "same_as_model_path"
    else:
        processor_fixture, processor_data2, processor_policy = _classify_path(
            path=resolved_processor_path,
            repo_root=repo_root,
            allow_real_vit=allow_real_vit,
            allow_external_vit_path=allow_external_vit_path,
            role="processor",
        )

    return VisualVitModelPathPolicy(
        policy_name="stage1_p19b_visual_vit_model_path_guard_v1",
        model_path=resolved_model_path,
        processor_path=resolved_processor_path,
        is_model_fixture_or_tempfile=model_fixture,
        is_processor_fixture_or_tempfile=processor_fixture,
        is_model_external_data2_path=model_data2,
        is_processor_external_data2_path=processor_data2,
        loads_real_vit=True,
        model_path_policy=model_policy,
        processor_path_policy=processor_policy,
        allow_real_vit=bool(allow_real_vit),
        allow_external_vit_path=bool(allow_external_vit_path),
    )
