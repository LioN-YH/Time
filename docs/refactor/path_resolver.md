# Stage 1 P4b Path Resolver 边界说明

创建日期：2026-06-19

## 1. 目标

`time_router/io/path_resolver.py` 只提供最小路径解析基础能力：

- `find_repo_root(...)`：从指定起点或本 helper 文件位置向上查找仓库根。
- `resolve_under_root(...)`：把 root 与路径片段拼接为 resolved path，并拒绝 `..` 或绝对路径逃逸 root。
- `resolve_status_path(...)`：只返回 `run_dir / "status.json"`。
- `resolve_metadata_path(...)`：只返回 `run_dir / "metadata.json"`。

该小步属于 P4 的低风险路径解析抽取，不代表完整 config system、logging framework 或正式 launcher 路径迁移已经实现。

## 2. 行为约束

- repo root marker 当前包括 `.git`、`WORKSPACE_STRUCTURE.md`、`pyproject.toml`、`setup.cfg`、`setup.py`。
- `find_repo_root(...)` 找不到 marker 时明确抛出 `FileNotFoundError`。
- `resolve_under_root(...)` 返回 root 内部或等于 root 的 resolved path。
- `resolve_under_root(..., must_exist=True)` 在路径不存在时明确抛出 `FileNotFoundError`。
- status / metadata path helper 只做路径拼接，不写文件、不创建目录。
- helper 不读取训练配置，不访问 `/data2` 或 full-scale 输出目录。

## 3. 明确不做

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何现有正式训练脚本。
- 不替换现有 launcher / monitor / resume 行为。
- 不接入 `/data2` 或任何 full-scale 输出目录。
- 不改变既有正式输出目录含义。
- 不改变既有 `status.json` / `metadata.json` schema。
- 不实现 config system。
- 不实现 logging framework。
- 不实现 checkpoint index。
- 不创建正式输出目录。
- 不写 JSON 文件。
- 不读取 oracle/TSF，不改 `PredictionBatchReader` / `OracleTsfReader`。

## 4. 验证

P4b 新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
```

该 smoke 覆盖：

- 从 `tests/smoke` 下找到仓库根；
- root 下可定位 `WORKSPACE_STRUCTURE.md`；
- 在 `tempfile.TemporaryDirectory` 下解析正常路径；
- `..` 逃逸 root 会明确失败；
- `must_exist=True` 对不存在路径会明确失败；
- `resolve_status_path(...)` / `resolve_metadata_path(...)` 只返回路径，不创建目录或文件。

完整 P4b 门禁包括：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```
