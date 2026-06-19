# Stage 1 P4c Run Metadata 边界说明

创建日期：2026-06-19

## 1. 目标

`time_router/io/run_metadata.py` 只提供最小 run metadata payload builder：

- `build_run_metadata(...)`：构造 metadata-like JSON payload，记录阶段、输入输出、命令、git/refactor 信息和补充说明。
- `write_run_metadata(...)`：可选 writer，内部调用 `build_run_metadata(...)` 和 P4a 的 `atomic_write_json(...)`，只写调用方显式传入的 path。

该小步属于 P4 的低风险 metadata payload 抽取，不代表既有正式 `metadata.json` schema 已迁移或替换。

## 2. 字段约束

- `stage` 必须是非空字符串。
- payload 至少包含 `stage`、`created_at_utc`、`inputs`、`outputs`。
- `created_at_utc` 使用 timezone-aware UTC ISO 字符串。
- `inputs`、`outputs`、`extra` 必须是 dict 或 None。
- `entrypoint`、`command`、`git_ref`、`notes` 为调用方显式传入的可选字段。
- `Path` / `os.PathLike` 会在内存中转换为字符串，便于 JSON 写入。
- `extra` 保留在 payload 的 `extra` 字段下，不展开覆盖基础字段。

## 3. 行为边界

- 不自动调用 git。
- 不自动读取当前命令行。
- 不自动读取训练配置。
- 不自动调用 `find_repo_root(...)`。
- 不自动解析 `/data2` 或 full-scale 输出目录。
- 不检查 inputs/outputs 路径是否存在。
- `write_run_metadata(...)` 不自行选择输出目录；父目录创建行为仅来自调用方显式 path 下的 `atomic_write_json(...)`。

## 4. 明确不做

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何现有正式训练脚本。
- 不替换现有 launcher / monitor / resume 行为。
- 不接入 `/data2` 或任何 full-scale 输出目录。
- 不改变既有正式输出目录含义。
- 不改变既有 `metadata.json` schema。
- 不实现 config system。
- 不实现 logging framework。
- 不实现 checkpoint index。
- 不自动调用 git。
- 不自动读取命令行或训练配置。
- 不创建正式输出目录。
- 不读取 oracle/TSF，不改 `PredictionBatchReader` / `OracleTsfReader`。

## 5. 验证

P4c 新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
```

该 smoke 覆盖：

- 在 `tempfile.TemporaryDirectory` 下构造 metadata；
- `stage` 必须非空；
- `inputs` / `outputs` / `extra` 非 dict 会报错；
- `created_at_utc` 存在且可解析为 timezone-aware ISO 时间；
- `Path` 会被转换为 JSON-safe 字符串；
- `write_run_metadata(...)` 只写 tempfile 下的 `metadata.json` 且 JSON 可读。

完整 P4c 门禁包括：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```
