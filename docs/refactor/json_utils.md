# Stage 1 P4a JSON Utils 边界说明

创建日期：2026-06-19

## 1. 目标

`time_router/io/json_utils.py` 只提供最小 I/O 基础能力：

- `atomic_write_json(...)`：同目录临时文件写入、`flush + fsync` 后通过 `os.replace` 原子替换目标 JSON。
- `build_status_payload(...)`：构造至少包含 `status` 的最小 status payload。
- `write_status_json(...)`：把最小 status payload 写入调用方显式传入的路径。

该小步属于 P4 的最低风险基础设施抽取，不代表完整 logging/path/config 系统已经实现。

## 2. 行为约束

- 默认使用 UTF-8 写入。
- 默认 `ensure_ascii=False`，保留中文 status / metadata 文本。
- 自动创建目标 parent directory。
- 临时文件写在目标同目录，避免跨文件系统替换破坏原子性。
- `extra` 必须是 `dict` 或 `None`。
- 只消费调用方传入的 `path` 和 payload，不自动读取训练状态。
- 不绑定 Visual Router、TimeFuse fusor、oracle、TSF 或 full-scale 输出目录。

## 3. 明确不做

- 不迁移 Visual Router / TimeFuse fusor 正式训练入口。
- 不修改任何既有正式训练脚本。
- 不改变任何现有正式输出目录含义。
- 不改变既有 `status.json` schema。
- 不接入 launcher / monitor / resume 行为。
- 不实现 path resolver、config system 或 logging framework。
- 不实现 comparison / calibration / report schema。
- 不读取 oracle/TSF，不改 `PredictionBatchReader` / `OracleTsfReader`。

## 4. 验证

P4a 新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
```

该 smoke 只在 `tempfile.TemporaryDirectory` 下写入临时 `status.json` 和 `metadata.json`，覆盖：

- 文件存在且 JSON 可读；
- 中文 message 不被 ASCII 转义；
- 第二次写入覆盖旧内容；
- nested parent directory 自动创建；
- `extra` 非 dict 时明确失败。

完整 P4a 门禁仍包括：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```
