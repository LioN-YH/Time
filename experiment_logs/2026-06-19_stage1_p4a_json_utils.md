# Stage 1 P4a minimal atomic JSON / run status writer extraction

日志日期：2026-06-19 19:30:35 CST

## 目的

在已完成 P1/P2/P3a-P3e 的基础上，进入 P4 的最小基础设施抽取，只新增低风险 JSON 原子写入工具和最小 run status writer，用于后续统一安全写入 `status.json` / metadata-like JSON。

## 背景

Stage 1 后续会继续收束 logging、path、config 和正式入口迁移，但本次不直接进入完整 P4，也不迁移 Visual Router / TimeFuse fusor 正式训练入口。当前需要先提供一个不依赖 pandas、torch、sklearn 的标准库 helper，并用临时目录 smoke 验证基础行为，避免改动正式输出目录或后台监控语义。

## 操作

1. 新增 `time_router/io/json_utils.py`：
   - `atomic_write_json(...)`：自动创建 parent directory，在目标同目录写临时文件，完成 `flush + fsync` 后使用 `os.replace` 原子替换目标 JSON。
   - `build_status_payload(...)`：构造至少包含 `status` 的最小 payload，可选 `phase`、`message`，并要求 `extra` 必须是 dict 或 None。
   - `write_status_json(...)`：把最小 status payload 写入调用方显式传入的路径。
2. 更新 `time_router/io/__init__.py`，只导出明确 public helper：`atomic_write_json`、`build_status_payload`、`write_status_json`。
3. 新增 `tests/smoke/stage1_json_utils_smoke.py`，只在 `tempfile.TemporaryDirectory` 下写入临时 `status.json` / `metadata.json`，验证：
   - 文件存在且 JSON 可读；
   - 中文 message 保持 UTF-8，不被 ASCII 转义；
   - 第二次写入会覆盖旧内容；
   - nested parent directory 自动创建；
   - `extra` 非 dict 时明确拒绝。
4. 新增 `docs/refactor/json_utils.md`，记录 P4a helper 边界和不做事项。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P4a 当前状态、完成范围和明确不做事项。
6. 更新 `WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`，登记新增长期文件与 smoke。

## 结果

已在 Quito conda 环境运行并通过以下命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

关键验证结果：

- P4a JSON utils smoke 通过，确认中文 message 保持 UTF-8、覆盖写入和 nested parent directory 创建行为正确。
- Stage 1 golden smoke 通过，确认 prediction/fusion/summary/rows 契约未受影响。
- Stage 1 oracle/TSF smoke 通过，确认 `OracleTsfReader` 契约未受影响。
- `compileall time_router tests/smoke` 通过。

## 结论

本次只完成 P4a minimal atomic JSON / run status writer extraction。新增 helper 不读取训练状态，不绑定 Visual Router / TimeFuse fusor，不访问正式输出目录，除非调用方显式传入 path；也没有实现 path resolver、config system 或 logging framework。

## 下一步方案

后续如继续 P4，应另起小步设计 path resolver、launcher/status 兼容检查或 logging/config 边界，并在迁移任何正式入口前继续运行 golden smoke、oracle/TSF smoke 和相关小规模门禁。本次不启动 full-scale 任务，也不修改现有正式训练脚本。
