# Stage 1 P10c Prediction Array IO Boundary Consolidation

日志日期：2026-06-20 03:56:32 CST

## 目的

把通用 prediction array IO 能力上收到 `time_router.io`，消除 `time_router.io` 对 `visual_router_experiments.common` 的反向依赖，同时保留旧实验脚本的 import 兼容。

## 背景

P10b 已新增最小 shared prediction SQLite backend helper，但 `time_router/io/prediction_sqlite_backend.py` 仍从 `visual_router_experiments.common.prediction_array_io` 导入路径解析函数。该方向会让核心 `time_router` 反向依赖实验目录，不符合长期架构边界。

## 操作

1. 新增 `time_router/io/prediction_array_io.py`，迁入 `PACKED_NPY_STORAGE`、`PER_SAMPLE_NPY_STORAGE`、`resolve_cache_array_path(...)`、`load_prediction_array(...)` 和 `load_prediction_arrays_grouped(...)`。
2. 更新 `time_router/io/__init__.py`，从 `time_router.io` public API 导出上述 prediction array IO 常量和函数。
3. 更新 `time_router/io/prediction_sqlite_backend.py` 与 `time_router/io/prediction_cache_reader.py`，改为依赖 `time_router.io.prediction_array_io`。
4. 更新 `tests/smoke/stage1_prediction_sqlite_backend_smoke.py`，改为从 `time_router.io` 导入 `load_prediction_arrays_grouped(...)`。
5. 将 `visual_router_experiments/common/prediction_array_io.py` 改为 compatibility wrapper，只 re-export canonical implementation，避免旧脚本 import 断裂。
6. 更新 `docs/refactor/prediction_sqlite_backend.md`、`docs/refactor/shared_prediction_sqlite_backend_audit.md`、`docs/refactor/stage1_refactor_roadmap.md` 和 `WORKSPACE_STRUCTURE.md`，记录 P10c 边界与旧路径兼容策略。
7. 使用 `rg` 检查 `time_router/` 与 `tests/smoke/` 中不再存在 `visual_router_experiments.common.prediction_array_io` 反向依赖。

## 结果

已执行验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router visual_router_experiments/common/prediction_array_io.py
```

结果均通过。P10b SQLite smoke 仍能构造 packed fixture、fetch records 并按 row index 读回 arrays；P6a provider smoke、P9f Visual Router training ExpertBatch bypass smoke、P7c TimeFuse protocol chain smoke 和 compileall 均通过。

## 结论

P10c 已完成 prediction array IO 边界整理：canonical implementation 位于 `time_router.io.prediction_array_io`，`time_router.io` 不再反向依赖 `visual_router_experiments.common.prediction_array_io`。旧实验路径仍可兼容导入同名 API。

本次未修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 或 `launch_timefuse_fusor_full_scale.py`，未接正式入口，未替换 Visual Router / TimeFuse 现有 SQLite index，未改 `PredictionCacheExpertProvider` 或 `EvaluationInputAdapter` 行为，未新增 provider/head/runtime/Bash/scripts，未访问 `/data2`，未启动 pressure/full-scale，未改正式 CSV / summary / metadata / status / checkpoint schema，也未改 loss、optimizer、scaler 或 checkpoint/resume。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit` 分支。
2. 后续若继续 P10d，可再整理 launcher / run scripts 边界；provider prepared backend 接入仍建议推迟到 Stage 1.5 / Stage 2，并继续小步 smoke 先行。
