# Stage 1 P5a Canonical Runtime Contract

日志日期：2026-06-19 20:58:07 CST

## 目的

基于 P4 后 architecture pivot，定义 Stage 1 新 canonical runtime contract，明确未来 Visual Router 主线和 TimeFuse-style fusor baseline 支线应共享的运行目录、状态文件、元数据、checkpoint、日志和评估输出契约。

## 背景

上一轮已经确认暂停 P4f config system，转向 P5 canonical entrypoint / FeatureProvider design。为了避免后续迁移时继续被历史输出 schema 牵制，本轮先把新 runtime contract 写清楚，并明确该 contract 不反向兼容所有历史 status/metadata/checkpoint schema。

## 操作

1. 只读复核了 `docs/refactor/stage1_architecture_pivot_after_p4.md`、`docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md` 和 `experiment_logs/README.md` 中的 P4/P5 结论。
2. 新增 `docs/refactor/stage1_canonical_runtime_contract.md`。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P5a canonical runtime contract only 小步记录。
4. 更新 `docs/refactor/stage1_target_architecture.md`，将 runtime 最小契约指向 P5a contract，并补齐 `evaluation/`、`predictions/` 或 `prediction_outputs/`、`logs/` 的职责。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P5a contract 文档。

## 结果

本轮定义的新 runtime contract 包含：

- 新 canonical `run_dir` 最小结构：`status.json`、`metadata.json`、`checkpoints/`、`logs/`、`evaluation/`、`predictions/` 或 `prediction_outputs/`。
- 新 `status.json` 最小字段：`status`、`phase`、`updated_at`、`run_dir`、`entrypoint`、`config_name`、`progress`、`latest_checkpoint_path`、`error`。
- 新 `metadata.json` 最小字段：`stage`、`entrypoint`、`config_name`、`args`、`inputs`、`outputs`、`model_columns`、`array_storage`、`feature_schema`、`split_strategy`、`created_at_utc`。
- Visual Router 与 TimeFuse-style fusor 的共享字段和 branch-specific extra 分层：视觉侧记录 pseudo image、ViT、router loss 等参数；TimeFuse 侧记录 feature columns、feature cache shards、feature-only scaler、fusor head、reader/index 参数等。
- P4a/P4b/P4c helper 只作为原子 JSON、路径解析和 metadata-like payload 的底层能力，不反向兼容所有历史 schema。
- checkpoint index 在新 runtime 中只保留 latest 指针和最小恢复线索的概念，本轮不实现 helper。
- 明确舍弃旧非 streaming metadata、LogisticRegression、offline embedding cache、旧 OOM、pilot launcher、prediction cache builder resume 等历史 schema 的强兼容目标。

验收命令均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结论

P5a 已把 canonical runtime contract 从上一轮 architecture pivot 的概要细化为可后续迁移使用的字段契约。下一步应进入 P5b FeatureProvider interface design，继续只写设计，等 runtime 和 feature 边界稳定后再考虑共享 config、checkpoint index helper 或 logging framework。

## 下一步方案

1. 更新 `experiment_logs/README.md` 记录本轮日志和验收结果。
2. 小步提交并 push 到 `refactor/stage1-route-audit` 分支。
3. 后续进入 P5b FeatureProvider interface design。
