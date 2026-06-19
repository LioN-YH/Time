# Stage 1 目标架构与重构路线图文档

日志日期：2026-06-19 16:26:07 CST

## 目的

为 Stage 1 后续重构补齐目标架构设计和小步迁移路线图，并把 `tests/smoke/stage1_golden_smoke.py` 固定为后续 reader、SQLite、metrics 和 output schema 迁移的前后验收门禁。

## 背景

当前 `refactor/stage1-route-audit` 已有路线审计、迁移候选、golden fixture 说明和 golden smoke。新的要求是只新增架构文档，不迁移正式代码、不新增 package、不新增空目录、不做 import 改动；同时需要运行 golden smoke、检查 git 状态、写中文实验日志，并以指定 commit message 提交推送。

## 操作

1. 阅读用户粘贴目标文件，确认新增文档范围为：
   - `docs/refactor/stage1_target_architecture.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
2. 阅读已有文档：
   - `docs/refactor/stage1_route_audit.md`
   - `docs/refactor/stage1_migration_candidates.md`
   - `docs/refactor/golden_fixture.md`
   - `tests/smoke/stage1_golden_smoke.py`
3. 新增 `stage1_target_architecture.md`，明确未来 `time_router/data/`、`time_router/io/`、`time_router/features/`、`time_router/models/`、`time_router/evaluation/`、`time_router/training/`、`scripts/`、`configs/`、`exp_scripts/` 和 `archive/` 边界。
4. 新增 `stage1_refactor_roadmap.md`，按 P0-P6 拆分后续 commit 顺序：
   - P0：architecture docs only；
   - P1：extract prediction batch reader；
   - P2：extract oracle/TSF reader；
   - P3：extract metrics/fusion；
   - P4：extract logging/path/config；
   - P5：introduce FeatureProvider interface；
   - P6：migrate visual router and TimeFuse fusor entrypoints。
5. 在两份文档中明确共享主干包括 manifest reader、prediction cache reader、oracle/TSF reader、SQLite index、metrics/fusion/report，并明确两个分支：
   - `VisualFeatureProvider: x -> pseudo image -> frozen ViT embedding`
   - `TimeFuseFeatureProvider: sample_key -> 17维 feature cache`
6. 在两份文档中明确未来进入 `archive/` 的旧代码类别：offline embedding cache、logistic regression fusor、old OOM routes、pilot-only scripts。
7. 更新 `WORKSPACE_STRUCTURE.md` 中 `docs/refactor/` 条目，登记新增长期文档。
8. 使用 `quito` 环境运行：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

9. 运行 `git status --short --branch` 检查当前改动范围。

## 结果

golden smoke 通过，关键输出为：

```text
通过：五专家顺序固定为 ['DLinear', 'PatchTST', 'CrossFormer', 'ES', 'NaiveForecaster']
通过：sample_key 顺序固定，sample_count=4
通过：y_pred shape=(4, 5, 48, 1)，y_true shape=(4, 48, 1)
通过：hard top-1 选择=['CrossFormer', 'DLinear', 'PatchTST', 'DLinear']，MAE=0.416048437，MSE=0.456369758
通过：raw soft fusion MAE=0.410296679，MSE=0.488154024
完成：Stage 1 golden smoke 全部通过
```

`git status --short --branch` 显示当前变更集中在文档和结构说明：

```text
## refactor/stage1-route-audit...origin/refactor/stage1-route-audit
 M WORKSPACE_STRUCTURE.md
?? docs/refactor/stage1_refactor_roadmap.md
?? docs/refactor/stage1_target_architecture.md
```

本轮未实现 `time_router/` package，未新增空目录，未做 import 改动，未移动、删除、重命名或改写正式代码。

## 结论

Stage 1 目标架构和后续重构路线图已文档化。后续迁移应从共享数据平面开始，小步提交，并在每次抽 reader、SQLite、metrics 或 output schema 前后运行 `tests/smoke/stage1_golden_smoke.py`。

## 下一步方案

按 `docs/refactor/stage1_refactor_roadmap.md` 从 P1 开始抽 prediction batch reader；启动任何代码迁移前先运行 golden smoke 记录基线，通过后再做最小范围改动。
