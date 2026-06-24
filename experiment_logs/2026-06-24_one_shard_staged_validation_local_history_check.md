# one-shard staged validation 本地历史核查

日志日期：2026-06-24 03:14:05 CST

## 目的

核查本地是否已经完成过 `Add one-shard staged fullscale validation` 目标，确认是否存在对应 git 提交、实验日志、轻量 summary 和 `/data2` 正式输出目录。

## 背景

用户需要判断之前本地是否已经干过 one-shard staged full-scale validation。该目标属于 Visual Router V2 Round2 staged full-scale validation 链路，关键判断依据应包括 git 历史、实验日志总览、结构文档、轻量 summary 和外部输出目录状态。

## 操作

1. 使用 `rg` 搜索 `one-shard`、`one_shard`、`staged fullscale`、`fullscale validation` 等关键词，覆盖代码、实验日志、summary 和结构文档。
2. 使用 `git log --all --grep` 检索相关提交，命中提交 `2e9d368 Add one-shard staged fullscale validation`。
3. 使用 `git show --stat --name-only 2e9d368` 查看提交时间、改动文件和入仓轻量产物。
4. 检查 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_one_shard_staged_validation/` 是否存在，并查看目录内容、`status.json`、metadata 和 summary。
5. 检查 `experiment_logs/README.md`、`WORKSPACE_STRUCTURE.md`、`EXTERNAL_OUTPUTS.md` 中是否已有对应索引记录。

## 结果

1. 本地 git 历史存在提交 `2e9d368ba324daba049de0b2ab03440d72b04724`，提交信息为 `Add one-shard staged fullscale validation`，提交时间为 `2026-06-22 13:10:31 +0800`。
2. 该提交新增或更新 19 个文件，包括 `EXTERNAL_OUTPUTS.md`、`WORKSPACE_STRUCTURE.md`、`experiment_logs/2026-06-22_visual_router_v2_round2_one_shard_staged_validation.md`、`experiment_logs/README.md` 和 `experiment_summaries/visual_router_v2_round2/one_shard_staged_validation/` 下的 CSV/JSON/Markdown 轻量 summary。
3. `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_one_shard_staged_validation/` 仍存在，包含 feature cache、subset SQLite、tasks、status、metadata 和各类 report。
4. `status.json` 显示 `status=completed`，best layout 为 `spatial_panel_3view`，backend 为 `film_mean_patch_aux`，selection basis 为 `staged_selection raw-soft MAE_mean`。
5. 原实验日志记录 one-shard dry-run 与正式 execution 均完成；四个 staged sample set 各 512 条，总计 2048 unique sample_key；prediction lookup 覆盖 `2048 × 5 = 10240` records；两个 layout 的 seed16 fixed FiLM train/eval 均完成；metadata 明确这是 pipeline validation，不是 1M 或 116M 正式长跑。

## 结论

本地已经完成过 `Add one-shard staged fullscale validation` 目标。该工作不只是计划或代码脚手架，而是包含已完成的 one-shard execution、正式 `/data2` 输出、入仓轻量 summary、实验日志和结构文档索引。

## 下一步方案

后续如果继续推进，应从 one-shard 之后的阶段接手：先设计并验证 multi-shard staged manifest 和小规模 smoke，再进入 1M staged validation；不要把 one-shard 指标当作 1M 或 116M 正式性能结论。
