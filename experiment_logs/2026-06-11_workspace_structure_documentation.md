# 工作区结构说明文档建立

日志日期：2026-06-11 01:24:36 CST

## 目的

随着 QuitoBench baseline 复盘、统计基线评估和多轮实验编排推进，工作区中文件和输出目录持续增加。本次建立统一的工作区结构说明文档，方便后续快速判断每个目录/文件的功能、结果口径和维护责任。

## 背景

当前 `/home/shiyuhong/Time` 下包含 `quito/`、`TimeFuse/` 两个主要代码库，以及 `experiment_scripts/`、`experiment_logs/`、`quito/outputs/` 等本轮实验新增或高频使用目录。用户提出需要创建一个 Markdown 文档介绍每个文件夹/文件的具体功能，并把“新增文件/文件夹后同步记录”的要求写入 `AGENTS.md`。

## 操作

1. 使用 `find`、`du` 和文件列表命令梳理了工作区顶层结构、`quito/`、`TimeFuse/`、`experiment_scripts/`、`experiment_logs/` 和主要输出目录。
2. 新增根目录文档 `WORKSPACE_STRUCTURE.md`，记录顶层目录、关键脚本、日志目录、Quito 代码结构、Quito 输出目录口径和 TimeFuse 代码结构。
3. 更新 `AGENTS.md`，新增“工作区结构文档维护规范”，要求后续新增/删除/移动长期文件或目录时同步维护 `WORKSPACE_STRUCTURE.md`。
4. 更新 `experiment_logs/README.md`，把本次文档整理纳入实验日志追踪表。

## 结果

已新增 `WORKSPACE_STRUCTURE.md`。其中明确记录：

- `quito/outputs/default_baseline/` 中 DLinear、PatchTST、CrossFormer 的原始 evaluate 结果采用 validation MAE-best checkpoint。
- `quito/outputs/default_baseline_mse_best/` 是后补 validation MSE-best 复盘口径，目前只包含 `PatchTST 576_288_S` 的补评估。
- `experiment_scripts/` 下四个本地编排/复盘脚本的职责。
- `experiment_logs/run_outputs/` 作为脚本运行目录，保存 `status.json`、生成配置、日志、汇总 CSV 和部分 cluster 分析产物。

## 结论

工作区现在有了统一结构索引，后续新增正式脚本、实验日志、输出根目录、数据产物或结果汇总时，应同步更新 `WORKSPACE_STRUCTURE.md`，并在必要时记录实验日志。

## 下一步方案

继续推进五个模型三组配置的结果汇总时，若新增汇总 CSV/Markdown、TSF cell 表或新的分析目录，需要同步更新 `WORKSPACE_STRUCTURE.md` 和对应实验日志。
