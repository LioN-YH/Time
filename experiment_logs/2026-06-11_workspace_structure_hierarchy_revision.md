# 工作区结构说明文档层次化改写

日志日期：2026-06-11 01:28:39 CST

## 目的

根据用户建议，将 `WORKSPACE_STRUCTURE.md` 从较长的平铺表格改写为分层结构，便于后续新增文件和目录时按层级维护。

## 背景

初版 `WORKSPACE_STRUCTURE.md` 已记录工作区主要目录、关键文件和输出口径，但整体更接近平铺清单。随着实验时间线继续拉长，平铺清单不利于快速定位新增内容应归入哪个层级。

## 操作

1. 重写 `WORKSPACE_STRUCTURE.md` 的组织方式。
2. 将内容拆分为工作区根目录层、实验管理层、QuitoBench/Quito 代码与实验层、TimeFuse 代码层、生成物和缓存层。
3. 保留并强化 `quito/outputs/default_baseline/` 的 validation MAE-best 口径说明，以及 `default_baseline_mse_best/` 的 MSE-best 补评估说明。
4. 更新 `experiment_logs/README.md`，记录本次结构文档层次化改写。

## 结果

`WORKSPACE_STRUCTURE.md` 现在采用层级编号和局部树形结构，后续新增脚本、日志、输出目录、结果汇总或缓存说明时，可以直接补到对应层级下。

## 结论

分层后的结构文档更适合长期维护，也更适合区分“正式结果来源”“编排运行产物”和“可再生成缓存”。

## 下一步方案

后续继续汇总五个模型三组配置结果时，若新增结果表或 TSF cell 分析文件，应补充到 `2.3`、`3.7` 或 `5.1` 的相应层级，并同步记录实验日志。
