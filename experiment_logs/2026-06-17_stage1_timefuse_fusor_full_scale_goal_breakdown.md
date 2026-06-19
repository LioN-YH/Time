# Stage 1 TimeFuse-style Fusor Full-Scale 目标拆分

日志日期：2026-06-17 19:33:30 CST

## 目的

将 `96_48_S` full-scale TimeFuse-style fusor baseline 拆分为适合多窗口 goal 模式推进的小目标，避免一次性实现过大、重复视觉路由 OOM 问题。

## 背景

full-scale prediction cache、oracle labels、TSF enrichment 和 TimeFuse feature cache 均已具备。现有 fusor pilot 入口可复刻 TimeFuse 单层 `Linear -> softmax -> weighted fusion -> SmoothL1Loss` 口径，但实现依赖全量 CSV/DataFrame 和全量 prediction lookup，不适合直接处理 full-scale `116,375,850` 行五专家 manifest。

## 操作

1. 只读检查 `train_visual_router_online_streaming.py` 中 OOM 修复后的 SQLite prediction index、batch 查询、checkpoint 和 status 设计。
2. 只读检查 `fusion_utils.py` 中 TimeFuse-style fusor 当前训练、预测、soft fusion 和 prediction lookup 的实现方式。
3. 阅读 `HANDOFF.md` 中视觉路由 OOM 经过和当前 v2 长跑状态，提炼 full-scale fusor 应复用的工程约束。
4. 形成按窗口推进的 goal 拆分：先做只读设计，再做索引/reader，随后做训练 smoke、评估 streaming、launcher 和最终正式运行。

## 结果

拆分建议为 6 个小目标：

1. full-scale fusor 设计与输入契约冻结；
2. streaming/shard-aware reader 与 SQLite/memmap 索引实现；
3. 单 shard / 双 shard 训练 smoke；
4. test streaming 评估与 summary 输出；
5. 后台 launcher、resume/checkpoint/status 和文档日志完善；
6. 正式 full-scale 启动与监控。

## 结论

TimeFuse-style fusor full-scale 的主要风险不是模型计算，而是数据对齐、I/O、内存和恢复语义。实现时必须避免全量 manifest lookup、全量 test predictions、全量 feature/label DataFrame 常驻内存；应参考视觉路由 v2 的 SQLite batch 查询和 status/checkpoint 设计。

## 下一步方案

用户后续可按拆分 goal 逐个启动新窗口。每个 goal 完成后都应写中文实验日志，并更新 `experiment_logs/README.md`；涉及新增正式脚本或长期输出目录时同步更新 `WORKSPACE_STRUCTURE.md`。
