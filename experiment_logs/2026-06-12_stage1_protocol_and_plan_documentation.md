# Stage 1 协议与任务规划文档固化

日志日期：2026-06-12 23:35:44 CST

## 目的

将 Stage 1 后续实验协议、per-config 主实验口径、跨 config 迁移实验定位和下一步任务写入对应代码目录，避免阶段性规划散落在对话或总览日志中。

## 背景

讨论后确认，不同历史-未来 config 的 router 不能在部署时自由跨 config 选择专家。例如 `96_48_S` 样本不能选择 `1024_512_S` 专家预测，因为输入长度、输出长度和 checkpoint 不匹配。因此 Stage 1 主实验应按 config 分开训练和评估；跨 config 数据更适合作为 shared encoder 或迁移学习实验，而不是单一混合动作空间 router。

## 操作

1. 新增 Stage 1 阶段性协议与任务规划文档：

   ```text
   visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md
   ```

2. 文档中明确：
   - 主实验是 per-config router；
   - 每个 config 的合法动作空间只包含同 config 下五专家；
   - mixed-config 不作为主实验可部署 router；
   - Stage 1B 使用 shared encoder + config-specific heads；
   - leave-one-config-out 用于评估结构表示迁移能力；
   - 下一步先更新 baseline evaluator，再跑通 `96_48_S` visual feature 和 per-config router。
3. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`，加入新文档索引。
4. 更新 `AGENTS.md`，固化“阶段性协议、实验规划、任务拆解和路线变更应优先写入对应 stage 代码目录下 Markdown 文档”的规范。
5. 更新 `WORKSPACE_STRUCTURE.md`，记录 `stage1_protocol_and_plan.md`。

## 结果

Stage 1 现在有明确的长期规划文档，后续实现正式脚本时可以直接按该文档推进。项目级协作规范也已要求后续阶段继续把阶段性规划放在对应 stage 目录下。

## 结论

Stage 1 主线确定为 per-config router；跨 config 只作为迁移学习或表征学习扩展。这个口径已经写入代码目录文档和项目级规范，后续 baseline、cache、router、summary 脚本都应遵守。

## 下一步方案

1. 优先改造 `evaluate_router_baselines.py`，默认按 `config_name` 分层学习和汇总。
2. 继续实现 `96_48_S` visual feature / embedding cache。
3. 在 `96_48_S` 上训练最小 per-config router，并和非视觉 baseline 同表评估。
