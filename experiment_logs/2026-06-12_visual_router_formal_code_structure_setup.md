# Visual Router 正式实验代码结构建立

日志日期：2026-06-12 07:49:45 CST

## 目的

为即将开始的 Visual Router 正式实验建立独立代码目录，并把“按 stage 建立二级目录”的长期规范写入 `AGENTS.md`。

## 背景

此前的 baseline 汇总、oracle 审计和路线设计主要使用 `experiment_scripts/`。随着实验进入正式阶段，继续把 Visual Router 代码堆叠在通用脚本目录中会导致阶段边界不清、复用逻辑分散、后续 Stage 1 / Stage 2 难以维护。因此需要新增专门代码根目录。

## 操作

1. 新增正式实验代码根目录：

   ```text
   visual_router_experiments/
   ```

2. 按 stage 建立二级目录：

   ```text
   visual_router_experiments/common/
   visual_router_experiments/stage0_oracle_audit/
   visual_router_experiments/stage1_vali_test_router/
   visual_router_experiments/stage2_heldout_cell/
   ```

3. 为根目录和各二级目录新增 `README.md`，记录目录职责和后续脚本建议。
4. 更新 `AGENTS.md`，新增“正式视觉路由实验代码目录规范”。
5. 更新 `WORKSPACE_STRUCTURE.md`，记录新目录的层级角色和维护口径。
6. 更新 `experiment_logs/README.md` 总览表。

## 结果

正式 Visual Router 实验代码结构已建立：

| 路径 | 角色 |
| --- | --- |
| `visual_router_experiments/` | Visual Router 正式实验代码根目录 |
| `visual_router_experiments/common/` | 跨阶段公共代码目录 |
| `visual_router_experiments/stage0_oracle_audit/` | oracle 上限审计阶段目录 |
| `visual_router_experiments/stage1_vali_test_router/` | vali->test window-level router 主实验目录 |
| `visual_router_experiments/stage2_heldout_cell/` | held-out cell zero-shot 泛化实验目录 |

`AGENTS.md` 已明确：

- Visual Router / Visual-Conditioned MoE 正式实验代码优先放在 `visual_router_experiments/`；
- 按 stage 建二级目录；
- 跨阶段复用逻辑放入 `common/`；
- 大规模 cache、checkpoint 和输出仍写入 `experiment_logs/run_outputs/`；
- 新增 stage 或正式脚本时同步更新结构文档和实验日志。

## 验证

已验证：

- 新目录和各级 `README.md` 已写入。
- `AGENTS.md` 已包含正式视觉路由实验代码目录规范。
- `WORKSPACE_STRUCTURE.md` 已包含 `visual_router_experiments/` 结构说明。
- `experiment_logs/README.md` 已新增本次结构建立记录。

## 结论

正式实验代码目录已具备基础结构。后续 Stage 1 的 prediction cache、伪图像 tensor/embedding、router 训练和评估脚本应放入 `visual_router_experiments/stage1_vali_test_router/`，跨阶段复用组件应放入 `visual_router_experiments/common/`。

## 下一步方案

1. 阅读 Quito evaluate 数据流，确定 window-level `y_true/y_pred` 导出位置。
2. 在 `visual_router_experiments/common/` 中设计 prediction cache schema。
3. 在 `visual_router_experiments/stage1_vali_test_router/` 中实现小规模 prediction cache pilot。
