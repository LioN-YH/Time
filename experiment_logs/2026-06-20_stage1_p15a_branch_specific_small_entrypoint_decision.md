# Stage 1 P15a branch-specific small entrypoint decision

日志日期：2026-06-21 00:03:57 CST

## 目的

基于 P13d/P13e/P14a-P14f 的结果，新增 Stage 1 branch-specific small entrypoint decision
文档，决策后续是否需要分别新增 TimeFuse-specific small canonical entrypoint 和
Visual-specific small canonical entrypoint。

## 背景

截至 P14f，Stage 1 canonical 协议骨架已经在 tiny/smoke 级别打通：

- P13d 已证明 prediction backend 可以输出 `ExpertBatch`。
- P13e 已证明 TimeFuse 17 维 `FeatureProvider` 可以输出 canonical `FeatureBatch`。
- P14a-P14f 已完成 Visual FeatureProvider 插入点审计、Visual mock provider smoke、
  Visual eval-only canonical bypass plan、Visual mock protocol eval smoke、Visual legacy MLP
  adapter audit 和 smoke-only adapter pattern 验证。

当前正式入口尚未迁移，真实 Visual ViT provider、正式 legacy `VisualMLPRouter` adapter、
正式 `SupervisionProvider` 和 branch-specific small entrypoint 也尚未新增。本步按计划只做
文档决策，不新增入口、不写 scripts、不启动训练。

## 操作

1. 读取任务说明，确认 P15a 范围为 docs-only 决策步。
2. 检查当前分支为 `refactor/stage1-route-audit`，并确认工作区起点干净。
3. 查阅相邻文档：
   - `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`
   - `docs/refactor/stage1_visual_legacy_mlp_adapter_smoke.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
4. 新增 `docs/refactor/stage1_branch_specific_small_entrypoints.md`，记录 P15a 决策。
5. 更新 `experiment_logs/README.md` 总览表，登记本实验日志。
6. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P15a 决策文档。
7. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md` 和
   `docs/refactor/stage1_refactor_roadmap.md`，把 P15a 决策接到后续 P15b/P15c 路线。

## 结果

P15a 决策结论如下：

- P14 可以收束。
- `scripts/run_stage1_canonical_small.py` 必须继续保持 generic thin CLI，只服务通用 tiny
  fixture 和 canonical dataflow 最小验证。
- 后续需要新增 TimeFuse-specific small canonical entrypoint，推荐未来命名为
  `scripts/run_stage1_timefuse_small.py`，但不在 P15a 实现。
- 后续需要新增 Visual-specific small canonical entrypoint，推荐未来命名为
  `scripts/run_stage1_visual_small.py`，但应分阶段推进，也不在 P15a 实现。
- P15b 建议先做 TimeFuse-specific thin slice，因为 TimeFuse-style fusor 是正式 baseline
  支线，且 17 维 feature/head/runtime 组合相对稳定。
- P15c 建议再做 Visual-specific thin slice，初期使用 `VisualMockFeatureProvider` 与
  smoke/legacy MLP adapter pattern，不加载真实 checkpoint、不接真实 ViT。

本步没有新增代码、脚本、provider/head/runtime core，没有访问 `/data2`，没有读取真实
checkpoint，没有启动 ViT embedding、训练、pressure 或 full-scale。

## 结论

P15a 已完成 branch-specific small entrypoint decision。后续不应把 Visual / TimeFuse 的
provider/head 细节塞回 `scripts/run_stage1_canonical_small.py`；generic small CLI 继续作为公共
协议骨架的 thin slice。TimeFuse 与 Visual 两条路线后续分别通过 branch-specific small entrypoint
验证 feature/head/evaluation/runtime artifact 组合边界。

## 下一步方案

1. P15b：新增 TimeFuse-specific small canonical entrypoint thin slice，使用已有 small fixture
   或 real-derived small input，写 canonical `run_dir`，不访问 `/data2`，不启动正式训练。
2. P15c：新增 Visual-specific small canonical entrypoint thin slice，先使用
   `VisualMockFeatureProvider + smoke/legacy MLP adapter pattern`，写 canonical `run_dir`，
   不加载真实 checkpoint，不接真实 ViT。
3. 后续单独设计正式 legacy `VisualMLPRouter` adapter smoke 和 real Visual feature /
   embedding provider audit。

## 验证

本步是 docs-only 决策步，未运行 Quito smoke 或 compileall。原因是本次只新增/更新文档和日志，
不新增代码、不修改脚本、不改变 runtime 行为。验证重点改为检查 diff 范围和关键标题：

```bash
git diff --name-only
rg -n "P14 可以收束|generic small CLI|TimeFuse-specific|Visual-specific|P15b|P15c" docs/refactor/stage1_branch_specific_small_entrypoints.md
```
