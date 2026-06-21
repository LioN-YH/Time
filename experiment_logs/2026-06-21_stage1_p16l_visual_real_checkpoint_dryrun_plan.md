# Stage 1 P16l Visual real checkpoint dry-run plan

日志日期：2026-06-21 16:36:57 CST

## 目的

制定并冻结 real Visual checkpoint dry-run 的前置方案和 guarded explicit-path policy，明确从 P16i/P16j tiny checkpoint payload 过渡到用户显式 real checkpoint dry-run 时的路径、授权、metadata 和分层边界。

## 背景

P16j 已让 Visual small path 可以串联 P16c precomputed feature、P16d optional scaler、P16i tiny checkpoint payload helper、legacy `VisualMLPRouter`、P16a adapter、Evaluator 和 Runtime writer。P16k 已验证默认 mock path 与显式 loaded legacy path 的 artifact parity。

下一步若接近真实 checkpoint，必须先明确真实 checkpoint 只能由用户显式传入，不能从 `/data2`、`run_dir` 或历史输出中自动发现；默认 smoke / CI 仍不能读取真实 checkpoint；checkpoint loading 属于 Runtime / entrypoint，不属于 P16a adapter 或 FeatureProvider。

## 操作

1. 读取 P16l 目标说明、P16k 文档、P16j/P16k roadmap 章节、entrypoint migration plan、`WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`。
2. 新增 `docs/refactor/stage1_visual_real_checkpoint_dryrun_plan.md`，记录 real checkpoint dry-run 的前置条件、三类路径策略、checkpoint payload 策略、候选 CLI、artifact metadata、失败策略和后续拆分。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，追加 P16l 章节，明确本步 docs-only，不读取真实 checkpoint，不访问 `/data2`，不启动 ViT，不修改正式入口，不新增 path guard helper。
4. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`，补充 P16j/P16k/P16l 当前状态，并把 real checkpoint guard 归入 Runtime / entrypoint 边界。
5. 更新 `WORKSPACE_STRUCTURE.md`，把 P16l 文档纳入 `docs/refactor/` 和 roadmap 说明。
6. 更新 `experiment_logs/README.md` 总览表，新增本次 P16l 日志记录。

## 结果

新增方案明确：

- real checkpoint dry-run 必须由用户显式传入 checkpoint path；
- 读取非 fixture checkpoint 必须有用户显式授权；
- 默认 smoke / CI 只允许 tests fixture 或 tempfile tiny payload；
- 不从 `/data2` 自动搜索，不从 `run_dir` 自动推断；
- checkpoint path、allow flag、`torch.load`、strict loading、敏感路径摘要和 dry-run 错误处理属于 Runtime / entrypoint；
- P16a adapter 仍只接收已加载 `torch.nn.Module` 和 head-ready `FeatureBatch`；
- `scaler_state` 可以记录 metadata，但不能 silent transform；
- future artifact metadata 应标记 `loads_real_checkpoint`、`loads_real_vit`、`formal_visual_router_migration`、`checkpoint_is_fixture`、`strict_checkpoint_load`、`scaler_state_present` 和 `scaler_transform_applied` 等字段；
- 后续建议拆分为 P16m path guard smoke、P16n manual real checkpoint dry-run、P17a Visual canonical eval entrypoint migration plan、P17b real Visual feature chain boundary 和 P17c full-scale / pressure plan。

本步没有新增代码、没有新增 smoke、没有读取真实 checkpoint、没有访问 `/data2`、没有启动 ViT、没有启动训练，也没有修改 `train_visual_router_online_streaming.py` 或正式 evaluation 入口。

实际验收命令与结果：

```text
rg -n "explicit|allow|fixture|/data2|run_dir|Runtime|P16a|scaler_state|loads_real_checkpoint|loads_real_vit|formal_visual_router_migration" docs/refactor/stage1_visual_real_checkpoint_dryrun_plan.md
结果：通过。目标关键词均可在 P16l 方案中定位到对应 policy。

git diff --check
结果：通过。

git status --short
结果：仅包含 P16l 新增文档、roadmap、entrypoint migration plan、WORKSPACE_STRUCTURE、实验日志和日志总览；未修改正式训练入口。
```

## 结论

P16l 已把真实 checkpoint dry-run 的显式路径策略冻结为 Runtime / entrypoint 层责任。默认 small smoke 与 CI 仍保持 tiny fixture / tempfile 口径，真实 checkpoint 只能在后续用户显式授权的手动 dry-run 中读取。

## 下一步方案

小步提交并 push 到 `origin/refactor/stage1-route-audit`。后续优先做 P16m explicit real-checkpoint dry-run CLI guard smoke，只验证读取前 path guard，不读取 `/data2`；真实 checkpoint 手动 dry-run 需等用户显式提供 artifact path 后另起 P16n。
