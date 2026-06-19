# Stage 1 P5f Launcher Architecture 设计

日志日期：2026-06-19 22:58:46 CST

## 目的

补充 Stage 1 及未来 Stage 的实验启动层设计，明确未来用户通过 Bash 启动实验时，`exp_scripts/*.sh -> scripts/*.py -> time_router runtime/protocol/provider/head/evaluator` 的分层职责和迁移顺序。

## 背景

P5a-P5e 已经完成 canonical runtime contract、provider interface、protocol types、provider adapter boundary 和 canonical entrypoint migration plan 的文档化设计。现有正式入口仍是 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 和 `launch_timefuse_fusor_full_scale.py`。下一步需要在不修改训练脚本的前提下，先明确未来 Bash launcher、极薄 Python entrypoint、config 和 `time_router` runtime/provider 的边界。

## 操作

1. 新增 `docs/refactor/launcher_architecture.md`。
2. 在新文档中明确 `exp_scripts/` 只负责 Bash launcher、config 选择、GPU/conda/env、logging、`nohup` 或后台运行策略、显式 `run_dir/output_root` 和可复现实验命令，不实现核心训练逻辑。
3. 在新文档中明确 `scripts/` 只作为极薄 Python entrypoint，负责解析 config/CLI、构造 `ExperimentProtocolSpec` 或等价 runtime spec，并调用 future runtime，不实现 provider 读取细节或训练主体逻辑。
4. 在新文档中明确 `time_router/` 负责 runtime contract、protocols、providers、features、heads/models、evaluation 和 IO helper，不知道 Bash 存在，不硬编码 `exp_scripts` 路径，也不决定 full-scale `run_dir`。
5. 在新文档中明确 `configs/` 保存 Stage/config/branch 参数、Visual Router 与 TimeFuse-style fusor branch-specific config，以及 future finetune ViT、joint training、online expert、online TimeFuse feature 等扩展点；full-scale 路径由 launcher 显式传入，不默认写死到 repo 内。
6. 在新文档中明确 `run_dir` 与 `/data2` 边界：full-scale 通常位于 `/data2/syh/Time/...`，但 provider 不决定 run_dir，repo 只保存代码、配置、文档、小 fixture 和 smoke。
7. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P5f launcher architecture design only 阶段。
8. 更新 `docs/refactor/stage1_target_architecture.md`，在目标架构中引用 P5f 启动层设计。
9. 更新 `docs/refactor/stage1_entrypoint_migration_plan.md`，补充 P5f 与 P5e 入口迁移计划的衔接关系。
10. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 `docs/refactor/launcher_architecture.md` 的长期文档职责。
11. 新增本实验日志，并准备同步更新 `experiment_logs/README.md`。

## 结果

本次只产生文档和索引更新，没有新增 Bash 脚本、Python entrypoint、config loader、runtime/run_dir helper 或 provider adapter，也没有修改 `PredictionBatchReader`、`OracleTsfReader`、evaluation、io、protocols、训练脚本、模型结构、loss 或正式输出目录。

P5f 后推荐的低风险顺序为：

1. 先实现 `PredictionCacheExpertProvider` smoke-only adapter。
2. 再实现 evaluator adapter smoke-only。
3. 补最小 config skeleton。
4. 新增 `scripts/stage1` thin entrypoint skeleton。
5. 最后新增 `exp_scripts/stage1` Bash launcher。

验收命令均已在 `quito` 环境通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结论

Stage 1 后续启动层应保持 Bash、Python entrypoint、config 和 `time_router` runtime/provider 的单向依赖关系。Bash 可以负责服务器资源和运行目录编排，但不能重新承载训练逻辑；`time_router` provider 和 runtime 不能反向依赖 Bash 或固定 `/data2` 路径。

## 下一步方案

执行小步提交并推送到远程 `refactor/stage1-route-audit` 分支。后续代码实现应优先从 `PredictionCacheExpertProvider` 的 smoke-only 接入开始，而不是先写完整 Bash launcher 或大而全 config system。
