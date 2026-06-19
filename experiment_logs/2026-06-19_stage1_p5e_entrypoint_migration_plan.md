# Stage 1 P5e canonical entrypoint migration plan

日志日期：2026-06-19 22:20:32 CST

## 目的

基于已完成的 P5a runtime contract、P5b provider interface、P5c protocol types 和 P5d provider adapter boundary review，补齐 Stage 1 Visual Router 与 TimeFuse-style fusor 两条 canonical entrypoint 的迁移路线文档。

本步骤只写文档，不改训练代码，不实现 provider adapter，不迁移入口，不创建 `run_dir`，不接入 `/data2`。

## 背景

P5d 已明确后续最小 adapter 实现前需要先写 entrypoint migration plan，避免直接把现有训练脚本整体改名为 provider。当前正式主线仍是：

- Visual Router：`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- TimeFuse-style fusor：`visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- TimeFuse full-scale launcher：`visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`

这三个入口当前都包含较多 runtime、I/O、模型、评估和产物写出逻辑，需要先文档化拆分边界。

## 操作

1. 读取用户粘贴的 P5e 任务说明，确认验收命令和明确不做范围。
2. 检查当前 Git 分支和工作树状态，确认位于 `refactor/stage1-route-audit` 分支且起始工作树干净。
3. 读取 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md`、`docs/refactor/stage1_canonical_runtime_contract.md`、`docs/refactor/stage1_provider_interface.md` 和 `docs/refactor/provider_adapter_boundary.md`。
4. 只读审查 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py`、`launch_timefuse_fusor_full_scale.py` 和 `stage1_timefuse_fusor_streaming_reader.py` 的函数边界、入口主流程和输出写出逻辑。
5. 新增 `docs/refactor/stage1_entrypoint_migration_plan.md`，按 runtime orchestration、ExpertProvider、FeatureProvider、RouterHead、Evaluator 和 launcher/runtime 边界拆分两条 canonical entrypoint。
6. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P5e 小步，明确本阶段完成范围和不做范围。
7. 更新 `docs/refactor/stage1_target_architecture.md`，补充 P5e 迁移结论和两条分支的下沉顺序。
8. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P5e 文档。
9. 新增本中文实验日志，并准备同步更新 `experiment_logs/README.md`。

## 结果

新增 P5e 迁移计划文档，核心结论如下：

- Visual Router 入口暂时保留 CLI/runtime/checkpoint/status/metadata/文件写出职责；prediction cache 读取未来下沉到 `PredictionCacheExpertProvider`，在线 pseudo image / ViT 前向下沉到 `VisualOnlineVitFeatureProvider`，MLP head 下沉到 Visual RouterHead，hard/raw-soft 指标和 rows/summary 复算下沉到 Evaluator。
- TimeFuse fusor 入口暂时保留 shard 准备、scaler fit、epoch/eval 编排、checkpoint/status/metadata/报告写出职责；prediction tensor 读取下沉到 `PredictionCacheExpertProvider`，17 维 feature cache streaming 下沉到 `TimeFuseFeatureCacheProvider`，`nn.Linear -> softmax` 下沉到 TimeFuse RouterHead，评估复算下沉到 Evaluator。
- `launch_timefuse_fusor_full_scale.py` 被明确为 preflight、脚本生成、后台 PID/PGID、stop/resume 和接手信息层，不实现 provider adapter，不决定 provider 内部行为。
- 第一批代码迁移建议顺序为：`PredictionCacheExpertProvider`、evaluator adapter、`TimeFuseFeatureCacheProvider`、TimeFuse linear-softmax head、Visual online ViT feature provider、Visual head。
- 新 adapter 先接入 smoke，不直接替换 full-scale streaming 入口；正式入口后续小步接入。
- 迁移计划不创建 `run_dir`，provider/entrypoint plan 不硬编码 `/data2`。

## 验证

已在 `quito` 环境完成全部验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

验证结果：

- `stage1_golden_smoke.py` 通过；确认五专家顺序、4 个 sample_key 顺序、`y_pred=(4,5,48,1)`、`y_true=(4,48,1)`、hard top-1、raw soft fusion、summary 和 per-sample rows 均未漂移。
- `stage1_oracle_tsf_smoke.py` 通过；确认 `allow_full_scan` 默认禁止、oracle/TSF 保序 join、缺失策略和冲突重复检查正常。
- `stage1_json_utils_smoke.py` 通过。
- `stage1_path_resolver_smoke.py` 通过。
- `stage1_run_metadata_smoke.py` 通过。
- `stage1_protocol_types_smoke.py` 通过。
- `compileall time_router tests/smoke` 通过。

## 结论

P5e 的文档化迁移路线已经补齐，后续可以在不破坏当前 streaming 入口的前提下，先从 `PredictionCacheExpertProvider` 和 evaluator adapter 两个最低风险方向进入代码迁移。

本步骤未修改任何训练脚本、模型结构、loss、正式输出目录、reader/evaluation/io helper 或 protocol types。

## 下一步方案

1. 检查 `git diff --name-only`，确认变更只包含文档、实验日志和结构索引。
2. 小步提交并 push 到远程 `refactor/stage1-route-audit` 分支。
3. 后续代码迁移从 `PredictionCacheExpertProvider` 的 smoke-only 接入开始。
