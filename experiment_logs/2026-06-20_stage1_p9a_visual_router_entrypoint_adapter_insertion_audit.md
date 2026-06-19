# Stage 1 P9a Visual Router 正式入口 adapter 插入点审计

日志日期：2026-06-20 01:49:59 CST

## 目的

在 P8d TimeFuse baseline parity review 之后，回到 Visual Router 主线，审计正式入口 `train_visual_router_online_streaming.py` 的最小 adapter 接入点，并形成只改文档、不改正式训练行为的 P9a 接入计划。

## 背景

当前 Stage 1 已有 `PredictionCacheExpertProvider`、`ExpertBatch` 和 `EvaluationInputAdapter` 等 smoke-only adapter。TimeFuse P8a-P8c 已验证可以在 TimeFuse-style fusor evaluation 阶段旁路接入 `EvaluationInputAdapter` 做一致性校验。Visual Router 入口更复杂，包含 Quito 历史窗口读取、在线 pseudo image、冻结 ViT 前向、MLP router、`fusion_huber_kl` loss、checkpoint/status/metadata 和历史 CSV schema，因此本步先做文档化审计。

## 操作

1. 读取用户粘贴目标文件，确认本步边界为只做 P9a 文档审计，不修改 `train_visual_router_online_streaming.py`，不新增 smoke/Bash/scripts，不访问 `/data2`，不启动 pressure/full-scale。
2. 核对当前分支为 `refactor/stage1-route-audit`。
3. 只读审查 `train_visual_router_online_streaming.py` 的函数边界，重点查看 CLI、SQLite prediction index、Quito history window、online embedding、ViT forward、训练 batch、预测 batch、CSV summary、checkpoint/status/metadata 等职责。
4. 新增 `docs/refactor/visual_router_entrypoint_adapter_insertion_audit.md`，记录 P9a 审计结论。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md`、`docs/refactor/stage1_target_architecture.md` 和 `WORKSPACE_STRUCTURE.md`，同步登记 P9a。
6. 运行目标验收命令，验证 smoke 与 compileall。

## 结果

新增 P9a 审计文档，主要结论如下：

- `train_visual_router_online_streaming.py` 当前同时承担 sample/manifest/prediction 读取、Quito history window、online pseudo image、ViT forward、Visual MLP router、loss/optimizer、evaluation rows/summary、checkpoint/status/metadata/run_dir。
- `PredictionCacheExpertProvider / ExpertBatch` 应优先作为专家输出 contract 规划，但 P9b 不应直接替换正式入口的 SQLite prediction index、batch query、packed row index 单行读取和 `fusion_huber_kl` expert error 计算。
- 第一批最小接入点应放在 test evaluation batch，旁路使用 `EvaluationInputAdapter.evaluate_input(...)` 或临时 `ExpertBatch + RouterOutput` 复算 hard top-1、raw soft fusion 和 weight diagnostics。
- P9b 不应改变 `visual_router_predictions.csv`、soft fusion predictions、summary、comparison、selected counts、metadata、status 或 checkpoint schema。
- Visual FeatureProvider / ViT provider 暂不作为第一批最小 adapter，因为该路径绑定 Quito 数据读取、pseudo image、ViT/Hugging Face cache、GPU dtype、DataParallel、latency、scaler 和 metadata。
- 对比 TimeFuse P8a-P8c，Visual Router feature/provider/head 更重，迁移应更保守。

验收结果：

- `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py` 通过；输出 hard MAE `1.093573928`、raw soft MAE `0.556751269`。
- `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py` 通过；输出 hard MAE `0.416048437`、raw soft MAE `0.410296679`。
- `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 通过。
- `git diff -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 无输出，确认本步未修改正式 Visual Router 入口。

## 结论

P9a 已完成文档化接入计划。下一步 P9b 若进入代码迁移，应只新增显式 flag 控制的 evaluation 旁路一致性校验，默认保持正式 full-scale 行为不变；不得迁移 feature extraction、ViT、training loop、router head 或正式 output schema。

## 下一步方案

提交并推送 `refactor/stage1-route-audit`。P9b 若改代码，只做显式 flag 控制的 evaluation 旁路一致性校验。
