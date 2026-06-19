# Stage 1 P10a Shared Prediction SQLite Backend Audit

日志日期：2026-06-20 03:26:14 CST

## 目的

在 P9f 完成 Visual Router training/evaluation `ExpertBatch` 旁路对齐之后，对 Visual Router 与 TimeFuse-style fusor 两条正式入口中的 prediction / oracle SQLite backend、index prepare、packed array batch loading 和 `ExpertProvider` 边界做文档化审计。

本步只做审计和文档更新，不抽 shared helper 代码，不修改正式入口行为。

## 背景

P9 已确认 Visual Router evaluation 侧可通过 `ExpertBatch + EvaluationInputAdapter` 旁路复算 hard/raw-soft 指标，training 侧可通过 `ExpertBatch.y_pred/y_true` 旁路复算 `fusion_huber_kl` 所需 `expert_errors`。

同时，P9e 已确认 `PredictionCacheExpertProvider` 当前不应直接替换 Visual Router 的 `SQLitePredictionIndex`：SQLite 是 full-scale prediction backend implementation，不是 framework interface；index prepare、run_dir、metadata、status 属于 runtime，不属于 provider。

因此 P10a 需要先把 shared prediction SQLite backend 的职责边界写清楚，再决定后续是否抽 smoke helper。

## 操作

1. 阅读用户目标文件 `/home/shiyuhong/.codex-tianyu/attachments/12428929-a615-4c6f-8a2d-b8808c8decfc/pasted-text-1.txt`，确认本步边界为纯文档化审计。
2. 只读审查 Visual Router 当前 SQLite prediction path：
   - `required_prediction_sample_keys(...)`
   - `build_lightweight_prediction_index(...)`
   - `SQLitePredictionIndex.fetch_records(...)`
   - `load_prediction_tensors_from_lightweight_index(...)`
   - `prediction_manifest_index.sqlite`
   - `fusion_huber_kl` 的 `expert_errors`
   - evaluation 侧 soft lookup / adapter bypass 过渡
3. 只读审查 TimeFuse-style fusor 当前 SQLite path：
   - feature-only scaler
   - feature shard split 过滤下推
   - shard-local oracle / prediction SQLite index
   - batch query
   - packed npy 按 batch 内 path 分组 mmap 切片
   - feature batch 通过 `sample_key` 查询 oracle / prediction backend
4. 新增 `docs/refactor/shared_prediction_sqlite_backend_audit.md`。
5. 更新以下路线与结构文档：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_target_architecture.md`
   - `WORKSPACE_STRUCTURE.md`
6. 新增本实验日志，并更新 `experiment_logs/README.md` 总览表。

## 结果

新增审计文档明确：

- Visual Router 当前 SQLite path 仍承担 required sample_key 推导、manifest chunk scan、SQLite 子集索引、batch query、run_dir 下 `prediction_manifest_index.sqlite`、index metadata、packed row index 读取、`fusion_huber_kl` `expert_errors` 和 eval soft lookup 兼容。
- TimeFuse-style fusor 当前 path 已具备 feature-only scaler、split 过滤提前到 feature CSV、shard-local oracle/prediction SQLite、batch query 和 packed npy grouped mmap，但其 reader 仍混合了 TimeFuse feature、oracle supervision 和 prediction backend。
- shared backend / index prepare 应只承担 manifest chunk scan、target sample_keys、SQLite 子集索引、batch fetch records、packed row index lineage、grouped mmap loading、index metadata 和 atomic replace / cleanup。
- Visual Quito history window / pseudo image / ViT、TimeFuse 17 维 feature streaming / scaler、training loss / optimizer / checkpoint、status / metadata / CSV、launcher / Bash / `/data2` 和 oracle deployable feature 边界都不应放进 shared backend。
- `ExpertProvider` 边界仍是 `load_batch(sample_keys) -> ExpertBatch`，可以消费 prepared backend，但不创建 run_dir、不写 status/metadata、不知道 Bash 或 `/data2`。
- prediction backend 可进入 `ExpertProvider`；oracle 只用于监督、诊断、baseline 和 upper-bound，不进入 deployable `FeatureProvider`。

本轮没有修改 Visual Router / TimeFuse 正式入口，没有修改 launcher，没有抽 shared helper，没有修改 `PredictionBatchReader`、`PredictionCacheExpertProvider` 或 `EvaluationInputAdapter`，没有新增 Bash/scripts，没有访问 `/data2`，没有启动 pressure/full-scale。

## 结论

Visual Router 和 TimeFuse-style fusor 已经收敛到同一类 full-scale prediction backend implementation：调用方先确定 sample_keys，再构建 SQLite 子集索引，训练/评估 batch 只查询当前 sample_keys，并通过 packed row index 分组 mmap 读取 `y_pred/y_true`。

但 shared backend 的边界必须保持狭窄。它应是 prepared prediction SQLite backend / index prepare，而不是 runtime、provider、feature、loss、oracle deployable feature 或 launcher。

## 下一步方案

1. P10b 可先抽 shared index prepare smoke helper，锁定 SQLite prepare / fetch / metadata / atomic replace / grouped loading 行为。
2. P10c 再整理 launcher / run scripts 边界，避免 provider 接手 run_dir、status、metadata、resume 或后台运行职责。
3. `PredictionCacheExpertProvider` 消费 prepared backend 的正式接入推迟到 Stage 1.5 / Stage 2，并在 smoke + pressure 后再考虑进入正式入口。

## 验收结果

本轮文档编辑完成后，目标验收命令均已通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

其中四个 smoke 分别确认 Visual Router training ExpertBatch 旁路、Visual Router evaluation adapter 旁路、TimeFuse protocol chain 和 `PredictionCacheExpertProvider` contract 均保持绿色；`compileall` 通过确认本轮没有引入语法或导入级错误。
