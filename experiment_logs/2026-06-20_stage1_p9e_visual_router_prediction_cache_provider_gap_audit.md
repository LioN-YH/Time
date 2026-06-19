# Stage 1 P9e Visual Router PredictionCacheExpertProvider full-scale gap audit

日志日期：2026-06-20 03:05:31 CST

## 目的

审计现有 `PredictionCacheExpertProvider` / `PredictionBatchReader` 与 Visual Router
正式入口 SQLite prediction path 的能力差距，明确哪些能力应归属 provider，哪些能力应保留在
runtime、SplitStrategy、index prepare 或 launcher/report 层。

本步骤只做架构审计和迁移计划，不替换正式入口，不新增 full-scale provider 实现。

## 背景

P9a 已完成 Visual Router entrypoint adapter insertion audit。P9b/P9c 已完成
默认关闭的 `--verify-evaluation-adapter` 旁路校验和小规模 pressure 验证。P9d 已把
Visual Router evaluation bypass 输入从直接 `EvaluationInput` 收敛为
`ExpertBatch + fusion_weights`。

当前还不能把 P9d 误解为 `PredictionCacheExpertProvider` 已可替换正式
`SQLitePredictionIndex`。Visual Router full-scale 入口仍承担 required sample_key 推导、
大 manifest chunk scan、SQLite 子集索引、batch query、packed row index 读取、训练 loss
所需 `expert_errors`、eval raw soft fusion lookup 和 index runtime artifact metadata。

## 操作

1. 阅读目标文件：
   `/home/shiyuhong/.codex-tianyu/attachments/f873414b-24a2-4f68-80a2-2862ee9be296/pasted-text-1.txt`。
2. 只读审计以下实现和文档：
   - `time_router/experts/prediction_cache.py`
   - `time_router/io/prediction_cache_reader.py`
   - `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
   - `docs/refactor/prediction_cache_expert_provider.md`
   - `docs/refactor/visual_router_expert_batch_evaluation_bridge.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_target_architecture.md`
3. 新增审计文档：
   - `docs/refactor/visual_router_prediction_cache_provider_gap_audit.md`
4. 更新迁移路线与结构索引：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_target_architecture.md`
   - `WORKSPACE_STRUCTURE.md`
5. 运行验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```

## 结果

新增 P9e 审计文档后，明确记录：

- `PredictionCacheExpertProvider` 当前已具备显式 `sample_keys` batch 输入、固定五专家
  `model_columns`、`y_pred/y_true`、`row_index_metadata`、`verify_metrics`、
  `packed_npy_v1` / `per_sample_npy` 和 `ExpertBatch` 输出能力。
- Visual Router 正式 SQLite path 仍承担 `required_prediction_sample_keys(...)`、
  `build_lightweight_prediction_index(...)`、大 manifest chunk scan、只为 required
  sample_keys 建 SQLite 子集索引、`prediction_index.fetch_records(...)`、batch-level
  packed row index 读取、`fusion_huber_kl` 训练 loss 所需 `expert_errors`、eval raw soft
  fusion lookup、`prediction_manifest_index.sqlite` runtime artifact 和 index metadata。
- provider 应只负责 `load_batch(sample_keys) -> ExpertBatch`、sample/model 保序、
  `y_pred/y_true` 读取和 row index lineage；不应创建 run_dir、写 status/metadata/CSV、
  推导 split required keys、决定 SQLite path、管理 checkpoint/resume、执行 full-scale
  preflight 或绑定 `/data2`。
- 三种迁移方案中，A 继续保留 Visual SQLitePredictionIndex 并在 batch 后包装
  `ExpertBatch` 风险最低；B 给 provider 增加可选 prepared index / batch query 后端可作为
  Stage 1.5 候选；C 直接用 `PredictionBatchReader` 替换 Visual SQLite path 默认不建议。

验收命令结果：

- `stage1_visual_router_evaluation_adapter_bypass_smoke.py` 通过。
- `stage1_prediction_cache_expert_provider_smoke.py` 通过，输出 hard MAE `0.416048437`、
  raw soft MAE `0.410296679`。
- `stage1_evaluation_input_adapter_smoke.py` 通过，确认 adapter 不重新读取 prediction
  cache 或 oracle/TSF。
- `compileall time_router tests/smoke train_visual_router_online_streaming.py` 通过。

## 结论

P9e 证明当前 gap 不是简单 reader API 缺口，而是 full-scale runtime 与 index prepare
职责边界差异。短期不应把 `PredictionCacheExpertProvider` 直接接入 Visual Router 正式入口；
应继续保留现有 SQLitePredictionIndex，并优先在 batch 后构造 `ExpertBatch` 做 training/evaluation
旁路校验。

本步骤未修改 `train_visual_router_online_streaming.py` 行为，未替换 SQLite index，未接
`PredictionCacheExpertProvider` 到正式入口，未改 `PredictionBatchReader`、
`PredictionCacheExpertProvider`、`EvaluationInputAdapter`、VisualFeatureProvider、ViT provider、
router head、training loop、`fusion_huber_kl` loss、checkpoint/status/metadata/CSV schema，
未新增 Bash/scripts，未访问 `/data2`，未启动 pressure/full-scale，未改 TimeFuse 正式入口。

## 下一步方案

1. P9f 可先做 training loss `ExpertBatch` bypass check：继续保留 SQLite arrays 作为来源，
   默认关闭校验，从 `ExpertBatch.y_pred/y_true` 复算 `expert_errors` 并与 legacy
   `expert_errors` 比较，不改变训练 loss。
2. P10 可先抽 shared prediction index prepare helper，明确它属于 runtime/index prepare，
   不属于 provider。
3. 真正让 provider 消费 prepared index / batch query backend 应推迟到 Stage 1.5 或 Stage 2，
   并先完成 smoke、small pressure 和 full-scale preflight。
