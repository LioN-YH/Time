# Stage 1 P14a Visual FeatureProvider 插入点审计

日志日期：2026-06-20 16:02:53 CST

## 目的

审计 Visual Router 正式入口中可迁移为 Visual `FeatureProvider` 的最小边界，明确未来
`FeatureBatch` 输出、不做范围、device/dtype/runtime 分层，以及与 `ExpertBatch` 的对齐关系。
本步骤只做文档审计，不抽 provider，不改正式入口，不访问 `/data2`，不启动训练或 full-scale。

## 背景

P13d 已完成 prediction backend -> `ExpertBatch` small smoke。P13e 已完成 TimeFuse 17 维
`FeatureProvider` -> `FeatureBatch` small smoke。当前 TimeFuse-style 支线已经有
`ExpertBatch`、17 维 `FeatureBatch`、`TimeFuseLinearSoftmaxHead` 和 protocol chain smoke；
Visual 主线仍把 history window、pseudo image、ViT forward、router feature、device/dtype、
latency、checkpoint/resume 等逻辑混在 `train_visual_router_online_streaming.py` 中。

因此 P14a 先做插入边界审计，避免直接抽 provider 时把 Runtime、ExpertProvider、RouterHead
或正式输出 schema 混入 pure Visual `FeatureProvider`。

## 操作

1. 读取目标文件 `/home/shiyuhong/.codex-tianyu/attachments/cbb5eb4c-7ae9-40db-9f3c-3d3f342e9011/pasted-text-1.txt`，确认 P14a 范围、边界和验收命令。
2. 检查当前分支为 `refactor/stage1-route-audit`，并确认工作区已有 P13d/P13e 文档、smoke 和结构索引。
3. 阅读 `train_visual_router_online_streaming.py` 中以下路径：
   - labels/sample metadata：`load_labels`、`filter_stream_shard`、`limit_samples_per_split`、`windows_from_labels`；
   - Visual feature path：`iter_online_embedding_batches`、Quito `load_datasets`、history window `x` 读取、`make_pseudo_images`、ViT forward、`pool_vit_outputs`；
   - Runtime path：`resolve_device`、`resolve_dtype`、`build_vit_model`、`load_vit_model_with_retry`、DataParallel、latency rows；
   - downstream path：`StandardScaler`、`VisualMLPRouter`、training/eval batch、SQLite prediction index、ExpertBatch bypass、checkpoint/status/metadata 写出。
4. 新增 `docs/refactor/stage1_visual_feature_provider_insertion_audit.md`。
5. 同步更新：
   - `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`
   - `docs/refactor/stage1_target_architecture.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `WORKSPACE_STRUCTURE.md`
6. 新增本中文实验日志，并更新 `experiment_logs/README.md` 总览追踪表。

## 结果

已完成 P14a 文档审计，核心结论如下：

- 未来 Visual `FeatureProvider` 最小输出应是 `FeatureBatch(sample_keys, features, feature_schema, extra)`。
- `sample_keys` 必须保持 manifest ordered sample_keys；`features` 当前最小口径是 router/head 消费的视觉特征或轻量表示，例如 ViT pooled embedding。
- `feature_schema` 应记录 visual schema name、feature dim、history window 来源、pseudo-image 参数、encoder name、pooling、normalization、image size 和 period selection。
- `extra` 只放轻量 lineage，不放 `run_dir`、checkpoint、status、正式输出路径或大型数组路径。
- 属于 Visual provider 的候选逻辑包括按 sample metadata 定位 history window、只读取历史 `x`、构造 pseudo image、可选执行 frozen ViT feature extraction、输出 batch-level visual features。
- 不属于 Visual provider 的逻辑包括 oracle/error、prediction cache、SQLite backend、`ExpertBatch`、loss、optimizer、backprop、checkpoint/resume、run_dir/status/metadata/logs、full evaluation rows/summary、Bash/`exp_scripts`/`/data2` 路径策略和 Visual RouterHead 权重计算。
- device、dtype、DataParallel、Hugging Face cache、retry、latency、checkpoint signature 由 Runtime 或 encoder factory 显式管理；pure provider 不私自决定全局 device、checkpoint 或 run_dir。
- `FeatureBatch` 与 `ExpertBatch` 只通过 ordered `sample_keys` 对齐；Visual provider 不读取 prediction cache。

本轮未修改以下正式代码：

- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`

P14a 指定验收命令均已通过：

```text
通过：/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
通过：/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
通过：/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
通过：/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
通过：/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
通过：/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 结论

P14a 已把 Visual FeatureProvider 的最小插入边界从正式 runtime 中审计出来。后续不应直接迁移
正式入口，而应先做 small smoke-only：

1. P14b：Visual FeatureProvider minimal mock/fixture smoke，先用 fake history window reader
   或 deterministic encoder stub 输出 `FeatureBatch`。
2. P14c：Visual eval-only canonical bypass plan，规划 legacy SQLite batch arrays 如何与 Visual
   `FeatureBatch` / head / evaluator 对齐，仍不替换正式入口、不改正式输出 schema。

## 下一步方案

检查 diff，提交并推送到远程 `refactor/stage1-route-audit` 分支，建议提交信息：

```text
docs: audit stage1 visual feature provider insertion
```
