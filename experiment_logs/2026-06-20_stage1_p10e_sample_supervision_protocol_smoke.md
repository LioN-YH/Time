# Stage 1 P10e canonical SampleManifest / SupervisionBatch protocol smoke

日志日期：2026-06-20 04:27:26 CST

## 目的

在 P10d canonical SampleManifest / supervision boundary 设计基础上，新增最小
`SampleManifest` / `SupervisionBatch` 协议骨架和纯内存 smoke，先锁定 sample identity、
split、ordered sample_keys 与 supervision shape 对齐的轻量 contract。

## 背景

P10d 已明确 Stage 1 后续主索引应从历史 labels CSV / feature CSV 提升为
canonical `SampleManifest`，并将 oracle / per-model error 监督信息从
`ExpertProvider` 和可部署 `FeatureProvider` 中分离出来。用户接受必要时重跑
Stage 1，因此本步优先服务新的 canonical schema，而不是继续把旧 CSV / runtime
artifact schema 作为最高兼容目标。

## 操作

1. 修改 `time_router/protocols/types.py`：
   - 新增 `SampleManifestRow`，记录 `sample_key`、`split`、`config_name`、
     `dataset_name`、`item_id`、`channel_id`、`window_index`、可选
     `seq_len/pred_len` 和 `extra`。
   - 新增 `SampleManifest`，提供 `validate_unique_sample_keys()`、
     `sample_keys(split=None)` 和 `split_counts()`。
   - 新增 `SupervisionBatch`，保存 `sample_keys`、`model_columns`、`metric`、
     `oracle_model`、`oracle_value`、`per_model_errors` 和 `extra`，并提供
     `validate_shapes()` 做最小维度对齐校验。
2. 修改 `time_router/protocols/__init__.py`，从 public API 导出
   `SampleManifestRow`、`SampleManifest` 和 `SupervisionBatch`。
3. 新增 `tests/smoke/stage1_sample_supervision_protocol_smoke.py`：
   - 构造 4 行 vali/test manifest。
   - 校验 `sample_key` 唯一、split 过滤保序和 split 统计。
   - 构造 vali/test 两个 supervision batch，使用五专家列和小型 numpy error matrix。
   - 覆盖重复 sample_key、专家维 shape mismatch 和 oracle shape mismatch 报错。
4. 更新 `docs/refactor/stage1_canonical_sample_supervision_boundary.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_target_architecture.md`、
   `docs/refactor/protocol_types.md` 和 `WORKSPACE_STRUCTURE.md`，记录 P10e
   已完成的最小协议骨架、边界和验收口径。

## 结果

以下命令均已在 conda `quito` 环境下通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

新增 smoke 输出确认：

- `SampleManifest` 唯一性、split 过滤、`split_counts()` 和 ordered sample_keys 通过。
- `SupervisionBatch` 保持 sample/model 顺序、metric 和监督矩阵 shape 通过。
- 重复 sample_key 和 supervision shape mismatch 均能给出清晰报错。

## 结论

P10e 已完成最小 canonical `SampleManifest` / `SupervisionBatch` 协议骨架与 smoke。
本步没有修改 `train_visual_router_online_streaming.py`、
`train_timefuse_fusor_streaming.py`、`launch_timefuse_fusor_full_scale.py`，没有接
Visual Router / TimeFuse 正式训练，没有新增 provider 实现，没有访问 `/data2`，
也没有改变正式 CSV、summary、metadata、status、checkpoint、loss、optimizer、
scaler 或 resume 口径。

## 下一步方案

后续可继续进入 P11/P12 schema 冻结设计，明确 `SampleManifest` 的物理存储格式、
版本号、`SplitStrategy` materialize/validate 方式、`SupervisionProvider` 的 metric
维度和缺失策略；正式入口接入仍应另起小步，并继续以 smoke 和中文实验日志作为门禁。
