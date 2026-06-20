# Stage 1 P10f Visual labels sample/supervision adapter

日志日期：2026-06-20 10:39:29 CST

## 目的

新增一个最小 Visual labels CSV / DataFrame 到 canonical `SampleManifest` 与
`SupervisionBatch` 的 smoke adapter，验证历史 Visual Router labels 表可以拆解为
sample manifest、split 与 supervision 协议对象。

## 背景

P10d 已冻结 canonical `SampleManifest` / `SplitStrategy` / `SupervisionProvider` 边界。
P10e 已新增 `SampleManifestRow`、`SampleManifest` 和 `SupervisionBatch` 最小协议骨架与
纯内存 smoke。P10f 进一步验证 Visual labels 历史 schema 的职责拆分，但不接正式
Visual Router 入口，也不改变正式输出 schema。

历史 Visual labels CSV 同时承担 sample manifest、split、oracle supervision 和 metadata。
新架构要求 manifest 只保存样本身份、split、顺序与轻量 lineage，oracle/error 只进入
supervision，不进入 deployable `FeatureProvider`。

## 操作

1. 新增 `time_router/data/visual_labels_adapter.py`：
   - `visual_labels_to_sample_manifest(...)` 支持小型 `pd.DataFrame` 或 CSV 路径输入；
   - 构造 `SampleManifest`，字段覆盖 `sample_key`、`split`、`config_name`、
     `dataset_name`、`item_id`、`channel_id`、`window_index` 和可选 `seq_len/pred_len`；
   - `SampleManifestRow.extra` 只保存白名单 lineage，例如 `manifest_shard`；
   - `visual_labels_to_supervision_batch(...)` 按显式 `sample_keys + model_columns + metric`
     保序输出 `SupervisionBatch`；
   - 使用 `{model_name}_{metric}_error` 作为 P10f smoke fixture 的 per-model error 列约定。
2. 更新 `time_router/data/__init__.py`，导出 P10f adapter public API。
3. 新增 `tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py`：
   - 构造 4 行 vali/test labels fixture；
   - 包含五专家 `mae` error 列；
   - 校验 manifest sample_key 唯一、split 保序、`split_counts()` 和 lineage extra；
   - 分别对 vali/test 构造 `SupervisionBatch`；
   - 校验 `oracle_model`、`oracle_value` 和 `[sample, expert]` shape；
   - 覆盖 CSV 入口、缺失专家列、重复 `sample_key` 和未知 split 报错。
4. 新增 `docs/refactor/visual_labels_sample_supervision_adapter.md`，说明 P10f 范围、fixture 字段、
   不接正式入口边界和后续 TimeFuse adapter 方向。
5. 更新 `docs/refactor/stage1_canonical_sample_supervision_boundary.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

已执行并通过完整验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

新增 smoke 输出确认：

- `SampleManifest` 保持 labels 原始 sample_key 顺序；
- vali/test split 过滤保序，`split_counts()` 为 `{"vali": 2, "test": 2}`；
- `SampleManifestRow.extra` 只保存 `manifest_shard` lineage，不保存专家 error；
- vali/test `SupervisionBatch` 的 oracle 专家和 oracle error 均来自每行最小专家误差；
- 缺失专家列、重复 `sample_key` 和未知 split 均触发清晰 `ValueError`。

## 结论

P10f 已完成 Visual labels 到 canonical sample/supervision 协议对象的最小 adapter smoke。
本步未修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py`、
`launch_timefuse_fusor_full_scale.py`，未接正式入口，未访问 `/data2`，未启动 pressure/full-scale，
未改变正式 CSV / summary / metadata / status / checkpoint schema，也未修改 loss、optimizer、scaler
或 checkpoint/resume。

## 下一步方案

1. 提交并推送到远程 `refactor/stage1-route-audit` 分支。
2. P10f 之后建议做 TimeFuse feature/oracle 到 `SampleManifest` / `SupervisionBatch` 的 smoke adapter；
   正式入口接入前，需要另行对齐真实 Visual labels CSV 字段映射。
