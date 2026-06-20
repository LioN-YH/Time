# Stage 1 P10g TimeFuse feature/oracle 到 SampleManifest / SupervisionBatch smoke adapter

日志日期：2026-06-20 10:52:20 CST

## 目的

新增一个最小 TimeFuse feature/oracle 到 canonical `SampleManifest` 与 `SupervisionBatch`
的 smoke adapter，验证 TimeFuse-style fusor 历史 feature source 与 oracle/supervision source
可以拆解为统一的 sample/split/supervision 协议对象。

## 背景

P10d 已冻结 canonical `SampleManifest` / `SplitStrategy` / `SupervisionProvider` 边界。
P10e 已新增最小 `SampleManifestRow`、`SampleManifest` 和 `SupervisionBatch` 协议骨架与 smoke。
P10f 已完成 Visual labels 到 `SampleManifest` / `SupervisionBatch` 的 smoke adapter。

P10g 需要补齐 TimeFuse 路线的对应 smoke adapter，但不得修改正式 TimeFuse 训练入口、
launcher、feature provider、prediction reader、evaluation adapter、loss、optimizer、scaler
或正式 artifact schema。

## 操作

1. 读取目标说明、当前 git 状态、P10e/P10f 协议与 Visual labels adapter 实现。
2. 新增 `time_router/data/timefuse_supervision_adapter.py`：
   - `timefuse_features_to_sample_manifest(...)` 从小型 feature DataFrame/CSV 构造
     `SampleManifest`；
   - `timefuse_oracle_to_supervision_batch(...)` 从小型 oracle DataFrame/CSV 构造
     `SupervisionBatch`；
   - manifest extra 只保留 `feature_shard`、`feature_schema_version` 等轻量 lineage；
   - 明确不把 17 维 TimeFuse feature 值放入 `SampleManifestRow.extra`。
3. 更新 `time_router/data/__init__.py`，导出 P10g adapter 函数。
4. 新增 `tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py`：
   - 构造 4 行 vali/test feature fixture，包含 17 维 feature 列；
   - 构造对应 4 行 oracle fixture，包含五专家 `mae` error 列；
   - 校验 sample_key 唯一、split 保序、`split_counts()`、oracle top-1、oracle value、
     `[sample, expert]` shape；
   - 覆盖 CSV/DataFrame 双入口、缺失 oracle 专家列、feature duplicate sample_key、
     oracle 缺失 sample_key 和未知 split 报错。
5. 新增 `docs/refactor/timefuse_sample_supervision_adapter.md`。
6. 更新 `docs/refactor/stage1_canonical_sample_supervision_boundary.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

已执行新 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
```

结果通过，输出确认：

- `SampleManifest` 唯一性、split 保序、`split_counts()` 和 feature lineage extra 通过；
- vali/test `SupervisionBatch` 的 oracle、oracle_value 和 per-model error shape 通过；
- feature/oracle CSV 与 DataFrame 入口一致；
- 缺失专家列、feature 重复 sample_key、oracle 缺失 sample_key 和未知 split 均给出清晰报错。

完整验收命令均已执行通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```

补充验证结果：

- P10f Visual labels adapter smoke 通过，确认 P10g 未破坏既有 Visual sample/supervision adapter。
- P10e SampleManifest / SupervisionBatch protocol smoke 通过，确认协议骨架仍满足 shape 与保序约束。
- P7c TimeFuse protocol chain smoke 通过，输出保持 `hard_mae=1.093573928`、
  `raw_soft_mae=0.556751269`，head/evaluator 阶段仍不回读 cache、不写产物。
- `compileall` 通过，新增 adapter、smoke 和既有 Stage 1 代码语法检查无错误。

## 结论

P10g 已补齐 TimeFuse feature/oracle 到 canonical sample/supervision 协议的最小 adapter
与 smoke。至此，Visual labels 和 TimeFuse feature/oracle 两条历史输入路径都已有
canonical `SampleManifest` / `SupervisionBatch` adapter smoke。

本步未修改正式 Visual Router / TimeFuse-style fusor 入口，未访问 `/data2`，未启动
pressure/full-scale，未改变正式 CSV / summary / metadata / status / checkpoint schema。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续正式入口接入前，应先审计真实 full-scale feature CSV 与 oracle SQLite/parquet schema，
   再冻结 `SampleManifest` 物理存储、`SupervisionProvider` 缺失策略和字段映射层。
