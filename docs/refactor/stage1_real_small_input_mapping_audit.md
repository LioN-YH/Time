# Stage 1 P13a Real Small Input Mapping Audit

创建日期：2026-06-20

## 1. 目标

本文审计真实 Visual Router 与 TimeFuse-style 小规模输入如何映射到 P12b
`tests/fixtures/stage1_canonical_small/` 的 fixture contract。P13a 只冻结 mapping
边界，不迁移正式入口，不新增真实数据脚本，不访问 `/data2`，不启动训练、pressure 或
full-scale。

P13a 的输入证据来自当前仓库内已完成的小步：

- P10f `time_router/data/visual_labels_adapter.py`：Visual labels 可拆解为
  `SampleManifest` 与 `SupervisionBatch`。
- P10g `time_router/data/timefuse_supervision_adapter.py`：TimeFuse feature/oracle source
  可拆解为 `SampleManifest` 与 `SupervisionBatch`。
- P7a `time_router/features/timefuse_cache.py`：TimeFuse feature CSV 可作为
  `FeatureProvider` 输出 `FeatureBatch`。
- P10b/P10c `time_router/io/prediction_sqlite_backend.py` 与 prediction array IO：
  prediction manifest 可映射为 SQLite backend records，再读取 packed/per-sample arrays。
- P6a `time_router/experts/prediction_cache.py`：prediction cache reader 可包装为
  `ExpertBatch`，但仍未接正式入口。
- P12b `scripts/run_stage1_canonical_small.py` 与 fixture contract：tiny
  `sample_manifest.csv`、`features.csv`、`expert_predictions.json` 已能驱动 canonical
  small run。

## 2. 结论摘要

1. Visual labels 和 TimeFuse feature/oracle source 都能提供
   `stage1_sample_manifest_v1` 的样本身份字段，但真实 full-scale schema 在正式迁移前仍需
   逐字段校验，不能由 P13a 文档替代数据扫描。
2. `SampleManifest` 只承载样本身份、split、顺序和轻量 lineage；oracle label、oracle
   value、per-model error、feature values、prediction cache path、SQLite index path 均不进入
   `SampleManifest`。
3. P12b `features.csv` 是 tiny feature fixture 的表达方式。真实 TimeFuse 17 维 feature
   属于 `FeatureProvider`；真实 Visual Quito history window、pseudo image 和 ViT embedding
   也属于 Visual `FeatureProvider`，P13a 不抽 Visual online ViT provider。
4. P12b `expert_predictions.json` 只是 tiny fixture 格式。正式路径仍应走 prediction
   backend / `ExpertProvider` / `ExpertBatch`，小规模 real fixture 可以临时派生 JSON，但不得把
   JSON 当作正式 prediction cache schema。
5. P13b 已从 P10f/P10g smoke fixture 派生
   `tests/fixtures/stage1_real_derived_small/`，验证真实字段风格小样例的字段映射与保序 join；
   该 fixture 仍不创建新的 full-scale 数据链路。

## 3. SampleManifest 映射表

`stage1_sample_manifest_v1` 的 canonical 字段如下表。P13a 写的是当前可审计 mapping，不表示
真实 full-scale schema 已经迁移。

| canonical 字段 | Visual labels / legacy metadata 来源 | TimeFuse feature/oracle 来源 | 映射边界 |
| --- | --- | --- | --- |
| `sample_key` | labels 表中的 `sample_key`；P10f 要求唯一并保序 | feature source 中的 `sample_key`；oracle source 只用于 supervision join | canonical join key；prediction / supervision / feature / evaluation 都按它 join |
| `split` | labels 表中的 `split`，允许 `train/vali/test/heldout` | feature source 中的 `split`，允许 `train/vali/test/heldout` | 后续应由 `SplitStrategy` 生成或校验；P13a 不重新切 split |
| `config_name` | labels 表中的 `config_name`，例如 `96_48_S` | feature source 中的 `config_name` | 标识 seq/pred 配置；不能只从 CLI 隐式反推 |
| `dataset_name` | labels 表中的 `dataset_name` | feature source 中的 `dataset_name` | Quito dataset 或等价数据集名称 |
| `item_id` | labels 表中的 `item_id` | feature source 中的 `item_id` | 样本身份字段；adapter 当前转为整数，物理 schema 可保存为 string |
| `channel_id` | labels 表中的 `channel_id` | feature source 中的 `channel_id` | 单变量或多变量 channel 标识；adapter 当前转为整数，物理 schema 可保存为 string |
| `window_index` | labels 表中的 `window_index` | feature source 中的 `window_index` | 同一 `item_id + channel_id` 下窗口顺序，不是全局行号 |
| `seq_len` | labels 表可选 `seq_len`；缺失时 P10f adapter 为 `None` | feature source 可选 `seq_len`；缺失时 P10g adapter 为 `None` | P11b 物理 schema 要求最终 manifest 写入；P13b 真实 fixture 应补齐 |
| `pred_len` | labels 表可选 `pred_len`；缺失时 P10f adapter 为 `None` | feature source 可选 `pred_len`；缺失时 P10g adapter 为 `None` | P11b 物理 schema 要求最终 manifest 写入；P13b 真实 fixture 应补齐 |
| `lineage` | 只允许轻量列，如 `source_label_path`、`label_schema_version`、`manifest_shard`、source row id | 只允许轻量列，如 `feature_shard`、`feature_schema_version`、source row id | 不保存 feature 值、oracle/error、prediction path、SQLite path、checkpoint 或 run_dir |

### 3.1 Visual labels 映射

Visual labels 侧的真实小规模 manifest 应从 labels / legacy sample metadata 中抽取
`sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、`channel_id`、
`window_index`、`seq_len`、`pred_len` 和轻量 lineage。P10f adapter 已证明小型
DataFrame/CSV 可以构造 canonical `SampleManifest`，并保持 labels 原始行顺序。

Visual labels 中用于 oracle 选择或 loss target 的字段不进入 manifest，包括：

- oracle top-1 model；
- oracle metric value；
- 每个专家的 MAE/MSE/error；
- loss target、diagnostic metric、regret 或 upper-bound 相关字段。

这些字段只能进入 `SupervisionProvider`、`SupervisionBatch`、evaluation diagnostics 或 legacy
reference artifact。

### 3.2 TimeFuse feature/oracle 映射

TimeFuse-style 侧的 manifest 应优先从 feature source 抽取样本身份字段，因为 feature source 是
可部署 fusor/router 的 test-time 输入索引。oracle source 只用于校验 sample 集合和构造
supervision，不应成为 deployable feature 的隐式来源。

P10g adapter 已证明小型 TimeFuse feature DataFrame/CSV 可构造 `SampleManifest`，并保持 feature
source 原始行顺序；对应 oracle DataFrame/CSV 可按显式 `sample_keys + model_columns + metric`
构造 `SupervisionBatch`。

TimeFuse 17 维 feature 值不进入 `SampleManifest.lineage` 或 manifest 额外字段；oracle
label/value/per-model error 也不进入 manifest。

## 4. Supervision 映射边界

`SupervisionProvider` / `SupervisionBatch` 承载训练监督、diagnostics、baseline 或 upper-bound
所需的 oracle/error 信息。当前 smoke adapter 的共同口径是：

- 输入显式 `sample_keys`，输出必须保持该顺序；
- 输入显式 `model_columns`，per-model error 矩阵第二维必须与之对齐；
- 输入显式 `metric`，例如 `mae` 或 `mse`；
- 从 `{model_name}_{metric}_error` 列构造 `[sample, expert]` `per_model_errors`；
- `oracle_model` 和 `oracle_value` 由 per-model errors 的最小值推导。

必须保持的边界：

- oracle label / oracle value / per-model error 不进入 `SampleManifest`。
- oracle/error 不进入 deployable test-time `FeatureProvider`。
- oracle/error 可以用于训练监督、evaluation diagnostics、baseline、upper-bound 或迁移一致性检查。
- 正式 `SupervisionProvider` 仍未实现；P13a 不把 P10f/P10g smoke adapter 提升为正式 provider。

## 5. Feature 映射边界

P12b `features.csv` 的最小 tiny contract 是 `sample_key + feature columns`，并要求 provider 按
manifest ordered sample_keys 返回 `FeatureBatch`。真实 feature source 与 P12b 的关系如下：

| 来源 | P12b fixture 口径 | 真实 provider 边界 |
| --- | --- | --- |
| TimeFuse 17 维 feature cache | 可派生为 `features.csv`，列为 `sample_key + 17` 个 feature columns；行顺序可以不同于 manifest | 属于 TimeFuse `FeatureProvider`；scaler fit 是 training/runtime 行为，不属于 manifest 或 expert backend |
| Visual Quito history window | 不适合直接落成 P12b 通用 `features.csv`，除非 P13b 做 branch-specific 小型派生 | 属于 Visual `FeatureProvider` 的输入；从 Quito window 生成 pseudo image 与 ViT embedding |
| Visual pseudo image / ViT feature | P13a 不抽取，不落盘为长期 cache；P13b 若需要只能做小规模 branch-specific fixture 或 mock feature | 属于 Visual online feature path；full-scale 主线仍是 batch runtime 内生成，不保存 ViT embedding `.npy` 或伪图像 tensor |

因此，P13b 若使用 P12b small entrypoint 做真实字段映射验证，可以有两种低风险方式：

1. TimeFuse branch：从已有小规模 TimeFuse feature cache 派生 `features.csv`，保留真实 17 维列。
2. Visual branch：先使用 branch-specific feature fixture 或保守 mock 数值验证 sample join，不在 P13b
   抽 Visual online ViT provider。

## 6. Expert Prediction 映射边界

P12b `expert_predictions.json` 是 tiny fixture 格式：

```text
{
  "model_columns": [...],
  "samples": [
    {"sample_key": "...", "y_true": [[...]], "y_pred": [[[...]], ...]}
  ]
}
```

它只适合 small smoke，约束是按 manifest ordered sample_keys 组装 `ExpertBatch`。真实 Stage 1
prediction cache / SQLite backend 的 canonical 映射应保持：

- prediction manifest / SQLite backend 使用 `(sample_key, model_name)` 定位专家记录；
- `model_columns` 由调用方显式传入并固定当前 Stage 1 五专家顺序；
- record 中的 `y_true_path`、`y_pred_path`、`array_storage`、`y_true_row_index`、
  `y_pred_row_index` 属于 prediction backend implementation；
- array 读取由 prediction array IO / `PredictionBatchReader` / `ExpertProvider` 完成；
- `ExpertBatch` 输出 `sample_keys`、`model_columns`、`y_pred`、共享 `y_true` 和 row index
  lineage；
- prediction cache path、SQLite index path、packed npy path 不进入 `SampleManifest`。

P13a 不替换 Visual `SQLitePredictionIndex`，不把 `PredictionCacheExpertProvider` 接到正式
Visual Router 或 TimeFuse-style fusor 入口。后续正式迁移前，仍需证明 prepared backend 在真实
small batch、pressure 和 full-scale 行为下与 legacy path 完全一致。

## 7. 责任分层

| 信息类型 | 所属层 | 不属于 |
| --- | --- | --- |
| `sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、`channel_id`、`window_index`、`seq_len`、`pred_len`、轻量 lineage | `SampleManifest` | `FeatureProvider`、prediction backend、supervision payload |
| split 生成或校验、split overlap、ordered sample_keys policy | `SplitStrategy` / Runtime input artifact | feature CSV、oracle reader、prediction reader 私自推导 |
| oracle model、oracle value、per-model error、metric | `SupervisionProvider` / `SupervisionBatch` | `SampleManifest`、deployable `FeatureProvider` |
| TimeFuse 17 维 feature、Visual history window、pseudo image、ViT embedding | `FeatureProvider` / `FeatureBatch` | `SampleManifest`、prediction backend |
| `(sample_key, model_name)` records、array paths、packed row indices、shared y_true 校验 | prediction backend / `ExpertProvider` / `ExpertBatch` | `SampleManifest`、FeatureProvider |
| CSV/JSON artifact 写出、run metadata/status、manifest snapshot/reference | Runtime artifact writer | Provider / Head / Evaluator |

## 8. P13b 真实小规模 fixture 建议

P13b 若做真实小规模 fixture，应控制范围在“派生并验证字段映射”，不启动正式训练，不访问
`/data2`，不改正式入口。建议顺序如下：

1. **选择来源**：优先使用仓库内已有 golden fixture、small dry-run fixture、P10f/P10g smoke
   fixture 或已提交的小样本输出；若来源需要外部数据，应另起目标并显式说明。
2. **派生 `sample_manifest.csv`**：从 Visual labels 或 TimeFuse feature source 抽取
   `stage1_sample_manifest_v1` 字段；补齐 `seq_len/pred_len`；写入轻量 lineage；校验
   sample_key 唯一、split 取值、manifest row order。
3. **派生 feature fixture**：TimeFuse branch 可写 `sample_key + 17` 维真实 feature；
   Visual branch 可先写 branch-specific 小型 feature fixture 或 mock feature，避免在 P13b 抽
   online ViT provider。
4. **派生 expert fixture 或复用 backend helper**：小型场景可以从 prediction cache 读出
   `ExpertBatch` 后写成 `expert_predictions.json`；更接近正式路径的 smoke 可直接用 shared
   prediction backend helper 构造 `ExpertBatch`，但不要改 P12b small entrypoint 的 tiny fixture
   contract。
5. **用 P12b entrypoint 验证映射**：只验证 manifest 保序、feature/expert join、canonical
   run_dir artifact 可读和 metadata inputs 摘要；不声称正式入口已迁移。

P13b 已按上述低风险路径新增
`docs/refactor/stage1_real_derived_small_fixture.md`、
`tests/fixtures/stage1_real_derived_small/` 和
`tests/smoke/stage1_real_derived_small_fixture_smoke.py`。本次 fixture 从 P10f/P10g smoke 的
ETTh1 / ETTm2 / weather 小样本身份派生 manifest；feature 仍是 P12b small entrypoint 支持的
三列 schema-style fixture，不是 TimeFuse 17 维 full-scale feature cache；expert JSON 仍只是
small fixture，不是正式 prediction backend。

P13c 已新增 `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`，进一步冻结真实
small batch 后续连接方案：`expert_predictions.json` 由 prediction backend / `ExpertProvider` /
`ExpertBatch` 替换；三列 `features.csv` 由 TimeFuse 17 维 feature provider 或 Visual history
window / pseudo image / ViT feature provider 替换；generic small entrypoint 继续保持 thin CLI，
branch-specific 输入 shape 或 provider/head 验证另起 smoke 或 small entrypoint。

## 9. P13a 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增真实数据派生脚本、Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 10. 验收

P13a 是纯文档审计，代码行为应保持不变。验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```
