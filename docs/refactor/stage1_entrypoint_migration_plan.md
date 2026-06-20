# Stage 1 Canonical Entrypoint Migration Plan

创建日期：2026-06-19
更新日期：2026-06-20

## 1. 目标

本文记录 Stage 1 entrypoint 后续迁移路线。P10h 起，迁移计划不再以
“Visual Router 和 TimeFuse-style fusor 分别拆解入口职责”为主视角，而是以统一
canonical dataflow 为主视角，再在每个层次上声明两条路线的 branch-specific 实现。

本阶段只做文档对齐，不修改正式入口，不新增 provider/head/runtime 代码，不新增 Bash
或 scripts，不访问 `/data2`，不启动 pressure/full-scale。

当前仍保留的正式入口：

- Visual Router：`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
- TimeFuse-style fusor：`visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- TimeFuse full-scale 后台编排入口：
  `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`

## 2. Canonical Dataflow 总览

Stage 1 长期数据流应收束为：

```text
SampleManifest + SplitStrategy
  -> ordered sample_keys
  -> ExpertProvider / prediction backend
  -> SupervisionProvider
  -> FeatureProvider
  -> RouterHead
  -> EvaluationInputAdapter / Evaluator
  -> Runtime / artifact writer
```

这个 dataflow 统一的是样本主索引、专家预测、监督、评估和运行产物契约；它不要求
Visual Router 与 TimeFuse-style fusor 共用同一个 feature extractor、head、loss 或训练目标。

### 2.1 SampleManifest / SplitStrategy

`SampleManifest` 是样本身份、split 和顺序的 canonical source。它至少承载
`sample_key`、`split`、`config_name`、`dataset_name`、`item_id`、`channel_id`、
`window_index`，以及可选 `seq_len`、`pred_len` 和轻量 lineage。

`SplitStrategy` 负责生成或校验 split，并按 split 输出 ordered `sample_keys`。split
不应继续由 Visual labels CSV、TimeFuse feature CSV、oracle reader 或 prediction reader
各自推导。

两条路线应共用：

- `SampleManifest` 语义和物理存储版本方向；
- `SplitStrategy` 语义、split 互斥性和 ordered sample_keys；
- 同一批 `sample_key` 在 prediction、supervision、feature 和 evaluation 之间的 join 口径。

当前状态：

- P10e 已新增 smoke-only `SampleManifestRow` / `SampleManifest` protocol skeleton。
- P10f 已验证 Visual labels 可拆解为 `SampleManifest` 与 `SupervisionBatch`。
- P10g 已验证 TimeFuse feature/oracle source 可拆解为 `SampleManifest` 与
  `SupervisionBatch`。
- P11b 已冻结 `stage1_sample_manifest_v1` 物理 schema、`stage1_split_summary_v1` split
  summary schema，以及 `run_dir/inputs/` 中 snapshot/reference 两种 manifest 保存方式，见
  `docs/refactor/stage1_sample_manifest_physical_schema.md`。
- 正式入口尚未整体迁移到 canonical `SampleManifest` 驱动。

### 2.2 ExpertProvider / Prediction Backend

`ExpertProvider` 负责提供专家预测 batch：`sample_keys`、`model_columns`、`y_pred`、
共享 `y_true` 和 row index lineage。prediction cache 是当前 Stage 1 实现选项，不应上升为
全局专家系统接口。

两条路线应共用：

- prediction SQLite backend / prepared index 方向；
- `ExpertBatch` contract；
- 固定当前 Stage 1 五专家顺序的 canonical experiment 校验；
- `packed_npy_v1` row index lineage、共享 `y_true` 校验和 batch-level grouped loading；
- `(sample_key, model_name)` 唯一性与五专家完整性校验。

当前状态：

- `PredictionBatchReader` 已抽出并用于 smoke。
- `PredictionCacheExpertProvider` 已作为 smoke-only adapter 输出 `ExpertBatch`。
- shared prediction SQLite backend smoke 已完成。
- Visual evaluation/training ExpertBatch bypass 已完成，但只是旁路校验，未替换正式
  SQLite path。
- TimeFuse protocol chain smoke 已完成，但正式 streaming reader 尚未整体替换。

### 2.3 SupervisionProvider

`SupervisionProvider` 提供训练监督、诊断、baseline 或 upper-bound 所需的
`oracle_model`、`oracle_value`、`per_model_errors`、`model_columns`、`metric` 和 lineage。
它必须与 `ExpertProvider` 分层，不能被当作 prediction backend 或 deployable feature source。

两条路线应共用：

- `SupervisionBatch` / `SupervisionProvider` contract；
- `sample_keys + model_columns + metric` 保序返回语义；
- oracle/error 只用于监督、诊断、baseline 或 upper-bound 的边界；
- supervision 缺失策略、metric 维度和 per-model error 来源的后续 schema 冻结方向。

当前状态：

- P10e 已新增最小 `SupervisionBatch` protocol skeleton 与 shape smoke。
- Visual labels adapter smoke 已完成。
- TimeFuse sample/supervision adapter smoke 已完成。
- 正式 `SupervisionProvider` 读取逻辑尚未实现，正式入口尚未迁移。

### 2.4 FeatureProvider

`FeatureProvider` 只提供可部署 router/fusor 特征，不读取 oracle、expert error 或未来
`y`。这里是两条路线最主要的 branch-specific 分叉点。

保留 branch-specific 实现：

- Visual Router：Quito history window、`x -> pseudo image -> frozen ViT -> embedding`
  online feature provider；伪图像 tensor 和 ViT embedding 只在 batch runtime 存在，不作为
  长期 cache 落盘。
- TimeFuse-style fusor：17 维 TimeFuse-derived feature cache 或后续 online feature
  computation；feature-only scaler 属于 training/runtime 行为，不属于 `ExpertProvider`。

当前状态：

- TimeFuse 17 维 feature cache已有 smoke-only `TimeFuseFeatureCacheProvider`。
- Visual online ViT provider 尚未抽取；该路径绑定 Quito 数据读取、pseudo image、ViT /
  Hugging Face cache、GPU dtype、DataParallel、latency 和 future finetune/joint training，
  不作为第一批最小迁移目标。

### 2.5 RouterHead

`RouterHead` 只消费 `FeatureProvider` 输出，并输出与 `model_columns` 对齐的 logits/weights。
loss、optimizer、scheduler、checkpoint/resume 和 epoch loop 仍由 branch-specific training/runtime
管理。

保留 branch-specific 实现：

- Visual Router：`VisualMLPRouterHead`，兼容当前 MLP router；
- TimeFuse-style fusor：`TimeFuseLinearSoftmaxHead`，对应 `nn.Linear -> softmax` 的
  sample-level adaptive expert fusion；
- Visual 目标：`fusion_huber_kl` / classification objective；
- TimeFuse 目标：SmoothL1 weighted fusion objective。

当前状态：

- TimeFuse smoke-only `TimeFuseLinearSoftmaxHead` 已完成。
- Visual MLP head 尚未抽为 canonical head adapter。
- 正式入口的 loss、optimizer、scaler、checkpoint/resume 均未改变。

### 2.6 EvaluationInputAdapter / Evaluator

`Evaluator` 负责从同一批 `sample_keys`、`model_columns`、`y_pred`、`y_true` 和
weights/logits 复算 hard top-1、raw soft fusion、diagnostics、summary 和 per-sample rows。
文件写出仍属于 runtime / artifact writer，不属于 evaluator 本体。

两条路线应共用：

- `EvaluationInputAdapter` / `Evaluator` metrics；
- hard top-1、raw soft fusion、MAE/MSE、selected counts、entropy、max weight 的复算口径；
- 未来 comparison、calibration-ready object 和 run artifact contract 方向；
- 不从 legacy CSV 反推专家顺序的原则。

当前状态：

- `EvaluationInputAdapter` 已完成 smoke-only adapter。
- Visual evaluation adapter bypass 与 training ExpertBatch bypass 已完成。
- TimeFuse protocol chain smoke 已完成。
- 正式入口仍保留各自 CSV/summary 写出逻辑；Evaluator 尚未整体替换正式 report 层。

### 2.7 Runtime / Artifact Writer

Runtime 负责 CLI/config、run_dir、seed/device、resource policy、checkpoint/resume、
epoch/batch loop、train-only/eval-only 分支、status/metadata/checkpoint/log/evaluation/
prediction artifact 写出和 launcher 接手信息。

两条路线应共用：

- run artifact contract 方向；
- `status.json` / `metadata.json` / checkpoint index / logs / evaluation / predictions 的未来
  schema 分层；
- 显式 `run_dir` 由 runtime/launcher 传入的原则；
- provider 不创建 run_dir、不写 status/metadata、不硬编码 `/data2` 的原则。

保留 branch-specific 实现：

- Visual Router 的 online ViT resource policy、DataParallel、latency 统计和 visual-specific
  metadata；
- TimeFuse 的 feature shard discovery、feature-only scaler、shard-local oracle/prediction
  index prepare 和 launcher preflight。

当前状态：

- P5a 已定义 canonical runtime contract 方向。
- P11c 已新增 `time_router.runtime.artifact_writer` 最小 helper，可在临时 `run_dir` 写出
  `run_metadata.json`、`run_status.json`、`inputs/sample_manifest_ref.json`、
  `inputs/split_summary.json`、`evaluation/evaluation_summary.json` 和
  `predictions/prediction_rows.csv`；该 helper 尚未接入正式入口。
- P11d 已新增 tiny canonical protocol run smoke，把 `SampleManifest`、`ExpertBatch`、
  `FeatureBatch`、`RouterOutput`、`EvaluationInputAdapter` 和 P11c Runtime artifact writer
  串成 tempfile canonical `run_dir`；Provider / Head / Evaluator 仍不知道 `run_dir`，正式入口尚未迁移。
- P12 已新增 `scripts/run_stage1_canonical_small.py` small canonical Python entrypoint thin slice，
  可通过 CLI 接收 `--output-root/--run-name`，运行 tiny canonical dataflow，并只由 Runtime
  artifact writer 写出 canonical `run_dir`；见
  `docs/refactor/stage1_canonical_small_entrypoint.md`。
- P12b 已固定 small fixture input contract：entrypoint 继续默认使用内联 tiny fixture，同时
  可选读取 `--sample-manifest`、`--feature-source` 和 `--expert-fixture`；显式 fixture 保持
  manifest row order、feature provider 按 manifest 保序、expert fixture 按 manifest 组装
  `ExpertBatch`，并在 `run_metadata.json inputs` 中记录 `sample_manifest`、`feature_source`
  和 `expert_fixture` 来源摘要；见
  `docs/refactor/stage1_canonical_small_fixture_contract.md`。
- `launch_timefuse_fusor_full_scale.py` 仍是当前 TimeFuse full-scale preflight、脚本生成、
  PID/PGID、stop/resume 和接手信息层。
- 正式 CSV / summary / metadata / status / checkpoint schema 本阶段不改。

## 3. 共用层与分支层边界

应两条路线共用的层：

- `SampleManifest`；
- `SplitStrategy` 语义；
- prediction SQLite backend / `ExpertBatch`；
- `SupervisionBatch` / `SupervisionProvider` contract；
- `EvaluationInputAdapter` / Evaluator metrics；
- run artifact contract 方向。

保留 branch-specific 实现的层：

- Visual Router 的 Quito history window / pseudo image / ViT feature provider；
- TimeFuse 的 17 维 feature cache / feature-only scaler；
- `VisualMLPRouterHead`；
- `TimeFuseLinearSoftmaxHead`；
- Visual `fusion_huber_kl` / classification objective；
- TimeFuse SmoothL1 weighted fusion objective。

## 4. 当前代码状态

已完成：

- Visual evaluation/training `ExpertBatch` bypass；
- Visual labels adapter smoke；
- TimeFuse protocol chain smoke；
- TimeFuse sample/supervision adapter smoke；
- shared prediction SQLite backend smoke。

尚未完成：

- 正式入口尚未整体迁移到 canonical dataflow；
- 正式 `SupervisionProvider` 尚未实现；
- Visual online ViT `FeatureProvider` 尚未抽取；
- P11a 已冻结 future canonical run artifact schema，见
  `docs/refactor/stage1_canonical_run_artifact_schema.md`；正式 legacy output schema 尚未改动；
- P11b 已冻结 future canonical `SampleManifest` physical schema，见
  `docs/refactor/stage1_sample_manifest_physical_schema.md`；正式 legacy input/output schema 尚未改动；
- P11c 已提供最小 Runtime artifact writer/helper 和 tempfile smoke，见
  `docs/refactor/stage1_runtime_artifact_writer.md`；正式 legacy entrypoint 尚未迁移；
- P11d 已提供 tiny canonical protocol run smoke，见
  `docs/refactor/stage1_canonical_protocol_run_smoke.md`；正式 legacy entrypoint 尚未迁移；
- P12 已提供 small canonical Python entrypoint thin slice，见
  `docs/refactor/stage1_canonical_small_entrypoint.md`；正式 legacy entrypoint 尚未迁移；
- P12b 已提供 small fixture input contract hardening，见
  `docs/refactor/stage1_canonical_small_fixture_contract.md`；该 fixture 后续用于 P13 审计真实
  Visual/TimeFuse 小规模输入映射；
- P13a 已完成真实小规模输入 mapping audit，见
  `docs/refactor/stage1_real_small_input_mapping_audit.md`；明确 Visual labels、
  TimeFuse feature/oracle、真实 feature source、prediction cache / SQLite backend 到
  P12b fixture contract 的映射边界，并确认 P13a 不迁移正式入口、不创建真实 fixture；
- pressure / full-scale canonical scripts 尚未准备。

## 5. 下一阶段路线

建议顺序：

1. P13b 可从已有 golden fixture、小规模真实样本或 P10f/P10g smoke fixture 派生真实小规模
   `sample_manifest.csv`、branch-specific feature fixture 和 expert fixture/backend smoke，用
   P12b entrypoint 验证字段映射、保序 join 与 artifact 可读。
2. P13b 仍需保持 Provider / Head / Evaluator 不知道 `run_dir`，且不把 Bash 语义下沉到
   `time_router`。
3. 准备 pressure / full-scale 方案时，`scripts/` 仍只作为 thin entrypoint 或 launcher，
   不承载 provider 内部逻辑；Bash launcher 另行分层，不能混入 P12 small CLI。
4. 以 legacy `96_48_S` full-scale 结果作为 reference baseline；canonical pipeline 后续需要
   重跑，不能把旧 schema 作为新 contract 的强兼容来源。

## 6. 当前阶段明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增 provider/head/runtime 代码。
- 不改 `time_router/protocols/types.py`。
- 不改 `PredictionBatchReader` / `PredictionCacheExpertProvider` / `EvaluationInputAdapter`。
- 不新增 Bash launcher。
- 不访问 `/data2`。
- 不启动 pressure/full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。

## 7. 本阶段验收

P10h 文档对齐后的验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router
```
