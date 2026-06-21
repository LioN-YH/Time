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
- P14a 已完成 Visual FeatureProvider 插入审计，见
  `docs/refactor/stage1_visual_feature_provider_insertion_audit.md`；未来 Visual provider
  最小输出应是 `FeatureBatch(sample_keys, features, feature_schema, extra)`，history window /
  pseudo image / 可选 frozen ViT forward 是候选 provider 边界，device/dtype/DataParallel/
  Hugging Face cache/latency/checkpoint 仍由 Runtime 或 encoder factory 管理。
- P16b 已完成 real Visual feature provider 边界审计，见
  `docs/refactor/stage1_real_visual_feature_provider_audit.md`；进一步明确 head-ready
  `FeatureBatch` 不应由 RouterHead adapter 隐式准备，真实 Visual feature chain 应拆成
  `HistoryWindowProvider` / `VisualRawInputProvider`、`PseudoImageTransformer`、
  `VisualEncoderProvider` / `FrozenViTFeatureProvider`、可选 `FeatureScaler` /
  `FeatureNormalizer` 和组合型 `VisualFeatureProvider`；scaler fit/state loading、ViT loading、
  device/dtype/batch size/DataParallel、feature cache 和 `run_dir` 均不应成为 provider 的隐式
  interface。

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
- P13b 已完成 real-derived / schema-style small fixture smoke，见
  `docs/refactor/stage1_real_derived_small_fixture.md`；新增
  `tests/fixtures/stage1_real_derived_small/` 和
  `tests/smoke/stage1_real_derived_small_fixture_smoke.py`，从 P10f/P10g smoke 的 ETTh1 /
  ETTm2 / weather 小样本身份派生 manifest，用 P12b entrypoint 验证 manifest 保序、
  feature/expert join、canonical `run_dir` 写出、metadata inputs 来源摘要和 evaluation
  sample_count；P13b 不迁移正式入口，也不把 expert JSON 升级为正式 prediction backend；
- P13c 已完成 real small backend / provider connection audit，见
  `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`；明确 P13b
  `expert_predictions.json` 后续由 prediction backend / `ExpertProvider` / `ExpertBatch`
  替换，shared prediction SQLite backend 属于 Runtime/backend prepare 层，
  `PredictionBatchReader` 属于底层 reader，`PredictionCacheExpertProvider` 属于 smoke-only
  prediction-cache adapter；明确三列 `features.csv` 后续由 TimeFuse 17 维
  `FeatureProvider` 或 Visual history window / pseudo image / ViT `FeatureProvider` 替换；
  generic small entrypoint 继续保持 thin CLI，branch-specific feature/head 验证另走
  branch-specific smoke 或 small entrypoint；
- P13d 已完成 prediction backend -> `ExpertBatch` small smoke，见
  `docs/refactor/stage1_prediction_backend_expertbatch_smoke.md`；新增
  `tests/smoke/stage1_prediction_backend_expertbatch_smoke.py`，使用 P13b
  real-derived manifest 的 ordered sample_keys 和 P13b `expert_predictions.json` 数值参考，
  在 tempfile 内构造 packed_npy_v1 prediction manifest、数组和 SQLite backend，经
  shared SQLite backend、`PredictionBatchReader` 和 `PredictionCacheExpertProvider` 输出
  `ExpertBatch` 并验证 sample_key、model_columns、shape、row index lineage 和数值一致性；
  P13d 不迁移正式入口、不替换 Visual `SQLitePredictionIndex`，P13b JSON 仍不是正式
  backend schema；
- P13e 已完成 TimeFuse 17 维 `FeatureProvider` small smoke，见
  `docs/refactor/stage1_timefuse_17dim_feature_provider_smoke.md`；新增
  `tests/fixtures/stage1_timefuse_17dim_small/` 和
  `tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py`，使用 P13b manifest 的
  ordered sample_keys 和仓库内小型 17 维 TimeFuse-style feature CSV，经
  `TimeFuseFeatureCacheProvider` 输出 `FeatureBatch`，验证 sample_key 保序、`[sample, 17]`
  shape、feature schema、provider extra 和数值一致性；P13e 不接 TimeFuse head/evaluator，
  不扩展 generic small entrypoint，不读取 oracle/error/prediction；
- P14a 已完成 Visual FeatureProvider insertion audit，见
  `docs/refactor/stage1_visual_feature_provider_insertion_audit.md`；审计确认未来 Visual
  provider 应输出 `FeatureBatch`，只提供可部署视觉特征和轻量 lineage，不读取 prediction
  cache、oracle/error、run_dir/checkpoint/status，也不接管 Visual RouterHead、loss、optimizer
  或正式输出 schema；
- P14b 已完成 Visual FeatureProvider minimal mock/fixture smoke，见
  `docs/refactor/stage1_visual_feature_provider_mock_smoke.md`；新增
  `time_router/features/visual_mock.py`、`tests/fixtures/stage1_visual_feature_mock/` 和
  `tests/smoke/stage1_visual_feature_provider_mock_smoke.py`，使用 P13b manifest ordered
  sample_keys 和小型 history window fixture，经 deterministic encoder stub 输出
  `FeatureBatch(features=(4, 8), dtype=float32)`；smoke 证明 provider 阶段不读取任何文件、
  prediction/oracle/y_true/run_dir/status/checkpoint 或 `/data2`，不接 Visual RouterHead 或
  evaluator；
- P14c 已完成 Visual eval-only canonical bypass plan，见
  `docs/refactor/stage1_visual_eval_canonical_bypass_plan.md`；冻结 future eval-only 链路：
  `SampleManifest ordered sample_keys -> VisualFeatureProvider / mock provider / legacy embedding path
  -> FeatureBatch -> legacy SQLite prediction arrays 或 PredictionCacheExpertProvider
  -> ExpertBatch -> Visual RouterHead / legacy MLP adapter -> RouterOutput
  -> EvaluationInputAdapter -> Evaluator summary/rows -> future Runtime artifact writer`；明确
  `ExpertBatch` 不读取视觉特征，`FeatureBatch` 不读取 prediction/oracle/run_dir，head 只输出
  `RouterOutput`，artifact writer 只在 future canonical run_dir 写出；P14c 不改正式入口、
  不替换 Visual `SQLitePredictionIndex`、不接 `PredictionCacheExpertProvider` 到正式入口、
  不改 legacy 输出 schema；
- P14d 已完成 Visual mock protocol eval smoke，见
  `docs/refactor/stage1_visual_mock_protocol_eval_smoke.md`；新增
  `tests/smoke/stage1_visual_mock_protocol_eval_smoke.py`，使用 P13b manifest ordered
  sample_keys、P14b `VisualMockFeatureProvider` 和 P13b expert JSON 数值参考，在内存中串联
  `FeatureBatch + ExpertBatch -> smoke-only mock RouterHead -> RouterOutput ->
  EvaluationInputAdapter -> summary/rows`；验证 sample_key 保序、model_columns 对齐、
  weights shape/归一化、hard top-1/raw soft fusion 指标和 selected counts 可生成；
  P14d 不改正式入口、不加载真实 ViT、不接 legacy MLP、不写 canonical `run_dir`；
- P14e 已完成 Visual eval-only legacy MLP adapter audit，见
  `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md`；审计确认 legacy
  `VisualMLPRouter` eval-only 输入是 scaler transform 后的 ViT pooled embedding，对应
  head-ready `FeatureBatch.features`；future thin adapter 只消费
  `FeatureBatch + model_columns + runtime-loaded MLP`，输出 `RouterOutput(logits, weights)`，
  并检查 sample_key 保序、model_columns 与 `ExpertBatch.model_columns` 对齐、shape 和
  softmax 权重；scaler fit/checkpoint state、checkpoint loading、resume、device/dtype 和
  DataParallel 仍归 Runtime/entrypoint 管理；P14e 不改正式入口、不新增 adapter 代码；
- P14f 已完成 Visual legacy MLP adapter smoke，见
  `docs/refactor/stage1_visual_legacy_mlp_adapter_smoke.md`；新增
  `tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py`，使用 P13b manifest ordered
  sample_keys、P14b `VisualMockFeatureProvider` 输出的 head-ready float32 `FeatureBatch`
  和 P13b expert JSON 数值参考，在内存中串联 `FeatureBatch + ExpertBatch ->
  smoke-only loaded torch MLP state_dict fixture -> smoke-only thin adapter ->
  RouterOutput -> EvaluationInputAdapter -> summary/rows`；验证 logits/weights shape、
  softmax row sum、有限值、sample_key 保序、model_columns 对齐和 rows 保序；adapter
  阶段 patch 文件 IO、`np.load`、`np.save` 和 `torch.load`，确认不读取 checkpoint、
  prediction、oracle、run_dir 或 `/data2`；P14f 不新增正式 adapter、不改正式入口；
- P15a 已完成 branch-specific small entrypoint decision，见
  `docs/refactor/stage1_branch_specific_small_entrypoints.md`；决策确认 P14 可以收束，
  `scripts/run_stage1_canonical_small.py` 必须继续保持 generic thin CLI，不承载 Visual
  legacy MLP / ViT embedding / SQLitePredictionIndex 逻辑，也不承载 TimeFuse 17 维 feature
  cache / oracle parquet / shard-local SQLite / linear-softmax fusor 逻辑；后续需要分别新增
  TimeFuse-specific 和 Visual-specific small canonical entrypoint，但 P15a 不实现入口、不写
  scripts、不迁移正式训练入口；
- P15b 已完成 TimeFuse-specific small canonical entrypoint thin slice，见
  `docs/refactor/stage1_timefuse_small_entrypoint.md`；新增
  `scripts/run_stage1_timefuse_small.py` 和
  `tests/smoke/stage1_timefuse_small_entrypoint_smoke.py`，使用 P13b real-derived
  sample manifest/expert JSON 与 P13e 17 维 feature fixture，串联
  `SampleManifest -> ExpertBatch -> TimeFuseFeatureCacheProvider / FeatureBatch ->
  TimeFuseLinearSoftmaxHead / RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer`，
  写出 canonical run_dir；P15b 仍不迁移正式 TimeFuse fusor 训练入口，不访问 `/data2`，
  不启动训练、pressure 或 full-scale，不修改 generic small CLI；
- P15c 已完成 Visual-specific small canonical entrypoint thin slice，见
  `docs/refactor/stage1_visual_small_entrypoint.md`；新增
  `scripts/run_stage1_visual_small.py` 和
  `tests/smoke/stage1_visual_small_entrypoint_smoke.py`，使用 P13b real-derived
  sample manifest/expert JSON 与 P14b Visual mock history window fixture，串联
  `SampleManifest -> VisualMockFeatureProvider / FeatureBatch -> ExpertBatch ->
  script-local smoke-only MLP adapter / RouterOutput -> EvaluationInputAdapter ->
  Runtime artifact writer`，写出 canonical run_dir；P15c 仍不迁移正式 Visual Router 训练入口，
  不读取真实 checkpoint，不接真实 ViT，不访问 `/data2`，不启动训练、pressure 或 full-scale，
  不修改 generic small CLI 或 TimeFuse small CLI；
- P15d 已完成 branch-specific small entrypoint artifact parity smoke，见
  `docs/refactor/stage1_branch_small_entrypoint_artifact_parity.md`；新增
  `tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py`，同一 tempfile 下分别运行
  TimeFuse small entrypoint 和 Visual small entrypoint，并比较两边 canonical run_dir 的共同结构、
  `run_metadata.json`、`run_status.json`、`inputs/`、`evaluation/evaluation_summary.json`、
  `predictions/prediction_rows.csv`、ordered sample_keys、split 列、`config_name`、`model_columns`
  和有限指标字段；P15d 只锁定 schema parity，不比较 TimeFuse/Visual 指标优劣，不修改三个
  small CLI，不迁移正式训练入口，不访问 `/data2`，不读取真实 checkpoint，不启动 ViT 或
  full-scale；
- P16a 已完成正式 Visual MLP RouterHead adapter 最小边界，见
  `docs/refactor/stage1_visual_mlp_routerhead_adapter.md`；新增
  `time_router/models/visual_mlp_adapter.py` 和
  `tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py`，通过
  `LoadedTorchMLPRouterHeadAdapter` 包装 Runtime 已加载的 `torch.nn.Module`，只消费
  head-ready `float32 FeatureBatch.features` 与显式 `model_columns`，输出
  `RouterOutput(logits, weights)` 并由 `EvaluationInputAdapter` 复算 summary/rows；P16a
  不读取 checkpoint、不处理 scaler、不启动 ViT、不访问 `/data2`，不导入 legacy
  `VisualMLPRouter` 或正式训练入口，也不把该 adapter 接入 P15c visual small entrypoint；
- P16b 已完成 real Visual feature provider boundary audit，见
  `docs/refactor/stage1_real_visual_feature_provider_audit.md`；明确未来正式 Visual route 可串为
  `SampleManifest / ordered sample_keys -> real VisualFeatureProvider / FeatureBatch ->
  LoadedTorchMLPRouterHeadAdapter / RouterOutput -> EvaluationInputAdapter / Evaluator ->
  Runtime artifact writer`，其中 scaler/checkpoint/ViT/device/cache/run_dir 均有显式边界；
- P16c 已完成 precomputed/head-ready Visual FeatureProvider minimal smoke，见
  `docs/refactor/stage1_visual_precomputed_feature_provider.md`；新增
  `time_router/features/visual_precomputed.py`、
  `tests/fixtures/stage1_visual_precomputed_small/` 和
  `tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py`，用 P13b manifest 的
  ordered sample_keys 读取打乱行顺序的 head-ready embedding fixture，输出
  `FeatureBatch(features=(4, 8), dtype=float32)`，并串到 P16a
  `LoadedTorchMLPRouterHeadAdapter` 与 `EvaluationInputAdapter`；P16c 不接真实 ViT、不做
  pseudo image、不处理 scaler、不读取 checkpoint、不迁移正式入口、不访问 `/data2`；
- pressure / full-scale canonical scripts 尚未准备。

## 5. 下一阶段路线

建议顺序：

1. P16c 已完成 precomputed embedding -> `FeatureBatch` 边界；后续 real Visual provider
   应继续按 fake encoder、scaler boundary 和 online ViT provider 分步推进，不直接迁移正式入口。
2. scaler boundary design/smoke 应单独验证 loaded scaler transform -> head-ready float32
   `FeatureBatch`，并禁止 provider 或 RouterHead adapter silent fit。
3. 后续正式 Visual entrypoint 迁移应在 Runtime 中加载 checkpoint/scaler、准备 head-ready
   features，再把已加载 module 交给 P16a adapter；legacy `VisualMLPRouter` 的 import/signature
   和 checkpoint state_dict 适配仍需单独处理。
4. online ViT provider audit/smoke 应单独处理 pseudo image + frozen ViT + batching +
   device/dtype/resource policy。
5. 后续仍需保持 Provider / Head / Evaluator 不知道 `run_dir`，且不把 Bash 语义下沉到
   `time_router`。
6. 准备 pressure / full-scale 方案时，`scripts/` 仍只作为 thin entrypoint 或 launcher，
   不承载 provider 内部逻辑；Bash launcher 另行分层，不能混入 P12 small CLI。
7. 以 legacy `96_48_S` full-scale 结果作为 reference baseline；canonical pipeline 后续需要
   重跑，不能把旧 schema 作为新 contract 的强兼容来源。

## 6. 当前阶段明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增正式 provider/head/runtime 代码。
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
