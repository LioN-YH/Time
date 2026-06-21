# 工作区结构说明

更新日期：2026-06-21 14:55:58 CST

本文档用于按层次说明 `/home/shiyuhong/Time` 工作区内主要目录、关键文件和生成物的功能。后续新增、删除或移动长期保留的文件/目录时，应同步更新本文档。

## 0. 维护规则

1. **先按层级定位，再补充文件职责**：优先把新增内容挂到已有层级下，例如实验脚本放到 `2.1`，Quito 输出放到 `3.7`。
2. **长期文件逐项记录**：脚本、配置、正式日志、汇总表、结构文档、重要数据产物需要明确用途。
3. **大规模生成物按模式记录**：checkpoint、TensorBoard event、评估日志、缓存目录不逐个文件展开，但需要写明来源、用途和是否能作为正式结果引用。
4. **结果口径必须写清楚**：例如 checkpoint 选择指标是 validation MAE-best 还是 validation MSE-best。
5. **结构变化要留痕**：如果新增结构文档、正式实验产物或结果目录，按 `AGENTS.md` 要求同步写实验日志并更新 `experiment_logs/README.md`。

## 1. 工作区根目录层

根目录负责承载项目级规范、结构索引、实验管理目录，以及两个主要代码库。

```text
/home/shiyuhong/Time
├── .gitignore
├── AGENTS.md
├── EXTERNAL_OUTPUTS.md
├── HANDOFF.md
├── WORKSPACE_STRUCTURE.md
├── docs/
├── scripts/
├── experiment_scripts/
├── experiment_logs/
├── time_router/
├── visual_router_experiments/
├── quito/
├── TimeFuse/
├── .agents/
├── .codex/
└── .git/
```

### 1.1 根目录长期文件

| 路径 | 功能 | 维护要求 |
| --- | --- | --- |
| `.gitignore` | 根仓库忽略规则；排除嵌套外部仓库、本地 agent 状态、大规模数据、checkpoint、cache、运行日志和密钥环境文件 | 新增长期输出根目录或大规模生成物类型时更新 |
| `AGENTS.md` | 项目级 agent 工作规范，记录实验日志、默认 conda `quito` 实验环境、中文计划、中文代码注释、工作区结构文档维护等长期要求 | 修改协作规则时更新 |
| `EXTERNAL_OUTPUTS.md` | 外部大规模输出索引，当前记录 `/data2/syh/Time/` 下的大盘输出和临时 cache shard 策略 | 新增外部输出根目录或调整缓存策略时更新 |
| `HANDOFF.md` | 上下文接近 65% 或长任务需要切换窗口时使用的交接模板，要求记录当前目标、已完成步骤、运行命令、失败点、关键路径、下一步命令和验证口径 | 触发 handoff 时用真实进展替换模板内容；完成继承后可按最新状态继续维护 |
| `WORKSPACE_STRUCTURE.md` | 当前文件，按层级说明工作区结构、关键文件和输出口径 | 新增长期文件/目录后更新 |
| `docs/refactor/` | 重构前审计与迁移设计文档目录；当前包含 Stage 1 路线审计、目标架构、重构路线图、公共模块迁移候选、golden fixture、共享 PredictionBatchReader 说明、共享 OracleTsfReader 说明、evaluation package 边界复核、P4a JSON utils 边界说明、P4b path resolver 边界说明、P4c run metadata 边界说明、P4d run artifacts 边界复核、P4e checkpoint index 边界复核、P4 后 architecture pivot 决策、P5a canonical runtime contract、P5b canonical provider interface design、P5c/P10e protocol types skeleton、P5d provider adapter boundary review、P5e/P10h canonical entrypoint migration plan、P5f launcher architecture、P6a PredictionCacheExpertProvider、P6a.5 expert system boundary review、P6b EvaluationInput adapter、兼容 FusionEvaluator adapter、P7a/P7b/P7c TimeFuse adapter smoke 文档、P8a TimeFuse 正式入口 adapter 插入审计、P8c evaluation adapter pressure 验证、P8d TimeFuse baseline parity review、P9a Visual Router 正式入口 adapter 插入审计、P9b Visual Router evaluation adapter 旁路校验说明、P9c Visual Router evaluation adapter pressure 验证、P9d Visual Router ExpertBatch evaluation bridge、P9e Visual Router prediction cache provider gap audit、P9f Visual Router training ExpertBatch bypass、P10a shared prediction SQLite backend audit、P10b Prediction SQLite backend helper、P10c prediction array IO boundary consolidation、P10d/P10e/P10h canonical SampleManifest / supervision boundary、P10f Visual labels sample/supervision adapter、P10g TimeFuse feature/oracle sample/supervision adapter、P11a canonical run artifact schema、P11b canonical SampleManifest physical schema、P11c minimal Runtime artifact writer、P11d canonical protocol run smoke、P12 small canonical entrypoint thin slice、P12b small fixture input contract、P13a real small-input mapping audit、P13b real-derived small fixture smoke、P13c real small backend/provider connection audit、P13d prediction backend -> ExpertBatch small smoke、P13e TimeFuse 17 维 FeatureProvider small smoke、P14a Visual FeatureProvider insertion audit、P14b Visual FeatureProvider mock smoke、P14c Visual eval-only canonical bypass plan、P14d Visual mock protocol eval smoke、P14e Visual legacy MLP adapter audit、P14f Visual legacy MLP adapter smoke、P15a branch-specific small entrypoint decision、P15b TimeFuse-specific small canonical entrypoint、P15c Visual-specific small canonical entrypoint、P15d branch-specific small entrypoint artifact parity、P16a formal Visual MLP RouterHead adapter、P16b real Visual feature provider boundary audit、P16c Visual precomputed FeatureProvider smoke、P16d loaded Visual FeatureScaler smoke、P16e Visual feature architecture variant boundary、P16f Visual feature chain protocol skeleton、P16g Visual legacy MLP checkpoint/signature audit、P16h Visual legacy MLP loaded-module smoke 和 P16i Visual legacy MLP checkpoint payload smoke | 路线或迁移结论变化时更新；代码迁移应另写实验日志和验证结果 |
| `docs/refactor/stage1_route_audit.md` | Stage 1 共享主干、Visual/TimeFuse 分支、废弃路线及 36 个 Python 文件标签审计 | 新增/归档 Stage 1 脚本或正式路线改变时同步复核 |
| `docs/refactor/stage1_target_architecture.md` | Stage 1 未来目标架构设计，定义 `time_router/{data,io,features,models,evaluation,training}`、`scripts/`、`configs/`、`exp_scripts/` 和 `archive/` 边界，并明确共享主干与 Visual/TimeFuse 两个 FeatureProvider 分支；P6a.5 起补充 ExpertProvider / ExpertBatch 长期专家系统边界与 PredictionCacheExpertProvider adapter 实现边界 | 当前只作为设计文档；实现 package、迁移入口或归档旧代码时需另行验证并更新 |
| `docs/refactor/stage1_refactor_roadmap.md` | Stage 1 后续小步重构路线图，按 P0-P16i 及 P2.5/P3a-P3e/P6a.5 等中间小步拆分 architecture docs、prediction reader、oracle/TSF reader、metrics/fusion、router weight diagnostics、summary、per-sample rows、evaluation package 边界复核、logging/path/config、FeatureProvider、专家系统边界审计、入口迁移、shared prediction SQLite backend、prediction array IO boundary consolidation、canonical SampleManifest / supervision boundary、最小协议 smoke、Visual labels adapter smoke、TimeFuse feature/oracle adapter smoke、canonical dataflow alignment review、canonical run artifact schema、canonical SampleManifest physical schema、minimal Runtime artifact writer、canonical protocol run smoke、small canonical entrypoint thin slice、small fixture input contract hardening、real small-input mapping audit、real-derived small fixture smoke、real small backend/provider connection audit、prediction backend -> ExpertBatch small smoke、TimeFuse 17 维 FeatureProvider small smoke、Visual mock protocol eval smoke、Visual legacy MLP adapter audit、Visual legacy MLP adapter smoke、branch-specific small entrypoint decision、TimeFuse small entrypoint、Visual small entrypoint、branch-specific artifact parity smoke、formal Visual MLP RouterHead adapter、real Visual feature provider boundary audit、Visual precomputed FeatureProvider smoke、loaded Visual FeatureScaler smoke、Visual feature architecture variant boundary、Visual feature chain protocol skeleton、Visual legacy MLP checkpoint/signature audit、Visual legacy MLP loaded-module smoke 和 Visual legacy MLP checkpoint payload smoke；后续正式入口迁移仍需另起步骤 | 每个迁移步骤前后都应运行对应 smoke 并写实验日志 |
| `docs/refactor/stage1_migration_candidates.md` | manifest、prediction cache、oracle/TSF、SQLite/batch reader、metrics、logging、路径和训练骨架的后续收束候选 | 只记录建议；实际重构完成后更新状态与兼容性结论 |
| `docs/refactor/golden_fixture.md` | Stage 1 重构前 golden fixture 说明，记录 4 sample packed dry-run fixture 来源、锁定契约和 smoke 运行命令 | 后续调整 golden fixture 或重构验收口径时同步更新；不代表正式逻辑已重构 |
| `docs/refactor/prediction_batch_reader.md` | Stage 1 P1 共享 `PredictionBatchReader` 接口说明，记录输入、输出、约束和后续正式入口迁移方式 | reader 接口或迁移策略变化时更新；正式入口接入另按 P6 记录 |
| `docs/refactor/oracle_tsf_reader.md` | Stage 1 P2 共享 `OracleTsfReader` 接口说明，记录 oracle/TSF 读取、保序、缺失报告、join lineage、用途约束、正式入口禁止 `allow_full_scan=True` 和 full-scale 后续 SQLite / shard-local / batch query 要求 | reader 接口或 oracle/TSF 迁移策略变化时更新；正式入口接入另按 P6 记录 |
| `docs/refactor/evaluation_package_boundary.md` | Stage 1 P3e `time_router/evaluation` package 边界复核与 consolidation 规划 | 记录 `metrics.py`、`summary.py`、`prediction_rows.py`、`__init__.py` 的职责、public/private API、当前不合并/不拆分判断和未来整理门禁；本身不改变 helper 行为或正式 output schema |
| `docs/refactor/json_utils.md` | Stage 1 P4a JSON 原子写入和最小 status writer 边界说明 | 记录 `atomic_write_json`、`build_status_payload`、`write_status_json` 的职责、UTF-8 / 同目录临时文件 / `flush + fsync + os.replace` 约束，以及不实现 path/config/logging framework、不迁移正式入口的边界 |
| `docs/refactor/path_resolver.md` | Stage 1 P4b 最小 path resolver 边界说明 | 记录 `find_repo_root`、`resolve_under_root`、`resolve_status_path`、`resolve_metadata_path` 的职责、root marker、逃逸 root 防护和不实现 config/logging/checkpoint index、不接入 full-scale 输出目录的边界 |
| `docs/refactor/run_metadata.md` | Stage 1 P4c 最小 run metadata payload builder 边界说明 | 记录 `build_run_metadata` 和 `write_run_metadata` 的字段约束、UTC 时间、Path 转字符串、`extra` 边界，以及不自动调用 git、不读取命令行/训练配置、不改变既有 metadata schema 的边界 |
| `docs/refactor/run_artifacts_boundary.md` | Stage 1 P4d run artifacts 边界复核与接入规划 | 复核 `time_router/io` 中 prediction cache reader、JSON/status writer、path resolver、run metadata 和 public API 聚合入口的职责边界；明确低风险 IO helper 与正式训练入口/launcher/resume 层的分工、接入前 status/metadata 字段审计要求、P4e/P4f/P4g 后续候选和不接入 full-scale 的边界 |
| `docs/refactor/checkpoint_index_boundary.md` | Stage 1 P4e checkpoint index 边界复核与接入规划 | 审查 Visual Router / TimeFuse-style fusor 当前 checkpoint、latest 指针、resume、launcher、monitor、`status.json` 和 `metadata.json` 约定；明确现有 `latest_checkpoint_index.json` 仍是入口私有文件，未来 helper 更适合 training/runtime 层而不是低风险 `time_router/io` helper；本身不实现 checkpoint index、不改训练入口 |
| `docs/refactor/stage1_architecture_pivot_after_p4.md` | Stage 1 P4 后 architecture pivot 决策 | 明确 P4 后暂停 config system，转向 P5 canonical entrypoint / FeatureProvider design；正式保留 streaming Visual Router 和 streaming TimeFuse-style fusor baseline 两条主干；将 LogisticRegression fusor、offline ViT embedding cache、旧 OOM lookup、pilot-only 和非 streaming full-scale 入口标记为 archive/deprecated/reference-only；定义新 canonical runtime 最小契约和 helper 接入边界 |
| `docs/refactor/stage1_canonical_runtime_contract.md` | Stage 1 P5a canonical runtime contract | 定义未来新 canonical `run_dir` 结构、`status.json` / `metadata.json` 最小字段、Visual Router 与 TimeFuse-style fusor 共享字段和 branch-specific extra、P4 helper 接入边界、checkpoint index 最小概念和旧 status/metadata/checkpoint schema 舍弃边界；本身不改训练入口、不实现 helper |
| `docs/refactor/stage1_canonical_run_artifact_schema.md` | Stage 1 P11a canonical run artifact schema | 冻结 future canonical `run_dir` 推荐结构：`run_metadata.json`、`run_status.json`、`inputs/`、`indexes/`、`predictions/`、`evaluation/`、`checkpoints/` 和 `logs/`；明确 metadata/status 动静态边界、evaluation 与 predictions 边界、Runtime 与 Provider 路径边界、最小 versioning strategy、legacy `96_48_S` full-scale policy，以及 Visual / TimeFuse-style branch-specific artifact policy；本身不改正式入口、不新增 runtime/helper/scripts、不访问 `/data2` |
| `docs/refactor/stage1_sample_manifest_physical_schema.md` | Stage 1 P11b canonical SampleManifest physical schema | 冻结 `stage1_sample_manifest_v1` 最小物理字段、`sample_key` 规则、`stage1_split_summary_v1` split summary schema、`run_dir/inputs/` snapshot/reference 保存方式、Visual labels 与 TimeFuse feature/oracle source 映射策略，以及 feature/oracle/error/prediction cache path 不进入 `SampleManifest` 的边界；本身不改正式入口、不新增 runtime/helper/scripts、不访问 `/data2` |
| `docs/refactor/stage1_runtime_artifact_writer.md` | Stage 1 P11c minimal Runtime artifact writer | 记录 `time_router.runtime.artifact_writer` 的最小 API、canonical `run_dir` 写出结构、Runtime / Provider / Head / Evaluator 边界、明确不做范围和 tempfile smoke 验收；说明 helper 只属于 Runtime artifact 写出层，不迁移正式 legacy entrypoint |
| `docs/refactor/stage1_canonical_protocol_run_smoke.md` | Stage 1 P11d canonical protocol run smoke | 记录 tiny smoke 如何串联 `SampleManifest -> ExpertBatch -> FeatureBatch -> RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer -> canonical run_dir`，验证 ordered sample_keys 贯通、predictions/evaluation 分层、Provider/Head/Evaluator 不知道 `run_dir`；说明本阶段仍不是正式入口迁移 |
| `docs/refactor/stage1_canonical_small_entrypoint.md` | Stage 1 P12 small canonical entrypoint thin slice | 记录 `scripts/run_stage1_canonical_small.py` 的薄 CLI 参数、tiny canonical dataflow、canonical `run_dir` artifact 写出、`scripts/` 与 provider/head/evaluator/runtime writer 边界、明确不做范围和 P12/P12b/P13 后续连接；说明本阶段仍不是正式入口迁移 |
| `docs/refactor/stage1_canonical_small_fixture_contract.md` | Stage 1 P12b small canonical fixture input contract | 记录 small entrypoint 显式 fixture 文件契约：`sample_manifest.csv` 使用 P11b 最小字段且行顺序作为 ordered sample_keys，`features.csv` 可乱序但 provider 必须按 manifest 保序，`expert_predictions.json` 保存小数组 `model_columns/y_true/y_pred` 并按 manifest 组装 `ExpertBatch`；明确 P12b 不是正式入口迁移，后续用于 P13 审计真实 Visual/TimeFuse 小规模输入映射 |
| `docs/refactor/stage1_real_small_input_mapping_audit.md` | Stage 1 P13a real small-input mapping audit | 审计真实 Visual labels / legacy metadata、TimeFuse feature/oracle source、真实 feature source、prediction cache / SQLite backend 到 P12b fixture contract 的映射；明确 `SampleManifest` 只保存样本身份、split、顺序和轻量 lineage，oracle/error 属于 `SupervisionProvider`，TimeFuse 17 维 feature 与 Visual history/pseudo image/ViT feature 属于 `FeatureProvider`，P12b expert JSON 只是 tiny fixture，正式路径仍走 prediction backend / `ExpertProvider` / `ExpertBatch`；本身不创建真实 fixture、不访问 `/data2`、不迁移正式入口 |
| `docs/refactor/stage1_real_derived_small_fixture.md` | Stage 1 P13b real-derived small fixture smoke | 记录 `tests/fixtures/stage1_real_derived_small/` 的来源、字段口径和 smoke 验收；该 fixture 从 P10f/P10g smoke 的 ETTh1 / ETTm2 / weather 小样本身份派生，用 P12b small entrypoint 验证 manifest 保序、feature/expert join、canonical `run_dir` 写出、metadata inputs 来源摘要和 evaluation sample_count；明确三列 feature 只是 schema-style fixture，不是 TimeFuse 17 维 full-scale cache，expert JSON 也不是正式 prediction backend |
| `docs/refactor/stage1_real_small_backend_provider_connection_audit.md` | Stage 1 P13c real small backend/provider connection audit | 冻结真实 small batch 后续从 P13b fixture-driven path 迁移到 backend/provider 的连接方案：`expert_predictions.json` 后续由 prediction backend / `ExpertProvider` / `ExpertBatch` 替换，shared prediction SQLite backend 属于 Runtime/backend prepare 层，`PredictionBatchReader` 属于底层 reader，`PredictionCacheExpertProvider` 属于 smoke-only adapter；三列 feature fixture 后续由 TimeFuse 17 维 `FeatureProvider` 或 Visual history window / pseudo image / ViT `FeatureProvider` 替换；generic small entrypoint 保持 thin CLI，branch-specific feature/head 验证另走 smoke 或 small entrypoint；本身不改正式入口、不访问 `/data2`、不启动训练 |
| `docs/refactor/stage1_prediction_backend_expertbatch_smoke.md` | Stage 1 P13d prediction backend -> ExpertBatch small smoke | 记录 P13d 如何使用 P13b manifest ordered sample_keys 和 P13b expert JSON 数值参考，在 tempfile 内构造 packed_npy_v1 prediction manifest、数组和 SQLite backend，经 shared SQLite backend、`PredictionBatchReader`、`PredictionCacheExpertProvider` 输出 `ExpertBatch`；明确 P13b expert JSON 只是参考而非正式 backend schema，`validate_manifest_schema=False` 只用于非 canonical sample_key smoke bridge，不接正式入口、不替换 Visual `SQLitePredictionIndex` |
| `docs/refactor/stage1_timefuse_17dim_feature_provider_smoke.md` | Stage 1 P13e TimeFuse 17 维 FeatureProvider small smoke | 记录 P13e 如何使用 P13b manifest ordered sample_keys 和仓库内小型 `features_17d.csv`，经 `TimeFuseFeatureCacheProvider` 输出 `FeatureBatch`；验证 sample_key 保序、`[sample, 17]` feature shape、17 维 schema metadata、provider extra 和数值一致性；明确本阶段不接 TimeFuse head/evaluator、不扩展 generic small entrypoint、不读取 oracle/error/prediction、不访问 `/data2` |
| `docs/refactor/stage1_visual_feature_provider_insertion_audit.md` | Stage 1 P14a Visual FeatureProvider insertion audit | 审计 `train_visual_router_online_streaming.py` 中 Visual history window、pseudo image、frozen ViT forward、router feature shape、Visual MLP head、SQLite prediction path、ExpertBatch bypass、device/dtype/DataParallel、latency 和 checkpoint/resume 的边界；明确未来 Visual provider 输出 `FeatureBatch(sample_keys, features, feature_schema, extra)`，provider 不读取 prediction cache、oracle/error、run_dir/checkpoint/status，也不接管 RouterHead/loss/evaluation 写出；本身不改正式入口、不新增 provider、不访问 `/data2` |
| `docs/refactor/stage1_visual_feature_provider_mock_smoke.md` | Stage 1 P14b Visual FeatureProvider mock smoke | 记录 P14b 如何使用 P13b manifest ordered sample_keys 和 `tests/fixtures/stage1_visual_feature_mock/history_windows.json`，经 `VisualMockFeatureProvider` 与 deterministic encoder stub 输出 `FeatureBatch(features=(4, 8), dtype=float32)`；验证 sample_key 保序、visual mock schema、history_source、pseudo_image/mock_not_materialized、encoder_stub 口径和 provider extra；明确本阶段不是正式 ViT provider，不接 Visual RouterHead/evaluator，不读取 prediction/oracle/run_dir，不访问 `/data2` |
| `docs/refactor/stage1_visual_eval_canonical_bypass_plan.md` | Stage 1 P14c Visual eval-only canonical bypass plan | 冻结 future eval-only 链路：`SampleManifest -> FeatureBatch -> ExpertBatch -> RouterOutput -> EvaluationInputAdapter -> Evaluator -> future Runtime artifact writer`；明确 legacy SQLite batch arrays 短期只在已加载后包装为 `ExpertBatch`，Visual mock/future/legacy embedding path 输出 `FeatureBatch`，Visual head 或 legacy MLP thin adapter 输出 `RouterOutput`；本阶段不改正式入口、不替换 `SQLitePredictionIndex`、不接 `PredictionCacheExpertProvider`、不改 legacy 输出 schema |
| `docs/refactor/stage1_visual_mock_protocol_eval_smoke.md` | Stage 1 P14d Visual mock protocol eval smoke | 记录 P14d 如何使用 P13b manifest ordered sample_keys、P14b `VisualMockFeatureProvider` 和 P13b expert JSON 数值参考，在内存中串联 `FeatureBatch + ExpertBatch -> smoke-only mock RouterHead -> RouterOutput -> EvaluationInputAdapter -> summary/rows`；明确 expert JSON 只是 small fixture 参考，不是真实 backend schema；本阶段不接真实 ViT、不接 legacy MLP、不写 `run_dir`、不迁移正式入口 |
| `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md` | Stage 1 P14e Visual eval-only legacy MLP adapter audit | 审计 legacy `VisualMLPRouter` eval-only 输入、输出和最小 adapter 边界；明确 head-ready `FeatureBatch.features` 对应 scaler transform 后的 ViT pooled embedding，legacy MLP logits 经 softmax 包装为 `RouterOutput(logits, weights)`，`model_columns` 必须与 `ExpertBatch.model_columns` 显式对齐；scaler fit/checkpoint state、checkpoint loading、resume、device/dtype/DataParallel 归 Runtime/entrypoint 管理；本阶段不新增正式 adapter、不改正式入口 |
| `docs/refactor/stage1_visual_legacy_mlp_adapter_smoke.md` | Stage 1 P14f Visual legacy MLP adapter smoke | 记录 P14f 如何使用 P13b manifest ordered sample_keys、P14b `VisualMockFeatureProvider` 输出的 head-ready float32 `FeatureBatch`、P13b expert JSON 数值参考和 smoke-only loaded torch MLP state_dict fixture，在内存中串联 `FeatureBatch + ExpertBatch -> smoke-only thin adapter -> RouterOutput -> EvaluationInputAdapter -> summary/rows`；明确该 adapter 定义在 smoke 内，不是正式 Visual RouterHead adapter，不读取 checkpoint/prediction/oracle/run_dir 或 `/data2` |
| `docs/refactor/stage1_branch_specific_small_entrypoints.md` | Stage 1 P15a branch-specific small entrypoint decision | 决策 P14 可以收束，generic `scripts/run_stage1_canonical_small.py` 必须继续保持 thin，不承载 Visual legacy MLP / ViT embedding / SQLitePredictionIndex 或 TimeFuse 17 维 feature cache / oracle parquet / shard-local SQLite / linear-softmax fusor 逻辑；明确后续需要分别新增 TimeFuse-specific 和 Visual-specific small canonical entrypoint，但 P15a 不新增入口、不写 scripts、不迁移正式训练入口 |
| `docs/refactor/stage1_timefuse_small_entrypoint.md` | Stage 1 P15b TimeFuse-specific small canonical entrypoint | 记录 `scripts/run_stage1_timefuse_small.py` 的 small fixture 输入、TimeFuse 17 维 provider/head/evaluator/runtime 串联链路、canonical run_dir 输出、与 generic small CLI 的区别、与 future full-scale TimeFuse fusor 的关系和明确不做范围；说明 P15b 不是正式训练入口迁移 |
| `docs/refactor/stage1_visual_small_entrypoint.md` | Stage 1 P15c Visual-specific small canonical entrypoint | 记录 `scripts/run_stage1_visual_small.py` 的 small fixture 输入、`VisualMockFeatureProvider`、script-local smoke-only MLP adapter、`ExpertBatch`、evaluator/runtime 串联链路、canonical run_dir 输出、与 generic/TimeFuse small CLI 的区别、与 future 正式 Visual Router 迁移的关系和明确不做范围；说明 P15c 不是正式训练入口迁移 |
| `docs/refactor/stage1_branch_small_entrypoint_artifact_parity.md` | Stage 1 P15d branch-specific small entrypoint artifact parity | 记录 P15d 在 P15b/P15c 之后、真实迁移之前锁定 TimeFuse/Visual small canonical run_dir 共同结构和 schema 的目标；说明共同 metadata/status/inputs/evaluation/prediction rows 字段、允许的 branch-specific metadata、不比较指标优劣、明确不访问 `/data2`、不迁移正式入口、不读取 checkpoint、不启动 ViT 或 full-scale |
| `docs/refactor/stage1_visual_mlp_routerhead_adapter.md` | Stage 1 P16a formal Visual MLP RouterHead adapter | 记录 `LoadedTorchMLPRouterHeadAdapter` 的正式最小边界：Runtime 已加载 `torch.nn.Module` + head-ready float32 `FeatureBatch` + 显式 `model_columns` -> `RouterOutput(logits, weights)`；明确 adapter 不读取 checkpoint、不处理 scaler、不启动 ViT、不接 prediction backend、不知道 `run_dir` 或 Bash |
| `docs/refactor/stage1_real_visual_feature_provider_audit.md` | Stage 1 P16b real Visual feature provider boundary audit | 审计真实 Visual feature chain 从 history window、pseudo image、frozen ViT embedding、可选 scaler/normalizer 到 head-ready `FeatureBatch` 的边界；明确 scaler/checkpoint/ViT/device/cache/run_dir 分层，cache 不是 provider interface，P16b 不新增 provider、不修改正式入口、不访问 `/data2` |
| `docs/refactor/stage1_visual_precomputed_feature_provider.md` | Stage 1 P16c Visual precomputed FeatureProvider smoke | 记录 `VisualPrecomputedFeatureProvider` 的最小边界：读取仓库内 precomputed/head-ready visual embedding CSV fixture，按 requested sample_keys 输出 `FeatureBatch(features=np.float32)`，并可接入 P16a `LoadedTorchMLPRouterHeadAdapter` 与 `EvaluationInputAdapter`；明确本阶段不是真实 ViT provider，不做 pseudo image/scaler/checkpoint/正式入口迁移 |
| `docs/refactor/stage1_visual_feature_scaler.md` | Stage 1 P16d loaded Visual FeatureScaler smoke | 记录 `LoadedFeatureScaler` 的最小边界：使用已加载 scaler state 对 raw/pre-head `FeatureBatch` 执行 `(raw - mean) / scale`，输出 head-ready `float32 FeatureBatch`，并可接入 P16a adapter 与 `EvaluationInputAdapter`；明确本阶段不做 scaler fit/state discovery/checkpoint/ViT/pseudo image/正式入口迁移 |
| `docs/refactor/stage1_visual_feature_architecture_variants.md` | Stage 1 P16e Visual feature architecture variant boundary | 记录 P16a-P16d 后的可替换视觉特征架构插槽边界；明确长期固定的是 ordered sample_keys、`FeatureProvider` / `FeatureTransform`、`FeatureBatch`、RouterHead adapter、Evaluator 和 Runtime artifact writer 交接契约，长期不固定的是 RevIN、resize、pseudo image、encoder、CLS/mean_patch pooling、scaler/normalizer、precompute embedding 和 cache；说明 `VisualPrecomputedFeatureProvider` 只是 fixture/debug/ablation 路径，`LoadedFeatureScaler` 只是一个显式 transform，CSV/cache/scaler 都不是长期强制方案 |
| `docs/refactor/stage1_visual_feature_chain_protocol.md` | Stage 1 P16f Visual feature chain protocol skeleton | 记录 `RawWindowProvider -> PreImageTransform -> PseudoImageTransformer -> ResizePolicy -> VisualEncoderProvider -> PoolingStrategy -> FeatureTransform -> FeatureBatch` 的最小协议骨架；明确 P16f 只定义可替换插槽和轻量 lineage，不实现真实 RevIN、pseudo image、resize、ViT、pooling、scaler fit、cache path、checkpoint 或正式入口迁移 |
| `docs/refactor/stage1_visual_legacy_mlp_checkpoint_signature_audit.md` | Stage 1 P16g Visual legacy MLP checkpoint/signature audit | 审计 legacy `VisualMLPRouter` constructor、forward、输入 feature_dim、输出 logits/model_columns、streaming checkpoint payload、`router_state_dict`、`scaler_state`、DataParallel `module.` 前缀和 device/runtime loading 边界；明确 checkpoint/scaler/device/strict loading 属于 Runtime，P16a adapter 只接收已加载 torch module 和 head-ready `FeatureBatch` |
| `docs/refactor/stage1_visual_legacy_mlp_loaded_module_smoke.md` | Stage 1 P16h Visual legacy MLP loaded-module smoke | 记录 P16h 如何 import legacy `VisualMLPRouter` 定义，用 P13b ordered sample_keys、P16c head-ready fixture、P13b expert JSON 和 in-memory fake state_dict 验证 normal / `module.` 前缀 key 清洗后 strict load，并将已加载 module 交给 P16a adapter 与 `EvaluationInputAdapter`；明确本阶段不实现 checkpoint loader、不读取真实 checkpoint、不处理真实 scaler、不启动 ViT、不迁移正式入口 |
| `docs/refactor/stage1_visual_legacy_mlp_checkpoint_payload.md` | Stage 1 P16i Visual legacy MLP checkpoint payload smoke | 记录 P16i 如何新增 Runtime-side checkpoint payload helper，并在 tempfile 内创建 tiny checkpoint payload 覆盖 `router_state_dict`、`scaler_state`、`config` 和 `metadata`，验证显式 checkpoint path 读取、`module.` 前缀清理、strict load 到已构造 legacy `VisualMLPRouter`、P16a adapter 消费和 `EvaluationInputAdapter` summary/rows；明确本阶段不读取真实 checkpoint、不访问 `/data2`、不处理真实 scaler transform、不启动 ViT、不迁移正式入口 |
| `docs/refactor/stage1_provider_interface.md` | Stage 1 P5b canonical provider interface design | 定义 `ExperimentProtocol -> SplitStrategy -> ExpertProvider -> FeatureProvider -> RouterHead -> Evaluator` 的共享接口边界；明确当前 fixed config / five experts / prediction cache / vali-test 只是默认实现，接口不写死 frozen ViT、17 维 feature cache、固定 split 或固定训练方式；记录 Visual Router 与 TimeFuse-style fusor 共享 contract、branch-specific extra、oracle/TSF 禁止作为可部署 test-time 动态调权特征、provider 不决定 `run_dir` 和 deprecated/reference-only 历史路线边界；本身不改训练入口、不实现接口代码 |
| `docs/refactor/protocol_types.md` | Stage 1 P5c/P10e protocol types skeleton | 记录 `time_router.protocols` 中 `SplitSpec`、`SampleManifestRow`、`SampleManifest`、`SupervisionBatch`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput` 和 `ExperimentProtocolSpec` 的字段、轻量 contract 边界、tuple/default_factory 约束、public API、smoke 覆盖和明确不做范围；本身不实现 provider/runtime/config/checkpoint/logging，也不迁移正式入口 |
| `docs/refactor/provider_adapter_boundary.md` | Stage 1 P5d provider adapter boundary review | 审查 `PredictionBatchReader`、`prediction_array_io`、Visual pseudo image / ViT feature 路径、TimeFuse 17 维 feature cache reader、Visual Router head、TimeFuse linear-softmax head 和 `time_router.evaluation` public API 的未来 adapter 适配边界；明确第一批应先做 entrypoint migration plan，再优先实现 `PredictionCacheExpertProvider`，TimeFuse feature cache provider 作为后续 feature-only adapter，Visual online ViT provider 不作为第一批最小实现；本身不实现 provider adapter、不改训练入口、不接入 `/data2` |
| `docs/refactor/stage1_entrypoint_migration_plan.md` | Stage 1 P5e/P10h canonical entrypoint migration plan | P10h 起从旧的 Visual Router / TimeFuse 分入口职责拆解视角，调整为 `SampleManifest + SplitStrategy -> ExpertProvider / prediction backend -> SupervisionProvider -> FeatureProvider -> RouterHead -> EvaluationInputAdapter / Evaluator -> Runtime / artifact writer` 的 canonical dataflow；明确共用层包括 `SampleManifest`、`SplitStrategy` 语义、prediction SQLite backend / `ExpertBatch`、`SupervisionBatch` / `SupervisionProvider`、Evaluator metrics 和 run artifact contract 方向，branch-specific 层保留 Visual Quito history / pseudo image / ViT / MLP / objective 与 TimeFuse 17 维 feature cache / scaler / linear-softmax / SmoothL1 objective；本身不改训练代码、不实现 adapter |
| `docs/refactor/launcher_architecture.md` | Stage 1 P5f launcher architecture | 设计未来 `exp_scripts/*.sh -> scripts/*.py -> time_router runtime/protocol/provider/head/evaluator` 启动分层；明确 `exp_scripts/` 负责 Bash launcher、config 选择、GPU/conda/env、logging、后台策略、显式 `/data2` run_dir/output_root 和可复现实验命令，`scripts/` 只做极薄 Python entrypoint，`configs/` 保存 Stage/config/branch 参数与扩展点，`time_router/` 不知道 Bash 存在且不决定 run_dir；给出 P5f 后先做 `PredictionCacheExpertProvider` smoke-only、再做 evaluator adapter、config skeleton、scripts skeleton、Bash launcher 的低风险顺序；本身不新增 Bash/Python 入口、不实现 config/runtime/provider、不改训练脚本 |
| `docs/refactor/prediction_cache_expert_provider.md` | Stage 1 P6a PredictionCacheExpertProvider | 记录 `time_router.experts.PredictionCacheExpertProvider` 的 smoke-only adapter API、与 `PredictionBatchReader` 的关系、`ExpertBatch.extra` 轻量 metadata、`row_index_metadata` lineage、明确不做范围和后续接入顺序；本身不迁移正式 Visual Router / TimeFuse fusor 入口 |
| `docs/refactor/expert_system_boundary_review.md` | Stage 1 P6a.5 expert system boundary review | 冻结 P6a 之后、P6b EvaluationInput adapter 之前的专家系统边界：`ExpertProvider / ExpertBatch` 是 Time framework 长期专家系统 contract，`PredictionCacheExpertProvider` 只是 Stage 1 canonical experiment 的 prediction-cache adapter；明确固定五专家顺序只属于当前 Stage 1 canonical experiment，不上升为全局专家系统契约；明确 P6b 后续消费 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 而不重新读取 prediction cache；本身不改 reader/provider 行为、不新增 runtime/config/launcher/entrypoint |
| `docs/refactor/evaluation_input_adapter.md` | Stage 1 P6b/P6c EvaluationInput adapter | 记录 `time_router.evaluation.EvaluationInputAdapter` 的 smoke-only canonical adapter API、`ExpertBatch + RouterOutput.weights / explicit fusion weights -> EvaluationInput -> evaluation public API` 适配流程、纯内存 summary/rows 输出、明确不做范围和 smoke 验收；P6c 起说明 `FusionEvaluator` 只作为 legacy/compat wrapper；本身不迁移正式 Visual Router / TimeFuse fusor 入口 |
| `docs/refactor/fusion_evaluator_adapter.md` | Stage 1 P6c FusionEvaluator compat adapter | 记录 `time_router.evaluation.FusionEvaluator` 的 legacy/compat API、`ExpertBatch + RouterOutput -> EvaluationInput -> EvaluationInputAdapter` 适配流程、纯内存 summary/rows 输出、明确不做范围和兼容 smoke 验收；本身不迁移正式 Visual Router / TimeFuse fusor 入口 |
| `docs/refactor/timefuse_feature_cache_provider.md` | Stage 1 P7a TimeFuseFeatureCacheProvider | 记录 `time_router.features.TimeFuseFeatureCacheProvider` 的 smoke-only adapter API、`feature CSV -> FeatureBatch` 适配流程、`feature_schema` / `extra` 轻量 metadata、明确不做 prediction/oracle/scaler/run_dir/正式入口迁移的范围和 smoke 验收 |
| `docs/refactor/timefuse_linear_head.md` | Stage 1 P7b TimeFuseLinearSoftmaxHead | 记录 `time_router.models.TimeFuseLinearSoftmaxHead` 的 smoke-only adapter API、`FeatureBatch.features -> RouterOutput(logits, weights)` 适配流程、固定线性权重和 stable softmax 约束、明确不训练/不读 cache/不写运行产物/不迁移正式入口的范围和 smoke 验收 |
| `docs/refactor/timefuse_protocol_chain_smoke.md` | Stage 1 P7c TimeFuse protocol chain smoke | 记录 smoke-only 链路 `PredictionCacheExpertProvider -> ExpertBatch -> TimeFuseFeatureCacheProvider -> FeatureBatch -> TimeFuseLinearSoftmaxHead -> RouterOutput -> EvaluationInputAdapter -> summary/rows` 的目标、IO 边界、deterministic 口径和验收命令；本身不迁移正式 TimeFuse fusor / Visual Router 入口 |
| `docs/refactor/timefuse_entrypoint_adapter_insertion_audit.md` | Stage 1 P8a TimeFuse entrypoint adapter insertion audit | 审计 `train_timefuse_fusor_streaming.py` 的最小 `EvaluationInputAdapter` 接入点，结论为先在 `evaluate_streaming(...)` 中 torch fusor 产出 `weights_np` 后旁路复算 batch metrics；明确 CSV/summary/checkpoint/status/metadata、scaler fit、optimizer/loss/epoch loop、reader/index 仍暂留正式入口，并说明 P7a/P7b smoke adapter 不能直接替换 full-scale streaming reader 或 torch training head |
| `docs/refactor/timefuse_evaluation_adapter_pressure_verification.md` | Stage 1 P8c TimeFuse evaluation adapter pressure verification | 记录 P8b `--verify-evaluation-adapter` 的 1-shard 小样本 pressure 验证；包含关闭/开启 verify 的显式命令、输出目录、CSV 字段顺序/行数/sample_key/指标/selected counts 对比结果，结论为开启旁路校验不改变正式输出 |
| `docs/refactor/timefuse_baseline_parity_review.md` | Stage 1 P8d TimeFuse baseline parity review | 审计当前 TimeFuse-style fusor baseline 与原版 TimeFuse 思路的 parity 边界；明确保留 linear logits、softmax expert weights、sample-level adaptive fusion、weighted prediction fusion 和 SmoothL1Loss，明确有意改造为单变量 17 维 feature、QuitoBench 五专家、Stage 1 packed prediction cache / streaming reader 和 Time evaluation adapter 口径；固定可声称和不可声称表述 |
| `docs/refactor/visual_router_entrypoint_adapter_insertion_audit.md` | Stage 1 P9a Visual Router entrypoint adapter insertion audit | 审计 `train_visual_router_online_streaming.py` 的最小 adapter 接入点；结论为 Visual Router 比 TimeFuse 更保守，P9b 优先只在 evaluation batch 旁路使用 `EvaluationInputAdapter` 或临时 `ExpertBatch` 做一致性校验，不改变正式 CSV/summary/metadata/status/checkpoint schema，不迁移 Quito history window、pseudo image、ViT provider、router head、loss 或 training loop |
| `docs/refactor/visual_router_evaluation_adapter_bypass.md` | Stage 1 P9b Visual Router evaluation adapter bypass | 记录 `train_visual_router_online_streaming.py` 新增默认关闭 `--verify-evaluation-adapter` 的边界和校验字段；该 flag 只在 test evaluation batch 内用 `EvaluationInputAdapter` 旁路复算 hard/raw-soft rows，并与正式 `soft_df` 逐样本比较，不写 adapter rows，不修改正式 CSV/summary/metadata/status/checkpoint schema |
| `docs/refactor/visual_router_evaluation_adapter_pressure_verification.md` | Stage 1 P9c Visual Router evaluation adapter pressure verification | 记录 P9b `--verify-evaluation-adapter` 的小规模正式入口 pressure 验证；使用仓库内 `2026-06-14_stage1_full_scale_dry_run_v2` dry-run 输入、CPU、`--local-files-only`、每 split 2 样本、1 epoch，对比关闭/开启 verify 后六个目标 CSV、streaming summary 核心表格、文件集合和 metadata/status/checkpoint schema，结论为除 run_dir 路径和生成时间外无正式口径漂移 |
| `docs/refactor/visual_router_expert_batch_evaluation_bridge.md` | Stage 1 P9d Visual Router ExpertBatch evaluation bridge | 记录 Visual Router evaluation bypass 从直接 `EvaluationInput` 收敛到 `ExpertBatch + fusion_weights` 的实现边界；明确 ExpertBatch 只包装当前 legacy SQLite batch arrays，不读取 manifest、不接 `PredictionCacheExpertProvider`、不改变正式输出 schema |
| `docs/refactor/visual_router_prediction_cache_provider_gap_audit.md` | Stage 1 P9e Visual Router PredictionCacheExpertProvider full-scale gap audit | 审计 `PredictionCacheExpertProvider` / `PredictionBatchReader` 与 Visual Router 正式 SQLite prediction path 的能力差距；明确 provider 已具备 `ExpertBatch` 包装能力，但正式入口仍需要 required sample_key 推导、manifest chunk scan、SQLite 子集索引、batch query、runtime index metadata 和 `fusion_huber_kl` expert error；结论为短期保留 SQLite path，只做 batch 后 ExpertBatch 旁路校验 |
| `docs/refactor/visual_router_training_expert_batch_bypass.md` | Stage 1 P9f Visual Router training ExpertBatch bypass | 记录默认关闭 `--verify-training-expert-batch` 的边界：只在 `fusion_huber_kl` training batch 内包装 legacy SQLite arrays 为 `ExpertBatch`，从 `ExpertBatch.y_pred/y_true` 显式复算 MAE/MSE `expert_errors` 并与 legacy 值比较；不替换 SQLite index、loss、optimizer、checkpoint 或正式输出 schema |
| `docs/refactor/shared_prediction_sqlite_backend_audit.md` | Stage 1 P10a shared prediction SQLite backend audit | 对比 Visual Router 与 TimeFuse-style fusor 当前 prediction/oracle SQLite path，明确 shared backend 只覆盖 manifest chunk scan、target sample_keys、SQLite 子集索引、batch fetch records、packed row index lineage、grouped mmap loading、index metadata 和 atomic replace；Visual/TimeFuse feature、loss、runtime artifact、launcher、`/data2` 和 oracle deployable feature 边界不进入 shared backend |
| `docs/refactor/prediction_sqlite_backend.md` | Stage 1 P10b Prediction SQLite backend helper | 记录 `time_router.io.prediction_sqlite_backend` 的 smoke-only API、SQLite schema、metadata、missing report、atomic replace / cleanup、fetch record 保序方式和 grouped packed loading 验收；明确不接 Visual Router / TimeFuse 正式入口、不改 provider/reader/adapter/loss/schema/launcher |
| `docs/refactor/stage1_canonical_sample_supervision_boundary.md` | Stage 1 P10d/P10e/P10f/P10g/P10h canonical SampleManifest 与 supervision boundary | 定义 Visual Router 与 TimeFuse-style fusor 可共用的 `SampleManifest`、`SplitStrategy` 和 `SupervisionProvider` 边界；明确 canonical manifest 字段、split 主索引、ExpertProvider 与 supervision 的区别、oracle/error 只用于训练监督/诊断/baseline/upper-bound、不进入 deployable FeatureProvider，并记录 P10e 最小协议骨架与 smoke、P10f Visual labels adapter smoke、P10g TimeFuse feature/oracle adapter smoke 和 P10h entrypoint migration plan canonical dataflow 对齐；不改正式入口、不新增正式 provider |
| `docs/refactor/visual_labels_sample_supervision_adapter.md` | Stage 1 P10f Visual labels sample/supervision adapter | 记录 `time_router.data.visual_labels_adapter` 的 smoke-only API、fixture 字段、历史 labels CSV 职责拆分、`SampleManifest` / `SupervisionBatch` 输出边界、oracle/error 不进入 FeatureProvider、真实 labels schema 后续需单独对齐，以及不接正式入口、不访问 `/data2`、不改正式 schema 的范围 |
| `docs/refactor/timefuse_sample_supervision_adapter.md` | Stage 1 P10g TimeFuse feature/oracle sample/supervision adapter | 记录 `time_router.data.timefuse_supervision_adapter` 的 smoke-only API、feature/oracle fixture 字段、历史 feature CSV 与 oracle SQLite/parquet 职责拆分、`SampleManifest` / `SupervisionBatch` 输出边界、17 维 feature 值不进入 manifest、oracle/error 不进入 FeatureProvider、真实 schema 后续需单独对齐，以及不接正式入口、不访问 `/data2`、不改正式 schema 的范围 |

### 1.2 根目录隐藏目录

| 路径 | 功能 | 备注 |
| --- | --- | --- |
| `.agents/` | agent 相关本地状态/配置目录 | 当前无需要长期维护的文件 |
| `.codex/` | Codex 相关本地状态/配置目录 | 当前无需要长期维护的文件 |
| `.git/` | 根仓库 Git 元数据目录 | 2026-06-13 已重新初始化为有效 Git 仓库；目录本身不进入版本控制 |

## 2. 实验管理层

这一层是本轮 QuitoBench baseline 复盘、统计基线评估、日志留痕和结果整理的主要协作区域。

```text
experiment_scripts/
├── run_default_baseline_finetune_eval.py
├── rescore_default_baseline_ckpts_by_mse.py
├── run_patchtst_crossformer_tuning.py
├── run_statistical_baseline_evaluate.py
├── summarize_five_model_three_config_results.py
└── audit_visual_router_phase1_oracle.py

experiment_logs/
├── README.md
├── 2026-06-10_*.md
├── 2026-06-11_*.md
├── 2026-06-12_*.md
├── 2026-06-13_*.md
└── run_outputs/

tests/
└── smoke/

/data2/syh/Time/
├── run_outputs/
└── cache_shards/
```

### 2.1 `experiment_scripts/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `experiment_scripts/run_default_baseline_finetune_eval.py` | 正式编排脚本 | 编排 QuitoBench 默认深度学习 baseline 的 finetune/evaluate；支持 checkpoint 选择指标、续训 checkpoint 和单任务过滤 |
| `experiment_scripts/rescore_default_baseline_ckpts_by_mse.py` | 复盘分析脚本 | 复盘已有深度学习 baseline checkpoint，按 validation MSE 重新选择 best checkpoint，并可对 MAE-best 与 MSE-best 不同的任务补跑 evaluate |
| `experiment_scripts/run_patchtst_crossformer_tuning.py` | 调参编排脚本 | PatchTST/CrossFormer tuning 编排；曾用于 4 卡顺序和单卡并发粗搜方案 |
| `experiment_scripts/run_statistical_baseline_evaluate.py` | 统计基线编排脚本 | 编排 ES/SNaive evaluate，输出到 `quito/outputs/statistical_baseline/` |
| `experiment_scripts/summarize_five_model_three_config_results.py` | 结果汇总脚本 | 汇总 DLinear、PatchTST、CrossFormer、ES、SNaive 在三组配置下的 overall mean MAE、TSF cell MAE、per-item 明细和 checkpoint lineage |
| `experiment_scripts/audit_visual_router_phase1_oracle.py` | 互补性审计脚本 | 基于五模型三配置 per-item 明细计算 best single expert、per-item oracle top-1、TSF cell 专家胜率和 regret，用于判断 Visual Router Phase 1 是否有足够上限 |
| `experiment_scripts/__pycache__/` | 可再生成缓存 | Python 解释器缓存，可删除后再生成，不作为正式产物 |

### 2.2 `scripts/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `scripts/run_stage1_canonical_small.py` | Stage 1 P12/P12b 薄 Python entrypoint | 使用 tiny `SampleManifest`、内联或显式 tiny expert fixture、临时或显式 tiny feature CSV、`TimeFuseFeatureCacheProvider`、`TimeFuseLinearSoftmaxHead`、`EvaluationInputAdapter` 和 Runtime artifact writer 跑通 small canonical dataflow；通过 `--output-root/--run-name` 显式创建 canonical `run_dir`，可选 `--sample-manifest` / `--expert-fixture` / `--feature-source` 读取 small fixture，并写出 `run_metadata.json`、`run_status.json`、`inputs/sample_manifest_ref.json`、`inputs/split_summary.json`、`evaluation/evaluation_summary.json` 和 `predictions/prediction_rows.csv`；不新增 Bash launcher、不访问 `/data2`、不启动训练、不迁移正式入口，Provider/Head/Evaluator 不接收 `run_dir` |
| `scripts/run_stage1_timefuse_small.py` | Stage 1 P15b TimeFuse-specific small canonical entrypoint | 默认使用 P13b real-derived small manifest/expert JSON 和 P13e 17 维 TimeFuse feature fixture，串联 `SampleManifest -> ExpertBatch -> TimeFuseFeatureCacheProvider / FeatureBatch -> TimeFuseLinearSoftmaxHead / RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer`，在 `--output-dir/--run-id` 下写 canonical run_dir；仅服务 small rehearsal，不访问 `/data2`、不启动训练/pressure/full-scale、不修改 generic small CLI、不迁移正式 TimeFuse fusor 入口 |
| `scripts/run_stage1_visual_small.py` | Stage 1 P15c Visual-specific small canonical entrypoint | 默认使用 P13b real-derived small manifest/expert JSON 和 P14b Visual mock history window fixture，串联 `SampleManifest -> VisualMockFeatureProvider / FeatureBatch -> ExpertBatch -> script-local smoke-only MLP adapter / RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer`，在 `--output-dir/--run-id` 下写 canonical run_dir；仅服务 Visual small rehearsal，不访问 `/data2`、不启动训练/pressure/full-scale、不读取真实 checkpoint、不启动 ViT embedding、不修改 generic 或 TimeFuse small CLI、不迁移正式 Visual Router 入口 |

### 2.3 `experiment_logs/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `experiment_logs/README.md` | 日志总览 | 追踪每篇实验日志的主题、状态、关键结果和下一步 |
| `experiment_logs/2026-06-10_*.md` | 正式实验日志 | 记录 cluster 信息整理、smoke test、baseline 编排、MSE-best 复盘、统计基线启动等步骤 |
| `experiment_logs/2026-06-11_*.md` | 正式实验日志 | 记录 2026-06-11 之后的结构文档维护、结果汇总或后续实验步骤 |
| `experiment_logs/2026-06-12_*.md` | 正式实验日志 | 记录视觉结构先验 Router/MoE 研究路线制定和 Visual Router Phase 1 oracle 审计 |
| `experiment_logs/2026-06-13_*.md` | 正式实验日志 | 记录 GitHub SSH key 配置、根仓库初始化、AGENTS 实验环境规范补充、Stage 1 结构特征/在线伪图像/ViT 成本估算、外部输出根目录接入、近期工作梳理、下一步计划更新和 HF ViT normalization 实现等 2026-06-13 后续步骤 |
| `experiment_logs/2026-06-14_*.md` | 正式实验日志 | 记录 Stage 1 ViT embedding 与 Visual Router MLP smoke 的实现、运行结果和后续计划 |
| `experiment_logs/2026-06-15_*.md` | 正式实验日志 | 记录 Stage 1 TimeFuse metadata baseline 口径审计、TimeFuse-style fusor baseline 实现与验证、续接复核、上下文 handoff 阈值协作规范补充等 2026-06-15 后续步骤 |
| `experiment_logs/run_outputs/` | 脚本运行输出根目录 | 保存每次编排脚本的 `status.json`、生成配置、运行日志、汇总 CSV 和部分 cluster/TSF cell 分析产物 |
| `/data2/syh/Time/run_outputs/` | 外部大盘运行输出根目录 | 用于后续大规模实验输出，避免继续占用 `/home`；仓库内通过 `EXTERNAL_OUTPUTS.md` 和实验日志记录索引 |
| `/data2/syh/Time/cache_shards/` | 外部临时 cache shard 根目录 | 用于抽样或短生命周期 shard；视觉路线默认不全量缓存伪图像或 ViT embedding，除非先证明缓存带来显著端到端加速 |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/` | Stage 1 `96_48_S` 正式 full-scale 全候选窗口输出根目录 | 保存正式 sample manifest、prediction cache launcher、merged cache、merged cache validation、oracle labels、TSF enrichment、TimeFuse feature cache launcher、`HANDOFF.md` 和后续 router/calibration；当前 `sample_manifest_full_scale/` 已完成，`prediction_cache_full_scale_launcher/merged_cache/` 已生成正式五专家 merged prediction cache，`prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/` 与 `tsf_enrichment_full_scale_2026-06-16/` 已生成并通过 join/覆盖验证，`timefuse_feature_cache_full_scale_launcher/` 为从 sample manifest 独立预计算 17 维 TimeFuse-derived 单变量元特征的正式输出，`launcher_compat_check/` 仅为 dry-run manifest 兼容性检查 |

### 2.3 `experiment_logs/run_outputs/`

| 路径模式 | 功能 | 引用口径 |
| --- | --- | --- |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_default_baseline_finetune_eval/` | 深度学习 baseline 编排运行目录 | 用于复核当次任务状态、生成配置和汇总 CSV |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_default_baseline_mse_best_rescore/` | MSE-best checkpoint 复盘运行目录 | 用于复核 MAE-best 与 MSE-best 差异和补评估结果 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_statistical_baseline_evaluate/` | ES/SNaive 统计基线 evaluate 运行目录 | 用于复核统计基线运行状态、生成配置和汇总结果 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_five_model_three_config_summary/` | 五模型三配置汇总目录 | 保存 overall mean MAE、TSF cell MAE、per-item 明细、checkpoint lineage 和 Markdown 汇总 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_phase1_oracle_audit/` | Visual Router Phase 1 专家互补性审计目录 | 保存配置级 oracle gap、TSF cell oracle gap、专家胜率、per-item oracle 选择明细和中文摘要；当前口径为 per-item top-1 oracle，不含窗口级 top-k 或 soft fusion |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_prediction_cache_pilot/` | Stage 1 prediction cache 小规模试运行目录 | 保存 window-level manifest、metadata、`y_true/y_pred` 数组、window oracle labels、TSF cell enrichment 结果和非视觉 router baseline 汇总；用于验证 cache schema、数组对齐和 Stage 1 baseline 口径，不作为全量正式结果 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_sample_manifest_96_48_s_1k/` | Stage 1 `96_48_S` 1k manifest-only 样本清单目录 | 保存 `sample_manifest.csv`、`sampling_metadata.json`、`sampling_summary.md` 和 `status.json`；不包含专家预测数组、embedding 或伪图像 tensor，用于中等规模 prediction cache / embedding / router 的共同样本来源 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/` | Stage 1 `96_48_S` 1k prediction cache launcher 目录 | 保存 `launcher.sh`、`launch_plan.md`、`status.json`、`pids/` 和 `shards/{model_name}/`；每个 shard 独立写 `main.log`、`status.json`、`manifest.csv` 和数组，合并前必须校验五专家完整性 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/` | Stage 1 `96_48_S` 1k 五专家合并 cache 目录 | 保存合并后的 `manifest.csv`、`window_oracle_labels*.csv`、`manifest_with_tsf_cell.csv`、`baseline_*.csv`、`summary.md` 和相关 `status.json`；是 1k router / calibration / baseline 的共同监督与对照来源 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_prediction_cache_full_scale_launcher/` | Stage 1 full-scale prediction cache launcher 目录 | 由 `launch_full_scale_prediction_cache.py` 生成，保存根级 `main.log`、`metadata.json`、`launcher.sh`、`launch_plan.md`、`status.json`、`pids/` 和 `shards/{model_name}/sample_shard_xxxx/`；默认 `packed_npy_v1`，DLinear/PatchTST/CrossFormer 绑定 GPU，ES/NaiveForecaster 走 CPU |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/sample_manifest_full_scale/` | Stage 1 `96_48_S` 正式 full-scale sample manifest 目录 | 保存 `sample_manifest_shard_index.csv`、`sample_shards/*.csv`、`sampling_metadata.json`、`sampling_summary.md`、`status.json` 和 `main.log`；当前全候选窗口 sample_count 为 `23,275,170`，64 个 shard，每 shard 约 `363,674` 到 `363,675` 个 sample_key，不包含专家预测、ViT embedding 或伪图像 tensor |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/` | Stage 1 `96_48_S` 正式 full-scale prediction cache launcher 目录 | 保存正式 `launcher.sh`、`launch_plan.md`、`status.json`、`metadata.json`、`pids/`、五专家 worker 日志和 `shards/{model_name}/sample_shard_XXXX_of_0064/`；五专家 shard 已 completed，completed shard 不应删除；`merged_cache/` 是正式合并结果，`record_count=116,375,850`、`sample_count=23,275,170`、`array_storage=packed_npy_v1`、`merge_strategy=packed_npy_v1_streaming_by_sample_shard`；`merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/` 保存完整性校验，`passed=true`；`oracle_labels_full_scale_2026-06-16/` 保存正式 `window_oracle_labels.parquet`、`window_oracle_summary.csv`、`status.json` 和 `main.log`，覆盖 `23,275,170` 个 sample_key、`46,550,340` 条 mae/mse label；首版同名 stopped 半成品已删除，completed 的 `_v2` 目录已重命名为该 canonical 路径，`_v2` 不再存在；`tsf_enrichment_full_scale_2026-06-16/` 保存正式 `sample_tsf_enrichment.parquet`、`tsf_missing_summary.csv`、`status.json` 和 `main.log`，关键 TSF 字段缺失全 0；`oracle_tsf_validation_2026-06-16/` 保存 join/覆盖验证，`status=passed`；`audits/` 保存运行中 completed shard 只读一致性抽检结果；`es_parallel_backfill_0016_0063/` 和 `es_accelerator_0010_0015_0048_0063/` 保存历史 ES 补跑 launcher、lane 日志和 PID |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/` | Stage 1 `96_48_S` 正式 full-scale TimeFuse-derived feature cache launcher 目录 | 由 `launch_timefuse_feature_cache_full_scale.py` 生成并启动，保存根级 `launcher.sh`、`launch_plan.md`、`status.json`、`metadata.json`、`main.log`、`pids/`、`lane_scripts/`、`logs/lane_*.log` 和 `shards/sample_shard_XXXX_of_0064/`；每个 shard 独立写 `feature_cache.csv`、`metadata.json`、`status.json`、`main.log`，特征只使用历史窗口 `x`，不读取未来 `y`、专家预测或 oracle label；当前按 8 lane CPU 并行运行，GPU 不参与 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_full_scale_dry_run/` | Stage 1 full-scale 框架 dry-run 目录 | 由 `run_full_scale_dry_run.py` 生成，保存根级 `main.log`、`metadata.json`、`status.json`、dry-run sample manifest、packed prediction cache shards、merged cache、oracle/TSF/baseline、streaming online router 和 calibration；用于验证可恢复流水线闭环，不作为正式指标 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/` | Stage 1 `96_48_S` 1k ViT embedding launcher 目录 | 保存 `launcher.sh`、`launch_plan.md`、`status.json` 和 `embedding_run/`；当前 online 路线下暂不启动，避免先长期缓存 ViT embedding `.npy`，不保存伪图像 tensor |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_structure_feature_pilot/` | Stage 1 结构特征 router 试运行目录 | 保存 TimeFuse-derived 单变量 `feature_cache.csv`、结构特征 router predictions/summary/metadata；该 hard-label LogisticRegression 结果已标注为 `legacy_deprecated`，仅作为轻量非视觉历史对照，不作为视觉主线正式结果 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_timefuse_fusor_baseline_pilot/` | Stage 1 TimeFuse-style fusor baseline 试运行目录 | 保存 `baseline_predictions.csv`、`baseline_summary.csv`、`baseline_comparison.csv`、`timefuse_fusor_predictions.csv`、`timefuse_fusor_raw_soft_fusion_predictions.csv`、`timefuse_fusor_summary.csv`、`timefuse_fusor_raw_soft_fusion_summary.csv`、`timefuse_fusor_selected_model_counts.csv`、`baseline_metadata.json` 和 `summary.md`；采用 TimeFuse-style 单层 `nn.Linear -> softmax -> weighted fusion -> SmoothL1Loss(beta=0.01)` 核心口径并适配当前单变量 QuitoBench 五专家设置，用于与 global/dataset/TSF-cell/oracle 和后续 visual router 公平同表比较 |
| `/data2/syh/Time/run_outputs/YYYY-MM-DD_stage1_timefuse_fusor_streaming_*/` | Stage 1 full-scale TimeFuse-style fusor streaming smoke/压力测试目录 | 由 `train_timefuse_fusor_streaming.py` 生成，保存 `metadata.json`、`status.json`、`summary.md`、`main.log`、`timefuse_fusor_predictions.csv`、`timefuse_fusor_summary.csv`、`timefuse_fusor_raw_soft_fusion_summary.csv`、`timefuse_fusor_selected_model_counts.csv`、`sample_predictions.csv`、`checkpoints/*.pt`、`indexes/*/*.sqlite` 和可选 `feature_subsets/*/feature_cache.csv`；当前只用于 1-2 个 feature shard 的 smoke/压力测试，不是正式 64-shard launcher |
| `experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_{off,on}/` | Stage 1 P8c TimeFuse evaluation adapter pressure 验证输出目录 | 由 `train_timefuse_fusor_streaming.py` 使用显式 `--output-dir` 生成；单个 `sample_shard_0008_of_0064`、每 split 8 行、CPU、1 epoch；用于比较关闭/开启 `--verify-evaluation-adapter` 后正式 CSV 是否漂移，结论为五个目标 CSV 字段顺序、行数、sample_key 顺序、selected_model、hard/raw-soft 指标和 selected counts 完全一致 |
| `experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_{off,on}/` | Stage 1 P9c Visual Router evaluation adapter pressure 验证输出目录 | 由 `train_visual_router_online_streaming.py` 使用显式 `--output-dir` 生成；复用仓库内 `2026-06-14_stage1_full_scale_dry_run_v2/merged_cache` labels/manifest 和本地 Quito config，CPU、`fp32`、`--local-files-only`、每 split 2 样本、1 epoch；用于比较关闭/开启 `--verify-evaluation-adapter` 后正式 artifact 是否漂移，结论为目标 CSV 字段顺序、行数、sample_key 顺序、selected_model、权重诊断、hard/raw-soft 指标、summary/comparison/selected counts 一致，开启 verify 不新增 adapter artifact |
| `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_cpu/` | Stage 1 `96_48_S` full-scale TimeFuse-style fusor CPU 半程停止留痕目录 | 由 `launch_timefuse_fusor_full_scale.py` 曾启动 CPU 版，保存 `preflight_report.json`、`metadata.json`、`status.json`、`launcher.log`、`main.log`、`pid.txt`、`pgid.txt`、脚本和半程 `indexes/*/*.sqlite`；用户要求正式公平比较至少训练时使用 GPU2/GPU3 后，该 CPU 进程已停止，`status=stopped_for_gpu_fairness_requirement`，不作为正式 baseline 结果引用 |
| `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` | Stage 1 `96_48_S` full-scale TimeFuse-style fusor baseline GPU2/3 正式后台运行目录 | 由 `launch_timefuse_fusor_full_scale.py --device cuda --cuda-visible-devices 2,3` 生成并启动，保存 `preflight_report.json`、`metadata.json`、`status.json`、`launcher.log`、`main.log`、`pid.txt`、`pgid.txt`、`command.sh`、`command_resume.sh`、`launcher.sh`、`stop.sh`、`resume.sh`，以及运行中/完成后的 `indexes/*/*.sqlite`、`checkpoints/*.pt`、`timefuse_fusor_predictions.csv`、summary、selected counts 和 sample predictions；`train_timefuse_fusor_streaming.py` 在 CUDA 多卡可见时使用 `nn.DataParallel`，checkpoint 保存未包裹模型 state_dict；2026-06-19 已优化为 shard-local index 复用、feature-only scaler、reader split 下推、packed npy batch-level grouped loading 和大块 CSV 后切 batch；当前状态为后台运行中，PID/PGID `1845436/1845436`，Python 子进程 `1845438`，`phase=train` |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_online_pseudo_image_pilot/` | Stage 1 在线伪图像化试运行目录 | 保存 `imageization_index.csv`、`latency_summary.csv`、`metadata.json`、`summary.md` 和少量 `debug_preview/*.png`；用于验证 Quito 历史窗口 x 到 3view/top3fold 视觉输入的在线 tensor-first 路径，不保存全量图像或 tensor cache，不作为 router 训练结果 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_vit_embedding_smoke/` | Stage 1 ViT embedding smoke 目录 | 保存 `embedding_manifest.csv`、`embedding_latency_summary.csv`、`embedding_metadata.json`、`embedding_summary.md` 和小规模 embedding npy；当前 2026-06-14 版本覆盖 120 个 `96_48_S metric=mae` sample_key，`google/vit-base-patch16-224` CLS embedding 维度为 768 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_visual_router_smoke/` | Stage 1 Visual Router smoke 目录 | 保存 `visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_soft_fusion_predictions.csv`、`visual_router_soft_fusion_summary.csv`、`visual_router_selected_model_counts.csv`、`visual_router_comparison.csv`、`visual_router_metadata.json` 和中文摘要；当前 2026-06-14 版本为 120 sample_key 小型 MLP smoke，不作为三 config 正式结论；新版默认 `fusion_huber_kl` 使用融合预测 SmoothL1 主损失与 KL soft oracle 辅助损失，旧版 `classification` 作为 baseline 模式保留 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_online_visual_router_smoke/` | Stage 1 Online Visual Router smoke 目录 | 保存 `online_embedding_manifest.csv`、`online_embedding_latency_summary.csv`、`visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_soft_fusion_predictions.csv`、`visual_router_soft_fusion_summary.csv`、`visual_router_selected_model_counts.csv`、`visual_router_comparison.csv`、`online_vs_offline_reference_comparison.csv`、`visual_router_online_metadata.json` 和中文摘要；online embedding 只在运行内内存暂存，不保存 ViT embedding `.npy`、`embeddings/` 目录或伪图像 tensor cache |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_online_visual_router_96_48_s_1k/` | Stage 1 `96_48_S` 1k online visual router 正式目录 | 保存 `online_embedding_manifest.csv`、`online_embedding_latency_summary.csv`、`visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_soft_fusion_predictions.csv`、`visual_router_soft_fusion_summary.csv`、`visual_router_selected_model_counts.csv`、`visual_router_comparison.csv`、`online_vs_offline_reference_comparison.csv`、`visual_router_online_metadata.json` 和中文摘要；采用本地 HF/ViT cache 与 `--local-files-only`，不保存长期 embedding `.npy`、`embeddings/` 目录或伪图像 tensor cache |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_online_visual_router_96_48_s_1k_local_only/` | Stage 1 `96_48_S` 1k online visual router 最终留痕目录 | 与上类同，作为显式记录 `local_files_only=True` 的自证版本；可与上一个目录并存，保留首次运行与最终自证运行的差异 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_online_visual_router_streaming_*/` | Stage 1 full-scale streaming online visual router 目录 | 保存 `online_embedding_manifest.csv`、`online_embedding_latency_summary.csv`、`visual_router_predictions.csv`、`visual_router_summary.csv`、`visual_router_selected_model_counts.csv`、`visual_router_comparison.csv`、标准 `visual_router_metadata.json`、`visual_router_online_metadata.json`、`status.json` 和中文摘要；ViT embedding 与伪图像 tensor 只在 batch 运行时存在，不保存 `.npy` 或 embedding shard |
| `/data2/syh/Time/run_outputs/YYYY-MM-DD_stage1_96_48_s_streaming_visual_router_eval_only_*/` | Stage 1 `96_48_S` full-scale streaming visual router eval-only 目录 | 使用已完成 train-only checkpoint 执行 `--resume-checkpoint ... --epochs 0`，保存 `launcher.sh`、`command.sh`、`launch_metadata.json`、`main.log`、`pid.txt`、`status.json`、SQLite prediction manifest index、`online_embedding_manifest.csv`、`online_embedding_latency_summary.csv`、`visual_router_predictions.csv`、`visual_router_summary.csv`、soft fusion 预测/summary、comparison 和 metadata；用于后续 full-scale calibration，不保存 ViT embedding `.npy` 或伪图像 tensor |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_soft_fusion_calibration_smoke/` | Stage 1 soft fusion calibration smoke 目录 | 保存 `soft_fusion_calibration_predictions.csv`、`soft_fusion_calibration_summary.csv`、`soft_fusion_calibration_selected_model_counts.csv`、`soft_fusion_calibration_comparison.csv`、`soft_fusion_calibration_metadata.json` 和中文摘要；用于在不改变 router 输入约束的前提下，对已有 test 权重做 temperature scaling、top-k 截断重归一化、raw soft、top1 hard、top2/top3 fusion 对比 |
| `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_soft_fusion_calibration_96_48_s_1k/` | Stage 1 `96_48_S` 1k soft fusion calibration 正式目录 | 保存 `soft_fusion_calibration_predictions.csv`、`soft_fusion_calibration_summary.csv`、`soft_fusion_calibration_selected_model_counts.csv`、`soft_fusion_calibration_comparison.csv`、`soft_fusion_calibration_metadata.json`、`soft_fusion_calibration_summary.md` 以及最终 summary 表；用于比较 raw soft、top1 hard、top2/top3 fusion 和 temperature sweep 的正式结果 |
| `experiment_logs/run_outputs/*/generated_configs/` | 派生配置 | 保存脚本实际写出的配置，便于复现实验参数 |
| `experiment_logs/run_outputs/*/logs/` | 子任务日志 | 保存单任务 stdout/stderr 或运行日志 |
| `experiment_logs/run_outputs/*/cluster_analysis/` | 分组分析结果 | 保存 cluster/TSF cell 等分层评估产物 |
| `experiment_logs/run_outputs/*.log` | 后台运行日志 | 早期 nohup/setsid/single-lane 后台运行日志 |

## 2.4 `visual_router_experiments/`

`visual_router_experiments/` 是视觉结构先验路由正式实验代码根目录。后续 Visual Router / Visual-Conditioned MoE 相关的可复用代码、正式实验入口和阶段性脚本优先放在这里；大规模输出仍写入 `experiment_logs/run_outputs/`。

```text
visual_router_experiments/
├── README.md
├── common/
├── stage0_oracle_audit/
├── stage1_vali_test_router/
└── stage2_heldout_cell/
```

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `visual_router_experiments/README.md` | 正式实验代码目录说明 | 记录按 stage 建二级目录、跨阶段公共代码和输出目录约定 |
| `visual_router_experiments/common/` | 跨阶段公共代码目录 | 保存 prediction cache schema、item-channel-window key、指标、伪图像张量构造、运行内视觉 embedding 工具和通用评估工具；当前已有 `prediction_cache_schema.py`、`prediction_array_io.py`、`pseudo_imageization.py` 和 `vit_embedding_utils.py`；P10c 后 `prediction_array_io.py` 是旧路径兼容层，re-export `time_router.io.prediction_array_io` 中的 `per_sample_npy` 与 `packed_npy_v1` prediction arrays 读取 API；`pseudo_imageization.py` 已支持 `hf_vit_0_5` 与 `torchvision_imagenet` encoder normalization，并新增固定候选周期桶与按周期分桶 fold 路径，减少在线伪图像化中的逐样本 CPU/GPU 同步；`vit_embedding_utils.py` 为 online 主线提供不落盘的 ViT 输入/输出处理工具 |
| `visual_router_experiments/stage0_oracle_audit/` | 上限审计阶段目录 | 承接专家互补性和 oracle 上限审计；当前 README 索引已有审计脚本与输出，后续扩展专家池或 window-level oracle 可在此补充正式脚本 |
| `visual_router_experiments/stage1_vali_test_router/` | Stage 1 主实验目录 | 保存 vali 训练 router、test 测试 router 的 prediction cache、oracle labels、TSF enrichment、TimeFuse feature cache、embedding、训练、评估和汇总脚本；`README.md` 现作为当前主线导航页，明确 visual router full-scale 正式入口、中小规模复现入口、baseline 支线入口、共享库、pilot 边界和下一步；`stage1_visual_router_mainline.md` 只记录视觉路由主线、`96_48_S` 正确路线、废弃路线和扩 config 标准步骤，明确 TimeFuse-style fusor 是 baseline 支线；`stage1_history_results.md` 保存从 README 拆出的 120 sample smoke、1k、dry-run 和 full-scale 长跑历史结果索引；当前已有 `prediction_cache_design.md`、`feature_and_rl_extension_notes.md`、`stage1_cache_contract.md`、`stage1_visual_router_mainline.md`、`stage1_protocol_and_plan.md`、`stage1_history_results.md`、`stage1_timefuse_fusor_streaming_reader_design.md`、`build_stage1_sample_manifest.py`、`build_full_scale_sample_manifest.py`、`build_prediction_cache_from_manifest.py`、`merge_prediction_cache_shards.py`、`launch_full_scale_prediction_cache.py`、`build_full_scale_window_oracle_labels.py`、`build_full_scale_tsf_enrichment.py`、`validate_full_scale_oracle_tsf_outputs.py`、`build_timefuse_feature_cache_from_manifest.py`、`launch_timefuse_feature_cache_full_scale.py`、`stage1_timefuse_fusor_streaming_reader.py`、`train_timefuse_fusor_streaming.py`、`launch_timefuse_fusor_full_scale.py`、`run_full_scale_dry_run.py`、`evaluate_router_baselines.py`、`fusion_utils.py`、`train_visual_router.py`、`train_visual_router_online.py`、`train_visual_router_online_streaming.py`、`evaluate_soft_fusion_calibration.py`、`pilot/` 和 package 初始化文件；`train_visual_router_online_streaming.py` 当前新增默认关闭的 `--verify-evaluation-adapter` 和 `--verify-training-expert-batch`，前者只在 test evaluation batch 内用 legacy SQLite batch arrays 构造 `ExpertBatch + fusion_weights` 并通过 `EvaluationInputAdapter` 旁路一致性校验，后者只在 `fusion_huber_kl` training batch 内从 `ExpertBatch.y_pred/y_true` 复算 `expert_errors` 并对照 legacy 值；两者均不改变正式输出 schema，不代表 `PredictionCacheExpertProvider` 已正式接入；baseline evaluator 现可同时输出统计 baseline、TimeFuse-style fusor hard/raw-soft、oracle 和统一 comparison；visual full-scale 路线使用 packed prediction cache 与 streaming online router，不落盘 ViT embedding `.npy` 或伪图像 tensor；TimeFuse-style fusor full-scale 读取层已支持 feature shard、oracle parquet、五专家 shard/merged prediction manifest 的 shard-local SQLite + batch reader，streaming train/eval 入口已完成 1-shard smoke、checkpoint eval-only 和 2-shard小切片压力测试，并已支持 CUDA 多卡 `DataParallel`；2026-06-19 起 fusor reader/train 支持 index 复用、feature-only scaler、split 下推、packed npy batch-level grouped loading 和大块 CSV 后切 batch；正式 64-shard GPU2/3 后台 launcher 独立追踪于 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` |
| `visual_router_experiments/stage1_vali_test_router/pilot/` | Stage 1 pilot 脚本目录 | 保存 `build_prediction_cache_pilot.py`、`build_vit_embeddings_pilot.py`、`build_online_pseudo_image_pilot.py`、`build_structure_feature_cache_pilot.py`、`train_structure_router_pilot.py`、`compute_window_oracle_from_cache.py`、`enrich_cache_with_tsf_cell.py`、`launch_96_48_s_1k_prediction_cache_pilot.py`、`launch_96_48_s_1k_vit_embedding_pilot.py` 等小规模验证、离线 embedding 历史对照、过渡性 launcher 和固定规模资源编排脚本；用于打通 cache/oracle/enrichment/feature/router 流程或复现 1k smoke 编排，不作为通用正式实验入口 |
| `visual_router_experiments/stage2_heldout_cell/` | Stage 2 泛化实验目录 | 后续保存 7-cell 训练、held-out cell 测试的 zero-shot 泛化实验脚本 |

### 2.5 `time_router/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `time_router/__init__.py` | 共享 package 入口 | Stage 1 后续重构使用的最小共享 package 骨架；当前承载低风险公共 reader、最小 evaluation helper、protocol types 和 P6a expert adapter，不代表正式训练入口已迁移 |
| `time_router/data/__init__.py` | 共享数据子包入口 | 导出 `OracleTsfBatch`、`OracleTsfReader`、P10f `visual_labels_to_sample_manifest` / `visual_labels_to_supervision_batch` 和 P10g `timefuse_features_to_sample_manifest` / `timefuse_oracle_to_supervision_batch`；入口只聚合稳定 smoke/data API，不读取配置或创建 run_dir |
| `time_router/data/oracle_tsf_reader.py` | Stage 1 共享 oracle/TSF reader | 按 sample_key 批量读取 window-level oracle labels 与 TSF enrichment / TSF-cell metadata；支持显式 sample_key 保序、CSV chunk 过滤、Parquet dataset 过滤、`missing_policy=error/report`、冲突重复/缺失检查和 oracle/TSF 一对一 join；只做读取、校验和 join，不提供训练策略，不把 oracle/TSF 作为可部署 test-time 动态特征 |
| `time_router/data/visual_labels_adapter.py` | Stage 1 P10f Visual labels sample/supervision adapter | 支持小型 `pd.DataFrame` 或 CSV 路径输入，构造 canonical `SampleManifest` 和 `SupervisionBatch`；manifest 覆盖 sample_key、split、config、dataset、item/channel/window、可选 seq/pred length 和轻量 lineage；supervision 按显式 `sample_keys + model_columns + metric` 保序输出 oracle_model、oracle_value 和 `[sample, expert]` per-model errors；只用于 smoke，不接正式入口、不访问 `/data2`、不改正式 schema |
| `time_router/data/timefuse_supervision_adapter.py` | Stage 1 P10g TimeFuse feature/oracle sample/supervision adapter | 支持小型 `pd.DataFrame` 或 CSV 路径输入，分别从 feature source 构造 canonical `SampleManifest`、从 oracle/supervision source 构造 `SupervisionBatch`；manifest 覆盖 sample_key、split、config、dataset、item/channel/window、可选 seq/pred length 和 feature lineage，明确 17 维 feature 值不进入 extra；supervision 按显式 `sample_keys + model_columns + metric` 保序输出 oracle_model、oracle_value 和 `[sample, expert]` per-model errors；只用于 smoke，不接正式入口、不访问 `/data2`、不改正式 schema |
| `time_router/evaluation/__init__.py` | 共享评估子包入口 | 导出 P3a/P3b/P3c/P3d 最小 fusion/metrics/router weight diagnostics/summary/rows helper，以及 P6b/P6c canonical `EvaluationInputAdapter` 和兼容 `FusionEvaluator`；当前不包含 calibration、正式报告 schema 或训练入口迁移 |
| `time_router/evaluation/evaluation_input_adapter.py` | Stage 1 P6b/P6c canonical EvaluationInput adapter | 纯内存实现 `EvaluationInputAdapter` 和 `EvaluationInputAdapterResult`，将 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 包装为 `EvaluationInput`；`evaluate_input()` 是 adapter 层唯一调用 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 和 `build_per_sample_fusion_rows` 的实现点；保持 sample/model 顺序，复用 `ExpertBatch.y_pred/y_true`，不读取 manifest、prediction cache、packed npy、oracle/TSF，不访问 `/data2`，不创建 run_dir，不写 status/metadata/CSV/JSON/Parquet |
| `time_router/evaluation/fusion_evaluator.py` | Stage 1 P6c FusionEvaluator compat wrapper | 纯内存保留 `FusionEvaluator` 和 `FusionEvaluationResult` 旧 public API；从 `ExpertBatch + RouterOutput` 或显式 `EvaluationInput` 进入后委托 `EvaluationInputAdapter` 构造/复算，不再直接调用 metrics/summary/rows helper；用于旧 smoke 和下游兼容，不作为新增逻辑入口 |
| `time_router/evaluation/metrics.py` | Stage 1 P3a/P3b 最小 fusion/metrics/diagnostics helper | 纯 numpy 实现 `compute_mae`、`compute_mse`、`validate_fusion_inputs`、`hard_top1_fusion`、`raw_soft_fusion`、`compute_selected_counts`、`compute_weight_entropy` 和 `compute_max_weight`；函数输入显式使用 `y_pred`、`y_true`、`weights`、`selected_indices`、`model_columns`，用于 golden smoke 复算 hard top-1/raw soft 指标和 router weight 诊断，不读取 manifest/oracle/TSF/正式输出目录，不引入 torch/sklearn 训练依赖 |
| `time_router/evaluation/prediction_rows.py` | Stage 1 P3d 最小 per-sample evaluation rows helper | 纯 numpy / Python 标准库实现 `build_per_sample_fusion_rows`，只消费显式传入的 `sample_keys`、`FusionMetricsResult`、`y_true`、`weights` 和 `model_columns`，输出当前 batch 的 sample_key、hard top-1 选择、逐样本 hard/raw-soft MAE/MSE、max weight 和 weight entropy；不读取 manifest、prediction cache、oracle/TSF 或正式输出目录，不写 CSV/JSON/Parquet，不实现 calibration、oracle regret 或正式 output schema 迁移 |
| `time_router/evaluation/summary.py` | Stage 1 P3c 最小 evaluation summary helper | 纯 numpy / Python 标准库实现 `build_fusion_summary`，只消费显式传入的 `FusionMetricsResult`、`weights` 和 `model_columns`，汇总 hard/raw-soft MAE/MSE、selected counts、mean entropy、mean max weight、样本数、专家数和专家顺序；不读取 manifest、prediction cache、oracle/TSF 或正式输出目录，不实现 calibration、oracle regret、comparison 或正式 output schema 迁移 |
| `time_router/experts/__init__.py` | 共享专家预测适配器子包入口 | 导出 P6a 最小 `PredictionCacheExpertProvider`；当前只用于 smoke，不接正式 Visual Router / TimeFuse fusor 入口 |
| `time_router/experts/prediction_cache.py` | Stage 1 P6a/P13d PredictionCacheExpertProvider | 复用 `time_router.io.PredictionBatchReader`，通过显式 `load_batch(sample_keys, verify_metrics=True)` 输出 `time_router.protocols.ExpertBatch`；保持 sample_key 顺序、固定五专家顺序或调用方显式 model_columns、共享 y_true 校验、packed row index lineage 和 verify_metrics 校验能力；`extra` 只记录 provider name、array_storage、轻量 reader metadata 和 P13d schema 校验开关；P13d 新增 `validate_manifest_schema=False` 仅用于非 canonical sample_key smoke bridge，默认仍严格校验正式 prediction cache schema；不读取 oracle/TSF、不生成 feature、不计算 loss、不做 evaluation、不访问 `/data2`、不创建 run_dir、不写 status/metadata、不改训练入口 |
| `time_router/features/__init__.py` | 共享特征适配器子包入口 | 导出 P7a 最小 `TimeFuseFeatureCacheProvider`、P14b smoke-only `VisualMockFeatureProvider` / `DeterministicVisualEncoderStub`、P16c `VisualPrecomputedFeatureProvider`、P16d `LoadedFeatureScaler` 和 P16f visual chain protocol skeleton 类型；当前只用于 smoke 和 provider/transform/protocol 边界验证，不接正式 TimeFuse fusor / Visual Router 入口 |
| `time_router/features/timefuse_cache.py` | Stage 1 P7a TimeFuseFeatureCacheProvider | 读取调用方显式传入的小规模 feature CSV，并通过显式 `load_batch(sample_keys)` 输出 `time_router.protocols.FeatureBatch`；保持 sample_key 顺序，features 当前为 `numpy.float32` array，`feature_schema` 记录 schema 名、feature columns、feature_dim 和 source；不读取 prediction cache、oracle/TSF、`y_true` 或 expert error，不做 scaler fit，不访问 `/data2`，不创建 run_dir，不写 status/metadata/CSV/JSON/Parquet，不改正式入口 |
| `time_router/features/visual_mock.py` | Stage 1 P14b VisualMockFeatureProvider | 提供 smoke-only `DeterministicVisualEncoderStub` 和 `VisualMockFeatureProvider`；provider 只消费调用方注入的内存 `sample_key -> history_window_x` 映射，并通过显式 `load_batch(sample_keys)` 输出 `FeatureBatch`；保持 sample_key 顺序，features 为 `[sample, 8]` `numpy.float32` array，schema 记录 visual mock、history_source、pseudo_image/mock_not_materialized 和 encoder_stub 口径；不读取 prediction cache、oracle/error、`y_true`、run_dir、status、checkpoint，不加载真实 ViT，不访问 `/data2`，不改正式入口 |
| `time_router/features/visual_precomputed.py` | Stage 1 P16c VisualPrecomputedFeatureProvider | 读取调用方显式传入的 precomputed/head-ready visual embedding CSV，并通过 `load_batch(sample_keys)` 输出 canonical `FeatureBatch`；自动识别 `feature_` 前缀列，校验 sample_key 非空唯一、feature column 非空、特征数值有限，输出 features 为 `[sample, feature_dim]` `numpy.float32` array，schema 记录 `head_ready=True`、`precomputed=True`、`loads_real_vit=False`、`handles_scaler=False`；不接真实 ViT、不构造 pseudo image、不处理 scaler、不读取 checkpoint、不接收 `run_dir`、不读取 prediction/oracle/expert error、不访问 `/data2` |
| `time_router/features/visual_scaler.py` | Stage 1 P16d LoadedFeatureScaler | 使用已加载 scaler state 对 raw/pre-head `FeatureBatch.features` 执行 `(raw - mean) / scale`，输出新的 head-ready `float32 FeatureBatch`；校验 mean/scale 有限、长度匹配、scale 非 0、输入二维有限、feature_columns 对齐和 sample_key 唯一；不执行 scaler fit，不读取 checkpoint，不接真实 ViT，不接收 `run_dir`，不读取 prediction/oracle/expert error，不访问 `/data2` |
| `time_router/features/visual_chain.py` | Stage 1 P16f Visual feature chain protocol skeleton | 定义 `RawWindowBatch`、`PreImageBatch`、`VisualInputBatch`、`VisualEmbeddingBatch`、`RawWindowProvider`、`PreImageTransform`、`PseudoImageTransformer`、`ResizePolicy`、`VisualEncoderProvider`、`PoolingStrategy`、`FeatureTransform` 和 `VisualFeatureChainSpec`；只表达 ordered sample_keys、payload 和轻量 metadata lineage 的输入输出契约，不实现真实算法，不导入 torch/transformers/sklearn，不绑定 cache/checkpoint/run_dir |
| `time_router/io/__init__.py` | 共享 IO 子包入口 | 导出 `DEFAULT_MODEL_COLUMNS`、`PredictionBatch`、`PredictionBatchReader`、P10c prediction array IO public API（`PACKED_NPY_STORAGE` / `PER_SAMPLE_NPY_STORAGE` / `resolve_cache_array_path` / `load_prediction_array` / `load_prediction_arrays_grouped`）、P10b `PreparedPredictionSQLiteBackend` / `PredictionSQLiteBackendMetadata` / `build_prediction_sqlite_backend` / `load_prediction_sqlite_backend` / `records_to_ordered_rows`、`atomic_write_json`、`build_status_payload`、`write_status_json`、`find_repo_root`、`resolve_under_root`、`resolve_status_path`、`resolve_metadata_path`、`build_run_metadata` 和 `write_run_metadata`；P4d 起明确该入口只聚合稳定 public API，不在导入时读取配置、创建输出目录或执行训练相关副作用 |
| `time_router/io/prediction_array_io.py` | Stage 1 P10c canonical prediction array IO | canonical 提供 `packed_npy_v1` 与 legacy `per_sample_npy` prediction arrays 的路径解析、单样本读取和 grouped batch 读取；`load_prediction_arrays_grouped(...)` 对同一路径 packed npy 复用 mmap 并按 row index 切片；只读取调用方传入 record 指向的数组，不写 cache、不创建 run_dir、不接正式入口；旧 `visual_router_experiments/common/prediction_array_io.py` 仅作为兼容 re-export |
| `time_router/io/prediction_cache_reader.py` | Stage 1 共享 prediction batch reader | 从 `merged_cache/manifest.csv` 或 fixture root 读取专家 `y_pred` 和共享 `y_true`；支持 `packed_npy_v1`、`per_sample_npy`、固定默认五专家顺序或调用方显式 model_columns、共享 y_true 校验、row index 元数据和 manifest MAE/MSE 复算；默认执行 canonical prediction manifest schema 校验，P13d 新增 `validate_manifest_schema=False` 仅用于 P13b 非 canonical sample_key 的 smoke-only bridge 并仍保留 `(sample_key, model_name)` 唯一与专家集合完整性校验；尚未迁移正式 Visual Router / TimeFuse fusor 入口 |
| `time_router/io/prediction_sqlite_backend.py` | Stage 1 P10b shared prediction SQLite backend helper | 提供 smoke-only `build_prediction_sqlite_backend(...)`、`PreparedPredictionSQLiteBackend.fetch_records(...)`、`load_prediction_sqlite_backend(...)` 和 `records_to_ordered_rows(...)`；按调用方传入的 `target_sample_keys + model_columns` 分块扫描 manifest 并构建 `(sample_key, model_name)` SQLite 子集索引，记录 array path、MAE/MSE、`array_storage`、packed row index、metadata 和 missing report；构建使用临时文件和原子替换，失败清理临时文件；不读取 feature/oracle/loss、不创建 run_dir、不写正式 status/metadata/CSV/checkpoint、不接正式入口 |
| `time_router/io/json_utils.py` | Stage 1 P4a 最小 JSON/status writer | 纯标准库实现 `atomic_write_json`、`build_status_payload` 和 `write_status_json`；只写调用方显式传入的 path，使用同目录临时文件、`flush + fsync` 和 `os.replace` 原子替换，默认 UTF-8 / `ensure_ascii=False`；不读取训练状态，不实现 path resolver、config system 或 logging framework |
| `time_router/io/path_resolver.py` | Stage 1 P4b 最小 path resolver | 纯标准库实现 `find_repo_root`、`resolve_under_root`、`resolve_status_path` 和 `resolve_metadata_path`；只做 repo root 查找、root 内安全拼接和 status/metadata path 计算，不创建目录、不写文件、不读取训练配置、不访问 `/data2` 或 full-scale 输出目录 |
| `time_router/io/run_metadata.py` | Stage 1 P4c 最小 run metadata payload builder | 纯标准库实现 `build_run_metadata` 和 `write_run_metadata`；构造至少包含 `stage`、`created_at_utc`、`inputs`、`outputs` 的 metadata-like payload，支持 Path 转字符串和 tempfile writer；不自动调用 git、不读取命令行/训练配置、不改变既有正式 metadata schema |
| `time_router/models/__init__.py` | 共享模型/head 适配器子包入口 | 导出 P7b 最小 `TimeFuseLinearSoftmaxHead` 和 P16a `LoadedTorchMLPRouterHeadAdapter`；当前用于 Stage 1 canonical smoke/adapter 边界验证，不接正式 TimeFuse fusor / Visual Router 训练入口 |
| `time_router/models/timefuse_linear.py` | Stage 1 P7b TimeFuseLinearSoftmaxHead | 纯 numpy 实现固定线性层和 stable softmax；通过 `predict(feature_batch, model_columns)` 输出 `time_router.protocols.RouterOutput`，保持 sample_key/model_columns 顺序，logits/weights 专家维度与 model_columns 对齐；不训练、不计算 loss、不建 optimizer、不保存 checkpoint、不读取 prediction cache/oracle/TSF/feature CSV、不访问 `/data2`、不创建 run_dir、不写 status/metadata/CSV/JSON/Parquet、不改正式入口 |
| `time_router/models/visual_mlp_adapter.py` | Stage 1 P16a Visual MLP RouterHead adapter | 提供 `LoadedTorchMLPRouterHeadAdapter`，只包装 Runtime 已加载的 `torch.nn.Module`，消费 head-ready `float32 FeatureBatch.features` 和显式 `model_columns`，在 `torch.inference_mode()` 下得到二维 logits 并 softmax 为 weights，输出 `RouterOutput`；不读取 checkpoint、不调用 `torch.load`、不处理 scaler、不启动 ViT、不访问 `/data2`、不接收 `run_dir`、不导入 legacy `VisualMLPRouter` 或正式训练入口 |
| `time_router/protocols/__init__.py` | Stage 1 P5c/P10e protocol 子包入口 | 导出 `SplitSpec`、`SampleManifestRow`、`SampleManifest`、`SupervisionBatch`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput` 和 `ExperimentProtocolSpec`；只聚合 lightweight contract public API，不实例化 provider、不读取配置或路径、不创建 run_dir |
| `time_router/protocols/types.py` | Stage 1 P5c/P10e 最小 protocol dataclass 类型骨架 | 纯标准库 `dataclass + typing.Any` 定义 split、canonical sample manifest row/manifest、supervision batch、expert batch、feature batch、router output、evaluation input 和 experiment protocol spec；array/tensor 字段不绑定 numpy/torch，P10e 仅在 `SupervisionBatch.validate_shapes()` 中做最小维度对齐校验；`SampleManifest` 提供 sample_key 唯一性、ordered sample_keys 和 split_counts helper；`extra`、`branch_specific` 和 `feature_schema` 使用 `field(default_factory=dict)`，不包含 `run_dir` |
| `time_router/runtime/__init__.py` | Stage 1 P11c Runtime artifact writer 子包入口 | 导出 `CANONICAL_RUN_SUBDIRS`、`create_run_dir`、`write_json_atomic`、`write_run_metadata`、`write_run_status`、`write_sample_manifest_ref`、`write_split_summary`、`write_evaluation_summary` 和 `write_prediction_rows_csv`；入口只聚合 Runtime artifact 写出 API，不读取配置、不访问 `/data2`、不启动训练 |
| `time_router/runtime/artifact_writer.py` | Stage 1 P11c 最小 Runtime artifact writer | 创建 canonical `run_dir` 子目录，写出 `run_metadata.json`、`run_status.json`、`inputs/sample_manifest_ref.json`、`inputs/split_summary.json`、`evaluation/evaluation_summary.json` 和 `predictions/prediction_rows.csv`；只校验最小必需字段并复用 JSON 原子写入，不实现 checkpoint/resume、provider/head/evaluator、launcher 或复杂 Runtime framework，不迁移 legacy entrypoint |

### 2.6 `tests/smoke/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `tests/smoke/stage1_golden_smoke.py` | Stage 1 只读 golden smoke | 默认读取 `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/` 的 4 sample packed fixture；当前通过 `time_router.io.PredictionBatchReader` 组装 `y_pred/y_true`，并通过 `time_router.evaluation.metrics` / `time_router.evaluation.summary` / `time_router.evaluation.prediction_rows` 复算 hard top-1、raw soft fusion、P3b router weight diagnostics、P3c minimal summary 和 P3d per-sample rows；锁定 sample_key 顺序、五专家顺序、shape、hard top-1、raw soft fusion MAE/MSE、selected counts、entropy/max_weight shape、summary 字段数值、per-sample row 字段与数值和 `packed_npy_v1` row index 读取一致性；用于后续公共 reader/metrics/入口迁移前后等价验证，不训练、不写正式输出 |
| `tests/smoke/stage1_oracle_tsf_smoke.py` | Stage 1 oracle/TSF reader 只读 smoke | 默认读取同一 dry-run fixture 的 `window_oracle_labels_with_tsf_cell.csv` 和 `manifest_with_tsf_cell.csv`；验证 `OracleTsfReader` 的 `allow_full_scan` 默认禁止无 sample_key 全扫描、显式 sample_key 保序、oracle label、TSF metadata、oracle/TSF join、`missing_policy=error` 缺失报错、`missing_policy=report` 缺失报告和冲突重复 sample_key 报错；不训练、不生成 oracle/TSF、不写正式输出 |
| `tests/smoke/stage1_json_utils_smoke.py` | Stage 1 P4a JSON utils 临时目录 smoke | 在 `tempfile.TemporaryDirectory` 下验证 `write_status_json` / `atomic_write_json` 的文件存在、JSON 可读、中文 message 不被 ASCII 转义、第二次写入覆盖旧内容、nested parent directory 自动创建和 `extra` 类型检查；不读取训练状态、不写正式输出目录 |
| `tests/smoke/stage1_path_resolver_smoke.py` | Stage 1 P4b path resolver 临时目录 smoke | 验证从 `tests/smoke` 定位仓库根、root 下定位 `WORKSPACE_STRUCTURE.md`、tempfile root 下正常路径解析、`..` 逃逸 root 报错、`must_exist=True` 不存在报错，以及 status/metadata helper 只返回路径不创建目录或文件；不访问 `/data2` 或 full-scale 输出目录 |
| `tests/smoke/stage1_run_metadata_smoke.py` | Stage 1 P4c run metadata 临时目录 smoke | 验证 `build_run_metadata` 的基础字段、timezone-aware UTC 时间、Path 转字符串、`stage` 非空校验、`inputs/outputs/extra` 类型校验，以及 `write_run_metadata` 只在 tempfile 下写入 JSON 且可读；不访问 `/data2` 或 full-scale 输出目录 |
| `tests/smoke/stage1_protocol_types_smoke.py` | Stage 1 P5c protocol dataclass 纯内存 smoke | 从 `time_router.protocols` public API 构造全部 6 个 dataclass，验证 sample_keys/model_columns/train_splits/eval_splits tuple 保序、`extra`/`branch_specific`/`feature_schema` default_factory 独立、RouterOutput/EvaluationInput 的 logits/weights 可选组合，以及 object/list 字段原样保存且不访问 `.shape`；不创建文件、不访问 `/data2` 或正式输出目录 |
| `tests/smoke/stage1_sample_supervision_protocol_smoke.py` | Stage 1 P10e SampleManifest / SupervisionBatch protocol smoke | 纯内存构造 4 行 vali/test `SampleManifestRow`，验证 `SampleManifest.validate_unique_sample_keys()`、按 split 返回 ordered sample_keys 和 `split_counts()`；构造 vali/test 两个 `SupervisionBatch`，使用 5 个专家列和小型 numpy `per_model_errors` 校验 shape、sample/model 顺序、metric 和 oracle 输出；覆盖重复 sample_key、专家维 shape mismatch 和 oracle shape mismatch 报错；不访问 `/data2`、不运行正式入口、不写正式输出 |
| `tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py` | Stage 1 P10f Visual labels adapter smoke | 纯内存构造 4 行 vali/test labels fixture，包含五专家 `mae` error 列；通过 `visual_labels_to_sample_manifest(...)` 校验 sample_key 唯一、split 保序、`split_counts()` 和 lineage extra；通过 `visual_labels_to_supervision_batch(...)` 分别构造 vali/test `SupervisionBatch`，校验 oracle_model、oracle_value 和 `[sample, expert]` shape；覆盖 CSV 入口、缺失专家列、重复 sample_key 和未知 split 报错；不访问 `/data2`、不运行正式入口、不写正式输出 |
| `tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py` | Stage 1 P10g TimeFuse feature/oracle adapter smoke | 纯内存构造 4 行 vali/test feature fixture 和对应 oracle fixture，feature fixture 包含 17 维 TimeFuse feature 列；通过 `timefuse_features_to_sample_manifest(...)` 校验 sample_key 唯一、split 保序、`split_counts()`、feature lineage extra 和 17 维 feature 值不进入 manifest；通过 `timefuse_oracle_to_supervision_batch(...)` 分别构造 vali/test `SupervisionBatch`，校验 oracle_model、oracle_value 和 `[sample, expert]` shape；覆盖 CSV 入口、缺失 oracle 专家列、feature 重复 sample_key、oracle 缺失 sample_key 和未知 split 报错；不访问 `/data2`、不运行正式入口、不写正式输出 |
| `tests/smoke/stage1_prediction_cache_expert_provider_smoke.py` | Stage 1 P6a PredictionCacheExpertProvider 只读 smoke | 默认读取同一 4 sample packed golden fixture，构造 `PredictionCacheExpertProvider` 并显式传入 golden sample_keys；验证 `ExpertBatch` 类型、sample_keys/model_columns tuple 保序、`y_pred/y_true` shape、packed row index metadata、provider extra 轻量 metadata，并通过 `time_router.evaluation` public API 复算 hard top-1 与 raw soft fusion golden 指标；不创建正式输出目录、不访问 `/data2`、不读取 oracle/TSF |
| `tests/smoke/stage1_prediction_sqlite_backend_smoke.py` | Stage 1 P10b/P10c shared prediction SQLite backend smoke | 在 tempfile 下构造 4 sample × 5 model 的 packed_npy_v1 manifest 和数组，验证 `build_prediction_sqlite_backend(...)`、SQLite metadata、`fetch_records(...)`、`records_to_ordered_rows(...)`、经 `time_router.io.load_prediction_arrays_grouped(...)` grouped packed loading、row index lineage、shape、默认缺失报错和 `allow_missing=True` missing report；不访问 `/data2`、不运行正式入口、不写正式输出 |
| `tests/smoke/stage1_runtime_artifact_writer_smoke.py` | Stage 1 P11c Runtime artifact writer smoke | 使用 `tempfile` 创建本地临时 `output_root/run_dir`，验证 canonical 子目录、JSON 可读、schema version 字段、split summary count、`predictions/` 与 `evaluation/` 分离、per-sample CSV 写出和 `ProviderWithoutRunDir` mock 不接收 `run_dir`；不访问 `/data2`、不启动训练、不改正式入口 |
| `tests/smoke/stage1_canonical_protocol_run_smoke.py` | Stage 1 P11d canonical protocol run smoke | 使用 tiny `SampleManifest`、测试内 `TinyExpertProvider`、临时 TimeFuse feature CSV、`TimeFuseLinearSoftmaxHead`、`EvaluationInputAdapter` 和 Runtime artifact writer 串通 canonical dataflow，在 tempfile 下写出 canonical `run_dir`；验证 sample_key 保序、schema/sample/split/row count、predictions/evaluation 分层、Provider/Head/Evaluator 不接收 `run_dir` 且不访问 `/data2` |
| `tests/smoke/stage1_canonical_small_entrypoint_smoke.py` | Stage 1 P12 small canonical entrypoint smoke | 使用 `tempfile` 通过 subprocess 调用 `scripts/run_stage1_canonical_small.py --output-root ... --run-name ...`，验证返回码为 0、stdout 包含 `run_dir`、canonical 子目录存在、`run_metadata.json` / `run_status.json` / inputs / evaluation / predictions artifact 可读、prediction rows 保持 manifest sample_key 顺序、未引用 `/data2` 且未启动正式训练入口 |
| `tests/smoke/stage1_timefuse_small_entrypoint_smoke.py` | Stage 1 P15b TimeFuse small entrypoint smoke | 使用 tempfile 通过 subprocess 调用 `scripts/run_stage1_timefuse_small.py`，验证 canonical run_dir、metadata/status、inputs、evaluation summary、prediction rows 和最小日志文件；同时复用 entrypoint provider/head 组合检查 17 维 FeatureBatch、ExpertBatch/RouterOutput model_columns 对齐、weights shape/finite/softmax row sum、sample_key 保序、generic small CLI 前后不变、未访问 `/data2` 且未启动正式训练入口 |
| `tests/smoke/stage1_visual_small_entrypoint_smoke.py` | Stage 1 P15c Visual small entrypoint smoke | 使用 tempfile 通过 subprocess 调用 `scripts/run_stage1_visual_small.py`，验证 canonical run_dir、metadata/status、inputs、evaluation summary、prediction rows 和最小日志文件；同时复用 entrypoint provider/head 组合检查 8 维 float32 Visual `FeatureBatch`、ExpertBatch/RouterOutput model_columns 对齐、weights shape/finite/softmax row sum、sample_key 保序、generic small CLI 与 TimeFuse small CLI 前后不变、未访问 `/data2`、未启动正式训练入口、未读取真实 checkpoint 且未启动 ViT |
| `tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py` | Stage 1 P15d branch small entrypoint artifact parity smoke | 使用 tempfile 通过 subprocess 分别调用 TimeFuse small entrypoint 和 Visual small entrypoint，比较两边 canonical run_dir 的共同结构、metadata/status/inputs/evaluation/prediction rows schema、sample_manifest row_count、split count、sample_key 顺序、split 列、config_name、model_columns 和有限指标字段；检查 TimeFuse/Visual branch-specific metadata，确认 stdout/stderr 未出现 `/data2`、正式训练入口、`torch.load`、ViT 相关 token，且 generic/TimeFuse/Visual small CLI 文件前后不变 |
| `tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py` | Stage 1 P16a Visual MLP RouterHead adapter smoke | 使用 P13b real-derived manifest ordered sample_keys、P14b `VisualMockFeatureProvider`、P13b expert JSON 和内存小型 torch MLP fixture，验证 `LoadedTorchMLPRouterHeadAdapter` 输出 `RouterOutput` 并可由 `EvaluationInputAdapter` 生成 summary/rows；覆盖 sample_key 保序、model_columns 对齐、logits/weights shape、finite、softmax row sum、selected counts、per-sample rows 保序，以及 feature dtype 非 float32、重复 model_columns、logits shape mismatch 三类负向用例；patch `torch.load` 并扫描源码，确认不读取 checkpoint、不访问 `/data2`、不启动 ViT、不调用 legacy `VisualMLPRouter` 或正式训练入口，且不替换 P15c script-local adapter |
| `tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py` | Stage 1 P16c Visual precomputed FeatureProvider smoke | 使用 P13b real-derived manifest ordered sample_keys、P16c precomputed head-ready visual embedding fixture、P13b expert JSON 和内存小型 torch MLP，验证 `VisualPrecomputedFeatureProvider -> FeatureBatch -> LoadedTorchMLPRouterHeadAdapter -> RouterOutput -> EvaluationInputAdapter` 链路；覆盖 fixture 行顺序打乱但输出保序、test split sample_key 覆盖、features shape/dtype/schema、missing sample_key、重复 fixture sample_key、非有限 feature、provider 不持有 `run_dir`、patch `torch.load` 和 per-sample rows 保序；不访问 `/data2`、不启动 ViT、不读取 checkpoint、不迁移正式入口、不创建 canonical run_dir |
| `tests/smoke/stage1_visual_feature_scaler_smoke.py` | Stage 1 P16d loaded Visual FeatureScaler smoke | 使用 P13b real-derived manifest ordered sample_keys、P16d raw visual feature/scaler state fixture、P13b expert JSON 和内存小型 torch MLP，验证 `raw FeatureBatch -> LoadedFeatureScaler -> LoadedTorchMLPRouterHeadAdapter -> RouterOutput -> EvaluationInputAdapter` 链路；覆盖 transform 数值、sample_key 保序、shape 不变、输出 `float32`、输入 FeatureBatch 未被修改、schema lineage、zero scale、长度不匹配、非有限 scaler state、非有限 input、missing/duplicate sample_key、patch `torch.load` 和 per-sample rows 保序；不访问 `/data2`、不启动 ViT、不读取 checkpoint、不迁移正式入口、不创建 canonical run_dir |
| `tests/smoke/stage1_visual_feature_chain_protocol_smoke.py` | Stage 1 P16f Visual feature chain protocol smoke | 使用 P13b real-derived manifest ordered sample_keys、smoke-local dummy raw/pre-image/pseudo-image/resize/encoder/pooling/transform components、P13b expert JSON 和内存小型 torch MLP，验证 `VisualFeatureChainSpec` 组合可输出 canonical `FeatureBatch` 并接入 P16a adapter 与 `EvaluationInputAdapter`；覆盖每层 sample_key 保序、dummy batch shape、最终 `float32` dtype、`raw_window/pre_image/pseudo_image/resize/encoder/pooling/transform` lineage、替换一个 dummy component 后仍输出合法 FeatureBatch、patch `torch.load` 和不创建 run_dir；不访问 `/data2`、不启动 ViT、不读取 checkpoint、不修改正式入口 |
| `tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py` | Stage 1 P12b small canonical fixture input contract smoke | 使用 `tests/fixtures/stage1_canonical_small/` 的 tiny manifest、feature CSV 和 expert JSON，通过 subprocess 分别运行默认内联 fixture 与显式 fixture，验证显式 fixture 输出与内联 fixture `prediction_rows.csv` 一致、prediction rows 保持 manifest 行顺序、`run_metadata.inputs` 记录 sample_manifest / feature_source / expert_fixture 来源摘要、未引用 `/data2` 且未启动正式训练 |
| `tests/fixtures/stage1_canonical_small/` | Stage 1 P12b small canonical entrypoint fixture | 保存 `sample_manifest.csv`、`features.csv` 和 `expert_predictions.json` 三个 tiny fixture；用于验证显式 small input contract，manifest 使用 P11b 最小字段且行顺序作为 ordered sample_keys，feature CSV 可乱序，expert JSON 保存小数组 `model_columns/y_true/y_pred`；不代表正式 Visual/TimeFuse 输入已经迁移 |
| `tests/smoke/stage1_real_derived_small_fixture_smoke.py` | Stage 1 P13b real-derived small fixture smoke | 使用 `tests/fixtures/stage1_real_derived_small/` 的 real-derived / schema-style manifest、feature CSV 和 expert JSON，通过 subprocess 调用 P12b small entrypoint，验证 fixture 不在 `/data2`、manifest 字段为 P11b 最小字段、feature/expert sample_key 集合对齐且顺序打乱、canonical `run_dir` artifact 写出、`prediction_rows.csv` 保持 manifest 行顺序、`run_metadata.inputs` 记录三个输入来源、`evaluation_summary.sample_count` 与 manifest 行数一致 |
| `tests/smoke/stage1_prediction_backend_expertbatch_smoke.py` | Stage 1 P13d prediction backend -> ExpertBatch smoke | 使用 P13b real-derived manifest 的 ordered sample_keys 和 P13b expert JSON 数值参考，在 tempfile 内构造 packed_npy_v1 prediction manifest、数组和 SQLite backend；验证 shared SQLite backend fetch records、grouped packed array loading、row index lineage、`PredictionBatchReader`、`PredictionCacheExpertProvider` 和 `ExpertBatch` 的 sample_key/model_columns/shape/数值一致性；显式使用 `validate_manifest_schema=False` 处理 P13b 非 canonical sample_key，且不访问 `/data2`、不运行正式入口、不写正式输出 |
| `tests/fixtures/stage1_real_derived_small/` | Stage 1 P13b real-derived small fixture | 保存从 P10f/P10g smoke 小样本身份派生的 `sample_manifest.csv`、schema-style `features.csv`、P12b 小数组 `expert_predictions.json` 和 README；用于验证真实字段风格小规模输入的保序 join 和 canonical artifact 写出；三列 feature 不是 TimeFuse 17 维 full-scale feature cache，expert JSON 不是正式 prediction backend schema |
| `tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py` | Stage 1 P13e TimeFuse 17 维 FeatureProvider smoke | 使用 P13b real-derived manifest 的 ordered sample_keys 和 `tests/fixtures/stage1_timefuse_17dim_small/features_17d.csv`，构造 `TimeFuseFeatureCacheProvider` 并输出 `FeatureBatch`；验证 fixture sample_key 集合对齐且 CSV 顺序打乱、FeatureBatch 保持 manifest 顺序、`features=(4, 17)`、schema/extra 来源信息和按 manifest 重排后的数值一致；provider 阶段只允许读取 feature CSV 并阻断 `np.load`，证明不读取 oracle/error/prediction、不写 canonical run_dir |
| `tests/fixtures/stage1_timefuse_17dim_small/` | Stage 1 P13e TimeFuse 17 维 small fixture | 保存 `features_17d.csv` 和 README；CSV 使用 P13b manifest 的 4 个 sample_key 和正式 TimeFuse feature builder 的 17 个列名，行顺序刻意不同于 manifest；用于验证 `TimeFuseFeatureCacheProvider -> FeatureBatch`，不是 full-scale feature cache，也不包含 oracle/error/prediction |
| `tests/smoke/stage1_visual_feature_provider_mock_smoke.py` | Stage 1 P14b Visual FeatureProvider mock smoke | 使用 P13b real-derived manifest 的 ordered sample_keys 和 `tests/fixtures/stage1_visual_feature_mock/history_windows.json`，构造 `VisualMockFeatureProvider` 并输出 `FeatureBatch`；验证 history fixture 只包含历史窗口 x、FeatureBatch 保持 manifest 顺序、`features=(4, 8)`、dtype 为 `float32`、schema/extra 轻量口径；provider 阶段阻断 `open`、`Path.open`、`Path.read_text` 和 `np.load`，证明不读取 prediction/oracle/y_true/run_dir/status/checkpoint、不写 canonical run_dir |
| `tests/smoke/stage1_visual_mock_protocol_eval_smoke.py` | Stage 1 P14d Visual mock protocol eval smoke | 使用 P13b real-derived manifest 的 ordered sample_keys、P14b history window fixture 和 P13b expert JSON 参考，在内存中构造 `FeatureBatch` 与 `ExpertBatch`，再通过 smoke-only deterministic mock RouterHead 输出 `RouterOutput` 并调用 `EvaluationInputAdapter`；验证 sample_key 保序、model_columns 对齐、weights shape/归一化、summary sample_count、hard/raw-soft 指标、selected counts 和 per-sample rows；mock head/evaluator 阶段阻断文件 IO、`np.load`、`np.save`，确认不写 canonical run_dir |
| `tests/fixtures/stage1_visual_feature_mock/` | Stage 1 P14b Visual mock fixture | 保存 `history_windows.json` 和 README；JSON 使用 P13b manifest 的 4 个 sample_key，值为小型 history window x，不包含 future y、y_true、oracle/error、prediction cache path、run_dir、metadata、status、checkpoint 或 `/data2`；用于验证 `VisualMockFeatureProvider -> FeatureBatch`，不是正式 ViT provider 输入 |
| `tests/fixtures/stage1_visual_precomputed_small/` | Stage 1 P16c Visual precomputed small fixture | 保存 `visual_embeddings.csv` 和 README；CSV 使用 P13b manifest 的 4 个 sample_key，覆盖 test split 两个 sample_key，行顺序刻意不同于 manifest，只包含 `sample_key` 与 `feature_0 ... feature_7` 固定数值；用于验证 `VisualPrecomputedFeatureProvider -> FeatureBatch`，不是真实 ViT embedding cache，也不包含 scaler、checkpoint、prediction、oracle、run_dir 或 `/data2` |
| `tests/fixtures/stage1_visual_scaler_small/` | Stage 1 P16d Visual scaler small fixture | 保存 `raw_visual_features.csv`、`scaler_state.json` 和 README；CSV 使用 P13b manifest 的 4 个 sample_key，行顺序刻意不同于 manifest，JSON 保存固定 `mean` / `scale` / `feature_columns`；用于验证 loaded scaler transform，不包含 checkpoint、ViT、pseudo image、prediction、oracle、run_dir 或 `/data2` |
| `tests/smoke/stage1_evaluation_input_adapter_smoke.py` | Stage 1 P6b EvaluationInput adapter smoke | 默认读取同一 4 sample packed golden fixture，先用 `PredictionCacheExpertProvider` 显式构造 `ExpertBatch`，再用 golden weights 构造 `RouterOutput` 并覆盖显式 fusion weights 输入路径；验证 `EvaluationInput` 保序、固定五专家顺序、summary golden 数值、per-sample rows 字段/数值、hard/raw-soft MAE/MSE、max_weight、weight_entropy 和 adapter `extra`；adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，并检查 `run_outputs` 一层目录集合不变，证明不重读 prediction cache/oracle/TSF、不创建正式输出目录 |
| `tests/smoke/stage1_fusion_evaluator_adapter_smoke.py` | Stage 1 P6c FusionEvaluator compat smoke | 默认读取同一 4 sample packed golden fixture，先用 `PredictionCacheExpertProvider` 显式构造 `ExpertBatch`，再用 golden weights 构造 `RouterOutput` 并调用兼容 `FusionEvaluator`；验证旧路径 `EvaluationInput` 保序、summary/rows 数值不漂移，并检查 diagnostics 中 `canonical_adapter_name=EvaluationInputAdapter`；adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，并检查 `run_outputs` 一层目录集合不变，证明不重读 prediction cache/oracle/TSF、不创建正式输出目录 |
| `tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py` | Stage 1 P7a TimeFuseFeatureCacheProvider smoke | 使用测试内临时 feature CSV 构造 `TimeFuseFeatureCacheProvider` 并显式传入 sample_keys；验证 `FeatureBatch` 类型、sample_keys tuple 保序、`features=(2, 17)`、`feature_schema`、`extra`、空/重复 sample_key 拒绝；provider 阶段只允许读取临时 feature CSV 并阻断 `np.load`，检查 `run_outputs` 一层目录集合不变，证明不读取 prediction/oracle、不创建输出目录 |
| `tests/smoke/stage1_timefuse_linear_head_smoke.py` | Stage 1 P7b TimeFuseLinearSoftmaxHead smoke | 使用测试内固定 `FeatureBatch`、固定线性权重和 bias 构造 `RouterOutput`；验证 sample_keys 保序、model_columns 对齐、logits/weights deterministic 数值、weights 逐样本和为 1、`__call__` 与 `predict` 一致以及非法 model_columns 拒绝；head 阶段阻断文件 IO、`np.load` 和 `np.save/np.savez`，检查 `run_outputs` 一层目录集合不变，证明不读取 cache/feature CSV、不训练、不写运行产物 |
| `tests/smoke/stage1_timefuse_protocol_chain_smoke.py` | Stage 1 P7c TimeFuse protocol chain smoke | 使用 golden prediction fixture 构造 `ExpertBatch`，使用测试内临时 feature CSV 构造 `FeatureBatch`，用固定 `TimeFuseLinearSoftmaxHead` 权重生成 `RouterOutput`，再通过 `EvaluationInputAdapter` 复算 summary/rows；验证 sample_keys/model_columns/features/weights/summary/rows 保序且 deterministic；head/evaluator 阶段阻断文件 IO、`np.load` 和 `np.save/np.savez`，检查 `run_outputs` 一层目录集合不变，证明不回读 cache/oracle/TSF、不训练、不写运行产物 |
| `tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py` | Stage 1 P9d Visual Router ExpertBatch evaluation bridge smoke | 使用测试内小型 numpy arrays 和 DataFrame 构造与 Visual Router test batch 等价的 `pred_df`、`soft_df`、weights、`y_pred` 和 `y_true`，直接调用 `verify_evaluation_adapter_bypass_batch(...)`，验证 helper 经 `ExpertBatch + fusion_weights` 调用 `EvaluationInputAdapter.evaluate(...)`，`sample_keys`/`model_columns` 保序，float32 `y_pred/y_true` 原样进入 adapter，adapter rows 与正式字段的 selected_model、hard MAE/MSE、raw soft MAE/MSE、max_weight、weight_entropy 一致，并覆盖故意 mismatch 时错误信息包含 config/split/batch/sample/字段/旧值/adapter 值/output_dir；不启动 ViT、不访问 `/data2`、不运行正式入口 |
| `tests/smoke/stage1_visual_router_training_expert_batch_bypass_smoke.py` | Stage 1 P9f Visual Router training ExpertBatch bypass smoke | 使用测试内小型 numpy arrays 构造 training batch 的 `y_pred/y_true`，直接调用 `verify_training_expert_errors_from_expert_batch(...)`，验证 MAE/MSE `expert_errors` 可由 `ExpertBatch.y_pred/y_true` 显式复算并与 legacy 值一致；覆盖故意 mismatch 时错误信息包含 phase/router_mode/metric/batch/sample/model/expert/value/output_dir 定位上下文；不启动 ViT、不访问 `/data2`、不运行正式入口 |

## 3. QuitoBench / Quito 代码与实验层

`quito/` 是当前 baseline 复盘和评估的主工作目录，包含 Quito Python 包源码、原始脚本、文档、数据审计结果和实验输出。

```text
quito/
├── configs/
├── data/
├── data_audit/
├── docs/
├── examples/
├── exp_scripts/
├── quito/
├── outputs/
├── quito.egg-info/
├── README.md
├── pyproject.toml
└── requirements*.txt
```

### 3.1 Quito 项目根文件

| 路径 | 功能 |
| --- | --- |
| `quito/README.md` | Quito 项目说明 |
| `quito/CONTRIBUTING.md` | Quito 贡献说明 |
| `quito/LICENSE` | Quito 许可证 |
| `quito/pyproject.toml` | Python 包构建、安装和 CLI 入口配置 |
| `quito/requirements.txt` | Quito 基础依赖 |
| `quito/requirements-optional.txt` | Quito 可选依赖 |

### 3.2 `quito/configs/`

| 路径 | 功能 | 备注 |
| --- | --- | --- |
| `quito/configs/` | Quito 配置目录 | 当前仅有示例配置；本轮正式运行配置主要由 `experiment_scripts/` 写入 `run_outputs/*/generated_configs/` |
| `quito/configs/example_config.yaml` | Quito 示例配置 | 用于参考配置字段 |

### 3.3 `quito/data/`

| 路径 | 功能 | 备注 |
| --- | --- | --- |
| `quito/data/` | Quito 数据目录 | 当前 `find` 未列出直接本地数据文件 |
| `quito/data/hf/` | Hugging Face 数据缓存/镜像目录 | 当前未展开到具体文件 |

### 3.4 `quito/data_audit/`

| 路径 | 功能 |
| --- | --- |
| `quito/data_audit/` | 数据审计与 cluster 映射产物根目录 |
| `quito/data_audit/quitobench_clusters/` | QuitoBench cluster/TSF cell 相关映射和质量统计产物 |
| `quito/data_audit/quitobench_clusters/all_item_cluster.csv` | 全量 item 到 cluster/TSF cell 的映射 |
| `quito/data_audit/quitobench_clusters/all_item_cluster_with_quality.csv` | 带数据质量字段的全量 item cluster 映射 |
| `quito/data_audit/quitobench_clusters/cluster_conflicts.csv` | cluster 映射冲突检查结果 |
| `quito/data_audit/quitobench_clusters/cluster_summary.csv` | cluster 统计汇总 |
| `quito/data_audit/quitobench_clusters/hour_item_cluster.csv` | 小时级 item cluster 映射 |
| `quito/data_audit/quitobench_clusters/min_item_cluster.csv` | 分钟级 item cluster 映射 |
| `quito/data_audit/quitobench_clusters/manifest.json` | cluster 审计产物清单和来源信息 |

### 3.5 Quito 文档、示例和原始脚本

| 路径 | 功能 |
| --- | --- |
| `quito/docs/` | Quito 文档，包含 finetune、evaluate、pretrain、tune 和数据质量说明 |
| `quito/examples/` | Quito 示例脚本和数据处理入口 |
| `quito/examples/item_csv.csv` | Quito 示例 item 元数据/映射 CSV |
| `quito/examples/data_analysis/` | 数据质量分析、cluster 构建、数据合并等辅助脚本 |
| `quito/examples/datasets/` | 示例数据预处理脚本 |
| `quito/exp_scripts/` | Quito 原始 shell 实验脚本 |
| `quito/exp_scripts/finetune/` | 原始 finetune shell 脚本，覆盖 DLinear、PatchTST、CrossFormer 等模型 |
| `quito/exp_scripts/evaluate/` | 原始 evaluate shell 脚本，覆盖深度学习、统计基线和外部 foundation model |
| `quito/exp_scripts/tune/` | 原始 tuning shell 脚本 |

### 3.6 `quito/quito/` Python 包源码

| 路径 | 功能 |
| --- | --- |
| `quito/quito/cli.py` | `quito-cli` 命令行入口 |
| `quito/quito/config/` | 配置 dataclass 和解析逻辑 |
| `quito/quito/models/` | 模型适配层，包含 DLinear、PatchTST、CrossFormer、ES、SNaive 等 |
| `quito/quito/scripts/` | `finetune.py`、`evaluate.py`、`tune.py`、`pretrain.py` 等实际脚本入口 |
| `quito/quito/trainers/` | 训练器实现和自动选择逻辑 |
| `quito/quito/utils/` | 数据、指标、分布式和可视化工具 |
| `quito/quito/__pycache__/` | Python 缓存，可再生成 |
| `quito/quito.egg-info/` | 本地 editable/install 生成的包元数据，可再生成 |

### 3.7 `quito/outputs/` 实验输出层

`quito/outputs/` 是模型训练、评估、checkpoint 和 smoke/tuning/statistical baseline 结果的根目录。

| 路径 | 层级角色 | 当前口径 |
| --- | --- | --- |
| `quito/outputs/default_baseline/` | 深度学习 baseline 原始输出 | DLinear/PatchTST/CrossFormer，3 个配置，`seed_16`，evaluate 使用 validation MAE-best checkpoint |
| `quito/outputs/default_baseline/{dlinear,patchtst,crossformer}/{96_48_S,576_288_S,1024_512_S}/seed_16/FINE_TUNE/ver_0/` | 单模型单配置 finetune 输出 | 包含配置、训练日志、TensorBoard event 和 `checkpoints/` |
| `quito/outputs/default_baseline/{dlinear,patchtst,crossformer}/{96_48_S,576_288_S,1024_512_S}/seed_16/EVALUATE/ver_0/` | 单模型单配置 test evaluate 输出 | 包含 `eval_results_*.json`、`config.yaml`、`log.txt` |
| `quito/outputs/default_baseline_mse_best/` | 后补 validation MSE-best 复盘 evaluate 输出 | 当前只有 `PatchTST 576_288_S` 的 MSE-best 补评估 |
| `quito/outputs/statistical_baseline/` | ES/SNaive 统计基线 evaluate 输出 | 按 `es/`、`snaive/` 和三组配置组织 |
| `quito/outputs/smoke/` | 早期 smoke test 输出 | 用于验证链路，不作为最终 baseline 表默认来源 |

## 4. TimeFuse 代码层

`TimeFuse/` 是独立的 TimeFuse 相关模型代码库，当前主要作为外部/对照代码资源保留。

```text
TimeFuse/
├── data_provider/
├── exp/
├── layers/
├── models/
├── pics/
├── scripts/
├── utils/
├── run.py
├── run_config.json
└── timefuse.py
```

### 4.1 TimeFuse 项目根文件

| 路径 | 功能 |
| --- | --- |
| `TimeFuse/README.md` | TimeFuse 项目说明 |
| `TimeFuse/CONTRIBUTING.md` | TimeFuse 贡献说明 |
| `TimeFuse/LICENSE` | TimeFuse 许可证 |
| `TimeFuse/requirements.txt` | TimeFuse 依赖清单 |
| `TimeFuse/run.py` | TimeFuse 实验主入口 |
| `TimeFuse/run_config.json` | TimeFuse 示例/默认运行配置 |
| `TimeFuse/run_timefuse_exp.ipynb` | TimeFuse notebook 实验入口 |
| `TimeFuse/timefuse.py` | TimeFuse 核心封装/入口文件 |
| `TimeFuse/args.py` | 命令行参数定义 |
| `TimeFuse/load_configs.py` | 配置加载逻辑 |
| `TimeFuse/meta_feature.py` | meta feature 相关逻辑 |

### 4.2 TimeFuse 子目录

| 路径 | 功能 |
| --- | --- |
| `TimeFuse/data_provider/` | 数据加载和数据工厂 |
| `TimeFuse/exp/` | TimeFuse 实验类和基础实验流程 |
| `TimeFuse/layers/` | 各类时间序列模型共享层和编码器/解码器模块 |
| `TimeFuse/models/` | TimeFuse 支持的模型定义，包括 PatchTST、DLinear、CrossFormer、iTransformer 等 |
| `TimeFuse/pics/` | README/论文说明用图片资源 |
| `TimeFuse/scripts/` | TimeFuse shell 脚本目录 |
| `TimeFuse/scripts/long_term_forecast/` | 长期预测实验脚本 |
| `TimeFuse/utils/` | 指标、时间特征、增强、DTW、工具函数等 |

## 5. 生成物和缓存层

这一层说明哪些内容应被当作“可复核产物”，哪些只是可再生成缓存。

### 5.1 可作为复核来源的生成物

| 路径模式 | 用途 |
| --- | --- |
| `quito/outputs/**/EVALUATE/ver_*/eval_results_*.json` | 模型 evaluate 的核心结果 JSON |
| `quito/outputs/**/FINE_TUNE/ver_*/checkpoints/*.ckpt` | 深度学习模型 checkpoint；引用时必须说明 MAE-best、MSE-best、last 或具体 epoch |
| `quito/outputs/**/config.yaml` | 对应训练/评估的配置快照 |
| `experiment_logs/run_outputs/*/status.json` | 编排脚本任务状态 |
| `experiment_logs/run_outputs/*/*.csv` | 编排脚本生成的汇总表或复盘表 |
| `experiment_logs/run_outputs/*/cluster_analysis/` | 分 cluster/TSF cell 分析结果 |
| `experiment_logs/run_outputs/*_five_model_three_config_summary/overall_mean_mae_pivot.csv` | 五模型三配置 overall mean MAE 主表 |
| `experiment_logs/run_outputs/*_five_model_three_config_summary/tsf_cell_mae_pivot.csv` | 五模型三配置分 TSF cell MAE 透视表 |
| `experiment_logs/run_outputs/*_five_model_three_config_summary/checkpoint_lineage.csv` | 五模型三配置 evaluate 来源和 checkpoint 口径 |

### 5.2 可再生成或不作为正式结果引用的内容

| 路径模式 | 用途 |
| --- | --- |
| `**/__pycache__/` | Python 字节码缓存 |
| `quito/quito.egg-info/` | 本地安装生成的包元数据 |
| `**/events.out.tfevents.*` | TensorBoard event；可用于复盘训练曲线，但最终表格应引用整理后的指标 |
| `experiment_logs/run_outputs/*.log` | 后台运行日志；用于排查过程，不单独作为最终结果表来源 |

## 6. 后续更新触发条件

出现以下情况时，需要同步更新本文档：

1. 新增或删除顶层目录、长期文档、正式脚本。
2. 新增实验输出根目录，例如 `quito/outputs/new_baseline/`。
3. 新增长期数据产物、汇总 CSV/JSON、cluster/TSF cell 分析结果。
4. 改变现有目录职责或结果口径，例如从 MAE-best 改为 MSE-best 作为默认表格来源。
5. 清理中止实验半成品后，相关输出目录状态发生变化。
