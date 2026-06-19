# 工作区结构说明

更新日期：2026-06-20 01:49:59 CST

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
| `docs/refactor/` | 重构前审计与迁移设计文档目录；当前包含 Stage 1 路线审计、目标架构、重构路线图、公共模块迁移候选、golden fixture、共享 PredictionBatchReader 说明、共享 OracleTsfReader 说明、evaluation package 边界复核、P4a JSON utils 边界说明、P4b path resolver 边界说明、P4c run metadata 边界说明、P4d run artifacts 边界复核、P4e checkpoint index 边界复核、P4 后 architecture pivot 决策、P5a canonical runtime contract、P5b canonical provider interface design、P5c protocol types skeleton、P5d provider adapter boundary review、P5e canonical entrypoint migration plan、P5f launcher architecture、P6a PredictionCacheExpertProvider、P6a.5 expert system boundary review、P6b EvaluationInput adapter、兼容 FusionEvaluator adapter、P7a/P7b/P7c TimeFuse adapter smoke 文档、P8a TimeFuse 正式入口 adapter 插入审计、P8c evaluation adapter pressure 验证、P8d TimeFuse baseline parity review 和 P9a Visual Router 正式入口 adapter 插入审计 | 路线或迁移结论变化时更新；代码迁移应另写实验日志和验证结果 |
| `docs/refactor/stage1_route_audit.md` | Stage 1 共享主干、Visual/TimeFuse 分支、废弃路线及 36 个 Python 文件标签审计 | 新增/归档 Stage 1 脚本或正式路线改变时同步复核 |
| `docs/refactor/stage1_target_architecture.md` | Stage 1 未来目标架构设计，定义 `time_router/{data,io,features,models,evaluation,training}`、`scripts/`、`configs/`、`exp_scripts/` 和 `archive/` 边界，并明确共享主干与 Visual/TimeFuse 两个 FeatureProvider 分支；P6a.5 起补充 ExpertProvider / ExpertBatch 长期专家系统边界与 PredictionCacheExpertProvider adapter 实现边界 | 当前只作为设计文档；实现 package、迁移入口或归档旧代码时需另行验证并更新 |
| `docs/refactor/stage1_refactor_roadmap.md` | Stage 1 后续小步重构路线图，按 P0-P6 及 P2.5/P3a-P3e/P6a.5 等中间小步拆分 architecture docs、prediction reader、oracle/TSF reader、metrics/fusion、router weight diagnostics、summary、per-sample rows、evaluation package 边界复核、logging/path/config、FeatureProvider、专家系统边界审计和入口迁移 | 每个迁移步骤前后都应运行 `tests/smoke/stage1_golden_smoke.py` 并写实验日志 |
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
| `docs/refactor/stage1_provider_interface.md` | Stage 1 P5b canonical provider interface design | 定义 `ExperimentProtocol -> SplitStrategy -> ExpertProvider -> FeatureProvider -> RouterHead -> Evaluator` 的共享接口边界；明确当前 fixed config / five experts / prediction cache / vali-test 只是默认实现，接口不写死 frozen ViT、17 维 feature cache、固定 split 或固定训练方式；记录 Visual Router 与 TimeFuse-style fusor 共享 contract、branch-specific extra、oracle/TSF 禁止作为可部署 test-time 动态调权特征、provider 不决定 `run_dir` 和 deprecated/reference-only 历史路线边界；本身不改训练入口、不实现接口代码 |
| `docs/refactor/protocol_types.md` | Stage 1 P5c protocol types skeleton | 记录 `time_router.protocols` 中 `SplitSpec`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput` 和 `ExperimentProtocolSpec` 的字段、轻量 contract 边界、tuple/default_factory 约束、public API、smoke 覆盖和明确不做范围；本身不实现 provider/runtime/config/checkpoint/logging，也不迁移正式入口 |
| `docs/refactor/provider_adapter_boundary.md` | Stage 1 P5d provider adapter boundary review | 审查 `PredictionBatchReader`、`prediction_array_io`、Visual pseudo image / ViT feature 路径、TimeFuse 17 维 feature cache reader、Visual Router head、TimeFuse linear-softmax head 和 `time_router.evaluation` public API 的未来 adapter 适配边界；明确第一批应先做 entrypoint migration plan，再优先实现 `PredictionCacheExpertProvider`，TimeFuse feature cache provider 作为后续 feature-only adapter，Visual online ViT provider 不作为第一批最小实现；本身不实现 provider adapter、不改训练入口、不接入 `/data2` |
| `docs/refactor/stage1_entrypoint_migration_plan.md` | Stage 1 P5e canonical entrypoint migration plan | 基于 P5a-P5d 设计拆分 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 和 `launch_timefuse_fusor_full_scale.py` 的 runtime orchestration、ExpertProvider、FeatureProvider、RouterHead、Evaluator 与 launcher 边界；建议第一批迁移先做 `PredictionCacheExpertProvider`，再做 evaluator adapter、TimeFuse feature provider、TimeFuse head 和 Visual online ViT provider；明确新 adapter 先由 smoke 使用、正式入口后续小步接入、不创建 run_dir、不硬编码 `/data2`；本身不改训练代码、不实现 adapter |
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

### 2.2 `experiment_logs/`

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
| `visual_router_experiments/common/` | 跨阶段公共代码目录 | 保存 prediction cache schema、item-channel-window key、指标、伪图像张量构造、运行内视觉 embedding 工具和通用评估工具；当前已有 `prediction_cache_schema.py`、`prediction_array_io.py`、`pseudo_imageization.py` 和 `vit_embedding_utils.py`；`prediction_array_io.py` 统一读取 `per_sample_npy` 与 `packed_npy_v1` prediction arrays；`pseudo_imageization.py` 已支持 `hf_vit_0_5` 与 `torchvision_imagenet` encoder normalization，并新增固定候选周期桶与按周期分桶 fold 路径，减少在线伪图像化中的逐样本 CPU/GPU 同步；`vit_embedding_utils.py` 为 online 主线提供不落盘的 ViT 输入/输出处理工具 |
| `visual_router_experiments/stage0_oracle_audit/` | 上限审计阶段目录 | 承接专家互补性和 oracle 上限审计；当前 README 索引已有审计脚本与输出，后续扩展专家池或 window-level oracle 可在此补充正式脚本 |
| `visual_router_experiments/stage1_vali_test_router/` | Stage 1 主实验目录 | 保存 vali 训练 router、test 测试 router 的 prediction cache、oracle labels、TSF enrichment、TimeFuse feature cache、embedding、训练、评估和汇总脚本；`README.md` 现作为当前主线导航页，明确 visual router full-scale 正式入口、中小规模复现入口、baseline 支线入口、共享库、pilot 边界和下一步；`stage1_visual_router_mainline.md` 只记录视觉路由主线、`96_48_S` 正确路线、废弃路线和扩 config 标准步骤，明确 TimeFuse-style fusor 是 baseline 支线；`stage1_history_results.md` 保存从 README 拆出的 120 sample smoke、1k、dry-run 和 full-scale 长跑历史结果索引；当前已有 `prediction_cache_design.md`、`feature_and_rl_extension_notes.md`、`stage1_cache_contract.md`、`stage1_visual_router_mainline.md`、`stage1_protocol_and_plan.md`、`stage1_history_results.md`、`stage1_timefuse_fusor_streaming_reader_design.md`、`build_stage1_sample_manifest.py`、`build_full_scale_sample_manifest.py`、`build_prediction_cache_from_manifest.py`、`merge_prediction_cache_shards.py`、`launch_full_scale_prediction_cache.py`、`build_full_scale_window_oracle_labels.py`、`build_full_scale_tsf_enrichment.py`、`validate_full_scale_oracle_tsf_outputs.py`、`build_timefuse_feature_cache_from_manifest.py`、`launch_timefuse_feature_cache_full_scale.py`、`stage1_timefuse_fusor_streaming_reader.py`、`train_timefuse_fusor_streaming.py`、`launch_timefuse_fusor_full_scale.py`、`run_full_scale_dry_run.py`、`evaluate_router_baselines.py`、`fusion_utils.py`、`train_visual_router.py`、`train_visual_router_online.py`、`train_visual_router_online_streaming.py`、`evaluate_soft_fusion_calibration.py`、`pilot/` 和 package 初始化文件；baseline evaluator 现可同时输出统计 baseline、TimeFuse-style fusor hard/raw-soft、oracle 和统一 comparison；visual full-scale 路线使用 packed prediction cache 与 streaming online router，不落盘 ViT embedding `.npy` 或伪图像 tensor；TimeFuse-style fusor full-scale 读取层已支持 feature shard、oracle parquet、五专家 shard/merged prediction manifest 的 shard-local SQLite + batch reader，streaming train/eval 入口已完成 1-shard smoke、checkpoint eval-only 和 2-shard小切片压力测试，并已支持 CUDA 多卡 `DataParallel`；2026-06-19 起 fusor reader/train 支持 index 复用、feature-only scaler、split 下推、packed npy batch-level grouped loading 和大块 CSV 后切 batch；正式 64-shard GPU2/3 后台 launcher 独立追踪于 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` |
| `visual_router_experiments/stage1_vali_test_router/pilot/` | Stage 1 pilot 脚本目录 | 保存 `build_prediction_cache_pilot.py`、`build_vit_embeddings_pilot.py`、`build_online_pseudo_image_pilot.py`、`build_structure_feature_cache_pilot.py`、`train_structure_router_pilot.py`、`compute_window_oracle_from_cache.py`、`enrich_cache_with_tsf_cell.py`、`launch_96_48_s_1k_prediction_cache_pilot.py`、`launch_96_48_s_1k_vit_embedding_pilot.py` 等小规模验证、离线 embedding 历史对照、过渡性 launcher 和固定规模资源编排脚本；用于打通 cache/oracle/enrichment/feature/router 流程或复现 1k smoke 编排，不作为通用正式实验入口 |
| `visual_router_experiments/stage2_heldout_cell/` | Stage 2 泛化实验目录 | 后续保存 7-cell 训练、held-out cell 测试的 zero-shot 泛化实验脚本 |

### 2.5 `time_router/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `time_router/__init__.py` | 共享 package 入口 | Stage 1 后续重构使用的最小共享 package 骨架；当前承载低风险公共 reader、最小 evaluation helper、protocol types 和 P6a expert adapter，不代表正式训练入口已迁移 |
| `time_router/data/__init__.py` | 共享数据子包入口 | 导出 `OracleTsfBatch` 和 `OracleTsfReader` |
| `time_router/data/oracle_tsf_reader.py` | Stage 1 共享 oracle/TSF reader | 按 sample_key 批量读取 window-level oracle labels 与 TSF enrichment / TSF-cell metadata；支持显式 sample_key 保序、CSV chunk 过滤、Parquet dataset 过滤、`missing_policy=error/report`、冲突重复/缺失检查和 oracle/TSF 一对一 join；只做读取、校验和 join，不提供训练策略，不把 oracle/TSF 作为可部署 test-time 动态特征 |
| `time_router/evaluation/__init__.py` | 共享评估子包入口 | 导出 P3a/P3b/P3c/P3d 最小 fusion/metrics/router weight diagnostics/summary/rows helper，以及 P6b/P6c canonical `EvaluationInputAdapter` 和兼容 `FusionEvaluator`；当前不包含 calibration、正式报告 schema 或训练入口迁移 |
| `time_router/evaluation/evaluation_input_adapter.py` | Stage 1 P6b/P6c canonical EvaluationInput adapter | 纯内存实现 `EvaluationInputAdapter` 和 `EvaluationInputAdapterResult`，将 `ExpertBatch + RouterOutput.weights` 或显式 fusion weights 包装为 `EvaluationInput`；`evaluate_input()` 是 adapter 层唯一调用 `hard_top1_fusion`、`raw_soft_fusion`、`build_fusion_summary` 和 `build_per_sample_fusion_rows` 的实现点；保持 sample/model 顺序，复用 `ExpertBatch.y_pred/y_true`，不读取 manifest、prediction cache、packed npy、oracle/TSF，不访问 `/data2`，不创建 run_dir，不写 status/metadata/CSV/JSON/Parquet |
| `time_router/evaluation/fusion_evaluator.py` | Stage 1 P6c FusionEvaluator compat wrapper | 纯内存保留 `FusionEvaluator` 和 `FusionEvaluationResult` 旧 public API；从 `ExpertBatch + RouterOutput` 或显式 `EvaluationInput` 进入后委托 `EvaluationInputAdapter` 构造/复算，不再直接调用 metrics/summary/rows helper；用于旧 smoke 和下游兼容，不作为新增逻辑入口 |
| `time_router/evaluation/metrics.py` | Stage 1 P3a/P3b 最小 fusion/metrics/diagnostics helper | 纯 numpy 实现 `compute_mae`、`compute_mse`、`validate_fusion_inputs`、`hard_top1_fusion`、`raw_soft_fusion`、`compute_selected_counts`、`compute_weight_entropy` 和 `compute_max_weight`；函数输入显式使用 `y_pred`、`y_true`、`weights`、`selected_indices`、`model_columns`，用于 golden smoke 复算 hard top-1/raw soft 指标和 router weight 诊断，不读取 manifest/oracle/TSF/正式输出目录，不引入 torch/sklearn 训练依赖 |
| `time_router/evaluation/prediction_rows.py` | Stage 1 P3d 最小 per-sample evaluation rows helper | 纯 numpy / Python 标准库实现 `build_per_sample_fusion_rows`，只消费显式传入的 `sample_keys`、`FusionMetricsResult`、`y_true`、`weights` 和 `model_columns`，输出当前 batch 的 sample_key、hard top-1 选择、逐样本 hard/raw-soft MAE/MSE、max weight 和 weight entropy；不读取 manifest、prediction cache、oracle/TSF 或正式输出目录，不写 CSV/JSON/Parquet，不实现 calibration、oracle regret 或正式 output schema 迁移 |
| `time_router/evaluation/summary.py` | Stage 1 P3c 最小 evaluation summary helper | 纯 numpy / Python 标准库实现 `build_fusion_summary`，只消费显式传入的 `FusionMetricsResult`、`weights` 和 `model_columns`，汇总 hard/raw-soft MAE/MSE、selected counts、mean entropy、mean max weight、样本数、专家数和专家顺序；不读取 manifest、prediction cache、oracle/TSF 或正式输出目录，不实现 calibration、oracle regret、comparison 或正式 output schema 迁移 |
| `time_router/experts/__init__.py` | 共享专家预测适配器子包入口 | 导出 P6a 最小 `PredictionCacheExpertProvider`；当前只用于 smoke，不接正式 Visual Router / TimeFuse fusor 入口 |
| `time_router/experts/prediction_cache.py` | Stage 1 P6a PredictionCacheExpertProvider | 复用 `time_router.io.PredictionBatchReader`，通过显式 `load_batch(sample_keys, verify_metrics=True)` 输出 `time_router.protocols.ExpertBatch`；保持 sample_key 顺序、固定五专家顺序、共享 y_true 校验、packed row index lineage 和 verify_metrics 校验能力；`extra` 只记录 provider name、array_storage 和轻量 reader metadata；不读取 oracle/TSF、不生成 feature、不计算 loss、不做 evaluation、不访问 `/data2`、不创建 run_dir、不写 status/metadata、不改训练入口 |
| `time_router/features/__init__.py` | 共享特征适配器子包入口 | 导出 P7a 最小 `TimeFuseFeatureCacheProvider`；当前只用于 smoke，不接正式 TimeFuse fusor / Visual Router 入口 |
| `time_router/features/timefuse_cache.py` | Stage 1 P7a TimeFuseFeatureCacheProvider | 读取调用方显式传入的小规模 feature CSV，并通过显式 `load_batch(sample_keys)` 输出 `time_router.protocols.FeatureBatch`；保持 sample_key 顺序，features 当前为 `numpy.float32` array，`feature_schema` 记录 schema 名、feature columns、feature_dim 和 source；不读取 prediction cache、oracle/TSF、`y_true` 或 expert error，不做 scaler fit，不访问 `/data2`，不创建 run_dir，不写 status/metadata/CSV/JSON/Parquet，不改正式入口 |
| `time_router/io/__init__.py` | 共享 IO 子包入口 | 导出 `DEFAULT_MODEL_COLUMNS`、`PredictionBatch`、`PredictionBatchReader`、`atomic_write_json`、`build_status_payload`、`write_status_json`、`find_repo_root`、`resolve_under_root`、`resolve_status_path`、`resolve_metadata_path`、`build_run_metadata` 和 `write_run_metadata`；P4d 起明确该入口只聚合稳定 public API，不在导入时读取配置、创建输出目录或执行训练相关副作用 |
| `time_router/io/prediction_cache_reader.py` | Stage 1 共享 prediction batch reader | 从 `merged_cache/manifest.csv` 或 fixture root 读取五专家 `y_pred` 和共享 `y_true`；支持 `packed_npy_v1`、`per_sample_npy`、固定专家顺序、共享 y_true 校验、row index 元数据和 manifest MAE/MSE 复算；P1 只接入 golden smoke，尚未迁移正式 Visual Router / TimeFuse fusor 入口 |
| `time_router/io/json_utils.py` | Stage 1 P4a 最小 JSON/status writer | 纯标准库实现 `atomic_write_json`、`build_status_payload` 和 `write_status_json`；只写调用方显式传入的 path，使用同目录临时文件、`flush + fsync` 和 `os.replace` 原子替换，默认 UTF-8 / `ensure_ascii=False`；不读取训练状态，不实现 path resolver、config system 或 logging framework |
| `time_router/io/path_resolver.py` | Stage 1 P4b 最小 path resolver | 纯标准库实现 `find_repo_root`、`resolve_under_root`、`resolve_status_path` 和 `resolve_metadata_path`；只做 repo root 查找、root 内安全拼接和 status/metadata path 计算，不创建目录、不写文件、不读取训练配置、不访问 `/data2` 或 full-scale 输出目录 |
| `time_router/io/run_metadata.py` | Stage 1 P4c 最小 run metadata payload builder | 纯标准库实现 `build_run_metadata` 和 `write_run_metadata`；构造至少包含 `stage`、`created_at_utc`、`inputs`、`outputs` 的 metadata-like payload，支持 Path 转字符串和 tempfile writer；不自动调用 git、不读取命令行/训练配置、不改变既有正式 metadata schema |
| `time_router/models/__init__.py` | 共享模型/head 适配器子包入口 | 导出 P7b 最小 `TimeFuseLinearSoftmaxHead`；当前只用于 smoke，不接正式 TimeFuse fusor / Visual Router 入口 |
| `time_router/models/timefuse_linear.py` | Stage 1 P7b TimeFuseLinearSoftmaxHead | 纯 numpy 实现固定线性层和 stable softmax；通过 `predict(feature_batch, model_columns)` 输出 `time_router.protocols.RouterOutput`，保持 sample_key/model_columns 顺序，logits/weights 专家维度与 model_columns 对齐；不训练、不计算 loss、不建 optimizer、不保存 checkpoint、不读取 prediction cache/oracle/TSF/feature CSV、不访问 `/data2`、不创建 run_dir、不写 status/metadata/CSV/JSON/Parquet、不改正式入口 |
| `time_router/protocols/__init__.py` | Stage 1 P5c protocol 子包入口 | 导出 `SplitSpec`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput` 和 `ExperimentProtocolSpec`；只聚合 lightweight contract public API，不实例化 provider、不读取配置或路径、不创建 run_dir |
| `time_router/protocols/types.py` | Stage 1 P5c 最小 protocol dataclass 类型骨架 | 纯标准库 `dataclass + typing.Any` 定义 split、expert batch、feature batch、router output、evaluation input 和 experiment protocol spec；array/tensor 字段不绑定 numpy/torch，不访问 `.shape`，不做数值校验；`extra`、`branch_specific` 和 `feature_schema` 使用 `field(default_factory=dict)`，不包含 `run_dir` |

### 2.6 `tests/smoke/`

| 路径 | 层级角色 | 功能 |
| --- | --- | --- |
| `tests/smoke/stage1_golden_smoke.py` | Stage 1 只读 golden smoke | 默认读取 `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/` 的 4 sample packed fixture；当前通过 `time_router.io.PredictionBatchReader` 组装 `y_pred/y_true`，并通过 `time_router.evaluation.metrics` / `time_router.evaluation.summary` / `time_router.evaluation.prediction_rows` 复算 hard top-1、raw soft fusion、P3b router weight diagnostics、P3c minimal summary 和 P3d per-sample rows；锁定 sample_key 顺序、五专家顺序、shape、hard top-1、raw soft fusion MAE/MSE、selected counts、entropy/max_weight shape、summary 字段数值、per-sample row 字段与数值和 `packed_npy_v1` row index 读取一致性；用于后续公共 reader/metrics/入口迁移前后等价验证，不训练、不写正式输出 |
| `tests/smoke/stage1_oracle_tsf_smoke.py` | Stage 1 oracle/TSF reader 只读 smoke | 默认读取同一 dry-run fixture 的 `window_oracle_labels_with_tsf_cell.csv` 和 `manifest_with_tsf_cell.csv`；验证 `OracleTsfReader` 的 `allow_full_scan` 默认禁止无 sample_key 全扫描、显式 sample_key 保序、oracle label、TSF metadata、oracle/TSF join、`missing_policy=error` 缺失报错、`missing_policy=report` 缺失报告和冲突重复 sample_key 报错；不训练、不生成 oracle/TSF、不写正式输出 |
| `tests/smoke/stage1_json_utils_smoke.py` | Stage 1 P4a JSON utils 临时目录 smoke | 在 `tempfile.TemporaryDirectory` 下验证 `write_status_json` / `atomic_write_json` 的文件存在、JSON 可读、中文 message 不被 ASCII 转义、第二次写入覆盖旧内容、nested parent directory 自动创建和 `extra` 类型检查；不读取训练状态、不写正式输出目录 |
| `tests/smoke/stage1_path_resolver_smoke.py` | Stage 1 P4b path resolver 临时目录 smoke | 验证从 `tests/smoke` 定位仓库根、root 下定位 `WORKSPACE_STRUCTURE.md`、tempfile root 下正常路径解析、`..` 逃逸 root 报错、`must_exist=True` 不存在报错，以及 status/metadata helper 只返回路径不创建目录或文件；不访问 `/data2` 或 full-scale 输出目录 |
| `tests/smoke/stage1_run_metadata_smoke.py` | Stage 1 P4c run metadata 临时目录 smoke | 验证 `build_run_metadata` 的基础字段、timezone-aware UTC 时间、Path 转字符串、`stage` 非空校验、`inputs/outputs/extra` 类型校验，以及 `write_run_metadata` 只在 tempfile 下写入 JSON 且可读；不访问 `/data2` 或 full-scale 输出目录 |
| `tests/smoke/stage1_protocol_types_smoke.py` | Stage 1 P5c protocol dataclass 纯内存 smoke | 从 `time_router.protocols` public API 构造全部 6 个 dataclass，验证 sample_keys/model_columns/train_splits/eval_splits tuple 保序、`extra`/`branch_specific`/`feature_schema` default_factory 独立、RouterOutput/EvaluationInput 的 logits/weights 可选组合，以及 object/list 字段原样保存且不访问 `.shape`；不创建文件、不访问 `/data2` 或正式输出目录 |
| `tests/smoke/stage1_prediction_cache_expert_provider_smoke.py` | Stage 1 P6a PredictionCacheExpertProvider 只读 smoke | 默认读取同一 4 sample packed golden fixture，构造 `PredictionCacheExpertProvider` 并显式传入 golden sample_keys；验证 `ExpertBatch` 类型、sample_keys/model_columns tuple 保序、`y_pred/y_true` shape、packed row index metadata、provider extra 轻量 metadata，并通过 `time_router.evaluation` public API 复算 hard top-1 与 raw soft fusion golden 指标；不创建正式输出目录、不访问 `/data2`、不读取 oracle/TSF |
| `tests/smoke/stage1_evaluation_input_adapter_smoke.py` | Stage 1 P6b EvaluationInput adapter smoke | 默认读取同一 4 sample packed golden fixture，先用 `PredictionCacheExpertProvider` 显式构造 `ExpertBatch`，再用 golden weights 构造 `RouterOutput` 并覆盖显式 fusion weights 输入路径；验证 `EvaluationInput` 保序、固定五专家顺序、summary golden 数值、per-sample rows 字段/数值、hard/raw-soft MAE/MSE、max_weight、weight_entropy 和 adapter `extra`；adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，并检查 `run_outputs` 一层目录集合不变，证明不重读 prediction cache/oracle/TSF、不创建正式输出目录 |
| `tests/smoke/stage1_fusion_evaluator_adapter_smoke.py` | Stage 1 P6c FusionEvaluator compat smoke | 默认读取同一 4 sample packed golden fixture，先用 `PredictionCacheExpertProvider` 显式构造 `ExpertBatch`，再用 golden weights 构造 `RouterOutput` 并调用兼容 `FusionEvaluator`；验证旧路径 `EvaluationInput` 保序、summary/rows 数值不漂移，并检查 diagnostics 中 `canonical_adapter_name=EvaluationInputAdapter`；adapter 调用阶段阻断 `open`、`Path.open` 和 `np.load`，并检查 `run_outputs` 一层目录集合不变，证明不重读 prediction cache/oracle/TSF、不创建正式输出目录 |
| `tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py` | Stage 1 P7a TimeFuseFeatureCacheProvider smoke | 使用测试内临时 feature CSV 构造 `TimeFuseFeatureCacheProvider` 并显式传入 sample_keys；验证 `FeatureBatch` 类型、sample_keys tuple 保序、`features=(2, 17)`、`feature_schema`、`extra`、空/重复 sample_key 拒绝；provider 阶段只允许读取临时 feature CSV 并阻断 `np.load`，检查 `run_outputs` 一层目录集合不变，证明不读取 prediction/oracle、不创建输出目录 |
| `tests/smoke/stage1_timefuse_linear_head_smoke.py` | Stage 1 P7b TimeFuseLinearSoftmaxHead smoke | 使用测试内固定 `FeatureBatch`、固定线性权重和 bias 构造 `RouterOutput`；验证 sample_keys 保序、model_columns 对齐、logits/weights deterministic 数值、weights 逐样本和为 1、`__call__` 与 `predict` 一致以及非法 model_columns 拒绝；head 阶段阻断文件 IO、`np.load` 和 `np.save/np.savez`，检查 `run_outputs` 一层目录集合不变，证明不读取 cache/feature CSV、不训练、不写运行产物 |
| `tests/smoke/stage1_timefuse_protocol_chain_smoke.py` | Stage 1 P7c TimeFuse protocol chain smoke | 使用 golden prediction fixture 构造 `ExpertBatch`，使用测试内临时 feature CSV 构造 `FeatureBatch`，用固定 `TimeFuseLinearSoftmaxHead` 权重生成 `RouterOutput`，再通过 `EvaluationInputAdapter` 复算 summary/rows；验证 sample_keys/model_columns/features/weights/summary/rows 保序且 deterministic；head/evaluator 阶段阻断文件 IO、`np.load` 和 `np.save/np.savez`，检查 `run_outputs` 一层目录集合不变，证明不回读 cache/oracle/TSF、不训练、不写运行产物 |

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
