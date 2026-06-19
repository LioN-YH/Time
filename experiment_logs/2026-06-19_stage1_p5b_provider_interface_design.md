# Stage 1 P5b canonical provider interface design

日志日期：2026-06-19 21:30:07 CST

## 目的

在 P4 后 architecture pivot 和 P5a canonical runtime contract 之后，继续设计 Stage 1 canonical provider interface，明确 `ExperimentProtocol -> SplitStrategy -> ExpertProvider -> FeatureProvider -> RouterHead -> Evaluator` 的稳定边界。

## 背景

当前 Stage 1 已有 Visual Router 主线和 TimeFuse-style fusor baseline 支线。P5a 已定义未来 canonical `run_dir/status/metadata/checkpoint/evaluation/logs` 契约，但还需要进一步把实验协议、split、专家预测、特征、head 和评估拆成可扩展接口，避免把当前 frozen ViT、17 维 feature cache、固定五专家、固定 vali/test split 或离线 prediction cache 误写成长期唯一接口。

## 操作

1. 新增 `docs/refactor/stage1_provider_interface.md`，只做设计文档，不修改训练代码、不迁移入口、不实现 Python interface / abstract class。
2. 在新文档中定义 `ExperimentProtocol`、`SplitStrategy`、`ExpertProvider`、`FeatureProvider`、`RouterHead` 和 `Evaluator` 的职责、最小输入输出、共享字段和边界。
3. 明确当前 `PredictionBatchReader + packed_npy_v1`、Visual Router pseudo image / ViT feature、TimeFuse 17 维 feature cache 都只是默认实现选项；接口未来需要兼容 online expert prediction、finetune ViT、router + ViT joint training、router + expert joint training、online TimeFuse feature computation、新 head、calibration 和 evaluator 扩展。
4. 明确 oracle/TSF 只可用于监督、诊断、baseline 或分层分析，不能作为可部署 Visual Router test-time 动态调权特征。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`，把 P5 从单一 FeatureProvider interface 扩展为 canonical provider interface，并补充 P5b 当前完成范围。
6. 更新 `docs/refactor/stage1_target_architecture.md`，把目标链路改为 provider chain，并补充 feature/head/evaluator 的新边界说明。
7. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 P5b 文档和维护时间。

8. 使用 `quito` conda 环境运行指定验收命令：
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke`

## 结果

已完成 P5b 设计文档和相关索引更新。全部指定 smoke 和 compileall 均通过。本次改动范围保持为文档、结构索引和实验日志；未修改 `time_router/`、`visual_router_experiments/` 训练代码、smoke 测试代码或正式输出目录。

## 结论

Stage 1 后续 interface 设计已从“FeatureProvider 单点抽象”收束为完整 provider chain。当前 fixed config / fixed five experts / prediction cache / vali train-test eval 仍可作为默认实现，但共享接口不再绑定 frozen ViT、17 维离线 feature cache、固定 split 或固定训练方式。

## 下一步方案

小步提交并推送到远程 `refactor/stage1-route-audit` 分支。后续若进入实现阶段，应先在小规模 smoke 中写出 protocol/provider metadata，再逐步接入正式 streaming 入口。
