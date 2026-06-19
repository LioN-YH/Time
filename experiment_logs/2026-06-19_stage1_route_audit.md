# Stage 1 路线审计与公共模块迁移候选整理

日志日期：2026-06-19 15:52:26 CST

## 目的

在不修改、移动、删除或重命名任何实验代码的前提下，审计 Stage 1 Visual Router 与 TimeFuse-style fusor 的共同实验协议、实际分叉点、正式路线、历史路线和废弃路线，并整理后续可收束的公共模块。

## 背景

现有 Stage 1 文档分别描述 visual router 主线、TimeFuse-style fusor baseline 和 full-scale reader/launcher，代码层又分别包含 SQLite、batch reader、metrics、logging 和路径处理，容易被误读为两套不同实验系统。进一步讨论确认，两边实际共享 `sample_key`、同 config 五专家 prediction cache、oracle labels、`y_pred/y_true` 和 vali/test 评估协议，主要差异应收敛为 FeatureProvider 与 RouterHead。

## 操作

1. 阅读 `WORKSPACE_STRUCTURE.md`、`HANDOFF.md`、`experiment_logs/` 中 Stage 1 相关日志、`visual_router_experiments/README.md`、`common/`、`stage1_vali_test_router/` 及其主线/README/协议/历史/reader 设计文档。
2. 枚举 `visual_router_experiments/common/` 与 `stage1_vali_test_router/` 下全部非 `__pycache__` Python 文件，共 36 个。
3. 对照文件 docstring、导入关系、README、实验日志与当前正式产物，为每个文件分配唯一主标签。
4. 新增 `docs/refactor/stage1_route_audit.md`，把当前路线描述为共享数据与评估主干，加 Visual/TimeFuse 两种特征与路由分支。
5. 新增 `docs/refactor/stage1_migration_candidates.md`，按数据平面、评估平面、训练骨架三个优先级整理迁移候选，不执行重构。
6. 创建 Git 分支 `refactor/stage1-route-audit`；本步骤只包含文档、日志和结构索引变更。

## 结果

- 正式 Visual Router 主线被确认使用在线 `x -> pseudo image -> frozen ViT -> MLP router`，full-scale 不保存长期 embedding cache。
- 正式 TimeFuse-style fusor 支线使用离线 17 维 feature cache 和 `Linear-softmax` fusor。
- 两边共享 sample/prediction/target/oracle/config/metrics 协议；现有大型工程差异主要来自数据规模、特征生命周期和独立演进。
- 36 个 Python 文件均已获得唯一主标签，没有 `UNKNOWN_NEEDS_REVIEW` 文件。
- 旧全量 Python lookup OOM、full-scale offline embedding、LogisticRegression hard router 和已停止 CPU fusor 被明确标为不可作为正式路线或正式结果。
- 后续第一优先级应抽取 manifest、prediction arrays、oracle/TSF、SQLite 和 batch reader；第二优先级统一 fusion/metrics/output/logging；最后才引入 FeatureProvider/RouterHead 训练骨架。

## 结论

Stage 1 更准确的架构是“一条共享数据与评估主干 + 两种 FeatureProvider/RouterHead”。可以明显收束工程实现，但第一轮重构不应直接合并两个已跑通的正式训练入口，应先抽公共数据与评估设施并做逐样本等价验证。

## 下一步方案

1. 审查并确认本次两份文档中的标签和迁移优先级。
2. 如启动重构，先为 packed prediction batch 建立小规模 golden fixture。
3. 每次只迁移一个公共模块，并分别用 Visual Router 与 TimeFuse smoke 验证 sample 顺序、数组 shape、指标和恢复语义不变。
4. full-scale 等价验证完成前保留现有正式入口，不提前移动或删除历史代码。
