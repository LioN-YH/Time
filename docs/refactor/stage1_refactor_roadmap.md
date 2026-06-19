# Stage 1 重构路线图

设计日期：2026-06-19

## 1. 执行原则

Stage 1 后续重构必须小步提交、先锁定行为再抽象共享模块。每一步提交都应保持 Visual Router 与 TimeFuse-style fusor 当前正式口径可回退，并且不得在同一提交里同时抽 reader、改 metrics、换训练入口和清理历史代码。

`tests/smoke/stage1_golden_smoke.py` 是所有迁移步骤的前后门禁：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

每次抽 reader、metrics、SQLite、output schema 之前必须先跑一次，确认基线绿色；迁移后再跑一次，确认 sample_key 顺序、五专家顺序、`y_pred/y_true` shape、packed row index、hard top-1 和 raw soft fusion 指标不漂移。若失败，停止后续迁移，先定位差异并写实验日志。

## 2. Commit 顺序

### P0：architecture docs only

目标：只提交架构设计和迁移路线图，不实现任何新 package，不新增空目录，不做 import 改动。

范围：

- 新增 `docs/refactor/stage1_target_architecture.md`。
- 新增 `docs/refactor/stage1_refactor_roadmap.md`。
- 同步维护 `WORKSPACE_STRUCTURE.md` 与中文实验日志。
- 运行 golden smoke 和 `git status`。

验收：

- `tests/smoke/stage1_golden_smoke.py` 通过。
- `git diff --name-only` 只包含文档、实验日志和结构说明。
- 正式代码零改动。

### P1：extract prediction batch reader

目标：抽出统一 prediction batch reader，使 Visual Router 和 TimeFuse-style fusor 能从同一批 sample key 读取五专家 `y_pred` 与共享 `y_true`。

范围：

- 读取 `per_sample_npy` 与 `packed_npy_v1`。
- 固定专家顺序 `["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]`。
- 保留 `y_true_row_index`、`y_pred_row_index` 和 manifest 指标复算校验。
- 先接入 smoke 或小规模 fixture，再接入现有入口。

门禁：

- 迁移前后运行 `tests/smoke/stage1_golden_smoke.py`。
- 对比迁移前后的 sample_key 顺序、shape、hard top-1 和 raw soft fusion。
- 不允许回退到每 sample 重复打开 packed 文件或全量 Python lookup。

### P2：extract oracle/TSF reader

目标：统一 oracle labels 与 TSF enrichment 的批量读取，明确其监督、上限和诊断用途。

范围：

- 按 `sample_key + metric` 读取 oracle。
- 按 `sample_key` 读取 TSF 元信息。
- 提供 required/diagnostic_only 缺失策略。
- 保证 oracle/TSF 不混入可部署 FeatureProvider 的 test-time 动态特征。

门禁：

- 迁移前后运行 golden smoke，确认 prediction/fusion 契约未被间接改动。
- 对小规模 oracle/TSF fixture 做 join 覆盖率和 sample_key 集合校验。
- 记录任何缺失策略变化，不用默认填充值掩盖数据问题。

### P3：extract metrics/fusion

目标：抽出统一 metrics、fusion 和报告基础，减少 `fusion_utils.py`、Visual Router、calibration 与 TimeFuse fusor 之间的重复实现。

范围：

- hard top-1、raw soft fusion、temperature/top-k fusion。
- MAE、MSE、oracle regret、selected counts、entropy、max weight。
- summary、comparison 和 per-sample prediction schema。

门禁：

- 迁移前后运行 golden smoke。
- 对已有 1k 或 pressure 输出做 schema golden comparison。
- 所有指标必须从同一批 `y_pred/y_true/weights` 复算，不能混用不同 reader 的中间结果。

### P4：extract logging/path/config

目标：收束日志、路径解析、状态落盘和配置默认值，减少脚本内硬编码。

范围：

- workspace root、输出根、manifest、prediction cache、oracle、TSF、feature cache、checkpoint 的集中解析。
- atomic JSON writer、`status.json`、metadata lineage、checkpoint index 和中文摘要。
- CLI 显式参数优先于集中默认值。

门禁：

- 迁移前后运行 golden smoke。
- 对现有 launcher 的 `status.json`、metadata 和恢复命令做兼容检查。
- 不改变既有正式输出目录含义，不破坏后台任务监控脚本。

### P5：introduce FeatureProvider interface

目标：引入显式 `FeatureProvider` 边界，把共享 reader 与路线特征生成解耦。

范围：

- `VisualFeatureProvider`：`x -> pseudo image -> frozen ViT embedding`，在线 batch 生成，不落盘 full-scale embedding。
- `TimeFuseFeatureProvider`：`sample_key -> 17维 feature cache`，支持 feature-only scaler 和 shard-aware 读取。
- 定义统一输出：`sample_keys`、`features`、dtype/device metadata 和可选诊断信息。

门禁：

- 迁移前后运行 golden smoke。
- Visual 小规模 smoke 需确认 online embedding 行为与既有入口一致。
- TimeFuse 小规模/pressure smoke 需确认 feature shape 为 17 维、split 下推和 scaler 不加载 prediction arrays。

### P6：migrate visual router and TimeFuse fusor entrypoints

目标：让两个正式入口逐步消费共享 reader、FeatureProvider、metrics/report 和 logging/path/config，但保留各自 head、loss 与实验变量。

范围：

- Visual Router 入口迁移到共享 batch reader + `VisualFeatureProvider` + Visual MLP head。
- TimeFuse fusor 入口迁移到共享 batch reader + `TimeFuseFeatureProvider` + Linear-softmax head。
- 保留 eval-only、train-only、checkpoint/resume、DataParallel 和现有 full-scale 路径兼容。

门禁：

- 每个入口迁移前后运行 golden smoke。
- 对小规模已知输出做逐样本 predictions/summary comparison。
- full-scale 只在小规模与 schema 对齐后启动；不能用重构首跑覆盖旧可引用结果。

## 3. Archive 触发条件

旧代码只能在以下条件全部满足后进入 `archive/`：

- 已有 package 化替代实现。
- golden smoke 在迁移前后均通过。
- 小规模或压力测试证明输出 schema 和关键指标等价。
- `WORKSPACE_STRUCTURE.md` 和实验日志记录归档原因、替代路径和保留口径。

优先归档候选：

- offline embedding cache；
- logistic regression fusor；
- old OOM routes；
- pilot-only scripts。

## 4. 本路线图明确不做

- P0 不实现新 package、不新增空目录、不做 import 改动。
- 任一迁移步骤都不同时改变实验协议、模型结构和正式结果引用口径。
- 不把 oracle/TSF 当作可部署 test-time 动态调权特征。
- 不把历史 OOM、legacy 或 pilot 结果重新包装为正式结果。
