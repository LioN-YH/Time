# Stage 1 Protocol And Plan

记录日期：2026-06-12 23:35:44 CST

## 目标

Stage 1 验证同分布设置下的 visual router 是否能在冻结专家之间做有效选择或融合。训练 split 为 `vali`，评估 split 为 `test`，路由粒度为 `item_id + channel_id + window_index`。

核心问题：

1. visual/structure feature 是否能预测 window-level 最优专家；
2. visual router 是否优于不使用视觉输入的 metadata/statistics baseline；
3. visual router 与 `oracle_top1` 之间还剩多少 regret；
4. 不同历史长度和预测长度 config 是否需要独立 router，以及结构表示能否跨 config 迁移。

## 主实验口径：Per-Config Router

主实验必须按 `config_name` 分开训练和评估，不允许一个 router 在不同历史-未来设置之间自由选择专家。

原因：

- `96_48_S`、`576_288_S`、`1024_512_S` 的输入长度、输出长度和 checkpoint 均不同；
- 对 `96_48_S` 样本，合法动作空间只能是同 config 下的五个专家；
- 即使某个 `1024_512_S` 专家族在结构上看起来更适合，也不能直接处理 `96_48_S` 的输入输出。

每个 config 的动作空间定义为：

```text
{DLinear, PatchTST, CrossFormer, ES, NaiveForecaster} within the same config_name
```

推荐主实验实例：

```text
router_96_48_S
router_576_288_S
router_1024_512_S
```

每个 router 独立使用该 config 的 `vali` window 训练，并在该 config 的 `test` window 上评估。

## Stage 1B：跨 Config 迁移实验

跨 config 数据可以用于迁移学习或表征学习，但不作为主实验的可部署 router。

推荐做法：

1. **Shared encoder + config-specific heads**

   ```text
   shared_visual_or_structure_encoder
   ├── router_head_96_48_S
   ├── router_head_576_288_S
   └── router_head_1024_512_S
   ```

2. **Leave-one-config-out**

   留出一个 config 不参与 encoder 预训练，用其他 config 学结构表示；在留出 config 上训练轻量 head 或少量 fine-tune，再评估 test。

   例子：

   ```text
   预训练 encoder: 576_288_S + 1024_512_S
   留出 config: 96_48_S
   留出 config 上训练: router_head_96_48_S
   留出 config 上评估: 96_48_S test
   ```

3. **不推荐作为主结论的 zero-shot head 迁移**

   直接把其他 config 训练出的完整 router head 用到留出 config，只能作为诊断实验。除非把 label 明确定义为专家族，例如 `PatchTST family -> PatchTST_96_48_S`，否则该设置不具备严格部署含义。

## 已完成

当前已完成 pilot 级别流程：

- Stage 1 代码目录和 `pilot/` 子目录整理；
- prediction cache schema 设计；
- `96_48_S` 小规模五专家 `vali/test` prediction cache pilot；
- window-level oracle label 和 regret 计算；
- TSF cell enrichment；
- 非视觉 router baseline 评估；
- baseline 输出包括整体、dataset、TSF cell、dataset+TSF cell 汇总。
- `evaluate_router_baselines.py` 已更新为按 `config_name` 独立训练 baseline，并输出 config-level 主汇总与 macro average。
- `stage1_cache_contract.md` 已固定正式 prediction cache、oracle labels、feature cache 和 router evaluation 的字段契约。

当前关键 pilot 结果：

- 可部署非视觉 baseline 中 `global_best_single` 最好，test MAE 为 `1.055190`；
- `oracle_top1` test MAE 为 `0.805392`；
- 当前 pilot 中 dataset/TSF-cell shortcut 未超过全局单专家。

## 未完成

主实验仍未完成以下部分：

1. 正式 per-config cache builder；
2. visual/structure feature 或 pseudo-image tensor 构造；
3. embedding cache；
4. per-config router 训练；
5. hard top-1 routing 评估；
6. softmax fusion 评估；
7. Stage 1B 迁移学习和 leave-one-config-out 评估；
8. 正式 summary/report 脚本。

## 下一步任务

### 1. 更新 Baseline Evaluator（已完成）

修改 `evaluate_router_baselines.py`，使其支持并默认保护 config 分层：

- 按 `config_name` 分开学习 `global_best_single`、`dataset_only`、`tsf_cell_only`、`dataset_tsf_cell` 等规则；
- 新增 `baseline_summary_by_config.csv`；
- 多 config 输入时不允许把不同 config 的专家动作空间混为一个全局选择；
- 输出整体汇总时保留 `config_name`，必要时另给 macro average。

当前实现已完成以上要求：

- `baseline_summary.csv` 与 `baseline_summary_by_config.csv` 均为按 `config_name` 的主汇总；
- `baseline_summary_macro.csv` 为跨 config macro average；
- `baseline_summary_by_dataset.csv`、`baseline_summary_by_tsf_cell.csv` 和 `baseline_summary_by_dataset_tsf_cell.csv` 均包含 `config_name`；
- 已用当前单 config pilot 和合成双 config labels 验证不会跨 config 学同一个动作空间。

### 2. 固定正式 Cache 口径（已完成）

正式 cache 中必须明确：

- `sample_key` 包含 `config_name`；
- oracle label 在同一 `config_name` 的五专家内计算；
- 不跨 config 比较 MAE/MSE 作为可部署动作；
- 多 config 可以同目录保存，但所有训练、baseline 和 summary 必须按 config 分层。

当前实现已完成以上要求：

- `stage1_cache_contract.md` 已记录 manifest、oracle labels、TSF cell enrichment、feature cache 和 router evaluation 的字段契约；
- `prediction_cache_schema.py` 已补强 `validate_manifest_frame()`，校验 `sample_key` 与字段一致性，以及同一 `sample_key` 下稳定元信息一致性；`y_true_path` 共享作为可选严格校验项。

### 3. 先跑通 `96_48_S` Visual Feature 链路

第一版不要立即扩全量。先用当前 pilot：

- 从 window 历史序列构造结构输入或 2D tensor；
- 写出 feature/embedding cache；
- 校验 feature cache 与 `window_oracle_labels_with_tsf_cell.csv` 按 `sample_key` 对齐。

### 4. 训练最小 Per-Config Router

先做 `96_48_S`：

- 输入：`vali` feature/embedding；
- 标签：同 config 内 `oracle_model`；
- 输出：五专家概率或 top-1；
- 评估：`test` hard top-1 MAE、oracle label accuracy、regret。

### 5. 同表报告

每个 config 的报告至少包含：

- `global_best_single`
- `dataset_only`
- `tsf_cell_only`
- `dataset_tsf_cell`
- `global_majority_label`
- `oracle_top1`
- `visual_router_top1`

如果 visual router 没有超过 `global_best_single`，优先诊断 feature、label 分布和 shortcut，而不是扩大规模。

### 6. 扩展到三个 Config

在 `96_48_S` 链路稳定后，扩展到：

- `576_288_S`
- `1024_512_S`

每个 config 单独训练和评估，同时保留同一套 summary schema。

### 7. 设计 Stage 1B 迁移实验

主实验稳定后再做：

- shared encoder + config-specific heads；
- freeze encoder，只训练留出 config head；
- leave-one-config-out 诊断。

## 输出要求

正式 Stage 1 输出目录继续使用：

```text
experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_*/
```

代码目录只保存源码、协议文档、README 和小型 schema 文档；prediction cache、embedding cache、checkpoint、评估结果和运行日志不得写入代码目录。
