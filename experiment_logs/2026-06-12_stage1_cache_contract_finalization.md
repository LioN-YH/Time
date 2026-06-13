# Stage 1 正式 Cache 口径固化

日志日期：2026-06-12 23:57:09 CST

## 目的

固定 Stage 1 正式 prediction cache、oracle labels、feature cache 和 router evaluation 的字段契约，确保后续正式 cache builder、visual feature builder 和 router trainer 都遵守 per-config 动作空间约束。

## 背景

前序步骤已经确认 Stage 1 主实验必须按 `config_name` 分开训练和评估。现有 `prediction_cache_schema.py` 已包含 `config_name`、`history_length`、`pred_length` 和 `sample_key`，但还缺一份明确的正式 contract 文档，也需要补强 manifest 校验，避免 `sample_key` 与字段不一致或同一窗口下不同专家的稳定元信息不一致。

## 操作

1. 新增正式 cache contract：

   ```text
   visual_router_experiments/stage1_vali_test_router/stage1_cache_contract.md
   ```

2. 文档中固定：
   - `sample_key = config_name + split + dataset_name + item_id + channel_id + window_index`；
   - `config_name` 是动作空间边界；
   - oracle labels 只能在同一 `config_name`、同一 `sample_key` 的五专家内计算；
   - 多 config 可以同目录保存，但训练、baseline 和 summary 必须按 config 分层；
   - feature cache 必须通过 `sample_key` 与 prediction cache 和 oracle labels 对齐；
   - `oracle_top1` 只作为上限，macro average 只作为总览。
3. 修改 `visual_router_experiments/common/prediction_cache_schema.py`：
   - 在模块注释中补充 per-config 约束；
   - `validate_manifest_frame()` 新增 `sample_key` 与字段一致性校验；
   - `validate_manifest_frame()` 新增同一 `sample_key` 下稳定元信息一致性校验，包括 `config_name`、`split`、`dataset_name`、`item_id`、`channel_id`、`window_index`、`history_length` 和 `pred_length`；
   - `validate_manifest_frame()` 新增 `require_shared_y_true_path` 可选严格校验，用于正式大规模 cache 要求同一 `sample_key` 共享一个 `y_true_path`。
4. 更新 Stage 1 README、`stage1_protocol_and_plan.md`、`WORKSPACE_STRUCTURE.md` 和实验日志总览。

## 结果

Stage 1 正式 cache 口径已经有独立 contract 文档，schema 校验也能在代码层面拦截常见错误：

- `sample_key` 字符串与字段拆分不一致；
- 同一 `sample_key` 下不同专家记录的历史长度或预测长度不一致；
- 可选严格模式下同一 `sample_key` 的 `y_true_path` 不一致；
- `sample_key + model_name` 重复；
- 可选的专家集合不完整。

## 结论

正式 cache 的字段契约已经固定。后续可以开始实现 `96_48_S` visual/structure feature cache；feature cache 必须只使用历史窗口构造特征，并通过 `sample_key` 与 `window_oracle_labels_with_tsf_cell.csv` 对齐。

## 下一步方案

1. 实现 `96_48_S` visual/structure feature pilot。
2. 校验 feature cache 的 `sample_key` 覆盖 `vali/test` oracle labels。
3. 在 `96_48_S` 上训练最小 per-config router，并与非视觉 baseline 同表评估。
