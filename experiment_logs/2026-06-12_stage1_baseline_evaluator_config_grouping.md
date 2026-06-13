# Stage 1 Baseline Evaluator 按 Config 分层改造

日志日期：2026-06-12 23:49:22 CST

## 目的

改造 `evaluate_router_baselines.py`，使非视觉 router baseline 默认按 `config_name` 独立训练和评估，避免未来多历史-未来 config 输入时把不同输入长度、输出长度和 checkpoint 的专家误合并为同一个动作空间。

## 背景

Stage 1 主实验已确定为 per-config router。对 `96_48_S` 样本，合法候选专家只能是同 config 下的 DLinear、PatchTST、CrossFormer、ES、NaiveForecaster；不能选择 `576_288_S` 或 `1024_512_S` 的专家。因此 baseline evaluator 也必须遵守同样动作空间约束。

## 操作

1. 修改 `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py`：
   - `load_labels()` 新增对 `config_name` 字段的必需校验；
   - 新增单 config 内训练逻辑，所有 baseline rule 在每个 `config_name` 内独立从 `vali` 学习；
   - `baseline_summary.csv` 改为 config-level 主汇总；
   - 新增 `baseline_summary_by_config.csv`；
   - 新增 `baseline_summary_macro.csv`，只用于跨 config 总览；
   - dataset、TSF cell、dataset+TSF cell 分层汇总均加入 `config_name`；
   - `summary.md` 增加 per-config 说明和 macro average 表。
2. 对脚本执行语法检查：

   ```bash
   python -m py_compile visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py
   ```

3. 使用当前 `96_48_S` pilot 输入复现运行：

   ```bash
   python visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
     --labels-path experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/window_oracle_labels_with_tsf_cell.csv \
     --metric mae
   ```

4. 构造 `/tmp` 下的合成双 config labels，验证两个 config 会分别学习规则。合成 config 将 DLinear 的误差缩小，使其应被独立选为 `global_best_single`，用于检查不会沿用 `96_48_S` 的专家选择。

## 结果

当前 pilot 输出目录新增或更新：

- `baseline_summary.csv`
- `baseline_summary_by_config.csv`
- `baseline_summary_macro.csv`
- `baseline_summary_by_dataset.csv`
- `baseline_summary_by_tsf_cell.csv`
- `baseline_summary_by_dataset_tsf_cell.csv`
- `baseline_predictions.csv`
- `summary.md`

单 config `96_48_S` 结果数值保持一致：

- `global_best_single` test MAE：`1.055190`
- `oracle_top1` test MAE：`0.805392`
- 可部署 baseline 中仍是 `global_best_single` 最好

新增列和文件校验：

- `baseline_summary.csv` 包含 `config_name`；
- `baseline_summary_by_config.csv` 与 `baseline_summary.csv` 同口径；
- `baseline_summary_macro.csv` 包含 `config_count`；
- dataset / TSF cell 分层汇总均包含 `config_name`。

合成双 config 验证结果：

| config_name | `global_best_single` 选择 |
| --- | --- |
| `96_48_S` | `CrossFormer` |
| `synthetic_dlinear_best_S` | `DLinear` |

这说明 baseline evaluator 已按 config 独立学习规则，没有跨 config 共享动作空间。

## 结论

Stage 1 非视觉 baseline evaluator 已符合 per-config router 协议。后续多 config cache 输入同一个目录时，baseline 训练、预测和分层汇总会默认保留 `config_name` 约束；macro average 只作为总览，不代表可部署的混合动作空间。

## 下一步方案

1. 固定正式 cache 口径，确保多 config cache 的 `sample_key` 和 oracle labels 都带 `config_name`。
2. 开始实现 `96_48_S` visual/structure feature 或 pseudo-image tensor cache。
3. 在 `96_48_S` 上训练最小 per-config router，并与本次 baseline 同表比较。
