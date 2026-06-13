# 五模型三配置 overall mean MAE 与 TSF cell 汇总

日志日期：2026-06-11 23:05:37 CST

## 目的

汇总 DLinear、PatchTST、CrossFormer、ES、SNaive 五个模型在 `96_48_S`、`576_288_S`、`1024_512_S` 三组配置下的 overall mean MAE，并按 TSF cell / cluster 统计各模型表现。

## 背景

深度学习 default baseline 和统计基线 ES/SNaive 已完成 evaluate。为了后续复盘 QuitoBench baseline，需要把五个模型放到同一口径下汇总：

- DLinear / PatchTST / CrossFormer：使用 `quito/outputs/default_baseline/` 下原始 evaluate 结果。
- ES / SNaive：使用 `quito/outputs/statistical_baseline/` 下 evaluate 结果。
- TSF cell 映射：使用 `quito/examples/datasets/cluster_data/item_clusters.csv`，共 1290 个 item、8 个 TSF cell。

## 操作

1. 新增统一汇总脚本 `experiment_scripts/summarize_five_model_three_config_results.py`。
2. 脚本读取 15 个 evaluate JSON，并与 `item_clusters.csv` 合并。
3. 生成 overall、TSF cell、per-item、checkpoint lineage 和 Markdown 汇总文件。
4. 对脚本执行语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     -m py_compile \
     experiment_scripts/summarize_five_model_three_config_results.py
   ```

5. 第一次运行因 `pandas.to_markdown()` 缺少可选依赖 `tabulate` 中断；已删除半成品目录 `experiment_logs/run_outputs/2026-06-11_230312_724906_five_model_three_config_summary/`。
6. 修改脚本为内置 Markdown 表格格式化函数，避免新增依赖；重新语法检查后正式运行成功。

正式输出目录：

```text
experiment_logs/run_outputs/2026-06-11_230450_825063_five_model_three_config_summary/
```

主要输出文件：

| 文件 | 功能 |
| --- | --- |
| `overall_mean_metrics.csv` | 15 组模型配置的整体 mean metrics |
| `overall_mean_mae_pivot.csv` | 模型 × 配置的 overall mean MAE 表 |
| `tsf_cell_metrics.csv` | 模型 × 配置 × TSF cell 的分组指标 |
| `tsf_cell_mae_pivot.csv` | TSF cell × 模型配置的 MAE 透视表 |
| `per_item_metrics.csv` | 15 组 evaluate 展开的 per-item 明细 |
| `checkpoint_lineage.csv` | evaluate 结果来源和 checkpoint 口径 |
| `summary.md` | 中文 Markdown 汇总 |
| `metadata.json` | 生成时间、输入数量、行数和口径元信息 |

## 结果

overall mean MAE：

标注规则：同一配置内 MAE 最低者加粗，第二低者下划线。

| 模型 | `96_48_S` | `576_288_S` | `1024_512_S` |
| --- | ---: | ---: | ---: |
| DLinear | <u>0.479345</u> | <u>0.491714</u> | <u>0.585809</u> |
| PatchTST | **0.471869** | **0.430489** | 0.627102 |
| CrossFormer | 0.492924 | 0.549232 | **0.478402** |
| ES | 0.629922 | 0.712879 | 0.743971 |
| SNaive | 0.615799 | 0.689365 | 0.719760 |

TSF cell mean MAE：

标注规则：同一 TSF cell、同一配置内，五个模型中 MAE 最低者加粗，第二低者下划线。

### `96_48_S`

| cluster | TSF cell | DLinear | PatchTST | CrossFormer | ES | SNaive |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | HIGH_HIGH_HIGH | <u>0.132606</u> | **0.131323** | 0.138479 | 0.274734 | 0.264973 |
| 2 | HIGH_HIGH_LOW | **0.483510** | <u>0.530564</u> | 0.553087 | 0.879123 | 0.859966 |
| 6 | HIGH_LOW_HIGH | 0.520728 | **0.368001** | <u>0.447586</u> | 0.698311 | 0.742031 |
| 8 | HIGH_LOW_LOW | 0.987834 | **0.916855** | 0.973275 | <u>0.944815</u> | 0.992463 |
| 18 | LOW_HIGH_HIGH | **0.390098** | 0.519360 | <u>0.492105</u> | 0.683083 | 0.522291 |
| 20 | LOW_HIGH_LOW | **0.475646** | 0.546529 | <u>0.537977</u> | 0.759078 | 0.707523 |
| 24 | LOW_LOW_HIGH | 0.240831 | **0.192962** | <u>0.209999</u> | 0.262826 | 0.282579 |
| 26 | LOW_LOW_LOW | 0.630465 | <u>0.612787</u> | 0.633126 | **0.606885** | 0.618134 |

### `576_288_S`

| cluster | TSF cell | DLinear | PatchTST | CrossFormer | ES | SNaive |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | HIGH_HIGH_HIGH | 0.307484 | **0.255789** | <u>0.286085</u> | 0.360682 | 0.353943 |
| 2 | HIGH_HIGH_LOW | 0.545359 | <u>0.538099</u> | **0.532121** | 0.929535 | 0.920668 |
| 6 | HIGH_LOW_HIGH | <u>0.355446</u> | **0.280716** | 0.652037 | 0.849803 | 0.846420 |
| 8 | HIGH_LOW_LOW | <u>1.066382</u> | **0.816441** | 1.177649 | 1.095209 | 1.116872 |
| 18 | LOW_HIGH_HIGH | <u>0.401033</u> | 0.435523 | **0.379774** | 0.698623 | 0.549320 |
| 20 | LOW_HIGH_LOW | 0.472594 | <u>0.459759</u> | **0.437524** | 0.782954 | 0.737465 |
| 24 | LOW_LOW_HIGH | <u>0.248504</u> | **0.208919** | 0.350858 | 0.374256 | 0.377347 |
| 26 | LOW_LOW_LOW | <u>0.581078</u> | **0.496463** | 0.602426 | 0.674305 | 0.673907 |

### `1024_512_S`

| cluster | TSF cell | DLinear | PatchTST | CrossFormer | ES | SNaive |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | HIGH_HIGH_HIGH | 0.358151 | <u>0.270670</u> | **0.269226** | 0.414078 | 0.403191 |
| 2 | HIGH_HIGH_LOW | <u>0.626666</u> | 0.640270 | **0.536209** | 0.956857 | 0.951476 |
| 6 | HIGH_LOW_HIGH | 0.409835 | **0.317191** | <u>0.323143</u> | 0.861036 | 0.861661 |
| 8 | HIGH_LOW_LOW | 1.365162 | 1.906207 | 1.191218 | **1.146488** | <u>1.163405</u> |
| 18 | LOW_HIGH_HIGH | <u>0.472354</u> | 0.536207 | **0.330676** | 0.725860 | 0.576235 |
| 20 | LOW_HIGH_LOW | <u>0.546307</u> | 0.555628 | **0.377541** | 0.807936 | 0.757701 |
| 24 | LOW_LOW_HIGH | 0.313342 | **0.259301** | <u>0.269066</u> | 0.408693 | 0.412647 |
| 26 | LOW_LOW_LOW | 0.648300 | <u>0.613434</u> | **0.579915** | 0.693894 | 0.693931 |

checkpoint 口径：

- DLinear / PatchTST / CrossFormer 均来自 `quito/outputs/default_baseline/` 原始 evaluate，checkpoint 选择为 validation MAE-best。
- ES / SNaive 是统计模型，无训练 checkpoint。
- `PatchTST 576_288_S` 另有 validation MSE-best 补评估，位于 `quito/outputs/default_baseline_mse_best/`，但本次五模型主表不采用该补评估结果。

深度学习模型具体 checkpoint 文件：

| 模型 | 配置 | checkpoint |
| --- | --- | --- |
| DLinear | `96_48_S` | `best_epoch=4_step=184130_MAE=0.356.ckpt` |
| DLinear | `576_288_S` | `best_epoch=4_step=161455_MAE=0.371.ckpt` |
| DLinear | `1024_512_S` | `best_epoch=4_step=140290_MAE=0.439.ckpt` |
| PatchTST | `96_48_S` | `best_epoch=4_step=184130_MAE=0.383.ckpt` |
| PatchTST | `576_288_S` | `best_epoch=4_step=161455_MAE=0.362.ckpt` |
| PatchTST | `1024_512_S` | `best_epoch=4_step=140290_MAE=0.405.ckpt` |
| CrossFormer | `96_48_S` | `best_epoch=4_step=184130_MAE=0.388.ckpt` |
| CrossFormer | `576_288_S` | `best_epoch=0_step=32291_MAE=0.368.ckpt` |
| CrossFormer | `1024_512_S` | `best_epoch=0_step=28058_MAE=0.309.ckpt` |

## 验证

已验证：

- 15 个 evaluate JSON 均存在。
- `per_item_metrics.csv` 行数为 19350，等于 `15 × 1290`。
- `overall_mean_metrics.csv` 行数为 15。
- `tsf_cell_metrics.csv` 行数为 120，等于 `15 × 8`。
- `group_name` 缺失数量为 0。
- TSF cell 共 8 个：`HIGH_HIGH_HIGH`、`HIGH_HIGH_LOW`、`HIGH_LOW_HIGH`、`HIGH_LOW_LOW`、`LOW_HIGH_HIGH`、`LOW_HIGH_LOW`、`LOW_LOW_HIGH`、`LOW_LOW_LOW`。

## 结论

五模型三配置的 overall mean MAE 与 TSF cell MAE 汇总已完成。当前 quick baseline 主表应使用 validation MAE-best 的深度学习 evaluate 结果；若后续要转向论文 protocol 修正版口径，需要重新明确 MSE-best、epoch、tuning 和 seeds 设置。

## 下一步方案

1. 如需论文可比口径，按 validation MSE-best、100 epoch 上限、3 seeds 和调参规则重新设计正式训练。
2. 如需报告图表，可基于 `overall_mean_mae_pivot.csv` 和 `tsf_cell_mae_pivot.csv` 生成论文式表格或热力图。
3. 若后续补充 MSE-best 或多 seed 结果，应新增独立输出目录，避免覆盖本次 MAE-best quick baseline 汇总。
