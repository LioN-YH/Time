# Stage 1 TimeFuse 单变量结构特征与 Router Pilot

日志日期：2026-06-13 11:41:14 CST

## 目的

为 Stage 1 主线补一个轻量非视觉数值 baseline：借鉴 TimeFuse 的 meta-feature 思路，从 Quito train-normalized 历史窗口 `x` 提取单变量元特征，并用这些特征训练最小 vali->test router。该支线用于对照视觉结构 router，不作为主要研究重点。

## 背景

前序 Stage 1 已完成 96_48_S 五专家 prediction cache、window oracle labels、TSF cell enrichment 和非视觉规则 baseline。用户明确指出结构化特征不是工作重心，因此本次不做复杂特征工程，只从 `TimeFuse/meta_feature.py` 的特征中删除多变量/跨变量项，保留单变量可计算项。

保留的 TimeFuse-derived 单变量特征为：

```text
mean
std
min
max
skewness
kurtosis
autocorrelation_mean
stationarity
rate_of_change_mean
rate_of_change_std
autoreg_coef_mean
residual_std_mean
frequency_mean
frequency_peak
spectral_entropy
spectral_skewness
spectral_kurtosis
```

删除的多变量/跨变量特征为：

```text
spectral_variation
covariance_mean
covariance_max
covariance_min
covariance_std
```

## 操作

1. 阅读 `TimeFuse/meta_feature.py`，确认原始 TimeFuse meta-feature 实现。
2. 新增 `visual_router_experiments/stage1_vali_test_router/pilot/build_structure_feature_cache_pilot.py`：
   - 默认读取 `window_oracle_labels_with_tsf_cell.csv` 中 `metric=mae` 的 sample_key 清单；
   - 通过 Quito `data_config` 重新加载 vali/test 历史窗口 `x`；
   - 只从历史 `x` 提取 17 个 TimeFuse-derived 单变量元特征；
   - 输出 `feature_cache.csv`、`metadata.json` 和 `summary.md`。
3. 第一次运行时，`summary.md` 写入阶段使用 `pandas.to_markdown()`，因 Quito 环境缺少可选依赖 `tabulate` 失败。
4. 删除失败产生的半成品目录：
   - `experiment_logs/run_outputs/2026-06-13_113503_926076_visual_router_stage1_structure_feature_pilot/`
5. 修改脚本，改为手写 Markdown 表格，避免依赖 `tabulate`。
6. 使用 Quito 环境重新运行 feature cache pilot：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/pilot/build_structure_feature_cache_pilot.py
   ```

7. 新增 `visual_router_experiments/stage1_vali_test_router/pilot/train_structure_router_pilot.py`：
   - 使用 `StandardScaler + LogisticRegression(class_weight='balanced')`；
   - `StandardScaler` 和 router 都只在 vali split 上 fit；
   - test split 上预测专家名，再读取对应专家 MAE 作为 router test MAE；
   - 每个 `config_name` 独立训练，保持 per-config 动作空间边界。
8. 运行结构特征 router pilot：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/pilot/train_structure_router_pilot.py
   ```

9. 更新 `visual_router_experiments/stage1_vali_test_router/pilot/README.md` 和 `WORKSPACE_STRUCTURE.md`，记录新增 pilot 脚本。

## 结果

最终输出目录：

```text
experiment_logs/run_outputs/2026-06-13_113713_308023_visual_router_stage1_structure_feature_pilot/
```

主要输出文件：

```text
feature_cache.csv
metadata.json
summary.md
structure_router_predictions.csv
structure_router_summary.csv
structure_router_selected_model_counts.csv
structure_router_metadata.json
structure_router_summary.md
```

feature cache 校验结果：

- `feature_cache.csv` 共有 120 行、28 列；
- 覆盖 `window_oracle_labels_with_tsf_cell.csv` 中 `metric=mae` 的 120 个唯一 `sample_key`；
- 无重复 `sample_key`；
- 17 个特征列全部为 finite 数值，无 NaN/Inf；
- 分层计数为：
  - `96_48_S / vali / TEST_DATA_MIN`: 30
  - `96_48_S / vali / TEST_DATA_HOUR`: 30
  - `96_48_S / test / TEST_DATA_MIN`: 30
  - `96_48_S / test / TEST_DATA_HOUR`: 30

结构特征 router pilot 结果：

| router | config | test samples | test MAE | oracle MAE | regret | oracle label accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TimeFuse 单变量特征 + LogisticRegression | 96_48_S | 60 | 1.079743 | 0.805392 | 0.274351 | 0.466667 |

test 预测专家分布：

| selected_model | rows |
| --- | ---: |
| CrossFormer | 10 |
| DLinear | 16 |
| ES | 20 |
| NaiveForecaster | 4 |
| PatchTST | 10 |

与已有非视觉规则 baseline 对照：

- `global_best_single` test MAE = 1.055190；
- `oracle_top1` test MAE = 0.805392；
- 当前 TimeFuse 单变量结构特征 router test MAE = 1.079743，略弱于 `global_best_single`，但优于 dataset/TSF-cell 分组规则。

## 结论

TimeFuse-derived 单变量元特征 cache 和最小结构特征 router 已经打通，证明 feature cache、`sample_key` join、vali-only scaler/router 训练和 test 评估流程可用。当前结构特征 router 只是轻量对照，结果没有超过 `global_best_single`，因此不应在这条支线上继续投入复杂特征工程。

## 下一步方案

1. Stage 1 主线应转向视觉/伪图像结构表示或视觉 embedding cache，而不是继续加细结构化特征。
2. 后续 visual router 报告中可将本次 TimeFuse 单变量结构特征 router 作为一个轻量非视觉 baseline。
3. 若后续扩大 prediction cache 到更多 item/window/config，可复用这两个 pilot 脚本，但正式流程应迁出 `pilot/` 并按 `config_name` 分层。
