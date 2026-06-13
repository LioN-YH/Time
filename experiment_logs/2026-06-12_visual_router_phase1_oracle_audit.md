# Visual Router Phase 1 专家互补性 Oracle 审计

日志日期：2026-06-12 00:51:52 CST

## 目的

基于已有五模型三配置 per-item 结果，审计 frozen experts 是否存在足够互补性，判断是否值得进入 Visual Router Phase 1 的视觉路由训练。

## 背景

上一阶段已经生成五模型三配置汇总目录：

```text
experiment_logs/run_outputs/2026-06-11_230450_825063_five_model_three_config_summary/
```

该目录包含 `per_item_metrics.csv`，共有 19350 行，即 5 个模型 × 3 个配置 × 1290 个 item。当前没有窗口级 prediction cache，因此本次只计算 per-item oracle top-1，不计算 top-k 或 soft fusion。

## 操作

1. 新增脚本 `experiment_scripts/audit_visual_router_phase1_oracle.py`，脚本内含中文注释和输入/输出说明。
2. 对脚本执行语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     -m py_compile \
     experiment_scripts/audit_visual_router_phase1_oracle.py
   ```

3. 运行 oracle 审计脚本：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     experiment_scripts/audit_visual_router_phase1_oracle.py
   ```

4. 审计输出目录为：

   ```text
   experiment_logs/run_outputs/2026-06-12_005130_659243_visual_router_phase1_oracle_audit/
   ```

5. 读取输出文件并校验行数、主要字段和摘要结论。

## 输出文件

| 文件 | 功能 |
| --- | --- |
| `config_oracle_summary.csv` | 配置级 best single expert 与 oracle top-1 对比，共 6 行，覆盖 MAE/MSE |
| `tsf_cell_oracle_summary.csv` | 配置 × TSF cell × 指标的 cell 内 best single 与 oracle top-1 对比，共 48 行 |
| `tsf_cell_win_rates.csv` | 配置 × TSF cell × 指标 × 专家的 oracle 胜率和平均 regret，共 240 行 |
| `per_item_oracle_choices.csv` | 配置 × item × 指标的 oracle 选择明细，共 7740 行 |
| `summary.md` | 中文审计摘要 |
| `metadata.json` | 输入输出路径、模型列表、指标列表和 oracle 口径 |

## 结果

配置级 MAE oracle top-1 结果：

| 配置 | best single expert | best single MAE | oracle top-1 MAE | 相对收益 |
| --- | --- | ---: | ---: | ---: |
| `96_48_S` | PatchTST | 0.471869 | 0.402112 | 14.78% |
| `576_288_S` | PatchTST | 0.430489 | 0.379912 | 11.75% |
| `1024_512_S` | CrossFormer | 0.478402 | 0.402622 | 15.84% |

专家胜率差异：

- `96_48_S` 中 oracle 选择非配置级 best single 的 item 比例为 52.25%。
- `576_288_S` 中 oracle 选择非配置级 best single 的 item 比例为 36.90%。
- `1024_512_S` 中 oracle 选择非配置级 best single 的 item 比例为 54.26%。
- cell 内 best single expert 覆盖 4 个不同模型，配置级 best single 覆盖 2 个不同模型。

最大 cell 级 MAE oracle 相对收益出现在 `1024_512_S` 的 `HIGH_LOW_LOW` cell：cell best single 为 ES，MAE 为 1.146488；oracle top-1 MAE 为 0.839939；相对收益为 26.74%。

## 补充统计：总体与 cell 级收益

追加说明日期：2026-06-12 01:16:35 CST

### Oracle top-1 MAE 的含义

`oracle top-1 MAE` 指：在每个 `config_name + item_id` 上，事后查看五个专家各自的真实 MAE，选择其中 MAE 最低的那个专家，并把这些“逐 item 最优专家”的 MAE 再取平均。

它不是一个可部署模型的实际测试结果，因为真实部署时不知道哪个专家会在当前 item 上误差最低；它是一个上限审计指标，用来回答“如果 router 能完美选择单个专家，最多能比 best single expert 好多少”。因此：

- `best single MAE`：同一配置下固定使用一个全局最优专家的平均 MAE。
- `oracle top-1 MAE`：同一配置下每个 item 都选择事后最优专家的平均 MAE。
- `oracle gap abs`：`best single MAE - oracle top-1 MAE`。
- `oracle gap pct`：`oracle gap abs / best single MAE`。
- `oracle_uses_non_best_item_rate`：oracle 在多少比例的 item 上没有选择配置级 best single expert；该值越高，说明 per-item 路由有更大空间。

### 与后续 softmax 加权输出的关系

如果后续 router 输出 softmax 权重并对专家预测做加权融合，本次 `oracle top-1` 仍然有价值，但它只回答“硬选择单专家”的上限，不等价于 softmax fusion 的理论上限。

原因是当前输入只有 per-item 聚合 MAE/MSE，没有每个预测窗口、每个时间点的专家预测值和真实值。softmax 加权输出需要在预测层面计算：

```text
y_hat_fusion = sum_i w_i * y_hat_i
```

然后再用 `y_hat_fusion` 与真实值计算 MAE/MSE。仅凭每个专家的 per-item MAE，无法还原不同专家误差的方向和互补抵消关系。实际 softmax fusion 可能低于、接近或高于 top-1 hard routing：

- 如果专家误差方向互补，softmax 加权可能比任何单个专家都更好。
- 如果权重分配不准，softmax 加权也可能比 best single 或 oracle top-1 更差。
- 因此本次 oracle top-1 可以作为“专家互补性存在”的必要上限审计，但不能替代后续基于窗口级 prediction cache 的 soft fusion 审计。

后续若目标是 softmax 加权输出，应新增窗口级 prediction cache，并补充：

1. learned softmax fusion 的真实 MAE/MSE；
2. per-window 或 per-item 的 convex weight oracle；
3. hard top-1 router 与 soft fusion router 的对照。

### 配置级 MAE/MSE 收益

| metric | config_name | best_single_model | best_single_metric | oracle_top1_metric | oracle_gap_abs | oracle_gap_pct | oracle_uses_non_best_item_rate | winner_entropy_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE | 96_48_S | PatchTST | 0.471869 | 0.402112 | 0.069757 | 14.78% | 52.25% | 0.796 |
| MAE | 576_288_S | PatchTST | 0.430489 | 0.379912 | 0.050577 | 11.75% | 36.90% | 0.715 |
| MAE | 1024_512_S | CrossFormer | 0.478402 | 0.402622 | 0.075780 | 15.84% | 54.26% | 0.794 |
| MSE | 96_48_S | DLinear | 109.663325 | 108.841684 | 0.821641 | 0.75% | 67.83% | 0.638 |
| MSE | 576_288_S | PatchTST | 70.642736 | 70.322346 | 0.320390 | 0.45% | 39.61% | 0.640 |
| MSE | 1024_512_S | CrossFormer | 88.510563 | 85.683730 | 2.826832 | 3.19% | 45.35% | 0.683 |

解读：MAE 口径下 oracle gap 明显，说明逐 item 专家偏好存在足够差异；MSE 口径下总体 gap 较小，说明少数大误差 item 对 MSE 的影响更强，后续如果训练 router 以 MSE 为主目标，需要单独检查高 MSE item 的分布和专家误差方向。

### Cell 级 MAE 收益

| config_name | cluster | group_name | cell_best_single_model | cell_best_single_metric | oracle_top1_metric | oracle_gap_abs | oracle_gap_pct | winner_entropy_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 96_48_S | 0 | HIGH_HIGH_HIGH | PatchTST | 0.131323 | 0.120134 | 0.011188 | 8.52% | 0.678 |
| 96_48_S | 2 | HIGH_HIGH_LOW | DLinear | 0.483510 | 0.457148 | 0.026361 | 5.45% | 0.677 |
| 96_48_S | 6 | HIGH_LOW_HIGH | PatchTST | 0.368001 | 0.364168 | 0.003834 | 1.04% | 0.178 |
| 96_48_S | 8 | HIGH_LOW_LOW | PatchTST | 0.916855 | 0.800900 | 0.115955 | 12.65% | 0.699 |
| 96_48_S | 18 | LOW_HIGH_HIGH | DLinear | 0.390098 | 0.331526 | 0.058572 | 15.01% | 0.695 |
| 96_48_S | 20 | LOW_HIGH_LOW | DLinear | 0.475646 | 0.440967 | 0.034679 | 7.29% | 0.726 |
| 96_48_S | 24 | LOW_LOW_HIGH | PatchTST | 0.192962 | 0.174915 | 0.018047 | 9.35% | 0.383 |
| 96_48_S | 26 | LOW_LOW_LOW | ES | 0.606885 | 0.559798 | 0.047088 | 7.76% | 0.860 |
| 576_288_S | 0 | HIGH_HIGH_HIGH | PatchTST | 0.255789 | 0.231934 | 0.023855 | 9.33% | 0.738 |
| 576_288_S | 2 | HIGH_HIGH_LOW | CrossFormer | 0.532121 | 0.474075 | 0.058046 | 10.91% | 0.871 |
| 576_288_S | 6 | HIGH_LOW_HIGH | PatchTST | 0.280716 | 0.280548 | 0.000168 | 0.06% | 0.040 |
| 576_288_S | 8 | HIGH_LOW_LOW | PatchTST | 0.816441 | 0.706335 | 0.110106 | 13.49% | 0.284 |
| 576_288_S | 18 | LOW_HIGH_HIGH | CrossFormer | 0.379774 | 0.318386 | 0.061388 | 16.16% | 0.849 |
| 576_288_S | 20 | LOW_HIGH_LOW | CrossFormer | 0.437524 | 0.389991 | 0.047533 | 10.86% | 0.865 |
| 576_288_S | 24 | LOW_LOW_HIGH | PatchTST | 0.208919 | 0.190408 | 0.018511 | 8.86% | 0.296 |
| 576_288_S | 26 | LOW_LOW_LOW | PatchTST | 0.496463 | 0.484926 | 0.011537 | 2.32% | 0.468 |
| 1024_512_S | 0 | HIGH_HIGH_HIGH | CrossFormer | 0.269226 | 0.244856 | 0.024370 | 9.05% | 0.764 |
| 1024_512_S | 2 | HIGH_HIGH_LOW | CrossFormer | 0.536209 | 0.477572 | 0.058637 | 10.94% | 0.745 |
| 1024_512_S | 6 | HIGH_LOW_HIGH | PatchTST | 0.317191 | 0.301919 | 0.015272 | 4.81% | 0.448 |
| 1024_512_S | 8 | HIGH_LOW_LOW | ES | 1.146488 | 0.839939 | 0.306549 | 26.74% | 0.828 |
| 1024_512_S | 18 | LOW_HIGH_HIGH | CrossFormer | 0.330676 | 0.283532 | 0.047144 | 14.26% | 0.487 |
| 1024_512_S | 20 | LOW_HIGH_LOW | CrossFormer | 0.377541 | 0.343187 | 0.034354 | 9.10% | 0.397 |
| 1024_512_S | 24 | LOW_LOW_HIGH | PatchTST | 0.259301 | 0.234052 | 0.025248 | 9.74% | 0.451 |
| 1024_512_S | 26 | LOW_LOW_LOW | CrossFormer | 0.579915 | 0.532838 | 0.047078 | 8.12% | 0.832 |

解读：除 `576_288_S / HIGH_LOW_HIGH` 的 cell 级 oracle gap 很小外，多数 cell 存在 5% 以上的 hard routing 上限；`HIGH_LOW_LOW`、`LOW_HIGH_HIGH`、`HIGH_HIGH_LOW` 等 cell 的收益尤其明显。`winner_entropy_norm` 较高的 cell 表明 oracle 胜者分布更分散，更适合作为 router 学习专家偏好的样本；熵很低的 cell 则可能主要由单一专家主导。

## 验证

已验证：

- 输入 `per_item_metrics.csv` 行数为 19350。
- 输入无 `config_name/item_id/model` 重复键。
- MAE/MSE 无缺失值。
- 审计输出行数符合预期：配置级 6 行、cell 级 48 行、胜率 240 行、per-item oracle 7740 行。
- `summary.md` 已写明当前没有窗口级 prediction cache，因此不计算 top-k 或 soft fusion。

## 结论

当前五专家池存在足够 per-item 互补性。配置级 MAE oracle top-1 相对 best single expert 有 11.75% 到 15.84% 的收益，且不同 TSF cell 的胜率分布存在明显差异。因此可以进入 Visual Router Phase 1 的视觉路由训练准备。

## 下一步方案

1. 构造视觉结构图像或视觉特征缓存，并与 item 级 oracle label / regret label 对齐。
2. 设计 7-cell -> held-out-cell 的 zero-shot router 评估协议。
3. 若 router 只能学到 cell-level 常数策略，再补充 TSMixer/iTransformer 等专家或生成窗口级 prediction cache。
