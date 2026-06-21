# Visual Router V2 Round 1 Global Summary and Final Recommendation

生成时间：2026-06-21 18:43:50 CST

## 1. Round 1 protocol recap

- `pilot_train` 只用于训练 router/adapter。
- `pilot_selection` 只用于 variant、seed/epoch 规则和轻量路线选择。
- `diagnostic_balanced` 只用于 oracle-balanced 行为、selected_model ratio 和 strata 解释，不参与选择。
- `pilot_test` 只用于 frozen final eval，不用于训练、调参、variant 选择、seed 选择、epoch 选择或 hyperparameter 选择。
- 本总结是 summary-only 归档步骤：不训练新模型、不重新评估 pilot_test、不读取 checkpoint/SQLite/逐样本 prediction/feature shard。

## 2. Stage-by-stage summary

- P0 样本冻结：固定 `pilot_train`、`pilot_selection`、`diagnostic_balanced` 和 `pilot_test` 的 sample_key 边界。
- P1 Round0 复现：提供 Round0 TimeFuse、Round0 original Visual、global_best_single 和 oracle_top1 参考线。
- P2a feature cache：冻结 ViT CLS、patch-token mean pooling 和 6 维 RevIN aux，feature 来自历史窗口 `x`。
- P2probe 表征诊断：证明 visual representation 对 oracle expert、结构标签和 dataset/TSF shortcut 具有有效信号。
- P2b visual pooling：`visual_mean_patch_only` 在 selection 上成为 visual-only 初始强线，final extension 显示 `visual_cls_mean_concat` 在 pilot_test 更强。
- P2c aux-only：RevIN aux 单独作为 router 输入不足以替代 visual embedding。
- P2d direct concat：selection 上 `cls_mean_concat_plus_aux` 略优，但 pilot_test extension 显示 direct concat aux 泛化不稳。
- P2d final extension：补齐 visual-only 和 concat baseline 的 frozen pilot_test 比较，暴露 `mean_patch_plus_aux` 的明显退化。
- P2e FiLM：aux 只作为 gamma/beta modulation signal 调制 visual hidden representation，selection 上 `film_mean_patch_aux` 最优。
- P2e final extension：两个 FiLM 变体在 frozen pilot_test 上均显著优于对应 visual/direct concat baseline。

## 3. Main quantitative result

frozen `pilot_test` raw-soft 排名如下，主指标是 raw-soft MAE / MSE / regret：

| variant | raw-soft MAE | raw-soft MSE | raw-soft regret | MAE std |
| --- | ---: | ---: | ---: | ---: |
| film_mean_patch_aux | 0.417824 | 183.353985 | 0.077539 | 0.000657 |
| film_cls_mean_concat_aux | 0.419568 | 183.463846 | 0.079283 | 0.001850 |
| visual_cls_mean_concat | 0.443062 | 244.238487 | 0.102777 | 0.021419 |
| cls_mean_concat_plus_aux | 0.452942 | 245.459475 | 0.112657 | 0.039445 |
| visual_mean_patch_only | 0.452976 | 303.486492 | 0.112691 | 0.044625 |
| mean_patch_plus_aux | 0.516108 | 486.102519 | 0.175823 | 0.048081 |
| Round0 TimeFuse | 0.535220 | 568.502401 | 0.194935 | 0.000000 |
| Round0 original Visual | 0.603009 | 510.975952 | 0.262724 | 0.000000 |

`film_mean_patch_aux` 是当前 Round 1 frozen pilot_test 最强变体：raw-soft MAE=0.417824、MSE=183.353985、regret=0.077539、MAE_std=0.000657。它同时改善 MAE、MSE、regret 和 seed stability。`mean_patch_plus_aux` raw-soft MAE=0.516108/MSE=486.102519，而 `film_mean_patch_aux` 在相同 mean_patch 主表示上显著成功，说明问题不在 aux 信息本身，而在 aux 注入机制。

## 4. Interpretation

visual embedding 本身有强信号，尤其 `visual_cls_mean_concat` 在 pilot_test 上已经显著优于 Round0 TimeFuse。`mean_patch` 更像 RevIN 后形状/结构摘要，`revin_aux` 则携带尺度、波动、范围和 clip 等统计。direct concat 把尺度统计直接并入 base visual input，容易改变表征几何并放大 seed/strata tail；FiLM 将 aux 用作 hidden representation 的条件调制，更符合“aux 调制 visual representation”而不是“aux 替代 visual representation”的机制假设。

## 5. Soft fusion vs hard oracle classifier

raw-soft 是最终主指标，hard top-1 / oracle-label accuracy 只解释 router 行为。FiLM 的优势不是因为它更像 TimeFuse 那样硬选中 oracle expert：`film_mean_patch_aux` hard top-1 MAE=0.431851，仍弱于 raw-soft MAE=0.417824；其 oracle-label accuracy=0.515560，也低于 Round0 TimeFuse 的 0.587240。FiLM 的收益主要来自更健康的 soft weight 分配和更低 MSE tail，而不是 hard oracle classifier accuracy 的单点提升。

## 6. Seed stability

| variant | seed | raw-soft MAE | raw-soft MSE | raw-soft regret |
| --- | ---: | ---: | ---: | ---: |
| film_cls_mean_concat_aux | 16 | 0.419974 | 183.612512 | 0.079689 |
| film_cls_mean_concat_aux | 17 | 0.421182 | 183.422099 | 0.080897 |
| film_cls_mean_concat_aux | 18 | 0.417549 | 183.356926 | 0.077264 |
| film_mean_patch_aux | 16 | 0.418138 | 183.265267 | 0.077853 |
| film_mean_patch_aux | 17 | 0.417069 | 183.261261 | 0.076784 |
| film_mean_patch_aux | 18 | 0.418265 | 183.535428 | 0.077980 |

`film_mean_patch_aux` 三个 seed 的 pilot_test raw-soft MAE 分别为 0.418138、0.417069、0.418265，std=0.000657，明显稳定于 `visual_cls_mean_concat` 和 direct concat baseline。`film_cls_mean_concat_aux` 也改善了 `visual_cls_mean_concat` 的 MAE/MSE tail，但 MAE_std=0.001850，略弱于 mean_patch FiLM。

## 7. Strata and selected_model diagnosis

selected_model ratio 摘要：

| variant | selected_model | ratio mean | ratio min | ratio max | status |
| --- | --- | ---: | ---: | ---: | --- |
| film_cls_mean_concat_aux | PatchTST | 0.420413 | 0.339467 | 0.575867 | needs_review |
| film_cls_mean_concat_aux | ES | 0.222098 | 0.139373 | 0.360947 | needs_review |
| film_cls_mean_concat_aux | CrossFormer | 0.145080 | 0.060787 | 0.309533 | needs_review |
| film_cls_mean_concat_aux | DLinear | 0.111027 | 0.101453 | 0.117107 | stable_or_expected |
| film_cls_mean_concat_aux | NaiveForecaster | 0.101382 | 0.077133 | 0.120147 | stable_or_expected |
| film_mean_patch_aux | PatchTST | 0.344347 | 0.327267 | 0.354587 | stable_or_expected |
| film_mean_patch_aux | ES | 0.275111 | 0.100227 | 0.367360 | needs_review |
| film_mean_patch_aux | DLinear | 0.215769 | 0.123440 | 0.394920 | needs_review |
| film_mean_patch_aux | NaiveForecaster | 0.122187 | 0.112813 | 0.131667 | stable_or_expected |
| film_mean_patch_aux | CrossFormer | 0.042587 | 0.027053 | 0.069120 | stable_or_expected |

`film_mean_patch_aux` 的 hard selected_model ratio 更偏 PatchTST / ES / DLinear，CrossFormer hard selected ratio 仍偏低。selected_model ratio 与 raw-soft MAE/MSE/regret 并不完全一致：Round0 TimeFuse oracle-label accuracy 更高，但 raw-soft MAE/MSE 明显更差；FiLM 虽未解决 hard CrossFormer selection，但 soft fusion 指标最好。

重点 strata 摘要：

- `oracle_model=CrossFormer`：film_mean_patch_aux MAE=0.635316；round0_timefuse MAE=0.658768；visual_cls_mean_concat MAE=0.622065。
- `oracle_model=PatchTST`：film_mean_patch_aux MAE=0.585689；round0_timefuse MAE=0.542706；visual_cls_mean_concat MAE=0.575456。
- `oracle_model=ES`：film_mean_patch_aux MAE=0.232849；round0_timefuse MAE=0.501407；visual_cls_mean_concat MAE=0.308741。
- `oracle_model=DLinear`：film_mean_patch_aux MAE=0.628059；round0_timefuse MAE=0.625626；visual_cls_mean_concat MAE=0.618699。
- `oracle_model=NaiveForecaster`：film_mean_patch_aux MAE=0.369585；round0_timefuse MAE=0.459349；visual_cls_mean_concat MAE=0.359015。

完整 distilled strata 表见 `round1_global_strata_summary.csv`，覆盖 `oracle_model`、`error_gap_quantile`、`dataset_name`、`group_name`、`forecastability_cat`、`season_strength_cat`、`trend_strength_cat`、`cv_cat` 和 `missing_ratio_cat`。后续需要继续关注 CrossFormer hard selection 偏低、error_gap 高分位和 dataset/group 层面的 tail。

## 8. Round 1 final recommendation

- 推荐 `film_mean_patch_aux` 作为 Round1 当前主线结构。
- 保留 `film_cls_mean_concat_aux` 作为强对照结构。
- 保留 `visual_cls_mean_concat` 作为 visual-only strong baseline。
- 保留 `visual_mean_patch_only` 作为简洁 visual-only baseline。
- 不建议把 `mean_patch_plus_aux` 作为后续主线。
- 不建议继续把 direct concat aux 作为主要路线。

## 9. Next-step options

1. P2f / Round1 calibration diagnostic：做 temperature scaling / post-hoc calibration，只能在 `pilot_selection` 选择温度，`pilot_test` 继续 frozen eval。
2. P2g FiLM hyperparameter small search：只在 `pilot_train` / `pilot_selection` 做小搜索，不碰 `pilot_test`。
3. Round2 view layout / pseudo image small screening：先小样本筛选，再扩大，避免直接 full-scale 重构。
4. Stage 1 canonical migration：把 FiLM 作为 Visual Router candidate head/adapter 的重要候选，但不要立刻 full-scale 重构。

## 10. Boundary and caveats

- 本总结不改变历史 selection 规则。
- `pilot_test` 结果只用于 frozen final eval 和路线解释。
- 后续任何超参搜索都必须回到 `pilot_train` / `pilot_selection`。
- 不要用 `pilot_test` 选择 temperature、FiLM hidden dim、dropout、epoch、seed 或 variant。
- 当前结论基于 Round1 pilot sample protocol；迁移到 canonical/full-scale 前仍需保留协议边界和 calibration 诊断。
