# Visual Router V2 Round2 Mainline Recommendation

生成时间：2026-06-22 09:31:55 CST

## 推荐结论

当前 Round2 mainline candidate 推荐为：

- layout：`spatial_panel_3view`
- backend：`film_mean_patch_aux`
- 主指标：raw-soft MAE / MSE / regret
- 解释性指标：hard top1 / oracle-label accuracy
- 重要 fallback：Round1 `film_mean_patch_aux`
- 重要 Round2 baseline：`current_rgb_3view + film_mean_patch_aux`

该推荐不是因为 `spatial_panel_3view` 带来巨大跃迁，而是因为它在 35k screening、65k expanded validation 和 P0-scale validation 三层中方向一致，并且在 P0 frozen `pilot_test` 上相对 Round1 `film_mean_patch_aux` 同时改善 MAE、MSE 和 regret。

## 证据摘要

| scale | split | layout/backend | raw-soft MAE | raw-soft MSE | raw-soft regret | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 35k | selection | `spatial_panel_3view + film_mean_patch_aux` | 0.310385 | 3.329199 | 0.046935 | Round2c selection best |
| 35k | frozen test_small | `spatial_panel_3view + film_mean_patch_aux` | 0.398598 | 3.484102 | 0.065249 | frozen screening best |
| 65k | selection | `spatial_panel_3view + film_mean_patch_aux` | 0.307233 | 2.043914 | 0.043993 | Round2e-b selection best |
| 65k | frozen test_expanded | `spatial_panel_3view + film_mean_patch_aux` | 0.394336 | 2.008546 | 0.069543 | frozen expanded best |
| P0 | selection | `spatial_panel_3view + film_mean_patch_aux` | 0.300227 | 1.269413 | 0.033496 | P0 selection validated |
| P0 | frozen pilot_test | `spatial_panel_3view + film_mean_patch_aux` | 0.413558 | 182.771166 | 0.073273 | 相对 Round1 fallback 改善 |
| P0 | frozen pilot_test | Round1 `film_mean_patch_aux` | 0.417824 | 183.353985 | 0.077539 | 历史最强 fallback |

P0 frozen `pilot_test` 上，`spatial_panel_3view + film_mean_patch_aux` 相比 Round1 `film_mean_patch_aux` 的 delta 为：

- MAE：-0.004266
- MSE：-0.582819
- regret：-0.004266

这些收益应写作稳定小幅收益，不应写成决定性突破。

## Baseline 角色

需要区分两个 baseline：

1. Round1 `film_mean_patch_aux` 是历史最强 fallback baseline，用于判断 Round2 mainline 是否值得继续推进。
2. Round2 `current_rgb_3view + film_mean_patch_aux` 是 Round2 registry 下的 layout baseline，用于同一 Round2 imageization registry 内比较 view layout。

二者不应默认完全等价。后续建议保留 `current layout equivalence check`，确认 Round1 current layout 与 Round2 `current_rgb_3view` 的实现差异是否影响横向比较。

## Period 方向

`top3fold_period_layout` 在 period continuity diagnostic 中有价值：它相对 `current_rgb_3view` 明显降低 image、mean patch embedding 和 router weight 的扰动变化，说明 hard top1 period fold 的不连续性是真实风险。

但该 continuity 优势没有直接转化为 65k expanded 或 P0 主指标优势。35k selection 中 `top3fold_period_layout` raw-soft MAE=0.320005，弱于 `spatial_panel_3view` 0.310385；65k selection 中为 0.312709，弱于 `spatial_panel_3view` 0.307233。因此 period direction 暂时保留为 side branch，不作为当前 mainline。

后续可探索 `period_soft_mixture`、period tokens、padding mask 和 panel-wise pooling，但这些方向不应阻塞当前主线收束。

## Deferred Directions

以下方向记录为后续支线，本步不展开：

- resize/interpolation systematic ablation
- padding mask as input
- richer value/difference bands
- absolute FFT energy redesign
- period_soft_mixture
- period tokens
- independent view encoder
- panel-wise / view-region pooling
- current layout equivalence check

## 下一步

推荐进入 `Visual Router V2 Round2 full-scale validation planning`，而不是继续无限扩展 small layout 支线。

planning 阶段只做方案设计，不直接启动 full-scale feature extraction、training 或 final eval。规划需要明确 staged validation scale，例如先 1M 级别而不是直接 116M；需要 shard-aware feature cache、batch/SQLite prediction lookup、不全量加载 116M prediction manifest、多 GPU 并行、frozen eval 协议，以及与 Stage 1 canonical migration 的边界。
