# Visual Router V2 Round2 Global Summary

生成时间：2026-06-22 09:31:55 CST

## 结论

在固定 `film_mean_patch_aux` 后端的前提下，Round2 的全局结论是：`spatial_panel_3view + film_mean_patch_aux` 应升级为当前 Round2 mainline candidate。

理由是三层验证方向一致：35k screening 和 65k expanded validation 均显示 `spatial_panel_3view` 是 selection 与 frozen test split 的 raw-soft MAE 最优 layout；P0-scale validation 进一步显示 frozen `pilot_test` 上 MAE/MSE/regret 均小幅优于 Round1 `film_mean_patch_aux` fallback。

该收益边界需要明确：`spatial_panel_3view` 不是决定性突破，而是稳定小幅收益。它主要支持 “view separation 减少 RGB channel mixing” 方向；不能完全解决 hard period fold discontinuity。raw-soft fusion 仍明显优于 hard top1，oracle-label accuracy 只能作为解释性诊断，不作为主指标。

## Round2 Timeline

| 阶段 | commit | 为什么做 | 输入 | 输出 | 结论 | 是否影响 mainline |
| --- | --- | --- | --- | --- | --- | --- |
| Round2a | `00d4a7b` | 建立 Round2 small screening 边界和 layout candidate registry | 35k small sample sets、Round1 最强后端判断 | `small_samples/` metadata、coverage、layout candidates、protocol | 固定 Round2 不改后端，只比较 pseudo image/view layout | 是，提供后续 screening 样本与候选集合 |
| Round2b | `7680f16` | 验证 6 个 layout 的 GPU tensor imageization 形状、数值和 latency | Round2a candidates 与 small sample | `layout_imageization_smoke/` latency、shape、period/value stats、summary | 未发现阻塞后续 screening 的 latency 或 shape 问题 | 是，确认 layout registry 可执行 |
| Round2c | `c395e54` | 在固定 `film_mean_patch_aux` 下筛选 layout | 35k small sample、6 layout、seeds 16/17/18 | `layout_screening/` comparison、delta、strata、test_small summary | `spatial_panel_3view` 是 selection 与 test_small raw-soft MAE best；推荐进入 65k | 是，首次把 spatial panel 提升为候选主线 |
| Round2d | `36741ef` | 检查 hard top1 period fold 对扰动的连续性风险 | `current_rgb_3view` 与 `top3fold_period_layout`，small selection/diagnostic samples | `period_continuity/` image/embedding/router/fused continuity、strata | `top3fold_period_layout` 明显更连续，period direction 有诊断价值 | 部分影响，支持把 top3fold 纳入 65k，但不改主线 |
| Round2d addendum | `9c2284f` | 补查 `spatial_panel_3view` 是否也受 period fold 不连续影响 | `spatial_panel_3view` 与已有 current/top3fold continuity 结果 | `period_continuity_addendum/` layout comparison、high-change examples、summary | spatial panel 仍受 hard fold 影响，但 selected flip 低于 current/top3fold，仍应进入 65k | 是，确认 spatial panel 不需被 continuity 问题挡住 |
| Round2e-a | `fede2be` | 扩展样本规模，降低 35k small screening 偶然性 | 35k small sets、expanded sample builder | `expanded_samples/` overlap、coverage、validation、metadata | 35k small sets 是 65k expanded sets 的严格子集 | 是，提供 65k validation 输入 |
| Round2e-b | `449b43a` | 验证 spatial/current/top3fold 在 65k 上是否稳定 | 65k expanded samples、3 layouts、fixed backend | `expanded_layout_validation/` selection/test/diagnostic summary | `spatial_panel_3view` 仍为 selection 与 test_expanded best；top3fold continuity 未转化为主指标优势 | 是，支持升级为 Round2 主线 |
| Round2f | `db0cd06` | 在 P0-scale 验证 spatial panel 是否优于 Round1 fallback | P0 pilot_selection/pilot_test、`spatial_panel_3view + film_mean_patch_aux` | `p0_spatial_panel_mainline/` final test、delta、strata、selected counts、metadata | frozen pilot_test MAE/MSE/regret 均小幅优于 Round1 `film_mean_patch_aux` | 是，形成本次 mainline recommendation |

## Key Results

详表见 `round2_global_key_results.csv`。核心结果如下：

| scale | split | layout/backend | raw-soft MAE | raw-soft MSE | raw-soft regret | seed mean/std | 备注 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 35k | selection | `spatial_panel_3view + film_mean_patch_aux` | 0.310385 | 3.329199 | 0.046935 | 3 seeds, MAE std 0.008199 | Round2c selection best |
| 35k | test_small | `spatial_panel_3view + film_mean_patch_aux` | 0.398598 | 3.484102 | 0.065249 | 3 seeds, MAE std 0.001161 | frozen screening best |
| 65k | selection | `spatial_panel_3view + film_mean_patch_aux` | 0.307233 | 2.043914 | 0.043993 | 3 seeds, MAE std 0.001518 | Round2e-b selection best |
| 65k | test_expanded | `spatial_panel_3view + film_mean_patch_aux` | 0.394336 | 2.008546 | 0.069543 | 3 seeds, MAE std 0.002355 | frozen expanded best |
| P0 | selection | `spatial_panel_3view + film_mean_patch_aux` | 0.300227 | 1.269413 | 0.033496 | 3 seeds, MAE std 0.000546 | P0 selection validation |
| P0 | pilot_test | `spatial_panel_3view + film_mean_patch_aux` | 0.413558 | 182.771166 | 0.073273 | 3 seeds, MAE std 0.002209 | frozen test, not used for selection |
| P0 | pilot_test | Round1 `film_mean_patch_aux` | 0.417824 | 183.353985 | 0.077539 | 3 seeds | historical fallback |

P0 frozen `pilot_test` 上，`spatial_panel_3view + film_mean_patch_aux` 相对 Round1 `film_mean_patch_aux` 的 delta 为 MAE=-0.004266、MSE=-0.582819、regret=-0.004266。Round2f strata 中 CrossFormer、PatchTST、ES、DLinear 均相对 Round1 fallback 改善，selected_model ratio 未出现单专家极端塌缩。

## Layout Conclusion

`spatial_panel_3view` 是当前 Round2 推荐 layout。35k 与 65k 上，它都同时赢得 selection 与 frozen test split；P0 frozen `pilot_test` 又给出相对 Round1 fallback 的小幅正收益。

`current_rgb_3view` 是 Round2 registry 下的 layout baseline，用于和同一 Round2 imageization registry 内其他 layout 比较。它不应被默认视为完全等价于 Round1 `film_mean_patch_aux` 的历史 current layout；后续需要做 current layout equivalence check。

`top3fold_period_layout` 在 continuity/diagnostic 上有价值，但没有转化为 65k/P0 主指标优势。`line_only`、`line_difference_band`、`fft_absolute_energy` 在 35k screening 中没有成为主指标 winner，不建议作为当前主线。

## Mechanism Interpretation

`spatial_panel_3view` 的收益更像是 view layout 表达的稳定微调，而不是后端能力跃迁。其主要机制解释是：把不同视图拆成 spatial panels，减少原始 RGB channel 混合对 ViT patch embedding 的干扰，使 `film_mean_patch_aux` 后端收到更稳定的 visual hidden representation。

这个机制不能完全解决 hard period fold discontinuity。Round2d addendum 显示 `spatial_panel_3view` 仍受 top1 period changed 影响，fold panel 仍可能成为高变化来源；只是它在主指标和 selected flip 上没有因此失去推进价值。

raw-soft fusion 在三层验证中仍优于 hard top1，因此主指标应继续看 raw-soft MAE/MSE/regret。hard top1 和 oracle-label accuracy 可用于解释 router 行为，但不能替代 soft fusion 作为主线判断。

## Period Continuity Conclusion

`top3fold_period_layout` 在扰动连续性上更好。Round2d 中它相对 `current_rgb_3view` 明显降低 image cosine distance、mean-patch embedding distance 和 router weight JS divergence，说明 hard top1 period fold 的不连续传播确实值得关注。

但 continuity 优势没有直接转化为 expanded/P0 主指标优势。35k selection 中 `top3fold_period_layout` raw-soft MAE=0.320005，弱于 `spatial_panel_3view` 0.310385；65k selection 中 `top3fold_period_layout` raw-soft MAE=0.312709，弱于 `spatial_panel_3view` 0.307233。因此 period direction 暂时保留为 side branch，不作为当前 mainline。

后续可以探索 `period_soft_mixture`、period tokens、padding mask 和 panel-wise pooling，但不应阻塞当前主线收束。这里的结论不是 period layout 已失败，而是当前 continuity 优势尚未转化为主指标优势。

## Baseline Clarification

本轮文档区分两个 baseline：

- Round1 `film_mean_patch_aux`：历史最强 fallback baseline，用于判断 Round2 是否值得继续推进。
- Round2 `current_rgb_3view + film_mean_patch_aux`：Round2 registry 下的 layout baseline，用于同 registry 内横向比较。

不要默认二者完全等价。下一步 full-scale validation planning 中建议保留 current layout equivalence check，确认 Round1 current layout 与 Round2 `current_rgb_3view` 的实现差异是否会影响横向比较。

## Deferred Directions

以下方向作为后续支线，不阻塞当前 mainline：

- resize/interpolation systematic ablation
- padding mask as input
- richer value/difference bands
- absolute FFT energy redesign
- period_soft_mixture
- period tokens
- independent view encoder
- panel-wise / view-region pooling
- current layout equivalence check

## 下一步建议

下一步推荐进入 `Visual Router V2 Round2 full-scale validation planning`，而不是继续扩展 small layout 支线。

原因是 Round2 已完成 35k、65k、P0 三层验证，`spatial_panel_3view` 收益虽小但方向稳定；继续在 small layout 上发散，边际收益不确定。更关键的问题已经变成：该方案在更大规模、更接近正式评估的设置下是否仍能保持收益。

full-scale planning 只做规划，不直接启动 feature extraction、training 或 final eval。规划需要明确 staged validation scale，例如先 1M 级别而不是直接 116M；需要 shard-aware feature cache、batch/SQLite prediction lookup、不全量加载 116M prediction manifest、多 GPU 并行、frozen eval 协议，以及与 Stage 1 canonical migration 的边界。
