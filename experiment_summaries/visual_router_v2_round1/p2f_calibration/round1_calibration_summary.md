# Visual Router V2 Round1 P2f post-hoc calibration diagnostic

生成时间：2026-06-21 19:47:28 CST

## 约束确认

- 本诊断未训练新 router，未读取 checkpoint，未重建 P2a feature，未保存 pseudo image tensor。
- Round1 prediction CSV 没有 logits，因此 temperature scaling 使用 `normalize(w ** (1/T))`；T=1、alpha=0 为原始未校准 baseline。
- calibration 参数只按 `pilot_selection` raw-soft MAE mean 选择；`diagnostic_balanced` 与 `pilot_test` 只做 frozen eval。

## 入选 calibration 参数

| variant | calibration_method | temperature | entropy_alpha | MAE_mean | MAE_std | MSE_mean | regret_to_oracle_mean | weight_entropy_mean | mean_max_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cls_mean_concat_plus_aux | power_temperature_T0p85 | 0.850000 | 0.000000 | 0.300404 | 0.001073 | 1.214417 | 0.033673 | 1.053483 | 0.544343 |
| film_cls_mean_concat_aux | power_temperature_T0p85 | 0.850000 | 0.000000 | 0.300224 | 0.001139 | 1.315696 | 0.033493 | 1.066779 | 0.535960 |
| film_mean_patch_aux | power_temperature_T0p85 | 0.850000 | 0.000000 | 0.300230 | 0.000857 | 1.295389 | 0.033498 | 1.072308 | 0.532835 |
| visual_cls_mean_concat | power_temperature_T0p85 | 0.850000 | 0.000000 | 0.301960 | 0.004137 | 1.232774 | 0.035229 | 1.044857 | 0.557873 |

## Frozen pilot_test 结果

| variant | calibration_method | MAE_mean | MAE_std | MSE_mean | MSE_std | regret_to_oracle_mean | weight_entropy_mean | mean_max_weight_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| film_mean_patch_aux | power_temperature_T0p85 | 0.417536 | 0.000669 | 183.369086 | 0.147836 | 0.077251 | 1.056659 | 0.544151 |
| film_cls_mean_concat_aux | power_temperature_T0p85 | 0.419259 | 0.001834 | 183.478104 | 0.146891 | 0.078974 | 1.048149 | 0.547219 |
| visual_cls_mean_concat | power_temperature_T0p85 | 0.441577 | 0.022351 | 247.844653 | 98.784888 | 0.101292 | 1.052236 | 0.557628 |
| cls_mean_concat_plus_aux | power_temperature_T0p85 | 0.452601 | 0.039396 | 245.478623 | 67.979818 | 0.112316 | 1.046188 | 0.551702 |

## 验收问题回答

1. `film_mean_patch_aux`：改善 MAE；入选 `power_temperature_T0p85`，pilot_test delta_MAE=-0.000288，delta_MSE=0.015101。
1. `film_cls_mean_concat_aux`：改善 MAE；入选 `power_temperature_T0p85`，pilot_test delta_MAE=-0.000309，delta_MSE=0.014259。
1. `visual_cls_mean_concat`：改善 MAE；入选 `power_temperature_T0p85`，pilot_test delta_MAE=-0.001485，delta_MSE=3.606165。
2. MSE tail：以 `film_mean_patch_aux` 为主观察，pilot_test delta_MSE=0.015101，没有下降。
3. `CrossFormer` stratum：calibrated `film_mean_patch_aux` MAE=0.637886，MSE=1.494021，regret=0.107235；delta_MAE_vs_original=0.002571，delta_MSE_vs_original=0.009384。
3. `PatchTST` stratum：calibrated `film_mean_patch_aux` MAE=0.580966，MSE=8.603557，regret=0.082461；delta_MAE_vs_original=-0.004724，delta_MSE_vs_original=0.009593。
4. selected_model ratio：`film_mean_patch_aux` CrossFormer ratio 从 original 0.042587 到 calibrated 0.042587。
5. 是否只是改变 entropy/max weight：`film_mean_patch_aux` entropy delta=-0.065573，MAE delta=-0.000288；若 MAE/MSE/regret 未同步下降，则只应视为权重形状诊断收益。
6. Round1 enhanced recommendation：不建议把 calibrated `film_mean_patch_aux` 升级为综合 enhanced recommendation；它有极小 MAE/regret 收益，但 MSE tail 没有改善，应保持 raw `film_mean_patch_aux` 为主推荐。
7. 后续路线：若本页 delta_MAE/delta_MSE 没有稳定改善，应优先进入 view layout Round2；只有 calibration 对 FiLM 主线有明确 frozen pilot_test 收益时，才值得先扩展 FiLM hyperparameter search。
