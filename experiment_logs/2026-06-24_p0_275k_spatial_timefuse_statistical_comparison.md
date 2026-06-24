# P0/275k spatial_panel_3view、TimeFuse 与统计基线全面对比

日志日期：2026-06-24 07:08:34 CST

## 目的

在 P0 275k 抽样数据上，补齐当前优化后的 `spatial_panel_3view + film_mean_patch_aux`、TimeFuse full-scale baseline 和非视觉统计基线的直接横向对比，并同时输出完整 275k、sample_set、TSF cell、dataset+TSF cell 分层结果。

## 背景

另一个窗口正在跑更大规模的 Visual Router fullscale 实验，但当前需要尽快得到更全面的可引用对比数据。此前已有 P0 spatial panel mainline 结果和 TimeFuse full-scale checkpoint，但旧日志只核查过 selection/test 级别对比，还缺少在完整 P0 275k 子集上统一评估 TimeFuse、统计 policy 与 TSF cell 分层输出。

## 操作

1. 复用脚本 `visual_router_experiments/stage1_vali_test_router/compare_p0_275k_spatial_timefuse_statistical.py`。
2. 输入 P0 manifest：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline/inputs/p0_sample_manifest.csv`。
3. 输入 P0 spatial 结果目录：`/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_p0_spatial_panel_mainline`。
4. 输入 TimeFuse full-scale checkpoint：`/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/checkpoints/latest_timefuse_fusor.pt`。
5. 输入 prediction cache：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache`。
6. 使用命令：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/compare_p0_275k_spatial_timefuse_statistical.py --overwrite --device auto --batch-size 2048 --output-dir /data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison
   ```

7. 运行完成后读取 `status.json`、`p0_275k_comparison_metadata.json`、`p0_275k_comparison_summary.md` 和 seed 聚合 CSV，核验完成状态与关键指标。

## 结果

1. 运行目录 `/data2/syh/Time/run_outputs/2026-06-24_p0_275k_spatial_timefuse_statistical_comparison/` 的 `status.json` 显示 `status=completed`，更新时间为 `2026-06-24 07:07:12 CST`。
2. 样本总数为 275000，sample_set 构成为 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000`、`pilot_test=75000`。
3. 输出文件包括：
   - `p0_275k_overall_comparison.csv`
   - `p0_275k_overall_seed_aggregated_comparison.csv`
   - `p0_275k_sample_set_comparison.csv`
   - `p0_275k_sample_set_seed_aggregated_comparison.csv`
   - `p0_275k_tsf_cell_comparison.csv`
   - `p0_275k_tsf_cell_seed_aggregated_comparison.csv`
   - `p0_275k_sample_set_tsf_cell_comparison.csv`
   - `p0_275k_sample_set_tsf_cell_seed_aggregated_comparison.csv`
   - `p0_275k_dataset_tsf_cell_comparison.csv`
   - `p0_275k_dataset_tsf_cell_seed_aggregated_comparison.csv`
   - `p0_275k_statistical_policy_mapping.csv`
   - `p0_275k_comparison_summary.md`
4. 完整 275k seed 聚合关键结果：
   - oracle top1：MAE `0.288582`，MSE `49.865488`。
   - `spatial_panel_3view_raw_soft_fusion`，seeds `16;17;18`：MAE `0.331642 ± 0.000931`，MSE `50.614281 ± 0.037746`，regret `0.043060`，oracle label accuracy `0.538656`。
   - `spatial_panel_3view_hard_top1`，seeds `16;17;18`：MAE `0.349132 ± 0.001113`，MSE `50.705945 ± 0.036831`，regret `0.060550`。
   - `timefuse_style_raw_soft_fusion`，`fullscale_epoch1_seed16`：MAE `0.379599`，MSE `155.894901`，regret `0.091017`，oracle label accuracy `0.577473`。
   - `timefuse_style_hard_top1`，`fullscale_epoch1_seed16`：MAE `0.395898`，MSE `155.968954`，regret `0.107316`。
   - 最强非视觉统计 policy 为 `dataset_tsf_cell_best_single`：MAE `0.424164`，MSE `225.843197`，regret `0.135581`。
   - `global_best_single`：MAE `0.429125`，MSE `87.112758`，regret `0.140543`。
   - 单统计专家 `single_ES` / `single_NaiveForecaster`：MAE `0.607955` / `0.590661`。
5. 按 TSF cell 的 MAE 结果显示，`spatial_panel_3view_raw_soft_fusion` 在 8 个 group 上均优于 TimeFuse raw-soft 和 `dataset_tsf_cell_best_single`：
   - `HIGH_HIGH_HIGH`：spatial `0.111480`，TimeFuse `0.116510`，dataset+TSF policy `0.129929`。
   - `HIGH_HIGH_LOW`：spatial `0.353192`，TimeFuse `0.368072`，dataset+TSF policy `0.411012`。
   - `HIGH_LOW_HIGH`：spatial `0.326182`，TimeFuse `0.333673`，dataset+TSF policy `0.344877`。
   - `HIGH_LOW_LOW`：spatial `0.653743`，TimeFuse `1.017122`，dataset+TSF policy `1.231495`。
   - `LOW_HIGH_HIGH`：spatial `0.325721`，TimeFuse `0.355627`，dataset+TSF policy `0.378658`。
   - `LOW_HIGH_LOW`：spatial `0.411495`，TimeFuse `0.445748`，dataset+TSF policy `0.489853`。
   - `LOW_LOW_HIGH`：spatial `0.165485`，TimeFuse `0.174369`，dataset+TSF policy `0.179270`。
   - `LOW_LOW_LOW`：spatial `0.488747`，TimeFuse `0.524140`，dataset+TSF policy `0.585333`。
6. 按 sample_set 看，`spatial_panel_3view_raw_soft_fusion` 在 `pilot_train`、`pilot_selection`、`diagnostic_balanced`、`pilot_test` 上 MAE 均优于 TimeFuse raw-soft；尤其 frozen `pilot_test` 为 spatial `0.413558` vs TimeFuse `0.535220`。

## 结论

本次已完成用户要求的 P0/275k 全面横向对比。完整 275k 上，优化后的 `spatial_panel_3view + film_mean_patch_aux` raw-soft 三 seed 平均 MAE 明显优于 TimeFuse full-scale epoch1 seed16 和统计 policy；按 TSF cell 分层后，spatial raw-soft 在 8 个 group 的 MAE 也全部优于 TimeFuse 与 `dataset_tsf_cell_best_single`。该结果可作为 fullscale 长跑完成前的较全面中间证据，但仍应标注其样本范围是 P0 275k subset，不是 23,275,170 full-scale 全候选窗口。

## 下一步方案

1. 对外引用时优先使用 `p0_275k_overall_seed_aggregated_comparison.csv` 和 `p0_275k_tsf_cell_seed_aggregated_comparison.csv`，避免只引用单 seed spatial 行。
2. 如果需要更细分论证，可继续读取 `p0_275k_dataset_tsf_cell_seed_aggregated_comparison.csv` 和 `p0_275k_sample_set_tsf_cell_seed_aggregated_comparison.csv`。
3. 等另一个窗口的 fullscale spatial 实验完成后，再把本次 P0/275k 对比与真正 fullscale 对比区分写入最终报告。
