# Visual Router V2 Round2 one-shard staged validation

日志日期：2026-06-22 12:54:14 CST

## 目的

把上一轮 staged full-scale smoke 从每个 sample set 32 条样本推进到单个真实 full-scale shard 的 one-shard staged validation，验证更大切片下 sample manifest、feature cache、prediction lookup、fixed FiLM train/eval 和 report schema 是否稳定。

## 背景

当前分支是 Visual Router V2 Round2 实验优化分支。本轮只运行主线 `spatial_panel_3view + film_mean_patch_aux` 和必需 baseline `current_rgb_3view + film_mean_patch_aux`，不加入新 layout search、period branch、calibration、panel-wise pooling、independent view encoder 或 1M/116M 长跑。

上一轮 `2026-06-22_visual_router_v2_round2_staged_fullscale_validation_thin_slice.md` 已完成 staged smoke：四个 staged sample set 各 32 条、两个 layout、seed 16、epochs 1，feature manifest 和 128 sample_key × 5 experts prediction lookup 均通过。

## 操作

1. 使用 `quito` 环境运行 one-shard dry-run，输出目录为 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_one_shard_staged_validation/`，轻量 summary 目录为 `experiment_summaries/visual_router_v2_round2/one_shard_staged_validation/`。
2. 确认 4 张 RTX 3090 空闲后，运行 `launch_visual_router_v2_round2_staged_validation_parallel.py --sample-scale one_shard --layouts spatial_panel_3view,current_rgb_3view --backend film_mean_patch_aux --seeds 16 --epochs 1 --local-files-only --overwrite`。
3. 运行后使用 `pandas`、`sqlite3` 和 `numpy` 复核 sample manifest、feature manifest、SQLite prediction index、reports 和 metadata。
4. 修正 one-shard 轻量 summary 的下一步说明，避免继续写成“下一步运行 one_shard”。

## 结果

1. Dry-run 通过，命令矩阵只包含 `spatial_panel_3view,current_rgb_3view`、`film_mean_patch_aux`、seed 16、epochs 1 和 `sample_scale=one_shard`。
2. 正式 one-shard execution 完成，launcher 状态为 `completed`。
3. Sample manifest 验证通过：总样本数 `2048`，unique sample_key `2048`；`staged_train=512`、`staged_selection=512`、`staged_diagnostic=512`、`staged_test=512`；前三个 set 均来自 `vali`，`staged_test` 来自 `test`；四个 set 内 `order_index` 均从 0 连续；`oracle_model`、`error_gap`、`error_gap_quantile` 缺失数为 0。
4. Feature cache 验证通过：unified feature manifest 行数 `64`，覆盖两个 layout 和四个 sample_set；每个 layout × sample_set 的 `sample_count=512`；feature shard 文件均存在；抽检 shard 中 `cls_embedding=(64,768)`、`mean_patch_embedding=(64,768)`、`revin_aux=(64,6)`，均为 `float32` 且 finite；未发现 pseudo image tensor 持久化目录或文件。
5. Prediction lookup 验证通过：SQLite 表 `prediction_index` 的 record_count 为 `10240 = 2048 × 5`，unique_samples 为 `2048`；CrossFormer、DLinear、ES、NaiveForecaster、PatchTST 各 `2048` 条；构建过程 RSS 约 750-780MB，未出现全量 116M manifest 常驻内存膨胀。
6. Fixed FiLM train/eval 完成：`spatial_panel_3view_seed16` 和 `current_rgb_3view_seed16` 均返回 0；两个 layout 均生成 checkpoint、selection/diagnostic/test prediction CSV、seed results 和 task metadata。
7. Report schema 稳定：overall report 6 行，strata report 96 行，tail report 6 行，router behavior report 6 行；metadata 明确记录 `not_1m_run=true`、`not_116m_full_scale_run=true`、`loaded_116m_prediction_manifest_to_memory=false`、`saved_pseudo_image_tensor=false`。
8. one-shard 指标只作为 pipeline validation，不作为最终性能结论：`staged_test` raw-soft MAE/MSE/regret 为 spatial `0.232323/0.768080/0.037208`，current `0.239509/0.772025/0.044395`。

## 结论

`--sample-scale one_shard` 已能稳定构建 staged manifest；每个 sample_set 512 条时 feature cache 正确生成；subset SQLite prediction lookup 覆盖所有 staged samples × 5 experts；两个 layout 的 seed16 fixed FiLM train/eval 完整跑通；overall / strata / tail / router behavior report schema 在 one-shard 规模下稳定。

本轮具备进入 multi-shard / 1M staged validation 设计和小规模预检的条件，但不应直接启动 116M full-scale。进入 1M 前需要先把 sample builder 从单 shard 扩展为 shard list，并继续保持 subset SQLite、feature shard、train/selection/diagnostic/test 分离和 test frozen eval 约束。

## 下一步方案

1. 设计 multi-shard staged manifest 参数，例如 shard list、每 shard 每 set 抽样数和总样本上限。
2. 先做 multi-shard dry-run 和较小样本数 smoke，确认 feature cache 与 subset SQLite 不因 shard list 扩展破坏 schema。
3. 再启动 1M staged validation；仍只比较 `spatial_panel_3view` 与 `current_rgb_3view`，后端固定 `film_mean_patch_aux`。
4. 继续监控 CrossFormer/PatchTST strata、`LOW_LOW_HIGH` group、q4/q5 error_gap、high-regret tail 中 PatchTST selected ratio 和 raw-soft/hard top1 gap。
