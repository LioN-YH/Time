# Visual Router V2 Round2 panel-wise pooling architecture probe

日志日期：2026-06-22 12:53:43 CST

## 目的

为 Round2 `spatial_panel_3view + film_mean_patch_aux` 主线补充一个轻量架构探针，确认 ViT patch embedding 是否可以按 line/fold/FFT panel region 分别 pooling，并为后续 35k screening 准备最小 feature 与训练入口。

## 背景

`spatial_panel_3view` 已把 line raster、top1 period fold 和 FFT power 三个 view 放在 224x224 图像的水平不同区域，减少 RGB channel mixing。但既有后端仍使用 global `mean_patch`，可能在 pooling 后再次混合 panel 归属信息。本窗口只做 Round2 exploration，不做 full-scale validation、不启动 1M/116M 长跑、不修改 full-scale pipeline。

## 操作

1. 从 `origin/exp/visual-router-v2-pilot` 新建并切换到 `exp/visual-router-v2-round2-exploration`。
2. 审计 `round2_layout_registry.py`，确认 `spatial_panel_3view` 在 224x224 下使用水平宽度 `[74, 74, 76]`，并在两个 panel 边界写入 2 像素 debug 白线。
3. 新增 `visual_router_v2_panel_pooling.py`，实现 224x224 / 16x16 ViT patch grid region mapping：默认忽略跨边界 patch 列 4 和 9；line/fold/FFT 三个 panel 各使用 56 个严格内部 patch，共 168 个 patch，保留 global mean_patch 作为 fallback。
4. 新增 `probe_visual_router_v2_panel_pooling.py`，固定 layout 为 `spatial_panel_3view`，运行 frozen ViT 后保存 `global_mean_patch`、`panel_mean_concat`、`global_plus_panel_mean`、`panel_variance` 和 `revin_aux`，不保存 pseudo image tensor。
5. 新增 `train_visual_router_v2_panel_pooling_probe.py`，为可选 35k probe 准备 `film_mean_patch_aux`、`film_panel_mean_aux`、`film_global_panel_mean_aux` 三个 FiLM backend 候选；aux 仍只通过 FiLM 注入。
6. 使用 quito 环境执行 `py_compile`，覆盖三个新增脚本。
7. 先跑 4 样本/集合 actual ViT smoke 验证路径，再跑正式 very-small 32 样本/集合 smoke：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_panel_pooling.py --max-samples-per-set 32 --embedding-batch-size 8 --device cuda:0 --local-files-only --output-dir /data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_probe --overwrite
```

8. 同步更新 `experiment_summaries/visual_router_v2_round2/panel_wise_pooling_architecture_probe.md`、`panel_wise_pooling_metadata.json`、Stage1 README 和 `WORKSPACE_STRUCTURE.md`。

## 结果

- 32 样本/集合 smoke 完成，输出目录为 `/data2/syh/Time/run_outputs/2026-06-22_visual_router_v2_round2_panel_pooling_probe/`。
- region mapping：image_size=224，patch_size=16，patch_grid=14x14，panel widths=`[74, 74, 76]`，ignored_patch_cols=`[4, 9]`，used_patch_count=168，ignored_patch_count=28。
- feature shape：
  - `global_mean_patch`: `[32, 768]`
  - `panel_mean_concat`: `[32, 2304]`
  - `global_plus_panel_mean`: `[32, 3072]`
  - `panel_variance`: `[32, 768]`
- finite check 为 `True`，写出 dtype 为 float32。
- global mean patch 由所有 patch token 重构的最大绝对误差为 `0.0`。
- 首个 train shard 的 panel 差异统计显示 panel means 没有完全塌缩：line/fold cosine mean=`0.807144`，fold/FFT cosine mean=`0.764465`，line/FFT cosine mean=`0.801636`；对应 L2 mean 分别为 `13.317383`、`15.520020`、`12.481323`。
- metadata 标记 `saved_pseudo_image_tensor=false`、`full_scale_validation=false`、`ran_vit=true`。

## 结论

panel-wise pooling 的基础可行性已通过 very-small actual ViT smoke：region mapping 明确、feature shape 正确、数值有限、global mean baseline 可精确复现，panel means 之间存在可观差异。基于 smoke，本方向值得进入 35k small screening；当前尚未运行 35k training，因此不能把 panel-wise pooling 写成主线结论，也不建议直接进入 65k expanded validation 或 full-scale。

## 下一步方案

1. 若继续本方向，先在 35k Round2 small 上构建 panel feature cache，再用 `train_visual_router_v2_panel_pooling_probe.py` 比较 `film_mean_patch_aux`、`film_panel_mean_aux`、`film_global_panel_mean_aux`。
2. 35k 只按 `round2_selection_small` raw-soft MAE 选择；`round2_diagnostic_balanced_small` 只诊断，`round2_test_small` 只做 frozen screening。
3. 如 35k 显示 panel variants 在 overall、CrossFormer/PatchTST strata、high-regret tail 和 selected_model ratio 上稳定优于 global mean baseline，再考虑进入 65k expanded validation。
