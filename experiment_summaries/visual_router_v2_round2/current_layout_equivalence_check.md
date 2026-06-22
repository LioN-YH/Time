# Visual Router V2 Round2 current layout equivalence check

## 结论

结论分类：**A. Equivalent**。

在 64 个固定 `round2_selection_small` 既有样本上，Round1 current layout 与 Round2 registry `current_rgb_3view` 生成的 pseudo image tensor 完全一致：

| 对比项 | shape | dtype | range | max_abs_diff | mean_abs_diff | cosine mean | bitwise equal |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| raw pseudo image | `[64,3,224,224]` vs `[64,3,224,224]` | `torch.float32` vs `torch.float32` | `[0,1]` vs `[0,1]` | 0.0 | 0.0 | 1.000004 | true |
| encoder-normalized pixel values | `[64,3,224,224]` vs `[64,3,224,224]` | `torch.float32` vs `torch.float32` | `[-1,1]` vs `[-1,1]` | 0.0 | 0.0 | 0.999998 | true |

最佳 channel permutation 为原始顺序 `(0,1,2)`，且该顺序 bitwise equal；其他 permutation 的 mean_abs_diff 为 0.179287 到 0.354944，说明不存在隐藏 channel reorder。

因此，后续 full-scale baseline matrix 可以把 `current_rgb_3view + film_mean_patch_aux` 作为 Round2 registry baseline；Round1 `film_mean_patch_aux` 可作为历史最强 fallback baseline 和结果对照，不需要仅为 layout path 等价性重新跑一条历史 current layout。

## 审计范围

本检查只做代码路径审计和小样本 tensor 对比：

- 未训练 router；
- 未运行 ViT encoder；
- 未生成 feature cache；
- 未新建样本；
- 未读取 future `y`、专家 prediction 或 oracle label；
- 未修改 imageization 主逻辑、router head 或 Stage 1 canonical 分支。

小样本来源为 `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_small_sample_manifest.csv` 中的前 64 个 `round2_selection_small` 样本。执行环境为 `quito` conda Python：`/home/shiyuhong/application/miniconda3/envs/quito/bin/python`。

## 代码路径对照

Round1 current layout 路径：

- 入口：`visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py::forward_visual_features`
- 调用：`visual_router_experiments/common/vit_embedding_utils.py::make_pseudo_images`
- checkpoint metadata：`variant=variant_a_3view`、`norm_mode=revin_aux`、`pixel_mode=vision`、`clip=5.0`、`image_size=224`、`period_selection=fixed_candidates`
- pseudo image 构造：`visual_router_experiments/common/pseudo_imageization.py::imageize_3view`
- encoder normalization：`visual_router_experiments/common/pseudo_imageization.py::encoder_normalize`

Round2 `current_rgb_3view` registry 路径：

- 入口：`visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_layout_features.py::forward_layout_features`
- 调用：`visual_router_experiments/common/round2_layout_registry.py::imageize_round2_layout`
- layout adapter：`_layout_current_rgb_3view`
- pseudo image 构造：`_layout_current_rgb_3view -> imageize_3view`
- encoder normalization：`build_visual_router_v2_round2_layout_features.py` 在 registry 输出 `[0,1]` tensor 后调用同一个 `encoder_normalize`

## 关键实现口径

两条路径在本次检查中一致：

- 输入来源：只使用历史窗口 `x`。
- 单变量化：`_as_series_batch`；`[B,L,1]` 取最后一维，若未来多通道则按通道均值折成单序列。
- normalization：`normalize_window(..., norm_mode="revin_aux")`，窗口内 mean/std 标准化。
- period selection：`fixed_candidates`，候选周期来自 checkpoint metadata：`[2,3,4,5,6,8,10,12,16,24,32,48,64,96]`。
- period fold：`_period_fold_batch` 按 top1 period 分桶，padding 使用窗口最后一个历史值。
- resize/interpolation：line profile 使用 linear interpolation；fold 使用 bilinear interpolation；`align_corners=True`。
- channel order：`channel0=line_raster`、`channel1=top1_period_fold`、`channel2=fft_power`。
- raw tensor：`[B,3,224,224]`、`torch.float32`、range `[0,1]`。
- ViT pixel tensor：同一个 `encoder_normalize(..., preset="hf_vit_0_5")`，range `[-1,1]`。
- padding mask：只记录 metadata，不作为 ViT 输入。

## 产物

- 指标：`experiment_summaries/visual_router_v2_round2/current_layout_equivalence_metrics.csv`
- 元信息：`experiment_summaries/visual_router_v2_round2/current_layout_equivalence_metadata.json`

## 对 full-scale baseline matrix 的影响

`current_rgb_3view + film_mean_patch_aux` 可以作为 Round2 baseline 进入 full-scale validation matrix。文档中可说明它与 Round1 current layout 的 pseudo image / encoder-normalized tensor 路径等价；但 Round2 full-scale baseline 仍应以 registry 名称 `current_rgb_3view` 记录，避免把历史 Round1 结果误写成 Round2 registry 运行结果。
