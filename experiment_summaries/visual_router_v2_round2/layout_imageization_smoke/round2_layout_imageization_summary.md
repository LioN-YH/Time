# Visual Router V2 Round2b layout imageization smoke

生成时间：2026-06-21 22:00:38 CST

## 输入与边界

- sample manifest：`/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_small_sample_manifest.csv`
- layout candidates：`/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/round2_layout_candidates.json`
- 输出目录：`/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_imageization_smoke`
- 设备：`cuda:0`
- image size：`224`
- smoke sample count：`320`
- 本步未读取 future y、专家 prediction、oracle label 或 116M prediction manifest。
- 本步未训练 router，未运行 ViT，未保存大规模 pseudo image tensor。

## 结论

- 已通过 layout：current_rgb_3view, spatial_panel_3view, line_only, line_difference_band, fft_absolute_energy, top3fold_period_layout。
- 未通过 layout：无。
- shape 检查：通过；finite 检查：通过；[0,1] range 检查：通过。
- 主 imageization 入口为 torch tensor path；debug PNG 才将少量生成后 tensor detach 到 CPU。
- `top3fold_period_layout` 当前是 channel-packed top1/top2/top3 fold，通过 registry adapter 复用 `imageize_top3fold`，不再暴露旧 `variant_b_top3fold` 参数语义。

## Layout Latency

| layout_name | batch_count | sample_count | total_time_ms | samples_per_sec | cpu_fallback |
| --- | --- | --- | --- | --- | --- |
| current_rgb_3view | 10 | 320 | 772.113096085377 | 414.4470565548026 | False |
| spatial_panel_3view | 10 | 320 | 44.172297115437686 | 7244.35949445255 | False |
| line_only | 10 | 320 | 16.58902404597029 | 19289.862930648567 | False |
| line_difference_band | 10 | 320 | 22.36432203790173 | 14308.504387375702 | False |
| fft_absolute_energy | 10 | 320 | 22.948627884034067 | 13944.18880366403 | False |
| top3fold_period_layout | 10 | 320 | 118.56762901879847 | 2698.881664819874 | False |

## Protocol 3.3 覆盖

- 插值 / resize：metadata 记录每个 layout 使用 `linear_for_1d_profile; bilinear_for_2d_panel_or_fold`，`antialias=false`，并记录 `L=96 -> H/W=224` 映射。
- value / difference bands：`line_difference_band` 使用 `first_diff=x[t]-x[t-1]`，首元素补 0；channel2 使用 `abs(first_diff)` 的窗口内 min-max band。
- padding mask：period fold/top3fold layout 记录 padding mask 可用性、padding 是否输入 ViT、pad_count 统计；本步不把 mask 输入 ViT。
- absolute FFT energy：`fft_absolute_energy` 使用 `abs(rfft(centered_x)[1:])**2` 和 `log1p(abs_energy)`；模型输入 profile 做窗口内归一化，absolute/log energy 原始统计写入 metadata。

## Protocol 3.4 后续准备

- top3fold metadata 已记录 top1/top2/top3 selected periods、period score summary 和 hard period bucket counts。
- 下一步 continuity diagnostic：对输入 x 加轻微扰动，比较 pseudo image cosine/L2 distance、ViT embedding cosine distance、router weights JS divergence、selected model flip rate，并对比 hard_top1_fold、top3fold、period_soft_mixture。

## 进入下一步条件

- 是否具备进入 35k small feature cache screening 的条件：是。
- 下一步仍需在 feature cache builder 中接入本 registry，并固定后端为 Round1 `film_mean_patch_aux` 风格：mean_patch_embedding + revin_aux FiLM。
