# Visual Router V2 Round2b layout registry / GPU tensor imageization smoke

日志日期：2026-06-21 22:02:31 CST

## 目的

实现并验证 Visual Router V2 Round2b 的最小闭环：通过统一 layout registry 生成六个默认 layout 的 ViT-compatible pseudo image tensor，确认 shape、dtype、value range、finite、padding/period metadata 和 latency，为下一步 35k small feature cache screening 做准备。

## 背景

Round2 small screening 已冻结 `round2_train_small=20000 vali`、`round2_selection_small=5000 vali`、`round2_diagnostic_balanced_small=5000 vali` 和 `round2_test_small=5000 test`。Round1 当前推荐后端为 `film_mean_patch_aux` 风格，即后续 screening 固定使用 mean patch visual input 与 RevIN aux FiLM modulation，本轮只改变 pseudo image / view layout 前端。

本步要求不生成完整 35k feature cache、不训练 router、不运行 ViT、不读取 future y、不读取专家 prediction 或 oracle label 作为 imageization 输入、不读取 116M prediction manifest。

## 操作

1. 阅读用户提供的 Round2b 目标文件，确认本步范围是 layout registry / GPU tensor imageization smoke。
2. 复核已有 tensor-first imageization 代码：
   - `visual_router_experiments/common/pseudo_imageization.py`
   - `visual_router_experiments/common/vit_embedding_utils.py`
   - Round2 small sample 输出 `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_small_samples/`
3. 新增 `visual_router_experiments/common/round2_layout_registry.py`，登记并实现：
   - `current_rgb_3view`
   - `spatial_panel_3view`
   - `line_only`
   - `line_difference_band`
   - `fft_absolute_energy`
   - `top3fold_period_layout`
   - deferred stub：`period_soft_mixture`、`independent_view_encoder`
4. 新增 `visual_router_experiments/stage1_vali_test_router/smoke_visual_router_v2_round2_layout_imageization.py`，默认从四个 Round2 sample set 取 `128/64/64/64` 个样本，只读取历史窗口 x，输出 CSV/JSON/Markdown 和少量 debug thumbnails。
5. 使用 Quito 环境执行语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/common/round2_layout_registry.py visual_router_experiments/stage1_vali_test_router/smoke_visual_router_v2_round2_layout_imageization.py
   ```

6. 先运行 4 样本 quick smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/smoke_visual_router_v2_round2_layout_imageization.py --output-dir /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_imageization_smoke_quick --max-samples-per-set 1 --batch-size 4 --device cuda:0 --save-debug-thumbnails --debug-thumbnail-count 1 --overwrite
   ```

   首次 quick smoke 的 imageization 主流程成功，但写 Markdown summary 时因 `pandas.to_markdown()` 缺少可选依赖 `tabulate` 失败；随后改为脚本内简单 Markdown 表格渲染，不新增依赖，并重跑 quick smoke 通过。

7. 运行默认 320 样本 smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/smoke_visual_router_v2_round2_layout_imageization.py --output-dir /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_imageization_smoke --batch-size 32 --device cuda:0 --save-debug-thumbnails --debug-thumbnail-count 8 --overwrite
   ```

8. 更新 `visual_router_experiments/common/README.md`、`visual_router_experiments/stage1_vali_test_router/README.md` 和 `WORKSPACE_STRUCTURE.md`，登记新 registry、smoke 入口、输出目录和轻量 summary 目录。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_imageization_smoke/
```

轻量 summary 副本：

```text
experiment_summaries/visual_router_v2_round2/layout_imageization_smoke/
```

生成文件包括：

- `round2_layout_imageization_smoke_results.csv`
- `round2_layout_imageization_latency.csv`
- `round2_layout_imageization_value_stats.csv`
- `round2_layout_imageization_shape_check.csv`
- `round2_layout_imageization_period_stats.csv`
- `round2_layout_imageization_metadata.json`
- `round2_layout_imageization_summary.md`
- `round2_layout_imageization_smoke_samples.csv`
- `debug_thumbnails/<layout_name>/*.png`

正式 smoke 样本数为 320：

| sample_set | count |
| --- | ---: |
| `round2_train_small` | 128 |
| `round2_selection_small` | 64 |
| `round2_diagnostic_balanced_small` | 64 |
| `round2_test_small` | 64 |

六个默认 layout 均通过：

| layout | sample_count | shape | dtype | device | finite_ratio | range |
| --- | ---: | --- | --- | --- | ---: | --- |
| `current_rgb_3view` | 320 | `[B,3,224,224]` | `torch.float32` | `cuda:0` | 1.0 | `[0,1]` |
| `spatial_panel_3view` | 320 | `[B,3,224,224]` | `torch.float32` | `cuda:0` | 1.0 | `[0,1]` |
| `line_only` | 320 | `[B,3,224,224]` | `torch.float32` | `cuda:0` | 1.0 | `[0,1]` |
| `line_difference_band` | 320 | `[B,3,224,224]` | `torch.float32` | `cuda:0` | 1.0 | `[0,1]` |
| `fft_absolute_energy` | 320 | `[B,3,224,224]` | `torch.float32` | `cuda:0` | 1.0 | `[0,1]` |
| `top3fold_period_layout` | 320 | `[B,3,224,224]` | `torch.float32` | `cuda:0` | 1.0 | `[0,1]` |

Latency 结果：

| layout | total_time_ms | samples_per_sec | cpu_fallback |
| --- | ---: | ---: | --- |
| `current_rgb_3view` | 772.113096 | 414.447057 | false |
| `spatial_panel_3view` | 44.172297 | 7244.359494 | false |
| `line_only` | 16.589024 | 19289.862931 | false |
| `line_difference_band` | 22.364322 | 14308.504387 | false |
| `fft_absolute_energy` | 22.948628 | 13944.188804 | false |
| `top3fold_period_layout` | 118.567629 | 2698.881665 | false |

协议覆盖结果：

- 插值 / resize 已写入 metadata：1D profile 使用 linear interpolation，2D fold/panel 使用 bilinear interpolation，`antialias=false`，并记录 `L=96 -> H/W=224`。
- `line_difference_band` 明确使用 `first_diff=x[t]-x[t-1]`，首元素补 0；channel2 使用 `abs(first_diff)` 的窗口内 min-max band。
- period fold/top3fold layout 记录 padding mask 可用性、是否输入 ViT、pad count 和 padded sample ratio；本步不把 mask 输入 ViT。
- `fft_absolute_energy` 使用 `abs(rfft(centered_x)[1:])**2` 和 `log1p(abs_energy)`；模型输入做窗口内 profile 归一化，absolute/log energy 原始统计写入 metadata。
- `top3fold_period_layout` 当前是 channel-packed top1/top2/top3 fold，通过 registry adapter 复用 `imageize_top3fold`，不再绑定旧 `variant_b_top3fold` CLI 语义；metadata 记录 selected periods、period score summary 和 top1 hard period bucket counts。

## 结论

Round2b layout registry / GPU tensor imageization smoke 已完成。六个默认 layout 均能通过统一 registry 生成 ViT-compatible `[B,3,224,224]` tensor，输出 finite 且 range 在 `[0,1]`。主路径使用 torch tensor 操作，没有 matplotlib/seaborn/PIL 逐样本画图；PIL 只用于少量 debug thumbnails。metadata 已覆盖 protocol 3.3 和 3.4 要求。

当前结果具备进入下一步 35k small feature cache screening 的条件，但下一步仍需在 feature cache builder 中接入本 registry，并固定后端为 Round1 `film_mean_patch_aux` 风格。

## 下一步方案

1. 新增或改造 Round2 small feature cache builder，使其按 layout 参数调用 `round2_layout_registry.py`，并生成 frozen ViT mean_patch embedding 与 RevIN aux。
2. 只在 35k Round2 fixed samples 上生成 small feature cache，不保存 pseudo image tensor。
3. 后端固定为 `film_mean_patch_aux` 风格：base visual input 为 mean_patch_embedding，condition input 为 revin_aux，aux 通过 FiLM gamma/beta 调制 visual hidden representation。
4. 后续 continuity diagnostic 对输入 x 加轻微扰动，比较 pseudo image cosine/L2、ViT embedding cosine distance、router weights JS divergence、selected model flip rate，并对比 hard_top1_fold、top3fold 和 deferred `period_soft_mixture`。
