# Visual Router V2 Round2 current layout 等价性检查

日志日期：2026-06-22 09:50:45 CST

## 目的

确认 Round1 current layout 与 Round2 registry 下的 `current_rgb_3view` 是否可视为等价，为后续 full-scale baseline matrix 判断 `current_rgb_3view + film_mean_patch_aux` 的基线角色提供证据。

## 背景

Round1 已确认 `film_mean_patch_aux` 是当前最强后端主线。Round2 固定该后端比较 pseudo image / view layout，并推荐 `spatial_panel_3view + film_mean_patch_aux` 作为 mainline candidate。后续 full-scale validation 需要一个 current layout baseline，但不能默认 Round1 current layout 与 Round2 `current_rgb_3view` 完全一致，因此本步只做轻量代码审计和小样本 tensor 对比。

## 操作

1. 阅读用户粘贴目标文件 `/home/shiyuhong/.codex-tianyu/attachments/b0339950-8be4-4a6f-8c31-3fa769d9f804/pasted-text-1.txt`，确认本窗口不做 full-scale planning、不训练、不跑 ViT、不生成 feature cache。
2. 审计 Round1 路径 `build_visual_router_v2_round1_features.py -> vit_embedding_utils.make_pseudo_images -> pseudo_imageization.imageize_3view -> encoder_normalize`。
3. 审计 Round2 路径 `build_visual_router_v2_round2_layout_features.py -> round2_layout_registry.imageize_round2_layout(current_rgb_3view) -> _layout_current_rgb_3view -> imageize_3view -> encoder_normalize`。
4. 使用 `/home/shiyuhong/application/miniconda3/envs/quito/bin/python` 读取 Round2 small manifest 中前 64 个 `round2_selection_small` 既有样本，只加载历史窗口 `x`。
5. 对同一批 `x` 分别构造 Round1 raw pseudo image、Round1 encoder-normalized pixel values、Round2 raw pseudo image 和 Round2 encoder-normalized pixel values。
6. 输出 `experiment_summaries/visual_router_v2_round2/current_layout_equivalence_metrics.csv`、`current_layout_equivalence_metadata.json` 和中文总结 `current_layout_equivalence_check.md`。

## 结果

64 个样本上，raw pseudo image 与 encoder-normalized pixel values 均为 `[64,3,224,224]`、`torch.float32`、finite。raw range 为 `[0,1]`，encoder-normalized range 为 `[-1,1]`。

核心指标如下：

- raw pseudo image：`max_abs_diff=0.0`，`mean_abs_diff=0.0`，bitwise equal 为 true。
- encoder-normalized pixel values：`max_abs_diff=0.0`，`mean_abs_diff=0.0`，bitwise equal 为 true。
- channel permutation 检查中，最佳 permutation 是原始 `(0,1,2)`；其他 permutation 的 mean_abs_diff 为 0.179287 到 0.354944。

本步未训练 router，未运行 ViT encoder，未生成 feature cache，未新建样本，未读取 future `y`、专家 prediction 或 oracle label，未修改 imageization 主逻辑或 router head。

## 结论

结论分类为 **A. Equivalent**。Round1 current layout 与 Round2 registry `current_rgb_3view` 在本次审计覆盖的输入、normalization、period selection、fold/padding、resize、channel order、dtype、shape、range 和 encoder normalization 路径上等价，且小样本 tensor 对比证明 bitwise identical。

## 下一步方案

后续 full-scale baseline matrix 可以使用 `current_rgb_3view + film_mean_patch_aux` 作为 Round2 registry baseline；Round1 `film_mean_patch_aux` 可作为历史最强 fallback baseline 和结果对照，不必仅为 layout path 等价性重复跑历史 current layout。full-scale 文档仍应使用 Round2 registry 名称 `current_rgb_3view` 记录新 baseline，避免把历史结果误写成同一次 registry 运行结果。
