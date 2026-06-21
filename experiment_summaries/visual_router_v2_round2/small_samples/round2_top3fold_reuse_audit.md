# Round2 top3fold 复用审计

生成时间：2026-06-21 21:18:56 CST

## 结论

已有 top3fold 实现可复用，路径为 `visual_router_experiments/common/pseudo_imageization.py`。当前实现已经支持从历史窗口 `x` 计算 FFT top-k 周期，并把 top1/top2/top3 period fold 作为 3 个 ViT 输入通道输出。

## 已有实现位置

- `select_fft_periods(x, top_k=3, period_candidates=None)`：基于历史窗口 FFT power 选择每个样本的 top-k 周期；支持动态 FFT top-k 和固定候选周期桶。
- `make_default_period_candidates(history_length, device=...)` / `parse_period_candidates(...)`：用于大规模 online 路径的固定候选周期解析。
- `_fold_fixed_period_batch(series, period, image_size, pixel_mode, clip)`：按指定周期对一批序列进行 padding、fold 和双线性 resize。
- `_period_fold_batch(series, periods, period_column, image_size, pixel_mode, clip)`：按每个样本选中的周期分桶批量 fold。
- `imageize_top3fold(x, image_size=224, periods=None, period_candidates=None, pixel_mode="vision", clip=5.0)`：构造 `[B, 3, H, W]` top3fold 伪图像，三个 channel 分别为 top1/top2/top3 fold。
- `visual_router_experiments/common/vit_embedding_utils.py::make_pseudo_images(...)`：当前仅通过 `variant_b_top3fold` 暴露 top3fold 入口。

## 输入输出

- 输入：历史窗口 tensor，支持 `[B, L, 1]` 或可被 `_as_series_batch` 归一成 `[B, L]` 的形式；Round1 正式路径先执行 RevIN-style window normalization。
- 输出：`imageize_top3fold` 返回 `[B, 3, image_size, image_size]`，数值范围 `[0, 1]`；进入 ViT 前由 `encoder_normalize(..., preset="hf_vit_0_5")` 标准化。
- 数据边界：只使用历史 `x`，不读取未来 `y`、专家预测或 oracle 标签作为输入特征。

## Round2 复用边界

- 可直接复用的部分：FFT 周期选择、固定候选周期、period fold 批处理、ViT-compatible tensor 输出。
- 需要补的部分：为 Round2 layout screening 增加 layout registry/adapter，使 `top3fold_period_layout` 与 `current_rgb_3view`、`spatial_panel_3view`、`line_only` 等候选共用同一 feature-cache 入口参数。
- 不建议本步做的部分：不要在 small sample builder 中生成 pseudo image tensor，不要跑 frozen ViT，不要把 top3fold 与 independent view encoder 同时作为一个混合因素测试。

## 风险

- hard top-k period 在周期估计不稳时可能放大噪声；
- top-k 周期本身可能形成 dataset shortcut；
- 当前 `variant_b_top3fold` 是 channel-packed 设计，若 Round2 想测试 spatial panel top3fold，需要新增 layout，而不是把它混同为已有实现。
