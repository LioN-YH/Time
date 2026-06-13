# Stage 1 在线伪图像化 Pilot

日志日期：2026-06-13 13:42:23 CST

## 目的

实现并验证一个 online、GPU-first、tensor-first 的伪图像化 pilot，为后续 Visual Router 的冻结视觉 encoder embedding 和 router 训练做输入准备。

## 背景

Stage 1 已经有扩大版五专家 window oracle pilot，输出 `window_oracle_labels_with_tsf_cell.csv`，覆盖 `96_48_S` 下 vali/test、`TEST_DATA_MIN`/`TEST_DATA_HOUR` 的 120 个 `metric=mae` sample_key。下一步需要从 Quito train-based normalized 历史窗口 `x` 在线构造视觉输入，但第一版只验证图像化，不训练 router，也不保存全量图像 tensor cache。

## 操作

1. 新增公共模块 `visual_router_experiments/common/pseudo_imageization.py`：
   - `normalize_window(x, norm_mode)` 支持 `quito`、`revin`、`revin_aux`；
   - `select_fft_periods(x, top_k=3)` 基于历史窗口 FFT power 选择周期，并对重复周期做确定性 fallback；
   - `imageize_3view()` 生成 `line_raster`、`top1_period_fold`、`fft_power` 三个语义通道；
   - `imageize_top3fold()` 生成 top1/top2/top3 FFT period fold 三个语义通道；
   - `to_vision_pixels()` 将标准化张量 clamp 到 `[-clip, clip]` 后映射到 `[0, 1]`；
   - `imagenet_normalize()` 仅用于 frozen ViT/MAE/CLIP encoder 前的 ImageNet mean/std normalization。
2. 新增 pilot 脚本 `visual_router_experiments/stage1_vali_test_router/pilot/build_online_pseudo_image_pilot.py`：
   - 默认读取 `experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/window_oracle_labels_with_tsf_cell.csv`；
   - 只筛选 `metric=mae` 的 120 个 sample_key；
   - 通过 Quito config 重新加载 vali/test dataset；
   - 直接从 `item_dataset.data` 切片历史窗口 `x`，不访问未来 `y`，不使用专家误差、regret 或 oracle label 作为图像化输入；
   - 在线生成 `variant_a=3view` 和 `variant_b=top3fold`；
   - 只保存 `imageization_index.csv`、`latency_summary.csv`、`metadata.json`、`summary.md` 和少量 `debug_preview/*.png`。
3. 使用 Quito conda 环境做语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/common/pseudo_imageization.py \
     visual_router_experiments/stage1_vali_test_router/pilot/build_online_pseudo_image_pilot.py
   ```

4. 做函数级 smoke test，覆盖：
   - `normalize_window()` 三种 mode 的输出 finite；
   - `select_fft_periods()` 返回 `[B, 3]` 周期且范围在 `[2, 96]`；
   - `imageize_3view()` 与 `imageize_top3fold()` 输出 `[B, 3, 64, 64]` 且范围在 `[0, 1]`；
   - CUDA 可用时在 `cuda:0` 上生成伪图像。
5. 运行默认 pilot：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/pilot/build_online_pseudo_image_pilot.py \
     --print-rows 8
   ```

6. 检查输出目录是否存在全量 tensor/cache 文件，并抽检一张 debug PNG。
7. 同步更新 `visual_router_experiments/common/README.md`、`visual_router_experiments/stage1_vali_test_router/pilot/README.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

1. 默认 pilot 输出目录：

   ```text
   experiment_logs/run_outputs/2026-06-13_134118_592280_visual_router_stage1_online_pseudo_image_pilot/
   ```

2. 输出文件：

   ```text
   imageization_index.csv
   latency_summary.csv
   metadata.json
   summary.md
   debug_preview/*.png
   ```

3. 样本覆盖：
   - `imageization_index.csv` 共有 120 行；
   - `sample_key` 无重复；
   - `test/TEST_DATA_HOUR`、`test/TEST_DATA_MIN`、`vali/TEST_DATA_HOUR`、`vali/TEST_DATA_MIN` 各 30 行；
   - 与 `metric=mae` 的 oracle label sample_key 集合完全一致。
4. 图像化校验：
   - `variant_a` 和 `variant_b` shape 均为 `3x224x224`；
   - 所有输出 finite；
   - 输出范围在 `[0, 1]`；
   - `clip_ratio` 均值约 `0.000434`，最大约 `0.010417`。
5. latency 记录：
   - `latency_summary.csv` 共 12 个 batch 记录；
   - CPU per-window mean latency 约 `3.750040 ms`；
   - GPU per-window mean latency 约 `2.923612 ms`；
   - 主验证设备为 `cuda`。
6. 输出目录中未发现 `.pt`、`.pth`、`.npy`、`.npz`、`.zarr` 等全量 tensor/cache 文件。
7. 抽检 `debug_preview/96_48_S__test__TEST_DATA_MIN__item153__ch0__win0__variant_a_3view.png`，可见 line raster、period fold 和 FFT power 语义通道叠加后的非空视觉信号。

## 结论

在线伪图像化 pilot 已达到当前验收标准：120 个 sample_key 全部成功从 Quito 历史窗口 `x` 在线生成 `variant_a` 和 `variant_b`，shape/range/finite 校验通过，metadata 记录了 `norm_mode`、`pixel_mode`、`clip`、`top3_periods`、`clip_ratio`、`norm_mean`、`norm_std` 和 `norm_range`，且没有保存全量图像或 tensor cache。

当前默认 `norm_mode=revin_aux`、`pixel_mode=vision`、`clip=5.0`、`image_size=224`、`period_policy=fft_top3` 可以作为后续视觉 embedding cache 的第一版输入口径。

## 下一步方案

1. 基于该在线图像化模块接入冻结 ViT/MAE/CLIP encoder，优先只缓存 embedding 而不是图像 tensor。
2. 将 embedding cache 与现有 window oracle label 按 `sample_key` 对齐，训练最小 per-config visual router。
3. 后续做 `norm_mode=quito/revin/revin_aux` 和 `variant_a/variant_b` 的 ablation，比较视觉 router 与现有非视觉 baseline。
