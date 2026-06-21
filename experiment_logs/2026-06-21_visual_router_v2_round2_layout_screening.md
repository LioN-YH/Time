# Visual Router V2 Round2c layout feature cache 与固定 FiLM screening

日志日期：2026-06-21 22:45:49 CST

## 目的

在 Round2 已冻结的 35k small sample sets 上，为六个默认 view layout 生成 frozen ViT feature cache，并用固定的 Round1 best `film_mean_patch_aux` 风格后端进行 layout 性能筛选。

## 背景

Round2a 已冻结 `round2_train_small=20000 vali`、`round2_selection_small=5000 vali`、`round2_diagnostic_balanced_small=5000 vali` 和 `round2_test_small=5000 test`。Round2b 已验证六个默认 layout 的 GPU tensor imageization 均输出 ViT-compatible `[B,3,224,224]` float32 tensor，finite 和 `[0,1]` range 检查通过。Round1 全局结论显示 `film_mean_patch_aux` 是 frozen pilot_test 当前最强变体，因此 Round2c 只比较 layout，不搜索后端结构或超参。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_layout_features.py`。
   - 支持单 layout worker 和 `--aggregate-only`。
   - 从 `round2_small_sample_manifest.csv` 读取四个 frozen sample set。
   - 通过 `round2_layout_registry.py` 生成 layout pseudo image，并用 Round1 visual checkpoint 的 frozen ViT 口径提取 `cls_embedding` 和 `mean_patch_embedding`。
   - 每个 shard 保存 `sample_key`、`order_index`、`layout_name`、`sample_set`、`cls_embedding`、`mean_patch_embedding` 和 `revin_aux`。
   - 不读取专家 prediction/oracle label 作为 imageization 输入，不保存 pseudo image tensor，不训练 router/encoder。

2. 新增 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_layout_film.py`。
   - 后端固定为 `film_mean_patch_aux`：base visual input 为 `mean_patch_embedding`，condition input 为 `revin_aux`。
   - `revin_aux` 只通过 FiLM gamma/beta 调制 visual hidden representation，不直接 concat 到 visual input。
   - 支持 `--build-index-only` 预构建 Round2c prediction subset SQLite，避免 layout×seed 并行任务竞争写同一 SQLite。
   - 单任务只写 `tasks/<layout>_seed<seed>/`，汇总步骤单独写 selection/diagnostic/test_small summary、delta、best layout、metadata 和 summary。

3. 新增 `visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py`。
   - 支持 `--devices cuda:0,cuda:1,cuda:2,cuda:3`、`--layouts`、`--feature-only`、`--train-only`、`--aggregate-only`、`--overwrite` 和 `--local-files-only`。
   - feature 阶段按 layout 级并行，training 阶段按 layout×seed 级并行，不使用 DataParallel/DDP。
   - unified manifest、prediction index 和最终 summary 都由单进程阶段写出。

4. 对照目标文件后半段补齐 aggregation 口径。
   - `round2_layout_selection_comparison.csv` 改为 reference-inclusive comparison，包含 Round2c layouts、Round1 `film_mean_patch_aux`、Round1 `visual_cls_mean_concat`、Round0 TimeFuse、oracle_top1 和 global_best_single；layout-only selection 副本另写 `round2_layout_selection_layout_only.csv`。
   - `round2_layout_delta_summary.csv` 补齐 `fft_absolute_energy - current_rgb_3view`、`top3fold_period_layout - current_rgb_3view`、`top3fold_period_layout - fft_absolute_energy`、best Round2 layout vs Round1 `film_mean_patch_aux`、best Round2 layout vs Round0 TimeFuse。
   - `round2_layout_screening_summary.md` 补齐 14 个必须回答的问题，包括 selection/test best、一致性、view separation、line-only、difference band、FFT/seasonality/PatchTST、top3fold、CrossFormer/PatchTST strata、MSE tail、latency、65k 推荐和 period continuity diagnostic。
   - `round2_layout_screening_metadata.json` 补齐 `round2_stage`、`trained_model`、`built_feature_cache`、`ran_vit`、`layout_registry_used`、`backend_fixed_to`、`changed_*`、`parallel_backend`、`single_task_output_isolated`、`recommended_expanded_layouts` 和 `next_step_recommendation` 等显式字段。

5. 使用 quito 环境做语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round2_layout_features.py \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round2_layout_film.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py
   ```

6. 运行极小 smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening_smoke \
     --layouts line_only \
     --devices cuda:0 \
     --feature-only \
     --max-samples-per-set 4 \
     --embedding-batch-size 4 \
     --feature-shard-size 4 \
     --overwrite \
     --poll-seconds 2
   ```

   随后在同一 smoke 输出上运行：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening_smoke \
     --layouts line_only \
     --devices cuda:0 \
     --train-only \
     --seeds 16 \
     --epochs 1 \
     --max-samples-per-set 4 \
     --batch-size 4 \
     --eval-batch-size 4 \
     --overwrite \
     --poll-seconds 2
   ```

7. 在 smoke 目录重跑新版 `--aggregate-only`，验证新增 metadata 字段、reference-inclusive comparison 和 14 问 summary 均能生成；随后删除 smoke 复制到 `experiment_summaries/visual_router_v2_round2/layout_screening/` 的轻量副本，避免误读为正式结果。

8. 启动正式后台任务：

   ```bash
   setsid bash /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening/command.sh \
     > /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening/main.log 2>&1 &
   ```

   `command.sh` 实际内容为：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python /home/shiyuhong/Time-visual-router-v2/visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_layout_screening_parallel.py --output-dir /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening --devices cuda:0,cuda:1,cuda:2,cuda:3 --layouts current_rgb_3view,spatial_panel_3view,line_only,line_difference_band,fft_absolute_energy,top3fold_period_layout --seeds 16,17,18 --epochs 3 --batch-size 256 --eval-batch-size 512 --embedding-batch-size 16 --feature-shard-size 2000 --local-files-only --poll-seconds 15
   ```

9. 根据 GPU 利用率偏低调整并行策略。
   - 观察到原正式 launcher 为每卡 1 个 feature worker、`embedding_batch_size=16`，显存约 650MB/卡，GPU 仍有大量空余。
   - 修正 launcher 中 `device_slots` 生成方式，避免 `--max-procs-per-device>1` 时先连续填满同一张 GPU，改为跨 GPU 轮转分配。
   - 停止旧 PGID `3160713`，停机前保留 15 个完整 shard。
   - 不传 `--overwrite` 重启同一输出目录，已完成 shard 会校验并跳过；新命令改为 `--embedding-batch-size 32 --max-procs-per-device 2`，并写入 `main_accel.log`。
   - 加速版启动后 6 个 layout worker 同时运行：cuda:0 与 cuda:1 各 2 个进程，cuda:2 与 cuda:3 各 1 个进程。

## 结果

1. 语法检查通过。
2. smoke feature-only 首次暴露 ViT loader 需要 `pooling` 参数，已在 builder 的 `make_encoder_args()` 中补齐 `pooling="cls"`，复验通过。
3. smoke feature-only 输出 `round2_layout_feature_manifest.csv`、cache size、latency、metadata 和 summary；`line_only` 四个 sample set 各 4 个样本，共 16 行 feature。
4. smoke train-only 完成 prediction subset index 预构建、`line_only seed=16` 1 epoch 训练、selection/diagnostic/test_small eval 和 aggregation。检查要求的 16 个核心输出名均存在，`status.json` 为 `completed`。
5. 新版 smoke aggregation 检查通过：metadata 必填字段存在，`round2_layout_selection_comparison.csv` 含 reference，summary 覆盖 1-14 个必答问题。
6. 正式后台任务已启动：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening/`
   - PID/PGID：`3160713 / 3160713`
   - 当前健康检查显示四个 feature worker 已启动：
     - `current_rgb_3view` -> `cuda:0`
     - `spatial_panel_3view` -> `cuda:1`
     - `line_only` -> `cuda:2`
     - `line_difference_band` -> `cuda:3`
   - `fft_absolute_energy` 和 `top3fold_period_layout` 将在第一批 layout worker 完成后排队启动。
   - 2026-06-21 22:53 CST 复查时，首批 4 个 layout worker 仍正常运行，已写出 11 个 feature shard，无失败状态。
   - 2026-06-21 22:57 CST 已切换到加速版正式任务，PID/PGID 更新为 `3167543 / 3167543`；加速版健康检查显示 6 个 layout worker 同时运行，显存约 cuda:0/1 1374MB、cuda:2/3 693MB，无 OOM 或失败状态。
7. 正式任务已于 2026-06-22 00:08:50 CST 完成：
   - feature cache 阶段六个 layout 均完成 35,000 个样本，覆盖 `round2_train_small=20000`、`round2_selection_small=5000`、`round2_diagnostic_balanced_small=5000`、`round2_test_small=5000`。
   - prediction subset SQLite `prediction_index_round2c_35k.sqlite` 覆盖 35,000 个 sample_key、175,000 条 `sample_key + model_name` 记录。
   - 18 个 `layout × seed` 训练任务全部完成，seeds 为 16、17、18，epochs 为 3。
   - aggregation 写出目标要求的 16 个核心输出文件，并复制正式轻量 summary 到 `experiment_summaries/visual_router_v2_round2/layout_screening/`。
8. 正式验收脚本已通过：
   - 检查所有必需 CSV/JSON/Markdown 输出存在。
   - 检查 feature manifest 中六个 layout、四个 sample set 的样本计数均符合冻结样本规模。
   - 检查 `round2_layout_variant_seed_results.csv` 覆盖六个 layout、三 seeds、selection/diagnostic/test_small 三个评估集合，且包含 hard top-1 与 raw-soft fusion。
   - 检查 `round2_layout_screening_metadata.json` 明确本轮唯一变量为 layout，base visual input 为 `mean_patch_embedding`，condition input 为 `revin_aux`，使用 FiLM、不 concat aux、不使用 test_small 做选择。
   - 检查 `round2_layout_selection_comparison.csv` 覆盖六个 Round2 layout、Round1 `film_mean_patch_aux`、Round1 `visual_cls_mean_concat`、Round0 TimeFuse、oracle_top1 和 global_best_single。
   - 检查 delta summary 覆盖目标要求的四组核心 pair。
9. 正式筛选结果：
   - `round2_selection_small` raw-soft 最优 layout 为 `spatial_panel_3view`，MAE=0.310385，MSE=3.329199。
   - `round2_test_small` frozen screening raw-soft 最优 layout 也是 `spatial_panel_3view`，MAE=0.398598，MSE=3.484102。
   - selection best 与 test_small best 一致。
   - 推荐进入 65k expanded validation 的 layout 为 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout`。

## 结论

Round2c 35k layout feature cache 与固定 FiLM screening 已完成。正式结果显示 `spatial_panel_3view` 在 selection 和 frozen test_small 上均为 raw-soft MAE 最优 layout，可作为下一轮 65k expanded validation 的首选候选；`current_rgb_3view` 和 `top3fold_period_layout` 保留为对照/候选。进入 65k 前仍需要做 period continuity diagnostic，重点检查 hard FFT period selection 对 pseudo image、ViT embedding、router weight 和 selected model flip 的连续性影响。

## 下一步方案

1. 先做 period continuity diagnostic，重点覆盖 `top3fold_period_layout` 与 current period fold 路径。
2. 基于 `spatial_panel_3view`、`current_rgb_3view`、`top3fold_period_layout` 设计 65k expanded validation，后端继续固定为 `film_mean_patch_aux` 风格，避免同时搜索 head/loss/calibration。
3. 保留 `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round2_layout_screening/` 作为正式结果目录；仓库内只引用 `experiment_summaries/visual_router_v2_round2/layout_screening/` 的轻量副本。
