# Visual Router V2 Round 1 P2e FiLM / aux modulation pilot 消融

日志日期：2026-06-21 17:32:28 CST

## 目的

在不使用 `pilot_test` 的前提下，新增并完成 P2e FiLM / aux modulation 小规模消融，验证 RevIN aux 是否更适合作为 hidden representation 的调制信号，而不是直接 concat 到 visual input。

## 背景

Round 1 frozen pilot_test extension 显示 `visual_cls_mean_concat` 在 pilot_test raw-soft MAE / regret 上优于 P2d concat best，而直接 concat aux 在 frozen pilot_test 上没有正向边际贡献。为避免继续用 `pilot_test` 做选择，本轮只使用 `pilot_train` 训练、`pilot_selection` 选择、`diagnostic_balanced` 诊断，并复用 P2a sharded feature cache。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_film.py`：
   - 支持 `film_cls_mean_concat_aux` 与 `film_mean_patch_aux` 两个变体；
   - base visual input 分别为 `concat(cls_embedding, mean_patch_embedding)` 与 `mean_patch_embedding`；
   - `revin_aux` 单独标准化后输入小 MLP 生成 `gamma/beta`，执行 `h_mod = h * (1 + gamma) + beta`；
   - 支持 `--run-single`、`--aggregate-only`、串行 fallback 和 `--overwrite`；
   - 单任务只写 `tasks/<variant>_seed<seed>/`，汇总阶段单独生成统一 CSV/JSON/Markdown。
2. 新增 `visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round1_film_parallel.py`：
   - 使用进程级多 GPU 并行，不使用 DataParallel/DDP；
   - 默认设备为 `cuda:1,cuda:2,cuda:3`；
   - 支持 `--max-procs-per-device`，用于显存充足时每张 GPU 同时排多个 seed。
3. 使用 Quito 环境执行语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_film.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round1_film_parallel.py
   ```

4. 先用 `--max-samples-per-set 16` 对两个 FiLM 变体各跑 seed 16 的 smoke，并执行 `--aggregate-only --seeds 16`，验证单任务输出隔离、summary、delta 和 metadata 写出。
5. 正式输出目录为 `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/`。先用 launcher 启动 `film_cls_mean_concat_aux` seeds 16/17/18；观察到 GPU 1/2/3 每卡显存约 355MB 后，停止 launcher 父进程但保留已运行的三个训练子进程，并手动同步启动 `film_mean_patch_aux` seeds 16/17/18。
6. 六个单任务全部完成后，手动运行：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router_v2_round1_film.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film \
     --seeds 16,17,18 \
     --devices-requested cuda:1,cuda:2,cuda:3 \
     --parallel-launcher-used \
     --aggregate-only \
     --overwrite
   ```

7. 将轻量 summary 复制到 `experiment_summaries/visual_router_v2_round1/p2e_film/`，未复制 checkpoint、SQLite、逐样本 prediction CSV、feature shard 或 cache。
8. 更新 `WORKSPACE_STRUCTURE.md`、`experiment_summaries/visual_router_v2_round1/README.md` 和本日志总览。

## 结果

正式训练完成两个 FiLM 变体 × seeds 16/17/18 × 3 epochs。输出文件完整：

- `round1_film_variant_seed_results.csv`
- `round1_film_selection_comparison.csv`
- `round1_film_diagnostic_summary.csv`
- `round1_film_selected_model_counts.csv`
- `round1_film_stratified_summary.csv`
- `round1_film_delta_summary.csv`
- `round1_film_best_variant.json`
- `round1_film_metadata.json`
- `round1_film_summary.md`

验收脚本已确认：

- 两个 variant 与 seeds 16/17/18 覆盖完整；
- `pilot_selection` 和 `diagnostic_balanced` 均有 hard top-1 / raw-soft fusion 行；
- comparison 包含 `film_cls_mean_concat_aux`、`film_mean_patch_aux`、P2b visual baselines、P2c aux-only、P2d concat baselines、Round0 TimeFuse、Round0 original Visual、global_best_single 和 oracle_top1；
- delta summary 包含用户指定的七组 delta；
- metadata 记录 `pilot_test_used_for_selection=false`、`pilot_test_evaluated=false`、`trained_new_model=true`、`rebuilt_p2a_feature=false`、`loaded_116m_prediction_manifest_to_memory=false`、`saved_pseudo_image_tensor=false`、`used_film=true`、`used_gating=false`、`used_attention=false`、`used_concat_aux=false`、`parallel_backend=process_per_variant_seed`、`single_task_output_isolated=true`。

关键 selection raw-soft 结果：

| 变体 | MAE_mean | MAE_std | MSE_mean | 备注 |
| --- | ---: | ---: | ---: | --- |
| `film_mean_patch_aux` | 0.300393 | 0.000542 | 1.289872 | P2e best FiLM |
| `film_cls_mean_concat_aux` | 0.300486 | 0.000859 | 1.313162 | 优于 visual_cls_mean_concat 和 cls_mean_concat_plus_aux 的 MAE，但 MSE 变差 |
| `cls_mean_concat_plus_aux` | 0.300605 | 0.001287 | 1.205401 | P2d concat baseline |
| `mean_patch_plus_aux` | 0.300831 | 0.000548 | 1.239938 | P2d concat baseline |
| `visual_mean_patch_only` | 0.300996 | 0.001000 | 1.234168 | P2b visual-only baseline |
| `visual_cls_mean_concat` | 0.302220 | 0.003929 | 1.217317 | P2b visual-only baseline |

关键 diagnostic raw-soft 结果：

- `film_mean_patch_aux`：MAE_mean=0.345809，MSE_mean=1.394350。
- `film_cls_mean_concat_aux`：MAE_mean=0.346838，MSE_mean=1.407221。

## 结论

1. `film_cls_mean_concat_aux` 在 pilot_selection raw-soft MAE 上优于 `visual_cls_mean_concat`，delta=-0.001734。
2. `film_cls_mean_concat_aux` 在 pilot_selection raw-soft MAE 上也略优于 `cls_mean_concat_plus_aux`，delta=-0.000120。
3. `film_mean_patch_aux` 优于 `visual_mean_patch_only`，delta=-0.000603。
4. mean_patch 路线中 FiLM 优于直接 concat aux，`film_mean_patch_aux - mean_patch_plus_aux` delta=-0.000438。
5. FiLM 改善 seed stability：`film_cls_mean_concat_aux` MAE_std=0.000859，低于 `visual_cls_mean_concat` 的 0.003929。
6. FiLM 未改善 MSE tail：`film_cls_mean_concat_aux` MSE_mean=1.313162，高于 `visual_cls_mean_concat` 的 1.217317。
7. CrossFormer / PatchTST strata 需要结合 `round1_film_stratified_summary.csv` 继续细看；summary 中已列出 oracle_model 分层。
8. 基于 selection MAE，P2e FiLM 有进入下一步 frozen pilot_test eval extension 的价值，但需要把 MSE tail 变差作为风险明确记录。

## 下一步方案

1. 单独开 frozen pilot_test eval extension，固定 P2e 已训练 checkpoint，不按 pilot_test 重新选择 seed/epoch/超参。
2. frozen eval 中重点比较 `film_mean_patch_aux`、`film_cls_mean_concat_aux`、P2b `visual_cls_mean_concat`、P2d `cls_mean_concat_plus_aux`、Round0 TimeFuse 和 oracle_top1。
3. 除 MAE/regret 外必须重点报告 MSE tail、CrossFormer / PatchTST oracle strata、dataset/group/forecastability 分层，避免只因 selection MAE 轻微改善而忽略 tail 风险。
