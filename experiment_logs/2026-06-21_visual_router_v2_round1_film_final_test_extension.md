# Visual Router V2 Round 1 P2e FiLM frozen pilot_test eval extension

日志日期：2026-06-21 18:13:37 CST

## 目的

对已经完成训练和 pilot_selection 选择的 P2e FiLM / aux modulation 变体做 frozen `pilot_test` final eval extension，验证 `film_mean_patch_aux` 与 `film_cls_mean_concat_aux` 在 P0 `pilot_test` 上的最终泛化表现。

本步骤只做评估，不训练新模型，不调参，不重新选择 variant、seed、epoch 或 hyperparameter，不改变 P2e selection best 历史结论。

## 背景

P2e 已完成两个 FiLM 变体 `film_mean_patch_aux`、`film_cls_mean_concat_aux` 的 seeds 16/17/18 pilot_train 训练、pilot_selection 选择和 diagnostic_balanced 诊断。P2e selection best 为 `film_mean_patch_aux`，pilot_selection raw-soft MAE mean=0.300393，但 selection 上的轻微改善不足以证明泛化有效，因此需要在冻结条件下评估 P0 `pilot_test`。

本次评估复用：

- P0 `pilot_test` sample keys：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/pilot_test_sample_keys.csv`
- final_test_only feature cache：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/`
- P2e FiLM checkpoint：`/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/tasks/*/checkpoint_*.pt`
- P2d final extension baseline 和 SQLite index：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test_extension/`

## 操作

1. 新增评估脚本 `visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_film_final_test_extension.py`。
   - 支持 `--run-single` 单个 variant/seed frozen eval。
   - 支持 `--aggregate-only` 汇总隔离 task 结果。
   - 默认串行 fallback 可完成全部 2 variants × 3 seeds。
   - 加载 FiLM checkpoint 的 `router_state_dict`、`visual_scaler_state`、`aux_scaler_state`，恢复 `FiLMRouter` 后只做 forward 与 fusion metric 计算。
   - 读取 final_test_only `pilot_test` feature cache，不重建 P2a feature，不保存 pseudo image tensor。

2. 新增并行 launcher `visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round1_film_final_test_parallel.py`。
   - 使用进程级 `process_per_variant_seed` 并行。
   - 默认设备为 `cuda:1,cuda:2,cuda:3`。
   - 每个 variant/seed 写入独立 `tasks/<variant>_seed<seed>/` 子目录和独立 prediction CSV/log。
   - 6 个 eval task 全部完成后单独运行 aggregation。

3. 使用 quito 环境通过语法编译：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_film_final_test_extension.py \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round1_film_final_test_parallel.py
   ```

4. 首次启动并行 launcher 时，多个子进程同时复制本次输出目录内的 SQLite index，导致本次 extension 目录中的半成品 SQLite 出现 `database disk image is malformed`。已停止 launcher，确认无遗留评估进程，并精确删除本次半成品目录：

   ```bash
   rm -rf /data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film_final_test_extension
   ```

   删除范围只包含本次刚启动失败的 P2e final extension 输出目录，未删除 P2e 训练输出、P2d final extension 输出、feature cache 或历史 summary。

5. 修补 `ensure_prediction_index`，并行 eval task 直接复用已完成 P2d final extension 的 SQLite index：

   ```text
   /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test_extension/prediction_index_round1_final_test_pilot_test.sqlite
   ```

   只有在没有可复用 index 时才在本次输出目录构建新 index，从而避免并行首次复制写竞争。

6. 重新编译通过后，使用 GPU 1/2/3 启动正式并行 frozen eval：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round1_film_final_test_parallel.py \
     --devices cuda:1,cuda:2,cuda:3 \
     --poll-seconds 10
   ```

7. 6 个单任务全部完成后 launcher 自动运行 aggregation。随后修正 metadata 中 prediction index 来源字段，并只重跑 `--aggregate-only`，未重跑 6 个 eval task：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round1_film_final_test_extension.py \
     --aggregate-only \
     --parallel-eval-used \
     --devices-requested cuda:1,cuda:2,cuda:3 \
     --overwrite
   ```

8. 已复制轻量汇总文件到：

   ```text
   experiment_summaries/visual_router_v2_round1/p2e_film_final_test_extension/
   ```

   只复制 CSV/JSON/Markdown 汇总，不复制 checkpoint、SQLite、逐样本 prediction CSV、feature shard 或大规模 cache。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film_final_test_extension/
```

已生成必需产物：

- `round1_film_final_test_extension_variant_seed_results.csv`
- `round1_film_final_test_extension_comparison.csv`
- `round1_film_final_test_extension_selected_model_counts.csv`
- `round1_film_final_test_extension_stratified_summary.csv`
- `round1_film_final_test_extension_delta_summary.csv`
- `round1_film_final_test_extension_metadata.json`
- `round1_film_final_test_extension_summary.md`
- `status.json`

实际完成 6 个 frozen eval task：

- `film_cls_mean_concat_aux` seeds 16/17/18
- `film_mean_patch_aux` seeds 16/17/18

核心 pilot_test raw-soft 指标如下：

| variant | MAE | MSE | regret_to_oracle | MAE_std |
| --- | ---: | ---: | ---: | ---: |
| `film_mean_patch_aux` | 0.417824 | 183.353985 | 0.077539 | 0.000657 |
| `film_cls_mean_concat_aux` | 0.419568 | 183.463846 | 0.079283 | 0.001850 |
| `visual_cls_mean_concat` | 0.443062 | 244.238487 | 0.102777 | 0.021419 |
| `visual_mean_patch_only` | 0.452976 | 303.486492 | 0.112691 | 0.044625 |
| `cls_mean_concat_plus_aux` | 0.452942 | 245.459475 | 0.112657 | 0.039445 |
| `mean_patch_plus_aux` | 0.516108 | 486.102519 | 0.175823 | 0.048081 |
| Round0 TimeFuse | 0.535220 | 568.502401 | 0.194935 | 0.000000 |

重点结论：

1. `film_mean_patch_aux` 在 pilot_test 上优于 `visual_mean_patch_only`，raw-soft MAE delta=-0.035152。
2. `film_mean_patch_aux` 避免了 `mean_patch_plus_aux` 的明显退化，raw-soft MAE delta=-0.098284。
3. `film_cls_mean_concat_aux` 优于 `visual_cls_mean_concat`，raw-soft MAE delta=-0.023493。
4. `film_cls_mean_concat_aux` 优于 `cls_mean_concat_plus_aux`，raw-soft MAE delta=-0.033373。
5. 两个 FiLM 变体中 `film_mean_patch_aux` 的 raw-soft MAE、MSE 和 regret 均更好。
6. 两条 FiLM 路线相对对应 baseline 均明显改善 MAE/MSE seed stability。
7. 两条 FiLM 路线相对对应 baseline 均改善 MSE tail。
8. CrossFormer / PatchTST 分层已写入 `round1_film_final_test_extension_stratified_summary.csv` 和 summary 摘录。
9. FiLM 权重仍保持较高 normalized entropy，主要收益来自 soft fusion，而不是单纯 hard oracle-label accuracy。
10. P2e 可进入下一轮 Round2/P2f，但必须保持 frozen test 风险约束；该建议不改变 P2e selection best 历史结论。

metadata 已明确记录：

- `pilot_test_used_for_selection=false`
- `pilot_test_evaluated=true`
- `trained_new_model=false`
- `changed_variant_by_test=false`
- `changed_seed_by_test=false`
- `changed_epoch_by_test=false`
- `changed_hyperparams_by_test=false`
- `rebuilt_p2a_feature=false`
- `used_final_test_only_feature_cache=true`
- `loaded_116m_prediction_manifest_to_memory=false`
- `saved_pseudo_image_tensor=false`
- `used_film=true`
- `used_gating=false`
- `used_attention=false`
- `used_concat_aux=false`
- `parallel_eval_used=true`
- `parallel_backend=process_per_variant_seed`
- `devices_requested=cuda:1,cuda:2,cuda:3`
- `devices_used=["cuda:1","cuda:2","cuda:3"]`
- `single_task_output_isolated=true`

## 结论

P2e FiLM frozen pilot_test eval extension 已完成。与 selection 结论相比，pilot_test 上 FiLM 的收益更明确：两个 FiLM 变体均优于对应 visual-only 与 concat aux baseline，且 `film_mean_patch_aux` 在 MAE、MSE、regret 和 seed stability 上都是本次 FiLM 中最好的变体。

本次评估没有训练新模型，没有使用 `pilot_test` 做模型选择，没有重建 P2a feature cache，没有保存 pseudo image tensor，也没有全量加载 116M prediction manifest。

## 下一步方案

1. 若进入 Round2/P2f，应以 `film_mean_patch_aux` 作为 FiLM 主候选，同时保留 `visual_cls_mean_concat` 或 visual-only cls+mean 路线作为强对照。
2. Round2/P2f 仍必须坚持 `pilot_train`/`pilot_selection` 选择、`pilot_test` 只做冻结 final eval 的边界。
3. 下一轮应重点检查 CrossFormer / PatchTST strata、MSE tail 和 soft fusion 权重熵，避免只依据 aggregate MAE 推进。
4. 如后续再次并行使用 SQLite prediction index，应优先复用只读既有 index 或预先单进程准备 index，避免多个子进程同时创建/复制同一个 SQLite 文件。
