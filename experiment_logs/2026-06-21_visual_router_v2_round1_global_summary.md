# Visual Router V2 Round 1 全局汇总与最终推荐

日志日期：2026-06-21 18:40:28 CST

## 目的

对 Visual Router V2 Round 1 从 P0 到 P2e FiLM frozen pilot_test extension 的轻量结果做全局汇总、归档、解释和路线推荐，生成仓库内可直接审阅的 CSV/JSON/Markdown 产物。

## 背景

Round 1 已完成 P0 样本冻结、P1 Round0 复现、P2a feature cache、P2probe、P2b visual pooling、P2c aux-only、P2d direct concat、P2d frozen final test extension、P2e FiLM 和 P2e FiLM frozen final test extension。当前 frozen `pilot_test` 最强变体为 `film_mean_patch_aux`，raw-soft MAE/MSE/regret=0.417824/183.353985/0.077539，优于 `film_cls_mean_concat_aux`、visual-only baseline、direct concat baseline 和 Round0 TimeFuse。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/summarize_visual_router_v2_round1_global.py`。
2. 脚本只读取 `experiment_summaries/visual_router_v2_round1/` 下既有轻量 summary/CSV/JSON，包括 P2probe、P2b、P2c、P2d、P2d final test、P2d final test extension、P2e FiLM 和 P2e FiLM final test extension 的归档结果。
3. 未读取 checkpoint、SQLite prediction index、逐样本 prediction CSV、feature shard 或 116M prediction manifest，未训练新模型，未重新评估 `pilot_test`。
4. 使用 quito 环境运行：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/summarize_visual_router_v2_round1_global.py
   ```

5. 生成 `experiment_summaries/visual_router_v2_round1/global_summary/` 下八个轻量产物：
   - `round1_global_comparison.csv`
   - `round1_global_selection_diagnostic.csv`
   - `round1_global_final_test.csv`
   - `round1_global_delta_summary.csv`
   - `round1_global_selected_model_summary.csv`
   - `round1_global_strata_summary.csv`
   - `round1_global_recommendation.md`
   - `round1_global_metadata.json`
6. 同步更新 `experiment_summaries/visual_router_v2_round1/README.md`、`visual_router_experiments/stage1_vali_test_router/README.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

`round1_global_final_test.csv` 共 10 行、25 列，覆盖 `film_mean_patch_aux`、`film_cls_mean_concat_aux`、`visual_cls_mean_concat`、`visual_mean_patch_only`、`cls_mean_concat_plus_aux`、`mean_patch_plus_aux`、Round0 TimeFuse、Round0 original Visual、global_best_single 和 oracle_top1。raw-soft MAE 排名显示：

1. `film_mean_patch_aux`：MAE=0.417824，MSE=183.353985，regret=0.077539，MAE_std=0.000657。
2. `film_cls_mean_concat_aux`：MAE=0.419568，MSE=183.463846，regret=0.079283，MAE_std=0.001850。
3. `visual_cls_mean_concat`：MAE=0.443062，MSE=244.238487，regret=0.102777。
4. `cls_mean_concat_plus_aux`：MAE=0.452942，MSE=245.459475，regret=0.112657。
5. `visual_mean_patch_only`：MAE=0.452976，MSE=303.486492，regret=0.112691。
6. `mean_patch_plus_aux`：MAE=0.516108，MSE=486.102519，regret=0.175823。
7. Round0 TimeFuse：MAE=0.535220，MSE=568.502401，regret=0.194935。

`round1_global_metadata.json` 记录 `summary_only=true`、`trained_new_model=false`、`evaluated_new_model=false`、`used_pilot_test_for_selection=false`、`read_checkpoint=false`、`read_prediction_csv_sample_level=false`、`read_sqlite_prediction_index=false`、`read_feature_shard=false`、`loaded_116m_prediction_manifest_to_memory=false`，并记录推荐主线 `recommended_main_variant=film_mean_patch_aux`。

## 结论

Round 1 的全局推荐是以 `film_mean_patch_aux` 作为当前主线结构，`film_cls_mean_concat_aux` 作为强对照结构，`visual_cls_mean_concat` 作为 visual-only strong baseline，`visual_mean_patch_only` 作为简洁 visual-only baseline。不建议把 `mean_patch_plus_aux` 或 direct concat aux 作为后续主要路线。

关键解释是：visual embedding 本身有强信号；RevIN aux 直接 concat 容易破坏泛化，尤其 `mean_patch_plus_aux` 在 frozen `pilot_test` 上明显退化；FiLM modulation 更符合 aux 调制 visual hidden representation 的机制假设，因此能同时改善 MAE、MSE、regret 和 seed stability。FiLM 的优势仍主要来自 raw-soft fusion 和更低 MSE tail，而不是 hard oracle-label classifier accuracy。

## 下一步方案

1. P2f / Round1 calibration diagnostic：只在 `pilot_selection` 选择 temperature 或 post-hoc calibration 参数，`pilot_test` 继续作为 frozen eval。
2. P2g FiLM hyperparameter small search：只在 `pilot_train` / `pilot_selection` 上进行，不用 `pilot_test` 选择 hidden dim、dropout、epoch、seed 或 variant。
3. Round2 view layout / pseudo image small screening：先小样本筛选，再扩大。
4. Stage 1 canonical migration 时，把 FiLM 作为 Visual Router candidate head/adapter 的重要候选，但不要立即做 full-scale 重构。
