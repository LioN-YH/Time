# Visual Router V2 Round2 staged full-scale validation plan

生成时间：2026-06-22 11:06:48 CST

## 目的

把 Round2 full-scale validation 拆成可恢复、可审计、可扩展的 staged pipeline。当前阶段只完成 thin slice，不启动 1M 或 116M 正式长跑。

## 固定实验矩阵

| 角色 | layout | backend |
| --- | --- | --- |
| mainline | `spatial_panel_3view` | `film_mean_patch_aux` |
| required baseline | `current_rgb_3view` | `film_mean_patch_aux` |

本阶段不加入 `top3fold_period_layout`、`line_only`、`line_difference_band`、`fft_absolute_energy`、`period_soft_mixture`、period tokens、panel-wise pooling 或 independent view encoder。

## Staged scale

| scale | 样本来源 | 目标 | 是否本步运行 |
| --- | --- | --- | --- |
| `smoke` | full-scale sample shard 的确定性小切片 | 验证 sample loading、imageization、feature writing、SQLite lookup、train/eval aggregation 和 schema | 允许运行 |
| `one_shard` | 单个真实 full-scale shard 的较大切片 | 验证 shard-aware feature cache、batch SQLite lookup、report generation | 允许 dry-run 或小规模执行 |
| `1M planning` | 多 shard staged list | 只固定 CLI 和扩展方案 | 本步不启动 |
| near-full scale | full-scale shards | frozen eval 与 tail/strata report | 本步不启动 |

## Pipeline

1. `build_visual_router_v2_round2_staged_samples.py`
   - 只读取 `sample_manifest_full_scale/sample_shards/sample_shard_XXXX_of_0064.csv`。
   - 从 `vali` 切出 `staged_train`、`staged_selection`、`staged_diagnostic`，从 `test` 切出 `staged_test`。
   - 用 oracle parquet batch scan 补 `oracle_model`、`error_gap`、`error_gap_quantile`。
   - 不读取 116M prediction manifest。

2. `launch_visual_router_v2_round2_staged_validation_parallel.py`
   - 复用现有 Round2 layout feature builder。
   - 按 `layout/sample_set/shard` 写 feature cache shard，不保存 pseudo image tensor。
   - 复用 fixed FiLM trainer，先构建 subset SQLite prediction index，再启动 layout × seed worker。
   - 支持 `--dry-run`、`--feature-only`、`--train-only`、`--eval-only`、`--aggregate-only`、`--local-files-only`、`--overwrite`。

3. `summarize_visual_router_v2_round2_staged_validation.py`
   - 复核 feature manifest、SQLite record count 和 per-model coverage。
   - 汇总 overall、strata、tail 和 router behavior。
   - 轻量 summary 复制到 `experiment_summaries/visual_router_v2_round2/staged_fullscale_validation/`。

## 推荐命令

Dry-run：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_staged_validation_parallel.py \
  --layouts spatial_panel_3view,current_rgb_3view \
  --backend film_mean_patch_aux \
  --sample-scale smoke \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --dry-run
```

Very small smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_staged_validation_parallel.py \
  --layouts spatial_panel_3view,current_rgb_3view \
  --backend film_mean_patch_aux \
  --sample-scale smoke \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --seeds 16 \
  --epochs 1 \
  --local-files-only \
  --overwrite
```

One-shard dry-run：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_staged_validation_parallel.py \
  --layouts spatial_panel_3view,current_rgb_3view \
  --backend film_mean_patch_aux \
  --sample-scale one_shard \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --dry-run
```

## 扩大到 1M 的路径

后续新增 shard list 参数，把 sample builder 从单 shard 扩展为多 shard deterministic slice；feature cache 继续按 `layout/sample_set/shard` 分片；prediction lookup 仍使用 subset SQLite，按 sample_key 批量查询；训练只使用 staged train，选择只使用 staged selection，test 只做 frozen eval。

## 监控项

- CrossFormer / PatchTST 是否继续改善。
- `LOW_LOW_HIGH` group 是否继续退化。
- q4 / q5 error_gap 是否稳定。
- low forecastability / strong trend / highly_variable CV 是否仍贡献主要收益。
- high-regret tail 中 PatchTST selected mode 偏重是否加剧。
- `spatial_panel_3view` 相对 `current_rgb_3view` 的收益是否仍存在。
