# Visual Router V2 Round 1 Feature Cache Summary

生成时间：2026-06-21 14:59:29 CST

## 输入路径

- P0 sample set：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples`
- P1 Round 0 output：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0`
- Visual checkpoint：`/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt`
- Quito config：`/home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml`

## 输出结构

- 输出目录：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only`
- `features/<sample_set>/shard_XXXXX.npz`：sharded feature cache
- `round1_feature_manifest.csv`：按 sample_set/order_index 恢复顺序的 shard manifest
- `round1_feature_cache_size_summary.csv`：shard 与累计缓存大小
- `round1_feature_metadata.json`、`status.json`：运行参数、状态和 lineage

## Feature Schema

- schema version：`visual_router_v2_round1_feature_cache_v1`
- 每个 shard 包含 `sample_key`、`order_index`、`cls_embedding`、`mean_patch_embedding`、`revin_aux`。
- `cls_embedding` 来自 ViT CLS token。
- `mean_patch_embedding` 只对 patch tokens 求均值，即 `last_hidden_state[:, 1:, :].mean(dim=1)`，不包含 CLS token。
- `revin_aux` 为 6 维 raw aux：mean, log_std, min, max, range, clip_ratio；只由历史窗口 x 计算。
- 本步骤不 fit scaler，不保存 `cls_mean_concat_embedding`，后续训练时按需 concat。

## Sample Counts

| sample_set | sample_count |
| --- | --- |
| pilot_test | 75000 |

## Cache Size

- 总缓存大小：198.192 MB
- shard 数：38

## Smoke 与正式运行结果

- 当前运行模式：正式 P2a
- `status.json` 状态：`completed`
- 所有 shard finite：True
- resume/skip existing 机制：`--overwrite` 未开启时会校验已有 shard 并跳过；本次 skipped_shards=0。

## 使用边界

该 cache 只覆盖 Round 1 pilot 的固定 P0 sample sets，用于 RevIN aux 与 pooling 消融。它不是 full-scale embedding cache：未处理全量 2327 万样本、未读取 116M prediction manifest、未保存 pseudo image tensor，默认也不生成 pilot_test feature，避免把 final test 用于架构选择。
