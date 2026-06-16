# 外部大规模输出索引

更新日期：2026-06-16 02:33:22 CST

本文档记录 `/home/shiyuhong/Time` 项目可使用但不纳入 Git 的外部大规模输出目录。

## `/data2/syh/Time/`

| 路径 | 用途 | 当前策略 |
| --- | --- | --- |
| `/data2/syh/Time/run_outputs/` | 保存后续需要较大磁盘空间的实验运行输出，例如大规模评估 manifest、summary、临时日志和可复核结果 | 可作为 `experiment_logs/run_outputs/` 的大盘替代路径；仓库内只记录索引和实验日志，不提交产物 |
| `/data2/syh/Time/cache_shards/` | 保存可删除或可再生成的临时 shard，例如抽样 embedding shard、临时 prediction shard、在线训练中间产物 | 默认不做全量长期缓存；除非先证明缓存能显著加速训练，否则优先使用 online 计算或短生命周期 shard |
| `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/` | Stage 1 `96_48_S` 正式 full-scale 全候选窗口输出根目录 | `sample_manifest_full_scale/` 已生成 23,275,170 个 sample_key 和 64 个 shard；`prediction_cache_full_scale_launcher/merged_cache/` 已完成正式五专家 merged prediction cache，`record_count=116,375,850`、`sample_count=23,275,170`、`array_storage=packed_npy_v1`；`prediction_cache_full_scale_launcher/merged_cache_validation/2026-06-16_011835_full_integrity_validation_compact_retry/` 保存完整性校验，`passed=true`；`timefuse_feature_cache_full_scale_launcher/` 为独立 TimeFuse-derived feature cache 输出；根目录 `HANDOFF.md` 记录接手命令和当前状态 |

## 视觉路线缓存原则

1. 不预存储全量伪图像张量。
2. 不默认预存储全量 ViT embedding。
3. 如需验证缓存收益，应先在小规模 shard 上同时记录：
   - online imageization + ViT forward latency；
   - embedding 读取 latency；
   - 训练/评估端到端耗时差异；
   - shard 空间开销和复现必要性。
4. 只有当缓存能带来明确的端到端加速，且磁盘空间、清理策略、metadata 对齐都可控时，才考虑扩大缓存范围。
