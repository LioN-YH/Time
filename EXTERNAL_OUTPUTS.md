# 外部大规模输出索引

更新日期：2026-06-13 16:39:18 CST

本文档记录 `/home/shiyuhong/Time` 项目可使用但不纳入 Git 的外部大规模输出目录。

## `/data2/syh/Time/`

| 路径 | 用途 | 当前策略 |
| --- | --- | --- |
| `/data2/syh/Time/run_outputs/` | 保存后续需要较大磁盘空间的实验运行输出，例如大规模评估 manifest、summary、临时日志和可复核结果 | 可作为 `experiment_logs/run_outputs/` 的大盘替代路径；仓库内只记录索引和实验日志，不提交产物 |
| `/data2/syh/Time/cache_shards/` | 保存可删除或可再生成的临时 shard，例如抽样 embedding shard、临时 prediction shard、在线训练中间产物 | 默认不做全量长期缓存；除非先证明缓存能显著加速训练，否则优先使用 online 计算或短生命周期 shard |

## 视觉路线缓存原则

1. 不预存储全量伪图像张量。
2. 不默认预存储全量 ViT embedding。
3. 如需验证缓存收益，应先在小规模 shard 上同时记录：
   - online imageization + ViT forward latency；
   - embedding 读取 latency；
   - 训练/评估端到端耗时差异；
   - shard 空间开销和复现必要性。
4. 只有当缓存能带来明确的端到端加速，且磁盘空间、清理策略、metadata 对齐都可控时，才考虑扩大缓存范围。
