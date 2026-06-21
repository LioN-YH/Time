# Visual Router V2 Round 1 Summary Artifacts

本目录收集 Visual Router V2 Round 1 三个诊断/消融步骤的轻量结果文件，便于远程仓库中的 GPT 或人工 reviewer 直接分析。

## 目录

| 子目录 | 来源输出目录 | 内容 |
| --- | --- | --- |
| `p2probe/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_feature_probe/` | feature suitability、结构语义 probe、shortcut baseline、per-expert recall、within-dataset summary 和中文摘要 |
| `p2b_visual_pooling/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/` | visual-only pooling 三变体三 seed 的 selection/diagnostic 汇总、stratified summary、selected model counts、best variant 和中文摘要 |
| `p2c_aux_only/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only/` | RevIN aux-only 三 seed 的 selection/diagnostic 汇总、stratified summary、selected model counts、best seed 和中文摘要 |

## 边界

这里只保留适合代码仓库审阅的汇总 CSV、JSON 和 Markdown 文件；没有复制 checkpoint、prediction SQLite、逐样本 prediction CSV、运行 PID 或大规模缓存。

P2probe、P2b 和 P2c 均遵守 Round 1 pilot 协议：训练/选择只使用 `pilot_train` 和 `pilot_selection`，`diagnostic_balanced` 仅用于诊断，不使用 `pilot_test` 做模型选择。
