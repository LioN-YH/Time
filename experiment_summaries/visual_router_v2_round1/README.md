# Visual Router V2 Round 1 Summary Artifacts

本目录收集 Visual Router V2 Round 1 诊断/消融步骤的轻量结果文件，便于远程仓库中的 GPT 或人工 reviewer 直接分析。

## 目录

| 子目录 | 来源输出目录 | 内容 |
| --- | --- | --- |
| `p2probe/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_feature_probe/` | feature suitability、结构语义 probe、shortcut baseline、per-expert recall、within-dataset summary 和中文摘要 |
| `p2b_visual_pooling/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_visual_pooling/` | visual-only pooling 三变体三 seed 的 selection/diagnostic 汇总、stratified summary、selected model counts、best variant 和中文摘要 |
| `p2c_aux_only/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_aux_only/` | RevIN aux-only 三 seed 的 selection/diagnostic 汇总、stratified summary、selected model counts、best seed 和中文摘要 |
| `p2d_concat/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/` | visual+RevIN aux concat 两变体三 seed 的 selection/diagnostic 汇总、stratified summary、selected model counts、best variant、metadata 和中文摘要 |
| `p2d_final_test/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test/` 与 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/` | P2d best `cls_mean_concat_plus_aux` frozen pilot_test final eval 的 comparison、metadata、中文摘要，以及保留的 final_test_only pilot_test feature cache metadata/summary |
| `p2d_final_test_extension/` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test_extension/` | Round 1 frozen pilot_test eval extension，包含 P2d best、`mean_patch_plus_aux`、`visual_mean_patch_only`、`visual_cls_mean_concat` 与 Round0/global/oracle 的 comparison、delta summary、metadata 和中文摘要 |
| `p2e_film/` | `/data2/syh/Time/run_outputs/2026-06-21_visual_router_v2_round1_film/` | FiLM / aux modulation 两变体三 seed 的 selection/diagnostic 汇总、stratified summary、selected model counts、delta summary、best variant、metadata 和中文摘要；本轮未使用或评估 `pilot_test` |
| `round1_all_variant_comparison.csv` / `round1_all_variant_summary.md` | `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_concat/` | 合并 P1 Round0、P2b、P2c、P2d 的 Round 1 总表和中文总结 |

## 边界

这里只保留适合代码仓库审阅的汇总 CSV、JSON 和 Markdown 文件；没有复制 checkpoint、prediction SQLite、逐样本 prediction CSV、运行 PID 或大规模缓存。

P2probe、P2b、P2c、P2d 和 P2e 均遵守 Round 1 pilot 协议：训练/选择只使用 `pilot_train` 和 `pilot_selection`，`diagnostic_balanced` 仅用于诊断，不使用 `pilot_test` 做模型选择。`p2d_final_test/` 和 `p2d_final_test_extension/` 中的 `pilot_test` 结果只用于冻结后的 final evaluation 和解释性补测；其中记录的 final_test_only feature cache 可以复用，但不得用于模型、seed、epoch、variant 或超参数选择。
