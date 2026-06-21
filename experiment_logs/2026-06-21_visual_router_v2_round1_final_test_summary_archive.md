# Visual Router V2 Round 1 Final Test Summary 入仓

日志日期：2026-06-21 15:45:00 CST

## 目的

将 `/data2` 下 P2d frozen final eval 的重要概括性结果文件复制到仓库内 `experiment_summaries/visual_router_v2_round1/`，便于远程仓库中的 GPT 或人工 reviewer 直接查看，不必访问服务器外部输出目录。

## 背景

P2d best frozen final eval 已完成，正式输出位于：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_final_test/
```

pilot_test final_test_only feature cache 位于：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_final_test_only/
```

用户建议把重要概括性结果文件上传到远程仓库 `Time/experiment_summaries/visual_router_v2_round1/` 下。

## 操作

1. 新建仓库内目录：

   ```text
   experiment_summaries/visual_router_v2_round1/p2d_final_test/
   ```

2. 从 final eval 输出目录复制：
   - `round1_final_test_summary.md`
   - `round1_final_test_comparison.csv`
   - `round1_final_test_metadata.json`

3. 从 final_test_only feature cache 输出目录复制并重命名：
   - `round1_feature_summary.md` -> `final_test_only_feature_summary.md`
   - `round1_feature_metadata.json` -> `final_test_only_feature_metadata.json`

4. 更新 `experiment_summaries/visual_router_v2_round1/README.md`，登记 `p2d_final_test/` 的来源、内容和边界。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记 `experiment_summaries/visual_router_v2_round1/` 作为轻量汇总结果目录。

## 结果

仓库内新增/更新的 summary 文件：

```text
experiment_summaries/visual_router_v2_round1/p2d_final_test/round1_final_test_summary.md
experiment_summaries/visual_router_v2_round1/p2d_final_test/round1_final_test_comparison.csv
experiment_summaries/visual_router_v2_round1/p2d_final_test/round1_final_test_metadata.json
experiment_summaries/visual_router_v2_round1/p2d_final_test/final_test_only_feature_summary.md
experiment_summaries/visual_router_v2_round1/p2d_final_test/final_test_only_feature_metadata.json
```

这些文件均为轻量 CSV/JSON/Markdown 概括性结果，不包含 checkpoint、SQLite、逐样本 prediction CSV、`.npz` feature shard 或大规模 cache。

## 结论

P2d final eval 的关键结果已经进入仓库 summary 区，可随 git 远程同步。该目录只作为结果审阅入口，正式可复核来源仍以 `/data2` 输出目录和实验日志记录的路径为准。

## 下一步方案

1. 后续新增 Round 2 或 P2e 结果时，继续把轻量 summary 文件同步到 `experiment_summaries/visual_router_v2_round1/` 或新的阶段目录。
2. 不把 checkpoint、SQLite、逐样本预测、feature shard 或大规模缓存提交到仓库。
3. 若 final eval 复跑，应同步更新 `p2d_final_test/` 中的轻量 summary，并在 README 中注明新旧结果口径。
