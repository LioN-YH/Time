# Stage 1 README 导航化与历史结果拆分

日志日期：2026-06-15 23:07:21 CST

## 目的

在完成 Stage 1 代码文件功能梳理后，对 `visual_router_experiments/stage1_vali_test_router/README.md` 做文档层面的清理，使后续打开目录时优先看到当前 full-scale 主线、可执行入口和不要混用的历史路线，而不是被 120 sample smoke、1k 结果和历史 launcher 细节淹没。

## 背景

此前 `stage1_vali_test_router/README.md` 同时承担目录说明、脚本职责、历史实验结果、命令记录、1k 结果、full-scale 状态等多重角色。用户确认可以继续整理后，本次选择只改文档、不移动代码、不改 Python 实验逻辑，避免影响当前 full-scale 后续 merge/oracle/router/calibration 推进。

## 操作

1. 重写 `visual_router_experiments/stage1_vali_test_router/README.md`：
   - 保留 Stage 1 基本目标和约束；
   - 明确当前 full-scale 主线顺序；
   - 将文件按 full-scale 正式入口、中小规模复现入口、共享库、文档、pilot 目录分层；
   - 增加“当前不要混用的路线”和下一步执行清单。
2. 新增 `visual_router_experiments/stage1_vali_test_router/stage1_history_results.md`：
   - 保存从 README 拆出的 TimeFuse-style fusor baseline、120 sample 离线/online smoke、soft fusion calibration、fixed candidates 对照、`96_48_S` 1k、full-scale dry-run 和 full-scale 正式长跑状态；
   - 保留关键输出目录和代表指标，避免历史信息丢失。
3. 更新 `WORKSPACE_STRUCTURE.md`：
   - 将更新日期改为 `2026-06-15 23:07:21 CST`；
   - 在 Stage 1 主实验目录说明中登记 `stage1_history_results.md`；
   - 说明 README 现在是当前主线导航页。
4. 本次未修改 Python 脚本、未移动代码文件、未运行训练或评估实验。

## 结果

完成以下长期文档变化：

- `stage1_vali_test_router/README.md` 从长篇历史结果混合文档改为当前执行导航；
- 新增 `stage1_vali_test_router/stage1_history_results.md` 作为历史结果索引；
- `WORKSPACE_STRUCTURE.md` 已同步记录新增文档和 README 职责变化。

整理后的 README 明确当前 full-scale 后续应从以下链路继续：

```text
merge_prediction_cache_shards.py
-> pilot/compute_window_oracle_from_cache.py
-> pilot/enrich_cache_with_tsf_cell.py
-> evaluate_router_baselines.py
-> train_visual_router_online_streaming.py
-> evaluate_soft_fusion_calibration.py
-> final unified report
```

并明确不应把离线 ViT embedding cache、`train_visual_router_online.py`、旧 LogisticRegression 结构特征 router 或 `launcher_compat_check/` 误当成当前 full-scale 主线。

## 结论

Stage 1 目录的可读性主要问题已经通过文档分层缓解：README 现在回答“当前应该跑什么、哪些入口是正式主线、哪些入口不要混用”，历史结果则集中到独立文档留痕。这样后续推进 full-scale merge 和 router 评估时，不需要在 README 中反复穿越早期 smoke 细节。

## 下一步方案

1. 若继续推进实验，优先合并 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/` 下的五专家 completed shards。
2. 合并后执行完整性校验、oracle labels、TSF cell enrichment、baseline/fusor、streaming visual router 和 calibration。
3. 若后续还要进一步整理代码目录，可考虑补一个正式 unified reporter，而不是继续在 README 里堆实验结果。
