# Stage 1 视觉路由主线梳理与 TimeFuse 支线拆分

日志日期：2026-06-19 11:42:04 CST

## 目的

把 Stage 1 的正确路线重新整理为“视觉路由主线优先”，避免 TimeFuse-style fusor baseline 与 visual router 执行顺序混在一起，影响后续扩展到其它 config。

## 背景

此前 `stage1_vali_test_router/README.md` 的当前主线把 TimeFuse feature cache、TimeFuse fusor reader、TimeFuse fusor full-scale launcher 插入 full-scale 执行顺序中。实际实验目标中，visual router 是主线，TimeFuse-style fusor 只是 baseline 支线，应单独追踪，不应作为视觉路由扩配置的前置步骤。

当前 `96_48_S` visual router 已完成：

- full-scale sample manifest；
- 五专家 `packed_npy_v1` prediction cache；
- merged cache 和完整性校验；
- oracle labels 和 TSF enrichment；
- 1 epoch streaming visual router train-only checkpoint；
- checkpoint eval-only。

## 操作

1. 阅读并复核 `HANDOFF.md`、`stage1_protocol_and_plan.md`、`stage1_history_results.md`、`stage1_cache_contract.md`、Stage 1 README 和最近的 visual router / TimeFuse 日志。
2. 新增 `visual_router_experiments/stage1_vali_test_router/stage1_visual_router_mainline.md`，只记录 visual router 主线。
3. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`：
   - 在顶部加入 visual router 主线文档引用；
   - 将当前 full-scale 顺序改为 visual router 闭环；
   - 明确 TimeFuse-style fusor 是 baseline 支线，不再作为视觉路由前置步骤；
   - 将下一步改为 calibration streaming/SQLite 审查、full-scale calibration 和视觉主线报告。
4. 同步更新 `WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`，记录新增长期文档和本次整理。

## 结果

新增主线文档固定了以下路线：

```text
build_full_scale_sample_manifest.py
-> launch_full_scale_prediction_cache.py
-> build_prediction_cache_from_manifest.py
-> merge_prediction_cache_shards.py
-> build_full_scale_window_oracle_labels.py
-> build_full_scale_tsf_enrichment.py
-> validate_full_scale_oracle_tsf_outputs.py
-> evaluate_router_baselines.py
-> train_visual_router_online_streaming.py
-> evaluate_soft_fusion_calibration.py
-> final visual router report
```

文档同时记录了 `96_48_S` 已完成的关键结果：

- train-only checkpoint：`/data2/syh/Time/run_outputs/2026-06-16_stage1_96_48_s_streaming_visual_router_1epoch_v2/checkpoints/latest_96_48_S.pt`；
- eval-only 输出：`/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/`；
- visual hard top-1 MAE=`0.5615367653135453`；
- raw soft fusion MAE=`0.5174675759559787`；
- oracle MAE=`0.33862214116809347`；
- `router_predictions=13,924,650`。

## 结论

Stage 1 当前应按 visual router 主线继续推进。TimeFuse-style fusor baseline 可以继续独立监控和完成，但不应阻塞或污染视觉路由的路线整理、calibration 和扩 config 方案。

## 下一步方案

1. 只读审查 `evaluate_soft_fusion_calibration.py`，确认 full-scale 下是否会全量加载 116M 行 manifest。
2. 如有需要，为 calibration 增加 SQLite/streaming 读取路径。
3. 对已完成的 `96_48_S` eval-only 输出运行 full-scale calibration。
4. 生成视觉主线报告，先汇总 `oracle_top1`、`global_best_single`、visual hard top-1、raw soft 和 best calibrated soft。
5. 再决定追加 `96_48_S` epoch，或复制主线到下一个 config。
