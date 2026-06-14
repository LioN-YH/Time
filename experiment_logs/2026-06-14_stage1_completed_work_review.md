# Stage 1 当前已完成工作回顾

日志日期：2026-06-14 16:36:27 CST

## 目的

根据用户纠正后的最新实验日志 `experiment_logs/2026-06-14_stage1_online_visual_router_smoke.md` 和代码目录 `visual_router_experiments/stage1_vali_test_router/`，回顾 Stage 1 Visual Router 目前已经完成的工作，明确已经验证的能力、当前主线口径、仍未完成的缺口和下一步优先级。

## 背景

最新实验步骤是 `2026-06-14_stage1_online_visual_router_smoke.md`：该步骤把 Stage 1 路线调整为 online Visual Router，不再先缓存 1k ViT embedding，不启动 1k ViT embedding launcher，也不长期保存伪图像 tensor 或 ViT embedding `.npy`。在此之前，已完成 `fusion_huber_kl` Visual Router 改造、soft fusion 校准、fixed candidates embedding 对照，以及 `96_48_S` 1k manifest 与 launcher 准备；在此之后还有一次代码目录整理，但它是结构清理，不改变最新实验结果口径。

本次回顾不启动新训练或新评估，只做阶段性梳理。

## 操作

1. 阅读用户纠正后的最新实验日志：
   - `experiment_logs/2026-06-14_stage1_online_visual_router_smoke.md`

2. 为理解最新 online smoke 的前置背景，阅读相关连续日志：
   - `experiment_logs/2026-06-14_stage1_fusion_huber_kl_visual_router.md`
   - `experiment_logs/2026-06-14_stage1_soft_fusion_calibration_fixed_candidates.md`
   - `experiment_logs/2026-06-14_stage1_96_48_s_1k_manifest_launchers.md`
   - `experiment_logs/2026-06-14_stage1_code_directory_cleanup.md`

3. 阅读 Stage 1 当前文档：
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `experiment_logs/README.md`

4. 检查 Stage 1 代码目录文件列表，确认当前正式入口与 `pilot/` 边界。

5. 使用 `rg` 汇总代码入口、主要函数和 CLI 参数，确认当前根目录保留的正式入口包括：
   - `build_stage1_sample_manifest.py`
   - `build_prediction_cache_from_manifest.py`
   - `merge_prediction_cache_shards.py`
   - `evaluate_router_baselines.py`
   - `train_visual_router.py`
   - `train_visual_router_online.py`
   - `evaluate_soft_fusion_calibration.py`

6. 检查 `git status --short`，回顾前工作区为干净状态。

## 结果

### 已完成的实验与代码能力

1. Stage 1 协议已基本固化：
   - 主实验口径为 per-config router；
   - 当前先聚焦 `96_48_S`；
   - 训练 split 为 `vali`，评估 split 为 `test`；
   - 路由粒度为 `item_id + channel_id + window_index`；
   - 主线动作空间为同一 config 内五专家：`DLinear`、`PatchTST`、`CrossFormer`、`ES`、`NaiveForecaster`。

2. Prediction cache 与 oracle 前置链路已完成 pilot 到 1k 准备：
   - 早期 120 sample_key 五专家 prediction cache pilot 已完成；
   - window-level oracle label、regret、TSF cell enrichment 和非视觉 baseline 已完成；
   - `96_48_S` 1k manifest-only 清单已生成，`sample_manifest.csv` 为 `1000 x 17`，`sample_key` 无重复；
   - prediction cache shard builder 和 merge 脚本已实现；
   - 8 sample_key DLinear CPU/GPU cache smoke、单专家 merge smoke 已通过。

3. 非视觉 baseline 已作为固定对照纳入：
   - 当前 120 sample_key pilot 中，可部署 baseline 里 `global_best_single` 最好，test MAE 为 `1.055190`；
   - `oracle_top1` test MAE 为 `0.805392`；
   - TimeFuse 单变量结构特征 LogisticRegression router test MAE 为 `1.079743`，弱于 `global_best_single`，因此不再作为主线继续扩复杂特征工程。

4. 伪图像化和视觉 embedding 路径已跑通：
   - 在线伪图像化 120 sample_key 通过 shape、range、finite 校验；
   - HF ViT smoke 已生成 768 维 CLS embedding；
   - `pseudo_imageization.py` 已支持 `hf_vit_0_5` normalization、固定候选周期桶和分桶 fold；
   - fixed candidates 去 warm-up 后图像化 latency 从旧口径 `0.469106 ms/window` 降到 `0.222156 ms/window`，但 120 sample_key 下指标略退化，不能直接视为效果改进。

5. Visual Router 训练已从分类 baseline 推进到权重式 fusion：
   - `train_visual_router.py` 保留 `classification` baseline；
   - 默认 `fusion_huber_kl` 使用五专家预测加权融合的 SmoothL1 主损失，并用 soft oracle distribution 做 KL 辅助；
   - 代表性 `fusion_huber_kl` hard top-1 MAE 为 `0.982425`，优于旧分类 router hard top-1 `1.013099`、`global_best_single=1.055190` 和 TimeFuse 结构特征 router `1.079743`；
   - raw soft fusion MAE 为 `1.085451`，仍弱于 hard top-1 和 `global_best_single`。

6. Soft fusion calibration 已有第一版正式评估脚本：
   - `evaluate_soft_fusion_calibration.py` 支持 temperature scaling、top-k 截断重归一化、raw soft、top1 hard、top2/top3 fusion；
   - 旧代表 router 上最佳 soft calibration 为 `top2_fusion_T0p25`，MAE 为 `0.999014`，超过 `global_best_single` 但仍弱于 hard top-1；
   - 当前结论是 router 排序信号比概率幅度更可靠，概率校准仍需继续改进。

7. Online Visual Router 已成为当前推荐路线：
   - 新增 `train_visual_router_online.py`，在线执行 `x -> pseudo image -> frozen ViT -> CLS embedding -> router`；
   - embedding 只在单次运行内暂存，不保存伪图像 tensor，不保存 ViT embedding `.npy`；
   - 120 sample_key online smoke hard top-1 MAE=`0.982425`、raw soft fusion MAE=`1.085451`、oracle MAE=`0.805392`，与离线代表完全对齐；
   - 输出目录未生成 `.npy`、`embeddings/` 或伪图像 tensor cache。

8. 代码目录边界已整理：
   - Stage 1 根目录保留长期复用正式入口；
   - `pilot/` 保留小规模验证、离线 embedding 历史对照和固定规模 launcher；
   - 离线 ViT embedding cache builder 已移入 `pilot/build_vit_embeddings_pilot.py`；
   - online 主线复用逻辑已上收到 `visual_router_experiments/common/vit_embedding_utils.py`。

### 当前关键判断

1. 120 sample_key 层面，视觉 router 已经证明有超过 `global_best_single` 的信号，尤其是 `fusion_huber_kl` hard top-1。
2. raw soft fusion 目前不可作为主结论，必须通过 temperature/top-k 校准或训练阶段稀疏化继续处理。
3. fixed candidates 周期桶明确提升图像化速度，但会改变部分 embedding 和专家选择，需要在更大样本上同时评估速度和指标稳定性。
4. 当前不应启动 1k ViT embedding launcher，不应长期保存伪图像 tensor 或 ViT embedding cache；后续 1k 路线应先完成五专家 prediction cache，再用 online router 训练。
5. 目前所有强结果仍是 `96_48_S`、120 sample_key smoke，不应外推为三 config 正式结论。

## 结论

截至最新实验步骤 `2026-06-14_stage1_online_visual_router_smoke.md`，Stage 1 已从“前置 cache 和 oracle pilot”推进到“可在线训练的 visual router smoke”。核心链路已经闭合：

```text
sample_key 清单 -> 五专家 prediction cache -> oracle/baseline -> 在线伪图像化 -> 冻结 ViT embedding -> fusion_huber_kl router -> hard top-1 / soft fusion / calibration 评估
```

当前最可信的阶段性结果是：在 120 个 `96_48_S metric=mae` sample_key 上，`fusion_huber_kl` visual router hard top-1 MAE=`0.982425`，相对 `global_best_single=1.055190` 有约 `6.90%` 提升；online 入口能够无长期 embedding cache 地复现该结果。

主要未完成事项是把该结论放大到 `96_48_S` 1k，并建立统一 evaluator/reporter。三 config 正式实验和 Stage 1B 迁移学习仍在后续阶段。

## 下一步方案

1. 用户确认后，优先启动 `96_48_S` 1k 五专家 prediction cache launcher：

   ```text
   bash experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh
   ```

2. 五专家 shard 完成后，执行 merge，重新生成 oracle labels、TSF cell enrichment 和非视觉 baseline。

3. 使用 `train_visual_router_online.py` 在 1k prediction cache 上训练 visual router，不启动 1k ViT embedding launcher。

4. 对 1k 输出统一汇总：
   - `oracle_top1`
   - `global_best_single`
   - TimeFuse 单变量结构特征 router
   - visual hard top-1
   - raw soft fusion
   - best calibrated soft

5. 如果 1k 结果稳定，再考虑扩展到 `576_288_S` 和 `1024_512_S`，并补统一 evaluator/reporter。
