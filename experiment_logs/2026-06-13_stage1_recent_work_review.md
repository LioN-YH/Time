# Stage 1 最近工作梳理

日志日期：2026-06-13 20:52:43 CST

## 目的

梳理 `experiment_logs/` 中最近的实验日志和 `visual_router_experiments/stage1_vali_test_router/` 下的 Stage 1 代码，帮助确认当前 Visual Router Stage 1 的真实进展、有效产物、pilot 限制和下一步优先级。

## 背景

近期工作从五专家 window-level prediction cache、oracle label、非视觉 baseline，快速推进到 TimeFuse-style 结构特征、在线伪图像化、ViT embedding cache 成本估算和 `/data2/syh` 外部输出目录接入。多个方向在 2026-06-12 到 2026-06-13 交错推进，容易把“已完成的可运行 pilot”和“尚未完成的 visual router 主实验”混在一起。

## 操作

1. 阅读 `experiment_logs/README.md` 中 2026-06-12 到 2026-06-13 的 Stage 1 相关总览记录。
2. 阅读 Stage 1 协议和契约文档：
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_cache_contract.md`
3. 阅读 Stage 1 关键代码：
   - `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py`
   - `visual_router_experiments/stage1_vali_test_router/pilot/build_structure_feature_cache_pilot.py`
   - `visual_router_experiments/stage1_vali_test_router/pilot/train_structure_router_pilot.py`
   - `visual_router_experiments/stage1_vali_test_router/pilot/build_online_pseudo_image_pilot.py`
   - `visual_router_experiments/common/prediction_cache_schema.py`
   - `visual_router_experiments/common/pseudo_imageization.py`
4. 抽查最近 Stage 1 输出目录：
   - `experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/`
   - `experiment_logs/run_outputs/2026-06-13_113713_308023_visual_router_stage1_structure_feature_pilot/`
   - `experiment_logs/run_outputs/2026-06-13_134118_592280_visual_router_stage1_online_pseudo_image_pilot/`
5. 阅读近期关键日志：
   - `2026-06-13_stage1_timefuse_structure_feature_router_pilot.md`
   - `2026-06-13_stage1_online_pseudo_image_pilot.md`
   - `2026-06-13_stage1_vit_embedding_cache_cost_estimate.md`
   - `2026-06-13_external_data2_output_root_setup.md`

## 结果

当前 Stage 1 已完成的是前置链路和小规模 pilot，不是完整 visual router 主实验。

已完成并可复核的内容：

1. `96_48_S` 扩大版五专家 prediction cache pilot 已形成 600 条 manifest、120 个 `sample_key`，覆盖 vali/test、`TEST_DATA_MIN`/`TEST_DATA_HOUR`。
2. window-level oracle label、TSF cell enrichment 和非视觉 baseline 已在同一输出目录落盘。
3. 非视觉可部署规则中，当前 `global_best_single` 最好，test MAE 为 `1.055190`；`oracle_top1` test MAE 为 `0.805392`，相对 `global_best_single` 仍有约 `23.67%` 上限空间。
4. Stage 1 主实验口径已经固定为 per-config router：不同 `config_name` 不能共享专家动作空间；跨 config 只能作为 Stage 1B 迁移或 shared encoder 诊断。
5. TimeFuse-derived 单变量结构特征 pilot 已打通：120 行、17 维特征，`StandardScaler + LogisticRegression` 在 `96_48_S` test 上 MAE 为 `1.079743`，略弱于 `global_best_single=1.055190`。
6. 在线伪图像化 pilot 已打通：120 个 `metric=mae` sample_key 均可在线生成 `variant_a=3view` 和 `variant_b=top3fold`，shape 为 `3x224x224`，范围和 finite 校验通过，只保存 index、metadata、latency 和少量 debug PNG，不保存全量 tensor cache。
7. ViT embedding 全量缓存成本已估算：三组 config 的 vali/test 共 `60,743,910` 个 window；单 variant fp16 约 `93.3GB`，双 variant fp16 约 `186.6GB`。在 `/home` 不适合全量缓存；即使接入 `/data2/syh`，也应先做 online 或小 shard 对照。

尚未完成的内容：

1. 正式 per-config prediction cache builder。
2. HF ViT / MAE / CLIP embedding smoke 脚本。
3. embedding cache 与 oracle labels 的正式 `sample_key` 对齐验证。
4. 最小 visual router 的 vali 训练和 test hard top-1 评估。
5. softmax fusion router。
6. 三个 config 的正式 Stage 1 主表。
7. Stage 1B shared encoder / leave-one-config-out 迁移实验。

## 结论

近期工作的核心不是已经训练出 visual router，而是把 Stage 1 的数据契约、非视觉对照、结构特征对照和在线伪图像输入链路逐步打通。当前最可靠的判断是：

- `global_best_single` 是 visual router 需要首先超过的可部署 baseline；
- `oracle_top1` 表明专家选择仍有上限空间；
- TimeFuse-style 单变量结构特征没有超过 `global_best_single`，不应继续作为主线投入；
- 在线伪图像化已经可用，下一步应接冻结视觉 encoder，而不是继续扩大手工结构特征；
- 全量 ViT embedding cache 成本高，下一步应先做 online ViT smoke 或小规模 shard 对照。

## 下一步方案

1. 优先新增 HF ViT embedding smoke，使用在线伪图像化模块，支持 `variant_a/variant_b`、fp16、抽样、`--output-root` 和 `--cache-root`。
2. 明确 HF ViT processor normalization 使用 `(x - 0.5) / 0.5`，不要误用 torchvision ImageNet mean/std。
3. 在 `96_48_S` 上训练最小 visual router，并与 `global_best_single`、TimeFuse 单变量结构特征 router、`oracle_top1` 同表比较。
4. visual router 若仍弱于 `global_best_single`，优先诊断特征、标签分布和样本规模，再决定是否扩大到更多 item/window/config。
