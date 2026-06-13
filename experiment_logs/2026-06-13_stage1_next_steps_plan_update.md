# Stage 1 下一步任务梳理与计划文档更新

日志日期：2026-06-13 21:50:32 CST

## 目的

在阅读当前 Stage 1 代码、协议文档和近期实验产物后，更新 `stage1_protocol_and_plan.md`，把当前真正还需要完成的 Stage 1 工作按优先级、输入输出和验收标准重新梳理清楚，避免继续把已完成 pilot、正式主实验和 Stage 1B 迁移实验混在一起。

## 背景

原 `stage1_protocol_and_plan.md` 写于 2026-06-12 晚上，当时在线伪图像化、TimeFuse-derived 结构特征 router、ViT embedding cache 成本估算和 `/data2/syh` 外部输出目录都尚未完成。到 2026-06-13 晚上，Stage 1 的实际状态已经变化：伪图像化和结构特征对照均已打通，但视觉 encoder embedding 和 visual router 训练仍未完成。

此外，近期讨论明确了一个重要工程口径：伪图像化本体可以保持 float `[0, 1]`，但进入不同预训练视觉 encoder 前必须按对应 processor 或 direct-forward 口径处理，不能把 ImageNet mean/std 当成唯一默认路径。

## 操作

1. 重新阅读 `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`。
2. 核对 Stage 1 当前已有代码入口：
   - `evaluate_router_baselines.py`
   - `pilot/build_prediction_cache_pilot.py`
   - `pilot/build_structure_feature_cache_pilot.py`
   - `pilot/train_structure_router_pilot.py`
   - `pilot/build_online_pseudo_image_pilot.py`
   - `common/prediction_cache_schema.py`
   - `common/pseudo_imageization.py`
3. 更新 `stage1_protocol_and_plan.md`：
   - 补充已完成的 TimeFuse-derived 结构特征 router、在线伪图像化、ViT embedding 成本估算和 `/data2/syh` 接入；
   - 重写未完成清单；
   - 将下一步任务重排为 encoder 输入路径、HF ViT embedding smoke、最小 visual router、soft fusion、同表报告、正式脚本、三 config 扩展和 Stage 1B；
   - 为关键步骤补充验收标准。
4. 更新 `visual_router_experiments/stage1_vali_test_router/README.md` 中 `stage1_protocol_and_plan.md` 的说明。
5. 更新 `WORKSPACE_STRUCTURE.md`：
   - 更新时间；
   - 补充 `visual_router_stage1_structure_feature_pilot` 输出目录模式；
   - 更新 Stage 1 目录说明。
6. 更新 `experiment_logs/README.md` 总览表，登记本次计划文档更新。

## 结果

`stage1_protocol_and_plan.md` 当前明确的最近任务顺序为：

1. 明确视觉 encoder 输入路径：
   - 保持伪图像化本体输出 float `[0, 1]`；
   - 区分 `hf_vit_0_5`、`torchvision_imagenet`、`openai_clip` 和 `processor_uint8`；
   - 如果 float `[0, 1]` 直接传 HF processor，需要显式 `do_rescale=False`。
2. 实现 `96_48_S` HF ViT embedding smoke：
   - 优先 `google/vit-base-patch16-224`；
   - 先用当前 120 个 sample_key；
   - 输出 embedding manifest、metadata、latency，并校验 `sample_key` 对齐和 embedding finite。
3. 训练最小 per-config visual router：
   - 先做 `96_48_S`；
   - `vali` fit scaler/router，`test` 评估；
   - 输出 hard top-1 MAE、oracle label accuracy 和 regret。
4. 增加 softmax fusion 评估：
   - 只融合相同 `config_name`、相同 `sample_key` 下的五专家预测；
   - 权重来自 router 的 test 概率；
   - 不使用 test oracle error 调整权重。
5. 形成同表报告：
   - `global_best_single`
   - dataset/TSF-cell baseline
   - TimeFuse 单变量结构特征 router
   - `oracle_top1`
   - `visual_router_top1`
   - `visual_router_soft_fusion`
6. 在 `96_48_S` 最小闭环稳定后，正式化 Stage 1 脚本，再扩展到三个 config。
7. Stage 1B shared encoder / leave-one-config-out 迁移实验不再列为最近任务，需等 Stage 1 主实验稳定后再做。

## 结论

当前 Stage 1 的下一步重点已经从“是否能生成视觉输入”转为“如何正确接入冻结视觉 encoder 并训练最小 visual router”。最近不应直接启动全量 embedding cache 或 Stage 1B 迁移实验；应先用 `96_48_S` 的 120 个 sample_key 完成 HF ViT embedding smoke 和最小 visual router，同表比较 `global_best_single`、TimeFuse 结构特征 router、visual router 和 `oracle_top1`。

## 下一步方案

1. 修改或新增 encoder normalization 工具，明确 direct-forward 与 processor 路径。
2. 新增 HF ViT embedding smoke 脚本，优先支持 `google/vit-base-patch16-224`、fp16、`variant_a`、`--output-root` 和 `--cache-root`。
3. embedding smoke 通过后，训练 `96_48_S` 最小 visual router，并生成 hard top-1 与 soft fusion 的同表结果。
