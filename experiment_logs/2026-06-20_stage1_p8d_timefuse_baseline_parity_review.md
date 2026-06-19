# Stage 1 P8d TimeFuse baseline parity review

日志日期：2026-06-20 01:39:06 CST

## 目的

在 P8c evaluation adapter pressure verification 之后，补充 TimeFuse-style fusor baseline 的 parity 文档审计，明确当前 baseline 与原版 TimeFuse 思路之间保留、改造和不可声称的边界。

## 背景

当前 Stage 1 TimeFuse-style fusor baseline 来自 TimeFuse 的动态专家融合思想，但已接入 Time 工作区的单变量 sample-level QuitoBench 五专家 prediction cache、streaming reader 和 evaluation adapter 口径。为了后续实验汇总、论文表述和 refactor 文档不误称“原版 TimeFuse 完全复现”，需要单独记录命名边界。

## 操作

1. 读取 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`，确认正式入口中 `StandardScaler`、`SmoothL1Loss(beta=args.huber_beta)`、weighted fusion loss、streaming train/eval 和 `--verify-evaluation-adapter` 的当前口径。
2. 读取 `visual_router_experiments/stage1_vali_test_router/fusion_utils.py`，确认 `TimeFuseFusor` 使用 `torch.nn.Linear -> torch.softmax` 生成专家权重。
3. 新增 `docs/refactor/timefuse_baseline_parity_review.md`，记录当前 baseline 保留的 TimeFuse-style 核心、有意改变的部分、不能声称和可以声称的表述、正式入口已保持的实现细节，以及后续更强 parity 需要补审的内容。
4. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P8d 文档审计小步。
5. 更新 `docs/refactor/stage1_target_architecture.md`，补充 TimeFuse 分支的 baseline 命名边界。
6. 更新 `docs/refactor/timefuse_entrypoint_adapter_insertion_audit.md`，追加 P8d parity 结论。
7. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 parity review 文档。
8. 更新 `experiment_logs/README.md`，登记本日志和关键结果。
9. 使用 Quito 环境运行目标 smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
   ```

10. 使用 Quito 环境运行 compileall：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py
   ```

## 结果

- 新增 `docs/refactor/timefuse_baseline_parity_review.md`。
- 明确当前 baseline 保留 `meta feature -> linear logits`、softmax expert weights、sample-level adaptive fusion、weighted prediction fusion 和 `SmoothL1Loss` 训练口径。
- 明确当前 baseline 有意从原版多变量 TimeFuse 改造为单变量 sample-level baseline，使用当前 17 维 TimeFuse-derived feature、QuitoBench 五专家、Stage 1 packed prediction cache / streaming reader，以及 `ExpertBatch` / `EvaluationInputAdapter` 可复算的 Time evaluation 口径。
- 明确后续不能声称完全复现原版 TimeFuse，不能直接和原版论文数值一一复现比较，不能把当前 baseline 写成未改造的 TimeFuse。
- 明确后续可以称为 `TimeFuse-style fusor baseline`、`TimeFuse-inspired sample-level adaptive expert fusion baseline` 或 `adapted TimeFuse-style baseline for single-variable QuitoBench expert routing`。
- 本步未修改 `train_timefuse_fusor_streaming.py`、`TimeFuseFusor`、reader、scaler、loss、evaluation adapter 或 Visual Router 入口；未访问 `/data2`，未启动 pressure/full-scale，未新增 smoke 或 Bash/scripts。
- `tests/smoke/stage1_timefuse_protocol_chain_smoke.py` 通过，输出确认链路保序且 deterministic，`hard_mae=1.093573928`，`raw_soft_mae=0.556751269`。
- `compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` 通过。

## 结论

当前 TimeFuse-style fusor baseline 的合理定位是“受 TimeFuse 启发并适配到单变量 QuitoBench 五专家 routing/fusion 的动态加权 baseline”。它保留 linear-softmax sample-level weighted fusion 与 SmoothL1Loss 核心，但不是原版 TimeFuse 的完整复现。

## 下一步方案

小步提交并推送 `refactor/stage1-route-audit`。如果未来需要更强 parity claim，应另起审计原版 TimeFuse 特征定义、训练 split/loss/normalization、多变量处理方式和当前 17 维 feature 的逐项对应关系。
