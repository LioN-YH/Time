# 视觉结构先验 Router/MoE 研究路线制定

日志日期：2026-06-12 00:50:10 CST

## 目的

制定一条从“视觉结构先验能否帮助专家选择”到“视觉先验同时调控时序专家内部表示”的研究路线，并明确 Phase 1 第一项可执行检查。

## 背景

当前已有 DLinear、PatchTST、CrossFormer、ES、SNaive 五个专家在 `96_48_S`、`576_288_S`、`1024_512_S` 三组配置下的统一汇总结果。五模型 TSF cell 表已经显示不同 cell 的最优模型并不完全一致，但直接训练 visual router 前仍需要先确认专家池存在足够 oracle 上限，否则 router 只能学习到弱收益或常数策略。

可用输入目录为：

```text
experiment_logs/run_outputs/2026-06-11_230450_825063_five_model_three_config_summary/
```

## 操作

1. 梳理视觉结构先验专家系统的阶段化路线。
2. 将第一步从“直接训练 router”调整为“专家互补性与 oracle gap 审计”。
3. 明确当前只使用已有 per-item 聚合结果，不生成窗口级预测缓存。
4. 约定后续 Phase 1 主协议优先采用 7-cell -> held-out-cell 的 zero-shot generalization 评估。

## 总体路线

### Phase 1：Visual Router over Frozen Experts

目标是冻结现有时序专家，只训练视觉结构先验到专家偏好的路由器。输入可以是 period 图、trend/seasonal 分解图或多尺度时序图像；输出是专家选择概率或专家 ranking。

关键验证问题：

- 视觉结构先验是否能预测 item/cell 对不同专家的偏好。
- router 是否优于 best single expert 和 cell-level 常数选择策略。
- 在 held-out TSF cell 上是否仍有泛化收益。

第一步不训练模型，而是先审计 per-item oracle top-1 是否明显优于 best single。

### Phase 1.5：ViT Time-Series Image Domain Adaptation

目标是缓解自然图像 ViT 与时序图像之间的 domain gap。可选路线包括：

- 使用大量无标签时序图像做 masked image modeling 或 DINO 风格自监督适配。
- 对 period 图、多尺度折线图、频谱图做轻量 contrastive adaptation。
- 对比 ImageNet ViT、CLIP image encoder、从头小型 CNN/ViT 在 router 任务上的差异。

该阶段应在 Phase 1 证明 oracle 上限存在后再做，避免先优化视觉 encoder 却没有可学习的专家互补信号。

### Phase 2：Visual-Conditioned PatchTST

目标是让视觉结构向量进入单个时序专家内部，而不是只在专家之间选择。候选注入方式：

- 作为 PatchTST encoder 的 adapter 条件向量。
- 作为 RevIN、patch embedding 或 attention block 的 FiLM/gating 条件。
- 作为预测头前的低秩条件偏置。

验证重点是视觉条件是否提升单专家在不同 TSF cell 的稳定性，而不是依赖多个专家的 oracle 选择。

### Phase 2+：Visual-Conditioned MoE

目标是把同一个视觉先验同时用于：

- router 权重：决定不同专家/adapter 的混合比例。
- expert adapter：调控每个专家内部的条件化参数。

该阶段需要窗口级 prediction cache 或可联合训练的专家框架，适合在 Phase 1/2 都看到正信号后推进。

## 结果

当前决策是先执行 Phase 1 的专家互补性审计，输出：

- 配置级 best single expert 与 per-item oracle top-1 的 MAE/MSE gap。
- 每个 TSF cell 的专家胜率分布。
- 每个专家相对 oracle 的平均 regret。
- 是否满足进入 visual router 训练的接受标准。

接受标准：

- 至少部分配置或 TSF cell 上 oracle top-1 明显优于 best single。
- 不同 TSF cell 的专家胜率分布存在差异。
- 如果 oracle gap 很小，则优先扩充专家池或重新选择专家，而不是马上训练 visual router。

## 结论

路线已确定为 Phase 1 -> Phase 1.5 -> Phase 2 -> Phase 2+。当前最小可执行步骤是用已有五专家三配置 per-item 结果做 oracle 上限审计，以确认 visual router 是否有可学习目标。

## 下一步方案

1. 新增并运行 `experiment_scripts/audit_visual_router_phase1_oracle.py`。
2. 将审计结果写入新的 `experiment_logs/run_outputs/*_visual_router_phase1_oracle_audit/` 目录。
3. 如果 oracle gap 足够，下一步构造视觉图像特征与 oracle/regret label；如果不足，先纳入 TSMixer/iTransformer 或重新训练专家池。
