# Stage 1 Visual Router V2 小规模架构优化实验协议整理

日志日期：2026-06-20 13:14:59 CST

## 目的

基于已完成的 `96_48_S` full-scale Visual Router 与 TimeFuse-style baseline 结果，整理 Visual Router 当前架构问题、可验证假设、小规模分轮实验协议、经济性门禁和并行 Git 工作方式，为后续独立实验分支提供可执行依据。

## 背景

当前 Visual Router 1 epoch full-scale eval 覆盖 `13,924,650` 个 test window，hard top-1 MAE 为 `0.5615367653`，raw soft fusion MAE 为 `0.5174675760`。同口径 TimeFuse-style baseline hard top-1 MAE 为 `0.4594660365`，raw soft fusion MAE 为 `0.4473909308`。Visual Router 的 MSE 更低，但 MAE、oracle accuracy、regret 和权重集中度均落后。

已有代码审查还显示：`revin_aux` 计算的窗口尺度 metadata 未进入 router；line/fold/spectrum 三个异质 view 被作为 RGB channel；`L=96` 被插值到 `224x224`；period fold 使用固定候选上的 hard top-1；ViT 默认使用 CLS pooling。这些问题需要先在受控小规模样本上逐项验证，不宜直接重跑 full-scale。

## 操作

1. 复核当前 Visual Router 与 TimeFuse-style full-scale 的同口径 MAE、MSE、regret、oracle accuracy、权重 entropy、mean max weight 和专家选择比例。
2. 阅读当前 `pseudo_imageization.py`、`vit_embedding_utils.py` 和 streaming Visual Router 的图像化、period selection、ViT pooling、head 与 loss 路径。
3. 新增中文协议文档：

   ```text
   visual_router_experiments/stage1_vali_test_router/visual_router_v2_pilot_protocol.md
   ```

4. 在协议中定义：
   - 当前 full-scale 结果和引用边界；
   - 五项主要架构问题及其他混杂因素；
   - 10–20 万 train、2–5 万 selection、5–10 万 test 的建议小规模协议；
   - Round 0–5 的逐轮消融、probe 和 ViT domain adaptation；
   - residual hybrid 与 cost-aware cascade 的经济性门禁；
   - 独立 `git worktree` 与 `exp/visual-router-v2-pilot` 分支建议。
5. 同步更新 Stage 1 README、`WORKSPACE_STRUCTURE.md` 和实验日志总览。

## 结果

已形成一份可执行的小规模研究协议。协议没有把当前诊断写成已确认因果结论，而是把 RevIN aux、异质 channel、图像信息密度、period 连续性、pooling、loss 口径和训练选择机制拆成可验证假设。

协议明确优先执行低成本的 RevIN aux/mean-patch 消融，再比较 spatial panel、独立 view 编码上限、soft period mixture 和更高信息密度 view；只有 frozen representation probe 通过门禁后才进入结构预测、expert regret ranking、masked time-patch 或时序基础模型对齐及有限联合微调。

本轮只新增和更新文档，没有修改正式 Visual Router / TimeFuse 入口、模型、loss、prediction cache、checkpoint 或 `/data2` 结果，也没有启动实验或创建/切换 Git 分支。

## 结论

Visual Router V2 后续可以与 canonical 重构并行，但应通过独立 worktree 隔离 checkout。实验分支优先新增 pilot-only 模块，避免在方案尚未胜出前直接修改共享 pseudo-image 工具或正式 streaming 入口。

## 下一步方案

1. 在共同基线提交上建立独立 `exp/visual-router-v2-pilot` worktree。
2. 先冻结 paired pilot sample keys，并复现旧 Visual 与 TimeFuse-style 的相对趋势。
3. 实现 Round 1 的 `CLS / mean_patch / CLS+mean / RevIN aux-only / visual+aux` 最小消融。
4. Round 1 通过后再实现 spatial panel 和 soft period mixture，避免一次引入过多变量。
5. 只有小规模门禁通过后才申请 full-scale 或 ViT 联合微调资源。
