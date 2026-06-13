# Stage 1 结构特征与 RL 支线扩展文档

日志日期：2026-06-13 11:07:20 CST

## 目的

记录 Visual Router Stage 1 后续可扩展的两条支线：TimeFuse-style 统计/结构特征 router，以及将当前动作/状态路由进一步形式化为 contextual bandit 或强化学习的可能方案。

## 背景

当前主线目标是视觉路由：将 item-channel-window 粒度的单变量历史窗口图像化，交给 ViT 或其他视觉 encoder 编码，再训练 router 在冻结专家之间做选择或融合。用户希望后续能与 TimeFuse 比较，但统计特征路由不是主线，因此需要把结构特征设计控制为克制的对照支线。同时，前序讨论认为当前 router 虽然可以写成状态、动作、奖励形式，但没有真实状态转移，更接近 supervised routing 或 contextual bandit，而不是完整强化学习。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/feature_and_rl_extension_notes.md`。
2. 在文档中记录以下结构特征数据流：

   ```text
   raw time series
   -> Quito train-based normalization
   -> x window

   从 x 提取：
     A. train-normalized statistical/shape features
     B. window-RevIN 后的 shape features
     C. RevIN mean/std/range 等 scale features

   concat(A, B, C)
   -> feature scaler fit on vali
   -> train router on vali
   -> evaluate on test
   ```

3. 明确该支线为了与 TimeFuse 思路对照，优先保留单变量窗口特征，暂时剔除多变量专属特征，例如 covariance、cross-correlation 和变量间 spectral variation。
4. 记录 feature scaler 应在 `concat(A, B, C)` 后按列用 vali split fit，再对 test split transform，避免 test leakage。
5. 记录当前动作/状态路由更适合称为 contextual bandit / supervised router，而不是完整 RL。
6. 展开视觉多步自适应支线，包括视觉证据逐步获取、专家级联与早退、图像化策略自适应、跨相邻窗口序列化路由等方向。
7. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`，加入新文档索引。
8. 更新 `WORKSPACE_STRUCTURE.md`，记录 Stage 1 目录新增长期保留文档。

## 结果

新增的支线扩展文档已经放入 Stage 1 主实验目录，并明确区分：

- 主线：Visual Router over Frozen Experts；
- 对照：TimeFuse-style statistical feature router；
- 支线：Contextual Bandit Router Policy；
- 远期：Cost-aware Visual Sequential Routing / Expert Cascade。

文档也固定了统计/结构特征支线的 leakage 约束：Quito normalization 只能使用训练统计量，feature scaler 只能在 router 的 vali split 上 fit，test 只能 transform。

## 结论

统计/结构特征 router 可以作为 TimeFuse-style 对照和 ablation，但不应成为当前主线。当前视觉路由若只做单步专家选择，更适合监督学习或 contextual bandit；真正有价值的 RL 扩展应引入预算、逐步视觉证据获取、专家级联或早退机制，使动作影响后续状态和计算成本。

## 下一步方案

1. 继续优先实现 `96_48_S` visual/structure feature cache，并用 `sample_key` 对齐 oracle labels。
2. 统计特征 router 第一版保持轻量，先做单变量 TimeFuse-style 特征，不追求复杂多变量 meta-feature。
3. 视觉路由主结果稳定后，再考虑把 reward/regret 口径整理成 contextual bandit 支线实验。
