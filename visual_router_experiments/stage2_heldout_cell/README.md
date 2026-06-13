# stage2_heldout_cell

本目录用于保存 Stage 2 held-out cell 泛化实验代码。

Stage 2 目标：

- 使用 7 个 TSF cell 的 vali/window 样本训练 router；
- 在 1 个 held-out TSF cell 的 test/window 样本评估；
- 8 个 TSF cell 轮流 held out；
- 验证 router 是否学习到可迁移的视觉结构特征，而不是只记住 item 或 cell 分布。

Stage 2 应复用 `common/` 和 Stage 1 已验证的 prediction cache / embedding cache schema。

当前仅建立目录和职责说明，尚未写入正式代码。
