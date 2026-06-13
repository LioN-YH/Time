# stage1_vali_test_router

本目录用于保存 Stage 1 主实验代码。

Stage 1 目标：

- 冻结五个专家；
- 在 vali/test 上生成 item-channel-window 级 prediction cache；
- 在 vali 上训练 visual router；
- 在 test 上评估 hard top-1 routing 和 softmax fusion；
- 验证视觉结构先验在同分布设置下是否能提升专家选择或加权融合。

后续脚本建议按职责拆分：

- `build_prediction_cache.py`：生成或整理五专家 window-level 预测缓存；
- `build_visual_embeddings.py`：从伪图像 2D 张量提取冻结视觉 encoder embedding；
- `train_router.py`：训练 hard/soft router；
- `evaluate_router.py`：基于 prediction cache 评估 test MAE/MSE；
- `summarize_results.py`：汇总结果表和日志摘要。

pilot 脚本统一放入 `pilot/` 子目录；正式可复用的评估、训练、汇总入口保留在 Stage 1 根目录，跨阶段通用逻辑上收到 `visual_router_experiments/common/`。

当前已有文件：

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 将 Stage 1 目录标记为可导入 Python package |
| `prediction_cache_design.md` | 记录 Quito evaluate/data/model 数据流阅读结论，以及 Stage 1 prediction cache 的推荐导出点和 pilot 限制 |
| `feature_and_rl_extension_notes.md` | 记录 TimeFuse-style 结构特征 router 支线、feature scaler 口径，以及视觉路由扩展为 contextual bandit / RL 的可行方案 |
| `stage1_cache_contract.md` | 固定 Stage 1 正式 prediction cache、oracle labels、feature cache 和 router evaluation 的字段契约 |
| `stage1_protocol_and_plan.md` | 记录 Stage 1 per-config 主实验协议、Stage 1B 迁移实验设计、已完成事项和下一步任务 |
| `evaluate_router_baselines.py` | 用 vali split 学 global/dataset/TSF-cell/dataset+TSF-cell 等非视觉 router baseline，并在 test split 上评估 |
| `pilot/` | 保存 Stage 1 正式实验前的数据流、cache schema、oracle label 和 TSF cell enrichment pilot 脚本 |
