# stage1_vali_test_router

本目录用于保存 Stage 1 主实验代码。

Stage 1 目标：

- 冻结五个专家；
- 在 vali/test 上生成 item-channel-window 级 prediction cache；
- 在 vali 上训练 visual router；
- 在 test 上评估 hard top-1 routing 和 softmax fusion；
- 验证视觉结构先验在同分布设置下是否能提升专家选择或加权融合。

脚本按职责拆分：

- `build_prediction_cache.py`：生成或整理五专家 window-level 预测缓存；
- `build_vit_embeddings.py`：从在线伪图像 2D 张量提取冻结 HF ViT encoder embedding；
- `train_visual_router.py`：训练 TimeFuse-style 小型 MLP visual router，并评估 hard top-1 与 soft fusion；
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
| `stage1_protocol_and_plan.md` | 记录 Stage 1 per-config 主实验协议、Stage 1B 迁移实验设计、已完成事项、当前未完成清单、视觉 encoder 输入口径和下一步任务验收标准 |
| `evaluate_router_baselines.py` | 用 vali split 学 global/dataset/TSF-cell/dataset+TSF-cell 等非视觉 router baseline，并在 test split 上评估 |
| `build_vit_embeddings.py` | 使用 `google/vit-base-patch16-224` 对在线伪图像编码，默认取 `last_hidden_state[:, 0]` CLS token，输出 768 维 embedding manifest 与 npy 缓存 |
| `train_visual_router.py` | 读取 ViT embedding manifest 和 oracle labels，按 `config_name` 独立训练小型 MLP router，输出五专家权重、hard top-1、soft fusion 和 baseline comparison |
| `pilot/` | 保存 Stage 1 正式实验前的数据流、cache schema、oracle label 和 TSF cell enrichment pilot 脚本 |

## 当前 120 Sample Smoke

最近一次 smoke 使用当前扩大版 `96_48_S` 五专家 pilot 的 120 个 `metric=mae` sample_key：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py \
  --local-files-only --batch-size 16

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
  --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv
```

输出目录：

- `experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/`
- `experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/`

当前结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy |
| --- | --- | --- | --- | --- |
| `visual_router_mlp_v1` hard top-1 | 1.013099 | 0.805392 | 0.207707 | 0.350000 |
| `visual_router_mlp_v1_soft_fusion` | 1.022590 | 0.805392 | 0.217198 | NA |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 |

该 smoke 说明最小视觉 MLP router 在当前 60 个 test window 上超过 `global_best_single`，但距离 `oracle_top1` 仍有明显 regret；后续需要扩大样本并诊断过拟合、label 分布和 soft fusion 校准。
