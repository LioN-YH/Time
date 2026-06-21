# Visual Router V2 Round 1 P2probe feature suitability 与结构语义诊断

日志日期：2026-06-21 08:43:33 CST

## 目的

基于已完成的 P2a sharded feature cache 做只读 feature probe，诊断 `cls_embedding`、`mean_patch_embedding`、`cls_mean_concat` 和 `revin_aux` 是否包含 expert suitability 信息、TSF/结构语义信息，以及 dataset / TSF shortcut 风险。该步骤只做诊断，不训练 Visual Router routing head，不做 hard routing / soft fusion 主实验，不重新生成 P2a feature。

## 背景

固定输入包括：

- P0 sample set：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/`
- P1 Round 0 output：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/`
- P2a feature cache：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/`
- full-scale oracle labels 与 TSF enrichment 的 canonical 路径保留在 metadata 中作为来源引用。

并行边界：本轮只新增 P2probe 脚本、P2probe 输出目录和本日志/结构文档，不修改 P2a feature builder/schema，不覆盖 P0/P1/P2a/P2b/P2c 输出。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_round1_features.py`。
   - 只读取 P2a `round1_feature_manifest.csv` 指向的 `.npz` shard。
   - 对每个 sample_set 严格校验 `sample_key` 与 `order_index` 对齐。
   - feature probe 使用 `StandardScaler + class-balanced SGDClassifier(loss="log_loss")`，scaler 和分类器只在 `pilot_train` fit。
   - dataset/TSF shortcut baseline 使用 `OneHotEncoder + class-balanced LogisticRegression`，同样只在 `pilot_train` fit。
   - `pilot_selection` 作为主评估，`diagnostic_balanced` 只做额外诊断；不使用 `pilot_test`。
2. 先用 smoke 命令验证 400 样本/集合的完整输出：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_round1_features.py --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_feature_probe_smoke --max-samples-per-set 400 --max-iter 20 --overwrite
   ```

3. 执行正式 P2probe：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/probe_visual_router_v2_round1_features.py --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_feature_probe --max-iter 20 --overwrite
   ```

4. 修正实现中的两个诊断细节：
   - 小样本或正式子集中的单类结构标签用 `DummyClassifier` 或单类指标分支处理，避免线性分类器无法 fit。
   - 对 SGD dense probe 使用 `decision_function` 手算 stable softmax 概率，避免 sklearn 内部 `predict_proba` 在高维小样本上出现 NaN 概率。
   - confusion matrix 使用显式 label 顺序手工计数，避免单类标签产生无意义 warning。
5. 验证正式输出文件存在、行数符合预期，并检查 metadata 中 `uses_pilot_test=False`、`read_prediction_manifest=False`。

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_feature_probe/
```

已生成核心文件：

- `feature_probe_expert_suitability_results.csv`：12 行，覆盖 6 个 feature group × 2 个 eval sample_set。
- `feature_probe_structure_results.csv`：64 行，覆盖 8 个结构目标 × 4 个必需 feature group × 2 个 eval sample_set。
- `feature_probe_shortcut_baselines.csv`：6 行，覆盖 dataset-only、TSF-only、train frequency prior × 2 个 eval sample_set。
- `feature_probe_confusion_matrices.csv`：2026 行。
- `feature_probe_within_dataset_summary.csv`：32 行。
- `feature_probe_per_expert_recall.csv`：60 行。
- `feature_probe_metadata.json`：`status=completed`，样本数为 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000`。
- `feature_probe_summary.md`：中文回答 6 个验收问题。

`pilot_selection` 主结果摘要：

- expert suitability：`cls_mean_concat` 最好，oracle expert accuracy 为 `0.635067`，macro F1 为 `0.547568`，top-2 recall 为 `0.803033`。
- `mean_patch_embedding` 略优于 `cls_embedding`：accuracy `0.626833` vs `0.623767`，macro F1 `0.535799` vs `0.528439`。
- shortcut baseline 明显较弱：TSF-only accuracy `0.414433`，dataset-only accuracy `0.319867`。
- `revin_aux` 单独预测 oracle expert 有信息但弱于 visual embedding：accuracy `0.566567`，macro F1 `0.421606`。
- 结构语义 probe 显示 visual embedding 能恢复多类结构标签：`forecastability_cat` 最好 accuracy 约 `0.845`，`season_strength_cat` 约 `0.753`，`trend_strength_cat` 约 `0.672`，`cv_cat` 约 `0.663`；`cluster/group_name` 也可被恢复到约 `0.46-0.47` accuracy，提示仍存在一定 shortcut 风险。

## 结论

当前证据支持 visual embedding 含有明显 expert suitability 和结构语义增量，并且优于 dataset-only / TSF-only shortcut baseline。mean patch pooling 在 expert suitability 上小幅优于 CLS，`cls_mean_concat` 是本轮最强单项视觉表示。RevIN aux 单独有信息，但强度低于 visual embedding；visual+aux 两个固定候选没有超过 `cls_mean_concat`，因此 P2d concat 仍值得做，但不应假设 aux 必然带来增益。

同时，visual embedding 对 `cluster/group_name` 的可预测性说明它可能也编码了 dataset/TSF cell 相关 shortcut。后续如果推进 P2d 或 Round 2 imageization/view 消融，应继续加入 held-out dataset / held-out TSF cell 或更严格 group split 诊断。

## 下一步方案

1. P2d visual+aux concat 若启动，应以 `cls_mean_concat`、`mean_patch_embedding`、`revin_aux` 的 probe 结果作为先验，并继续只用 `pilot_train`/`pilot_selection` 做选择。
2. Round 2 view/imageization 消融优先关注 mean-patch 与 concat 表示，保留对 `cluster/group_name` 的 shortcut 诊断。
3. 如果需要更强泛化证据，补做 held-out dataset split probe；当前本轮已满足基础 P2probe 验收，不使用 `pilot_test`。
