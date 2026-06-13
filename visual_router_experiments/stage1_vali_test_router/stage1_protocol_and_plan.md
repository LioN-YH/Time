# Stage 1 Protocol And Plan

记录日期：2026-06-12 23:35:44 CST

最近更新：2026-06-14 01:09:37 CST

## 目标

Stage 1 验证同分布设置下的 visual router 是否能在冻结专家之间做有效选择或融合。训练 split 为 `vali`，评估 split 为 `test`，路由粒度为 `item_id + channel_id + window_index`。

核心问题：

1. visual/structure feature 是否能预测 window-level 最优专家；
2. visual router 是否优于不使用视觉输入的 metadata/statistics baseline；
3. visual router 与 `oracle_top1` 之间还剩多少 regret；
4. 不同历史长度和预测长度 config 是否需要独立 router，以及结构表示能否跨 config 迁移。

## 主实验口径：Per-Config Router

主实验必须按 `config_name` 分开训练和评估，不允许一个 router 在不同历史-未来设置之间自由选择专家。

原因：

- `96_48_S`、`576_288_S`、`1024_512_S` 的输入长度、输出长度和 checkpoint 均不同；
- 对 `96_48_S` 样本，合法动作空间只能是同 config 下的五个专家；
- 即使某个 `1024_512_S` 专家族在结构上看起来更适合，也不能直接处理 `96_48_S` 的输入输出。

每个 config 的动作空间定义为：

```text
{DLinear, PatchTST, CrossFormer, ES, NaiveForecaster} within the same config_name
```

推荐主实验实例：

```text
router_96_48_S
router_576_288_S
router_1024_512_S
```

每个 router 独立使用该 config 的 `vali` window 训练，并在该 config 的 `test` window 上评估。

## Stage 1B：跨 Config 迁移实验

跨 config 数据可以用于迁移学习或表征学习，但不作为主实验的可部署 router。

推荐做法：

1. **Shared encoder + config-specific heads**

   ```text
   shared_visual_or_structure_encoder
   ├── router_head_96_48_S
   ├── router_head_576_288_S
   └── router_head_1024_512_S
   ```

2. **Leave-one-config-out**

   留出一个 config 不参与 encoder 预训练，用其他 config 学结构表示；在留出 config 上训练轻量 head 或少量 fine-tune，再评估 test。

   例子：

   ```text
   预训练 encoder: 576_288_S + 1024_512_S
   留出 config: 96_48_S
   留出 config 上训练: router_head_96_48_S
   留出 config 上评估: 96_48_S test
   ```

3. **不推荐作为主结论的 zero-shot head 迁移**

   直接把其他 config 训练出的完整 router head 用到留出 config，只能作为诊断实验。除非把 label 明确定义为专家族，例如 `PatchTST family -> PatchTST_96_48_S`，否则该设置不具备严格部署含义。

## 已完成

当前已完成的 Stage 1 前置链路和 pilot 级别流程：

- Stage 1 代码目录和 `pilot/` 子目录整理；
- prediction cache schema 设计；
- `96_48_S` 小规模到扩大版五专家 `vali/test` prediction cache pilot；
- window-level oracle label 和 regret 计算；
- TSF cell enrichment；
- 非视觉 router baseline 评估；
- baseline 输出包括整体、dataset、TSF cell、dataset+TSF cell 汇总。
- `evaluate_router_baselines.py` 已更新为按 `config_name` 独立训练 baseline，并输出 config-level 主汇总与 macro average。
- `stage1_cache_contract.md` 已固定正式 prediction cache、oracle labels、feature cache 和 router evaluation 的字段契约。
- TimeFuse-derived 单变量结构特征 cache pilot 已完成，覆盖当前 `96_48_S` pilot 的 120 个 `metric=mae` sample_key。
- TimeFuse-derived 结构特征 LogisticRegression router pilot 已完成，可作为轻量非视觉对照。
- 在线伪图像化 pilot 已完成，可从 Quito 历史窗口 `x` 在线生成 `variant_a=3view` 和 `variant_b=top3fold`，并记录 index、metadata、latency 和少量 debug PNG。
- ViT embedding cache 成本已估算，确认不应在 `/home` 直接做全量 embedding cache。
- `/data2/syh/Time/run_outputs/` 和 `/data2/syh/Time/cache_shards/` 已接入为后续大规模输出或临时 shard 的可选位置。
- `build_vit_embeddings.py` 已实现为 Stage 1 根目录正式入口，默认使用 `google/vit-base-patch16-224`、direct `pixel_values`、`hf_vit_0_5` normalization 和 `last_hidden_state[:, 0]` CLS token 输出 768 维视觉特征。
- `train_visual_router.py` 已实现为 Stage 1 根目录正式入口，参考 TimeFuse fusor 设计，用 `StandardScaler` + 小型 MLP 输出五专家 softmax 权重，并同时评估 hard top-1 routing 与 soft fusion。
- 当前 120 个 `96_48_S metric=mae` sample_key 的 ViT embedding smoke 已完成，manifest 与 oracle labels 的 sample_key 集合完全一致。
- 当前 120 个 sample_key 的最小 visual router smoke 已完成，训练 split 为 60 个 vali window，测试 split 为 60 个 test window。

当前关键 pilot 结果：

- 可部署非视觉 baseline 中 `global_best_single` 最好，test MAE 为 `1.055190`；
- `oracle_top1` test MAE 为 `0.805392`；
- 当前 pilot 中 dataset/TSF-cell shortcut 未超过全局单专家。
- TimeFuse 单变量结构特征 router test MAE 为 `1.079743`，略弱于 `global_best_single`，不应继续作为主线投入复杂特征工程。
- 在线伪图像化 120 个 sample_key 全部通过 shape/range/finite 校验，默认 `image_size=224`、`norm_mode=revin_aux`、`pixel_mode=vision`、`clip=5.0`。
- 三组 config 的 vali/test 共 `60,743,910` 个 window；单 variant fp16 ViT embedding 全量约 `93.3GB`，双 variant fp16 约 `186.6GB`。
- ViT embedding smoke 输出 `120` 行 manifest，`embedding_dim=768`，`variant=variant_a_3view`，`pooling=cls`，`forward_dtype=float16`，`saved_dtype=float32`。
- Visual Router hard top-1 在当前 60 个 test window 上 MAE 为 `1.013099`，oracle MAE 为 `0.805392`，regret 为 `0.207707`，oracle label accuracy 为 `0.350000`。
- Visual Router soft fusion MAE 为 `1.022590`，略弱于 hard top-1，但仍优于当前 `global_best_single=1.055190`。

## 未完成

主实验仍未完成以下部分：

1. 正式 per-config cache builder；
2. processor/direct-forward 双路径的统一测试仍不完整，目前已跑通的是 direct `pixel_values` + `hf_vit_0_5`；
3. MAE/CLIP embedding smoke 尚未完成，当前只完成 HF ViT smoke；
4. soft fusion 权重校准仍粗糙，当前 `soft_fusion_mae=1.022590` 弱于 hard top-1；
5. Stage 1 正式脚本仍缺 prediction cache builder、统一 evaluator 和 summarizer；
6. 扩展到三个 config 的正式 summary/report；
7. Stage 1B 迁移学习和 leave-one-config-out 评估。

## 下一步任务与状态

### 1. 明确视觉 encoder 输入路径

状态：direct `pixel_values` 路径已用于 2026-06-14 smoke；HF processor / `processor_uint8` 路径仍待补充单元级 smoke。

目标：先消除伪图像 `[0, 1]`、TimeVLM-style `0..255 uint8`、HF processor 和 direct `pixel_values` 之间的口径歧义。

推荐实现：

- 保持 `pseudo_imageization.py` 的图像化本体输出为 float `[0, 1]`；
- 新增或扩展 encoder normalization 工具，显式区分：
  - `hf_vit_0_5`：用于 `google/vit-base-patch16-224` direct forward，执行 `(x - 0.5) / 0.5`；
  - `torchvision_imagenet`：用于 torchvision/MAE/timm ImageNet mean/std 口径；
  - `openai_clip`：用于 OpenAI CLIP mean/std 口径；
  - `processor_uint8`：用于 TimeVLM-style 路径，先把 `[0, 1]` 转 `0..255 uint8`，再交给 HF processor 处理。
- 如果把 float `[0, 1]` 直接传给 HF processor，必须显式设置 `do_rescale=False`，避免 processor 再除以 255。

验收标准：

- 单元级 smoke 覆盖 direct-forward 和 processor 两条路径；
- metadata 记录 `encoder_name`、`input_mode`、`normalization_preset`、`processor_do_rescale`；
- 不再把 `imagenet_normalize()` 当作所有 encoder 的默认入口。

### 2. 实现 `96_48_S` HF ViT Embedding Smoke

状态：已完成 `variant_a=3view` + `google/vit-base-patch16-224` + CLS pooling smoke，输出目录为 `experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/`。

目标：基于当前 120 个 sample_key 的在线伪图像化结果，跑通冻结视觉 encoder embedding。

第一版范围：

- config：`96_48_S`；
- 样本：当前扩大版 pilot 的 120 个 `metric=mae` sample_key；
- encoder：优先 `google/vit-base-patch16-224`；
- variant：先做 `variant_a=3view`，再可选 `variant_b=top3fold`；
- dtype：优先 fp16；
- 输出根目录：默认 `experiment_logs/run_outputs/`，可通过 `--output-root` 写到 `/data2/syh/Time/run_outputs/`；
- cache 根目录：如需写 shard，通过 `--cache-root` 指向 `/data2/syh/Time/cache_shards/`。

验收标准：

- 输出 embedding manifest，字段包含 `sample_key`、`config_name`、`split`、`variant`、`encoder_name`、`embedding_path`、`embedding_dim`、`dtype`；
- embedding shape 与 encoder hidden size 一致，`google/vit-base-patch16-224` 预期为 768；
- 所有 embedding finite；
- 与 `window_oracle_labels_with_tsf_cell.csv` 的 `sample_key` 集合完全一致；
- 输出 latency summary，区分 imageization、encoder forward、write/read。

### 3. 训练最小 Per-Config Visual Router

状态：已完成小型 MLP 版本，输出目录为 `experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/`。

先做 `96_48_S`：

- 输入：`vali` visual embedding；
- 标签：同 config 内 `oracle_model`；
- 输出：五专家概率或 top-1；
- 评估：`test` hard top-1 MAE、oracle label accuracy、regret。

第一版模型建议：

- `StandardScaler + LogisticRegression(class_weight='balanced')` 或小型 MLP；
- scaler 和 router 只能在 `vali` split fit；
- test 只做 transform/predict/evaluate；
- 若 `vali` label 类别过少，必须在 summary 中显式记录，不能静默退化。

验收标准：

- 输出 `visual_router_predictions.csv`；
- 输出 `visual_router_summary.csv`；
- summary 至少包含 `selected_value`、`oracle_value`、`regret_to_oracle`、`oracle_label_accuracy`；
- 结果与非视觉 baseline 同表比较。

### 4. 增加 Softmax Fusion 评估

状态：已完成基于 router softmax 权重的五专家预测数组加权融合 smoke；当前 soft fusion 弱于 hard top-1，需要后续做温度、正则化或 reward-aware 目标诊断。

目标：不只评估 hard top-1，还评估 router 概率对五专家预测的加权融合是否能降低 MAE/MSE。

约束：

- soft fusion 只能融合同一 `config_name`、同一 `sample_key` 下五专家的 `y_pred`；
- 融合权重来自 vali 训练出的 router 在 test 上的输出概率；
- 不允许使用 test oracle error 调整权重。

验收标准：

- 输出 `soft_fusion_predictions.csv` 或等价文件；
- 对每个 test sample 记录五专家权重、融合后 MAE/MSE、hard top-1 MAE/MSE、oracle MAE/MSE；
- summary 同时给出 hard top-1 和 soft fusion。

### 5. 同表报告

每个 config 的报告至少包含：

- `global_best_single`
- `dataset_only`
- `tsf_cell_only`
- `dataset_tsf_cell`
- `global_majority_label`
- `timefuse_single_variable_logistic_regression`
- `oracle_top1`
- `visual_router_top1`
- `visual_router_soft_fusion`

如果 visual router 没有超过 `global_best_single`，优先诊断 feature、label 分布和 shortcut，而不是扩大规模。

### 6. 正式化 Stage 1 脚本

当前很多流程仍在 `pilot/`。在 `96_48_S` 最小闭环跑通后，应把长期复用逻辑迁出或重写为正式入口：

- `build_visual_embeddings.py`
- `train_router.py`
- `evaluate_router.py`
- `summarize_results.py`

正式脚本要求：

- 支持 `--config-name`；
- 支持 `--labels-path`、`--manifest-path`、`--output-root`、`--cache-root`；
- 所有输出写入 `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_*/` 或 `/data2/syh/Time/run_outputs/`；
- 代码目录只保存源码、协议文档、README 和小型 schema 文档。

### 7. 扩展到三个 Config

在 `96_48_S` 链路稳定后，扩展到：

- `576_288_S`
- `1024_512_S`

每个 config 单独训练和评估，同时保留同一套 summary schema。

扩展前置条件：

- `96_48_S` visual router 至少完成 hard top-1 和 soft fusion 两种评估；
- 与 `global_best_single`、TimeFuse 结构特征 router 和 `oracle_top1` 的差距已经诊断清楚；
- embedding 路径已经证明不会因 normalization 或 processor rescale 造成输入尺度错误。

### 8. 设计 Stage 1B 迁移实验

主实验稳定后再做：

- shared encoder + config-specific heads；
- freeze encoder，只训练留出 config head；
- leave-one-config-out 诊断。

Stage 1B 不作为当前最近任务，除非三个 config 的 Stage 1 主实验已经有稳定结果。

## 输出要求

正式 Stage 1 输出目录继续使用：

```text
experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_stage1_*/
```

代码目录只保存源码、协议文档、README 和小型 schema 文档；prediction cache、embedding cache、checkpoint、评估结果和运行日志不得写入代码目录。
