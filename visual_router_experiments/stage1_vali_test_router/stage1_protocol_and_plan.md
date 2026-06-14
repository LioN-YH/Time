# Stage 1 Protocol And Plan

记录日期：2026-06-12 23:35:44 CST

最近更新：2026-06-14 14:23:47 CST

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
- 历史离线 embedding cache 入口已收拢到 `pilot/build_vit_embeddings_pilot.py`，默认使用 `google/vit-base-patch16-224`、direct `pixel_values`、`hf_vit_0_5` normalization 和 `last_hidden_state[:, 0]` CLS token 输出 768 维视觉特征；该脚本会写 embedding manifest 和 `.npy`，当前只作为历史 smoke / 小规模缓存对照保留。
- `visual_router_experiments/common/vit_embedding_utils.py` 已抽出为 online 主线共享工具，提供运行内 `make_pseudo_images()`、`pool_vit_outputs()`、dtype 解析和窗口 batch 索引工具；伪图像化路径已新增固定候选周期桶口径，减少逐样本 `.item()`/`.tolist()` 同步。
- `train_visual_router.py` 已实现为 Stage 1 根目录正式入口，参考 TimeFuse fusor 设计，用 `StandardScaler` + 小型 MLP 输出五专家 softmax 权重，并同时评估 hard top-1 routing 与 soft fusion；当前默认训练模式为 `fusion_huber_kl`，保留 `classification` 作为旧版 oracle hard-label baseline。
- `evaluate_soft_fusion_calibration.py` 已实现为 Stage 1 根目录正式入口，可基于已有 router test 权重评估 raw soft、top1 hard、top2/top3 fusion、temperature sweep 和 top-k 截断重归一化，并输出 entropy、max-weight、active weight count 与 selected-model 分布诊断。
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
- 旧版分类 Visual Router hard top-1 在当前 60 个 test window 上 MAE 为 `1.013099`，oracle MAE 为 `0.805392`，regret 为 `0.207707`，oracle label accuracy 为 `0.350000`。
- 旧版分类 Visual Router soft fusion MAE 为 `1.022590`，略弱于 hard top-1，但仍优于当前 `global_best_single=1.055190`。
- 新版 `fusion_huber_kl` Visual Router 代表 smoke 使用 `SmoothL1Loss(beta=0.1)` 主损失、`tau=0.1`、`lambda_kl=0.01`、`dropout=0.0`；hard top-1 MAE 为 `0.982425`，oracle MAE 为 `0.805392`，regret 为 `0.177033`，oracle label accuracy 为 `0.466667`，优于旧版分类 router 和 `global_best_single`。
- 新版 `fusion_huber_kl` soft fusion MAE 为 `1.085451`，弱于 hard top-1 和 `global_best_single`；权重归一化熵为 `0.757180`、平均最大权重为 `0.483784`，说明当前权重仍偏平滑，soft fusion 校准还不是可用结论。
- 基于旧代表 `fusion_huber_kl` router 的 calibration smoke 已完成：`top2_fusion_T0p25` MAE 为 `0.999014`，`soft_T0p25` MAE 为 `1.000585`，均超过 `global_best_single=1.055190`，但仍弱于 hard top-1 `0.982425`。
- 使用 `fixed_candidates` 周期桶重建 120 sample_key ViT embedding 后，去掉首批 warm-up 的图像化 latency 从旧 run 的 `0.469106 ms/window` 降到 `0.222156 ms/window`，端到端每窗口均值从 `1.692935 ms` 降到 `1.405437 ms`。
- 新 `fixed_candidates` embedding 重跑同参数 `fusion_huber_kl` router 后，hard top-1 MAE 为 `1.011773`，soft fusion MAE 为 `1.088799`；最佳 calibration 为 `soft_T0p25`，MAE 为 `1.021081`，超过 `global_best_single` 但弱于旧代表 hard top-1。
- 新旧 embedding 数组对比显示 120 个样本中有 22 个 embedding 最大绝对差异超过 `1e-6`，test hard top-1 有 13 个 sample_key 改变专家选择；说明固定候选周期口径提升了速度，但在当前小样本上会改变视觉表征和下游选择，不能直接视作指标改进。
- `96_48_S` 1k manifest-only 样本清单已完成，输出目录为 `experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/`。
- `build_prediction_cache_from_manifest.py`、`merge_prediction_cache_shards.py` 已新增为根目录可复用正式入口；`pilot/launch_96_48_s_1k_prediction_cache_pilot.py` 已收拢为 `96_48_S` 1k 专用 launcher；8 sample_key DLinear CPU/GPU smoke 和单专家 merge smoke 已通过。
- `pilot/build_vit_embeddings_pilot.py` 已支持 `--sample-manifest-path`；8 sample_key GPU embedding smoke 已通过，embedding `.npy` 写入 `/data2/syh/Time/cache_shards/`，未保存伪图像 tensor。该验证只证明离线 cache smoke 可复现，不改变当前 online 路线。
- `pilot/launch_96_48_s_1k_vit_embedding_pilot.py` 已新增，用于生成不自动启动的 1k embedding cache smoke launcher；当前 online 路线下不作为正式入口。
- `train_visual_router_online.py` 已新增为当前推荐入口，在线执行 `x -> pseudo image -> frozen ViT -> CLS embedding -> router`，运行内暂存 vali/test embedding，不保存伪图像 tensor 或 ViT embedding npy。
- 120 sample_key online Visual Router smoke 已完成，输出目录为 `experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/`；hard top-1 MAE=`0.982425`、raw soft fusion MAE=`1.085451`、oracle MAE=`0.805392`，与离线 embedding 代表结果完全对齐，且未生成 `.npy` 或 `embeddings/` 目录。

## 未完成

主实验仍未完成以下部分：

1. 正式 per-config cache builder 已有第一版 shard builder/merge/launcher，但尚未完成 1k 五专家实跑；
2. processor/direct-forward 双路径的统一测试仍不完整，目前已跑通的是 direct `pixel_values` + `hf_vit_0_5`；
3. MAE/CLIP embedding smoke 尚未完成，当前只完成 HF ViT smoke；
4. soft fusion calibration 已完成 120 sample_key smoke，但最佳 soft 策略仍弱于 hard top-1；需要在更大 `96_48_S` 样本上验证 sharpen/top-k 的稳定性；
5. Stage 1 正式脚本仍缺统一 evaluator 和 summarizer；
6. 扩展到三个 config 的正式 summary/report；
7. Stage 1B 迁移学习和 leave-one-config-out 评估。

## 下一步任务与状态

### 0. `96_48_S` 1k 中等规模方案

状态：manifest-only 已完成；prediction cache launcher 已收拢为 `pilot/launch_96_48_s_1k_prediction_cache_pilot.py`，ViT embedding launcher 已收拢为 `pilot/launch_96_48_s_1k_vit_embedding_pilot.py` 且未启动。当前路线改为先完成 1k 五专家 prediction cache，再用 online embedding router 训练；不再先跑 1k ViT embedding cache。

样本方案：

- 总数：`1000` unique sample_key；
- split：`vali=500`、`test=500`；
- dataset：每个 split 下 `TEST_DATA_MIN=250`、`TEST_DATA_HOUR=250`；
- item：每个 split/dataset 选择 50 个 item；
- window：每个 item 的 ch0 选择 5 个中心等距 window；
- item 抽样：TSF cell 均衡轮转，再在 cell 内按 item_id 等距取样；
- channel：本轮保持 `channel_id=0`，延续 120 sample smoke 的单通道口径。

当前样本清单：

```text
experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/sample_manifest.csv
```

dry-run / manifest 验证结果：

- shape 为 `1000 x 17`；
- `sample_key` 唯一数为 `1000`；
- 四个 split/dataset block 各 `250` 条；
- 候选窗口规模为：
  - vali TEST_DATA_MIN: `1,820,415`
  - vali TEST_DATA_HOUR: `7,530,105`
  - test TEST_DATA_MIN: `12,619,225`
  - test TEST_DATA_HOUR: `1,305,425`

成本估算：

- prediction manifest：`1000 * 5 = 5000` 行；
- prediction cache 若沿用旧 pilot 重复 y_true 小文件口径，按 120 sample_key `du=5.9MiB` 线性外推约 `49.17MiB`；
- 新 shard builder 已实现同一 sample_key 共享 y_true，小数组逻辑体积约 `1.83MiB`，实际目录占用仍会受小文件块大小影响；
- 1k ViT embedding float32 长期缓存约 `2.93MiB`，fp16 约 `1.46MiB`；当前不把该估算作为执行理由，路线改为 `train_visual_router_online.py` 运行内暂存 embedding；
- 5k 方案只作为估算：prediction manifest 约 `25,000` 行，旧小文件口径目录占用约 `245.8MiB`，float32 ViT embedding 约 `14.65MiB`。当前不直接跑 5k。

GPU 策略：

- 2026-06-14 10:14:01 CST 检查时，4 张 RTX 3090 基本空闲，仅 Xorg 占用少量显存；
- prediction cache 深度专家可并发：DLinear/PatchTST/CrossFormer 分别绑定 GPU 0/1/2，脚本内部 `--local-rank 0`；
- ES/SNaive 是统计模型，默认 CPU，不占 GPU；
- online Visual Router smoke 可绑定 GPU 3；当前 120 sample_key 已在 GPU 3 单卡完成，不使用 DDP；
- 不强行 DDP；并行方式为独立进程 + `CUDA_VISIBLE_DEVICES` 单卡隔离；
- 每个后台任务写独立 `main.log`、`status.json` 和 shard 目录；合并前必须校验 `sample_key + model_name` 唯一、五专家完整性和 y_true 一致性。

已生成但未启动的后台 launcher：

```text
prediction cache:
experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh

ViT embedding:
experiment_logs/run_outputs/2026-06-14_101500_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/launcher.sh
```

执行约束：

- 暂不启动 1k ViT embedding launcher；
- 不长期缓存伪图像 tensor 或 ViT embedding npy；
- prediction cache launcher 仍可用于后续 1k，因为 router 训练、oracle、soft fusion 和 calibration 需要五专家 `y_pred/y_true`；
- 1k router 训练阶段改用 `train_visual_router_online.py`，从合并后的 labels 和 prediction manifest 在线读取历史窗口 x 并运行冻结 ViT。

查看进度命令模板：

```text
tail -f experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/DLinear/main.log
cat experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/shards/DLinear/status.json
nvidia-smi
```

中等规模汇总必须包含：

- `oracle_top1`
- `global_best_single`
- `timefuse_single_variable_logistic_regression`
- `visual_hard_top1`
- `raw_soft`
- `best_calibrated_soft`

其中 `best_calibrated_soft` 只能来自固定 temperature/top-k sweep 的 config-level 汇总最优；不能读取 test oracle error 做 per-sample 动态调权。

当前判断：

- 120 sample_key online smoke 已证明 online 入口可复现离线代表 router，且不产生长期 embedding/伪图像缓存；
- 可以在用户确认后启动 1k prediction cache launcher；
- 不应启动 1k ViT embedding launcher。

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

### 2. 实现 `96_48_S` HF ViT Embedding / Online Embedding Smoke

状态：离线 embedding smoke 已完成，输出目录为 `experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/`，对应代码已迁入 `pilot/build_vit_embeddings_pilot.py` 作为历史对照；online embedding smoke 已通过 `train_visual_router_online.py` 集成完成，输出目录为 `experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/`，这是当前推荐路线。

目标：基于当前 120 个 sample_key 的在线伪图像化结果，跑通冻结视觉 encoder embedding。

第一版范围：

- config：`96_48_S`；
- 样本：当前扩大版 pilot 的 120 个 `metric=mae` sample_key；
- encoder：优先 `google/vit-base-patch16-224`；
- variant：先做 `variant_a=3view`，再可选 `variant_b=top3fold`；
- dtype：优先 fp16；
- 输出根目录：默认 `experiment_logs/run_outputs/`，可通过 `--output-root` 写到 `/data2/syh/Time/run_outputs/`；
- 当前推荐：router 训练时用 online/in-memory embedding，不写 shard，不保存 ViT embedding `.npy`；
- cache 根目录：仅保留给历史离线 embedding smoke 或临时 ablation，正式 1k 路线暂不使用。

验收标准：

- 离线 pilot 路径输出 embedding manifest，字段包含 `sample_key`、`config_name`、`split`、`variant`、`encoder_name`、`embedding_path`、`embedding_dim`、`dtype`；该路径只用于历史对照或小规模 ablation；
- online 路径输出 `online_embedding_manifest.csv`，不含 `embedding_path`，因为 embedding 只在运行内内存字典暂存；
- embedding shape 与 encoder hidden size 一致，`google/vit-base-patch16-224` 预期为 768；
- 所有 embedding finite；
- 与 `window_oracle_labels_with_tsf_cell.csv` 的 `sample_key` 集合完全一致；
- 输出 latency summary，区分 imageization、encoder forward、write/read。

### 3. 训练最小 Per-Config Visual Router

状态：已完成小型 MLP 版本；旧版分类 baseline 输出目录为 `experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/`，新版 fusion router 离线代表输出目录为 `experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/`，online 代表输出目录为 `experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/`。

先做 `96_48_S`：

- 输入：`vali` visual embedding；
- online 入口输入：`sample_key` 对齐的 Quito 历史窗口 x，运行内生成 vali/test embedding 后传给同一套 router 训练评估函数；
- 默认训练目标：按 sample_key 读取五专家 `y_pred` 与共享 `y_true`，router softmax 权重直接融合预测，主损失使用 `SmoothL1Loss(fused_pred, y_true)`；
- 辅助监督：用五专家误差构造 `q_i = softmax(-error_i / tau)`，最小化 `KL(q || p_router)`；
- baseline 模式：`--router-mode classification` 保留同 config 内 `oracle_model` hard-label 交叉熵训练；
- 输出：五专家概率或 top-1；
- 评估：`test` hard top-1 MAE、oracle label accuracy、regret。

当前模型建议：

- `StandardScaler + 小型 MLP`；
- scaler 和 router 只能在 `vali` split fit；
- test 只做 transform/predict/evaluate；
- 若 `vali` label 类别过少，必须在 summary 中显式记录，不能静默退化。

验收标准：

- 输出 `visual_router_predictions.csv`；
- 输出 `visual_router_summary.csv`；
- summary 至少包含 `selected_value`、`oracle_value`、`regret_to_oracle`、`oracle_label_accuracy`；
- 结果与非视觉 baseline、TimeFuse 结构特征 router、oracle top-1 同表比较；
- summary 记录权重熵、归一化权重熵、平均最大权重和 hard top-1 专家选择分布。
- online 入口还需输出 `online_embedding_latency_summary.csv`、`visual_router_online_metadata.json` 和 `online_vs_offline_reference_comparison.csv`，并确认没有 `.npy`、`embeddings/` 或伪图像 tensor cache。

### 4. 增加 Softmax Fusion 评估

状态：已完成基于 router softmax 权重的五专家预测数组加权融合 smoke，并新增 `evaluate_soft_fusion_calibration.py` 完成 temperature/top-k calibration smoke。旧代表 router 的最佳 soft calibration `top2_fusion_T0p25` MAE 为 `0.999014`，超过 `global_best_single=1.055190`，但仍弱于 hard top-1 `0.982425`；新 fixed_candidates embedding 的最佳 soft calibration `soft_T0p25` MAE 为 `1.021081`，同样超过 global single 但弱于 hard top-1。

目标：不只评估 hard top-1，还评估 router 概率对五专家预测的加权融合是否能降低 MAE/MSE。

约束：

- soft fusion 只能融合同一 `config_name`、同一 `sample_key` 下五专家的 `y_pred`；
- 融合权重来自 vali 训练出的 router 在 test 上的输出概率；
- 不允许使用 test oracle error 调整权重。

验收标准：

- 输出 `soft_fusion_predictions.csv` 或等价文件；
- 对每个 test sample 记录五专家权重、融合后 MAE/MSE、hard top-1 MAE/MSE、oracle MAE/MSE；
- summary 同时给出 hard top-1 和 soft fusion。
- calibration 额外输出 `soft_fusion_calibration_summary.csv`、`soft_fusion_calibration_comparison.csv`、`soft_fusion_calibration_selected_model_counts.csv`，并记录 entropy、max-weight、active weight count、selected-model 分布。

后续诊断重点：

- 当前最有效的方向是降低温度或 top-k 截断，说明 router 排序信号可用，但概率幅度过平滑；
- 若要让 soft fusion 超过 hard top-1，需要在训练阶段加入更强的稀疏/置信约束，或学习一个只在高置信样本启用 fusion 的 gating 规则；
- fixed_candidates 周期桶提升 latency，但会改变 embedding，需要扩大样本后再决定是否作为正式 embedding 口径。

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

当前数据流、oracle 和结构特征验证脚本仍保留在 `pilot/`。`96_48_S` 最小闭环跑通后，已把长期复用逻辑迁出或重写为正式入口；后续若继续正式化，应优先补齐：

- 统一 evaluator / reporter，用于把 hard top-1、raw soft、best calibrated soft、baseline 和 oracle 同表汇总；
- 面向三个 config 的 sample manifest / prediction cache 编排层，避免长期依赖 `96_48_S` 专用 launcher；
- 可选的 encoder 输入路径 smoke，用于覆盖 HF processor 与 direct `pixel_values` 两条路径。

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
