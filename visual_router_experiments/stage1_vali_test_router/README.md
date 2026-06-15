# stage1_vali_test_router

本目录用于保存 Stage 1 主实验代码。

Stage 1 目标：

- 冻结五个专家；
- 在 vali/test 上生成 item-channel-window 级 prediction cache；
- 在 vali 上训练 visual router；
- 在 test 上评估 hard top-1 routing 和 softmax fusion；
- 验证视觉结构先验在同分布设置下是否能提升专家选择或加权融合。

脚本按职责拆分：

- `build_stage1_sample_manifest.py`：生成 `96_48_S` 中等规模 manifest-only 样本清单，默认 1k sample_key，vali/test、dataset、item 和 window 尽量均衡；
- `build_prediction_cache_from_manifest.py`：基于 sample manifest 生成 prediction cache shard，支持按专家或 sample shard 并发，shard 内共享 `y_true_path`；
- `merge_prediction_cache_shards.py`：合并 prediction cache shard，校验 `sample_key + model_name` 唯一、五专家完整和共享 y_true 一致；
- `launch_full_scale_prediction_cache.py`：生成 full-scale prediction cache launcher，按专家和 sample shard 拆分任务，默认深度专家绑 GPU、统计专家走 CPU，数组默认 `packed_npy_v1`；
- `run_full_scale_dry_run.py`：执行小样本 full-scale 框架 dry-run，验证 manifest -> packed cache -> merge -> oracle/baseline -> streaming router -> calibration 闭环；
- `evaluate_router_baselines.py`：基于 vali split 学统计规则 baseline，并可同时训练 TimeFuse-style 单层 fusor baseline，在 test split 上生成统一 comparison；
- `fusion_utils.py`：共享 prediction manifest 读取、五专家预测数组读取、hard top-1/soft fusion 指标复算，以及 TimeFuse-style 单层 fusor 训练与评估逻辑；
- `train_visual_router.py`：训练 TimeFuse-style 小型 MLP visual router；默认 `fusion_huber_kl` 模式用五专家预测加权融合的 SmoothL1 主损失训练权重，同时保留 `classification` 旧版 oracle hard-label baseline；
- `train_visual_router_online.py`：在线执行 `x -> pseudo image -> frozen ViT -> CLS embedding -> router`，适合 120/1k 规模，在一次运行内把 embedding 暂存在内存中训练 MLP router，不保存伪图像 tensor 或 ViT embedding npy；
- `train_visual_router_online_streaming.py`：full-scale streaming online router，batch 运行时生成 ViT embedding，`StandardScaler.partial_fit` 只遍历 vali，test 流式 forward；不保存 embedding `.npy`，不构建全量 embedding 字典；
- `evaluate_soft_fusion_calibration.py`：读取已训练 router 的 test 权重和 prediction cache，评估 temperature scaling、top-k 截断重归一化、raw soft 与 hard top-1 的统一 soft fusion calibration 表。

当前推荐路线是 online embedding，不再先缓存 ViT embedding，不启动 ViT embedding launcher，不长期保存伪图像 tensor 或 ViT embedding `.npy`。小规模/1k 复现可用 `train_visual_router_online.py`，full-scale 使用 `train_visual_router_online_streaming.py`。`pilot/` 子目录只保留小规模验证、离线 embedding cache 历史对照、过渡性 launcher 和特定资源编排脚本；正式可复用的评估和训练入口保留在 Stage 1 根目录，跨阶段通用逻辑上收到 `visual_router_experiments/common/`。

当前已有文件：

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 将 Stage 1 目录标记为可导入 Python package |
| `prediction_cache_design.md` | 记录 Quito evaluate/data/model 数据流阅读结论，以及 Stage 1 prediction cache 的推荐导出点和 pilot 限制 |
| `feature_and_rl_extension_notes.md` | 记录 TimeFuse-style 结构特征 router 支线、feature scaler 口径，以及视觉路由扩展为 contextual bandit / RL 的可行方案 |
| `stage1_cache_contract.md` | 固定 Stage 1 正式 prediction cache、oracle labels、feature cache 和 router evaluation 的字段契约 |
| `stage1_protocol_and_plan.md` | 记录 Stage 1 per-config 主实验协议、Stage 1B 迁移实验设计、已完成事项、当前未完成清单、视觉 encoder 输入口径和下一步任务验收标准 |
| `build_stage1_sample_manifest.py` | 基于 Quito evaluate 数据边界生成 manifest-only 样本清单；当前默认 `96_48_S` 1k sample_key，按 split/dataset 均衡、TSF cell 轮转 item、每 item 取中心等距窗口，不启动专家推理 |
| `build_prediction_cache_from_manifest.py` | 读取 sample manifest 和专家 evaluate config，生成单专家或多专家 prediction cache shard；支持 `per_sample_npy` 和 full-scale 推荐的 `packed_npy_v1`，只对清单指定窗口做前向，支持 GPU 单卡绑定 |
| `merge_prediction_cache_shards.py` | 合并多个 prediction cache shard，复制并重写数组路径，校验五专家完整性、`sample_key + model_name` 唯一性和 y_true 一致性；packed 模式会重建 merged 共享 y_true row index |
| `launch_full_scale_prediction_cache.py` | 生成正式 full-scale prediction cache launcher；按专家和 sample shard 拆任务，DLinear/PatchTST/CrossFormer 绑定 GPU，ES/NaiveForecaster 走 CPU |
| `run_full_scale_dry_run.py` | 小样本执行 full-scale 框架 dry-run；每步写 `main.log`、`status.json`，用于验证 packed cache、merge、oracle/baseline、streaming router 和 calibration ABI |
| `evaluate_router_baselines.py` | 用 vali split 学 global/dataset/TSF-cell/dataset+TSF-cell 等统计规则 baseline，并可训练原生 TimeFuse-style 单层 fusor；输出统计 baseline、fusor hard top-1、raw soft fusion、oracle 和统一 `baseline_comparison.csv` |
| `fusion_utils.py` | Stage 1 router/fusor 共享工具；统一读取 prediction manifest、加载五专家 `y_pred/y_true` 数组、复算 hard/soft fusion MAE/MSE，并实现 `nn.Linear -> softmax` 的 TimeFuse-style fusor baseline |
| `train_visual_router.py` | 读取 ViT embedding manifest、oracle labels 和 prediction manifest，按 `config_name` 独立训练小型 MLP router；支持 `classification` 与 `fusion_huber_kl`，输出五专家权重、hard top-1、soft fusion、专家选择分布、权重熵/最大权重占比和 baseline comparison |
| `train_visual_router_online.py` | 从 `sample_key` 对齐的 Quito 历史窗口在线构造伪图像、前向冻结 HF ViT、运行内暂存 CLS embedding，并复用 `train_visual_router.py` 的 MLP router 训练、hard top-1 和 soft fusion 评估逻辑；适合小规模/1k，不落盘 embedding npy 或伪图像 tensor |
| `train_visual_router_online_streaming.py` | 从 Quito 历史窗口流式生成 ViT embedding；scaler 只在 vali 上 `partial_fit`，训练 epoch 重新流式生成 vali embedding，test 流式预测；输出兼容 calibration 的 `visual_router_predictions.csv` 和标准 `visual_router_metadata.json` |
| `evaluate_soft_fusion_calibration.py` | 读取 `visual_router_predictions.csv` 与 prediction manifest，在不改变 router 输入约束的前提下评估 softmax temperature、top-k weight truncation、raw soft、top1 hard、top2/top3 fusion，并输出 entropy、max-weight、selected-model 分布诊断 |
| `pilot/` | 保存 Stage 1 正式实验前的数据流、cache schema、oracle label、TSF cell enrichment pilot 脚本、离线 embedding cache 历史对照脚本，以及 `96_48_S` 1k 专用 launcher |

## TimeFuse-style Fusor Baseline

`evaluate_router_baselines.py` 现在是统一 baseline 入口：默认保留 global/dataset/TSF-cell/dataset+TSF-cell 等统计规则 baseline，并在 feature cache 与 prediction manifest 可对齐时训练 TimeFuse-style 单层 fusor。该 fusor 复刻 `TimeFuse/timefuse.py` 与 notebook 训练口径：`nn.Linear(input_dim, output_dim)` 输出 logits，forward 后 softmax 得到五专家权重，训练时用权重融合五专家 `y_pred`，再用 `SmoothL1Loss(beta=0.01)` 对 `y_true` 反传。旧 `timefuse_single_variable_logistic_regression` 是 legacy/deprecated hard-label 分类 baseline，不再作为新的主比较口径。

120 sample_key pilot 复验命令：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
  --labels-path experiment_logs/run_outputs/2026-06-12_125902_319469_visual_router_stage1_prediction_cache_pilot/window_oracle_labels_with_tsf_cell.csv \
  --output-dir experiment_logs/run_outputs/2026-06-15_014918_visual_router_stage1_timefuse_fusor_baseline_pilot \
  --timefuse-fusor on \
  --device cpu \
  --fusor-epochs 5 \
  --fusor-batch-size 64 \
  --fusor-lr 0.0005 \
  --fusor-beta 0.01 \
  --seed 16
```

代表输出目录：

- `experiment_logs/run_outputs/2026-06-15_014918_visual_router_stage1_timefuse_fusor_baseline_pilot/`

当前 120 sample_key pilot 结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `timefuse_style_fusor` hard top-1 | 1.490870 | 0.805392 | 0.685478 | 0.216667 | 0.576704 | 0.619892 |
| `timefuse_style_fusor_raw_soft_fusion` | 1.509144 | 0.805392 | 0.703752 | NA | 0.576704 | 0.619892 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |

输出文件包括 `timefuse_fusor_predictions.csv`、`timefuse_fusor_raw_soft_fusion_predictions.csv`、`timefuse_fusor_summary.csv`、`timefuse_fusor_raw_soft_fusion_summary.csv`、`timefuse_fusor_selected_model_counts.csv`、`baseline_comparison.csv` 和 `baseline_metadata.json`。其中每个 test sample 都保留 `weight_{model_name}`，comparison 表可与 visual router 后续结果按 `config_name`、MAE、oracle regret 和权重诊断同表比较。

## 历史离线 120 Sample Smoke

早期离线 smoke 使用当前扩大版 `96_48_S` 五专家 pilot 的 120 个 `metric=mae` sample_key；该路径会生成 embedding `.npy`，目前只作为历史对照和小规模调试保留：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/pilot/build_vit_embeddings_pilot.py \
  --local-files-only --batch-size 16

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
  --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv
```

输出目录：

- `experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/`
- `experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/`

旧版分类 router 结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy |
| --- | --- | --- | --- | --- |
| `visual_router_mlp_v1_classification` hard top-1 | 1.013099 | 0.805392 | 0.207707 | 0.350000 |
| `visual_router_mlp_v1_classification_soft_fusion` | 1.022590 | 0.805392 | 0.217198 | NA |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 |

## 当前 Fusion Router Smoke

本次改造后，`train_visual_router.py` 默认使用 `fusion_huber_kl`：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
  --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv \
  --router-mode fusion_huber_kl \
  --epochs 300 \
  --batch-size 32 \
  --hidden-dim 64 \
  --dropout 0.0 \
  --huber-beta 0.1 \
  --kl-tau 0.1 \
  --lambda-kl 0.01
```

代表输出目录：

- `experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/`

当前结果：

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | --- | --- | --- | --- | --- | --- |
| `visual_router_mlp_v2_fusion_huber_kl` hard top-1 | 0.982425 | 0.805392 | 0.177033 | 0.466667 | 0.757180 | 0.483784 |
| `visual_router_mlp_v2_fusion_huber_kl_soft_fusion` | 1.085451 | 0.805392 | 0.280059 | NA | 0.757180 | 0.483784 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `timefuse_single_variable_logistic_regression` legacy/deprecated | 1.079743 | 0.805392 | 0.274351 | 0.466667 | NA | NA |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |

结论口径：

- `fusion_huber_kl` hard top-1 在当前 60 个 test window 上超过旧分类 hard top-1、`global_best_single` 和 TimeFuse 结构特征 router；
- soft fusion 仍弱于 hard top-1 和 `global_best_single`，说明当前权重校准还不适合作为最终 fusion 结论；
- 该结果仍只是 `96_48_S`、120 sample_key 的 smoke，不作为三 config 正式结论。

## Online Visual Router Smoke

当前推荐路线改为 online embedding：不再先启动 ViT embedding cache，也不长期保存伪图像 tensor 或 ViT embedding `.npy`。`train_visual_router_online.py` 是 120/1k 规模的小规模复现入口，会在一次运行内读取 Quito 历史窗口 `x`，在线构造伪图像、前向冻结 HF ViT，并把 vali/test CLS embedding 暂存在内存字典中。full-scale 路线使用 `train_visual_router_online_streaming.py`，按 batch 生成 embedding 并立即消费，不构建全量 embedding 字典。

代表 120 sample_key smoke：

```text
CUDA_VISIBLE_DEVICES=3 \
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py \
  --device cuda \
  --local-files-only \
  --embedding-batch-size 16 \
  --router-mode fusion_huber_kl \
  --epochs 300 \
  --batch-size 32 \
  --hidden-dim 64 \
  --dropout 0.0 \
  --huber-beta 0.1 \
  --kl-tau 0.1 \
  --lambda-kl 0.01
```

代表输出目录：

- `experiment_logs/run_outputs/2026-06-14_142004_461629_visual_router_stage1_online_visual_router_smoke/`

验证结果：

| 检查项 | 结果 |
| --- | --- |
| `online_embedding_manifest.csv` | `120 x 19`，覆盖 labels 的 120 个 `metric=mae` sample_key |
| `visual_router_predictions.csv` | `60 x 22`，覆盖 test split 的 60 个 sample_key |
| `visual_router_soft_fusion_predictions.csv` | `60 x 36` |
| 设备与 dtype | GPU 3，metadata 中 device=`cuda`，forward_dtype=`float16` |
| online embedding latency | imageization `2.473705 ms/window`，encoder forward `2.591103 ms/window`，in-memory store `0.023265 ms/window` |
| 落盘缓存 | 未生成 `.npy`、未生成 `embeddings/` 目录、未生成伪图像 tensor cache |

online 与离线代表 router 指标完全对齐：

| 方法 | hard top-1 MAE | raw soft fusion MAE | oracle MAE | global_best_single |
| --- | ---: | ---: | ---: | ---: |
| online in-memory ViT | 0.982425 | 1.085451 | 0.805392 | 1.055190 |
| offline embedding reference | 0.982425 | 1.085451 | 0.805392 | 1.055190 |

实现细节：

- online 入口复用 `visual_router_experiments/common/vit_embedding_utils.py` 的 `make_pseudo_images()` 和 `pool_vit_outputs()`；
- `train_visual_router.py` 的 `load_embedding_matrix()` 支持运行内 `sample_key -> embedding` lookup，因此 `VisualMLPRouter`、`fusion_huber_kl`、`classification`、hard top-1、soft fusion 和 baseline comparison 都复用原实现；
- ViT 构造会消耗 PyTorch RNG，online 入口在 embedding 完成后重新 `set_seed(seed)`，保证 MLP 初始化和 DataLoader shuffle 与离线训练入口一致。

## Soft Fusion Calibration Smoke

基于代表性 fusion router 输出，已新增并运行校准评估：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py \
  --router-predictions-path experiment_logs/run_outputs/2026-06-14_025727_562553_visual_router_stage1_visual_router_smoke/visual_router_predictions.csv \
  --temperatures 0.25,0.5,0.75,1.0,1.5,2.0 \
  --top-k-values all,1,2,3
```

输出目录：

- `experiment_logs/run_outputs/2026-06-14_032303_451482_visual_router_stage1_soft_fusion_calibration_smoke/`

旧代表 router 的校准结果：

| 策略 | test MAE | oracle MAE | normalized weight entropy | mean max weight | 相对 global_best_single |
| --- | --- | --- | --- | --- | --- |
| `top1_hard` | 0.982425 | 0.805392 | 0.000000 | 1.000000 | +6.895970% |
| `top2_fusion_T0p25` | 0.999014 | 0.805392 | 0.232572 | 0.822011 | +5.323768% |
| `soft_T0p25` | 1.000585 | 0.805392 | 0.295159 | 0.799357 | +5.174969% |
| `raw_soft` | 1.085451 | 0.805392 | 0.757180 | 0.483784 | -2.867833% |
| `global_best_single` | 1.055190 | 0.805392 | NA | NA | 0.000000% |

结论：

- 温度 sharpen 和 top-k 截断可以把 soft fusion 从 `1.085451` 拉回到 `0.999014`，已经超过 `global_best_single=1.055190`；
- 最佳校准策略仍弱于 hard top-1 `0.982425`，说明当前 router 概率的排序比概率幅度更可靠；
- 随着温度变大或权重更平滑，MAE 单调变差，主要问题仍是低置信权重混入较差专家。

## Fixed Candidates Embedding 对照

使用当前默认 `fixed_candidates` 周期桶重新生成同一批 120 个 sample_key：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/pilot/build_vit_embeddings_pilot.py \
  --local-files-only \
  --batch-size 16 \
  --period-selection fixed_candidates
```

输出目录：

- embedding：`experiment_logs/run_outputs/2026-06-14_032340_154510_visual_router_stage1_vit_embedding_smoke/`
- router：`experiment_logs/run_outputs/2026-06-14_032518_167365_visual_router_stage1_visual_router_smoke/`
- calibration：`experiment_logs/run_outputs/2026-06-14_032647_499280_visual_router_stage1_soft_fusion_calibration_smoke/`

latency 对比已写入新 embedding run：

- `embedding_latency_comparison_vs_old.csv`
- `embedding_latency_speed_ratio_vs_old.csv`

去掉首批 warm-up 后，fixed_candidates 图像化均值为 `0.222156 ms/window`，旧 embedding run 为 `0.469106 ms/window`，图像化阶段约为旧口径的 `47.36%`；encoder forward 基本不变，端到端每窗口均值从 `1.692935 ms` 降到 `1.405437 ms`。

新 embedding 对 router 指标的影响：

| 方法 | test MAE | oracle MAE | regret | label accuracy | normalized weight entropy | mean max weight |
| --- | --- | --- | --- | --- | --- | --- |
| `visual_router_mlp_v2_fusion_huber_kl` hard top-1 | 1.011773 | 0.805392 | 0.206381 | 0.433333 | 0.783611 | 0.453640 |
| `visual_router_mlp_v2_fusion_huber_kl_soft_fusion` | 1.088799 | 0.805392 | 0.283407 | NA | 0.783611 | 0.453640 |
| `calibration_soft_T0p25` | 1.021081 | 0.805392 | 0.215689 | 0.433333 | 0.365525 | 0.756458 |
| `calibration_top2_fusion_T0p25` | 1.023443 | 0.805392 | 0.218051 | 0.433333 | 0.265783 | 0.794654 |
| `global_best_single` | 1.055190 | 0.805392 | 0.249798 | 0.050000 | NA | NA |
| `timefuse_single_variable_logistic_regression` legacy/deprecated | 1.079743 | 0.805392 | 0.274351 | 0.466667 | NA | NA |
| `oracle_top1` | 0.805392 | 0.805392 | 0.000000 | 1.000000 | NA | NA |

新旧 embedding 数组对比显示 120 个样本中有 22 个 embedding 的最大绝对差异超过 `1e-6`，hard top-1 有 13 个 test window 改变专家选择。新口径加速了伪图像化，但当前 120 sample_key 指标弱于旧代表 embedding，因此不能把 fixed_candidates 直接视为性能改进；更合理的下一步是在更大 `96_48_S` 样本上同时比较速度和路由稳定性。

## 96_48_S 1k 中等规模准备

当前已生成 manifest-only 1k sample_key 清单：

- `experiment_logs/run_outputs/2026-06-14_095911_486696_visual_router_stage1_sample_manifest_96_48_s_1k/`

抽样口径：

- `vali=500`、`test=500`；
- 每个 split 下 `TEST_DATA_MIN=250`、`TEST_DATA_HOUR=250`；
- 每个 dataset 选 50 个 item，每个 item 选 ch0 的 5 个中心等距 window；
- item 使用 TSF cell 均衡轮转后在 cell 内等距抽样，避免旧 pilot 的前缀 item/window 偏置。

验证结果：

- `sample_manifest.csv` 为 `1000 x 17`；
- `sample_key` 唯一数为 `1000`，无重复；
- 候选窗口规模：vali/test 四个 dataset split 合计约 `23,275,170` 个候选 item-channel-window；
- prediction cache 估算为 `5000` 行 manifest；沿用旧 pilot 小文件口径约 `49.17 MiB`，共享 y_true 后逻辑小数组体积约 `1.83 MiB`；
- 1k ViT embedding 长期缓存 float32 约 `2.93 MiB`，fp16 约 `1.46 MiB`；当前路线已改为 online/运行内缓存，不先跑 1k ViT embedding cache，不长期保存伪图像 tensor 或 ViT embedding npy。

已完成最小验证：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/build_prediction_cache_from_manifest.py \
  --sample-manifest-path experiment_logs/run_outputs/2026-06-14_100000_manual_prediction_cache_builder_smoke/sample_manifest_8.csv \
  --models DLinear \
  --batch-size 512 \
  --local-rank -1
```

- CPU DLinear 8 sample_key cache smoke 通过；
- GPU DLinear 8 sample_key smoke 首次暴露模型权重未迁移到 CUDA 的问题，已在 `prepare_model()` 显式 `model.to(model.device)` 后复验通过；
- `merge_prediction_cache_shards.py --expected-models DLinear` 单专家 merge smoke 通过；
- `pilot/build_vit_embeddings_pilot.py --sample-manifest-path ... --cache-root /data2/syh/Time/cache_shards/...` 8 sample_key GPU embedding smoke 通过，embedding dim 为 768；该结果只作为历史离线 cache smoke 保留，当前 1k 路线不启动 ViT embedding cache。

已生成但未自动启动的 launcher：

- prediction cache：`experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/launcher.sh`
- ViT embedding：`experiment_logs/run_outputs/2026-06-14_101500_visual_router_stage1_vit_embedding_96_48_s_1k_launcher/launcher.sh`，当前路线暂不启动，仅作为历史生成物保留

1k 链路状态：

- 1k 五专家 prediction cache、merge、oracle/TSF/baseline、online router 和 calibration 均已完成；
- 1k online router 保留为中等规模实证结果，后续真正 full-scale 不再使用全量 in-memory embedding 字典；
- 历史 1k ViT embedding launcher 不启动，仅作为已生成但废弃的离线 cache 对照入口留痕。

中等规模 comparison 表必须至少包含：`oracle_top1`、`global_best_single`、`timefuse_style_fusor` hard top-1、`timefuse_style_fusor_raw_soft_fusion`、`visual_hard_top1`、`raw_soft`、`best_calibrated_soft`。旧 `timefuse_single_variable_logistic_regression` 只能作为 legacy/deprecated 历史附录引用，不再作为主表必选 baseline。其中 `best_calibrated_soft` 只能来自固定 temperature/top-k sweep 的 config-level 汇总最优，不能按 test sample 读取 oracle error 动态调权。

## Full-Scale Dry-Run

首个 full-scale 框架 dry-run 已完成：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/run_full_scale_dry_run.py \
  --samples-per-split 2 \
  --sample-shard-count 2 \
  --embedding-batch-size 2 \
  --router-epochs 1 \
  --device auto \
  --local-files-only \
  --output-dir experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2
```

验证结果：

- merged manifest 为 `20` 行，覆盖 `4` 个 sample_key，每个 sample_key 五专家完整；
- prediction cache 全部使用 `array_storage=packed_npy_v1`；
- streaming router 输出 `2` 条 test prediction，权重行和约为 `1.0`；
- calibration 输出 `raw_soft`、`soft_T0p5`、`top1_hard`、`top2_fusion`、`top2_fusion_T0p5` 共 `5` 个策略；
- `streaming_online_router/` 下未生成 `.npy`、`embeddings/` 或 embedding shard 文件，只保存 manifest、latency、router predictions、summary 和 metadata；
- dry-run 根目录现在也写入 `main.log`、`status.json` 和 `metadata.json`，可作为后续 full-scale 长任务模板。
