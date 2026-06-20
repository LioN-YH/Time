# Stage 1 Visual Router V2 小规模架构优化实验协议

创建日期：2026-06-20

## 1. 文档目的

本文把现有 `96_48_S` full-scale Visual Router 与 TimeFuse-style baseline 的结果、模型架构诊断和后续小规模实验路线收束为一份可执行协议。目标不是立即重跑 full-scale，也不是在尚未确认瓶颈前直接全量微调 ViT，而是先用固定、可复核的小规模样本回答以下问题：

1. 当前 Visual Router 的主要损失来自尺度信息被 RevIN 移除、伪图像构造、冻结 ImageNet ViT 的 domain gap、pooling，还是 routing objective；
2. 修复后的视觉表示是否获得稳定且独立于 TimeFuse 结构特征的增量信息；
3. Visual 分支带来的误差收益是否足以覆盖其训练和推理成本；
4. 哪些变体值得进入后续 ViT domain adaptation、联合微调和 full-scale 验证。

本文属于 Stage 1 后续研究协议，不修改当前已完成 full-scale 结果的引用口径。当前正式结果继续作为 sanity reference，新实验必须使用新的 run directory 和新的版本标识。

## 2. 当前实验背景与结果

### 2.1 当前 Visual Router

当前正式视觉路径为：

```text
x window
-> per-window RevIN
-> variant_a_3view pseudo image
   channel 0 = line raster
   channel 1 = hard top-1 FFT period fold
   channel 2 = per-window normalized FFT power
-> frozen google/vit-base-patch16-224
-> CLS pooling
-> StandardScaler
-> MLP(768 -> 64 -> 5)
-> softmax expert weights
-> hard top-1 / raw soft fusion
```

训练采用 `fusion_huber_kl`：

```text
loss = SmoothL1(fused_prediction, y_true, beta=0.1)
     + 0.01 * KL(router_weights, soft_oracle(tau=0.1))
```

当前 checkpoint 只完成 1 epoch，覆盖 `9,350,520` 个 vali window；eval-only 覆盖 `13,924,650` 个 test window。

### 2.2 当前 TimeFuse-style baseline

当前 baseline 使用历史窗口提取的 17 维 `timefuse_single_variable_meta_v1` 特征，经 vali-fitted `StandardScaler` 和单层 `Linear -> softmax` 产生五专家权重，训练目标为 `SmoothL1Loss(beta=0.01)`。该 baseline 是针对单变量 QuitoBench 五专家设置改造后的 TimeFuse-style baseline，不代表原论文完整复现。

### 2.3 Full-scale 同口径结果

两边使用相同的 `96_48_S` test sample、相同五专家 prediction cache 和相同 oracle，test sample count 均为 `13,924,650`。

| 指标 | Visual Router 1 epoch | TimeFuse-style | 观察 |
| --- | ---: | ---: | --- |
| hard top-1 MAE | 0.5615367653 | **0.4594660365** | TimeFuse-style 低 18.18% |
| raw soft fusion MAE | 0.5174675760 | **0.4473909308** | TimeFuse-style 低 13.54% |
| hard regret to oracle | 0.2229146241 | **0.1208438953** | TimeFuse-style 缩小 45.79% |
| oracle-label accuracy | 0.4621166780 | **0.5864217772** | TimeFuse-style 高 12.43 个百分点 |
| hard top-1 MSE | **151.2007561569** | 181.4999427123 | Visual Router 尾部指标更好 |
| raw soft fusion MSE | **143.5674979157** | 181.4316851409 | Visual Router 尾部指标更好 |
| normalized weight entropy | 0.7846485994 | **0.4530491615** | Visual 权重明显更分散 |
| mean max weight | 0.4499556117 | **0.7021691159** | Visual 对单个专家更不确定 |

专家选择分布：

| 专家 | Visual Router | TimeFuse-style |
| --- | ---: | ---: |
| DLinear | 31.38% | 11.11% |
| PatchTST | 6.15% | 40.99% |
| CrossFormer | 2.28% | 3.22% |
| ES | 34.15% | 36.99% |
| NaiveForecaster | 26.05% | 7.70% |

该结果说明 Visual Router 并非完全没有信号：它的 MSE 更低，可能对少量极端误差更保守；但在当前主指标 MAE、oracle accuracy、regret 和权重可分性上明显落后。Visual 更倾向 DLinear、ES、NaiveForecaster，较少识别 PatchTST 的优势区域。

正式结果来源：

```text
/data2/syh/Time/run_outputs/2026-06-18_stage1_96_48_s_streaming_visual_router_eval_only_1epoch_ckpt/
/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/
```

## 3. 从当前证据反推的架构问题

以下诊断是由当前代码路径和 full-scale 结果共同支持的候选解释，不应被表述为已经通过消融实验确认的因果结论。

### 3.1 RevIN 移除了尺度信息，但 aux metadata 未进入 router

`normalize_window(..., norm_mode="revin_aux")` 计算 `mean/std/min/max/range`，但当前 `make_pseudo_images(...)` 只保留标准化序列，调用方丢弃 aux metadata。因此 visual feature 主要描述窗口内相对形状，缺少 level、volatility、range 和异常幅度。

TimeFuse-style 输入明确包含 mean、std、min、max 等尺度统计。当前结果不能区分“视觉形状无效”和“视觉路径主动删除了有效尺度信号”。这是优先级最高且最便宜的修复点。

### 3.2 三个异质 view 被伪装为 RGB channel

当前 channel 0/1/2 分别表示 line raster、period fold 和 FFT power；它们不是自然图像中空间对齐的 RGB 分量。ImageNet ViT 的 patch projection 会在同一空间位置混合三个 channel，而这些位置在三种 view 中没有统一物理语义。冻结 encoder 无法通过当前 routing loss 重学输入 stem。

需要比较 spatial panel、三 view 独立共享 encoder 和 view-specific patch stem，确认问题来自 channel mixing 还是 view 本身信息不足。

### 3.3 图像化引入信息压缩与人为纹理

`L=96` 的序列被插值到 `224x224`：line raster 产生大量空白和平滑纹理；fold view 受 padding 与双线性插值影响；FFT power 只是一维 profile 沿另一维复制，并做 per-window min-max，绝对频谱能量丢失。模型可能优先编码渲染器纹理，而不是专家相对能力。

需要通过更高有效像素密度的 value/difference bands、padding mask、保留绝对频谱能量，以及 `112 vs 224` 对比判断这一问题。

### 3.4 固定候选周期与 hard top-1 fold 造成不连续

当前先在固定候选周期上选 hard top-1，再由该周期构造 fold。相似序列在候选周期分数轻微变化时可能从一个 fold 结构跳到另一个 fold 结构，导致 pseudo image、embedding 和 router weights 不连续。

需要以 soft period mixture 或 period tokens 替代单一 hard fold，并用输入轻微扰动前后的 embedding cosine distance 与 router-weight JS divergence量化连续性。

### 3.5 CLS pooling 未必适合结构摘要

ImageNet ViT 的 CLS token服务于自然图像分类，不保证能汇总平稳性、频率结构、局部异常和专家 regret。当前高权重熵与较低 mean max weight 符合“embedding 有图像响应，但专家偏好不容易由小型 head 分离”的表现。

需要首先比较几乎零额外成本的 `mean_patch` 和 `concat(CLS, mean_patch)`；若使用 spatial panel，再比较 view-region pooling 与轻量 attention pooling。

### 3.6 其他需要同时控制的混杂因素

- Visual 使用 `SmoothL1 beta=0.1`，TimeFuse-style 使用 `beta=0.01`，需要在公平消融中对齐；
- 当前 KL 实际平均贡献远小于 fusion Huber，可能不足以形成清晰专家排序；
- 只有 1 epoch 且 streaming 顺序稳定，没有独立 selection split、shuffle/balanced sampling 和 checkpoint selection；
- 当前结果不能排除 dataset/TSF-cell/scale shortcut，也不能证明增加 epoch 可以解决表示问题。

## 4. 总体实验原则

1. 不直接修改或覆盖旧 full-scale 输出；所有 V2 pilot 使用新的 run directory。
2. 先判断信息是否保留，再判断 routing 是否改善，最后才做 ViT 微调。
3. 不做五类变量的完整笛卡尔积，采用逐轮淘汰和固定样本 paired comparison。
4. 每轮至少保留旧 Visual Router、aux-only、TimeFuse-style 和 oracle 四个参照。
5. 同一轮所有变体共享 sample keys、五专家 prediction arrays、seed、训练步数和 evaluator。
6. pilot 允许保存受控规模的 pooled embedding 或 debug image；不得把该策略误扩展为 full-scale 长期 embedding cache。
7. test pilot 只在方案冻结后使用；架构选择必须使用从 vali 内划出的 selection split。

## 5. 固定小规模数据协议

### 5.1 推荐规模

第一版建议：

| 数据 | 建议规模 | 用途 |
| --- | ---: | --- |
| pilot train | 100,000–200,000 vali windows | 训练 head / adapter |
| pilot selection | 20,000–50,000 独立 vali windows | checkpoint 和架构选择 |
| pilot test | 50,000–100,000 test windows | 方案冻结后的最终小规模比较 |
| diagnostic balanced set | 10,000–25,000 windows | 专家均衡、边界和稳定性诊断 |

1k smoke 只适合检查代码闭环，不足以比较高方差架构；pilot train 不需要直接扩大到百万级。

### 5.2 分层抽样

主样本保持自然分布，同时按以下字段做最低覆盖约束：

- `dataset_name` / TSF cell；
- oracle expert；
- oracle 第一名与第二名误差差距的 quantile；
- TimeFuse 胜、Visual 胜、两者都失败的历史诊断类别；
- 序列尺度、频谱熵、趋势强度等结构统计分位区间。

另保留 oracle expert 近似均衡的 diagnostic set，但不得用均衡集指标替代自然分布主指标。

### 5.3 防泄漏与 checkpoint 选择

- pilot train 与 pilot selection 都来自 vali，但 sample key 不重叠；
- pilot test 只来自 test，不能用于超参数选择；
- 所有 scaler 只在 pilot train fit；selection/test 只 transform；
- future `y`、oracle error 和 expert prediction 只用于训练监督与评估，不进入 deployable feature；
- 每个变体至少运行 3 个 seed，报告均值和标准差。

## 6. 分轮实验设计

### Round 0：基线复现与诊断仪表

目标：确保小规模 protocol 能复现 full-scale 的相对趋势，并建立后续共同 evaluator。

固定输出：

- hard/soft MAE、MSE、regret、oracle accuracy；
- weight entropy、normalized entropy、mean max weight；
- selected model counts；
- 按 dataset、TSF cell、oracle expert、error-gap quantile 的分层结果；
- imageization、encoder forward 和总推理 latency；
- 三 seed mean/std。

Round 0 必须包含：旧 Visual Router、TimeFuse-style、global best single 和 oracle。若小规模样本无法复现“TimeFuse MAE 更好、Visual MSE 更好”的大致方向，应先调整抽样，不进入架构比较。

### Round 1：RevIN aux 与 pooling

这是成本最低、优先级最高的一轮。图像化和 frozen ViT 保持不变，只比较：

1. `CLS visual only`；
2. `mean_patch visual only`；
3. `concat(CLS, mean_patch) visual only`；
4. `RevIN aux only`；
5. `best visual pooling + RevIN aux concat`；
6. 可选：`best visual pooling + RevIN aux FiLM/gating`。

第一版 RevIN aux 建议只包含归一化过程实际移除或审计所需的低成本字段：

```text
mean, log_std, min, max, range, clip_ratio
```

这些字段需要用 pilot-train fitted scaler；不要在本轮直接加入完整 17 维 TimeFuse feature，以便区分“恢复被删除的信息”和“复制完整 baseline”。

Round 1 解释口径：

- aux-only 已接近 concat：主要收益来自尺度信息；
- concat 明显优于 aux-only 和 visual-only：视觉提供条件增量；
- mean-patch 明显优于 CLS：pooling 是主要瓶颈；
- 所有变体均无提升：优先检查 view/imageization，不直接增加 epoch。

### Round 2：异质 view 表达

固定 Round 1 最佳 pooling/head，比较：

1. 当前三异质 channel 直接作为 RGB；
2. spatial panel：line/fold/spectrum 分区排布后复制为灰度 RGB，只做一次 ViT forward；
3. 三 view 分别复制为灰度 RGB，经共享 ViT 独立编码后融合，作为约 3 倍计算的效果上限；
4. 仅在 2/3 显示正收益后，再实现 view-specific patch stem/token fusion。

spatial panel 应优先于三 encoder 路线，因为它能消除 channel 语义错配，同时保持一次 ViT forward。若 panel 接近独立编码的效果，应优先保留 panel。

### Round 3：信息密度与周期连续性

固定 Round 1/2 最佳方案，逐项比较：

- line raster vs value/difference stacked bands；
- fold view 是否显式提供 valid/padding mask；
- per-window normalized spectrum vs 同时保留 absolute log-power/total energy aux；
- hard top-1 fold vs soft fold mixture；
- `112x112` vs `224x224`；
- 必要时比较与 patch grid 对齐的低分辨率 token map。

soft fold mixture 第一版建议使用少量代表周期，例如 `4, 8, 12, 24, 48, 96`：

```text
period_weights = softmax(candidate_fft_scores / temperature)
fold_mix = sum(period_weights[p] * fold_image[p])
```

除主指标外，本轮必须执行连续性测试：对 `x` 加入不改变语义的小幅噪声或幅值扰动，记录 period weights、pseudo image、embedding 和 router weights 的变化。

### Round 4：表示能力 probe

在进入 ViT 微调前，对最佳 frozen visual representation 做下列轻量 probe：

1. 预测 RevIN aux / 17 维 TimeFuse-derived feature，报告逐特征 R²/MAE；
2. 预测 oracle expert，报告 class-balanced accuracy；
3. 预测五专家 error/regret vector，报告 rank correlation 与 top-1 accuracy；
4. 预测 dataset/TSF cell，检查 shortcut 强度；
5. 比较 visual embedding 与结构特征在 TimeFuse 失败样本上的条件增量。

如果 frozen visual 无法恢复简单结构特征，优先修复 imageization/encoder；如果结构 probe 良好但 regret probe 差，问题更接近 supervision/objective；如果 regret probe 良好但正式 routing 差，优先检查 head、loss、calibration 和训练顺序。

### Round 5：ViT domain adaptation 与有限联合微调

只有 Round 1–4 的最佳方案通过门禁后才进入。建议顺序：

1. 结构特征预测：保证 encoder 能恢复已知有效结构；
2. 五专家 error/regret ranking：直接对齐 routing 所需表示；
3. masked time-patch reconstruction 或跨 view reconstruction；不优先重建包含插值和 padding 的伪图像像素；
4. 可选的时序基础模型中间表征 distillation；
5. 冻结 encoder 训练 router；
6. 最后只解冻 ViT 后 1–2 个 block，或使用 adapter/LoRA 小学习率联合微调。

推荐多任务形式：

```text
L = L_fusion
  + lambda_rank * L_expert_regret_ranking
  + lambda_struct * L_structure_prediction
  + optional lambda_ssl * L_masked_or_cross_view
```

不建议第一步直接 full ViT + noisy routing loss 端到端训练。

## 7. Visual 独立价值与经济性门禁

TimeFuse + Visual 不应默认采用无条件大向量拼接。应先冻结 TimeFuse，令 Visual 只学习 residual：

```text
final_logits = timefuse_logits + alpha(x) * visual_residual_logits
```

必须报告：

- `alpha(x)` 的分布；
- Visual residual 改善覆盖的样本比例；
- 改善集中在哪些 TSF cell / expert pair / error tail；
- TimeFuse-only、Visual-only、hybrid 的 paired error difference；
- 每个样本增加的 encoder latency 和显存成本。

若 Visual 只对少量困难样本有效，优先采用 cost-aware cascade：

```text
cheap TimeFuse first
-> confidence high: direct output
-> entropy / expected regret high: invoke Visual encoder
```

Visual V2 进入 full-scale 前至少应满足以下一项，并且不能明显破坏另一主指标：

- 相对旧 Visual Router 的 pilot soft MAE 稳定降低至少 3%；
- 相对 TimeFuse-style 的 hybrid soft MAE 有跨 seed 的稳定改善；
- 在保持 TimeFuse MAE 基本不退化时显著降低 MSE/error-tail；
- cost-aware gate 只调用少量 Visual 样本，却保留大部分 hybrid 收益。

如果 visual residual 增益很小且调用比例很高，应停止大型 ViT 路线，转向更轻量的 raw-series CNN、patch encoder 或直接使用结构特征 router。

## 8. 建议的 V2 最小架构

第一版不追求最大容量，建议：

```text
x
├── RevIN normalized series
│   ├── dense line/value-difference view
│   ├── soft multi-period fold view
│   └── spectrum view with absolute-energy aux
│
├── spatial panel -> frozen ViT
│   -> mean-patch or view-region pooling
│
└── RevIN aux
    mean, log_std, min, max, range, clip_ratio

[visual pooled feature, scaled RevIN aux]
-> LayerNorm / small MLP
-> five expert logits
```

该架构仍只需一次 ViT forward，能够逐项处理尺度丢失、异质 channel mixing、CLS 限制、hard period 不连续和图像有效信息密度不足。

## 9. 运行产物与复现要求

pilot 输出继续写入：

```text
experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_v2_pilot_*/
```

较大 pilot 或 GPU 长跑可写入：

```text
/data2/syh/Time/run_outputs/YYYY-MM-DD_*_visual_router_v2_pilot_*/
```

每个 run 至少保存：

- sample manifest/reference 与 split summary；
- 完整 config 与 git commit；
- seed、sample key hash、feature/view schema version；
- checkpoint 与 selection metric；
- hard/soft summary、selected counts、分层结果；
- latency/resource summary；
- perturbation stability summary；
- status 与主日志。

小规模 pilot 可保存 pooled embeddings 以复用 frozen encoder 计算，但必须记录 pooling/view/imageization signature；不同 imageization 或 ViT 参数的 embedding 不得混用。正式 full-scale 仍遵守 online pseudo image / embedding 不长期落盘的主线约束。

## 10. 与重构任务并行的 Git 工作方式

当前代码重构和 Visual V2 实验可以并行，但不能在同一个 checkout 中由两个窗口反复切换 branch。推荐使用独立 worktree：

```bash
cd /home/shiyuhong/Time
git worktree add ../Time-visual-router-v2 -b exp/visual-router-v2-pilot <共同基线提交>
```

建议职责边界：

- `refactor/stage1-route-audit`：继续 canonical protocol/provider/runtime/artifact 重构，不改变研究口径；
- `exp/visual-router-v2-pilot`：新增 pilot sampler、view builder、pooling/head、probe 和小规模 launcher；
- 实验分支优先新增独立 V2 模块，不直接重写 `common/pseudo_imageization.py` 或正式 streaming entrypoint；
- 等某个变体通过 pilot 门禁，再把最小稳定模块迁移到 common/canonical FeatureProvider；
- prediction cache、SampleManifest、ExpertBatch、Evaluator 和 artifact schema 尽量沿用重构分支的稳定接口；
- 两边需要共享的修复以小而独立的 commit cherry-pick，避免大范围双向 merge。

推荐实验分支的初始文件边界：

```text
visual_router_experiments/stage1_vali_test_router/pilot/
  visual_router_v2_sample_protocol.py
  visual_router_v2_views.py
  visual_router_v2_models.py
  train_visual_router_v2_pilot.py
  evaluate_visual_router_v2_pilot.py
```

在 V2 方案胜出前，不修改旧 full-scale entrypoint 的默认行为，不覆盖旧 checkpoint，不把 pilot 输出写进代码目录。

## 11. 推荐执行顺序

1. 冻结共同 sample manifest 和 pilot train/selection/test sample key；
2. 复现旧 Visual 与 TimeFuse-style 小规模相对趋势；
3. 完成 Round 1：RevIN aux 与 pooling；
4. 完成 Round 2：spatial panel 与独立 view 上限；
5. 完成 Round 3：soft period、有效信息密度和连续性；
6. 用 Round 4 probe 判断是否值得微调 ViT；
7. 只有通过门禁后执行 Round 5 domain adaptation；
8. 最后评估 residual hybrid 与 cost-aware cascade；
9. 仅将明确胜出的最小方案提升为新的 full-scale candidate。

该顺序的核心是先用便宜、可解释的实验定位瓶颈，再决定是否投入昂贵的 ViT 微调与 full-scale 资源。
