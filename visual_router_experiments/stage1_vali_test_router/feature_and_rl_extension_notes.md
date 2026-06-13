# Stage 1 结构特征与 RL 支线扩展记录

本文档记录 Stage 1 后续可扩展但不作为当前主线优先级最高的两个方向：

- TimeFuse-style 统计/结构特征 router 对照支线；
- 将当前动作/状态路由进一步形式化为 contextual bandit 或强化学习的可能路径。

当前主线仍是视觉结构先验路由：将单变量历史窗口转为内生图像，使用 ViT 或其他视觉 encoder 得到表示，再在冻结专家之间做 item-channel-window 粒度的 hard routing 或 soft fusion。统计/结构特征支线主要用于建立非视觉对照和增强论文分析，不追求过度复杂的手工特征工程。

## TimeFuse-Style 结构特征支线

### 数据流口径

推荐采用如下特征构造流程：

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

这里的 `x window` 已经经过 Quito 的 train-based normalization，与冻结专家实际看到的输入尺度一致。后续所有 router feature 都只允许使用历史窗口 `x`，不得使用预测目标 `y` 或 test split 的全局统计量。

### A/B/C 的含义

`A. train-normalized statistical/shape features`

在 Quito train-based normalization 后的窗口上直接计算统计和形状特征。它保留了专家实际输入尺度下的 level、volatility、trend 和周期结构，因此可以作为 TimeFuse-style meta-feature 对照。

`B. window-RevIN 后的 shape features`

对当前窗口做 RevIN-style 标准化后再提取形状特征。该分支用于弱化窗口内 level/scale，强调相对形状、自相关、频域形态、局部变化复杂度等信息。RevIN 后不应再保留近似常数的 mean/std 特征。

`C. RevIN mean/std/range 等 scale features`

显式保存 window-RevIN 被移除的尺度信息，例如窗口均值、标准差、range、绝对 level、波动强度等。这样可以避免 B 分支丢掉可能对专家选择有用的 scale 信号。

### 与 TimeFuse 比较时的约束

为了后续与 TimeFuse 比较，这条支线应尽量贴近 TimeFuse 的思想：用输入窗口的 meta-feature 学习一个 fusor/router。但当前项目主线不是统计特征路由，因此建议做一个克制版本：

- 优先面向单变量 item-channel-window 样本提取特征；
- 暂时剔除多变量专属特征，例如跨变量 covariance、cross-correlation、变量间 spectral variation；
- 保留少量可解释的单变量统计、趋势、自相关和频域特征；
- 统计特征 router 作为非视觉 baseline 或 ablation，不作为最终主贡献。

可选的单变量特征池包括：

- 基础统计：mean、std、min、max、range、分位数、偏度、峰度；
- 趋势变化：线性斜率、首尾差、平均一阶差分、变化率统计；
- 自相关结构：lag-1 autocorrelation、若干固定 lag 的 autocorrelation、acf 衰减摘要；
- 频域结构：主频位置、谱熵、低/中/高频能量比例；
- 稳定性和复杂度：近似 stationarity 标记、zero crossing、局部峰谷数量。

### 标准化方式

`concat(A, B, C)` 后使用一个 feature scaler，在 router 训练 split `vali` 上 fit，在 `test` 上只做 transform：

```text
features_vali = concat(A_vali, B_vali, C_vali)
features_test = concat(A_test, B_test, C_test)

feature_scaler.fit(features_vali)
features_vali_scaled = feature_scaler.transform(features_vali)
features_test_scaled = feature_scaler.transform(features_test)
```

如果使用 `StandardScaler`，其含义是按列做 z-score：

```text
X_scaled[:, j] = (X[:, j] - mean_vali[j]) / std_vali[j]
```

因此不需要为 A、B、C 分别维护三个 scaler。concat 后 fit 一个 scaler，在数学上仍然是逐列标准化，工程上更容易保证特征列顺序和保存/加载一致。

### 冗余与控制

A、B、C 之间会存在冗余，尤其是 A 的 mean/std/range 与 C 的 RevIN scale 参数可能高度相关。该冗余可以接受为第一版诊断性特征池，但正式报告时需要做去冗余或 ablation。

建议的控制方式：

- 删除常数列和近似常数列；
- 删除完全重复列；
- 训练 router 时使用 L2 regularization 或 weight decay；
- 做 A-only、B-only、C-only、A+B、B+C、A+B+C 的 ablation；
- 如果统计特征 router 表现很好，应检查是否主要来自 dataset/TSF-cell/scale shortcut，而不是视觉结构信息。

## 动作/状态路由与强化学习

### 当前 Stage 1 更像 contextual bandit

当前 Stage 1 的 router 可以形式化为动作/状态问题：

- 状态 `s`：窗口视觉/结构特征、metadata、`config_name`、dataset、TSF cell 等；
- 动作 `a`：在同一个 `config_name` 下选择冻结专家，动作空间边界见 `stage1_cache_contract.md`；
- 奖励 `r`：可以取 `-MAE`、`-MSE` 或 `-regret_to_oracle`；
- 转移：当前 item-channel-window 样本之间没有真实 sequential decision dynamics，一个窗口上的专家选择不会改变下一个窗口的状态。

因此第一版不建议称为完整强化学习。它更适合称为 supervised router 或 contextual bandit：

1. 监督学习：用 `oracle_model` 作为 hard label，或用每个专家的误差/regret 构造 soft target。
2. Contextual bandit 扩展：用 reward-weighted classification、IPS/DR 离线评估或无状态 policy gradient 表达 router policy。
3. 完整 RL 只有在引入多步观测、预算、级联调用或逐步修正时才更自然。

当前结论是：能引入 RL 语言，但主线应先打实 supervised router + regret/reward 口径。完整 RL 会显著增加训练不稳定性、离线评估难度和审稿解释成本。

### 视觉多步自适应的可行扩展

当前视觉路由的粒度是 item-channel-window：每个单变量历史窗口被图像化后交给 ViT 编码，然后 router 在同 config 的专家集合内选择或加权融合。这个设置天然是单步决策。要扩展成更像 RL 的视觉多步自适应，需要让一个样本的决策过程包含多个阶段，并让早期动作影响后续可用信息、计算成本或最终预测。

可考虑的多步自适应方向如下。

#### 1. 视觉证据逐步获取

将一个窗口图像拆成不同视觉视图或不同成本的编码步骤：

```text
step 1: 低分辨率/轻量 CNN/浅层 ViT embedding
action: 直接选择专家，或继续请求更高成本视觉证据

step 2: 高分辨率图像/更深层 ViT embedding/多尺度图像
action: 选择专家，或继续请求额外统计特征/局部裁剪

step 3: 完整视觉表示 + 可选结构特征
action: 最终专家选择或 soft fusion
```

奖励可以设计为：

```text
reward = -forecast_error - lambda * compute_cost
```

这时动作会影响后续状态，因为“是否继续看更贵的视觉证据”会改变下一步可用表示和累计成本。这比当前单步 router 更接近强化学习，也更适合解释为 cost-aware dynamic routing。

#### 2. 专家级联与早退

将专家选择改成级联过程，而不是一次性从五专家中选一个：

```text
step 1: 先调用低成本专家或统计基线
action: 接受当前预测，或升级到深度专家

step 2: 调用一个深度专家
action: 接受预测，或继续调用另一个互补专家做 fusion

step 3: 最多调用 K 个专家后输出加权融合
```

该方向的优势是部署意义更强：router 不只优化误差，还优化推理成本。它适合在论文中作为后续扩展，而不是 Stage 1 第一版主实验。

#### 3. 图像化策略自适应

当前默认是将单变量历史窗口转为一种内生图像。如果后续存在多种图像化方式，可以让 policy 选择图像构造策略：

```text
action examples:
- 使用 line plot / recurrence plot / Gramian angular field / spectrogram；
- 使用单尺度或多尺度窗口图像；
- 选择是否加入趋势去除、RevIN、频域视图；
- 选择 ViT patch size 或视觉 encoder 层级。
```

这类动作直接影响后续视觉状态，但动作空间会迅速变大。更稳的做法是先离线构造固定候选视图，然后把“选择视图 + 选择专家”做成 contextual bandit 或小规模 sequential policy。

#### 4. 跨相邻窗口的序列化路由

如果将同一 item-channel 的相邻 windows 按时间顺序组织，router 可以维护一个历史状态：

```text
state_t = current visual embedding + previous routing decisions + recent errors/proxy signals
action_t = expert choice or fusion weights
```

这看起来更像 RL，但要谨慎：test 时通常拿不到真实未来误差作为即时反馈，除非部署场景允许在线更新或延迟反馈。因此离线论文实验中，该方向容易变成使用事后误差的模拟环境，解释成本较高。

### 推荐优先级

短期建议：

1. 继续完成 supervised visual router：图像化历史窗口、ViT embedding、vali 训练、test 评估。
2. 加一个轻量 TimeFuse-style 统计特征 router，对照视觉 router 是否真的使用了视觉结构信息。
3. 在结果表中报告 `regret_to_oracle`、`oracle_label_accuracy` 和相对 best single / metadata baseline 的收益。

中期建议：

1. 把 supervised router 改写为 contextual bandit 口径，奖励使用 `-MAE` 或 `-regret_to_oracle`。
2. 加入 cost-aware 指标，例如视觉 encoder 成本、专家调用数量、平均调用深度。
3. 做离线 policy evaluation 的小规模诊断，但不把它作为主结论。

长期建议：

1. 设计视觉证据逐步获取或专家级联，让 action 真正影响后续状态和计算成本。
2. 将问题表述为 budgeted sequential decision making。
3. 在有稳定 supervised router 和 oracle gap 结果后，再尝试 RL / policy gradient / actor-critic 等训练方法。

### 对当前视觉路线的判断

以当前 item-channel-window 单变量图像化粒度来看，扩展 RL 的可能性存在，但第一阶段最自然的是 contextual bandit。真正有价值的 RL 扩展不是简单把 expert selection 换成 policy gradient，而是引入“看多少视觉证据、调用多少专家、是否早退”的预算约束。

因此后续论文路线可以这样分层：

- 主线：Visual Router over Frozen Experts；
- 对照：TimeFuse-style statistical feature router；
- 支线：Contextual Bandit Router Policy；
- 远期：Cost-aware Visual Sequential Routing / Expert Cascade。
