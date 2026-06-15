# Stage 1 TimeFuse Metadata Baseline 口径审计

日志日期：2026-06-15 00:55:47 CST

## 目的

审计当前 Stage 1 中被称作 metadata / TimeFuse-derived baseline 的实现口径，确认它与原生 TimeFuse 仓库的 router/fusor 结构和训练 loss 是否一致，并为后续公平对比的修改方案提供依据。

## 背景

用户指出当前 metadata baseline 记忆中来源于 TimeFuse 改造，但路由模块可能不是 TimeFuse 原仓库里的单层 MLP/linear fusor，而是另一种架构；为避免视觉 router 与 TimeFuse-style baseline 的比较不公平，需要先核实现有实现，再决定是否修改 router 模块和训练 loss。

## 操作

1. 阅读 `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py`，确认当前 metadata/statistics baseline 是基于 `dataset_name`、`group_name` 等元信息的规则映射，不训练神经 fusor。
2. 阅读 `visual_router_experiments/stage1_vali_test_router/pilot/build_structure_feature_cache_pilot.py`，确认 TimeFuse-derived 结构特征 cache 使用历史窗口 `x` 提取 17 个单变量元特征，其中 `autoreg_coef_mean` 和 `residual_std_mean` 来自 `statsmodels.tsa.ar_model.AutoReg`，这是特征提取中的 AR landmarker，不是 router 架构。
3. 阅读 `visual_router_experiments/stage1_vali_test_router/pilot/train_structure_router_pilot.py`，确认当前 TimeFuse-derived 结构特征 router 为 `StandardScaler + LogisticRegression(class_weight='balanced')`，训练目标是 `oracle_model` hard label 分类。
4. 阅读 `TimeFuse/timefuse.py` 和 `TimeFuse/run_timefuse_exp.ipynb`，确认原生 TimeFuse 的 `ModelFusor` 是单层 `nn.Linear(input_dim, output_dim)` 后接 `softmax` 输出模型融合权重；训练时用这些权重加权 `y_model_preds`，得到 `fused_output`，再用 `SmoothL1Loss(beta=0.01)` 等预测误差 loss 对 `y_true` 反传。
5. 对照当前视觉 router 入口 `train_visual_router.py`、`train_visual_router_online.py`、`train_visual_router_online_streaming.py`，确认视觉主线当前已使用小型 MLP 输出五专家 softmax 权重，并在 `fusion_huber_kl` 模式下用融合预测 SmoothL1 主损失加 KL 辅助损失训练。

## 结果

当前实现口径存在三类不同 baseline / router：

1. `evaluate_router_baselines.py` 的 metadata/statistics baseline：
   - 只根据全局、dataset、TSF cell、dataset+TSF cell 在 vali 上选择专家；
   - 不使用 TimeFuse meta-feature；
   - 不训练神经网络；
   - 不是用户所说的 TimeFuse-derived router。
2. `pilot/train_structure_router_pilot.py` 的 TimeFuse-derived 结构特征 router：
   - 输入是从历史 `x` 提取的 17 维 TimeFuse-derived 单变量元特征；
   - 当前 router 是 `StandardScaler + LogisticRegression`；
   - loss 等价于 hard oracle label 分类，不直接优化融合预测误差；
   - 与 TimeFuse 原生 `Linear -> softmax weights -> weighted_sum(y_pred) -> SmoothL1(y_true)` 不一致。
3. 当前视觉 router：
   - 输入是 ViT embedding；
   - router head 是两层小型 MLP，不是 TimeFuse 原生单层 linear fusor；
   - `fusion_huber_kl` 已接近 TimeFuse 的融合预测误差训练，但额外加入了 KL soft oracle 辅助项，且默认 `SmoothL1Loss(beta=0.1)`，不同于 notebook 中 `beta=0.01` 的代表配置。

因此，用户关于公平比较的担忧成立：现有 TimeFuse-derived baseline 只借用了 TimeFuse 的元特征思路，没有复刻 TimeFuse 原生 fusor head 和融合预测 loss。

## 结论

本次没有修改训练代码，只完成实现口径审计。严格说，当前 `timefuse_single_variable_logistic_regression` 不应命名为原生 TimeFuse baseline，更准确是 “TimeFuse-derived meta-feature + LogisticRegression hard routing baseline”。若要和视觉 router 公平比较，应新增或改造一个 TimeFuse-style fusor baseline：使用相同 feature cache 和 prediction cache，按 `config_name` 独立训练单层 linear softmax fusor，并用五专家融合预测对 `y_true` 的 SmoothL1/MSE/MAE loss 训练。

## 下一步方案

1. 新增正式或 pilot 入口，例如 `pilot/train_timefuse_fusor_router_pilot.py`，保留旧 LogisticRegression 结果作为历史对照，不直接覆盖。
2. 模型结构按 TimeFuse 原生实现：
   - `StandardScaler` 只在 vali feature 上 fit；
   - `nn.Linear(feature_dim, num_experts)` 输出 logits；
   - `softmax(logits)` 得到五专家权重；
   - hard top-1 取最大权重专家，soft fusion 按权重融合五专家预测。
3. 训练 loss 先复刻 TimeFuse 代表 notebook：
   - 默认 `SmoothL1Loss(beta=0.01)`；
   - 可通过参数切换 `mse`、`mae`、`huber`、`mix`，但主表优先引用 huber 口径；
   - 不加 KL 辅助项，避免把视觉 router 的额外设计带入 TimeFuse baseline。
4. 输出文件复用现有 router 生态字段：
   - `structure_router_predictions.csv` 或新的 `timefuse_fusor_predictions.csv`；
   - summary、selected counts、metadata；
   - 保存每个 test sample 的 `weight_{model}`、hard top-1 指标和 raw soft fusion 指标，便于接入 calibration / comparison。
5. 完成代码修改后用 Quito 环境在现有 120 sample_key pilot 上复验，再决定是否扩到 `96_48_S` 1k 或 full-scale。
