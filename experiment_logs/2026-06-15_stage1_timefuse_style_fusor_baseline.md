# Stage 1 TimeFuse-style Fusor Baseline 实现与 Pilot 验证

日志日期：2026-06-15 01:53:30 CST

## 目的

实现一个公平的 TimeFuse-style fusor baseline，替代当前仅做 hard-label 分类的 TimeFuse-derived LogisticRegression 口径，并把它接入 `evaluate_router_baselines.py` 作为统一基线入口。

## 背景

此前 Stage 1 的 TimeFuse-derived baseline 只使用了单变量结构特征 `feature_cache.csv`，但训练器是 `StandardScaler + LogisticRegression` hard-label 分类器，没有复刻原生 TimeFuse 的单层 linear fusor 与融合预测 SmoothL1 loss 口径。为了与 visual router 公平比较，需要把 fusor 训练、hard top-1 选择、raw soft fusion 和统一 comparison 放到同一入口里。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/fusion_utils.py`，集中实现：
   - `feature_cache.csv` 读取与字段校验；
   - prediction manifest 读取；
   - 五专家 `y_pred/y_true` 数组读取；
   - hard top-1 / raw soft fusion MAE 与 MSE 复算；
   - 原生 TimeFuse-style `nn.Linear -> softmax -> weighted fusion -> SmoothL1Loss` fusor 训练与测试预测。
2. 改造 `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py`：
   - 保留统计规则 baseline；
   - 增加 `--timefuse-fusor {auto,on,off}`；
   - 默认可同时输出统计 baseline、TimeFuse-style fusor hard/raw-soft、oracle 和统一 comparison；
   - metadata 中标注旧 `timefuse_single_variable_logistic_regression` 为 `legacy/deprecated`。
3. 调整 `visual_router_experiments/stage1_vali_test_router/train_visual_router.py`，让视觉 router 与 fusor baseline 复用同一套 prediction lookup、数组读取和 soft fusion 指标逻辑。
4. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`、`WORKSPACE_STRUCTURE.md`，记录新共享模块、统一 evaluator 入口和新 pilot 输出目录。
5. 使用 Quito 环境执行验证命令：

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

## 结果

1. 新输出目录 `experiment_logs/run_outputs/2026-06-15_014918_visual_router_stage1_timefuse_fusor_baseline_pilot/` 已生成，包含：
   - `baseline_predictions.csv`
   - `baseline_summary.csv`
   - `baseline_comparison.csv`
   - `baseline_metadata.json`
   - `timefuse_fusor_predictions.csv`
   - `timefuse_fusor_raw_soft_fusion_predictions.csv`
   - `timefuse_fusor_summary.csv`
   - `timefuse_fusor_raw_soft_fusion_summary.csv`
   - `timefuse_fusor_selected_model_counts.csv`
2. 120 sample_key pilot 上，TimeFuse-style fusor 的 test 结果为：
   - hard top-1 MAE = `1.490870`
   - raw soft fusion MAE = `1.509144`
   - oracle MAE = `0.805392`
   - regret = `0.685478`
   - normalized weight entropy = `0.576704`
   - mean max weight = `0.619892`
3. `baseline_comparison.csv` 已同时包含统计规则 baseline、TimeFuse-style fusor hard/raw-soft 和 oracle 对照；旧 LogisticRegression 结果保留为 legacy/deprecated，不再作为主比较口径。

## 结论

TimeFuse-style fusor baseline 已按原生 TimeFuse 口径接入统一 evaluator，并在 120 sample_key pilot 上跑通。当前 fusor 在这批 pilot 上不优于 `global_best_single`，但它的训练和输出口径已经与原生 TimeFuse 对齐，后续可作为公平对照继续扩展到更大样本或更多 config。

## 下一步方案

1. 将 `evaluate_router_baselines.py` 的 fusor 路径作为默认统一入口继续沿用，后续只运行这个脚本即可得到统计 baseline、TimeFuse-style fusor 和 oracle 对照。
2. 根据需要把 `timefuse_fusor_*` 输出接入后续 visual router 报告或同表比较。
3. 若后续要扩大样本，优先复用同一套 `feature_cache.csv` / prediction manifest / comparison 口径，避免重新引入 hard-label LogisticRegression 作为主 baseline。
