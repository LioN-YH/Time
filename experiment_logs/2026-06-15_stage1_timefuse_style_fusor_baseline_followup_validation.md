# Stage 1 TimeFuse-style Fusor Baseline 续接复核与口径修正

日志日期：2026-06-15 02:41:24 CST

## 目的

对前一轮已经实现的 TimeFuse-style fusor baseline 做收尾复核，确认实现口径、输出文件、统一评估入口和文档说明都已经落稳，并把 README 里仍然把旧 LogisticRegression 口径写成主 comparison 必选项的表述修正掉。

## 背景

上个窗口因为上下文限制中断时，工作树里已经有 fusor baseline 实现、pilot 输出和部分文档改动，但还需要重新核一遍当前状态，避免只凭记忆判断“已经完成”。这次续接重点不是再改训练逻辑，而是把证据闭环补齐。

## 操作

1. 复核 `TimeFuse/timefuse.py` 和 `TimeFuse/run_timefuse_exp.ipynb`，再次确认原生口径是单层 `nn.Linear(input_dim, output_dim)`，`forward` 后接 `softmax`，训练时用五专家加权后的 `fused_output` 配合 `SmoothL1Loss(beta=0.01)` 反传。
2. 复核 `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py` 与 `fusion_utils.py`，确认统一入口已经接入 TimeFuse-style fusor baseline，且按 `config_name` 独立训练，不跨 config 共享动作空间。
3. 检查现有 pilot 输出目录 `experiment_logs/run_outputs/2026-06-15_014918_visual_router_stage1_timefuse_fusor_baseline_pilot/`，确认关键产物齐全：
   - `baseline_predictions.csv`
   - `baseline_summary.csv`
   - `baseline_comparison.csv`
   - `baseline_metadata.json`
   - `timefuse_fusor_predictions.csv`
   - `timefuse_fusor_raw_soft_fusion_predictions.csv`
   - `timefuse_fusor_summary.csv`
   - `timefuse_fusor_raw_soft_fusion_summary.csv`
   - `timefuse_fusor_selected_model_counts.csv`
4. 用 Quito 环境执行语法检查：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
  visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
  visual_router_experiments/stage1_vali_test_router/fusion_utils.py \
  visual_router_experiments/stage1_vali_test_router/train_visual_router.py
```

5. 复核缓存和 manifest 规模：
   - `feature_cache.csv`：120 行；
   - `window_oracle_labels_with_tsf_cell.csv`（`metric=mae`）：120 行；
   - `manifest.csv`：600 行，120 个 sample_key，每个 sample_key 5 个专家完整。
6. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`，把“中等规模 comparison 必选项”中的旧 `timefuse_single_variable_logistic_regression` 改成 `timefuse_style_fusor` 的 hard top-1 和 raw soft fusion，并明确旧 LogisticRegression 只作为 legacy/deprecated 历史附录。
7. 补充旧 `pilot/train_structure_router_pilot.py` 的 legacy/deprecated 标记，重跑旧结构特征 LogisticRegression pilot，使 `structure_router_summary.csv` 与 `structure_router_metadata.json` 显式记录 `legacy_deprecated`；同时让 visual router 的 comparison helper 在读取该历史结果时带上 `method_status` 字段。

## 结果

1. 语法检查通过，未引入新的 Python 级错误。
2. 统一入口的 pilot 输出已经包含统计 baseline、TimeFuse-style fusor hard top-1、raw soft fusion 和 oracle 对照。
3. 当前 120 sample_key pilot 的关键数值保持为：
   - hard top-1 MAE = `1.490870`
   - raw soft fusion MAE = `1.509144`
   - oracle MAE = `0.805392`
   - regret = `0.685478`
   - normalized weight entropy = `0.576704`
   - mean max weight = `0.619892`
4. `README.md` 已把旧 LogisticRegression 口径降级为历史参考，不再作为主 comparison 的默认必选项。
5. 旧 `timefuse_single_variable_logistic_regression` 的历史输出已标注为 `legacy_deprecated`；后续 visual router comparison 若继续读取该表，也会通过 `method_status` 明确标识它不是 active 主 baseline。

## 结论

TimeFuse-style fusor baseline 的实现、统一入口、pilot 输出和文档说明已经连到一起了。当前这条 baseline 在 120 sample_key pilot 上不优于 `global_best_single`，但它的训练和评估口径已经与原生 TimeFuse 对齐，足以作为后续 visual router 的公平同表基线。旧 LogisticRegression 口径仍可复核历史结果，但已经不再以 active 主 baseline 形式出现。

## 下一步方案

1. 后续只要继续扩样本或扩 config，优先复用 `evaluate_router_baselines.py` 这一统一入口。
2. 旧 `timefuse_single_variable_logistic_regression` 只保留作历史对照，不再回到主比较口径。
3. 如果要继续向正式结果推进，优先在更大样本或更多 config 上复跑同一套 fusor 流水线，再和 visual router 的 hard/soft/校准结果同表比较。
