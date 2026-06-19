# Stage 1 P3a minimal fusion/metrics extraction

日志日期：2026-06-19 17:56:56 CST

## 目的

抽出 Stage 1 最小共享 fusion/metrics helper，让 golden smoke 使用统一 helper 复算 hard top-1 fusion、raw soft fusion、MAE 和 MSE，同时保持当前 golden 数值、sample_key 顺序、五专家顺序、row index 和 shape 契约不变。

## 背景

P3 的完整 metrics/fusion 收束后续仍会涉及更多报告、summary 和入口迁移。本次只做 P3a minimal extraction：新增纯 numpy helper，不引入 torch/sklearn 训练依赖，不迁移 Visual Router 或 TimeFuse fusor 正式训练入口，不实现 calibration，不改变 output schema，不修改 `PredictionBatchReader` / `OracleTsfReader`，不改变模型结构、loss 或正式输出目录。

## 操作

1. 审查 `tests/smoke/stage1_golden_smoke.py` 中原有 hard top-1、raw soft fusion、MAE 和 MSE 的本地 numpy 计算口径。
2. 新增 `time_router/evaluation/metrics.py` 和 `time_router/evaluation/__init__.py`：
   - `compute_mae(...)` 和 `compute_mse(...)` 只计算同形数组的全元素 MAE/MSE。
   - `validate_fusion_inputs(...)` 显式校验 `y_pred`、`y_true`、`weights`、`model_columns` 的 shape 和专家数量。
   - `hard_top1_fusion(...)` 返回 `selected_indices`、`selected_models`、`fused_pred`、`mae`、`mse`。
   - `raw_soft_fusion(...)` 保持现有 golden smoke 的线性加权口径，只做 `sum_expert(weights * y_pred)`。
3. 修改 `tests/smoke/stage1_golden_smoke.py`，改为调用共享 helper 复算 hard top-1 和 raw soft fusion；原有 golden 数值、sample_key 顺序、五专家顺序、row index 和 shape 断言全部保留。
4. 使用 Quito 环境运行两条指定 smoke。
5. 2026-06-19 18:04:12 CST 追加补漏：根据协作复核反馈，同步更新 `docs/refactor/stage1_refactor_roadmap.md`，补充 P3a 已完成状态、完成范围和明确不做事项。前一版提交只按粘贴任务的直接交付项更新了 helper、golden smoke、日志和结构索引，遗漏了路线图状态留痕；本次仅补文档追踪，不改 helper 或 smoke 逻辑。

## 结果

验收命令均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
```

关键结果：

- golden smoke 通过，五专家顺序仍为 `['DLinear', 'PatchTST', 'CrossFormer', 'ES', 'NaiveForecaster']`。
- sample_key 顺序仍为 4 个 dry-run fixture key，`y_pred` shape 仍为 `(4, 5, 48, 1)`，`y_true` shape 仍为 `(4, 48, 1)`。
- hard top-1 选择仍为 `['CrossFormer', 'DLinear', 'PatchTST', 'DLinear']`，MAE 为 `0.416048437`，MSE 为 `0.456369758`。
- raw soft fusion MAE 为 `0.410296679`，MSE 为 `0.488154024`。
- oracle/TSF smoke 继续通过，说明 P3a 没有影响 P2 reader smoke。
- 已补充 `docs/refactor/stage1_refactor_roadmap.md` 的 P3a 状态，明确最小 helper 已完成，完整 P3 和正式入口迁移仍留到后续小步。

## 结论

本次只完成 P3a minimal metrics/fusion extraction。共享 helper 已能覆盖 golden smoke 需要的 hard top-1、raw soft fusion、MAE、MSE 和必要输入校验；路线图已同步记录 P3a 状态；正式 Visual Router / TimeFuse fusor 入口、calibration、报告 schema、reader、模型结构、loss 和正式输出目录均未迁移或修改。

## 下一步方案

后续如进入完整 P3，应另起小步提交，在 golden smoke 通过的前提下再逐步收束更多 metrics、summary 和报告逻辑；正式入口迁移仍保留到 P6，迁移前后必须继续运行 golden smoke 并对比关键契约。
