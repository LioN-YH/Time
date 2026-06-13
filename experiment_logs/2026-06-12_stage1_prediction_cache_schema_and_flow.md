# Stage 1 Prediction Cache Schema 与 Quito 数据流阅读

日志日期：2026-06-12 07:58:18 CST

## 目的

按照 Visual Router Stage 1 主实验计划，阅读 Quito evaluate/data/model 数据流，确定 window-level prediction cache 的导出点，并建立第一版公共 schema。

## 背景

Stage 1 需要在 vali 上训练 router、在 test 上测试 router。由于后续 hard routing 和 softmax fusion 都依赖每个专家的窗口级预测，必须先建立 item-channel-window 级 prediction cache。此前已建立正式代码目录 `visual_router_experiments/`，本次开始向该目录写入 Stage 1 相关设计和公共 schema。

## 操作

1. 阅读 `quito/quito/scripts/evaluate.py`，确认当前 evaluate 脚本固定加载 `ModeType.TEST`，并在 `ModelEvaluator.evaluate_user()` 的 batch 循环中拿到 `batch` 和 `predictions`。
2. 阅读 `quito/quito/models/base.py`，确认 `eval_step()` 返回 `score_dict, y_pred`，真实未来序列可由 `batch["y"][:, -forecast_horizon:, :]` 获取。
3. 阅读 `quito/quito/datasets.py`，确认 `features=S` 会将原始 item-channel 展平为 `(N * C, L, 1)`，`shuffle=False` 时可从 batch 顺序恢复窗口序号。
4. 阅读 ES/SNaive 统计模型实现，确认统计模型也通过 `StatisticalModel` 返回 `[B, pred_len, C]` 形状预测。
5. 新增 `visual_router_experiments/common/prediction_cache_schema.py`，定义 cache key、manifest record、窗口级 MAE/MSE 计算和 manifest 校验工具。
6. 新增 `visual_router_experiments/stage1_vali_test_router/prediction_cache_design.md`，记录数据流阅读结论和 Stage 1 pilot 建议。
7. 对 schema 文件执行语法检查：

   ```bash
   python -m py_compile visual_router_experiments/common/prediction_cache_schema.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     -m py_compile visual_router_experiments/common/prediction_cache_schema.py
   ```

## 结果

新增文件：

| 文件 | 功能 |
| --- | --- |
| `visual_router_experiments/common/prediction_cache_schema.py` | item-channel-window prediction cache schema 与基础工具 |
| `visual_router_experiments/stage1_vali_test_router/prediction_cache_design.md` | Quito 数据流阅读结论与 Stage 1 cache pilot 设计 |

关键结论：

- 不建议第一版直接修改 Quito 原始 `evaluate.py`。
- 推荐在 `stage1_vali_test_router/` 中实现独立 cache builder，复用 Quito 的 `AutoConfig`、`load_datasets` 和 `AutoModel`。
- cache builder 应同时支持 `ModeType.VALID` 和 `ModeType.TEST`。
- 第一版 `sample_key` 为 `config_name + split + dataset_name + item_id + channel_id + window_index`。
- 初版 pilot 可先固定 `channel_id=0`，后续多通道时再显式恢复 channel 映射。

## 验证

已验证：

- `prediction_cache_schema.py` 在系统 Python 和 Quito conda 环境下均可通过 `py_compile`。
- `WORKSPACE_STRUCTURE.md` 已更新 `common/` 和 Stage 1 目录下的新文件说明。
- `visual_router_experiments/common/README.md` 和 `stage1_vali_test_router/README.md` 已更新当前文件索引。

## 结论

Stage 1 的 prediction cache schema 和导出方案已经明确。下一步可以在 `visual_router_experiments/stage1_vali_test_router/` 中实现小规模 cache builder，先用一个模型、一个配置、少量 item/window 验证 `y_true/y_pred`、window index、MAE/MSE 和 manifest 对齐。

## 下一步方案

1. 实现 `build_prediction_cache_pilot.py`，读取现有 `96_48_S` evaluate config。
2. 先选择 DLinear 和少量 item/window 验证单专家 cache。
3. 单专家 cache 正确后扩展到五专家，并检查同一 `sample_key` 下专家集合完整。
