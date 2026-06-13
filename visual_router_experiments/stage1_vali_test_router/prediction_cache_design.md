# Stage 1 Prediction Cache 设计记录

记录日期：2026-06-12 07:58:18 CST

## 目标

为 Stage 1 的 vali -> test visual router 主实验设计 window-level prediction cache 的导出位置、样本 key 和最小 pilot 实现路径。

## Quito 数据流阅读结论

### evaluate 入口

Quito 当前 evaluate 入口是：

```text
quito/quito/scripts/evaluate.py
```

关键逻辑：

- `main()` 固定调用 `load_datasets(..., mode=ModeType.TEST, concat=False)`，因此现有 evaluate 脚本只评估 test split。
- `ModelEvaluator.evaluate_user()` 会对每个 `user_id` 深拷贝 dataset，然后调用 `dataset.select_user_data(user_id)`。
- 每个 user 单独构造 `DataLoader(..., shuffle=False)`。
- batch 循环内调用：

  ```python
  loss_dict, predictions = self.model.eval_step(batch)
  ```

此时可以同时拿到：

- `batch["x"]`：当前窗口历史；
- `batch["y"]`：decoder label + 真实未来；
- `predictions`：模型预测；
- `loss_dict`：当前 batch 聚合指标。

### 模型 eval_step

通用模型实现位于：

```text
quito/quito/models/base.py
```

关键结论：

- `BaseModel.eval_step()` 会把 batch 移到模型 device，然后调用 `_eval_step()`。
- `_eval_step()` 返回 `score_dict, y_pred_point`。
- `y_pred_point` 形状为 `[batch_size, forecast_horizon, n_features]`。
- 真实未来序列是 `batch["y"][:, -forecast_horizon:, :]`。
- ES/SNaive 等统计模型继承 `StatisticalModel`，最终也返回同形状 tensor。

### Dataset 与 window_index

数据集实现位于：

```text
quito/quito/datasets.py
```

关键结论：

- `TimeSeriesDataset.__getitem__()` 根据滑动窗口索引 `j` 构造样本。
- `features == S` 时，原始 `(N, L, C)` 会被 reshape 为 `(N * C, L, 1)`。
- 因此 S 配置下 dataset 的样本轴已经是 item-channel 展开后的序列。
- 在 `select_user_data(user_id)` 后，dataset 保留该 user 的所有 channel。
- 如果当前数据实际只有单变量 target，则 `channel_id` 可先固定为 `0`。
- 因为 `DataLoader` 在 evaluate 中使用 `shuffle=False`，在单 item-channel pilot 中可用 batch 顺序恢复 `window_index`。

## Cache key

Stage 1 使用 item-channel-window 级 key：

```text
sample_key = config_name + split + dataset_name + item_id + channel_id + window_index
```

其中：

- `split` 为 `vali` 或 `test`；
- `dataset_name` 为 Quito dataset 配置名，例如 `TEST_DATA_MIN`；
- `window_index` 是当前 item-channel 在当前 split 内的滑动窗口序号。

## 推荐导出点

第一版 pilot 不建议直接改动 Quito 原始 evaluate.py，而是在 `visual_router_experiments/stage1_vali_test_router/pilot/` 中实现独立 cache builder：

1. 读取现有 evaluate config。
2. 使用 `AutoConfig.from_config()` 构造 data/model/training config。
3. 分别加载 `ModeType.VALID` 和 `ModeType.TEST` dataset。
4. 复用 `AutoModel.from_config()` 加载冻结专家。
5. 对少量 `item_id` 和窗口循环调用 `model.eval_step(batch)`。
6. 用 `visual_router_experiments.common.prediction_cache_schema` 写出 manifest 和数组。

这样可以避免为了 pilot 修改 Quito 主流程，也不会影响已有 baseline evaluate 口径。

## Pilot 限制

第一版 pilot 先限制：

- 单配置：`96_48_S`；
- 单变量 channel：`channel_id = 0`；
- 少量 item/cell；
- 少量 vali/test window；
- 五专家都使用现有 checkpoint/evaluate config 口径。

## 需要特别检查的问题

1. `select_user_data(user_id)` 后如果同一个 item 有多个 channel，如何稳定区分 `channel_id`。
2. `features=S` 的 reshape 顺序是 `(n c) l 1`，如果未来使用多变量数据，需要从原始 feature 列恢复 channel 映射。
3. 深度模型输出和统计模型输出应统一为 `[B, pred_len, C]`。
4. prediction cache 保存的 `y_true/y_pred` 应记录是否是标准化空间；初版建议与 Quito evaluate 指标保持一致，使用当前 eval_step 口径。
5. 如果后续要报告原始量纲指标，需要额外设计 inverse transform 口径。

## 下一步

1. 在 `stage1_vali_test_router/pilot/` 中实现小规模 cache builder。
2. 先用一个模型和少量 item 验证 `y_true/y_pred` shape、window_index 和 MAE/MSE。
3. 再扩展到五专家，检查同一 sample_key 下五专家是否完整对齐。
