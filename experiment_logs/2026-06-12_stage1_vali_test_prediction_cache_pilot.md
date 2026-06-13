# Stage 1 Vali/Test Prediction Cache Pilot

日志日期：2026-06-12 12:46:33 CST

## 目的

将 Stage 1 prediction cache pilot 从 test-only 扩展到 vali+test，验证同一脚本能同时导出 router 训练 split 和测试 split 的 window-level `y_true/y_pred`。

## 背景

上一轮 test-only pilot 已经证明 DLinear `96_48_S` 的少量 test window 可以生成 manifest 和数组。本轮继续按计划验证 vali+test split 对齐，为后续在 vali 上训练 router、test 上评估 router 做准备。

## 操作

运行命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
  --splits vali test
```

默认 pilot 范围：

- 模型：DLinear；
- 配置：`96_48_S`；
- split：`vali` 和 `test`；
- 每个 dataset：1 个 item；
- 每个 item：channel 0；
- 每个 item-channel：前 2 个 window。

输出目录：

```text
experiment_logs/run_outputs/2026-06-12_124604_976311_visual_router_stage1_prediction_cache_pilot/
```

## 结果

生成 `manifest.csv` 共 8 条记录：

| split | dataset | 记录数 |
| --- | --- | ---: |
| `vali` | `TEST_DATA_MIN` | 2 |
| `vali` | `TEST_DATA_HOUR` | 2 |
| `test` | `TEST_DATA_MIN` | 2 |
| `test` | `TEST_DATA_HOUR` | 2 |

代表性窗口指标：

| sample_key | MAE | MSE |
| --- | ---: | ---: |
| `96_48_S__vali__TEST_DATA_MIN__item153__ch0__win0` | 0.572505 | 0.586542 |
| `96_48_S__vali__TEST_DATA_HOUR__item100011__ch0__win0` | 0.101723 | 0.014443 |
| `96_48_S__test__TEST_DATA_MIN__item153__ch0__win0` | 1.178757 | 1.803547 |
| `96_48_S__test__TEST_DATA_HOUR__item100011__ch0__win0` | 0.088284 | 0.017548 |

## 验证

已验证：

- `manifest.csv` 行数为 8。
- `vali/test` 各覆盖 `TEST_DATA_MIN` 和 `TEST_DATA_HOUR`。
- 每条 `y_true/y_pred` 文件均存在。
- 每条数组形状均为 `(48, 1)`。
- 逐条重算 MAE/MSE 与 manifest 记录一致。

## 结论

单专家 DLinear 的 `vali+test` window-level prediction cache pilot 已跑通。Stage 1 现在具备最小 router 训练/测试 split 的 cache 结构基础。

## 下一步方案

1. 将 pilot 扩展到五专家，保持同样的小范围，检查同一 `sample_key` 下专家集合是否完整。
2. 在五专家 manifest 上计算 window-level oracle label 和 expert regret。
3. 如果五专家小范围对齐无误，再逐步扩大 item/window 范围。
