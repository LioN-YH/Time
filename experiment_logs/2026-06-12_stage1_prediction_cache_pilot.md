# Stage 1 Prediction Cache 小规模 Pilot

日志日期：2026-06-12 08:03:23 CST

## 目的

实现并运行 Stage 1 的小规模 window-level prediction cache pilot，验证 `sample_key`、`y_true/y_pred` 数组落盘、窗口级 MAE/MSE 和 manifest schema 是否可用。

## 背景

前一步已经完成 Quito evaluate 数据流阅读和 `prediction_cache_schema.py`。当前需要用最小范围实际跑通一条 prediction cache 生成链路，确认后续扩展到五专家、vali/test 和更多 item/window 时不会出现 key 对齐或数组形状问题。

## 操作

1. 新增脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py
   ```

2. 脚本默认配置：

   - 模型：DLinear；
   - 配置：`96_48_S`；
   - split：`test`；
   - 每个 dataset：1 个 item；
   - 每个 item：channel 0；
   - 每个 item-channel：前 2 个 window；
   - 输出：`manifest.csv`、`metadata.json`、`arrays/**/*.npy`。

3. 首次运行失败，原因是脚本直接运行时工作区根目录未加入 `sys.path`，导致无法导入 `visual_router_experiments` 包。
4. 修复方式：

   - 在脚本中同时加入 `WORKSPACE` 和 `quito/` 到 `sys.path`；
   - 为 `visual_router_experiments/`、`common/`、`stage0_oracle_audit/`、`stage1_vali_test_router/`、`stage2_heldout_cell/` 增加 `__init__.py`。

5. 数据检查发现原始数据每个 item 有 `ind_1` 到 `ind_5` 共 5 个指标通道；修正 pilot 脚本，使其从全局样本序号恢复：

   ```text
   channel_id = global_sample_index // len_per_channel
   window_index = global_sample_index % len_per_channel
   ```

6. 对脚本执行语法检查：

   ```bash
   python -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
     visual_router_experiments/common/prediction_cache_schema.py

   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     -m py_compile \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py \
     visual_router_experiments/common/prediction_cache_schema.py
   ```

7. 运行默认 pilot：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py
   ```

## 输出

输出目录：

```text
experiment_logs/run_outputs/2026-06-12_080303_768841_visual_router_stage1_prediction_cache_pilot/
```

输出文件：

| 文件/目录 | 功能 |
| --- | --- |
| `manifest.csv` | 4 条 window-level prediction cache manifest 记录 |
| `metadata.json` | pilot 参数、模型、配置和输出目录 |
| `arrays/test/TEST_DATA_MIN/DLinear/*.npy` | 分钟级 dataset 的 `y_true/y_pred` 数组 |
| `arrays/test/TEST_DATA_HOUR/DLinear/*.npy` | 小时级 dataset 的 `y_true/y_pred` 数组 |

## 结果

本次 pilot 生成 4 条记录：

| sample_key | MAE | MSE |
| --- | ---: | ---: |
| `96_48_S__test__TEST_DATA_MIN__item153__ch0__win0` | 1.178757 | 1.803547 |
| `96_48_S__test__TEST_DATA_MIN__item153__ch0__win1` | 1.284034 | 2.078254 |
| `96_48_S__test__TEST_DATA_HOUR__item100011__ch0__win0` | 0.088284 | 0.017548 |
| `96_48_S__test__TEST_DATA_HOUR__item100011__ch0__win1` | 0.088261 | 0.016615 |

数组形状均为：

```text
y_true: (48, 1)
y_pred: (48, 1)
```

## 验证

已验证：

- `manifest.csv` 行数为 4。
- 每条记录的 `y_true_path` 和 `y_pred_path` 均存在。
- 逐条重新读取 `.npy` 后重算 MAE/MSE，与 manifest 中记录一致。
- `history_length=96`，`pred_length=48`，与 `96_48_S` 配置一致。
- `sample_key` 中包含 `config_name/split/dataset_name/item_id/channel_id/window_index`。

运行时出现 `torch.load(weights_only=False)` 的 FutureWarning，这是 PyTorch 对 checkpoint 反序列化默认行为的提示；本次加载的是本地已有 checkpoint，不影响 pilot 结果。后续如改造 Quito checkpoint 加载，可单独评估是否设置 `weights_only=True`。

## 结论

Stage 1 prediction cache 的最小链路已经跑通。当前 schema 可以保存 item-channel-window 级 `y_true/y_pred`、窗口级 MAE/MSE 和专家版本信息；脚本也能从 S 配置的展开样本顺序中恢复 channel/window。

## 下一步方案

1. 将 pilot 从 test-only 扩展到 vali+test，仍先单专家验证 split 对齐。
2. 将单专家 DLinear 扩展到五专家，检查同一 `sample_key` 下专家集合是否完整。
3. 在五专家 cache 上计算 window-level oracle label 和 expert regret。
4. 再接入伪图像 tensor/embedding 生成。
