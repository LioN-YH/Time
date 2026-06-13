# DLinear 96_48_S smoke test 实验日志

## 日志信息

- 日志日期：2026-06-10 02:52:51 CST
- 工作目录：`/home/shiyuhong/Time/quito`
- 目标模型：DLinear
- 任务配置：`96_48_S`
- GPU 数量：4
- 相关配置：
  - `configs/smoke/dlinear/96_48_S_smoke.yaml`
  - `configs/smoke/dlinear/96_48_S_smoke_eval.yaml`

## 目的

用少量 QuitoBench item 验证 DLinear 在 `96_48_S` 任务上的最小闭环是否可运行，包括训练、验证、checkpoint 保存、checkpoint 加载、测试集评估和结果 JSON 输出。

## 背景

原始 `configs/finetune/dlinear/96_48_S.yaml` 会使用全量 QuitoBench 数据，并训练 5 个 epoch，不适合作为快速 smoke test。为了提高验证效率，本次创建了专用 smoke 配置：

- 只取 4 个 10 分钟粒度 item。
- 只取 4 个小时粒度 item。
- 只训练 1 个 epoch。
- 使用 4 张 GPU 进行 DDP 训练。
- 输出目录独立设置为 `outputs/smoke/dlinear/96_48_S`，避免污染正式实验输出。

## 操作

1. 从 cluster 映射中选择少量 item：
   - min item：`153, 191, 286, 288`
   - hour item：`100011, 100012, 100019, 100044`
2. 新增训练 smoke 配置：
   - `configs/smoke/dlinear/96_48_S_smoke.yaml`
3. 第一次运行 `quito-cli finetune` 失败，原因是当前 shell 的 `PATH` 中没有 conda 环境里的 `torchrun`。
4. 将 `/home/shiyuhong/application/miniconda3/envs/quito/bin` 临时加入 `PATH` 后重新运行训练。
5. 训练成功后，生成 smoke 子集 parquet：
   - `examples/datasets/smoke_data/open_min_data.parquet`
   - `examples/datasets/smoke_data/open_hour_data.parquet`
6. 新增评估 smoke 配置：
   - `configs/smoke/dlinear/96_48_S_smoke_eval.yaml`
7. 使用训练得到的最佳 checkpoint 运行 `quito-cli evaluate`。
8. 检查评估输出 JSON。

## 结果

训练命令成功完成，关键结果如下：

- 训练样本数：`272640`
- 验证样本数：`67680`
- DLinear 参数量：`9312`
- 最佳验证 MAE：`0.14116157591342926`
- 最佳 checkpoint：
  - `outputs/smoke/dlinear/96_48_S/FINE_TUNE/ver_0/checkpoints/best_epoch=0_step=266_MAE=0.141.ckpt`

评估命令成功完成，输出文件如下：

- `outputs/smoke/dlinear/96_48_S/EVALUATE/ver_0/eval_results_DLinear.json`

评估结果包含 8 个 item，每个 item 均有 MSE 和 MAE：

- `153`
- `191`
- `286`
- `288`
- `100011`
- `100012`
- `100019`
- `100044`

示例结果：

| item_id | MSE | MAE |
| ---: | ---: | ---: |
| 153 | 0.2021941664 | 0.1677672505 |
| 191 | 0.7060929518 | 0.3167390167 |
| 286 | 0.8746324814 | 0.5085471609 |
| 288 | 0.4529374355 | 0.3110186540 |
| 100011 | 0.0038270892 | 0.0285188183 |

## 结论

DLinear 在 `96_48_S` 上的最小训练和评估闭环已经跑通。当前本地环境可以正常完成：

1. 4 卡 DDP 训练。
2. 验证集评估。
3. 最佳 checkpoint 保存。
4. Ray evaluator 加载 checkpoint。
5. 小规模 test split 评估。
6. 结果 JSON 输出。

本次还发现一个环境注意点：如果直接调用 conda 环境中的 `quito-cli`，需要确保该环境的 `bin` 目录在 `PATH` 中，否则 `quito-cli` 无法找到 `torchrun`。

## 下一步方案

1. 对 `PatchTST / 96_48_S` 和 `CrossFormer / 96_48_S` 复用同样 smoke 流程。
2. 如果两个模型也通过，再编写批量脚本自动完成：
   - smoke 配置生成；
   - 训练；
   - checkpoint 路径收集；
   - evaluate 配置生成；
   - 评估结果汇总。
3. 在正式跑全量实验前，先决定使用快速配置还是更接近论文的 `100 epoch + 3 seed` 设置。
