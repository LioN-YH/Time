# PatchTST 和 CrossFormer 96_48_S smoke test 实验日志

## 日志信息

- 日志日期：2026-06-10 02:58:40 CST
- 工作目录：`/home/shiyuhong/Time/quito`
- 目标模型：PatchTST、CrossFormer
- 对照模型：DLinear
- 任务配置：`96_48_S`
- GPU 数量：4
- 数据子集：`examples/datasets/smoke_data`

## 目的

在 DLinear smoke test 已经跑通的基础上，继续验证 PatchTST 和 CrossFormer 是否也能在 QuitoBench `96_48_S` 小规模子集上完成训练、验证、checkpoint 保存、checkpoint 加载和测试集评估。

## 背景

正式实验需要运行 CrossFormer、PatchTST 和 DLinear。此前 DLinear 已经完成最小闭环验证。为了在进入全量配置前提前发现模型级问题，本次沿用同一份 smoke 数据子集，并为 PatchTST 和 CrossFormer 分别创建训练和评估配置。

## 操作

1. 新增 PatchTST smoke 训练配置：
   - `configs/smoke/patchtst/96_48_S_smoke.yaml`
2. 新增 CrossFormer smoke 训练配置：
   - `configs/smoke/crossformer/96_48_S_smoke.yaml`
3. 使用 4 卡分别运行 PatchTST 和 CrossFormer 的 `quito-cli finetune`。
4. 检查两个模型的 best checkpoint 是否生成。
5. 新增 PatchTST smoke 评估配置：
   - `configs/smoke/patchtst/96_48_S_smoke_eval.yaml`
6. 新增 CrossFormer smoke 评估配置：
   - `configs/smoke/crossformer/96_48_S_smoke_eval.yaml`
7. 使用 4 个 Ray evaluator 分别运行 PatchTST 和 CrossFormer 的 `quito-cli evaluate`。
8. 读取三个模型的 smoke evaluation JSON，汇总平均 MSE 和 MAE。

## 结果

PatchTST 训练成功：

- 参数量：`2448064`
- 最佳验证 MAE：`0.1698804497718811`
- best checkpoint：
  - `outputs/smoke/patchtst/96_48_S/FINE_TUNE/ver_0/checkpoints/best_epoch=0_step=266_MAE=0.170.ckpt`
- 评估结果：
  - `outputs/smoke/patchtst/96_48_S/EVALUATE/ver_0/eval_results_PatchTST.json`

CrossFormer 训练成功：

- 参数量：`525728`
- 最佳验证 MAE：`0.2315782904624939`
- best checkpoint：
  - `outputs/smoke/crossformer/96_48_S/FINE_TUNE/ver_0/checkpoints/best_epoch=0_step=266_MAE=0.232.ckpt`
- 评估结果：
  - `outputs/smoke/crossformer/96_48_S/EVALUATE/ver_0/eval_results_CrossFormer.json`

三个模型在 8 个 smoke item 上的评估均值如下：

| 模型 | item 数 | mean MSE | mean MAE |
| --- | ---: | ---: | ---: |
| DLinear | 8 | 0.3044409591 | 0.2156414958 |
| PatchTST | 8 | 0.3091445394 | 0.2279613437 |
| CrossFormer | 8 | 0.3846419056 | 0.2756668682 |

## 结论

三个目标模型在 `96_48_S` smoke 子集上均已跑通完整链路：

1. 4 卡 DDP 训练可用。
2. 验证集评估可用。
3. best checkpoint 保存可用。
4. Ray evaluator 加载 checkpoint 可用。
5. 小规模 test split 评估可用。
6. 结果 JSON 输出可用。

这些 smoke 指标只用于确认流程，不代表正式模型效果。由于只训练 1 个 epoch 且只使用 8 个 item，不能用于和论文结果对比。

## 下一步方案

1. 将 smoke 流程抽象成可复用脚本，减少手动创建 evaluate 配置和收集 checkpoint 的工作。
2. 决定正式实验采用快速配置还是论文式配置：
   - 快速配置：沿用当前 5 epoch 配置，先跑单 seed。
   - 论文式配置：调参后训练 100 epoch，并跑 3 个 seed。
3. 建议下一步先做 `96_48_S` 全量单 seed 试跑，优先从 DLinear 开始，再扩展到 PatchTST 和 CrossFormer。
