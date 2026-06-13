# Visual Router Phase 1 主实验协议设计澄清

日志日期：2026-06-12 01:38:06 CST

## 目的

记录 Visual Router Phase 1 主实验的关键协议澄清，尤其是两阶段实验顺序、伪图像张量缓存方式、item-channel-window 级自适应粒度，以及 prediction cache 的可扩展设计要求。

## 背景

前序 oracle 审计证明五专家池在 per-item MAE 口径下存在足够互补性，但该审计仍是 item 级上限分析。实际目标不是只为整个 item 选择一个专家，而是在每个 item 的每个 channel、每个滑动窗口上，根据当前历史窗口的结构进行自适应路由。

因此 Phase 1 主实验需要从 per-item oracle 审计推进到 window-level prediction cache 和 router 训练协议。

## 操作

1. 明确 Phase 1 主实验不是直接进入 held-out cell zero-shot，而是先做同分布 vali -> test 路由验证。
2. 明确视觉输入不应频繁转为 PNG 或依赖 CPU 绘图库，而应优先保持为伪图像 2D 堆叠张量。
3. 明确缓存对象优先是冻结 ViT/视觉 encoder 的 embedding，而不是原始图像文件。
4. 明确路由粒度为 item-channel-window，即通道独立读入当前窗口历史并输出路由决策。
5. 明确 prediction cache 需要从一开始按后续 hard routing、softmax fusion、专家扩展和指标扩展设计。

## 两阶段主实验顺序

### Stage 1：vali 训练 router，test 测试 router

目标是验证视觉路由本身是否有效。

协议：

- 冻结 DLinear、PatchTST、CrossFormer、ES、SNaive 五个专家。
- 为 vali/test 的每个 item-channel-window 保存五专家预测缓存。
- 在 vali 上基于真实值和专家预测计算 window-level oracle label 或 expert regret。
- router 只用 vali 的视觉张量/视觉 embedding 训练。
- router 在 test 上只看当前历史窗口的视觉输入，输出 hard expert 或 softmax 权重。
- test 指标通过 prediction cache 组合预测后计算，避免重新运行专家模型。

该阶段回答：在同分布 cell/item 上，视觉结构能否学到有效路由。

### Stage 2：held-out cell zero-shot generalization

目标是验证 router 是否学习到可泛化视觉结构特征，而不是只记住 cell 或 item 分布。

协议：

- 7 个 TSF cell 的 vali 窗口训练 router。
- 1 个 held-out TSF cell 的 test 窗口评估 router。
- 8 个 TSF cell 轮流作为 held-out cell。

该阶段应在 Stage 1 证明路由有效后进行。

## 伪图像与视觉缓存设计

视觉输入采用伪图像形式，但默认不保存为 PNG/JPEG。

推荐口径：

- 原始视觉输入：`C x H x W` 或 `H x W x C` 的 2D 堆叠张量。
- 生成方式：基于当前 item-channel-window 的历史序列构造多通道结构图，例如 raw history、归一化趋势、季节/周期视图、频域或自相关通道。
- 默认存储：优先保存 tensor 或视觉 encoder embedding。
- 仅在需要人工检查或报告展示时，将少量样本导出为图片文件。

这样可以避免：

- 大量 PNG 编码/解码开销。
- CPU 绘图库成为瓶颈。
- 频繁 CPU/GPU 数据交换。
- 图片文件数量过大导致 I/O 和 inode 压力。

## 路由粒度

当前主目标是 item-channel 上的窗口级自适应：

```text
sample_key = config_name + split + item_id + channel_id + window_index
```

每个样本读取当前窗口的历史序列，并输出该 item-channel-window 的路由结果。

这意味着：

- 同一个 item 的不同窗口可以选择不同专家。
- 同一个多变量 item 的不同 channel 可以选择不同专家。
- router label 和 regret 应在 window-level 计算，而不是只使用 item-level 平均 MAE。

如果当前 Quito 输出中的 `S` 配置是单变量口径，则 `channel_id` 可先固定为 `0`；后续扩展到 `M` 配置或多通道数据时保留该字段。

## Prediction Cache 可扩展设计

prediction cache 是冻结专家在每个 window 上的预测结果缓存。它不只是为了当前 hard router，也要支持后续 softmax fusion、专家池扩展和诊断分析。

建议每条记录至少包含：

| 字段 | 含义 |
| --- | --- |
| `cache_version` | 缓存格式版本，便于后续升级 |
| `config_name` | Quito 配置，例如 `96_48_S` |
| `split` | `vali` 或 `test` |
| `item_id` | item 标识 |
| `channel_id` | 通道标识，单变量时为 `0` |
| `window_index` | 当前 split 内窗口序号 |
| `window_start` | 可选，原序列中的窗口起点 |
| `history_length` | 输入历史长度 |
| `pred_length` | 预测长度 |
| `model_name` | 专家模型名 |
| `expert_version` | 专家 checkpoint 或统计模型版本 |
| `checkpoint_selection` | deep model 的 checkpoint 选择口径 |
| `y_true_path` 或 `y_true` | 真实未来序列，建议数组大时外置存储 |
| `y_pred_path` 或 `y_pred` | 专家预测序列，建议数组大时外置存储 |
| `MAE` | 当前 window 的专家 MAE |
| `MSE` | 当前 window 的专家 MSE |

推荐存储方式：

- 元信息用 Parquet/CSV。
- 大数组用 NumPy `.npy`、`.npz`、Zarr 或 memmap。
- 同一个 `sample_key` 下五专家预测应能快速对齐。
- 保留 `cache_version` 和 `expert_version`，避免后续新增专家或改 checkpoint 时混淆。

## 是否还需要小规模生成

仍然需要小规模生成，但生成对象不是 PNG 图像，而是 prediction cache 与视觉 embedding pipeline 的最小闭环。

建议先做小规模 pilot 的原因：

- window-level cache 的 key 对齐、数组形状、split 口径、专家输出维度很容易出错。
- softmax fusion 必须逐窗口组合专家预测，必须先确认 `y_true` 和五个 `y_pred` 可以严格对齐。
- item-channel-window 级样本数量会远大于 per-item 结果，直接全量生成风险较高。
- 先用少量 item/cell/window 验证 I/O、显存、磁盘占用和运行时间，再扩展全量更稳。

建议 pilot 范围：

- 先选 1 个配置：`96_48_S`。
- 先选 2 到 3 个 TSF cell。
- 每个 cell 选少量 item。
- 每个 item-channel 抽样少量 vali/test window。
- 跑通五专家预测缓存、oracle label 计算、伪图像 tensor 生成、ViT embedding 缓存和一个轻量 router 训练/测试闭环。

pilot 通过后，再生成全量 Stage 1 cache。

## 结果

Phase 1 主实验协议更新为：

1. 先做 Stage 1：vali -> test 的 window-level visual router 有效性验证。
2. 再做 Stage 2：held-out cell zero-shot generalization。
3. 视觉输入默认保持为伪图像张量，不批量保存 PNG。
4. 缓存优先保存冻结视觉 encoder embedding。
5. 路由粒度为 item-channel-window。
6. prediction cache 从一开始按 softmax fusion 和专家扩展设计。

## 结论

当前不应直接全量生成视觉图片，也不应直接进入 held-out cell 主实验。下一步应做小规模 prediction cache + 伪图像 tensor/embedding + router 训练测试的最小闭环，验证协议和数据结构正确后再全量展开。

## 下一步方案

1. 阅读 Quito evaluate 数据流，确定如何导出 vali/test window-level `y_true` 和 `y_pred`。
2. 设计 prediction cache 的目录结构和 metadata schema。
3. 实现一个小规模 pilot 脚本，先覆盖 `96_48_S` 的少量 item/cell/window。
4. 在 pilot 中同时验证 hard top-1 router 和 softmax fusion 所需的数据对齐。
