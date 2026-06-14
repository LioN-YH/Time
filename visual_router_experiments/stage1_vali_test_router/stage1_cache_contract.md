# Stage 1 Cache Contract

记录日期：2026-06-12 23:57:09 CST

## 目的

本文档固定 Stage 1 正式 prediction cache、oracle labels、feature cache 和 router evaluation 之间的字段契约。后续新增正式 cache builder、visual feature builder、router trainer 和 evaluator 时，必须遵守本文档，避免不同历史-未来 config、专家 checkpoint 或窗口粒度混用。

## 核心原则

1. **路由粒度是 item-channel-window**

   一个待路由样本由 `config_name + split + dataset_name + item_id + channel_id + window_index` 唯一标识。

2. **config_name 是动作空间边界**

   `config_name` 表示历史长度、预测长度和特征模式的组合，例如 `96_48_S`。同一个 router 或 baseline 只能在同一 `config_name` 内比较专家。

3. **专家动作空间只在同 config 内成立**

   对 `96_48_S` 样本，合法候选专家是：

   ```text
   DLinear_96_48_S
   PatchTST_96_48_S
   CrossFormer_96_48_S
   ES_96_48_S
   NaiveForecaster_96_48_S
   ```

   不允许选择 `576_288_S` 或 `1024_512_S` 的专家作为可部署预测动作。

4. **多 config 可以同目录保存，但必须分层训练和汇总**

   一个正式 cache 目录可以包含多个 `config_name`，但 oracle label、baseline、router 训练和结果汇总必须保留 `config_name` 分层。跨 config macro average 只能作为总览，不代表可部署动作空间。

## Prediction Cache Manifest

正式 manifest 文件建议命名为：

```text
manifest.csv
```

必需字段：

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `cache_version` | cache schema 版本 | 当前为 `visual_router_prediction_cache_v1` |
| `sample_key` | 窗口唯一 key | 必须等于 `config_name__split__dataset_name__item{item_id}__ch{channel_id}__win{window_index}` |
| `config_name` | 历史-未来-特征模式设置 | 例如 `96_48_S` |
| `split` | 数据 split | 只能是 `vali` 或 `test` |
| `dataset_name` | Quito dataset 配置名 | 例如 `TEST_DATA_MIN` |
| `item_id` | 原始 item id | 整数 |
| `channel_id` | 原始通道 id | 单变量 pilot 暂为 0 |
| `window_index` | 当前 item-channel 在 split 内的窗口序号 | 从 0 开始，`shuffle=False` 口径 |
| `history_length` | 输入历史长度 | 必须与 `config_name` 一致 |
| `pred_length` | 预测未来长度 | 必须与 `config_name` 一致 |
| `model_name` | 专家名称 | 五专家之一 |
| `expert_version` | 专家版本或来源 | 用于区分 checkpoint / 统计模型实现 |
| `checkpoint_selection` | checkpoint 选择口径 | 深度模型需说明 validation MAE/MSE best 等 |
| `y_true_path` | 真实未来数组路径 | 建议相对 cache 目录 |
| `y_pred_path` | 专家预测数组路径 | 建议相对 cache 目录 |
| `mae` | 当前窗口当前专家 MAE | 与 Quito evaluate 指标空间一致 |
| `mse` | 当前窗口当前专家 MSE | 与 Quito evaluate 指标空间一致 |
| `array_storage` | 数组存储模式 | 可选；缺省视为 `per_sample_npy`，全量推荐 `packed_npy_v1` |
| `y_true_row_index` | packed y_true 行号 | `array_storage=packed_npy_v1` 时必需 |
| `y_pred_row_index` | packed y_pred 行号 | `array_storage=packed_npy_v1` 时必需 |

唯一性约束：

- `sample_key + model_name` 必须唯一；
- 同一 `sample_key` 下，`config_name`、`split`、`dataset_name`、`item_id`、`channel_id`、`window_index`、`history_length` 和 `pred_length` 必须一致；
- 正式大规模 cache 推荐让同一 `sample_key` 共享一个 `y_true_path`，减少重复落盘；早期 pilot/legacy cache 若按专家重复保存 `y_true`，则必须保证其内容表示同一个真实未来窗口；
- `packed_npy_v1` 模式下，`y_true_path` / `y_pred_path` 指向包含多个窗口的 `.npy` 大数组，第一维为窗口维；读取单条记录时使用对应 row index。合并后的正式 cache 必须让同一 `sample_key` 共享同一个 `y_true_path + y_true_row_index`；
- `per_sample_npy` 只用于早期 smoke、历史 pilot 或很小样本调试；全量和 full-scale dry-run 默认使用 `packed_npy_v1`，避免 per-sample 小文件爆炸；
- 同一 `sample_key` 下，正式五专家 cache 应覆盖同一组专家。

## Full-Scale Shard / Resume 约定

- sample manifest 是所有 prediction cache shard、oracle labels、baseline、streaming router 和 calibration 的共同来源；
- prediction cache shard 目录应按 `model_name / sample_shard_xxxx` 或等价结构拆分，每个 shard 独立写 `manifest.csv`、`metadata.json`、`status.json`、`main.log` 和数组文件；
- DLinear、PatchTST、CrossFormer 优先用独立 GPU 进程并行；ES、NaiveForecaster 走 CPU 或独立进程；
- 失败重试时优先重跑失败 shard，不覆盖已完成 shard；merge 前必须确认所有目标 shard `status=completed`；
- merge 只接受完整专家集合，合并后必须重新校验 `sample_key + model_name` 唯一、五专家完整、共享 y_true 一致；
- oracle labels、TSF enrichment 和 baseline 默认只在 merged cache 上运行，避免 downstream 读取半成品 shard。

## Oracle Labels

正式 oracle labels 文件建议命名为：

```text
window_oracle_labels.csv
window_oracle_labels_with_tsf_cell.csv
```

生成约束：

- oracle label 必须在同一 `config_name`、同一 `sample_key` 的五专家内计算；
- `oracle_model` 是同 config 内误差最小的专家名称；
- `oracle_value` 是对应的窗口误差；
- 不允许跨 config 比较 MAE/MSE 并选择专家；
- MAE 与 MSE 是两个不同 label 口径，训练 router 前必须明确使用哪一个。

## TSF Cell Enrichment

TSF cell 字段来自 `item_id` 映射，不等同于 `dataset_name`。

推荐补充字段：

```text
cluster
group_name
forecastability_cat
season_strength_cat
trend_strength_cat
cv_cat
missing_ratio_cat
```

后续 summary 至少应支持：

- `config_name`
- `config_name + dataset_name`
- `config_name + group_name`
- `config_name + dataset_name + group_name`

## Feature / Embedding Cache

visual/structure feature cache 必须以 `sample_key` 对齐 prediction cache 和 oracle labels。注意：当前正式 online 主线不要求也不鼓励长期 ViT embedding cache；下面的 feature cache 契约主要用于历史离线 smoke、结构特征对照或明确声明的 ablation。

建议字段：

| 字段 | 含义 |
| --- | --- |
| `feature_version` | feature schema 或构造方法版本 |
| `sample_key` | 与 prediction cache 完全一致的窗口 key |
| `config_name` | 冗余保存，便于分层校验 |
| `split` | `vali` 或 `test` |
| `feature_type` | 例如 `numeric_structure`、`pseudo_image_tensor`、`vit_embedding` |
| `feature_path` | 外置特征数组路径 |
| `feature_dim` / `feature_shape` | 特征维度或形状 |

约束：

- feature cache 不保存专家预测结果；
- router 训练时通过 `sample_key` join feature cache 与 oracle labels；
- 如果 feature 构造使用历史窗口 `x`，不能读取未来 `y` 或专家误差信息；
- 同一 feature version 的 `sample_key` 必须唯一。
- 正式 online / streaming router 的 ViT embedding 与伪图像 tensor 只在 batch 运行时生成，不写 `.npy`，不作为长期缓存引用；可写 `online_embedding_manifest.csv` 和 latency summary 记录覆盖范围与口径，但不得包含长期 `embedding_path`。

## Baseline / Router Evaluation

baseline 和 visual router 的输出应遵守：

- 训练规则或模型时只使用 `vali`；
- 评估只使用 `test`；
- 所有可部署方法必须在同一 `config_name` 内选择专家；
- `oracle_top1` 只作为上限；
- summary 必须包含 `config_name`；
- macro average 只能作为总览，不能替代 per-config 主表。
- streaming online router 必须输出兼容 calibration 的 `visual_router_predictions.csv`，至少包含 `router_name`、`config_name`、`sample_key`、`split=test`、稳定元信息、`selected_model`、`oracle_model`、`oracle_value` 和五个 `weight_*` 字段；
- streaming online router 应写标准 `visual_router_metadata.json`，便于 `evaluate_soft_fusion_calibration.py` 自动定位 `labels_path` 与 `prediction_manifest_path`。

## 当前已落实的代码检查

`visual_router_experiments/common/prediction_cache_schema.py` 已提供：

- `PredictionCacheKey.as_string()`：统一生成 `sample_key`；
- `make_prediction_record()`：统一生成 manifest 记录；
- `validate_manifest_frame()`：校验必需字段、schema version、`sample_key` 与字段一致性、同一 sample 的稳定元信息一致性、`sample_key + model_name` 唯一性，以及可选专家集合完整性；如传入 `require_shared_y_true_path=True`，还会要求同一 `sample_key` 共享一个 `y_true_path`，在 packed 模式下同时共享 `y_true_row_index`。

`visual_router_experiments/common/prediction_array_io.py` 已提供：

- `load_prediction_array()`：统一读取 `per_sample_npy` 与 `packed_npy_v1` 两种 prediction cache 数组；
- `resolve_cache_array_path()`：按 manifest 所在目录解析相对数组路径。

`visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py` 已提供：

- 按 `config_name` 独立学习 baseline；
- config-level 主汇总；
- macro average 总览；
- dataset / TSF cell 分层时保留 `config_name`。
