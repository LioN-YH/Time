# QuitoBench cluster 信息整理实验日志

## 日志信息

- 日志日期：2026-06-10 02:23:38 CST
- 工作目录：`/home/shiyuhong/Time`
- 相关项目：`/home/shiyuhong/Time/quito`
- 相关数据集：Hugging Face `hq-bench/quitobench`
- 相关 revision：
  - 当前本地数据：`ed4bf8eea27862aa01a063393ee48732910ecd87`
  - cluster 来源旧版本：`17362dcb`

## 目的

确认 QuitoBench 数据中的 `cluster` 信息来源、语义和可用形式，解决当前下载的 `v20260315` parquet 文件没有 `cluster` 列的问题，并为后续实验准备可直接使用的 cluster 标签。

## 背景

本地已经下载了 QuitoBench 当前版本数据：

- `quito/data/hf/hq-bench/quitobench/v20260315/test_hour-00001-of-00001.parquet`
- `quito/data/hf/hq-bench/quitobench/v20260315/test_min-00001-of-00001.parquet`

检查发现当前版本 parquet 只包含：

- `date_time`
- `ind_1` 到 `ind_5`
- `item_id`

当前版本不包含 `cluster` 列。

之前实验日志指出旧 revision `17362dcb` 的相同 parquet 文件仍保留 `cluster` 列，可以作为官方 cluster 标签来源。

## 操作

1. 检查本地 QuitoBench 当前版本 parquet schema。
2. 使用 `quito` conda 环境中的 Hugging Face 工具访问 `hq-bench/quitobench`。
3. 下载旧 revision `17362dcb` 中的两个 parquet 文件到独立目录：
   - `quito/data/hf/hq-bench/quitobench/cluster_source_17362dcb/v20260315/test_hour-00001-of-00001.parquet`
   - `quito/data/hf/hq-bench/quitobench/cluster_source_17362dcb/v20260315/test_min-00001-of-00001.parquet`
4. 从旧 revision parquet 中抽取 `item_id -> cluster` 映射。
5. 将抽取结果和本地 `quito/examples/item_csv.csv` 中的 cluster 信息进行一致性校验。
6. 基于 `item_csv.csv` 给当前版本 parquet 生成带 `cluster` 列的副本。
7. 将带 `cluster` 列的 parquet 通过软链接接到 Quito 默认配置期望的数据目录。

## 结果

当前版本 parquet 确认无 `cluster` 列；旧 revision `17362dcb` 的 parquet 确认包含 `cluster:int64`。

抽取出的 cluster 标签统计如下：

| 频率 | item 数 | cluster |
| --- | ---: | --- |
| hour | 517 | 0, 2, 8, 18, 20 |
| min | 773 | 2, 6, 8, 18, 20, 24, 26 |
| 合计 | 1290 | 0, 2, 6, 8, 18, 20, 24, 26 |

校验结果：

- `item_csv.csv` 共 1290 个唯一 `item_id`。
- 旧 revision parquet 抽取结果共 1290 个唯一 `item_id`。
- 两者 `item_id -> cluster` 完全一致。
- 每个 `item_id` 只有一个 cluster。
- 没有发现 cluster 冲突。

生成的主要文件：

- `quito/data_audit/quitobench_clusters/hour_item_cluster.csv`
- `quito/data_audit/quitobench_clusters/min_item_cluster.csv`
- `quito/data_audit/quitobench_clusters/all_item_cluster.csv`
- `quito/data_audit/quitobench_clusters/all_item_cluster_with_quality.csv`
- `quito/data_audit/quitobench_clusters/cluster_summary.csv`
- `quito/data_audit/quitobench_clusters/manifest.json`

生成的带 cluster 当前版本 parquet：

- `quito/data/hf/hq-bench/quitobench/v20260315_with_cluster_from_17362dcb/test_hour-00001-of-00001.parquet`
- `quito/data/hf/hq-bench/quitobench/v20260315_with_cluster_from_17362dcb/test_min-00001-of-00001.parquet`

Quito 默认数据目录已建立软链接：

- `quito/examples/datasets/cluster_data/open_hour_data.parquet`
- `quito/examples/datasets/cluster_data/open_min_data.parquet`
- `quito/examples/datasets/cluster_data/item_clusters.csv`

## cluster 语义

根据之前反推的 codebook，`cluster` 是三位三进制编码，三位分别对应：

1. trend
2. seasonality
3. forecastability

其中：

- `0` 表示 high
- `2` 表示 low

对应关系：

| cluster | 编码 | 语义 |
| ---: | --- | --- |
| 0 | 000 | high trend, high seasonality, high forecastability |
| 2 | 002 | high trend, high seasonality, low forecastability |
| 6 | 020 | high trend, low seasonality, high forecastability |
| 8 | 022 | high trend, low seasonality, low forecastability |
| 18 | 200 | low trend, high seasonality, high forecastability |
| 20 | 202 | low trend, high seasonality, low forecastability |
| 24 | 220 | low trend, low seasonality, high forecastability |
| 26 | 222 | low trend, low seasonality, low forecastability |

## 结论

本地 `quito/examples/item_csv.csv` 可以作为最完整、最方便的官方 cluster 信息源；旧 revision `17362dcb` 的 parquet 可作为可复核来源。当前版本 parquet 缺失的 `cluster` 列已经通过 `item_id` 映射补回到独立副本中，原始下载文件没有被覆盖。

后续实验如果需要按 cluster 过滤、分组或复现实验配置，可以直接使用：

- `quito/examples/datasets/cluster_data/item_clusters.csv`
- `quito/examples/datasets/cluster_data/open_hour_data.parquet`
- `quito/examples/datasets/cluster_data/open_min_data.parquet`

## 下一步方案

1. 明确下一阶段实验目标：例如按 cluster 分层评估、复现论文表格，或分析不同 regime 下模型表现。
2. 如果要跑 Quito 默认配置，先用一个轻量配置验证 `open_hour_data.parquet` 和 `open_min_data.parquet` 能被 `TimeSeriesDataset` 正常加载。
3. 如果要做按 cluster 的实验，优先基于 `item_clusters.csv` 做筛选，而不是重新读取旧 revision parquet。
4. 后续每完成一个独立实验步骤，都在 `experiment_logs/` 下新增一份日期加工作描述的中文日志，并同步更新总览追踪表。
