# ES / SNaive 统计基线 evaluate 启动与完成复盘

日志日期：2026-06-10 22:38:37 CST

改写日期：2026-06-11 22:56:14 CST

## 目的

在 `96_48_S`、`576_288_S`、`1024_512_S` 三个配置上补齐 ES 与 SNaive 的 test evaluate 结果，为后续把 DLinear、PatchTST、CrossFormer、ES、SNaive 放入同一张结果表做整体 mean MAE 和 TSF cell / cluster 对比做准备。

## 背景

ES/SNaive 是统计基线，不需要 finetune。官方 evaluate 配置中的 `resume.checkpoint_path: [null]` 只是为了满足 `evaluate.py` 的 checkpoint 列表接口，模型本身不加载训练权重。

本轮统计基线输出与深度学习 baseline 隔离，写入：

```text
quito/outputs/statistical_baseline/{es,snaive}/{config}/seed_16/EVALUATE/ver_0/
```

## 操作

新增并使用统计基线 evaluate 编排脚本：

```text
experiment_scripts/run_statistical_baseline_evaluate.py
```

脚本职责：

1. 从官方 evaluate 配置生成临时 YAML，不直接修改官方配置。
2. 将 evaluate 输出写到 `quito/outputs/statistical_baseline/`。
3. 对 evaluate JSON 生成 per-item、cluster 指标和 summary JSON。
4. 写出 `statistical_baseline_summary.csv` 与 `status.json`，便于追踪任务状态。

先做过语法检查和 dry-run：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  -m py_compile \
  experiment_scripts/run_statistical_baseline_evaluate.py

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  experiment_scripts/run_statistical_baseline_evaluate.py \
  --dry-run \
  --num-processes 8 \
  --use-gpu 0
```

正式执行分三段完成：

| 阶段 | run 目录 | 范围 | 资源参数 | 结果 |
| --- | --- | --- | --- | --- |
| 第一段 | `experiment_logs/run_outputs/2026-06-10_175742_314573_statistical_baseline_evaluate/` | `es:96_48_S` 先运行 | `--num-processes 8 --use-gpu 0 --eval-batch-size 128 --num-workers 2` | ES `96_48_S` 于 2026-06-10 23:09:41 CST 完成；原 `status.json` 曾停在 `running`，后续已按实际输出修正 |
| 第二段 | `experiment_logs/run_outputs/2026-06-11_010553_665102_statistical_baseline_evaluate/` | `snaive:96_48_S,snaive:576_288_S,snaive:1024_512_S` | `--num-processes 32 --use-gpu 0 --eval-batch-size 512 --num-workers 0` | SNaive 三组于 2026-06-11 01:12:51 CST 全部完成 |
| 第三段 | `experiment_logs/run_outputs/2026-06-11_011413_215790_statistical_baseline_evaluate/` | `es:576_288_S,es:1024_512_S` | `--num-processes 32 --use-gpu 0 --eval-batch-size 512 --num-workers 0` | ES `576_288_S` 于 2026-06-11 02:49:59 CST 完成；ES `1024_512_S` 于 2026-06-11 04:21:34 CST 完成 |

## 完成状态

截至 2026-06-11 22:56:14 CST，ES 与 SNaive 三组配置均已完成 evaluate，且输出目录中均存在 `eval_results_*.json`：

| 模型 | 配置 | evaluate JSON | 分析 summary |
| --- | --- | --- | --- |
| ES | `96_48_S` | `quito/outputs/statistical_baseline/es/96_48_S/seed_16/EVALUATE/ver_0/eval_results_ES.json` | `experiment_logs/run_outputs/2026-06-10_175742_314573_statistical_baseline_evaluate/cluster_analysis/es/96_48_S/seed_16/summary.json` |
| ES | `576_288_S` | `quito/outputs/statistical_baseline/es/576_288_S/seed_16/EVALUATE/ver_0/eval_results_ES.json` | `experiment_logs/run_outputs/2026-06-11_011413_215790_statistical_baseline_evaluate/cluster_analysis/es/576_288_S/seed_16/summary.json` |
| ES | `1024_512_S` | `quito/outputs/statistical_baseline/es/1024_512_S/seed_16/EVALUATE/ver_0/eval_results_ES.json` | `experiment_logs/run_outputs/2026-06-11_011413_215790_statistical_baseline_evaluate/cluster_analysis/es/1024_512_S/seed_16/summary.json` |
| SNaive | `96_48_S` | `quito/outputs/statistical_baseline/snaive/96_48_S/seed_16/EVALUATE/ver_0/eval_results_NaiveForecaster.json` | `experiment_logs/run_outputs/2026-06-11_010553_665102_statistical_baseline_evaluate/cluster_analysis/snaive/96_48_S/seed_16/summary.json` |
| SNaive | `576_288_S` | `quito/outputs/statistical_baseline/snaive/576_288_S/seed_16/EVALUATE/ver_0/eval_results_NaiveForecaster.json` | `experiment_logs/run_outputs/2026-06-11_010553_665102_statistical_baseline_evaluate/cluster_analysis/snaive/576_288_S/seed_16/summary.json` |
| SNaive | `1024_512_S` | `quito/outputs/statistical_baseline/snaive/1024_512_S/seed_16/EVALUATE/ver_0/eval_results_NaiveForecaster.json` | `experiment_logs/run_outputs/2026-06-11_010553_665102_statistical_baseline_evaluate/cluster_analysis/snaive/1024_512_S/seed_16/summary.json` |

## 结果

下表为 `summary.json` 中的 `metrics_mean_over_items`，即对 1290 个 item/user 的指标做均值。

| 模型 | 配置 | MSE | MAE | MASE | SMAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| ES | `96_48_S` | 115.188259 | 0.629922 | 696.432702 | 77.683609 |
| ES | `576_288_S` | 98.199574 | 0.712879 | 14.124256 | 90.745136 |
| ES | `1024_512_S` | 112.661086 | 0.743971 | 10.228390 | 97.906280 |
| SNaive | `96_48_S` | 153.007701 | 0.615799 | 17.711990 | 67.935006 |
| SNaive | `576_288_S` | 145.601073 | 0.689365 | 14.112895 | 77.484400 |
| SNaive | `1024_512_S` | 154.877549 | 0.719760 | 10.206011 | 84.586307 |

补充说明：

- ES `96_48_S` 原始 run 的 `status.json` 曾停在 `running`，但 evaluate 日志显示 `Progress: 1290/1290 evaluations complete`，且 `eval_results_ES.json` 于 2026-06-10 23:09:40 CST 写出、`log.txt` 于 2026-06-10 23:09:41 CST 写出。
- 已补生成 ES `96_48_S` 的 `per_item_results.csv`、`cluster_metrics.csv` 和 `summary.json`，并将该 run 的 `status.json` 修正为 ES `96_48_S` completed；其余 5 个任务标记为已由后续拆分 run 完成。
- ES/SNaive evaluate 均为 CPU 任务，`--use-gpu 0`，不会占用深度学习训练使用的 GPU。

## 结论

统计基线 ES 与 SNaive 在三组配置下均已完成，可以进入五模型三配置统一汇总阶段。后续汇总时，统计基线可直接引用上述 `eval_results_*.json` 和对应 `cluster_analysis/*/summary.json`、`cluster_metrics.csv`。

## 下一步方案

1. 汇总 DLinear、PatchTST、CrossFormer、ES、SNaive 在 `96_48_S`、`576_288_S`、`1024_512_S` 下的 overall mean MAE。
2. 读取各模型的 `cluster_metrics.csv` 或重新生成统一 TSF cell 表，整理分 TSF cell 的 MAE。
3. 在最终表格中明确 checkpoint 口径：`quito/outputs/default_baseline/` 下 DLinear、PatchTST、CrossFormer 的 evaluate 使用 validation MAE-best checkpoint；`quito/outputs/default_baseline_mse_best/` 是后补 validation MSE-best 复盘口径，目前只补了 `PatchTST 576_288_S`。
