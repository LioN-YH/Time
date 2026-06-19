# Stage 1 P8c TimeFuse Evaluation Adapter Pressure Verification

## 目标

验证 P8b 在 `train_timefuse_fusor_streaming.py` 中新增的 `--verify-evaluation-adapter` 只做 evaluation 阶段内存旁路一致性校验，不改变正式输出。

本次验证使用单个 feature shard 和每 split 8 行的小样本配置，分别运行关闭和开启 `--verify-evaluation-adapter` 的 train+eval pressure run，并比较正式 CSV 输出是否漂移。

## 输入与边界

- 分支：`refactor/stage1-route-audit`
- feature shard：`/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0008_of_0064/feature_cache.csv`
- 输出目录：
  - verify 关闭：`experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_off/`
  - verify 开启：`experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_on/`
- 样本规模：`--max-rows-per-split-per-shard 8`，即 8 个 vali 样本和 8 个 test 样本。
- 设备：`--device cpu`

本次不启动 full-scale，不修改 reader、scaler、optimizer、loss、checkpoint、launcher 或 Visual Router 入口。

## Pressure 命令

关闭 verify：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py \
  --feature-shard-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0008_of_0064/feature_cache.csv \
  --output-dir /home/shiyuhong/Time/experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_off \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --max-feature-shards 1 \
  --max-rows-per-split-per-shard 8 \
  --feature-read-chunk-rows 50000 \
  --prediction-chunk-rows 50000 \
  --oracle-parquet-batch-rows 50000 \
  --prediction-num-workers 1 \
  --prefetch-batches 0 \
  --status-update-interval 1 \
  --sample-prediction-limit 8
```

开启 verify：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py \
  --feature-shard-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0008_of_0064/feature_cache.csv \
  --output-dir /home/shiyuhong/Time/experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_on \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --max-feature-shards 1 \
  --max-rows-per-split-per-shard 8 \
  --feature-read-chunk-rows 50000 \
  --prediction-chunk-rows 50000 \
  --oracle-parquet-batch-rows 50000 \
  --prediction-num-workers 1 \
  --prefetch-batches 0 \
  --status-update-interval 1 \
  --sample-prediction-limit 8 \
  --verify-evaluation-adapter
```

## 对比结果

两次运行均完成：

- `sample_key_count=16`
- `train_samples=8`
- `test_samples=8`
- `timefuse_fusor_predictions.csv` 行数均为 8
- `timefuse_fusor_selected_model_counts.csv` 行数均为 5

逐文件比较结果：

| 文件 | 是否都生成 | 字段顺序 | 行数 | 内容 |
| --- | --- | --- | --- | --- |
| `timefuse_fusor_predictions.csv` | 通过 | 一致 | 8 / 8 | 完全一致 |
| `timefuse_fusor_summary.csv` | 通过 | 一致 | 1 / 1 | 完全一致 |
| `timefuse_fusor_raw_soft_fusion_summary.csv` | 通过 | 一致 | 1 / 1 | 完全一致 |
| `timefuse_fusor_selected_model_counts.csv` | 通过 | 一致 | 5 / 5 | 完全一致 |
| `sample_predictions.csv` | 通过 | 一致 | 8 / 8 | 完全一致 |

重点字段比较：

- `sample_key` 顺序一致。
- `selected_model` 一致。
- `hard_top1_mae_from_array` / `hard_top1_mse_from_array` 一致。
- `soft_fusion_mae` / `soft_fusion_mse` 一致。
- selected counts 一致。

## 结论

`--verify-evaluation-adapter` 在该小规模 pressure 配置下没有造成正式输出漂移。P8b 的旁路校验只在 evaluation batch 内构造 `EvaluationInput` 并通过 `EvaluationInputAdapter.evaluate_input(...)` 复算，不改变 CSV 字段、字段顺序、行数、sample_key 顺序、关键指标或 selected counts。

后续如果继续扩大 pressure 范围，应仍使用显式 `--output-dir` 和 1-2 shard 小样本配置，避免误启动 full-scale。
