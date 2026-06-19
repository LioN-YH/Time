# Stage 1 P8c TimeFuse evaluation adapter pressure 验证

日志日期：2026-06-20 01:33:57 CST

## 目的

对 P8b 新增的 `--verify-evaluation-adapter` 做小规模 pressure 验证，确认开启旁路一致性校验不会改变 `train_timefuse_fusor_streaming.py` 的正式 CSV 输出。

## 背景

P8b 已在 `evaluate_streaming(...)` 中接入默认关闭的 `EvaluationInputAdapter` 内存旁路校验。P8c 需要用同一份小规模输入分别运行关闭和开启 verify flag 的 pressure run，并比较 CSV 是否生成、字段顺序、行数、`sample_key` 顺序、`selected_model`、hard MAE/MSE、raw soft MAE/MSE 和 selected counts 是否一致。本步骤只验证 P8b 旁路行为，不启动 full-scale，不修改 reader、scaler、optimizer、loss、checkpoint、launcher 或 Visual Router 入口。

## 操作

1. 读取目标文件 `/home/shiyuhong/.codex-tianyu/attachments/35eae658-c1a3-4c5b-90fc-15dfe25c012e/pasted-text-1.txt`，确认 P8c 目标、边界和验收命令。
2. 检查当前分支为 `refactor/stage1-route-audit`，开始前工作树干净且与 `origin/refactor/stage1-route-audit` 同步。
3. 选用单个 feature shard：
   `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0008_of_0064/feature_cache.csv`
4. 运行关闭 verify 的 pressure 命令，显式输出到：
   `/home/shiyuhong/Time/experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_off`
5. 运行开启 verify 的 pressure 命令，显式输出到：
   `/home/shiyuhong/Time/experiment_logs/run_outputs/2026-06-20_stage1_p8c_timefuse_eval_adapter_pressure_verify_on`
6. 两次命令均使用：
   - `--device cpu`
   - `--epochs 1`
   - `--batch-size 4`
   - `--max-feature-shards 1`
   - `--max-rows-per-split-per-shard 8`
   - `--feature-read-chunk-rows 50000`
   - `--prediction-chunk-rows 50000`
   - `--oracle-parquet-batch-rows 50000`
   - `--prediction-num-workers 1`
   - `--prefetch-batches 0`
   - `--status-update-interval 1`
   - `--sample-prediction-limit 8`
7. 使用内联 Python 比较两次输出的 `timefuse_fusor_predictions.csv`、`timefuse_fusor_summary.csv`、`timefuse_fusor_raw_soft_fusion_summary.csv`、`timefuse_fusor_selected_model_counts.csv` 和 `sample_predictions.csv`。
8. 新增文档 `docs/refactor/timefuse_evaluation_adapter_pressure_verification.md`，记录 pressure 命令、输出目录和对比结果。

## 结果

1. 关闭 verify 的 pressure run 完成：
   - `sample_key_count=16`
   - `train_samples=8`
   - `test_samples=8`
   - `verify_evaluation_adapter=false`
2. 开启 verify 的 pressure run 完成：
   - `sample_key_count=16`
   - `train_samples=8`
   - `test_samples=8`
   - `verify_evaluation_adapter=true`
3. 对比结果：
   - 五个目标 CSV 均在两个输出目录中生成。
   - 五个目标 CSV 的字段顺序一致。
   - `timefuse_fusor_predictions.csv` 行数均为 8，`sample_predictions.csv` 行数均为 8。
   - `timefuse_fusor_summary.csv` 行数均为 1。
   - `timefuse_fusor_raw_soft_fusion_summary.csv` 行数均为 1。
   - `timefuse_fusor_selected_model_counts.csv` 行数均为 5。
   - `sample_key` 顺序一致。
   - `selected_model` 一致。
   - `hard_top1_mae_from_array`、`hard_top1_mse_from_array`、`soft_fusion_mae`、`soft_fusion_mse` 均一致。
   - selected counts 完全一致。
   - `pandas.testing.assert_frame_equal(..., atol=1e-8, rtol=1e-8)` 对五个 CSV 均通过。
4. 本次没有因校验失败修改 P8b 旁路逻辑，也没有修改正式输出 schema。

## 结论

P8b 的 `--verify-evaluation-adapter` 在 1-shard / 每 split 8 行的小规模 pressure 配置下不会导致正式输出漂移。开启 flag 后，`EvaluationInputAdapter` 内存复算通过，且正式 CSV、字段顺序、行数、样本顺序、hard/raw-soft 指标和 selected counts 与关闭 flag 的运行完全一致。

## 下一步方案

1. 运行目标 smoke 与 compileall 验收命令。
2. 小步提交 P8c 文档、日志和结构索引更新，并推送到远程 `refactor/stage1-route-audit` 分支。
3. 后续若扩大 pressure 验证，仍应使用显式 `--output-dir`、1-2 shard 和小样本限制，避免误启动 full-scale。
