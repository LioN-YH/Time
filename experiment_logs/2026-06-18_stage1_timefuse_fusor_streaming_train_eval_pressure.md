# Stage 1 TimeFuse-style Fusor Streaming 训练/Eval 入口与压力测试

日志日期：2026-06-18 00:40:16 CST

## 目的

基于已实现的 `stage1_timefuse_fusor_streaming_reader.py`，完成 Stage 1 `96_48_S` full-scale TimeFuse-style fusor 的 streaming 训练与 test streaming eval 入口，并只用 1-2 个 feature/prediction shard 做闭环与压力测试，不启动正式 64-shard 全量训练。

## 背景

已有 pilot 版 TimeFuse-style fusor 复刻了原生 `nn.Linear -> softmax -> weighted fusion -> SmoothL1Loss` 口径，但旧入口会把 feature/label/prediction 读成内存 DataFrame 或 lookup，不适合直接承载 full-scale `116,375,850` 行 prediction manifest。前一步已经实现并 smoke 通过 `stage1_timefuse_fusor_streaming_reader.py`，该 reader 能按 feature shard 建 oracle/prediction SQLite 索引并按 batch 读取 packed 五专家预测数组。

本次工作继续沿用该 reader，重点补齐训练/eval 入口、checkpoint、状态文件、summary、sample predictions 和资源记录。

## 操作

1. 新增正式入口脚本：
   - 路径：`visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
   - 口径：单层 `TimeFuseFusor(input_dim=17, output_dim=5)`，forward 为 `Linear -> softmax`。
   - 训练：`StandardScaler` 只在 vali feature streaming 上 `partial_fit`；训练阶段按 vali batch 读取 packed 五专家 `y_pred/y_true`，用 weighted fusion 输出和 `y_true` 的 `SmoothL1Loss(beta=0.01)` 反传。
   - eval：test split 只 transform/eval，不 fit scaler；输出 hard top-1 与 raw soft fusion 的数组级 MAE/MSE。
   - 安全限制：默认最多 2 个 feature shard，避免误启动 64-shard 全量训练；GPU 模式只允许 `CUDA_VISIBLE_DEVICES=2,3` 或其单卡子集。
   - smoke/压力测试辅助：增加 `--max-rows-per-split-per-shard`，为每个输入 shard 生成只含少量 vali/test 行的 feature subset，避免 full-scale shard 开头全是 test 导致训练 smoke 没有 vali。

2. 修复两处 metadata 输出问题：
   - 首次 smoke 已完成 index 构建、训练/eval，但写 `metadata.json` 时发现 `Path` 不能直接 JSON 序列化。
   - 增加 `to_jsonable()`，把 `Path`、numpy 标量和 numpy 数组转换为稳定 JSON 表示。
   - 第二次 smoke 的文件输出已完成，但 stdout 打印 metadata 时仍使用未清洗对象；改为 `json.dumps(to_jsonable(metadata))`。
   - 复查 eval-only metadata 时，发现 `checkpoint_path` 会指向当前 eval-only 输出目录下并不存在的新 checkpoint；已改为 eval-only 时记录实际加载的 `--resume-checkpoint`。
   - 复查同名输出目录重跑语义时，发现 feature subset 会追加写入；已在重建 subset 前删除旧 subset 文件，避免重复 sample_key。
   - 删除了本次由我创建的半成品目录 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000/` 后重新复跑，避免误读。

3. 运行语法检查：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py
   ```

4. 运行 1-shard smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py \
     --feature-shard-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0000_of_0064/feature_cache.csv \
     --output-dir /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000 \
     --device cpu \
     --epochs 1 \
     --batch-size 8 \
     --max-rows-per-split-per-shard 16 \
     --prediction-num-workers 2 \
     --status-update-interval 1 \
     --sample-prediction-limit 20
   ```

5. 用 smoke checkpoint 做 eval-only 复验：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py \
     --feature-shard-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0000_of_0064/feature_cache.csv \
     --output-dir /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000_eval_only \
     --device cpu \
     --epochs 1 \
     --batch-size 8 \
     --max-rows-per-split-per-shard 16 \
     --prediction-num-workers 2 \
     --status-update-interval 1 \
     --sample-prediction-limit 20 \
     --resume-checkpoint /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000/checkpoints/latest_timefuse_fusor.pt \
     --eval-only
   ```

6. 运行 2-shard 压力测试：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py \
     --feature-shard-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0000_of_0064/feature_cache.csv \
     --feature-shard-path /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/timefuse_feature_cache_full_scale_launcher/shards/sample_shard_0001_of_0064/feature_cache.csv \
     --output-dir /data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_pressure_0000_0001_64ps \
     --device cpu \
     --epochs 1 \
     --batch-size 32 \
     --max-rows-per-split-per-shard 64 \
     --prediction-num-workers 2 \
     --status-update-interval 2 \
     --sample-prediction-limit 50
   ```

7. 抽查压力测试输出：
   - `summary.md`
   - `timefuse_fusor_summary.csv`
   - `timefuse_fusor_raw_soft_fusion_summary.csv`
   - `timefuse_fusor_selected_model_counts.csv`
   - `sample_predictions.csv`
   - `checkpoints/latest_timefuse_fusor.pt`

8. 同步更新：
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`

## 结果

1. 代码与语法检查：
   - `train_timefuse_fusor_streaming.py` 已新增，Quito 环境 `py_compile` 通过。
   - 入口复用 `Stage1TimeFuseFusorStreamingReader`，没有新增全量 manifest lookup、全量 DataFrame join 或全量 prediction 常驻内存逻辑。

2. 1-shard smoke：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000/`
   - feature shard：`sample_shard_0000_of_0064`
   - 子集规模：vali 16、test 16，共 32 个 sample_key。
   - oracle SQLite：32 条记录。
   - prediction SQLite：160 条记录，即 32 个 sample_key × 5 专家。
   - scaler：2 个 vali batch，16 个 vali sample。
   - 训练：1 epoch，2 个 train batch，16 个 train sample，mean loss `0.093338`。
   - eval：16 个 test sample。
   - checkpoint：`checkpoints/latest_timefuse_fusor.pt` 已写出。
   - 资源：CPU 运行，未设置 `CUDA_VISIBLE_DEVICES`；torch CUDA memory allocated/reserved 均为 0。

3. checkpoint eval-only 复验：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000_eval_only/`
   - 成功加载 `latest_timefuse_fusor.pt`，重新构建 shard-local SQLite 后完成 test streaming eval。
   - eval-only 输出 16 个 test sample，证明 checkpoint 中 fusor state 与 scaler state 可加载复验。
   - `metadata.json` 中 `checkpoint_path` 指向实际加载的 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_smoke_0000/checkpoints/latest_timefuse_fusor.pt`。

4. 2-shard 压力测试：
   - 输出目录：`/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_streaming_pressure_0000_0001_64ps/`
   - feature shards：`sample_shard_0000_of_0064`、`sample_shard_0001_of_0064`
   - 子集规模：每个 shard vali 64、test 64；合计 256 个 sample_key。
   - oracle SQLite：每个 shard 128 条记录。
   - prediction SQLite：每个 shard 640 条记录，即 128 个 sample_key × 5 专家。
   - scaler：4 个 vali batch，128 个 vali sample。
   - 训练：1 epoch，4 个 train batch，128 个 train sample，mean loss `0.075633`，last loss `0.067081`。
   - eval：128 个 test sample。
   - hard top-1 summary：test MAE/selected_value `0.079428`，oracle_value `0.061342`，regret `0.018086`，oracle_label_accuracy `0.125000`。
   - raw soft fusion summary：soft_fusion_mae `0.080372`，soft_fusion_mse `0.032963`，hard_top1_mae_from_array `0.079428`。
   - selected model counts：DLinear 14、PatchTST 27、CrossFormer 18、ES 0、NaiveForecaster 69。
   - checkpoint 抽查：`checkpoint_version=stage1_timefuse_fusor_streaming_v1`，`completed_epoch=1`，`completed_shards=['sample_shard_0000_of_0064','sample_shard_0001_of_0064']`，包含 `fusor_state_dict`、`scaler_state` 和 `train_args`。
   - 资源：CPU 运行；最终 RSS 约 `951.33 MB`，进程 I/O 读约 `923.52 MB`、写约 `1.06 MB`；torch CUDA allocated/reserved 为 0；`nvidia-smi` 快照显示本任务未占用 GPU。

5. 输出文件：
   - `metadata.json`
   - `status.json`
   - `summary.md`
   - `main.log`
   - `timefuse_fusor_predictions.csv`
   - `timefuse_fusor_summary.csv`
   - `timefuse_fusor_raw_soft_fusion_summary.csv`
   - `timefuse_fusor_selected_model_counts.csv`
   - `sample_predictions.csv`
   - `checkpoints/timefuse_fusor_epoch_0001.pt`
   - `checkpoints/latest_timefuse_fusor.pt`
   - `checkpoints/latest_checkpoint_index.json`
   - `indexes/*/oracle_labels_index.sqlite`
   - `indexes/*/prediction_manifest_index.sqlite`
   - `feature_subsets/*/feature_cache.csv`

## 结论

Stage 1 `96_48_S` full-scale TimeFuse-style fusor 的 streaming 训练与 test streaming eval 入口已完成，并在 1-shard smoke、checkpoint eval-only 复验和 2-shard 小切片压力测试中闭环通过。当前实现保持 reader 的内存口径：feature 逐 batch 流式读取，oracle/prediction 只为当前 1-2 个 feature shard 的 sample_key 建 SQLite，训练/eval 阶段按 batch 读取 packed 五专家 `y_pred/y_true`，不随 `116,375,850` 行 manifest 线性增长。

本次没有启动正式 64-shard 全量训练 launcher，也没有使用 GPU。

## 下一步方案

1. 如需更强压力测试，可在同一入口上把 `--max-rows-per-split-per-shard` 提高到更大数值，仍限制 `--max-feature-shards 2`，先观察 SQLite 构建时间、RSS 和 I/O。
2. 若决定跑正式 TimeFuse fusor baseline，应另写后台 launcher，并明确 PID、status、停止命令和 shard 恢复策略；启动前再次确认不会与 visual router GPU 任务抢资源。
3. 可把 `timefuse_fusor_summary.csv` 与后续 visual router eval/calibration summary 放入统一 comparison 表。
