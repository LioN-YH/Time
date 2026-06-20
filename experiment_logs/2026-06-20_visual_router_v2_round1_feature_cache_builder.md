# Visual Router V2 Round 1 feature cache builder

日志日期：2026-06-20 20:34:13 CST

## 目的

为 Visual Router V2 Round 1 RevIN aux 与 pooling 消融建立可复用的 sharded pilot feature cache。当前步骤只生成 pilot feature cache、manifest、metadata、status、cache size summary 和中文 summary，不训练 router/head/encoder，不启动 full-scale embedding cache。

## 背景

P0 已冻结 Visual Router V2 pilot sample sets，路径为 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/`。P1 Round 0 evaluator 已验证 P0 v1 `pilot_test` 能复现 full-scale 关键方向，路径为 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/`。Round 1 后续需要比较 `visual_cls_only`、`visual_mean_patch_only`、`visual_cls_mean_concat`、`revin_aux_only` 和 `best_visual_pooling_plus_revin_aux_concat`，因此本轮先复用 frozen ViT 前向结果，避免五个变体重复计算。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/visual_router_v2_features.py`，定义 `visual_router_v2_round1_feature_cache_v1` schema、6 维 RevIN aux 字段、P0 sample CSV 校验、feature finite/shape 校验、`.npz` shard 原子写出和已有 shard 校验逻辑。
2. 新增 `visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py`，从 Visual checkpoint 的 `embedding_metadata` 读取实际伪图像和 ViT 口径：`variant_a_3view`、`revin_aux`、`hf_vit_0_5`、`fixed_candidates`、候选周期 `[2,3,4,5,6,8,10,12,16,24,32,48,64,96]`。
3. builder 默认只处理 `pilot_train`、`pilot_selection`、`diagnostic_balanced`，默认禁止 `pilot_test`；如确需生成 final test feature，必须显式传入 `--include-pilot-test-final-test-only`，metadata 会标记 `final_test_only`。
4. builder 按 P0 CSV 的 `order_index` 顺序分 shard；每个 shard 内按 `split/dataset/item` 懒加载 Quito 历史窗口 `x`，只读取 `window_index : window_index + seq_len`，不访问 future `y`、oracle error 或 expert prediction。
5. 每个 shard 写出 `sample_key`、`order_index`、`cls_embedding`、`mean_patch_embedding` 和 `revin_aux`。其中 `mean_patch_embedding` 明确使用 `last_hidden_state[:, 1:, :].mean(dim=1)`，不包含 CLS token；`revin_aux` 字段为 `mean, log_std, min, max, range, clip_ratio`，仅由历史窗口 `x` 计算。
6. 使用 Quito conda 环境执行语法检查：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall \
     visual_router_experiments/stage1_vali_test_router/visual_router_v2_features.py \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py
   ```

7. 先运行 8 样本 dev smoke，发现并修复两个问题：`pandas.itertuples()` 会改写以下划线开头字段名，因此将 `_row_pos` 改为 `row_pos`；`pandas.DataFrame.to_markdown()` 依赖 quito 环境未安装的 `tabulate`，因此改为脚本内生成 Markdown 表格。
8. 运行任务要求的 128 样本 smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py \
     --max-samples-per-set 128 \
     --shard-size 64 \
     --embedding-batch-size 8 \
     --device cpu \
     --dtype fp32 \
     --local-files-only \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_smoke
   ```

9. 对同一 smoke 命令重复执行一次，不加 `--overwrite`，验证已有 shard 会先校验 sample_key、order_index、shape、dtype 和 finite，再 skip。第二次运行 `skipped_shards=6`，耗时约 0.81 秒。
10. 用独立 Python 检查 smoke 输出：三组 sample_set 均生成 feature；每个 shard 可独立读取；`cls_embedding=(N,768)`、`mean_patch_embedding=(N,768)`、`revin_aux=(N,6)`；所有 feature 为 float32 且 finite；shard 内 `sample_key/order_index` 与 P0 CSV 前 128 行完全一致；manifest 6 行、总样本 384；cache size summary 总大小约 1.636 MB。
11. 检查 GPU 状态后，GPU1/2/3 空闲，`/data2` 可用约 2.2T；正式 P2a 通过后台方式启动：

   ```bash
   cd /home/shiyuhong/Time-visual-router-v2
   export CUDA_VISIBLE_DEVICES=1,2,3
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_round1_features.py \
     --sample-sets pilot_train pilot_selection diagnostic_balanced \
     --shard-size 2000 \
     --embedding-batch-size 48 \
     --device cuda \
     --local-files-only \
     --vit-data-parallel \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features
   ```

   启动目录为 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/`，已写出 `command.sh`、`stop.sh`、`pid.txt`、`pgid.txt` 和 `main.log`。当前 PID/PGID 为 `2469769/2469769`，Python 子进程 PID 为 `2469772`。

## 结果

- 新增代码已通过 compileall。
- 128 样本 smoke 已完成，输出目录为 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features_smoke/`。
- smoke 产物包含 `status.json`、`round1_feature_metadata.json`、`round1_feature_manifest.csv`、`round1_feature_cache_size_summary.csv`、`round1_feature_summary.md`、`round1_feature_latency.csv` 和三个 sample_set 的 sharded `.npz` features。
- smoke 独立核验结果为 `status=passed`、`manifest_rows=6`、`total_samples=384`、`size_mb=1.6356325149536133`。
- resume/skip existing 机制已通过同参数复跑验证，第二次运行 `skipped_shards=6`。
- 正式 P2a 已后台启动。截至 2026-06-20 20:33 CST，`status.json` 显示 `status=running`、`current_sample_set=pilot_train`、`current_shard_id=1`、`processed_count=2000`、`completed_shards=1`，已写出 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/features/pilot_train/shard_00000.npz`，文件大小约 5.6 MB。

## 结论

Round 1 feature cache builder 的核心口径已经打通：按 P0 `order_index` 保序，从历史窗口 `x` 在线生成 pseudo image，冻结 ViT 同时输出 CLS 与 patch-token mean pooling，并写出 6 维 RevIN aux。smoke 证明 schema、shape、finite、P0 对齐、manifest、cache size summary 和 resume/skip 均满足要求。正式 P2a 已按后台长任务方式运行，目前仍在生成 pilot features，尚未完成最终验收。

## 下一步方案

1. 持续轻量监控正式 P2a：

   ```bash
   ps -p 2469769,2469772 -o pid,ppid,pgid,stat,etime,%cpu,%mem,rss,cmd
   cat /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/status.json
   find /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/features -maxdepth 3 -type f -name 'shard_*.npz' | wc -l
   tail -n 80 /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/main.log
   nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
   ```

2. 正式完成后核验三组 sample_count 是否分别为 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000`，总计 200000；核验 manifest 能按 sample_set/order_index 恢复 P0 顺序，所有 shard shape/finite/dtype 正确，`status.json` 为 `completed`。
3. 正式完成后追加更新本日志、`experiment_logs/README.md`、`WORKSPACE_STRUCTURE.md` 和输出目录内 `round1_feature_summary.md` 的最终结果。

## 追加查收记录

日志日期：2026-06-21 01:34:47 CST

### 目的

查收后台 GPU Feature Cache Builder 的正式 P2a 产物，确认是否满足 Round 1 pilot feature cache 的验收标准，并把实验日志、总览表和工作区结构文档从“进行中”更新为最终状态。

### 背景

上一阶段已完成 builder 实现、128 样本 smoke 和 resume/skip 验证，正式任务通过 `CUDA_VISIBLE_DEVICES=1,2,3`、`--device cuda`、`--vit-data-parallel` 后台运行。用户在新窗口说明 GPU 上的 Feature Cache Builder 已完成，因此本次以当前磁盘产物和状态文件为准重新核验。

### 操作

1. 检查正式输出目录 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/` 的后台进程、`status.json`、`main.log` 和 shard 文件数量。
2. 确认后台 PID `2469769/2469772` 已退出，`status.json` 显示 `status=completed`、`phase=done`、`processed_count=200000`、`completed_shards=100`、`failed_reason=null`。
3. 统计 `features/` 下 shard 数量：`pilot_train=75`、`pilot_selection=15`、`diagnostic_balanced=10`，合计 100 个 shard；没有生成 `features/pilot_test/`。
4. 使用 Quito conda 环境运行独立核验脚本，全量读取 `round1_feature_manifest.csv`、`round1_feature_cache_size_summary.csv` 和每个 `.npz` shard，逐项检查 sample count、shape、dtype、finite、order_index 连续性，以及全量 `sample_key/order_index` 与 P0 CSV 完全对齐。
5. 检查 `round1_feature_metadata.json` 中关键约束：`mean_patch_excludes_cls=true`、`read_prediction_manifest=false`、`train_router_or_encoder=false`、`default_excludes_pilot_test=true`。
6. 检查输出目录内 `round1_feature_summary.md`，确认已记录输入路径、输出结构、feature schema、sample counts、cache size、正式运行结果和“不是 full-scale embedding cache”的使用边界。

### 结果

- 正式 P2a 输出目录状态为 `completed`，总耗时约 `17881.26` 秒。
- `round1_feature_manifest.csv` 共 100 行，总样本数 200000；三组样本计数为 `pilot_train=150000`、`pilot_selection=30000`、`diagnostic_balanced=20000`。
- shard 分布为 `pilot_train=75`、`pilot_selection=15`、`diagnostic_balanced=10`，每个 shard 可独立读取。
- 所有 shard 均包含 `sample_key`、`order_index`、`cls_embedding`、`mean_patch_embedding`、`revin_aux`。
- `cls_embedding` shape 为 `(N, 768)`，`mean_patch_embedding` shape 为 `(N, 768)`，`revin_aux` shape 为 `(N, 6)`；三类 feature dtype 均为 `float32` 且全部 finite。
- 每个 shard 内 `order_index` 连续，按 manifest 的 `sample_set/start_order_index` 拼接后，`sample_key/order_index` 与 P0 对应 CSV 全量完全一致。
- `round1_feature_cache_size_summary.csv` 与 manifest 的文件大小合计一致，总缓存大小约 `546.102 MB`。
- `features/pilot_test/` 不存在，正式 P2a 未生成 final test feature。
- 独立核验脚本输出：

  ```text
  {'status': 'passed', 'checked': [('diagnostic_balanced', 20000), ('pilot_selection', 30000), ('pilot_train', 150000)], 'manifest_rows': 100, 'total_samples': 200000, 'size_mb': 546.1020946502686}
  ```

### 结论

Visual Router V2 Round 1 sharded pilot feature cache 已完成并通过全量验收。该 cache 符合 P2a 边界：只覆盖 P0 `pilot_train`、`pilot_selection`、`diagnostic_balanced`，不处理 `pilot_test`；只从历史窗口 `x` 生成 CLS、patch-token mean pooling 和 6 维 RevIN aux；不训练 router/head/encoder，不读取 prediction manifest，不保存 pseudo image tensor，也不是 full-scale embedding cache。

### 下一步方案

后续 Round 1 消融可从 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round1_features/round1_feature_manifest.csv` 读取 sharded feature cache，并在训练阶段按需组合 `visual_cls_only`、`visual_mean_patch_only`、`visual_cls_mean_concat`、`revin_aux_only` 和 `best_visual_pooling_plus_revin_aux_concat`。进行架构选择时仍应只使用 `pilot_train` 和 `pilot_selection`，保留 `pilot_test` 作为最终代表性验证集合。
