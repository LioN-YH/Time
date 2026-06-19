# Stage 1 TimeFuse Fusor Streaming Reader 实现与 Smoke Test

日志日期：2026-06-18 00:27:39 CST

## 目的

设计并实现 Stage 1 `96_48_S` full-scale TimeFuse-style fusor 的 streaming / shard-aware 数据读取层，冻结 reader 输入输出契约，并完成 1-shard 少量 batch smoke test。本轮只验证数据层，不启动正式训练。

## 背景

现有 `fusion_utils.py` 中的 TimeFuse-style fusor pilot 依赖全量 feature/label join 和全量 prediction lookup，不能直接承载 full-scale `116,375,850` 行 prediction manifest。此前 streaming visual router 已因全量 Python lookup 触发 OOM，后续修复为 SQLite 磁盘索引 + batch 查询。本轮 reader 复用该经验，将 oracle parquet、prediction manifest 和 packed arrays 都限制在 shard-local 与 batch-local 读取口径。

## 操作

1. 阅读并复核了 `train_visual_router_online_streaming.py` 的 SQLite OOM 修复、`prediction_array_io.py` 的 `packed_npy_v1` 读取接口、`fusion_utils.py` 的 TimeFuse fusor 旧口径，以及 full-scale feature shard / oracle / prediction cache 路径。
2. 新增 `visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py`：
   - 从 feature shard 按 batch 读取 17 维 `timefuse_single_variable_meta_v1` 特征；
   - 从 oracle parquet 为当前 shard/sample 子集构建 SQLite index；
   - 从五专家同编号 prediction shard manifest 或单个 merged manifest 构建 SQLite index；
   - 按 batch 查询 label 和五专家 record；
   - 通过 `load_prediction_array()` 按 packed row index 读取单行 `y_pred/y_true`；
   - 支持当前 batch 内多线程读取 prediction arrays，以及至多一个 batch 的安全预取。
3. 首次 smoke 时发现预取线程复用 SQLite 连接触发 `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread`。
4. 修复 SQLite index 类：连接增加 `check_same_thread=False`，并用 `threading.Lock` 串行化连接级查询，保证预取线程不会并发破坏 SQLite 访问顺序。
5. 新增设计文档 `visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader_design.md`。
6. 更新 `visual_router_experiments/stage1_vali_test_router/README.md` 和 `WORKSPACE_STRUCTURE.md`，登记新增 reader 和设计文档。
7. 使用 Quito conda 环境执行语法检查和真实 shard smoke test。

## 结果

语法检查通过：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py
```

真实 shard smoke 命令：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py --max-rows 16 --batch-size 8 --smoke-batches 2 --prediction-num-workers 2 --prefetch-batches 1
```

有效输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-18_002728_217172_stage1_timefuse_fusor_streaming_reader_smoke/
```

关键验证结果：

- oracle SQLite index 写入 16 条 `metric=mae` label；
- prediction SQLite index 写入 80 条 record，即 16 个 sample_key × 5 专家；
- batch 1 和 batch 2 均成功输出：
  - `feature_shape=[8,17]`
  - `y_pred_shape=[8,5,48,1]`
  - `y_true_shape=[8,48,1]`
  - `expert_errors_shape=[8,5]`
- DLinear 数组复算 MAE 与 reader `expert_errors` 的最大差异为 `0.0`；
- 本轮未启动正式训练，未使用 GPU。

## 结论

Stage 1 full-scale TimeFuse-style fusor 的数据读取层已具备基本可用契约：能在真实 full-scale shard 上按 batch 读取 feature、oracle label、五专家 `y_pred/y_true` 和专家误差，且不依赖全量 manifest lookup 或全量 DataFrame join。并行策略限制在当前 batch 内 prediction array 读取和单 batch 预取，内存峰值不随 full manifest 行数线性增长。

## 下一步方案

1. 基于该 reader 实现 streaming StandardScaler 和单层 TimeFuse fusor 训练循环。
2. 先做单 shard train/eval smoke，验证 loss、checkpoint、status 和输出格式。
3. 再设计 64 shard launcher；正式训练若使用 GPU，严格限制 `CUDA_VISIBLE_DEVICES=2,3`。
