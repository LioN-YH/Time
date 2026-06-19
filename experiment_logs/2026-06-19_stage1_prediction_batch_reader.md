# Stage 1 共享 PredictionBatchReader 抽取

日志日期：2026-06-19 17:15:03 CST

## 目的

完成 Stage 1 重构路线图 P1：抽取共享 prediction batch reader，并用 golden smoke 验证与重构前手写组装逻辑等价。

## 背景

重构前 `tests/smoke/stage1_golden_smoke.py` 在脚本内手工读取 `manifest.csv`、按五专家顺序重排、读取 `packed_npy_v1` row index、校验共享 y_true 和复算 MAE/MSE。该逻辑后续需要被 Visual Router 与 TimeFuse-style fusor 复用，因此先抽成共享 reader，但本阶段不迁移正式训练入口。

## 操作

1. 在 `visual_router_experiments/common/prediction_array_io.py` 新增 `load_prediction_arrays_grouped()`，对同一 batch 的 packed npy 按路径分组读取，legacy per-sample npy 仍复用 `load_prediction_array()`。
2. 新增 `time_router/` 最小 package 骨架与 `time_router/io/prediction_cache_reader.py`。
3. 实现 `PredictionBatchReader` 和 `PredictionBatch`，支持 `fixture_root` 或 `manifest_path`、显式 sample_key 或 manifest 首次出现顺序、固定五专家 `model_columns`、共享 y_true 校验、row index 元数据和 manifest MAE/MSE 复算。
4. 修改 `tests/smoke/stage1_golden_smoke.py`，让 smoke 使用 `PredictionBatchReader` 组装 `y_pred/y_true`，保留 sample_key 顺序、五专家顺序、shape、hard top-1、raw soft fusion 和 row index golden 检查。
5. 新增 `docs/refactor/prediction_batch_reader.md`，更新 `docs/refactor/stage1_refactor_roadmap.md`。

## 结果

已运行并通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

关键输出：

- `y_pred shape=(4, 5, 48, 1)`；
- `y_true shape=(4, 48, 1)`；
- hard top-1 选择为 `['CrossFormer', 'DLinear', 'PatchTST', 'DLinear']`；
- hard top-1 MAE=`0.416048437`，MSE=`0.456369758`；
- raw soft fusion MAE=`0.410296679`，MSE=`0.488154024`。

已运行并通过：

```bash
python -m compileall time_router tests/smoke
```

## 结论

P1 的共享 reader 抽取与 golden smoke 接入已完成。当前改动没有迁移正式 Visual Router / TimeFuse fusor 训练入口，也没有改变 prediction cache schema、sample_key、专家顺序、模型结构、loss 或正式输出目录。

## 下一步方案

后续 P2 可继续抽取 oracle/TSF reader。正式入口迁移应等到 P6，再按小规模 fixture、单 shard smoke、逐样本 comparison 和 golden smoke 前后门禁逐步替换。
