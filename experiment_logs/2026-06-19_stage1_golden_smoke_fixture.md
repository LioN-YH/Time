# Stage 1 Golden Smoke Fixture

日志日期：2026-06-19 16:16:47 CST

## 目的

为 Stage 1 后续公共 batch reader、prediction cache reader 和 fusion metrics 重构建立一个极小 golden smoke，用当前已有输出锁定关键契约，避免重构时无意改变 sample 顺序、专家顺序、数组 shape、packed row index 或 hard/raw-soft 指标。

## 背景

`docs/refactor/stage1_migration_candidates.md` 建议第一步先为现有 batch array 读取建立小规模 golden fixture，再抽公共 reader。当前要求明确不移动、不删除、不重命名现有脚本，不改变正式 full-scale 路径和输出，因此本轮只新增只读 smoke 脚本和文档。

选用的 fixture 是已有 dry-run 输出：

```text
experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/
```

该目录只有 4 个 `sample_key`、20 条五专家 manifest 记录，且使用 `packed_npy_v1`，适合快速 smoke。

## 操作

1. 阅读 `docs/refactor/stage1_migration_candidates.md`，确认 P0/P1 重构前需要锁定 sample 顺序、shape、packed row index 和 fusion metrics。
2. 读取 `visual_router_experiments/common/prediction_array_io.py` 和 `visual_router_experiments/stage1_vali_test_router/fusion_utils.py`，确认当前数组读取接口与正式五专家顺序。
3. 新增 `tests/smoke/stage1_golden_smoke.py`：
   - 只读默认 fixture 目录；
   - 从 `fusion_utils.py` 源码解析 `MODEL_COLUMNS`，避免为 smoke 导入训练依赖；
   - 固定 4 个 `sample_key` 的顺序；
   - 检查 manifest 原始专家顺序和正式动作空间顺序的差异；
   - 按 `MODEL_COLUMNS=["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]` 组装 `y_pred`；
   - 检查 `y_pred=(4,5,48,1)`、`y_true=(4,48,1)`；
   - 复算每条记录的 MAE/MSE，确认 `packed_npy_v1` row index 读取一致；
   - 用固定 golden 权重检查 hard top-1 选择和 raw soft fusion MAE/MSE。
4. 新增 `docs/refactor/golden_fixture.md`，记录 fixture 来源、锁定契约和运行命令。
5. 使用 `quito` 环境运行验证命令：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

## 结果

验证命令通过，关键输出为：

```text
通过：五专家顺序固定为 ['DLinear', 'PatchTST', 'CrossFormer', 'ES', 'NaiveForecaster']
通过：sample_key 顺序固定，sample_count=4
通过：y_pred shape=(4, 5, 48, 1)，y_true shape=(4, 48, 1)
通过：hard top-1 选择=['CrossFormer', 'DLinear', 'PatchTST', 'DLinear']，MAE=0.416048437，MSE=0.456369758
通过：raw soft fusion MAE=0.410296679，MSE=0.488154024
完成：Stage 1 golden smoke 全部通过
```

本轮未移动、删除、重命名现有脚本；未改变正式 full-scale 路径和输出；未启动训练或长期后台任务。

## 结论

Stage 1 重构前 golden fixture 已建立。该 smoke 能快速读取 4 个样本的 packed cache，并锁定后续 reader/metrics 重构最容易出错的行为边界。

## 下一步方案

后续如果抽取公共 prediction batch reader 或 fusion metrics，应先运行：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

若 smoke 失败，应先解释 sample_key 顺序、五专家顺序、shape、row index 或 MAE/MSE 差异，再继续迁移正式入口。
