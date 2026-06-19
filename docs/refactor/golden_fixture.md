# Stage 1 Golden Fixture

记录日期：2026-06-19

## 目的

为 Stage 1 后续公共 batch reader / metrics / output schema 重构建立极小 golden smoke。该 fixture 只用于锁定既有行为，不改变正式 full-scale 路径、输出目录或训练逻辑。

## Fixture 来源

- 默认路径：`experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/`
- 数据规模：4 个 `sample_key`、20 条 `(sample_key, model_name)` manifest 记录。
- 存储格式：`packed_npy_v1`。
- 数组形状：`y_pred=(4, 5, 48, 1)`，`y_true=(4, 48, 1)`。

## 锁定契约

- `sample_key` 顺序固定为 manifest 首次出现顺序。
- 五专家动作空间顺序固定为 `["DLinear", "PatchTST", "CrossFormer", "ES", "NaiveForecaster"]`。
- manifest 原始行顺序不同于动作空间顺序，smoke 会显式重排并检查。
- `packed_npy_v1` 的 `y_true_row_index` / `y_pred_row_index` 必须能读回与 manifest `mae` / `mse` 一致的窗口数组。
- golden 权重下的 hard top-1 选择固定为 `["CrossFormer", "DLinear", "PatchTST", "DLinear"]`。
- golden 权重下 raw soft fusion 固定为 `MAE=0.4102966785430908`、`MSE=0.48815402388572693`。

## 运行方式

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

脚本只读 fixture，不训练、不写正式输出目录。后续重构公共读取器或评估模块时，应先跑该 smoke；若失败，应先解释 sample 顺序、专家顺序、shape、row index 或指标差异，再继续迁移。
