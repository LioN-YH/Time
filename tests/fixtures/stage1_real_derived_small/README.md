# Stage 1 P13b Real-Derived Small Fixture

本目录保存 P13b 使用的 real-derived / schema-style 小型 fixture，用于驱动
`scripts/run_stage1_canonical_small.py` 验证 sample_key 保序、feature/expert join 和
canonical `run_dir` 写出。

## 来源

- `sample_manifest.csv` 的样本身份字段派生自 P10f
  `tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py` 与 P10g
  `tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py` 中的 4 行 ETTh1 /
  ETTm2 / weather 小型 fixture。
- `features.csv` 使用 P12b small entrypoint 当前支持的三列
  `trend_strength`、`seasonality_strength`、`recent_volatility`，并刻意打乱行顺序，用于验证
  `FeatureProvider` 按 manifest ordered sample_keys join。
- `expert_predictions.json` 使用 P12b 小数组格式，包含 `model_columns` 和每个 sample 的
  `sample_key`、`y_true`、`y_pred`，并刻意打乱 sample 顺序，用于验证 expert fixture 按
  manifest ordered sample_keys join。

## 字段口径

`sample_manifest.csv` 只包含 P11b `stage1_sample_manifest_v1` 最小字段：

```text
sample_key, split, config_name, dataset_name, item_id, channel_id, window_index, seq_len, pred_len
```

文件行顺序是 ordered sample_keys 的唯一来源。oracle label、oracle value、per-model error、
feature 值和 prediction cache 路径都不进入 manifest。

## 边界

本 fixture 是仓库内小样例派生的真实字段风格验证，不是 full-scale feature cache，也不是正式
prediction cache schema。`features.csv` 没有保存 TimeFuse 17 维真实 feature；这是为了复用
P12b small entrypoint 的固定三列 head contract。`expert_predictions.json` 仍只是 small smoke
fixture，不代表 packed npy、SQLite backend 或正式 `ExpertProvider` 已接入正式入口。
