# Stage 1 P14b Visual FeatureProvider Mock Fixture

该目录保存 P14b smoke-only Visual FeatureProvider mock 使用的小型 history window fixture。

- `sample_manifest.csv` 不在本目录重复保存，smoke 直接使用
  `tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 作为 ordered sample_keys 来源。
- `history_windows.json` 保存 `sample_key -> history_window_x`，每个窗口只代表历史输入 `x`。
- fixture 不包含 future `y`、`y_true`、oracle、expert error、prediction cache path、
  run_dir、metadata、status、checkpoint 或 `/data2` 路径。

本 fixture 只用于验证 `VisualMockFeatureProvider -> FeatureBatch` 的保序、shape、
schema 和边界，不是正式 ViT provider，也不作为 Visual Router 训练输入。

