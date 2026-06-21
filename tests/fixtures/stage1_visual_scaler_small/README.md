# Stage 1 P16d Visual Feature Scaler Small Fixture

本目录保存 P16d loaded Visual FeatureScaler boundary smoke 使用的最小 fixture。

- `raw_visual_features.csv` 使用 P13b real-derived small manifest 的 4 个 `sample_key`，行顺序刻意不同于 manifest，用于验证 smoke helper 按 requested sample_keys 重排。
- `scaler_state.json` 是已加载 scaler state 的 JSON 表示，包含固定 `mean` / `scale` 和 `feature_columns`。
- 本 fixture 只表达 raw/pre-head feature + loaded scaler state，不包含 `/data2`、checkpoint、ViT、pseudo image、run_dir、prediction、oracle 或 expert error 信息。
