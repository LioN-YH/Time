# Stage 1 P16c Visual Precomputed Small Fixture

本目录保存 P16c smoke 使用的最小 head-ready visual embedding CSV。

- `visual_embeddings.csv` 使用 P13b real-derived small manifest 的 4 个 sample_key，覆盖其中 test split 的两个 sample_key。
- CSV 行顺序故意不同于 manifest 行顺序，用于验证 `VisualPrecomputedFeatureProvider.load_batch(...)` 按 requested sample_keys 输出。
- `feature_0` 到 `feature_7` 是固定数值的 head-ready fixture，不代表真实 ViT embedding。
- fixture 不包含 `/data2`、checkpoint、ViT、scaler、run_dir、oracle、prediction cache 或 expert error。
