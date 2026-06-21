# Stage 1 P19a Visual Feature Chain Dry-run Fixture

本目录保存 P19a `VisualFeatureChainRunner` dry-run skeleton 使用的显式
raw window fixture。

- `raw_windows.json`：按 `sample_key` 保存 4 个小型 raw history window，每个
  window 长度为 6，只用于 smoke 验证 shape、保序和 lineage。
- sample_key 来自 `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`。
- 该 fixture 不来自 `/data2`，不包含真实 ViT embedding，不代表正式 full-scale
  Visual Router 数值。
