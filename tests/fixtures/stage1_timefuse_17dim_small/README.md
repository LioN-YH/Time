# Stage 1 P13e TimeFuse 17-Dim Small Fixture

本目录保存 P13e 的仓库内小型 TimeFuse-style 17 维 feature fixture，用于
`TimeFuseFeatureCacheProvider -> FeatureBatch` small smoke。

## 来源与用途

- `features_17d.csv` 使用 P13b
  `tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 中相同的
  4 个 `sample_key`。
- CSV 行顺序刻意不同于 P13b manifest 行顺序，用于验证 provider 按调用方传入的
  ordered sample_keys 重排，而不是依赖 CSV 原始顺序。
- 17 个 feature column 名称复用正式 TimeFuse feature cache builder 的
  `FEATURE_COLUMNS`：`mean`、`std`、`min`、`max`、`skewness`、`kurtosis`、
  `autocorrelation_mean`、`stationarity`、`rate_of_change_mean`、
  `rate_of_change_std`、`autoreg_coef_mean`、`residual_std_mean`、
  `frequency_mean`、`frequency_peak`、`spectral_entropy`、
  `spectral_skewness`、`spectral_kurtosis`。

## 边界

本 fixture 只是 small smoke 输入，不是 full-scale feature cache，不来自 `/data2`，
也不代表正式 TimeFuse fusor 入口已经迁移。它不包含 oracle label、oracle value、
per-model error、prediction cache path、`y_true` 或专家预测；这些信息不应进入
`FeatureProvider`。
