# Stage 1 P13e TimeFuse 17 维 FeatureProvider Small Smoke

创建日期：2026-06-20

## 1. 目标

P13e 在 P13d 的 prediction backend -> `ExpertBatch` small smoke 之后，补齐
TimeFuse-style 真实分支的 feature-only 小链路验证。该 smoke 使用 P13b
`tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 的行顺序作为 ordered
sample_keys，读取仓库内小型 17 维 TimeFuse-style feature CSV，经
`TimeFuseFeatureCacheProvider` 输出 `FeatureBatch`。

本阶段只验证：

```text
P13b sample_manifest.csv ordered sample_keys
  + tests/fixtures/stage1_timefuse_17dim_small/features_17d.csv
  -> TimeFuseFeatureCacheProvider
  -> FeatureBatch
```

## 2. 新增内容

新增 fixture：

```text
tests/fixtures/stage1_timefuse_17dim_small/
├── README.md
└── features_17d.csv
```

新增 smoke：

```text
tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
```

## 3. 检查项

P13e smoke 覆盖：

- P13b manifest 是 ordered sample_keys 的唯一来源；
- `features_17d.csv` 包含相同 4 个 sample_key，但 CSV 行顺序刻意不同于 manifest；
- `features_17d.csv` 只包含 `sample_key + 17` 个 TimeFuse-style feature columns；
- `TimeFuseFeatureCacheProvider.load_batch(...)` 输出 `FeatureBatch`；
- `FeatureBatch.sample_keys` 与 manifest 行顺序完全一致；
- `FeatureBatch.features` shape 为 `[num_samples, 17]`，dtype 为 `float32`；
- `feature_schema` 记录 `timefuse_single_variable_meta_v1`、17 个 feature column、feature dim
  和 source；
- `extra` 记录 provider name、sample_key column、feature CSV path、可用行数和 dtype；
- feature 数值与 fixture 按 manifest 顺序重排后的参考矩阵一致；
- provider 阶段只读取 feature CSV，不读取 oracle label、oracle value、per-model error、
  `y_true`、prediction cache 或 `.npy`。

## 4. 边界

P13e 明确不做：

- 不接 `TimeFuseLinearSoftmaxHead`。
- 不接 `EvaluationInputAdapter`。
- 不写 canonical `run_dir`。
- 不扩展 `scripts/run_stage1_canonical_small.py` 的三列 generic fixture 逻辑。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不迁移正式入口。
- 不把 17 维 feature 塞进 `SampleManifest`。
- 不把 oracle/error 塞进 `FeatureProvider`。

## 5. 与 P13b/P13d 的关系

P13b 的 `features.csv` 继续作为 P12b generic small entrypoint 的三列
schema-style fixture，只验证 sample_key join 和 thin CLI artifact 写出。

P13e 的 `features_17d.csv` 是 branch-specific TimeFuse FeatureProvider small fixture，
不经过 P12b generic small entrypoint，也不要求 generic head/input shape 接受 17 维特征。

P13d 证明 prediction backend / `PredictionBatchReader` /
`PredictionCacheExpertProvider` 可以在 small smoke 中输出 `ExpertBatch`。P13e 只补齐
feature-only 侧的 `FeatureBatch` 输出，不把两者接到 RouterHead 或 Evaluator。

## 6. 验收命令

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
```

本阶段回归验收：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 7. 后续连接

P13e 后可以继续进入 P14a Visual feature provider insertion audit。若后续需要把
`ExpertBatch + FeatureBatch` 接到 TimeFuse branch head 或 evaluator，应另起
branch-specific protocol chain 或 small entrypoint，不扩展 P12b generic three-column small
entrypoint，也不声称正式入口已迁移。
