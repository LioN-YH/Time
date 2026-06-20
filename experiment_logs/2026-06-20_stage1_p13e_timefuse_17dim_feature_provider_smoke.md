# Stage 1 P13e TimeFuse 17 维 FeatureProvider Small Smoke

日志日期：2026-06-20 15:37:49 CST

## 目的

新增 P13e TimeFuse 17 维 `FeatureProvider` small smoke，验证 P13b real-derived manifest
的 ordered sample_keys 可以驱动仓库内小型 17 维 TimeFuse-style feature CSV，经
`TimeFuseFeatureCacheProvider` 输出 `FeatureBatch`，并保持 sample_key 顺序、feature shape、
feature schema 和数值一致。

## 背景

P13d 已完成 prediction backend -> `ExpertBatch` small smoke，证明 P13b
`expert_predictions.json` 只是数值参考，后续 expert 侧可通过 prediction backend /
`PredictionBatchReader` / `PredictionCacheExpertProvider` 输出 `ExpertBatch`。

P13b 的 `features.csv` 仍是三列 schema-style fixture，只验证 sample_key join 和 generic
small entrypoint 的输入 contract；它不是真实 TimeFuse-style 17 维 feature cache。P13e
因此只补齐 feature-only small smoke，不接 RouterHead，不接 Evaluator，不扩展 generic small
entrypoint，也不访问 `/data2`。

## 操作

1. 新增 `tests/fixtures/stage1_timefuse_17dim_small/` 目录。
2. 新增 `tests/fixtures/stage1_timefuse_17dim_small/features_17d.csv`，包含 P13b
   manifest 相同的 4 个 sample_key，以及正式 TimeFuse feature builder 使用的 17 个 feature
   column：`mean`、`std`、`min`、`max`、`skewness`、`kurtosis`、
   `autocorrelation_mean`、`stationarity`、`rate_of_change_mean`、
   `rate_of_change_std`、`autoreg_coef_mean`、`residual_std_mean`、`frequency_mean`、
   `frequency_peak`、`spectral_entropy`、`spectral_skewness`、`spectral_kurtosis`。
3. `features_17d.csv` 行顺序刻意不同于 P13b
   `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`，用于验证 provider 按
   manifest ordered sample_keys 保序。
4. 新增 `tests/fixtures/stage1_timefuse_17dim_small/README.md`，说明该 fixture 是 small
   smoke，不是 full-scale feature cache，不包含 oracle/error/prediction。
5. 新增 `tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py`，从 P13b manifest
   读取 ordered sample_keys，读取 `features_17d.csv` 并按 manifest 顺序构造参考矩阵。
6. smoke 使用 `TimeFuseFeatureCacheProvider` 输出 `FeatureBatch`，检查
   `FeatureBatch.sample_keys`、`features=(4, 17)`、`float32` dtype、`feature_schema`、
   `extra` 和数值一致性。
7. smoke 在 provider 阶段阻断除 feature CSV 外的文件读取和 `np.load`，验证
   `FeatureProvider` 不读取 oracle label、oracle value、per-model error、`y_true` 或
   prediction cache。
8. 新增 `docs/refactor/stage1_timefuse_17dim_feature_provider_smoke.md`，并同步更新
   `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

新增 smoke 已运行通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
```

输出确认：

- P13b manifest 和 17 维 feature fixture 存在，sample_key 集合对齐且 CSV 顺序已打乱；
- `FeatureBatch` 按 manifest 保序，features shape 为 `(4, 17)`，17 维数值一致；
- `feature_schema` / `extra` 记录 17 维 schema 与来源；
- provider 阶段未读取 oracle/error/prediction，未创建 `run_outputs` 运行目录。

完整回归验收已全部通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

回归结果：

- P13d prediction backend -> `ExpertBatch` small smoke 通过，expert 侧小链路未漂移。
- P13b real-derived small fixture smoke 通过，generic small entrypoint 的三列 fixture contract
  未被 P13e 扩展或破坏。
- P10g TimeFuse sample/supervision adapter smoke 通过，oracle/error 仍只进入
  `SupervisionBatch`，未混入 FeatureProvider。
- P11d canonical protocol run smoke 通过，Provider/Head/Evaluator 与 Runtime writer 分层未漂移。
- compileall 通过，覆盖 `time_router`、`scripts`、`tests/smoke` 和
  `visual_router_experiments/stage1_vali_test_router`。

## 结论

P13e 已补齐 TimeFuse feature-only 侧的 smoke 输入和验收脚本。该工作仍是 small smoke，
不迁移正式入口，不接 TimeFuse head/evaluator，不写 canonical run_dir，不把 17 维 feature
塞进 `SampleManifest`，也不把 oracle/error 塞进 `FeatureProvider`。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续进入 P14a Visual feature provider insertion audit，先审计 history window、pseudo
   image、frozen ViT embedding 的插入点，再决定是否需要 branch-specific Visual small smoke。
