# Stage 1 P13d Prediction Backend -> ExpertBatch Small Smoke

日志日期：2026-06-20 15:27:43 CST

## 目的

新增 P13d prediction backend -> `ExpertBatch` small smoke，验证 P13b real-derived
manifest 的 ordered sample_keys 可以通过小型 prediction backend、SQLite records、
`PredictionBatchReader` 和 `PredictionCacheExpertProvider` 输出 `ExpertBatch`，并与
P13b `expert_predictions.json` 数值参考一致。

## 背景

P13c 已完成 backend/provider connection audit，结论是 P13b
`expert_predictions.json` 只是 small fixture，后续真实路径应逐步替换为
prediction backend -> `ExpertProvider` -> `ExpertBatch`。本轮只做 smoke，不迁移正式入口，
不访问 `/data2`，不启动训练、pressure 或 full-scale。

P13b sample_key 使用 `::` 风格的 schema-style 字符串，不是 strict canonical
`PredictionCacheKey.as_string()` 双下划线格式。为忠实使用 P13b manifest 行顺序，本轮为
`PredictionBatchReader` 和 `PredictionCacheExpertProvider` 增加显式
`validate_manifest_schema=False` bridge 开关；默认仍为 `True`，正式 prediction cache 继续使用
strict canonical schema 校验。

## 操作

1. 新增 `tests/smoke/stage1_prediction_backend_expertbatch_smoke.py`。
2. smoke 从 `tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 读取 ordered
   sample_keys，从 `expert_predictions.json` 按该顺序组装参考 `y_pred/y_true`。
3. smoke 在 `tempfile.TemporaryDirectory` 下构造 packed_npy_v1 prediction manifest、共享
   `y_true.npy`、每专家 `y_pred.npy` 和临时 SQLite backend。
4. smoke 通过 `build_prediction_sqlite_backend(...)`、`fetch_records(...)`、
   `records_to_ordered_rows(...)` 和 `load_prediction_arrays_grouped(...)` 验证 backend records、
   row index lineage 和数值 parity。
5. smoke 通过 `PredictionBatchReader(validate_manifest_schema=False)` 与
   `PredictionCacheExpertProvider(validate_manifest_schema=False)` 输出 `ExpertBatch`，检查
   sample_key 保序、model_columns、`y_pred/y_true` shape、数值、`row_index_metadata` 和
   `ExpertBatch.extra` 来源信息。
6. 新增 `docs/refactor/stage1_prediction_backend_expertbatch_smoke.md`，并同步更新
   `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

新增 smoke 已先行运行通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
```

输出确认：

- 已按 P13b sample_manifest 行顺序读取 sample_keys，并按该顺序组装参考数组；
- 已在 tempfile 构造 packed_npy_v1 prediction cache/backend fixture；
- shared SQLite backend 可按 manifest 顺序 fetch records，并按 row index 读回参考数组；
- `PredictionBatchReader` / `PredictionCacheExpertProvider` 输出 `ExpertBatch` 且数值对齐 P13b 参考。

完整回归验收已全部通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

回归结果：

- P13b real-derived small fixture smoke 通过，确认 P12b small entrypoint 仍可使用 P13b fixture
  写出 canonical run_dir，并保持 manifest 顺序。
- P12b canonical small entrypoint fixture smoke 通过，确认内联 fixture 与显式 fixture 口径未漂移。
- P10b shared prediction SQLite backend smoke 通过，确认完整 fixture、缺失报错和 missing report
  路径未受 P13d 影响。
- P11d canonical protocol run smoke 通过，确认 protocol chain 与 runtime writer 分层未漂移。
- compileall 通过，覆盖 `time_router`、`scripts`、`tests/smoke` 和
  `visual_router_experiments/stage1_vali_test_router`。

## 结论

P13d 已打通 smoke-only 的 prediction backend -> `ExpertBatch` 小链路。P13b
`expert_predictions.json` 仍只作为数值参考，不是正式 backend schema；本轮没有接正式入口，
没有替换 Visual `SQLitePredictionIndex`，也没有改变正式 CSV / summary / metadata / status /
checkpoint schema。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续 P13e 可做 TimeFuse 17 维 `FeatureProvider` small smoke；Visual history window /
   pseudo image / ViT provider 仍先进入 P14a 插入点审计。
