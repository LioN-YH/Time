# Stage 1 P14b Visual FeatureProvider Mock Smoke

日志日期：2026-06-20 16:20:21 CST

## 目的

新增 Stage 1 P14b Visual `FeatureProvider` minimal mock/fixture smoke，验证 Visual-style
provider 可以按 manifest ordered `sample_keys` 输出 `FeatureBatch`，同时守住不读取
prediction cache、oracle/error、`y_true`、run_dir、metadata、status、checkpoint 和 `/data2`
的边界。

## 背景

P14a 已完成 Visual FeatureProvider insertion audit，结论是未来 Visual provider 的最小输出
应为 `FeatureBatch(sample_keys, features, feature_schema, extra)`。provider 应负责按
ordered sample_keys 提供可部署视觉特征，但不读取 prediction cache、oracle/error、run_dir、
checkpoint、status，不接管 Visual RouterHead、loss、optimizer 或 evaluation 写出。

本轮只做 smoke-only mock，不迁移正式入口，不加载真实 Hugging Face ViT，不访问 `/data2`，
不启动训练、pressure 或 full-scale。

## 操作

1. 新增 `time_router/features/visual_mock.py`：
   - `DeterministicVisualEncoderStub` 将一维 history window x 编码为 8 维 `float32` 统计特征；
   - `VisualMockFeatureProvider` 显式接收内存 `sample_key -> history_window_x` 映射，并通过
     `load_batch(sample_keys)` 按传入顺序输出 `FeatureBatch`。
2. 更新 `time_router/features/__init__.py`，导出 `VisualMockFeatureProvider` 和
   `DeterministicVisualEncoderStub`。
3. 新增 `tests/fixtures/stage1_visual_feature_mock/`：
   - `history_windows.json` 保存 4 个 P13b sample_key 对应的小型 history window x；
   - README 明确 fixture 不包含 future y、y_true、oracle/error、prediction cache path、
     run_dir、metadata、status、checkpoint 或 `/data2`。
4. 新增 `tests/smoke/stage1_visual_feature_provider_mock_smoke.py`：
   - 使用 `tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 作为 ordered sample_keys；
   - 在 provider 阶段 patch `open`、`Path.open`、`Path.read_text` 和 `np.load`，禁止任何文件读取；
   - 检查 `FeatureBatch.sample_keys`、`features` shape/dtype、`feature_schema`、`extra` 和
     `experiment_logs/run_outputs/` 一层目录集合。
5. 新增 `docs/refactor/stage1_visual_feature_provider_mock_smoke.md`。
6. 更新 `docs/refactor/stage1_visual_feature_provider_insertion_audit.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md`、
   `docs/refactor/stage1_refactor_roadmap.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

新增 smoke 已运行通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
```

关键输出：

```text
开始 Stage 1 P14b Visual FeatureProvider mock smoke
通过：P13b manifest 和 Visual history window fixture 存在，且 sample_key 集合对齐
通过：FeatureBatch 按 manifest 保序，features shape=(4, 8)，dtype=float32
通过：feature_schema 记录 visual_mock schema、history_source、pseudo_image 和 encoder_stub 口径
通过：provider 阶段未读取文件、oracle/error/prediction/y_true/run_dir，未创建 run_outputs 运行目录
完成：Stage 1 P14b Visual FeatureProvider mock smoke 全部通过
```

回归 smoke 和 compileall 也已运行通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

回归结果：

- P13e TimeFuse 17 维 FeatureProvider smoke 通过。
- P13d prediction backend -> ExpertBatch small smoke 通过。
- P13b real-derived small fixture smoke 通过。
- P11d canonical protocol run smoke 通过。
- `compileall` 覆盖 `time_router`、`scripts`、`tests/smoke` 和
  `visual_router_experiments/stage1_vali_test_router` 通过。

## 结论

P14b 的最小 Visual mock provider contract 已成立：

- provider 由调用方显式传入 ordered `sample_keys`；
- provider 按 manifest 顺序读取内存 history window x；
- deterministic encoder stub 输出 `[sample, 8]` `float32` 特征；
- `FeatureBatch.feature_schema` 记录 visual mock schema、feature_dim、history_source、
  pseudo_image/mock_not_materialized 和 encoder_stub 口径；
- `FeatureBatch.extra` 只记录 provider_name、source 和 num_available_rows；
- provider 阶段不读取任何文件，不读取 prediction/oracle/`y_true`/run_dir/status/checkpoint，
  不创建 canonical run_dir。

## 下一步方案

1. 小步提交并 push 到 `refactor/stage1-route-audit`。
2. 后续 P14c 可做 Visual eval-only canonical bypass plan，继续不替换正式入口、不改正式输出 schema。
