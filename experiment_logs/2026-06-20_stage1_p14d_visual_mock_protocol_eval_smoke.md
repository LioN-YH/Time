# Stage 1 P14d Visual mock protocol eval smoke

日志日期：2026-06-20 20:24:23 CST

## 目的

新增 Visual mock `FeatureBatch + mock RouterHead + EvaluationInputAdapter` protocol smoke，
验证 Visual 分支可以在 tiny fixture 上用纯内存协议对象从 `FeatureBatch + ExpertBatch`
走到 `RouterOutput + evaluation summary/rows`。

## 背景

P14b 已完成 `VisualMockFeatureProvider -> FeatureBatch` smoke，P14c 已冻结 Visual
eval-only canonical bypass plan。下一步需要证明 Visual mock 特征、专家预测批次、router
输出和 evaluator adapter 能在不迁移正式入口的情况下接通。

本轮仍保持 smoke-only 边界：不改正式入口，不加载真实 ViT，不接 legacy MLP，不写
canonical `run_dir`，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 操作

1. 新增 `tests/smoke/stage1_visual_mock_protocol_eval_smoke.py`。
2. smoke 从 `tests/fixtures/stage1_real_derived_small/sample_manifest.csv` 读取 ordered
   sample_keys。
3. smoke 使用 P14b `VisualMockFeatureProvider` 和
   `tests/fixtures/stage1_visual_feature_mock/history_windows.json` 构造
   `FeatureBatch(features=(4, 8), dtype=float32)`。
4. smoke 使用 `tests/fixtures/stage1_real_derived_small/expert_predictions.json` 作为 small
   fixture 数值参考，在内存中构造最小 `ExpertBatch`，并明确该 JSON 不是正式 prediction
   backend schema。
5. 在 smoke 内定义 `DeterministicVisualMockRouterHead`，只消费
   `FeatureBatch.features` 和显式 `model_columns`，输出 `RouterOutput`。
6. 调用 `EvaluationInputAdapter.evaluate(...)`，生成 in-memory summary 和 per-sample rows。
7. 在 mock head/evaluator 阶段用 patch 阻断 `open`、`Path.open`、`Path.read_text`、
   `np.load`、`np.save` 和 `np.savez`，确认该阶段不回读 cache、不写产物。
8. 新增 `docs/refactor/stage1_visual_mock_protocol_eval_smoke.md`。
9. 更新 `docs/refactor/stage1_visual_eval_canonical_bypass_plan.md`、
   `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。
10. 运行新增 smoke 和指定回归验收：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```

## 结果

新增 smoke 和指定回归验收均已通过：

- `FeatureBatch.sample_keys`、`ExpertBatch.sample_keys`、`RouterOutput.sample_keys`、
  `EvaluationInput.sample_keys` 和 per-sample rows 均保持 P13b manifest 行顺序。
- `RouterOutput.model_columns` 等于 `ExpertBatch.model_columns`。
- `RouterOutput.weights` shape 为 `[4, 3]`，每行 sum 接近 1。
- `EvaluationInput` 原样引用 `ExpertBatch.y_pred/y_true` 和 `RouterOutput.weights`。
- summary 中 `num_samples=4`，hard top-1 / raw soft fusion MAE/MSE、selected counts、
  entropy 和 max weight 均可生成。
- per-sample rows 数量为 4，且 sample_key 顺序保持 manifest 顺序。
- mock head/evaluator 阶段未触发文件 IO、`np.load` 或 `np.save`。
- `experiment_logs/run_outputs/` 一层目录没有新增项，说明 smoke 未写 canonical run_dir。

单项输出中的关键指标：

- hard top-1 MAE：`0.174999982`
- raw soft fusion MAE：`0.081910767`

回归验证通过项：

- `tests/smoke/stage1_visual_mock_protocol_eval_smoke.py`
- `tests/smoke/stage1_visual_feature_provider_mock_smoke.py`
- `tests/smoke/stage1_prediction_backend_expertbatch_smoke.py`
- `tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py`
- `tests/smoke/stage1_real_derived_small_fixture_smoke.py`
- `tests/smoke/stage1_canonical_protocol_run_smoke.py`
- `python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router`

## 结论

P14d 已证明 Visual mock 分支可以在 tiny fixture 上按以下内存协议链路接通：

```text
P13b manifest ordered sample_keys
  -> P14b VisualMockFeatureProvider
  -> FeatureBatch
  -> P13b expert JSON reference -> ExpertBatch
  -> smoke-only mock RouterHead
  -> RouterOutput
  -> EvaluationInputAdapter
  -> in-memory summary / rows
```

该结果只说明 smoke protocol 边界成立，不代表正式 Visual 入口已迁移；当前仍未接真实
ViT、legacy MLP、正式 SQLite prediction path、Runtime artifact writer 或 full-scale 输出。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续 P14e 可做 Visual eval-only legacy MLP adapter audit or smoke，重点检查
   `FeatureBatch -> legacy MLP -> RouterOutput` 的 sample/model 保序、dtype/device 和
   checkpoint 边界。
