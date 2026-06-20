# Stage 1 P14c Visual Eval-Only Canonical Bypass Plan

创建日期：2026-06-20

## 1. 目标

P14c 冻结 Visual eval-only 的 canonical bypass 方案。它回答 legacy Visual Router
evaluation 阶段如何从当前 SQLite batch arrays 逐步过渡到
`ExpertBatch + FeatureBatch + RouterOutput + EvaluationInputAdapter` 的内存协议链路。

本轮只做文档审计和迁移方案冻结，不改正式入口，不新增正式 Visual provider/head 代码，
不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 2. 目标链路

Visual eval-only 的 future canonical bypass 目标链路为：

```text
SampleManifest ordered sample_keys
  -> VisualFeatureProvider / VisualMockFeatureProvider / legacy embedding path
  -> FeatureBatch(sample_keys, features, feature_schema, extra)
  -> legacy SQLite prediction arrays 或 PredictionCacheExpertProvider
  -> ExpertBatch(sample_keys, model_columns, y_pred, y_true, extra)
  -> Visual RouterHead / legacy MLP adapter
  -> RouterOutput(sample_keys, model_columns, logits, weights, extra)
  -> EvaluationInputAdapter
  -> Evaluator summary / per-sample rows
  -> Runtime artifact writer, future canonical run_dir only
```

当前正式入口仍保持 legacy 路径。P14c 的作用是冻结上述连接口径，防止后续把
prediction cache、视觉特征、head、evaluation 写出和 runtime artifact 混在同一层。

## 3. ExpertBatch 边界

P9d/P9f 已证明 Visual legacy batch arrays 可旁路包装为 `ExpertBatch`：

- P9d 在 evaluation batch 中把当前 SQLite prediction lookup 得到的 `y_pred/y_true`
  包装为 `ExpertBatch`，再用 `EvaluationInputAdapter` 旁路复算 hard/raw-soft rows。
- P9f 在 training batch 中把 legacy `y_pred/y_true` 包装为 `ExpertBatch`，并从
  `ExpertBatch.y_pred/y_true` 复算 MAE/MSE expert errors，对照 legacy `expert_errors`。

P14c 固定的 `ExpertBatch` contract：

- 提供 `sample_keys`，顺序必须等于调用方 manifest ordered sample_keys。
- 提供 `model_columns`，当前 Stage 1 canonical experiment 仍是固定五专家顺序。
- 提供 `y_pred`，shape 语义为 `[sample, model, horizon, target]`。
- 提供共享 `y_true`，shape 语义为 `[sample, horizon, target]`。
- `extra` 只保存 row index、backend/storage/version 等轻量 lineage。

`ExpertBatch` 明确不做：

- 不读取 Visual history window、pseudo image、ViT feature 或 router feature。
- 不保存 `FeatureBatch.features`、视觉 embedding 或 feature schema。
- 不读取 oracle/error，也不实现 `SupervisionProvider`。
- 不在本轮替换 Visual `SQLitePredictionIndex`。
- 不在本轮把 `PredictionCacheExpertProvider` 接到正式 Visual 入口。

短期 eval-only 迁移应优先保留 legacy SQLite path：正式入口继续使用
`SQLitePredictionIndex.fetch_records(...)` 与现有 batch array loading；仅在 batch arrays
已经存在后构造 `ExpertBatch` 作为旁路或未来 adapter 输入。

## 4. FeatureBatch 边界

P14b 的 `VisualMockFeatureProvider` 只是 smoke-only 证明：在不读取文件、不访问
prediction/oracle/run_dir 的情况下，Visual-style provider 可以按 ordered sample_keys 输出
`FeatureBatch(features=(4, 8), dtype=float32)`。

future `FeatureBatch` 可来自三类来源：

- `VisualMockFeatureProvider`：仅用于 smoke，输入为 tiny in-memory history window fixture，
  encoder 为 deterministic stub。
- future `VisualFeatureProvider`：输入为 history window `x`，可在 runtime 注入的 device /
  dtype / encoder policy 下执行 `x -> pseudo image -> frozen ViT -> embedding`。
- legacy embedding path：迁移过渡期可在正式入口已有 `iter_online_embedding_batches(...)`
  得到 embedding 后，薄包装成 `FeatureBatch`，用于 adapter/head smoke 或旁路校验。

P14c 固定的 `FeatureBatch` contract：

- 只保存 router/head 所需视觉特征和轻量 feature schema。
- `sample_keys` 顺序必须由调用方 manifest 决定。
- `features` 不包含 oracle、expert error、future `y`、`y_true` 或专家预测。
- `feature_schema` 记录 history source、pseudo-image variant、encoder、pooling、feature dim
  和 runtime-only storage policy。
- `extra` 只保存 provider/source/version 等轻量 lineage。

`FeatureBatch` 明确不做：

- 不读取 prediction cache、SQLite prediction index、oracle/error 或 run_dir。
- 不写 checkpoint、status、metadata、CSV、summary 或 canonical artifacts。
- 不保存 `ExpertBatch.y_pred/y_true`。
- 不决定 loss、optimizer、scaler、checkpoint/resume 或 full-scale launcher 策略。

`FeatureBatch` 与 `ExpertBatch` 的唯一连接点是 ordered `sample_keys` 对齐。任何
sample_key 缺失、重复、顺序不一致或 model_columns 不一致，都应由 protocol/runtime
adapter 显式报错，而不是由 provider 私自重排。

## 5. RouterHead / Legacy MLP 边界

Visual RouterHead adapter 尚未抽取。当前正式入口仍使用 legacy `VisualMLPRouter` 和
正式训练/evaluation helper 生成 logits、weights 与 legacy prediction rows。

P14c 固定的 head 迁移方向：

- 后续可先做 smoke-only `VisualRouterHead` mock，用 `FeatureBatch.features` 生成固定
  `RouterOutput`。
- 也可对 legacy MLP 包一层 thin adapter：调用方仍负责加载 checkpoint/scaler/device，
  adapter 只消费 `FeatureBatch` 并输出 `RouterOutput`。
- `RouterOutput.sample_keys` 必须与 `FeatureBatch.sample_keys` 一致。
- `RouterOutput.model_columns` 必须与 `ExpertBatch.model_columns` 一致。
- `RouterOutput` 至少包含 `weights`，可选包含 `logits` 和 diagnostics `extra`。

RouterHead 明确不做：

- 不读取 prediction cache 或 `ExpertBatch.y_pred/y_true`。
- 不读取 oracle/error 或 run_dir。
- 不写 evaluation CSV/summary/metadata/status/checkpoint。
- 不在 P14c 修改 loss、optimizer、scheduler、scaler、checkpoint/resume 或 training loop。

## 6. Evaluation / Artifact 边界

`EvaluationInputAdapter` 是 eval-only canonical bypass 的内存连接层。它消费同一批
`ExpertBatch + RouterOutput`，检查 `sample_keys` 和 `model_columns` 对齐，再生成
Evaluator 所需输入。

固定边界：

- `EvaluationInputAdapter` 负责从 `ExpertBatch.y_pred/y_true` 与
  `RouterOutput.weights` 或显式 fusion weights 生成 evaluation input。
- Evaluator 负责产生 in-memory summary 和 per-sample rows，包括 hard top-1、raw soft
  fusion、MAE/MSE、selected counts、entropy 和 max weight 等口径。
- Runtime artifact writer 只负责 future canonical `run_dir` 写出，例如
  `evaluation/evaluation_summary.json` 与 `predictions/prediction_rows.csv`。

本轮明确不改：

- 不改正式 legacy CSV / summary / metadata / status / checkpoint schema。
- 不把 Runtime artifact writer 接到 legacy Visual 入口。
- 不让 provider/head/evaluator 直接知道 `run_dir`。
- 不从 legacy CSV 反推专家顺序或 sample order 作为 canonical source。

P11c/P11d 已证明 canonical artifact writer 可以在 tiny protocol smoke 中写出 future
`evaluation/` 与 `predictions/`，但这不代表 legacy Visual 输出已经迁移。

## 7. Smoke-Only 与正式迁移等待项

可先做 smoke-only：

- P14d 已完成：`VisualMockFeatureProvider -> mock VisualRouterHead -> RouterOutput ->
  EvaluationInputAdapter` protocol smoke，输入使用 P13b/P14b tiny fixture 与 P13b
  expert JSON 数值参考，输出只在内存检查 summary/rows；见
  `docs/refactor/stage1_visual_mock_protocol_eval_smoke.md`。
- P14d 可选覆盖 legacy embedding path wrapper 的最小 shape/order smoke，但不接真实
  Hugging Face ViT、不访问 `/data2`。
- P14e 已完成 legacy Visual MLP eval-only adapter audit，重点检查
  `FeatureBatch -> legacy MLP -> RouterOutput` 的 sample/model 保序和 dtype/device 边界。
- P14f 已完成 legacy Visual MLP adapter smoke，使用 smoke-only 小型 torch MLP 和
  state_dict fixture 验证 head-ready `FeatureBatch -> RouterOutput -> EvaluationInputAdapter`
  的内存链路。

必须等正式入口迁移阶段：

- 接入真实 `VisualFeatureProvider`，包括 Quito dataset history window、pseudo image 和
  frozen ViT forward。
- 把 `PredictionCacheExpertProvider` 或 prepared backend 替换正式 Visual SQLite path。
- 把 legacy `VisualMLPRouter` 完整迁移为正式 `VisualRouterHead` adapter。
- 改正式 CSV/summary/metadata/status/checkpoint schema。
- 改 loss、optimizer、scaler、checkpoint/resume 或 training loop。
- 接入 Runtime artifact writer 写正式 Visual canonical `run_dir`。
- 启动 pressure/full-scale 或访问 `/data2`。

## 8. 后续小步建议

推荐后续编号：

1. P14d 已完成：Visual mock FeatureBatch + mock RouterHead + EvaluationInputAdapter
   protocol smoke。
2. P14e 已完成：Visual eval-only legacy MLP adapter audit，见
   `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md`。结论是 future eval-only
   legacy MLP adapter 只消费 head-ready `FeatureBatch.features` 和已加载 MLP，输出
   `RouterOutput`；scaler fit/checkpoint/device/dtype/DataParallel 仍归 Runtime/entrypoint。
3. P14f 已完成：Visual legacy MLP adapter smoke，见
   `docs/refactor/stage1_visual_legacy_mlp_adapter_smoke.md`。新增 smoke-only thin adapter
   只消费 head-ready float32 `FeatureBatch`、显式 `model_columns` 和已加载 MLP，输出
   `RouterOutput` 并由 `EvaluationInputAdapter` 生成 summary/rows；它不是正式
   Visual RouterHead adapter。
4. P15：branch-specific small entrypoint decision，根据 P13d/P13e/P14a/P14b/P14c/P14d/P14e/P14f
   结果判断是否新增 Visual/TimeFuse branch-specific small entrypoint。

P15 前仍应保持 generic small canonical entrypoint thin，不把 Visual provider/head 的
branch-specific 逻辑塞回通用 CLI。

## 9. P14c 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增正式 VisualFeatureProvider。
- 不抽真实 ViT provider。
- 不新增 Visual RouterHead adapter 代码。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler、checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 10. P14c 验收

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_17dim_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_backend_expertbatch_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```
