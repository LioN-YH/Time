# Stage 1 P13c Real Small Backend / Provider Connection Audit

创建日期：2026-06-20

## 1. 目标

P13c 在 P13a/P13b 之后，只审计真实 small batch 如何从 fixture-driven path 过渡到
prediction backend / `ExpertProvider` / `FeatureProvider` 连接。本阶段冻结连接方案和边界，不迁移
正式入口，不新增真实数据脚本，不访问 `/data2`，不启动训练、pressure 或 full-scale。

P13c 的核心结论是：P12b/P13b 的 `expert_predictions.json` 与三列 `features.csv` 只保留为
generic small fixture；真实 small batch 后续应在 P12b generic thin CLI 之外，通过 smoke-only
backend/provider 或 branch-specific small entrypoint 逐步接入。

## 2. ExpertProvider 连接方案

P13b 的 `expert_predictions.json` 是 small fixture，不是正式 prediction cache schema。后续替换路线应为：

```text
SampleManifest ordered sample_keys
  -> prediction backend / prepared index
  -> PredictionBatchReader 或 PredictionCacheExpertProvider
  -> ExpertBatch
  -> RouterHead / Evaluator / Runtime artifact writer
```

分层判断如下：

| 组件 | 适合接入层 | P13c 判断 |
| --- | --- | --- |
| shared prediction SQLite backend | Runtime / backend prepare 层 | 适合从 manifest subset 构建 SQLite index，再按 ordered sample_keys fetch `(sample_key, model_name)` records；记录 array path、packed row index、missing report 和 metadata，但不进入 `SampleManifest` |
| `PredictionBatchReader` | ExpertProvider 底层 reader | 适合读取 packed/per-sample cache，并完成固定五专家顺序、共享 `y_true`、row index lineage、metrics 复算和 grouped array loading |
| `PredictionCacheExpertProvider` | prediction-cache adapter 层 | 适合作为 smoke-only `ExpertProvider`，把 reader 输出包装为 `ExpertBatch`；短期不接正式 Visual / TimeFuse 入口 |
| `ExpertBatch` | canonical protocol object | 是下游 RouterHead / Evaluator 应消费的专家输出契约，不暴露 prediction cache 内部路径给 `SampleManifest` |

必须保持的边界：

- prediction cache path、SQLite path、packed npy path、per-sample npy path 不进入 `SampleManifest`。
- `SampleManifest` 只提供 ordered sample_keys、split 和样本身份字段。
- SQLite backend 的 index artifact 属于 Runtime/backend prepare 层，不属于 provider protocol object 的必填字段。
- 本轮不替换 Visual Router 的 `SQLitePredictionIndex`，不把 `PredictionCacheExpertProvider` 接到正式入口。
- P13d 若做真实 small backend smoke，应只在 tempfile 或仓库内 fixture 上构造 `ExpertBatch`，并对照 P13b JSON fixture 输出，不改变正式入口。

## 3. FeatureProvider 连接方案

P13b 的三列 `features.csv` 是 schema-style fixture，只验证 `sample_key` join 与保序，不代表真实
TimeFuse 17 维 feature cache，也不代表 Visual online ViT feature。

后续替换路线应分支处理：

| 分支 | 真实 feature 来源 | 后续接入方式 | P13c 边界 |
| --- | --- | --- | --- |
| TimeFuse-style | 17 维 TimeFuse feature cache | 使用 `TimeFuseFeatureCacheProvider` 或 branch-specific small provider smoke 输出 `FeatureBatch`；必要时另起 small entrypoint/head contract 验证 17 维输入 shape | feature values 不进入 `SampleManifest`；scaler fit 属于 training/runtime，不属于 pure provider |
| Visual Router | Quito history window -> pseudo image -> frozen ViT embedding | 后续先审计 Visual history window / pseudo image / ViT feature provider 插入点，再做 branch-specific smoke；full-scale 仍在 batch runtime 内生成伪图像和 ViT embedding | 本轮不抽 Visual online ViT `FeatureProvider`，不落盘 pseudo image tensor 或 ViT embedding cache |

必须保持的边界：

- `FeatureProvider` 只提供可部署特征，不读取 oracle label、oracle value、per-model error 或未来 `y`。
- feature values 不进入 `SampleManifest`，只可在 `FeatureBatch.feature_schema` / `extra` 中记录轻量 lineage。
- TimeFuse 17 维 feature 与 Visual ViT embedding 可能有不同 input shape 和 head 约束，不应塞进 P12b generic small CLI 的三列 fixture 逻辑。
- Visual online ViT provider 牵涉 Quito window reader、pseudo image、Hugging Face cache、GPU dtype/DataParallel、latency 和 future finetune/joint training，必须晚于 smoke-only backend/provider 验证。

## 4. Entrypoint 策略

`scripts/run_stage1_canonical_small.py` 继续保持 generic thin small entrypoint。它负责：

- 接收 `--sample-manifest`、`--feature-source`、`--expert-fixture` 和 output 参数；
- 运行 tiny canonical dataflow；
- 验证 fixture contract、manifest 保序、feature/expert join 和 canonical `run_dir` 写出；
- 不承载 branch-specific backend prepare、feature extractor、training loop、Bash launcher 或 `/data2` 路径策略。

后续如果需要验证真实 TimeFuse 17 维 feature 或 Visual ViT feature，有两种低风险路线：

1. **branch-specific smoke**：直接构造 provider/head/evaluator protocol chain，不经过 generic small CLI。
2. **branch-specific small entrypoint**：另起薄 CLI，显式声明 feature schema、head/input shape 和 backend/provider spec。

不应把 TimeFuse 17 维 feature schema、Visual pseudo image / ViT feature、SQLite backend prepare 或 Bash /
`exp_scripts` 逻辑塞进 P12b generic small entrypoint。`scripts/` 仍只做薄 Python 入口；Bash /
`exp_scripts` 仍不进入 `time_router`。

## 5. Smoke-Only 与正式迁移边界

可以先做 smoke-only 的连接点：

- prediction backend -> `ExpertBatch` small smoke，使用 real-derived manifest 的 ordered sample_keys；
- `PredictionCacheExpertProvider` 对照 P13b `expert_predictions.json` 的 small fixture parity smoke；
- `TimeFuseFeatureCacheProvider` 读取 17 维 small feature CSV 并输出 `FeatureBatch` 的 branch-specific smoke；
- `ExpertBatch + FeatureBatch + branch head + EvaluationInputAdapter` 的 deterministic protocol chain smoke；
- Visual history window / pseudo image / ViT provider 插入点文档审计。

必须等正式入口迁移阶段的小步：

- 把 `PredictionCacheExpertProvider` 接入 Visual Router 或 TimeFuse full-scale 正式入口；
- 替换 Visual `SQLitePredictionIndex`；
- 抽 Visual online ViT `FeatureProvider` 并接入训练/eval；
- 抽 Visual `RouterHead` adapter；
- 实现正式 `SupervisionProvider`；
- 改正式 CSV / summary / metadata / status / checkpoint schema；
- 改 loss、optimizer、scaler、checkpoint/resume 或 launcher 行为；
- 引入复杂 config/runtime framework；
- 启动 pressure/full-scale 或访问 `/data2`。

## 6. 后续小步建议

建议按下列小步继续推进，保持 smoke-first：

1. **P13d：prediction backend -> ExpertBatch small smoke**
   - 使用 P13b real-derived manifest 的 ordered sample_keys。
   - 在 tempfile 中构造 packed/per-sample prediction manifest 或复用仓库内 small fixture。
   - 通过 shared prediction SQLite backend / `PredictionBatchReader` / `PredictionCacheExpertProvider`
     输出 `ExpertBatch`。
   - 对照 P13b `expert_predictions.json` 检查 sample_key、model_columns、`y_pred/y_true` shape 和指标。

2. **P13e：TimeFuse 17 维 FeatureProvider small smoke**
   - 新增或复用小型 17 维 feature CSV。
   - 用 `TimeFuseFeatureCacheProvider` 输出 `FeatureBatch`，验证 manifest ordered sample_keys 保序。
   - 若需要 head 输入 shape 验证，另走 TimeFuse branch-specific protocol chain，不扩展 generic small CLI。

3. **P14a：Visual feature provider insertion audit**
   - 只审计 history window、pseudo image、frozen ViT embedding 在 Visual 正式入口中的可插入点。
   - 明确哪些 runtime/encoder/device/latency/checkpoint 逻辑不能进入 pure `FeatureProvider`。

4. **P14b：Visual eval-only canonical bypass plan**
   - 规划 eval-only 阶段如何从 legacy SQLite batch arrays 旁路包装 `ExpertBatch`，再接 Visual feature/head/evaluator。
   - 仍不替换正式入口、不改输出 schema。

5. **P15：branch-specific small entrypoint decision**
   - 根据 P13d/P13e/P14a 结果判断是否需要 TimeFuse/Visual 各自 small CLI。
   - 若新增，必须保持 thin CLI，不把 Bash launcher 或 full-scale path strategy 下沉。

## 7. 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增真实数据脚本、Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 8. 验收

P13c 是纯文档审计，代码行为应保持不变。验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_fixture_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_runtime_artifact_writer_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_labels_sample_supervision_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_sample_supervision_protocol_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_sqlite_backend_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router scripts tests/smoke visual_router_experiments/stage1_vali_test_router
```
