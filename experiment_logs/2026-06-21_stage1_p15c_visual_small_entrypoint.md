# Stage 1 P15c Visual-specific small canonical entrypoint

日志日期：2026-06-21 01:25:50 CST

## 目的

新增 Visual-specific small canonical entrypoint thin slice，验证 Visual Router 主线可以在
small fixture / mock feature / smoke adapter pattern 级别串联 canonical dataflow 并写出
canonical run_dir。

本步不是正式 Visual Router 训练入口迁移，不应读取真实 checkpoint、不应启动 ViT embedding、
不应访问 `/data2`，也不应修改 generic small CLI 或 TimeFuse small CLI。

## 背景

P15b 已完成 TimeFuse-specific small canonical entrypoint：

- `scripts/run_stage1_timefuse_small.py`
- `tests/smoke/stage1_timefuse_small_entrypoint_smoke.py`
- `docs/refactor/stage1_timefuse_small_entrypoint.md`

P14b/P14d/P14f 已分别验证：

- `VisualMockFeatureProvider` 可从内存 history window 输出 8 维 float32 `FeatureBatch`；
- Visual mock `FeatureBatch + ExpertBatch -> RouterOutput -> EvaluationInputAdapter` 的内存链路可行；
- smoke-only loaded torch MLP adapter pattern 可把 head-ready `FeatureBatch.features` 转为
  canonical `RouterOutput`。

P15c 需要把这些边界组合成 Visual-specific small CLI，但仍保持正式训练入口、真实 checkpoint、
scaler、ViT provider 和 Bash launcher 不迁移。

## 操作

1. 新增 `scripts/run_stage1_visual_small.py`。
   - 默认读取 `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`。
   - 默认读取 `tests/fixtures/stage1_real_derived_small/expert_predictions.json`。
   - 默认读取 `tests/fixtures/stage1_visual_feature_mock/history_windows.json`。
   - 支持 `--sample-manifest-csv`、`--history-windows-json`、`--expert-predictions-json`、
     `--output-dir`、`--split-name`、`--run-id`、`--config-name`、`--feature-dim` 和
     `--strict / --no-strict`。
   - 串联 `SampleManifest -> VisualMockFeatureProvider / FeatureBatch ->
     JsonExpertSmallProvider / ExpertBatch -> script-local SmokeOnlyVisualMLPAdapter / RouterOutput ->
     EvaluationInputAdapter -> Runtime artifact writer`。
   - 在 `--output-dir/--run-id` 下写出 canonical run_dir。

2. 在 `scripts/run_stage1_visual_small.py` 内部定义 script-local smoke-only MLP adapter。
   - `SmokeOnlyVisualMLP` 使用固定 seed 和固定 uniform 初始化。
   - `SmokeOnlyVisualMLPAdapter` 只消费内存 `FeatureBatch` 和显式 `model_columns`。
   - adapter 不读取 checkpoint、不调用 `torch.load`、不处理 scaler、不接真实视觉路由 head。

3. 新增 `tests/smoke/stage1_visual_small_entrypoint_smoke.py`。
   - subprocess 调用 `scripts/run_stage1_visual_small.py`。
   - 验证 canonical run_dir、metadata/status、inputs、evaluation summary、prediction rows 和最小日志。
   - 在内存中复用 entrypoint provider/head 组合，验证 `FeatureBatch`、`ExpertBatch`、
     `RouterOutput` 和 evaluator 对齐。
   - 检查 generic small CLI 与 TimeFuse small CLI 文件在 smoke 前后不变。
   - 检查 stdout/stderr 不出现 `/data2`、正式训练入口、真实 checkpoint 或 ViT。

4. 新增 `docs/refactor/stage1_visual_small_entrypoint.md`。
   - 说明 P15c 目标、输入 fixture、串联链路、输出 canonical run_dir。
   - 说明它与 generic small CLI、TimeFuse small CLI 的区别。
   - 说明与未来正式 Visual Router 迁移的关系。
   - 明确不做 `/data2`、训练、full-scale、真实 checkpoint、真实视觉路由 head、scaler、
     ViT embedding、Bash 和正式入口迁移。

5. 同步更新：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`

## 结果

已新增长期文件：

- `scripts/run_stage1_visual_small.py`
- `tests/smoke/stage1_visual_small_entrypoint_smoke.py`
- `docs/refactor/stage1_visual_small_entrypoint.md`
- `experiment_logs/2026-06-21_stage1_p15c_visual_small_entrypoint.md`

P15c entrypoint 当前写出：

- `run_metadata.json`
- `run_status.json`
- `inputs/sample_manifest_ref.json`
- `inputs/split_summary.json`
- `evaluation/evaluation_summary.json`
- `predictions/prediction_rows.csv`
- `logs/visual_small_entrypoint.log`

已运行的验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_small.py tests/smoke/stage1_visual_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_protocol_run_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_canonical_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_small.py time_router tests/smoke/stage1_visual_small_entrypoint_smoke.py
```

验证结果：

- compileall 通过。
- 新增 P15c smoke 通过。
- P14b Visual FeatureProvider mock smoke 通过。
- P14d Visual mock protocol eval smoke 通过。
- P14f Visual legacy MLP adapter smoke 通过。
- P15b TimeFuse small entrypoint smoke 通过。
- P11d canonical protocol run smoke 通过。
- P12 canonical small entrypoint smoke 通过。
- smoke 确认 `FeatureBatch(features=(2, 8), dtype=float32)` 与 `ExpertBatch` sample_keys 对齐。
- smoke 确认 `RouterOutput.model_columns` 与 `ExpertBatch.model_columns` 对齐。
- smoke 确认 weights shape 正确、全为有限值，且 softmax row sum 约等于 1。
- smoke 确认 `prediction_rows.csv` 保持 manifest test split sample_key 顺序。
- smoke 确认 generic small CLI 和 TimeFuse small CLI 文件在运行前后不变。
- smoke 确认 stdout/stderr 不出现 `/data2`、正式训练入口、真实 checkpoint 或 ViT。

## 结论

P15c Visual-specific small canonical entrypoint thin slice 已完成初步实现和新增 smoke 验证。
它只证明 Visual branch-specific small rehearsal 可串通 canonical protocol objects 与 Runtime
artifact writer，不代表正式 Visual Router 训练入口、真实 checkpoint、scaler、ViT provider 或
full-scale 运行已迁移。

## 下一步方案

1. 用 `git diff` 确认没有修改正式训练入口，没有修改 generic small CLI 或 TimeFuse small CLI 行为。
2. 小步提交并推送到 `origin/refactor/stage1-route-audit`。
3. 后续如继续推进，应另起正式 Visual RouterHead adapter 或真实 Visual feature provider 审计/smoke；
   不能把 P15c 的 script-local smoke adapter 直接提升为正式 adapter。
