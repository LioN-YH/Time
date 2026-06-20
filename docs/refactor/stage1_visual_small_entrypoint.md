# Stage 1 P15c Visual Small Entrypoint

创建日期：2026-06-21

## 1. 目标

P15c 新增 Visual-specific small canonical entrypoint：

```text
scripts/run_stage1_visual_small.py
```

本阶段只做 Visual Router 主线的 small fixture / mock feature / smoke adapter pattern 级别
canonical rehearsal。它不是正式 Visual Router 训练入口迁移，不修改
`train_visual_router_online_streaming.py`，不访问 `/data2`，不启动训练、pressure 或
full-scale。

## 2. 输入 Fixture

默认输入全部来自仓库内 small fixture：

- `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`
- `tests/fixtures/stage1_real_derived_small/expert_predictions.json`
- `tests/fixtures/stage1_visual_feature_mock/history_windows.json`

默认 `--split-name test`，使用 manifest 行顺序过滤后的两个 test sample。传
`--split-name all` 可使用全部 small fixture；该选项仍只服务 small rehearsal。

CLI 参数：

```text
--sample-manifest-csv
--history-windows-json
--expert-predictions-json
--output-dir
--split-name
--run-id
--config-name
--feature-dim
--strict / --no-strict
```

## 3. 串联链路

P15c entrypoint 串联：

```text
SampleManifest
  -> VisualMockFeatureProvider / FeatureBatch
  -> JsonExpertSmallProvider / ExpertBatch
  -> script-local SmokeOnlyVisualMLPAdapter / RouterOutput
  -> EvaluationInputAdapter / Evaluator summary + rows
  -> Runtime artifact writer / canonical run_dir
```

`FeatureBatch` 和 `ExpertBatch` 是并列输入，二者都只通过 ordered `sample_keys` 对齐。
`VisualMockFeatureProvider` 只产生 head-ready float32 `FeatureBatch(features=[sample, 8])`，
不产生 `ExpertBatch`，不读取 prediction cache、oracle、run_dir、checkpoint 或 ViT 资源。

脚本内的 `SmokeOnlyVisualMLPAdapter` 只是 P15c 局部 smoke-only adapter pattern。它不是正式
Visual RouterHead adapter，不加载真实 checkpoint，不处理 scaler，不接真实视觉路由 head，
也不决定 device/DataParallel 策略。固定初始化的小型 MLP 只用于让 `RouterOutput(logits,
weights)` 可复现，并验证 evaluator 与 Runtime artifact writer 的组合边界。

## 4. 输出 Run Dir

`--output-dir/--run-id` 下写 canonical run_dir。当前 Runtime writer 能力下包含：

- `run_metadata.json`
- `run_status.json`
- `inputs/sample_manifest_ref.json`
- `inputs/split_summary.json`
- `evaluation/evaluation_summary.json`
- `predictions/prediction_rows.csv`
- `logs/visual_small_entrypoint.log`

`evaluation/evaluation_summary.json` 包含 hard top-1 与 raw-soft fusion 的 MAE/MSE、
selected counts、平均 entropy、平均 max weight 和 model columns。当前 writer 的 per-sample
rows 文件名仍是 `predictions/prediction_rows.csv`，不是 `evaluation/rows.csv`；P15c 不为补齐
目录名写假 artifact。

## 5. 与 Generic / TimeFuse Small CLI 的区别

`scripts/run_stage1_canonical_small.py` 继续保持 branch-neutral thin CLI，只服务通用 tiny
fixture 和 canonical dataflow 最小回归。

`scripts/run_stage1_timefuse_small.py` 是 TimeFuse-specific thin CLI，固定使用 17 维
TimeFuse feature schema、`TimeFuseFeatureCacheProvider` 和 `TimeFuseLinearSoftmaxHead`。

`scripts/run_stage1_visual_small.py` 是 Visual-specific thin CLI：

- 固定使用 `VisualMockFeatureProvider` 和 history window mock fixture。
- 固定验证 `FeatureBatch + ExpertBatch -> smoke-only MLP adapter -> RouterOutput`。
- 在 metadata 中记录 `visual_router` branch-specific feature/head lineage。
- 不承载 TimeFuse 逻辑，不修改 generic small CLI，也不修改 TimeFuse small CLI。

## 6. 与未来正式 Visual Router 迁移的关系

P15c 只证明 Visual 分支可以用 canonical protocol object 和 Runtime writer 串通 small
entrypoint。future 正式 Visual Router 迁移仍需要另起步骤处理：

- 真实 history window / pseudo image / frozen ViT feature provider；
- scaler、checkpoint state、checkpoint loading、resume；
- 正式 Visual RouterHead adapter；
- loss、optimizer、train/eval split 策略；
- full-scale streaming prediction backend 与 shard-aware runtime；
- launcher、GPU 资源、后台运行和 status 监控。

P15c 的 mock feature 和 smoke-only MLP 不能解释为正式 Visual Router 已迁移或已训练。

## 7. 明确不做

- 不修改 `scripts/run_stage1_canonical_small.py`。
- 不修改 `scripts/run_stage1_timefuse_small.py`。
- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改正式 evaluation 入口。
- 不新增 TimeFuse 逻辑。
- 不访问 `/data2`。
- 不读取真实 checkpoint。
- 不接真实视觉路由 head。
- 不启动 ViT embedding。
- 不启动训练、pressure 或 full-scale。
- 不新增 Bash launcher。
- 不把 Bash 引入 `time_router`。
- 不把 `run_dir` 传入 provider。
- 不把 prediction cache 设计成 interface。
- 不为兼容旧版 `96_48_S` full-scale 输出 schema 写适配逻辑。

## 8. 验收

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_smoke.py
```

该 smoke 覆盖：

- entrypoint subprocess 可运行完成；
- canonical run_dir 被创建；
- `run_metadata.json` / `run_status.json` / inputs / evaluation JSON 存在；
- summary 包含 hard/raw-soft MAE/MSE 和 selected counts；
- prediction rows 保持 sample_key 顺序；
- `FeatureBatch` sample_keys 与 `ExpertBatch` 对齐；
- `FeatureBatch` shape 为 `[sample, 8]` 且 dtype 为 `float32`；
- `ExpertBatch.model_columns` 与 `RouterOutput` / evaluator 对齐；
- weights shape 正确、有限值、softmax row sum 约等于 1；
- generic small CLI 和 TimeFuse small CLI 文件在 smoke 前后不变；
- stdout/stderr 不出现 `/data2`、正式训练入口、真实 checkpoint 或 ViT。
