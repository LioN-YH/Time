# Stage 1 P15b TimeFuse Small Entrypoint

创建日期：2026-06-21

## 1. 目标

P15b 新增 TimeFuse-specific small canonical entrypoint：

```text
scripts/run_stage1_timefuse_small.py
```

本阶段只做 TimeFuse-style fusor baseline 支线的 small fixture / real-derived small input 级别
canonical rehearsal。它不是正式 TimeFuse fusor 训练入口迁移，不修改
`train_timefuse_fusor_streaming.py`，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 2. 输入 Fixture

默认输入全部来自仓库内 small fixture：

- `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`
- `tests/fixtures/stage1_real_derived_small/expert_predictions.json`
- `tests/fixtures/stage1_timefuse_17dim_small/features_17d.csv`

默认 `--split-name test`，使用 manifest 行顺序过滤后的两个 test sample。传
`--split-name all` 可使用全部 small fixture；该选项仍只服务 small rehearsal。

CLI 参数：

```text
--sample-manifest-csv
--features-csv
--expert-predictions-json
--output-dir
--split-name
--run-id
--config-name
--strict / --no-strict
```

## 3. 串联链路

P15b entrypoint 串联：

```text
SampleManifest
  -> JsonExpertSmallProvider / ExpertBatch
  -> TimeFuseFeatureCacheProvider / FeatureBatch
  -> TimeFuseLinearSoftmaxHead / RouterOutput
  -> EvaluationInputAdapter / Evaluator summary + rows
  -> Runtime artifact writer / canonical run_dir
```

`FeatureBatch` 和 `ExpertBatch` 都只通过 ordered `sample_keys` 对齐。
`TimeFuseFeatureCacheProvider` 只读取 17 维 feature CSV，不产生 `ExpertBatch`。
`TimeFuseLinearSoftmaxHead` 使用 deterministic fixed weights 做 small rehearsal，不训练、
不保存 checkpoint。

## 4. 输出 Run Dir

`--output-dir/--run-id` 下写 canonical run_dir。当前 Runtime writer 能力下包含：

- `run_metadata.json`
- `run_status.json`
- `inputs/sample_manifest_ref.json`
- `inputs/split_summary.json`
- `evaluation/evaluation_summary.json`
- `predictions/prediction_rows.csv`
- `logs/timefuse_small_entrypoint.log`

`evaluation/evaluation_summary.json` 包含 hard top-1 与 raw-soft fusion 的 MAE/MSE、
selected counts、平均 entropy、平均 max weight 和 model columns。当前 writer 的 per-sample
rows 文件名仍是 `predictions/prediction_rows.csv`，不是 `evaluation/rows.csv`；P15b 不为补齐
目录名写假 artifact。

## 5. 与 Generic Small CLI 的区别

`scripts/run_stage1_canonical_small.py` 继续保持 branch-neutral thin CLI，只服务通用 tiny
fixture 和 canonical dataflow 最小回归。

`scripts/run_stage1_timefuse_small.py` 是 branch-specific thin CLI：

- 固定使用 TimeFuse 17 维 feature schema。
- 固定使用 `TimeFuseFeatureCacheProvider`。
- 固定使用 `TimeFuseLinearSoftmaxHead`。
- 在 metadata 中记录 `timefuse_fusor` branch-specific 输入与 head lineage。

P15b 没有修改 generic small CLI 行为。

## 6. 与 Future Full-Scale TimeFuse Fusor 的关系

P15b 只证明 TimeFuse 支线可以用 canonical protocol object 和 Runtime writer 串通 small
entrypoint。future full-scale TimeFuse fusor 仍需要另起步骤处理：

- shard-aware full-scale feature source；
- prediction backend / `ExpertProvider` 的正式接入；
- scaler、loss、optimizer、checkpoint/resume；
- train/eval split 策略；
- launcher、GPU/CPU 资源、后台运行和 status 监控。

P15b 的 fixed-weight linear-softmax head 不能解释为训练完成的 TimeFuse fusor baseline。

## 7. 明确不做

- 不修改 `scripts/run_stage1_canonical_small.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改正式 Visual Router 入口。
- 不新增 Visual-specific small entrypoint。
- 不访问 `/data2`。
- 不读取 full-scale artifact。
- 不启动训练、pressure 或 full-scale。
- 不新增 Bash launcher。
- 不把 Bash 引入 `time_router`。
- 不把 `run_dir` 传入 provider。
- 不把 feature cache path 设计成长期 interface。
- 不为兼容旧版 `96_48_S` full-scale 输出 schema 写适配逻辑。

## 8. 验收

新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_small_entrypoint_smoke.py
```

该 smoke 覆盖：

- entrypoint subprocess 可运行完成；
- canonical run_dir 被创建；
- `run_metadata.json` / `run_status.json` / inputs / evaluation JSON 存在；
- summary 包含 hard/raw-soft MAE/MSE 和 selected counts；
- prediction rows 保持 sample_key 顺序；
- `FeatureBatch` shape 为 17 维；
- `ExpertBatch.model_columns` 与 `RouterOutput` / evaluator 对齐；
- weights shape 正确、有限值、softmax row sum 约等于 1；
- generic small CLI 文件在 smoke 前后不变；
- stdout/stderr 不出现 `/data2`，不启动正式训练入口。
