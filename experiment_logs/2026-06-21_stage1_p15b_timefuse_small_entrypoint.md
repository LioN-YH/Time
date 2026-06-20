# Stage 1 P15b TimeFuse-specific small canonical entrypoint

日志日期：2026-06-21 00:27:53 CST

## 目的

新增 TimeFuse-style fusor baseline 支线的 small canonical entrypoint thin slice，验证
real-derived small input 可以沿 canonical dataflow 串联到 Runtime artifact writer。

## 背景

P15a 已决策：`scripts/run_stage1_canonical_small.py` 必须继续保持 generic thin CLI；后续
需要分别新增 TimeFuse-specific 和 Visual-specific small canonical entrypoint。P15b 先做
TimeFuse 支线，因为 17 维 `TimeFuseFeatureCacheProvider`、`TimeFuseLinearSoftmaxHead`、
`EvaluationInputAdapter` 和 Runtime writer 已有稳定 small smoke 基础。

本阶段不是正式 TimeFuse fusor 训练入口迁移，不修改 `train_timefuse_fusor_streaming.py`，
不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 操作

1. 阅读任务说明、当前 git 分支和现有 Stage 1 TimeFuse/provider/head/runtime/smoke 代码。
2. 新增 `scripts/run_stage1_timefuse_small.py`：
   - 默认读取 `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`。
   - 默认读取 `tests/fixtures/stage1_real_derived_small/expert_predictions.json`。
   - 默认读取 `tests/fixtures/stage1_timefuse_17dim_small/features_17d.csv`。
   - 支持 `--sample-manifest-csv`、`--features-csv`、`--expert-predictions-json`、
     `--output-dir`、`--split-name`、`--run-id`、`--config-name` 和 strict 开关。
   - 串联 `SampleManifest -> ExpertBatch -> TimeFuseFeatureCacheProvider / FeatureBatch ->
     TimeFuseLinearSoftmaxHead / RouterOutput -> EvaluationInputAdapter -> Runtime artifact writer`。
3. 新增 `tests/smoke/stage1_timefuse_small_entrypoint_smoke.py`：
   - 通过 subprocess 调用新增 entrypoint。
   - 验证 canonical run_dir artifact。
   - 在内存中复用 provider/head 组合检查 17 维 FeatureBatch、model_columns 对齐、weights
     shape/有限值/softmax row sum 和 sample_key 保序。
   - 检查 generic small CLI 文件在 smoke 前后不变。
4. 新增 `docs/refactor/stage1_timefuse_small_entrypoint.md`。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。
6. 运行新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_small_entrypoint_smoke.py
```

## 结果

新增 smoke 已通过，输出确认：

- entrypoint subprocess 返回码为 0。
- stdout 包含 `run_dir`，stdout/stderr 未出现 `/data2`。
- canonical run_dir、`run_metadata.json`、`run_status.json`、inputs 和 evaluation JSON 均存在。
- evaluation summary 包含 hard/raw-soft MAE/MSE 与 selected counts。
- `predictions/prediction_rows.csv` 保持 test split sample_key 顺序。
- `logs/timefuse_small_entrypoint.log` 已写出。
- 内存协议对象验证覆盖 17 维 FeatureBatch、ExpertBatch/RouterOutput 对齐和 softmax 权重。

本次未修改 `scripts/run_stage1_canonical_small.py`，未修改正式 TimeFuse 或 Visual Router 训练入口。

## 结论

P15b thin slice 已完成：TimeFuse-style small canonical entrypoint 可以在仓库内 small fixture 上
串通 canonical protocol objects，并由 Runtime writer 写出 canonical run_dir。

该结果只能证明 small rehearsal 链路成立；fixed deterministic linear-softmax head 不是训练后的
TimeFuse fusor baseline，不能作为正式 full-scale 结果引用。

## 下一步方案

1. 运行完整 P15b 回归 smoke 和 compileall。
2. 检查 git diff，确认没有修改正式训练入口、没有修改 generic small CLI 行为。
3. 提交并推送 `refactor/stage1-route-audit`。
4. 后续进入 P15c，新增 Visual-specific small canonical entrypoint thin slice。
