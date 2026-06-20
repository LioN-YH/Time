# Stage 1 P13b real-derived small fixture smoke

日志日期：2026-06-20 14:55:19 CST

## 目的

从仓库内已有小型 smoke fixture 派生真实字段风格的 Stage 1 small fixture，并用 P12b small
canonical entrypoint 验证 sample_key 保序、feature/expert join 和 canonical run_dir 写出。

## 背景

P13a 已完成真实小规模输入 mapping audit，明确 `SampleManifest` 只保存样本身份、split、顺序和
轻量 lineage；oracle/error 属于 `SupervisionProvider`；feature 属于 `FeatureProvider`；
P12b `expert_predictions.json` 只是 small fixture，不是正式 prediction cache schema。P13b
在该边界内只做仓库内小样例派生验证，不访问 `/data2`，不启动训练、pressure 或 full-scale。

## 操作

1. 读取用户目标文件，确认 P13b 范围、验收命令和禁止范围。
2. 检查当前分支、工作树、P12b entrypoint、P12b fixture smoke、P13a mapping audit、P10f/P10g
   smoke 和 golden fixture 文档。
3. 新增 `tests/fixtures/stage1_real_derived_small/`：
   - `sample_manifest.csv`：使用 P10f/P10g smoke 中 ETTh1 / ETTm2 / weather 小样本身份，字段为
     P11b 最小 manifest 字段；
   - `features.csv`：使用 P12b entrypoint 当前支持的三列 schema-style feature，并刻意打乱行顺序；
   - `expert_predictions.json`：使用 P12b 小数组格式，并刻意打乱 sample 顺序；
   - `README.md`：说明 fixture 来源、字段含义、不是 full-scale feature cache 或正式 prediction
     backend。
4. 新增 `tests/smoke/stage1_real_derived_small_fixture_smoke.py`，通过 subprocess 调用
   `scripts/run_stage1_canonical_small.py`，检查 canonical `run_dir`、manifest 保序、
   metadata inputs、evaluation sample_count 和 `/data2` 禁止边界。
5. 新增 `docs/refactor/stage1_real_derived_small_fixture.md`，并更新 P12b contract、P13a audit、
   entrypoint migration plan、roadmap 和 `WORKSPACE_STRUCTURE.md`。

## 结果

- 新增 P13b real-derived / schema-style fixture，manifest 行顺序为 ordered sample_keys 来源。
- 新增 smoke 已验证 feature/expert 行顺序不同于 manifest 时，P12b entrypoint 仍按 sample_key
  join 并输出按 manifest 保序的 `prediction_rows.csv`。
- `run_metadata.inputs` 已验证记录 `sample_manifest`、`feature_source` 和 `expert_fixture` 三个
  显式输入来源。
- `evaluation_summary.sample_count` 已验证等于 manifest 行数 4。
- 本步骤未修改正式训练入口、launcher、loss、optimizer、scaler、checkpoint/resume 或正式输出
  schema，未访问 `/data2`，未启动训练、pressure 或 full-scale。

已执行：

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

结果：全部通过。

## 结论

P13b 已完成仓库内真实字段风格小 fixture 的保序 join 和 canonical artifact 写出验证。该验证只是
small fixture smoke，不代表正式 Visual Router 或 TimeFuse-style fusor 入口已经迁移；三列
feature 不是 TimeFuse 17 维 full-scale feature cache，expert JSON 也不是正式 prediction
backend schema。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续 P13c 可继续审计真实 small batch 的 prediction backend / feature provider 连接点，但仍应
   保持 smoke-only 或 bypass，不直接迁移正式入口。
