# Stage 1 P16k Visual small loaded path artifact parity

日志日期：2026-06-21 15:27:53 CST

## 目的

新增 Visual small entrypoint 默认 mock path 与显式 loaded legacy path 的 artifact parity smoke，确认两条路径写出的 canonical run_dir 共同结构、metadata schema、evaluation summary schema、prediction rows schema 和 sample_key 顺序保持一致。

## 背景

P16j 已让 `scripts/run_stage1_visual_small.py` 支持默认 mock path 和显式 loaded legacy path。loaded path 可串联 P16c precomputed feature、可选 P16d scaler、P16i tiny checkpoint payload helper、legacy `VisualMLPRouter`、P16a adapter 和 Runtime writer。

本步不读取真实 checkpoint，不访问 `/data2`，不启动 ViT，不迁移正式 Visual full-scale 入口。目标是防止 loaded path 引入新的输出 schema 分叉，而不是比较两条路径的数值优劣。

## 操作

1. 读取 P16k 目标说明、现有 P16j loaded path smoke、P15d branch artifact parity smoke 和 `scripts/run_stage1_visual_small.py` 输出 schema。
2. 新增 `tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py`：
   - 使用 tempfile output root；
   - 先运行默认 mock path；
   - 再运行 loaded legacy path；
   - loaded path 使用 P16c precomputed fixture 和 tempfile tiny checkpoint payload；
   - 本步选择 no-scaler 简化，P16j smoke 已覆盖 scaler-enabled loaded path；
   - 检查两边 run_dir completed、共同 artifact 存在、metadata/status schema、evaluation summary schema、prediction rows schema、sample_key 顺序、split 列、model_columns 和有限指标字段。
3. 新增 `docs/refactor/stage1_visual_small_loaded_path_artifact_parity.md`，说明 P16k 是 Visual small entrypoint 内部两条路径的 artifact parity smoke。
4. 同步更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md`、`WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`。

## 结果

新增 smoke 已验证默认 mock path 记录 `loaded_legacy_mlp=false`，loaded path 记录 `feature_source=precomputed`、`loaded_legacy_mlp=true`、`checkpoint_payload_source=explicit_small_fixture`、`scaler_enabled=false`、`p16i_helper_used=true` 和 `p16a_adapter_used=true`。

两条路径允许 metrics 数值不同，但必须共享 `sample_count`、`model_columns`、prediction row 行数、`sample_key` 顺序和 `split` 列。

实际验收命令与结果：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
结果：通过。

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
结果：通过。默认 mock path 与 loaded legacy path 均 completed，canonical run_dir 共同结构、metadata/evaluation/prediction rows schema、sample_key/split/model_columns parity 成立。

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py
结果：通过。P16j loaded legacy path 仍可串联 precomputed feature、scaler、tiny checkpoint payload、P16a adapter 和 Runtime writer。

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
结果：通过。P15d branch small entrypoint artifact parity 未回归。

git diff --check
结果：通过。
```

`git diff --name-only` 显示本步未修改正式训练入口；新增/修改范围集中在 P16k smoke、文档、实验日志和结构索引。禁止 token 搜索中 `/data2`、ViT 和 streaming 入口只出现在新 smoke 的防护断言或文档的“不做”说明中。

## 结论

P16k 将 loaded legacy path 纳入 Visual small entrypoint artifact parity 门禁，确保 small rehearsal 的 canonical run_dir schema 不随 path 分叉。

## 下一步方案

小步提交并 push 到 `origin/refactor/stage1-route-audit`。真实 checkpoint dry-run 或 real Visual feature chain 需另起步骤。
