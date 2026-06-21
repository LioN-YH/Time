# Stage 1 P17a Visual canonical evaluation entrypoint thin slice

日志日期：2026-06-21 16:50:52 CST

## 目的

新增 Visual Router evaluation 的 canonical entrypoint thin slice，作为未来正式 Visual Router evaluation 入口的雏形，但不启动训练、不启动 ViT、不迁移 full-scale 训练入口。

## 背景

P16a-P16l 已完成 Visual Router canonical small migration foundation：已有 `VisualPrecomputedFeatureProvider`、`LoadedFeatureScaler`、Runtime-side tiny checkpoint payload loader、legacy `VisualMLPRouter` loaded path 和 Visual small artifact parity。当前仍不能修改正式 `train_visual_router_online_streaming.py`，也不能改变 P15c/P16j Visual small entrypoint 的默认行为。

## 操作

1. 新增 `scripts/run_stage1_visual_eval_canonical.py`。
   - 使用显式 CLI 参数读取 SampleManifest CSV、expert prediction JSON、precomputed visual feature CSV、router checkpoint payload、output dir、run id、config 和 split。
   - 串联 `SampleManifest -> VisualPrecomputedFeatureProvider -> optional LoadedFeatureScaler -> Runtime-loaded legacy VisualMLPRouter -> LoadedTorchMLPRouterHeadAdapter -> ExpertBatch -> EvaluationInputAdapter -> Runtime artifact writer`。
   - 默认禁止 `/data2`，默认 `--allow-real-checkpoint=false` 时只允许 `tests/fixtures` 或 `/tmp` checkpoint payload。
2. 新增 `tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py`。
   - 使用 P13b manifest / expert prediction fixture。
   - 使用 P16c precomputed visual feature fixture。
   - 在 tempfile 中构造 tiny legacy `VisualMLPRouter` checkpoint payload。
   - 验证 canonical run_dir、metadata、status、evaluation summary 和 prediction rows。
3. 新增 `docs/refactor/stage1_visual_eval_canonical_entrypoint.md`。
4. 更新 `WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`，登记新增入口、smoke、文档和本日志。

## 结果

已实现 P17a Visual eval canonical entrypoint thin slice。新入口 metadata 明确记录：

- `entrypoint = visual_eval_canonical`
- `feature_source = precomputed`
- `loaded_legacy_mlp = true`
- `scaler_enabled`
- `loads_real_checkpoint`
- `loads_real_vit = false`
- `training_started = false`
- `formal_training_migration = false`

验证结果：

- `python -m compileall scripts/run_stage1_visual_eval_canonical.py tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py` 通过。
- `tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py` 通过。
- `tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py` 通过。
- `tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py` 通过。
- `git diff --check` 通过。
- diff 边界审计确认未修改正式 streaming 训练入口、未新增 Bash launcher、未启动 ViT/transformers、未访问 `/data2`。

## 结论

P17a 的实现范围保持在 evaluation canonical thin slice，没有迁移正式训练入口，没有接真实 ViT，没有自动搜索 checkpoint/feature，也没有新增 Bash launcher。checkpoint path 与 scaler path 均停留在 Runtime/entrypoint 侧，不进入 provider/head/evaluator interface。

## 下一步方案

运行新增 smoke、P16j/P16k 相关回归 smoke、compileall 和 diff 审计；通过后提交并 push 到 `origin/refactor/stage1-route-audit`。
