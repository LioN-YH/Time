# Stage 1 P17c Visual Eval External Feature Guard

日志日期：2026-06-21 17:42:00 CST

## 目的

在 P17a/P17b Visual canonical evaluation entrypoint 上新增受控 external precomputed feature/scaler dry-run path，让用户未来显式提供真实 precomputed visual feature CSV 和可选 scaler state JSON 时，可以通过同一个 canonical eval entrypoint 做 evaluation-only dry-run。

## 背景

P17a 已串通 Visual canonical evaluation thin slice，P17b 已新增 checkpoint path guard。当前入口仍只安全支持 fixture/tmp 风格的 precomputed visual feature/scaler。本步需要在不启动 ViT、不训练、不跑 full-scale、不迁移 streaming 训练入口的前提下，把真实 precomputed feature/scaler artifact 的显式授权入口补齐。

## 操作

1. 新增 `time_router/runtime/visual_eval_feature_guard.py`，实现 `authorize_visual_eval_feature_path(...)`、`authorize_visual_eval_scaler_path(...)`、`VisualEvalPathPolicy` 和 `is_fixture_or_tempfile_visual_eval_artifact(...)`。
2. 更新 `time_router/runtime/__init__.py` 导出 feature/scaler path guard helper。
3. 更新 `scripts/run_stage1_visual_eval_canonical.py`：
   - 新增 `--allow-external-feature-path`、`--allow-external-scaler-path`；
   - 新增 `--feature-path-label`、`--scaler-path-label`；
   - feature CSV 和 scaler JSON 先走 guard，再分别由 `VisualPrecomputedFeatureProvider` 与 `LoadedFeatureScaler.from_json(...)` 明确读取；
   - 去除 feature CSV 的默认 `/data2` 硬禁止，改为外部路径显式授权策略；
   - metadata 增加 `feature_path_policy`、`feature_path_label`、`allow_external_feature_path`、`scaler_path_policy`、`scaler_path_label`、`allow_external_scaler_path`、`scaler_fit_performed=false`；
   - 增加 feature/scaler 读取后的轻量 contract 检查，覆盖 sample_key 保序、feature dim、dtype、finite、scaler transform 后顺序与 shape 不变。
4. 新增 `tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py`，覆盖默认 fixture feature/no scaler、非 fixture/tmp feature 未授权失败、授权成功、非 fixture/tmp scaler 未授权失败/授权成功和 `/data2` helper policy。
5. 新增 `docs/refactor/stage1_visual_eval_external_feature_guard.md` 记录 CLI、metadata、helper、contract、smoke 和不做范围。
6. 更新 `WORKSPACE_STRUCTURE.md` 与 `experiment_logs/README.md` 追踪新增长期文件。

## 结果

代码已完成 P17c guarded external feature/scaler path 改造。默认 fixture feature CSV + no scaler 行为保持可用；非 fixture/tmp feature CSV 必须显式 `--allow-external-feature-path`；非 fixture/tmp scaler JSON 必须显式 `--allow-external-scaler-path`；`/data2` feature/scaler path 默认不读取，只有显式授权后才允许进入 evaluation-only dry-run 文件读取阶段。

已执行的验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py time_router/runtime/visual_eval_feature_guard.py time_router/runtime/__init__.py tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
# 通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
# 通过：Stage 1 P17c Visual eval external feature/scaler guard smoke 全部通过；manual dry-run 因未设置环境变量按预期跳过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
# 通过：Stage 1 P17a Visual eval canonical thin slice smoke 全部通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
# 通过：Stage 1 P17b Visual eval real-checkpoint guard smoke 全部通过；manual real-checkpoint dry-run 因未设置环境变量按预期跳过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py
# 通过：Stage 1 P16j Visual small loaded legacy path smoke 全部通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
# 通过：Stage 1 P16k Visual small loaded path artifact parity smoke 全部通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py time_router/runtime/visual_eval_feature_guard.py time_router/runtime/__init__.py tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
# 通过

git diff --check
# 通过

rg -n "ViTModel|AutoImageProcessor|transformers|train_visual_router_online_streaming|nohup|setsid|tmux|launch_|/data2|torch\\.load|run_dir" scripts/run_stage1_visual_eval_canonical.py time_router/runtime/visual_eval_feature_guard.py tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py docs/refactor/stage1_visual_eval_external_feature_guard.md
# 仅命中允许的注释、文档、helper policy、Runtime artifact writer 和 smoke guard 文案；未修改正式 streaming 训练入口，未新增 Bash launcher，默认测试不读取真实 /data2 feature/scaler/checkpoint 内容，未启动 ViT/训练/full-scale。
```

## 结论

P17c 将 external precomputed feature/scaler 接入能力限制在显式 CLI 与 Runtime metadata 边界内，没有把 run_dir、path 或 allow flag 下沉到 `VisualPrecomputedFeatureProvider`、`LoadedFeatureScaler` 或 `LoadedTorchMLPRouterHeadAdapter` interface。默认测试不读取真实 `/data2` feature、scaler 或 checkpoint 内容，不启动 ViT、训练或 full-scale。

## 下一步方案

1. 小步提交并推送到 `origin/refactor/stage1-route-audit`。
2. 后续如需真实 artifact manual dry-run，必须由用户显式提供 checkpoint、feature CSV 和可选 scaler state 环境变量或 CLI 路径。
3. ViT provider 迁移、训练入口迁移和 full-scale 正式运行继续另起步骤。
