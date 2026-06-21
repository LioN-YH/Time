# Stage 1 P16j Visual Small Loaded Legacy Path

日志日期：2026-06-21 15:14:12 CST

## 目的

扩展 `scripts/run_stage1_visual_small.py`，在默认 P15c mock path 不变的前提下，新增一个显式启用的 loaded legacy path，用 small fixture / tempfile tiny checkpoint 串联 P16c precomputed feature、P16d optional scaler、P16i checkpoint payload helper、P16a `LoadedTorchMLPRouterHeadAdapter`、`EvaluationInputAdapter` 和 Runtime artifact writer。

## 背景

P16a 已完成正式 Visual MLP RouterHead adapter 边界；P16c 已提供 `VisualPrecomputedFeatureProvider`；P16d 已提供 `LoadedFeatureScaler`；P16i 已提供 Runtime-side checkpoint payload helper。此前 Visual-specific small entrypoint 仍只走 script-local smoke adapter，没有读取 tiny checkpoint payload，也没有在 entrypoint 层集成 P16a/P16c/P16d/P16i。

本步仍严格限制为 small integrated rehearsal：不读取真实 checkpoint，不访问 `/data2`，不启动 ViT/transformers，不调用或迁移 `train_visual_router_online_streaming.py`，不新增 Bash launcher，不改变 TimeFuse small entrypoint，不把 checkpoint/scaler/run_dir 放入 P16a adapter interface。

## 操作

1. 修改 `scripts/run_stage1_visual_small.py`：
   - 新增 CLI：`--feature-source mock|precomputed`、`--visual-features-csv`、`--use-loaded-legacy-mlp`、`--router-checkpoint-payload`、`--scaler-state-json`、`--disable-scaler`。
   - 默认不传新参数时继续走 P15c `VisualMockFeatureProvider` + `SmokeOnlyVisualMLPAdapter`。
   - `--feature-source precomputed` 时使用 `VisualPrecomputedFeatureProvider`。
   - `--scaler-state-json` 且未 `--disable-scaler` 时执行 `LoadedFeatureScaler.transform(...)`，不执行 silent fit。
   - `--use-loaded-legacy-mlp` 和 `--router-checkpoint-payload` 同时传入时，Runtime/entrypoint 侧调用 P16i helper：`load_checkpoint_payload`、`extract_router_state_dict`、`load_router_state_dict`；动态导入 legacy `VisualMLPRouter`，strict load 后交给 P16a `LoadedTorchMLPRouterHeadAdapter`。
   - run metadata 记录 `feature_source`、`loaded_legacy_mlp`、`checkpoint_payload_source`、`scaler_enabled`、`loads_real_checkpoint=false`、`loads_real_vit=false`、`formal_visual_router_migration=false`、`p16i_helper_used` 和 `p16a_adapter_used`。
2. 更新 `tests/smoke/stage1_visual_small_entrypoint_smoke.py`：
   - 保持默认路径回归验证；
   - 调整旧静态 token 规则，允许 entrypoint 源码存在显式 loaded path 所需的 legacy `VisualMLPRouter` 引用，但默认运行时仍验证不读取真实 checkpoint、不启动 ViT、不访问 `/data2`。
3. 新增 `tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py`：
   - 在 tempfile 内创建 tiny checkpoint payload；
   - 使用 P16d `raw_visual_features.csv` 作为 precomputed small fixture，并传入 `scaler_state.json`；
   - 通过 subprocess 调用 loaded path；
   - 验证 canonical run_dir、metadata、evaluation summary、prediction rows、sample_key 保序、model_columns 对齐，以及 stdout/stderr 边界。
4. 新增 `docs/refactor/stage1_visual_small_loaded_legacy_path.md`。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

已通过的验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_small.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_small.py tests/smoke/stage1_visual_small_entrypoint_smoke.py tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
git diff --check
```

默认路径 smoke 结果：

- canonical run_dir、metadata/status、inputs、evaluation summary、prediction rows 和最小日志文件均存在；
- `VisualMockFeatureProvider`、ExpertBatch、RouterOutput sample_key/model_columns 对齐；
- stdout/stderr 未出现 `/data2`、正式训练入口、真实 checkpoint 或 ViT；
- generic small CLI 与 TimeFuse small CLI 未被修改。

loaded legacy path smoke 结果：

- CLI loaded path 完成，stdout/stderr 未触碰 `/data2`、ViT/transformers 或 streaming 正式入口；
- `run_status.json` 为 completed；
- `run_metadata.json` 记录 `feature_source=precomputed`、`loaded_legacy_mlp=true`、`scaler_enabled=true`、`checkpoint_payload_source=explicit_small_fixture`、`loads_real_checkpoint=false`、`loads_real_vit=false`、`formal_visual_router_migration=false`、`p16i_helper_used=true`、`p16a_adapter_used=true`；
- `evaluation_summary.json` 包含 hard/raw-soft MAE/MSE；
- `prediction_rows.csv` 保持 test split sample_key 顺序；
- `model_columns` 与 small ExpertBatch 对齐。

## 结论

P16j 已完成 Visual small entrypoint 的 loaded legacy path integrated rehearsal。现在 small entrypoint 能在显式 CLI 下串起 precomputed feature、可选 scaler、tiny checkpoint payload strict load、P16a adapter、Evaluator 和 canonical run_dir writer；默认 P15c mock path 仍保持可用。

该结果仍不代表正式 Visual Router 入口已迁移：没有读取真实 checkpoint，没有启动真实 ViT feature chain，没有处理 full-scale data，也没有修改正式 streaming 训练入口。

## 下一步方案

1. 小步提交并 push 到 `origin/refactor/stage1-route-audit`。
2. 后续可单独进入真实 checkpoint dry-run、real Visual feature chain 或正式 eval entrypoint migration plan。
3. 后续真实路径仍需保持 checkpoint/scaler/run_dir 不进入 P16a adapter interface，并继续用 small smoke 先验证边界。
