# Stage 1 P17d Visual canonical eval real-artifact manual dry-run

日志日期：2026-06-21 18:10:24 CST

## 目的

在 P17a/P17b/P17c 的 Visual canonical evaluation entrypoint 基础上，新增 real-artifact manual dry-run 口径。目标是在用户显式提供真实 legacy `VisualMLPRouter` checkpoint payload、真实 precomputed visual feature CSV 和可选真实 scaler state JSON 时，能通过新的 canonical eval entrypoint 执行 evaluation-only dry-run，并写出 canonical `run_dir`。

## 背景

P17a 已建立 `scripts/run_stage1_visual_eval_canonical.py`，串联 `SampleManifest -> VisualPrecomputedFeatureProvider -> optional LoadedFeatureScaler -> Runtime-loaded legacy VisualMLPRouter -> LoadedTorchMLPRouterHeadAdapter -> ExpertBatch -> EvaluationInputAdapter -> Runtime artifact writer`。P17b/P17c 已补充真实 checkpoint 和 external feature/scaler 的 path guard，但缺少一个明确的 manual real-artifact dry-run contract smoke，用来证明三类真实 artifact 可以在显式授权下同时进入 canonical evaluation entrypoint。

## 操作

1. 修改 `scripts/run_stage1_visual_eval_canonical.py`：
   - 新增 `--manual-real-artifact-dryrun`；
   - 在 `run_metadata.json` 的 `visual_router` 段写入 `manual_real_artifact_dryrun`；
   - 在 stdout summary 中写入 `manual_real_artifact_dryrun` 和 checkpoint/feature/scaler allow flag；
   - 对 evaluation summary metrics 增加 finite 校验。
2. 新增 `tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py`：
   - 环境变量不完整时按预期 skip；
   - 在仓库受控临时目录构造 synthetic real-artifact checkpoint、external feature CSV 和 external scaler JSON；
   - 显式 allow 后运行 canonical eval entrypoint，并检查 canonical `run_dir`、metadata、finite metrics 和 `prediction_rows.csv` sample_key 保序；
   - 负向覆盖 checkpoint `input_dim` 与 feature dim 不一致；
   - 负向覆盖 feature CSV 缺少 manifest sample_key。
3. 新增 `docs/refactor/stage1_visual_eval_real_artifact_manual_dryrun.md`，记录触发方式、环境变量、metadata、contract、smoke 和明确不做范围。
4. 更新 `WORKSPACE_STRUCTURE.md`，登记 P17d 文档、入口脚本新 flag 和 P17d smoke。
5. 更新 `experiment_logs/README.md` 总览表。

## 结果

已完成新增入口 metadata/stdout 口径、P17d smoke、refactor 文档和结构索引。

已执行验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
git diff --check
rg -n "train_visual_router_online_streaming|ViTModel|AutoImageProcessor|transformers|launch_|nohup|setsid|tmux|/data2|torch\\.load" scripts/run_stage1_visual_eval_canonical.py tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py docs/refactor/stage1_visual_eval_real_artifact_manual_dryrun.md experiment_logs/2026-06-21_stage1_p17d_visual_eval_real_artifact_manual_dryrun.md
```

当前结果：

- compileall 通过。
- P17d smoke 通过：环境变量不完整时 skip；synthetic real-artifact manual dry-run 成功；checkpoint input_dim mismatch fail-fast；feature CSV 缺 sample_key fail-fast。
- P17a thin slice smoke 通过。
- P17b checkpoint guard smoke 通过。
- P17c external feature/scaler guard smoke 通过。
- P16j Visual small loaded legacy path smoke 通过。
- P16k Visual small loaded path artifact parity smoke 通过。
- `git diff --check` 通过。
- 边界审计只命中文档、注释、guard 字符串和新 smoke 中的 `/data2` 授权判断；没有新增 ViT/transformers 调用、Bash launcher、后台命令或正式 streaming 训练入口修改。

## 结论

P17d 已把 manual real-artifact dry-run 从 P17b/P17c 的分散可选路径收束为明确 contract：默认 fixture smoke 行为不变，只有显式 CLI flag 或完整环境变量才会运行 manual dry-run；路径授权仍停留在 entrypoint/runtime 侧，不下沉到 `FeatureProvider` 或 RouterHead adapter；本步未迁移训练入口、未启动 ViT、未启动 full-scale、未新增 Bash launcher，也未自动搜索 `/data2`。

## 下一步方案

1. 小步提交，并 push 到 `origin/refactor/stage1-route-audit`。
2. 后续 ViT provider 迁移、训练入口迁移和 full-scale 正式运行另起步骤；不要复用 P17d smoke 作为默认真实 `/data2` 自动发现路径。
