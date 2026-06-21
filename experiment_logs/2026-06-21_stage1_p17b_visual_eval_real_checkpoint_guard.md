# Stage 1 P17b Visual Eval Real Checkpoint Guard

日志日期：2026-06-21 17:15:29 CST

## 目的

在 P17a Visual canonical evaluation entrypoint 上新增受控 real-checkpoint dry-run path，让用户未来显式提供真实 legacy `VisualMLPRouter` checkpoint payload 时，可以通过同一个 canonical eval entrypoint 做 evaluation-only dry-run。

## 背景

P17a 已串通 `SampleManifest -> VisualPrecomputedFeatureProvider -> optional LoadedFeatureScaler -> Runtime-loaded legacy VisualMLPRouter checkpoint payload -> LoadedTorchMLPRouterHeadAdapter -> ExpertBatch -> EvaluationInputAdapter -> Runtime artifact writer -> canonical run_dir`。但 P17a 默认只允许 fixture 或 `/tmp` tiny checkpoint，并且将 `/data2` checkpoint 彻底禁止。本步需要保持默认行为不变，同时新增更明确的真实 checkpoint 授权入口。

## 操作

1. 新增 `time_router/runtime/visual_eval_checkpoint_guard.py`，实现 `authorize_visual_eval_checkpoint_path(...)`、`CheckpointPathPolicy`、`is_fixture_or_tempfile_checkpoint(...)` 和 `is_data2_path(...)`。
2. 更新 `time_router/runtime/__init__.py` 导出 checkpoint guard helper。
3. 更新 `scripts/run_stage1_visual_eval_canonical.py`：
   - 新增 `--allow-external-checkpoint-path`；
   - 新增 `--checkpoint-path-label`；
   - 保持 `--allow-real-checkpoint` 默认关闭；
   - checkpoint path 先走 guard，再由 Runtime loader 读取；
   - metadata 增加 `checkpoint_payload_source`、`checkpoint_payload_sha256`、`checkpoint_path_policy`、`checkpoint_path_label`、`allow_real_checkpoint`、`allow_external_checkpoint_path`、`loads_real_checkpoint`、`loads_real_vit=false`、`training_started=false`、`formal_training_migration=false`。
4. 新增 `tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py`，覆盖默认 tempfile tiny checkpoint、非 fixture/tmp 未授权失败、`/data2` path guard 双重授权。
5. 新增 `docs/refactor/stage1_visual_eval_real_checkpoint_guard.md` 记录 CLI、metadata、helper、smoke 和不做范围。
6. 更新 `WORKSPACE_STRUCTURE.md` 与 `experiment_logs/README.md` 追踪新增长期文件。

## 结果

代码已完成 P17b guard path 改造。默认 tiny checkpoint path 仍不需要 `--allow-real-checkpoint`；非 fixture/tmp checkpoint 需要显式 `--allow-real-checkpoint`；`/data2` checkpoint path 需要同时显式 `--allow-real-checkpoint` 与 `--allow-external-checkpoint-path`。`/data2` smoke 只调用 guard helper，不创建文件、不执行 `torch.load`。

验证结果：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
# 通过：Stage 1 P17a Visual eval canonical thin slice smoke 全部通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
# 通过：Stage 1 P17b Visual eval real-checkpoint guard smoke 全部通过；manual real-checkpoint dry-run 因未设置环境变量按预期跳过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py
# 通过：Stage 1 P16j Visual small loaded legacy path smoke 全部通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py
# 通过：Stage 1 P16k Visual small loaded path artifact parity smoke 全部通过

/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py time_router/runtime/visual_eval_checkpoint_guard.py tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
# 通过

git diff --check
# 通过

rg -n "train_visual_router_online_streaming|ViTModel|AutoImageProcessor|launch_|nohup|setsid|tmux|/data2|torch.load" scripts/run_stage1_visual_eval_canonical.py time_router/runtime/visual_eval_checkpoint_guard.py tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py docs/refactor/stage1_visual_eval_real_checkpoint_guard.md
# 仅命中允许的注释、文档、helper policy 和 smoke guard 文案；未修改正式 streaming 训练入口，未新增 Bash launcher，默认测试不读取真实 checkpoint、不访问 /data2 文件内容。
```

## 结论

P17b 将真实 checkpoint dry-run 能力限制在显式 CLI 和 Runtime metadata 边界内，没有把 checkpoint path、allow flag 或 run_dir 下沉到 `FeatureProvider` / `RouterHead` adapter。默认 CI/smoke 仍使用 tiny checkpoint，不读取真实 checkpoint，不启动 ViT、训练或 full-scale。

## 下一步方案

1. 运行 P17a 原 smoke、新增 P17b guard smoke、P16j/P16k 相关 smoke、compileall 和 diff 检查。
2. 若验证通过，补充本日志结果并提交推送到 `origin/refactor/stage1-route-audit`。
3. 后续如需真实 artifact manual dry-run，必须由用户显式提供 checkpoint、feature CSV 和可选 scaler state 环境变量或 CLI 路径。
