# Stage 1 P21a Visual eval small launcher/config pack

日志日期：2026-06-21 23:27:35 CST

## 目的

为 Stage 1 Visual canonical eval 新增小样本 Python launcher 和 config pack，包装已跑通的
precomputed path 与 visual-chain-dryrun path，减少日常复现时的长 CLI 参数。

## 背景

P17a-P17d 已完成 Visual canonical eval entrypoint、checkpoint guard、external
feature/scaler guard 和 manual real-artifact dry-run contract。P19a/P19b/P20a 已完成
VisualFeatureChainRunner dry-run skeleton、guarded VisualVitEncoderProvider 和
visual-chain-dryrun 接入 canonical eval entrypoint。本步目标不是 full-scale，也不迁移训练入口，
只把两个 small canonical eval 路径封装为受控 launcher/config。

## 操作

1. 新增 `configs/stage1/visual_eval_small_precomputed.json` 和
   `configs/stage1/visual_eval_small_visual_chain.json`，只引用 repo 内 small fixture，默认
   `vit_provider_mode=injected-fake`、`manual_real_vit=false`、`safety.allow_data2=false`。
2. 新增 `scripts/run_stage1_visual_eval_small.py`，作为 thin wrapper 读取 JSON config、做
   `/data2` / `training_started` / `full_scale_run` / manual real ViT safety 校验，在 `/tmp`
   生成 tiny legacy `VisualMLPRouter` checkpoint payload，并调用
   `scripts/run_stage1_visual_eval_canonical.py`。
3. 新增 `tests/smoke/stage1_visual_eval_small_launcher_smoke.py`，覆盖 precomputed、
   visual-chain、both、dry-print 和 safety negative。
4. 新增 `docs/refactor/stage1_visual_eval_small_launcher.md`，记录 CLI、config、安全边界和验收口径。
5. 更新 `WORKSPACE_STRUCTURE.md`、`experiment_logs/README.md`、
   `docs/refactor/stage1_refactor_roadmap.md` 和
   `docs/refactor/stage1_entrypoint_migration_plan.md` 的 P21a 索引。

## 结果

已运行新增 smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_small_launcher_smoke.py
```

结果通过。覆盖项包括：

- `--mode precomputed` 成功写 canonical run_dir，`run_status.status=completed`，
  metadata `feature_source=precomputed`；
- `--mode visual-chain` 成功写 canonical run_dir，metadata
  `feature_source=visual-chain-dryrun`、`visual_chain_enabled=true`、`loads_real_vit=false`；
- `--mode both` 生成 `p21a_both_precomputed` 与 `p21a_both_visual_chain` 两个 run_dir；
- `--dry-print-command` 只打印 canonical command，不创建 run_dir；
- config 中出现 `/data2` path、`training_started=true` 或 `full_scale_run=true` 时 fail-fast。

随后已补跑验收与回归：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_small.py tests/smoke/stage1_visual_eval_small_launcher_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_boundary_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py
git diff --check
git diff -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```

结果均通过或无输出。P20a smoke 内部也覆盖了 P17a 默认 precomputed 回归；本日志仍额外显式运行了
P17a thin-slice smoke。`git diff -- visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
无输出，确认本步未修改正式 streaming 训练入口。

## 结论

P21a small launcher 已能一键复现 precomputed 和 visual-chain-dryrun 两条小样本 canonical
eval 路径。该 launcher 保持 thin wrapper 边界，不复制 evaluation 逻辑，不默认访问 `/data2`，
不默认加载真实 checkpoint 或真实 ViT，不启动训练或 full-scale。

## 下一步方案

小步提交并 push 到 `origin/refactor/stage1-route-audit`。真实 ViT artifact small/manual 接入、
full-scale 和训练入口迁移后续另起步骤。
