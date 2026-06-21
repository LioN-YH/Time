# Stage 1 P20a Visual eval visual-chain path

日志日期：2026-06-21 22:31:45 CST

## 目的

把 P19a `VisualFeatureChainRunner` 和 P19b guarded `VisualVitEncoderProvider` 接入 P17 Visual canonical eval entrypoint 的小样本 dry-run feature source，证明 visual-chain 基础设施能进入 canonical eval 后半段。

## 背景

P17a-P17d 已完成 precomputed feature、guarded checkpoint、external feature/scaler 和 manual real-artifact dry-run contract。P19a/P19b 已分别提供 visual-chain runner 和默认不导入 transformers 的 guarded ViT provider。但在本步骤前，`scripts/run_stage1_visual_eval_canonical.py` 仍只支持 `--feature-source precomputed`，visual-chain 只能停留在独立 smoke 中。

## 操作

1. 修改 `scripts/run_stage1_visual_eval_canonical.py`：
   - 将 `--feature-source` 扩展为 `precomputed` 和 `visual-chain-dryrun`。
   - 新增 `--raw-window-json`、`--visual-chain-mode`、`--vit-provider-mode`、`--manual-real-vit`、`--vit-model-path`、`--vit-processor-path`、`--allow-real-vit` 和 `--allow-external-vit-path`。
   - 新增显式 raw-window JSON provider、identity pre-image transform、tiny pseudo-image transform、identity resize policy、injected fake ViT processor/model 和 mean patch pooling。
   - 默认 visual-chain path 通过 `VisualVitEncoderProvider(processor=InjectedFakeVitProcessor, model=InjectedFakeVitModel)` 注入 fake provider，不调用 guarded real builder，不导入 transformers。
   - manual real ViT 分支要求 `--manual-real-vit`、`--allow-real-vit` 和 `--vit-model-path`，`/data2` path 还需要 `--allow-external-vit-path`。
   - visual-chain `FeatureBatch` 继续复用 P17 既有 checkpoint/head/eval/runtime writer 链路。
2. 新增 `tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py`：
   - 先调用 P17a smoke，确认默认 precomputed path 回归通过。
   - 使用 P13b manifest/expert JSON、P19a raw window fixture 和 tempfile tiny checkpoint payload 运行 visual-chain dry-run。
   - 覆盖 raw-window-json 缺失和 raw-window fixture 缺少 manifest sample_key 两个 fail-fast 负例。
3. 新增文档 `docs/refactor/stage1_visual_eval_visual_chain_path.md`，记录 CLI、metadata、smoke 和不做范围。
4. 更新 `WORKSPACE_STRUCTURE.md`、`docs/refactor/stage1_refactor_roadmap.md` 和 `experiment_logs/README.md`。

## 结果

- 临时 visual-chain CLI 运行成功，`run_status=completed`，stdout 摘要包含 `feature_source=visual-chain-dryrun`、`loads_real_vit=false`、`full_scale_run=false`。
- 验证已通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_time_router_public_api_smoke_scaffold_cleanup_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py
```

- P20a smoke 输出确认 P17a 默认 precomputed smoke 仍通过，visual-chain canonical run_dir、metadata、summary、prediction rows 成立，两个 raw-window 负例均 fail-fast。
- P19a/P19b/P17b/P17c/P17d/P18b 回归 smoke 均通过；P17b/P17c/P19b 的 manual real artifact/ViT 分支因未设置环境变量按预期 skip。
- `git diff --check` 通过；`visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 无 diff；diff 文件名审计未发现新增 launcher 或修改 TimeFuse small entrypoint。

## 结论

P20a 已把 small visual-chain dry-run feature source 接入 Visual canonical eval entrypoint。默认路径仍保持 P17 precomputed 行为；新增路径只在显式 `--feature-source visual-chain-dryrun` 时启用，不默认导入 transformers，不加载真实 ViT，不访问 `/data2`，不启动训练，也不修改正式 streaming 训练入口。

## 下一步方案

小步提交并 push 到 `origin/refactor/stage1-route-audit`。后续真实 ViT artifact small/manual 接入、正式训练入口迁移和 full-scale 运行另起步骤。
