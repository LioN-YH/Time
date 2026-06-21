# Stage 1 P17d Visual Eval Real Artifact Manual Dry-Run

P17d 在 P17a/P17b/P17c 的 `scripts/run_stage1_visual_eval_canonical.py` 上补齐 manual real-artifact dry-run 口径。目标不是默认 CI 路径，也不是训练、ViT provider 或 full-scale 迁移，而是在用户显式提供真实 legacy `VisualMLPRouter` checkpoint payload、真实 precomputed visual feature CSV 和可选真实 scaler state JSON 时，通过同一个 canonical evaluation entrypoint 做 evaluation-only dry-run，并写出 canonical `run_dir`。

## 链路

```text
SampleManifest
-> guarded explicit visual feature CSV
-> VisualPrecomputedFeatureProvider
-> optional guarded scaler state JSON
-> LoadedFeatureScaler
-> guarded explicit checkpoint payload
-> Runtime-loaded legacy VisualMLPRouter
-> LoadedTorchMLPRouterHeadAdapter
-> ExpertBatch
-> EvaluationInputAdapter / Evaluator
-> Runtime artifact writer
-> canonical run_dir
```

## 触发方式

默认 P17a/P17b/P17c fixture smoke 行为保持不变。manual real-artifact dry-run 必须显式触发：

- CLI 路径：传入真实 `--router-checkpoint-payload`、`--visual-features-csv` 和可选 `--scaler-state-json`，并同时传入 `--manual-real-artifact-dryrun`。
- 环境变量路径：`tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py` 在检测到必要环境变量完整时执行 manual dry-run；缺少必要变量时打印 skip 并成功退出。

支持的环境变量：

- `STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD`
- `STAGE1_VISUAL_REAL_FEATURE_CSV`
- `STAGE1_VISUAL_REAL_SCALER_STATE_JSON`，可选
- `STAGE1_VISUAL_REAL_SAMPLE_MANIFEST_CSV`，可选，默认使用 P13b fixture
- `STAGE1_VISUAL_REAL_EXPERT_PREDICTIONS_JSON`，可选，默认使用 P13b fixture
- `STAGE1_VISUAL_REAL_OUTPUT_DIR`，可选，默认使用临时目录

如果 checkpoint、feature CSV 或 scaler JSON 位于 `/data2`，manual smoke 只会根据用户已提供的路径自动追加对应 allow flag，不搜索 `/data2`，不从旧 run_dir 推断 artifact。stdout 和 `run_metadata.json` 都会记录 allow 状态。

## Metadata

P17d 在 `run_metadata.json` 的 `visual_router` 段新增：

- `manual_real_artifact_dryrun`

manual dry-run 成功时还必须保留或记录：

- `entrypoint = visual_eval_canonical`
- `loads_real_checkpoint`
- `feature_source = precomputed`
- `allow_external_feature_path`
- `scaler_enabled`
- `allow_external_scaler_path`
- `loads_real_vit = false`
- `training_started = false`
- `formal_training_migration = false`

CLI stdout summary 同步输出 `manual_real_artifact_dryrun` 和各类 allow flag，便于人工 dry-run 时直接确认授权状态。

## Contract 检查

入口和 P17d smoke 覆盖以下 contract：

1. checkpoint config `input_dim` 必须与 `FeatureBatch.features.shape[1]` 一致。
2. checkpoint config `output_dim` 必须与 `ExpertBatch.model_columns` 数量一致。
3. feature CSV 必须覆盖 manifest ordered sample_keys。
4. `ExpertBatch` 必须覆盖同一批 ordered sample_keys。
5. `RouterOutput.sample_keys` 必须保序。
6. `prediction_rows.csv` 的 sample_key 必须保序。
7. evaluation summary 的所有 metrics 必须 finite。
8. scaler 如果启用，只执行 loaded-state transform，不做 fit 或 partial_fit；transform 后 shape、sample_key 顺序保持不变，features 必须 finite。

## Smoke

`tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py` 覆盖：

1. 未设置环境变量时，manual dry-run 按预期 skip，smoke 成功。
2. 在仓库受控临时目录构造 synthetic real-artifact checkpoint、external feature CSV 和 external scaler JSON，显式 allow 后 dry-run 成功，写出 canonical `run_dir`。
3. 成功路径检查 `manual_real_artifact_dryrun=true`、`loads_real_checkpoint=true`、external feature/scaler policy、no ViT、no training、no formal training migration、finite metrics 和 `prediction_rows.csv` sample_key 保序。
4. checkpoint `input_dim` 与 feature dim 不一致时 fail-fast，并输出清楚原因。
5. feature CSV 缺少 manifest sample_key 时 fail-fast，并输出清楚原因。

运行命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_manual_real_artifact_contract_smoke.py
```

## 明确不做

- 不迁移训练入口。
- 不修改 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`。
- 不启动 ViT 或 transformers。
- 不启动 full-scale。
- 不新增 Bash launcher。
- 不自动搜索 `/data2`。
- 不从旧 `run_dir` 自动推断 artifact。
- 不为旧版 `96_48_S` 输出 schema 写强兼容逻辑。
- 不修改 Visual small entrypoint 默认行为。
- 不修改 TimeFuse small entrypoint。
- 不把 run_dir、path 或 allow flags 下沉到 FeatureProvider 或 RouterHead adapter。
