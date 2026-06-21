# Stage 1 P17c Visual Eval External Feature Guard

P17c 在 P17a/P17b `scripts/run_stage1_visual_eval_canonical.py` 上新增受控 external precomputed feature/scaler dry-run path。目标不是 ViT provider 迁移，也不是训练入口迁移，而是允许用户未来显式提供真实 precomputed visual feature CSV 和可选 scaler state JSON 时，仍通过同一个 canonical evaluation entrypoint 做 evaluation-only dry-run。

## 链路

```text
SampleManifest
-> guarded explicit visual feature CSV
-> VisualPrecomputedFeatureProvider
-> optional guarded scaler state JSON
-> LoadedFeatureScaler
-> guarded checkpoint payload
-> Runtime-loaded legacy VisualMLPRouter
-> LoadedTorchMLPRouterHeadAdapter
-> ExpertBatch
-> EvaluationInputAdapter / Evaluator
-> Runtime artifact writer
-> canonical run_dir
```

## CLI Guard

默认行为保持 P17a/P17b：

- `--visual-features-csv` 必须显式提供。
- 默认 feature CSV 只允许 `tests/fixtures` 或 `/tmp`。
- 非 fixture/tmp feature CSV 必须额外传入 `--allow-external-feature-path`。
- 如果传入 `--scaler-state-json`，默认 scaler JSON 也只允许 `tests/fixtures` 或 `/tmp`。
- 非 fixture/tmp scaler JSON 必须额外传入 `--allow-external-scaler-path`。
- `/data2` feature/scaler path 只有显式 external allow 后才允许进入后续文件读取阶段，且仍只做 evaluation-only dry-run。
- `--feature-path-label` 和 `--scaler-path-label` 只用于 metadata 脱敏说明，不参与读取、搜索或推断。

示例：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python scripts/run_stage1_visual_eval_canonical.py \
  --sample-manifest-csv tests/fixtures/stage1_real_derived_small/sample_manifest.csv \
  --expert-predictions-json tests/fixtures/stage1_real_derived_small/expert_predictions.json \
  --visual-features-csv /path/to/precomputed_visual_features.csv \
  --scaler-state-json /path/to/scaler_state.json \
  --router-checkpoint-payload /tmp/tiny_or_user_provided_visual_mlp_payload.pt \
  --output-dir /tmp/stage1_p17c_visual_eval \
  --run-id p17c_external_feature_dry_run \
  --config-name 96_48_S \
  --split-name test \
  --strict-checkpoint-load \
  --allow-external-feature-path \
  --feature-path-label "manual:user-provided-feature-csv" \
  --allow-external-scaler-path \
  --scaler-path-label "manual:user-provided-scaler-state"
```

如果 checkpoint 同时是非 fixture/tmp 或 `/data2` 路径，仍需遵守 P17b 的 `--allow-real-checkpoint` 与 `--allow-external-checkpoint-path`。

## Runtime Helper

`time_router/runtime/visual_eval_feature_guard.py` 提供：

- `authorize_visual_eval_feature_path(...)`
- `authorize_visual_eval_scaler_path(...)`
- `is_fixture_or_tempfile_visual_eval_artifact(...)`
- `VisualEvalPathPolicy`

该 helper 只做 path policy，不检查文件存在，不读取 CSV/JSON 内容，不自动搜索 feature/scaler，不从旧 `run_dir` 推断 artifact。入口拿到 policy 后，才明确调用 `VisualPrecomputedFeatureProvider` 或 `LoadedFeatureScaler.from_json(...)`。

## Contract 检查

P17c 在入口侧对读取后的内存对象增加轻量检查：

- feature CSV 必须覆盖 manifest ordered sample_keys。
- `FeatureBatch.sample_keys` 必须保持 manifest 顺序。
- feature dim 必须大于 0。
- feature dtype 必须是 `float32`。
- features 必须全部 finite。
- `VisualPrecomputedFeatureProvider` 不接收 `run_dir`。
- 只有显式传入 scaler JSON 时才执行 transform。
- scaler 不执行 fit 或 partial_fit。
- scaler transform 后 sample_key 顺序不变。
- scaler transform 后 shape 不变。
- scaler transform 后 features 必须全部 finite。

## Metadata

`run_metadata.json` 的 `visual_router` 段记录：

- `feature_source = precomputed`
- `feature_path_policy`
- `feature_path_label`
- `allow_external_feature_path`
- `scaler_path_policy`
- `scaler_path_label`
- `allow_external_scaler_path`
- `scaler_enabled`
- `scaler_fit_performed = false`
- `loads_real_vit = false`
- `training_started = false`
- `formal_training_migration = false`

`feature_lineage` 和 `scaler` 中保留 Runtime 内部读取所需的 path reference、checksum 与 helper 细节；feature/scaler path 与 allow flag 不进入 `FeatureProvider` 或 `LoadedTorchMLPRouterHeadAdapter` interface。

## Smoke

`tests/smoke/stage1_visual_eval_canonical_external_feature_guard_smoke.py` 覆盖：

1. 默认 fixture feature CSV + no scaler：不传 external allow，CLI 成功，metadata `allow_external_feature_path=false` 且 `scaler_enabled=false`。
2. 仓库根目录下非 fixture/tmp feature CSV：不传 `--allow-external-feature-path`，CLI fail-fast，并提示 external feature 授权。
3. 非 fixture/tmp feature CSV 显式授权：CLI 成功，metadata `feature_path_policy=explicit_external_feature_authorized`。
4. 非 fixture/tmp scaler JSON：未传 `--allow-external-scaler-path` 时失败，授权后成功，metadata `scaler_enabled=true` 且 `scaler_fit_performed=false`。
5. `/data2` feature/scaler path policy：只调用 guard helper，不创建 `/data2` 文件，不读取内容；未授权失败，授权后 policy 通过。

可选 manual dry-run 由环境变量控制，默认跳过：

- `STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD`
- `STAGE1_VISUAL_REAL_FEATURE_CSV`
- `STAGE1_VISUAL_REAL_SCALER_STATE_JSON`

manual dry-run 仍只做 evaluation-only，不启动 ViT、训练或 full-scale。若 checkpoint 或 feature/scaler 位于 `/data2`，manual smoke 会追加对应 allow flag，并通过 stdout/metadata 留痕。

## 明确不做

- 不迁移训练入口。
- 不修改 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`。
- 不启动 ViT 或 transformers。
- 不启动 full-scale。
- 不新增 Bash launcher。
- 不自动搜索 `/data2`。
- 不从旧 `run_dir` 自动推断 feature/scaler。
- 不为旧版 `96_48_S` 输出 schema 写强兼容逻辑。
- 不修改 P15c/P16j Visual small entrypoint 默认行为。
- 不修改 TimeFuse small entrypoint。
- 不把 run_dir、path 或 allow flags 下沉到 provider/head adapter。
