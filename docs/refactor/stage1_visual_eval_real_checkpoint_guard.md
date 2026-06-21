# Stage 1 P17b Visual Eval Real Checkpoint Guard

P17b 在 P17a `scripts/run_stage1_visual_eval_canonical.py` 上新增受控 real-checkpoint dry-run path。目标不是默认 CI 路径，也不是训练迁移，而是允许用户未来显式提供真实 legacy `VisualMLPRouter` checkpoint payload 时，仍通过同一个 canonical evaluation entrypoint 做 evaluation-only dry-run。

## 链路

```text
SampleManifest
-> precomputed Visual feature fixture / explicit feature CSV
-> optional LoadedFeatureScaler
-> explicit checkpoint payload
-> guarded checkpoint authorization
-> Runtime-loaded legacy VisualMLPRouter
-> LoadedTorchMLPRouterHeadAdapter
-> ExpertBatch
-> EvaluationInputAdapter / Evaluator
-> Runtime artifact writer
-> canonical run_dir
```

## CLI Guard

默认行为保持 P17a：

- `--router-checkpoint-payload` 必须显式提供。
- 默认只允许 `tests/fixtures` 或 `/tmp` 下的 tiny checkpoint payload。
- 非 fixture/tmp checkpoint 必须额外传入 `--allow-real-checkpoint`。
- `/data2` checkpoint path 必须同时传入 `--allow-real-checkpoint` 和 `--allow-external-checkpoint-path`。
- `--checkpoint-path-label` 只用于 metadata 脱敏说明，不参与读取、搜索或推断。

示例：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python scripts/run_stage1_visual_eval_canonical.py \
  --sample-manifest-csv tests/fixtures/stage1_real_derived_small/sample_manifest.csv \
  --expert-predictions-json tests/fixtures/stage1_real_derived_small/expert_predictions.json \
  --visual-features-csv /path/to/explicit_visual_features.csv \
  --router-checkpoint-payload /path/to/legacy_visual_mlp_payload.pt \
  --output-dir /tmp/stage1_p17b_visual_eval \
  --run-id p17b_real_checkpoint_dry_run \
  --config-name 96_48_S \
  --split-name test \
  --strict-checkpoint-load \
  --allow-real-checkpoint \
  --checkpoint-path-label "manual:user-provided-legacy-payload"
```

如果 checkpoint 位于 `/data2`，还需要追加：

```bash
--allow-external-checkpoint-path
```

## Runtime Helper

`time_router/runtime/visual_eval_checkpoint_guard.py` 提供：

- `authorize_visual_eval_checkpoint_path(...)`
- `is_fixture_or_tempfile_checkpoint(...)`
- `is_data2_path(...)`
- `CheckpointPathPolicy`

该 helper 只做 path policy，不调用 `torch.load`，不检查文件存在，不读取 `/data2` 内容，不自动搜索 checkpoint，不从旧 `run_dir` 推断 checkpoint。入口拿到 policy 后才继续走 Runtime-side `load_checkpoint_payload(...)`。

## Metadata

`run_metadata.json` 的 `visual_router` 段记录：

- `loads_real_checkpoint`
- `checkpoint_payload_source`
- `checkpoint_payload_sha256`
- `checkpoint_path_policy`
- `checkpoint_path_label`
- `allow_real_checkpoint`
- `allow_external_checkpoint_path`
- `loads_real_vit = false`
- `training_started = false`
- `formal_training_migration = false`

`head_lineage` 中保留 Runtime 内部读取所需的 `checkpoint_payload_path` 与 helper 细节；checkpoint path 不进入 `FeatureProvider` 或 `LoadedTorchMLPRouterHeadAdapter` interface。

## Smoke

`tests/smoke/stage1_visual_eval_canonical_real_checkpoint_guard_smoke.py` 覆盖：

1. 默认 tempfile tiny checkpoint：不传 `--allow-real-checkpoint`，CLI 成功，metadata `loads_real_checkpoint=false`。
2. 仓库根目录下非 fixture/tmp checkpoint：不传 `--allow-real-checkpoint`，CLI 在 `torch.load` 前失败，并提示必须显式授权。
3. `/data2` checkpoint path policy：只调用 guard helper，不创建 `/data2` 文件，不读取内容；缺少 `--allow-external-checkpoint-path` 时失败，双重授权时 policy 通过。

可选 manual dry-run 由环境变量控制，默认跳过：

- `STAGE1_VISUAL_REAL_CHECKPOINT_PAYLOAD`
- `STAGE1_VISUAL_REAL_FEATURE_CSV`
- `STAGE1_VISUAL_REAL_SCALER_STATE_JSON`

manual dry-run 仍只做 evaluation-only，不启动 ViT、训练或 full-scale。

## 明确不做

- 不迁移训练入口。
- 不修改 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`。
- 不启动 full-scale。
- 不接真实 ViT。
- 不自动搜索 `/data2`。
- 不新增 Bash launcher。
- 不为旧版 `96_48_S` 输出 schema 写强兼容逻辑。
- 不修改 P15c/P16j Visual small entrypoint 默认行为。
- 不修改 TimeFuse small entrypoint。
- 不把真实 checkpoint 读取逻辑下沉到 RouterHead adapter。
