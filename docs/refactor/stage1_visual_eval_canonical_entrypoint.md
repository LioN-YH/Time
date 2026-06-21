# Stage 1 P17a Visual Eval Canonical Entrypoint

P17a 新增 `scripts/run_stage1_visual_eval_canonical.py`，目标是把 Visual Router evaluation 的未来正式入口先切出一个受限 thin slice：

```text
SampleManifest
-> VisualPrecomputedFeatureProvider
-> optional LoadedFeatureScaler
-> Runtime-loaded legacy VisualMLPRouter checkpoint payload
-> LoadedTorchMLPRouterHeadAdapter
-> ExpertBatch
-> EvaluationInputAdapter
-> Runtime artifact writer
-> canonical run_dir
```

## 边界

- 这是 Visual evaluation canonical entrypoint，不是训练入口。
- 不修改 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`。
- 不修改 `scripts/run_stage1_visual_small.py` 或 `scripts/run_stage1_timefuse_small.py` 的默认行为。
- CLI 必须显式传入 manifest、expert prediction、visual feature、checkpoint payload、output dir、run id、config 和 split。
- feature source 当前只支持 `precomputed` fixture；真实 VisualFeatureChain / ViT provider 留到后续。
- scaler 是可选 `LoadedFeatureScaler` transform；只有显式传入 `--scaler-state-json` 才执行，不做 silent fit。
- checkpoint loading 只属于 Runtime/entrypoint；adapter 只接收已加载 torch module。
- 默认 `--allow-real-checkpoint=false`，只允许 `tests/fixtures` 或 `/tmp` tiny payload；`/data2` checkpoint 始终禁止。

## CLI

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python scripts/run_stage1_visual_eval_canonical.py \
  --sample-manifest-csv tests/fixtures/stage1_real_derived_small/sample_manifest.csv \
  --expert-predictions-json tests/fixtures/stage1_real_derived_small/expert_predictions.json \
  --visual-features-csv tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv \
  --router-checkpoint-payload /tmp/tiny_legacy_visual_mlp_payload.pt \
  --output-dir /tmp/stage1_p17a_visual_eval \
  --run-id p17a_visual_eval_canonical \
  --config-name 96_48_S \
  --split-name test \
  --strict-checkpoint-load
```

可选：

- `--scaler-state-json PATH`：显式加载 scaler state 并 transform feature。
- `--allow-real-checkpoint`：允许非 fixture/tempfile checkpoint，但 P17a 仍禁止 `/data2`。

## Metadata

`run_metadata.json` 的 `visual_router` 段必须明确记录：

- `entrypoint = visual_eval_canonical`
- `feature_source`
- `loaded_legacy_mlp`
- `scaler_enabled`
- `loads_real_checkpoint`
- `loads_real_vit = false`
- `training_started = false`
- `formal_training_migration = false`

## Smoke

`tests/smoke/stage1_visual_eval_canonical_thin_slice_smoke.py` 使用：

- P13b `tests/fixtures/stage1_real_derived_small/sample_manifest.csv`
- P13b `tests/fixtures/stage1_real_derived_small/expert_predictions.json`
- P16c `tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv`
- tempfile tiny legacy `VisualMLPRouter` checkpoint payload

验证 canonical run_dir、`run_status=completed`、evaluation summary hard/raw-soft MAE/MSE、`selected_counts`、`model_columns`、prediction rows sample_key 保序、metadata 字段完整，并确认 stdout/stderr 不出现 `/data2`、ViT/transformers 或 streaming entrypoint。
