# Stage 1 P16k Visual Small Loaded Path Artifact Parity

## 1. 目的

P16k 在 P16j 之后新增 Visual small entrypoint 内部两条路径的 artifact parity smoke：

- 默认 mock path：`VisualMockFeatureProvider -> SmokeOnlyVisualMLPAdapter`；
- loaded legacy path：P16c precomputed feature fixture -> tempfile tiny checkpoint payload -> legacy `VisualMLPRouter` -> P16a `LoadedTorchMLPRouterHeadAdapter`。

本步只比较 canonical run_dir 的 artifact 契约，不比较两条路径的 metrics 数值优劣。默认 mock path 和 loaded legacy path 可以产生不同 hard/raw-soft 数值，但共同结构、共同 schema、`model_columns` 和 `sample_key` 顺序必须保持一致。

## 2. 检查范围

新增 smoke：`tests/smoke/stage1_visual_small_loaded_path_artifact_parity_smoke.py`。

它在 tempfile output root 下同时运行：

```text
scripts/run_stage1_visual_small.py 默认 mock path
scripts/run_stage1_visual_small.py --feature-source precomputed --use-loaded-legacy-mlp --router-checkpoint-payload <tempfile>
```

loaded path 使用仓库内 P16c `stage1_visual_precomputed_small/visual_embeddings.csv` fixture 和 tempfile tiny checkpoint payload。本步选择 no-scaler 简化；P16j loaded legacy path smoke 已覆盖 scaler-enabled 组合。

## 3. Artifact 契约

两条路径都必须写出 completed canonical run_dir，并包含：

- `run_metadata.json`
- `run_status.json`
- `inputs/sample_manifest_ref.json`
- `inputs/split_summary.json`
- `evaluation/evaluation_summary.json`
- `predictions/prediction_rows.csv`
- `logs/visual_small_entrypoint.log`

共同 metadata 必须包含 `branch_name`、`feature_source`、`loaded_legacy_mlp`、`loads_real_checkpoint=false`、`loads_real_vit=false` 和 `formal_visual_router_migration=false`。默认 path 必须记录 `loaded_legacy_mlp=false`；loaded path 必须记录 `checkpoint_payload_source=explicit_small_fixture`、`scaler_enabled=false`、`p16i_helper_used=true` 和 `p16a_adapter_used=true`。

`evaluation_summary.json` 共同包含 `sample_count`、`model_columns`、`selected_counts`，以及 `hard_mae`、`hard_mse`、`raw_soft_mae`、`raw_soft_mse`、`mean_entropy` 和 `mean_max_weight`。

`prediction_rows.csv` 共同包含 `sample_key`、`split`、`selected_model`、`selected_index`、`y_true`、`y_pred`、hard/raw-soft 逐样本指标和 weight diagnostics。两条路径的行数、`sample_key` 顺序、`split` 列和 `model_columns` 必须一致；`selected_model` 必须属于 `model_columns`；指标字段必须是有限数值。

## 4. 明确不做

P16k 仍然只是 small fixture rehearsal：

- 不读取真实 checkpoint；
- 不访问 `/data2`；
- 不启动 ViT、`AutoImageProcessor` 或 transformers；
- 不迁移 `train_visual_router_online_streaming.py`；
- 不修改 TimeFuse small entrypoint；
- 不新增 Bash launcher；
- 不把 checkpoint、scaler 或 run_dir 放进 P16a adapter；
- 不为了 parity 强行改 metrics 数值；
- 不声称正式 Visual Router 已迁移完成。

## 5. 后续

P16k 只锁定 Visual small entrypoint 两条路径的 artifact schema 不分叉。下一步才考虑真实 checkpoint dry-run 或 real Visual feature chain，并且仍应保持 Runtime checkpoint/scaler/ViT/device 责任与 RouterHead adapter 分层。
