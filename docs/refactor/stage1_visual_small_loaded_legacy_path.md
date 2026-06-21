# Stage 1 P16j Visual Small Loaded Legacy Path

## 1. 目的

P16j 扩展 `scripts/run_stage1_visual_small.py`，给 Visual-specific small entrypoint 增加一个显式启用的 loaded legacy path。该路径把前面 P16a/P16c/P16d/P16i 的单组件 smoke 串成一次 integrated rehearsal：

```text
SampleManifest
-> VisualPrecomputedFeatureProvider 或默认 VisualMockFeatureProvider
-> optional LoadedFeatureScaler
-> P16i checkpoint payload helper strict load legacy VisualMLPRouter
-> LoadedTorchMLPRouterHeadAdapter
-> EvaluationInputAdapter / Evaluator
-> Runtime artifact writer / canonical run_dir
```

这比单组件 smoke 更接近正式 Visual evaluation path，但仍只跑 small fixture 和 tempfile tiny checkpoint。

## 2. 边界

- checkpoint loading 属于 Runtime/entrypoint。P16a adapter 仍只接收已加载的 `torch.nn.Module`，不接 checkpoint path、scaler path 或 run_dir。
- scaler 是可选 `FeatureTransform`。只有显式传入 `--scaler-state-json` 且未 `--disable-scaler` 时才执行 `LoadedFeatureScaler`，不会 silent fit。
- precomputed feature 是 small/debug implementation，不是正式 Visual provider 的唯一或长期路径。
- 默认不传新参数时，entrypoint 仍走 P15c mock feature + script-local `SmokeOnlyVisualMLPAdapter`。
- P16j 不读取真实 checkpoint，不访问 `/data2`，不启动 ViT/transformers，不调用或迁移 `train_visual_router_online_streaming.py`，不声称正式 Visual Router 已迁移完成。

## 3. CLI

新增参数：

```bash
--feature-source mock|precomputed
--visual-features-csv PATH
--use-loaded-legacy-mlp
--router-checkpoint-payload PATH
--scaler-state-json PATH
--disable-scaler
```

典型 loaded path smoke：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python scripts/run_stage1_visual_small.py \
  --feature-source precomputed \
  --visual-features-csv tests/fixtures/stage1_visual_scaler_small/raw_visual_features.csv \
  --scaler-state-json tests/fixtures/stage1_visual_scaler_small/scaler_state.json \
  --use-loaded-legacy-mlp \
  --router-checkpoint-payload /tmp/tiny_legacy_visual_mlp_payload.pt \
  --output-dir /tmp/stage1_p16j_run_outputs \
  --split-name test
```

## 4. Runtime Artifacts

`run_metadata.json` 中新增或明确记录：

- `branch_name = visual_router_small`
- `visual_router.feature_source = mock|precomputed`
- `visual_router.loaded_legacy_mlp = true|false`
- `visual_router.checkpoint_payload_source = explicit_small_fixture|none`
- `visual_router.scaler_enabled = true|false`
- `visual_router.loads_real_checkpoint = false`
- `visual_router.loads_real_vit = false`
- `visual_router.formal_visual_router_migration = false`
- `visual_router.p16i_helper_used` 和 `visual_router.p16a_adapter_used`

`checkpoint_payload_path` 只作为 Runtime metadata 的 lineage 留痕，不进入 adapter interface。

## 5. 验证

本步新增 `tests/smoke/stage1_visual_small_entrypoint_loaded_legacy_path_smoke.py`，覆盖：

- tempfile tiny checkpoint payload；
- `module.` prefix state_dict 清理和 strict load；
- `VisualPrecomputedFeatureProvider` + `LoadedFeatureScaler`；
- `LoadedTorchMLPRouterHeadAdapter` 输出 finite logits/weights；
- softmax row sum、sample_key 顺序、model_columns 对齐；
- canonical run_dir、run_status、evaluation_summary、prediction_rows；
- stdout/stderr 不出现 `/data2`、ViT/transformers 或正式 streaming 入口。

默认路径仍由 `tests/smoke/stage1_visual_small_entrypoint_smoke.py` 覆盖。

## 6. 后续

P16j 完成后，可以继续拆：

1. 真实 checkpoint dry-run：仍只做显式 path，不做 discovery。
2. real Visual feature chain：`x -> pseudo image -> frozen ViT -> optional scaler -> FeatureBatch`。
3. 正式 eval entrypoint migration plan：把 Runtime checkpoint/scaler/feature chain 和 canonical writer 接到正式 evaluation path。

上述步骤都不能回写 P16a adapter 的接口，也不能把 small/debug precomputed feature 当成正式唯一方案。
