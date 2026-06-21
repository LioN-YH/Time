# Stage 1 P20a Visual Eval Visual-chain Path

## 目标

P20a 在 P17 canonical eval entrypoint 中新增显式 small visual-chain dry-run feature source：

```text
SampleManifest
-> explicit raw window JSON
-> VisualFeatureChainRunner
-> injected/fake VisualVitEncoderProvider
-> PoolingStrategy
-> optional LoadedFeatureScaler FeatureTransform
-> FeatureBatch
-> Runtime-loaded legacy VisualMLPRouter checkpoint payload
-> LoadedTorchMLPRouterHeadAdapter
-> ExpertBatch
-> EvaluationInputAdapter
-> Runtime artifact writer
-> canonical run_dir
```

本步证明 P19 的 visual-chain 基础设施可以进入 P17 canonical eval 后半段；它不是 full-scale，不启动训练，不迁移 `train_visual_router_online_streaming.py`，默认不导入 transformers，也不默认加载真实 ViT。

## CLI

新增 feature source：

- `--feature-source precomputed`：默认值，保持 P17a-P17d 行为。
- `--feature-source visual-chain-dryrun`：新增 small visual-chain dry-run path。

新增参数：

- `--raw-window-json PATH`：visual-chain-dryrun 必填；只读取显式文件，不自动搜索。
- `--visual-chain-mode dryrun`：当前唯一支持模式。
- `--vit-provider-mode injected-fake`：默认 fake provider，不导入 transformers。
- `--manual-real-vit --allow-real-vit --vit-model-path PATH`：显式 manual real ViT dry-run。
- `--allow-external-vit-path`：当 real ViT path 位于 `/data2` 或仓库外时的额外授权。

## Metadata

`run_metadata.json` 的 `visual_router` 段在 visual-chain-dryrun 下记录：

- `feature_source = visual-chain-dryrun`
- `visual_chain_enabled = true`
- `raw_window_source`
- `vit_provider_mode = injected-fake`
- `loads_real_vit = false`
- `visual_chain_runner = VisualFeatureChainRunner`
- `encoder_provider = VisualVitEncoderProvider`
- `training_started = false`
- `formal_training_migration = false`
- `full_scale_run = false`

`inputs.visual_features_csv` 在该路径下为 `null`，`inputs.raw_window_json` 保存显式 fixture 的路径和 sha256。checkpoint path 仍只记录在 Runtime/head lineage，不下沉到 provider、runner 或 head adapter。

## Smoke

`tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py` 覆盖：

- 调用 P17a thin-slice smoke，确认默认 precomputed path 回归通过。
- 使用 P13b manifest/expert JSON、P19a raw window fixture 和 tempfile tiny checkpoint payload 运行 `--feature-source visual-chain-dryrun`。
- 验证 canonical run_dir、`run_status=completed`、`evaluation_summary.json` 和 `prediction_rows.csv` sample_key 保序。
- 验证 metadata 中 `visual_chain_enabled=true`、`loads_real_vit=false`、`full_scale_run=false`。
- 验证 stdout/stderr 不出现 `ViTModel`、`AutoImageProcessor`、`train_visual_router_online_streaming.py` 或 `/data2`。
- 验证缺少 `--raw-window-json` 和 raw-window fixture 缺少 manifest sample_key 时 fail-fast。

## 不做范围

- 不启动 full-scale。
- 不启动训练。
- 不新增 Bash launcher。
- 不自动搜索 `/data2`。
- 不默认加载真实 ViT。
- 不默认导入 transformers 或下载 HuggingFace 模型。
- 不迁移或修改 `train_visual_router_online_streaming.py`。
- 不删除 P17 precomputed feature path。
- 不要求与 legacy Visual Router 数值对齐。

## 验收命令

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_dryrun_skeleton_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_vit_encoder_guard_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall scripts/run_stage1_visual_eval_canonical.py tests/smoke/stage1_visual_eval_canonical_visual_chain_path_smoke.py
```
