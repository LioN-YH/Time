# Stage 1 P9c Visual Router Evaluation Adapter Pressure Verification

日志日期：2026-06-20 02:42:08 CST

## 目标

P9c 验证 P9b 在 `train_visual_router_online_streaming.py` 中新增的
`--verify-evaluation-adapter` 只是 evaluation batch 内存旁路校验，不改变 Visual
Router 正式输出 artifact。

本轮是 pressure / consistency verification，不是新一轮迁移。正式入口仍保留：

- `x -> pseudo image -> frozen ViT -> router` feature 路径；
- `predict_stream_batch(...)`；
- `add_soft_fusion_metrics(...)`；
- `EvaluationInputAdapter` 实现；
- `VisualMLPRouter` / router head；
- training loop、`fusion_huber_kl` loss、optimizer、checkpoint/resume 语义；
- 正式 CSV / summary / metadata / status schema。

adapter rows 仍只作为内存对象用于校验，不写入正式 CSV，也不新增 adapter 输出文件。

## 基础验证

在 conda 环境 `quito` 下完成以下命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
```

结果：

- P9b Visual Router evaluation adapter bypass smoke 通过。
- P6b EvaluationInput adapter smoke 通过。
- P7c TimeFuse protocol chain smoke 通过。
- `compileall` 通过。

## Pressure 输入与边界

本轮使用仓库内已有小规模 dry-run 输入，不访问 `/data2`，不下载模型，不启动 full-scale。

- 分支：`refactor/stage1-route-audit`
- labels：
  `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv`
- prediction manifest：
  `experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv`
- config：
  `/home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml`
- config 数据目录：`examples/datasets/cluster_data`，未指向 `/data2`
- ViT：`google/vit-base-patch16-224`，`--local-files-only` 本地可加载
- 设备：`--device cpu`
- dtype：`--dtype fp32`
- 样本规模：`--max-samples-per-split 2`，即 2 个 vali 样本和 2 个 test 样本
- shard：`--stream-shard-index 0 --stream-shard-count 1`
- 训练：`--epochs 1 --batch-size 2 --embedding-batch-size 2`
- seed：`16`
- soft fusion：开启，未使用 `--skip-soft-fusion`

输出目录：

- verify 关闭：
  `experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_off/`
- verify 开启：
  `experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_on/`

## Pressure 命令

关闭 verify：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
  --labels-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv \
  --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv \
  --config-path /home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml \
  --output-dir experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_off \
  --router-mode fusion_huber_kl \
  --epochs 1 \
  --batch-size 2 \
  --embedding-batch-size 2 \
  --max-samples-per-split 2 \
  --stream-shard-index 0 \
  --stream-shard-count 1 \
  --device cpu \
  --dtype fp32 \
  --local-files-only \
  --seed 16 \
  --print-rows 5
```

开启 verify：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py \
  --labels-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv \
  --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv \
  --config-path /home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml \
  --output-dir experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_on \
  --router-mode fusion_huber_kl \
  --epochs 1 \
  --batch-size 2 \
  --embedding-batch-size 2 \
  --max-samples-per-split 2 \
  --stream-shard-index 0 \
  --stream-shard-count 1 \
  --device cpu \
  --dtype fp32 \
  --local-files-only \
  --seed 16 \
  --print-rows 5 \
  --verify-evaluation-adapter
```

两次运行均完成：

- manifest index：`rows_seen=20`、`matched_rows=20`、`target_sample_keys=4`
- hard summary：`sample_count=2`
- soft fusion summary：`sample_count=2`
- selected counts：5 个专家行

## 对比结果

对比的正式 artifact：

| 文件 | 字段顺序 | 行数 | sample_key 顺序 | 内容 |
| --- | --- | --- | --- | --- |
| `visual_router_predictions.csv` | 一致 | 2 / 2 | 一致 | 完全一致 |
| `visual_router_soft_fusion_predictions.csv` | 一致 | 2 / 2 | 一致 | 浮点容差内一致 |
| `visual_router_summary.csv` | 一致 | 1 / 1 | 不适用 | 浮点容差内一致 |
| `visual_router_soft_fusion_summary.csv` | 一致 | 1 / 1 | 不适用 | 浮点容差内一致 |
| `visual_router_selected_model_counts.csv` | 一致 | 5 / 5 | 不适用 | 完全一致 |
| `visual_router_comparison.csv` | 一致 | 12 / 12 | 不适用 | 除 run_dir 路径型 `source` 值外一致；归一化 run_dir 后完全一致 |
| `visual_router_streaming_summary.md` | 不适用 | 不适用 | 不适用 | 除生成时间和输出路径外，核心表格归一化后一致 |

重点字段结果：

- `selected_model` 一致。
- `weight_DLinear`、`weight_PatchTST`、`weight_CrossFormer`、`weight_ES`、`weight_NaiveForecaster` 在 `rtol=1e-9, atol=1e-9` 内一致。
- `weight_entropy`、`normalized_weight_entropy`、`max_weight` 在 `rtol=1e-9, atol=1e-9` 内一致。
- `hard_top1_mae_from_array`、`hard_top1_mse_from_array` 在 `rtol=1e-9, atol=1e-9` 内一致。
- `soft_fusion_mae`、`soft_fusion_mse` 在 `rtol=1e-9, atol=1e-9` 内一致。
- summary、comparison、selected counts 的指标一致。
- 文件集合一致，共 16 个文件；开启 verify 后没有新增 adapter artifact。
- `visual_router_metadata.json`、`visual_router_online_metadata.json`、`status.json` 和
  `checkpoints/latest_checkpoint_index.json` 的 top-level schema 一致。

路径型差异说明：

- `visual_router_comparison.csv` 的 `source` 列包含当前 run 的 summary CSV 路径，因此 off/on 输出目录不同会产生预期路径差异。
- `visual_router_streaming_summary.md` 包含生成时间和输出文件路径，因此 off/on run 会有预期路径和时间差异。
- 这些差异不是 `--verify-evaluation-adapter` 改变 evaluation 结果或 schema 造成的漂移；归一化 run_dir 和生成时间后，核心表格口径一致。

## 结论

在该小规模正式入口 pressure 配置下，开启 `--verify-evaluation-adapter` 没有改变 Visual Router 正式输出的字段顺序、行数、样本顺序、selected model、权重诊断、hard top-1 指标、raw soft fusion 指标、summary/comparison/selected counts 或 schema。

P9b 的 flag 仍只是旁路校验：它在 test evaluation batch 内构造内存 `EvaluationInput`，通过 `EvaluationInputAdapter` 复算并比较正式路径结果；adapter rows 不写正式 CSV，不改变 feature / ViT / router head / training loop / loss / checkpoint / metadata / status。

后续如果继续扩大验证范围，应仍使用显式 `--output-dir`、小样本、`--local-files-only` 和明确设备参数；不得把该验证扩展成 provider/head 迁移或 full-scale 启动。
