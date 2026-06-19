# Stage 1 P9c Visual Router evaluation adapter pressure 验证

日志日期：2026-06-20 02:42:08 CST

## 目的

验证 P9b 在 Visual Router 正式入口中新增的 `--verify-evaluation-adapter` 只做内存旁路一致性校验，不改变正式输出 artifact。

## 背景

P9a 已完成 Visual Router 正式入口 adapter 插入点审计。P9b 已在
`train_visual_router_online_streaming.py` 中新增默认关闭的
`--verify-evaluation-adapter`，并新增 `verify_evaluation_adapter_bypass_batch(...)`、
smoke 测试和说明文档。

本轮 P9c 不继续迁移正式入口，不改 Visual FeatureProvider、ViT、router head、training
loop、`fusion_huber_kl` loss、checkpoint/resume 语义或正式 CSV/metadata/status schema。

## 操作

1. 在 `refactor/stage1-route-audit` 分支检查现有 P9a/P9b 文档和工作区状态。
2. 使用 conda 环境 `quito` 运行基础验收：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_router_evaluation_adapter_bypass_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py
   ```

3. 检查仓库内已有小规模正式入口输入：
   - labels：`experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/window_oracle_labels_with_tsf_cell.csv`
   - prediction manifest：`experiment_logs/run_outputs/2026-06-14_stage1_full_scale_dry_run_v2/merged_cache/manifest.csv`
   - Quito config：`/home/shiyuhong/Time/quito/outputs/default_baseline/dlinear/96_48_S/seed_16/EVALUATE/ver_0/config.yaml`
   - config 数据目录为 `examples/datasets/cluster_data`，未指向 `/data2`
   - `google/vit-base-patch16-224` 可通过 `--local-files-only` 本地加载
4. 使用 CPU、`fp32`、`--local-files-only`、每 split 2 样本、1 epoch、同一 seed 和同一 shard 做两组正式入口对照：
   - verify 关闭输出到 `experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_off/`
   - verify 开启输出到 `experiment_logs/run_outputs/2026-06-20_stage1_p9c_visual_router_eval_adapter_pressure_verify_on/`
5. 对比以下 artifact：
   - `visual_router_predictions.csv`
   - `visual_router_soft_fusion_predictions.csv`
   - `visual_router_summary.csv`
   - `visual_router_soft_fusion_summary.csv`
   - `visual_router_selected_model_counts.csv`
   - `visual_router_comparison.csv`
   - `visual_router_streaming_summary.md` 核心表格
   - 文件集合、metadata/status/checkpoint index top-level schema
6. 新增 `docs/refactor/visual_router_evaluation_adapter_pressure_verification.md`，并更新 roadmap、entrypoint migration plan、`WORKSPACE_STRUCTURE.md` 和实验日志 README。

## 结果

基础验收全部通过：

- P9b Visual Router evaluation adapter bypass smoke 通过。
- P6b EvaluationInput adapter smoke 通过。
- P7c TimeFuse protocol chain smoke 通过。
- `compileall` 通过。

两组正式入口 pressure run 均完成：

- `rows_seen=20`
- `matched_rows=20`
- `target_sample_keys=4`
- hard summary `sample_count=2`
- soft fusion summary `sample_count=2`
- selected counts 5 行

artifact 对比结果：

- `visual_router_predictions.csv`：字段顺序一致，行数 2/2，`sample_key` 顺序一致，内容完全一致。
- `visual_router_soft_fusion_predictions.csv`：字段顺序一致，行数 2/2，`sample_key` 顺序一致，权重、hard 指标、raw soft 指标在 `rtol=1e-9, atol=1e-9` 内一致。
- `visual_router_summary.csv`：字段顺序一致，行数 1/1，指标一致。
- `visual_router_soft_fusion_summary.csv`：字段顺序一致，行数 1/1，指标一致。
- `visual_router_selected_model_counts.csv`：字段顺序一致，行数 5/5，selected counts 一致。
- `visual_router_comparison.csv`：字段顺序一致，行数 12/12；除 run_dir 路径型 `source` 值外一致，归一化 run_dir 后完全一致。
- `visual_router_streaming_summary.md`：除生成时间和输出路径外，核心表格归一化后一致。
- 两个输出目录文件集合一致，共 16 个文件；开启 verify 后没有新增 adapter artifact。
- `visual_router_metadata.json`、`visual_router_online_metadata.json`、`status.json` 和 `checkpoints/latest_checkpoint_index.json` top-level schema 一致。

## 结论

P9c 小规模正式入口 pressure 验证通过。开启 `--verify-evaluation-adapter` 不改变 Visual Router 正式输出的字段顺序、行数、样本顺序、selected model、权重诊断、hard top-1 指标、raw soft fusion 指标、summary/comparison/selected counts 或 schema。

`--verify-evaluation-adapter` 仍只是旁路校验：adapter rows 只在 test evaluation batch 内作为内存对象存在，不写入正式 CSV，不改变 Visual feature、ViT、router head、training loop、loss、checkpoint、metadata 或 status。

## 下一步方案

1. 提交并推送本轮 P9c 文档、日志和输出目录记录。
2. 后续如扩大验证范围，继续使用显式 `--output-dir`、小样本、`--local-files-only`、明确设备和固定 seed。
3. 不把 P9c 扩展为 VisualFeatureProvider、ViT provider、router head 或 training loop 迁移；这些必须另开阶段并设置独立 smoke/pressure 门禁。
