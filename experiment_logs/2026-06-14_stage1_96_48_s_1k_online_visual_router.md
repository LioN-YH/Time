# Stage 1 `96_48_S` 1k Online Visual Router

日志日期：2026-06-14 17:52:36 CST

## 目的

使用 `train_visual_router_online.py` 在 `96_48_S` 1k prediction cache 上训练 online visual router，验证在线 ViT embedding 只在运行内存中暂存，并使用本地 HF/ViT cache 与 `--local-files-only`，不长期保存伪图像 tensor 或 ViT embedding `.npy`。

## 背景

前置步骤已经生成：

- 合并后的五专家 prediction cache：`merged_cache/manifest.csv`；
- 带 TSF cell 的 oracle labels：`merged_cache/window_oracle_labels_with_tsf_cell.csv`；
- 非视觉 baseline：`merged_cache/baseline_summary.csv`。

online router 训练应复用这些产物，输入只使用 Quito 历史窗口 `x` 生成伪图像和冻结 ViT CLS embedding，不使用未来 `y`、专家误差或 oracle label 作为输入特征。

## 操作

1. 先验证本地 ViT cache 可离线读取：

   ```text
   HF_HUB_CACHE=/data2/syh/Time/hf_models/google-vit-base-patch16-224 \
   HF_HUB_OFFLINE=1 \
   TRANSFORMERS_OFFLINE=1 \
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -c "from transformers import ViTConfig; ViTConfig.from_pretrained('google/vit-base-patch16-224', local_files_only=True)"
   ```

   验证结果为 `hidden_size=768`、`image_size=224`、`patch_size=16`。

2. 第一次运行 online router 到目录：

   ```text
   experiment_logs/run_outputs/2026-06-14_174758_visual_router_stage1_online_visual_router_96_48_s_1k/
   ```

   该运行完成且结果有效，但 `visual_router_online_metadata.json` 中没有显式保存 `local_files_only` 布尔字段，只能从 `main.log` 命令反推。

3. 为增强产物自证性，修改 `train_visual_router_online.py`，在 run metadata 中新增：

   ```text
   "local_files_only": bool(args.local_files_only)
   ```

4. 使用同样输入、同样训练参数重跑最终 online router：

   ```text
   HF_HUB_CACHE=/data2/syh/Time/hf_models/google-vit-base-patch16-224 \
   HF_HUB_OFFLINE=1 \
   TRANSFORMERS_OFFLINE=1 \
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     /home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py \
     --labels-path experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/window_oracle_labels_with_tsf_cell.csv \
     --prediction-manifest-path experiment_logs/run_outputs/2026-06-14_101000_visual_router_stage1_prediction_cache_96_48_s_1k_launcher/merged_cache/manifest.csv \
     --router-mode fusion_huber_kl \
     --metric mae \
     --local-files-only \
     --output-dir experiment_logs/run_outputs/2026-06-14_175036_visual_router_stage1_online_visual_router_96_48_s_1k_local_only
   ```

## 结果

最终输出目录：

```text
experiment_logs/run_outputs/2026-06-14_175036_visual_router_stage1_online_visual_router_96_48_s_1k_local_only/
```

关键 metadata 校验：

- `local_files_only = True`；
- `embedding_storage = in_memory_only`；
- `persistent_embedding_npy_written = False`；
- `persistent_pseudo_image_tensor_written = False`；
- `online_embedding_manifest.csv` 行数为 1000；
- embedding dim 为 768；
- `online_embedding_manifest.csv` 不包含 `embedding_path`；
- 输出目录下没有 `.npy` 文件；
- 输出目录下没有 `embeddings/`、`arrays/` 或 `pseudo_images/` 目录。

Online embedding latency：

- imageization per-window mean ms：0.924820；
- encoder forward per-window mean ms：1.524758；
- in-memory store per-window mean ms：0.038487。

Hard top-1 结果：

| router_name | config_name | sample_count | selected_value | oracle_value | regret_to_oracle | oracle_label_accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| visual_router_mlp_v2_fusion_huber_kl_online_vit | 96_48_S | 500 | 0.459729 | 0.356273 | 0.103456 | 0.392 |

Raw soft fusion 结果：

| router_name | sample_count | soft_fusion_mae | soft_fusion_mse | hard_top1_mae_from_array | oracle_value |
| --- | ---: | ---: | ---: | ---: | ---: |
| visual_router_mlp_v2_fusion_huber_kl_online_vit_soft_fusion | 500 | 0.437221 | 0.578778 | 0.459729 | 0.356273 |

与非视觉 baseline 对照：

- `global_best_single` MAE = 0.467657；
- `dataset_tsf_cell` MAE = 0.439672；
- online hard top-1 MAE = 0.459729；
- online raw soft fusion MAE = 0.437221；
- oracle top-1 MAE = 0.356273。

## 结论

1k online visual router 已完成，并满足本轮本地模型/cache 和不落盘约束。hard top-1 优于 `global_best_single`，但弱于 `dataset_tsf_cell`；raw soft fusion 略优于 `dataset_tsf_cell`，是当前可部署结果中最好的未校准视觉融合结果。

## 下一步方案

1. 对最终 online router 输出运行 `evaluate_soft_fusion_calibration.py`。
2. 比较 raw soft、top-1 hard、top-2/top-3 fusion 和 temperature sweep，确认最佳 calibrated soft fusion。
3. 完成后写最终统一中文汇总日志。
