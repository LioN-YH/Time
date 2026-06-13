# Stage 1 ViT Embedding 与 Visual Router Smoke

日志日期：2026-06-14 01:13:32 CST

## 目的

在已有 `96_48_S` 扩大版五专家 prediction cache pilot 的 120 个 `metric=mae` sample_key 上，完成冻结 ViT embedding 与 TimeFuse-style Visual Router 的最小闭环：

1. 使用 `google/vit-base-patch16-224` 对在线伪图像编码，输出视觉特征；
2. 参考 TimeFuse `ModelFusor` 的 softmax 权重设计，实现小型 MLP Visual Router；
3. 在 `vali` split 训练，在 `test` split 评估 hard top-1 routing 和 soft fusion；
4. 输出可复核 manifest、summary、comparison，并更新 Stage 1 文档和工作区结构说明。

## 背景

此前 Stage 1 已完成 prediction cache、window oracle label、TSF cell enrichment、非视觉 baseline、TimeFuse-derived 结构特征 router、在线伪图像化和 HF ViT normalization 口径。当前未完成的是冻结视觉 encoder embedding 与最小 visual router 训练评估。

本次继续处理前序因网络中断遗留的工作。检查发现 Hugging Face 本地缓存中 `models--google--vit-base-patch16-224` 只有 `config.json` 和 `preprocessor_config.json`，缺少 `model.safetensors` 权重文件，因此 `--local-files-only` 加载失败。

## 操作

1. 阅读并核对了以下 Stage 1 相关文件：
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_cache_contract.md`
   - `visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py`
   - `visual_router_experiments/stage1_vali_test_router/pilot/build_online_pseudo_image_pilot.py`
   - `visual_router_experiments/stage1_vali_test_router/pilot/train_structure_router_pilot.py`
   - `visual_router_experiments/common/pseudo_imageization.py`
   - `visual_router_experiments/common/prediction_cache_schema.py`
   - `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py`
   - 近期 `experiment_logs/README.md` 与 `WORKSPACE_STRUCTURE.md`

2. 阅读 TimeFuse 的 fusor 设计：
   - `TimeFuse/timefuse.py` 中 `ModelFusor` 使用 `Linear(input_dim, output_dim)` 后接 `softmax`，输出模型融合权重；
   - 本次 Visual Router 沿用“输出专家权重”的思想，但将线性层扩展为小型 MLP：`Linear -> GELU -> Dropout -> Linear -> softmax`。

3. 完成或修正 ViT embedding 路径：
   - `build_vit_embeddings.py` 默认使用 `google/vit-base-patch16-224`；
   - 输入为在线伪图像 `variant_a_3view`，先按 `hf_vit_0_5` 执行 `(x - 0.5) / 0.5`；
   - 采用 direct `pixel_values` 路径，不经过 HF processor；
   - 默认取 `last_hidden_state[:, 0]` CLS token 作为 768 维视觉特征；
   - 为避免默认 CLS 路径创建未使用的随机 pooler，加载 ViT 时在非 pooler pooling 下设置 `add_pooling_layer=False`。

4. 由于本地 HF 缓存缺少权重，先执行一次联网补齐：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python - <<'PY'
   from transformers import ViTModel
   m = ViTModel.from_pretrained('google/vit-base-patch16-224')
   print(type(m).__name__, m.config.hidden_size)
   PY
   ```

   该步骤下载了 `model.safetensors`，确认 hidden size 为 `768`。随后再用 `--local-files-only` 运行 smoke。

5. 新增正式入口脚本：
   - `visual_router_experiments/stage1_vali_test_router/train_visual_router.py`

   该脚本读取 ViT embedding manifest 和 oracle labels，按 `config_name` 独立训练：
   - `StandardScaler` 只在 `vali` split 上 fit；
   - 小型 MLP 只用 `vali` 的 `oracle_model` 监督训练；
   - 输出五专家 softmax 权重；
   - hard top-1 取最大权重专家；
   - soft fusion 用五专家权重加权同一 `sample_key` 下的 `y_pred` 数组；
   - 不使用未来 `y`、test oracle 误差或专家误差作为输入特征。

6. 运行 ViT embedding smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py \
     --local-files-only \
     --batch-size 16 \
     --print-rows 5
   ```

7. 运行 Visual Router smoke：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
     --embedding-manifest-path experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/embedding_manifest.csv \
     --epochs 300 \
     --batch-size 32 \
     --hidden-dim 64 \
     --print-rows 5
   ```

8. 完成验证：
   - 使用 quito conda 环境执行 `py_compile`；
   - 校验 ViT `add_pooling_layer=False` 路径可本地加载；
   - 读取输出 CSV，确认 embedding manifest、hard summary、soft fusion summary 和 comparison 均可正常解析。

9. 更新文档：
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`

## 结果

### ViT Embedding

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_010821_165988_visual_router_stage1_vit_embedding_smoke/
```

关键输出：

- `embedding_manifest.csv`
- `embedding_latency_summary.csv`
- `embedding_metadata.json`
- `embedding_summary.md`
- `embeddings/*.npy`

关键结果：

- 覆盖 `120` 个 `metric=mae` sample_key；
- split 计数为 `vali=60`、`test=60`；
- dataset 计数为每个 split 下 `TEST_DATA_HOUR=30`、`TEST_DATA_MIN=30`；
- embedding 维度为 `768`；
- encoder 为 `google/vit-base-patch16-224`；
- pooling 为 `last_hidden_state[:, 0]` CLS token；
- `variant=variant_a_3view`；
- `normalization_preset=hf_vit_0_5`；
- `forward_dtype=float16`，保存为 `float32`；
- manifest 与 oracle labels 的 sample_key 集合完全一致；
- latency mean：imageization `7.766617 ms/window`，encoder forward `17.626119 ms/window`，write `0.165546 ms/window`。

### Visual Router

输出目录：

```text
experiment_logs/run_outputs/2026-06-14_010907_224073_visual_router_stage1_visual_router_smoke/
```

关键输出：

- `visual_router_predictions.csv`
- `visual_router_summary.csv`
- `visual_router_soft_fusion_predictions.csv`
- `visual_router_soft_fusion_summary.csv`
- `visual_router_comparison.csv`
- `visual_router_metadata.json`
- `visual_router_summary.md`

hard top-1 结果：

| 方法 | config | test sample | test MAE | oracle MAE | regret | oracle label accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `visual_router_mlp_v1` | `96_48_S` | 60 | 1.013099 | 0.805392 | 0.207707 | 0.350000 |

soft fusion 结果：

| 方法 | config | test sample | soft fusion MAE | soft fusion MSE | hard top-1 MAE | hard top-1 MSE |
| --- | --- | --- | --- | --- | --- | --- |
| `visual_router_mlp_v1_soft_fusion` | `96_48_S` | 60 | 1.022590 | 4.862953 | 1.013099 | 4.729046 |

同表比较中的关键项：

| 方法 | test MAE | 相对 `global_best_single` |
| --- | --- | --- |
| `oracle_top1` | 0.805392 | +23.6733% |
| `visual_router_mlp_v1` hard top-1 | 1.013099 | +3.9889% |
| `visual_router_mlp_v1_soft_fusion` | 1.022590 | +3.0895% |
| `global_best_single` | 1.055190 | 0.0000% |
| `timefuse_single_variable_logistic_regression` | 1.079743 | -2.3269% |

Visual Router hard top-1 的 test 选中专家计数：

```text
CrossFormer        31
DLinear            16
ES                 11
PatchTST            1
NaiveForecaster     1
```

### 验证结果

执行以下验证均通过：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile \
  visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py \
  visual_router_experiments/stage1_vali_test_router/train_visual_router.py \
  visual_router_experiments/common/pseudo_imageization.py
```

ViT 本地加载验证通过：

```text
ViTModel 768 True True
```

输出 CSV 读取验证通过：

```text
embedding_manifest.csv: (120, 20)
visual_router_summary.csv: (1, 7)
visual_router_soft_fusion_summary.csv: (1, 8)
visual_router_comparison.csv: (12, 9)
```

## 结论

1. `google/vit-base-patch16-224` 冻结视觉 embedding 已在当前 120 个 `sample_key` 上跑通，默认 CLS token 输出 768 维视觉特征。
2. TimeFuse-style 小型 MLP Visual Router 已完成最小闭环，可以输出五专家权重，并评估 hard top-1 与 soft fusion。
3. 在当前小规模 pilot 中，hard top-1 visual router 的 MAE `1.013099` 优于 `global_best_single=1.055190` 和 TimeFuse-derived 结构特征 router `1.079743`。
4. soft fusion 的 MAE `1.022590` 略弱于 hard top-1，说明当前权重校准仍不够好，后续需要做温度、正则化、训练目标或融合策略诊断。
5. 当前结果仍只是 `96_48_S`、120 sample_key 的 smoke，不应解释为三 config 正式结论。

## 下一步方案

1. 补充 HF processor 路径 smoke，特别是 `do_rescale=False` 与 `processor_uint8` 口径，避免后续多 encoder 对比时输入尺度混乱。
2. 扩大 `96_48_S` prediction cache 和 embedding 样本，优先检查 visual router 是否在更大样本上仍超过 `global_best_single`。
3. 诊断 visual router 过拟合、类别不平衡和权重过尖锐问题；可尝试温度 softmax、label smoothing、balanced sampler 或 reward/regret-aware 目标。
4. 逐步迁出正式 prediction cache builder、统一 evaluator 和 summarizer，保持 run output 写入 `experiment_logs/run_outputs/` 或 `/data2/syh/Time/run_outputs/`。
5. 在 `96_48_S` 链路稳定后，再扩展到 `576_288_S` 与 `1024_512_S`，并保持 per-config router 口径。
