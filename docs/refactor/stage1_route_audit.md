# Stage 1 路线审计

审计日期：2026-06-19

## 1. 审计结论

Stage 1 不是两套彼此独立的实验系统，而是“一条共享数据与评估主干，加两种特征提供器和路由头”。两边共享同一批 `sample_key`、同一 `config_name` 内的五专家 prediction cache、同一套 `y_pred/y_true`、同一套 oracle 上限和 `vali` 训练、`test` 评估协议。这个共同契约由 `visual_router_experiments/stage1_vali_test_router/stage1_cache_contract.md` 固定，正式主线与当前结果由 `visual_router_experiments/stage1_vali_test_router/stage1_visual_router_mainline.md`、`visual_router_experiments/stage1_vali_test_router/README.md` 和 `HANDOFF.md` 交叉确认。

两条路线真正需要不同的部分只有：

```text
共享 sample / prediction / target batch
├── Visual Router
│   历史窗口 x -> 在线伪图像 -> frozen ViT -> MLP router -> 五专家权重
└── TimeFuse-style fusor
    历史窗口 x -> 离线 17 维 feature cache -> Linear-softmax fusor -> 五专家权重
```

当前代码看起来像两套大型流程，主要是 full-scale 数据规模、特征生命周期不同，以及两边先后独立实现了 SQLite、shard reader、恢复和日志设施；这属于工程实现漂移，不是实验协议的根本差异。Visual Router 的 OOM 修复证据见 `experiment_logs/2026-06-17_stage1_96_48_s_streaming_visual_router_oom_fix_review_restart.md`，TimeFuse reader/scaler 的独立优化证据见 `experiment_logs/2026-06-19_stage1_timefuse_fusor_reader_scaler_optimization_restart.md`。

## 2. 共享主干

### 2.1 样本与动作空间

- 路由粒度固定为 `config_name + split + dataset_name + item_id + channel_id + window_index`，稳定 key 由 `visual_router_experiments/common/prediction_cache_schema.py` 定义。
- 每个 router/fusor 只在一个 `config_name` 内选择 `{DLinear, PatchTST, CrossFormer, ES, NaiveForecaster}`，不得跨 config 选择专家；依据为 `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`。
- 正式 `96_48_S` full-scale sample manifest 共 `23,275,170` 个 sample，64 个 shard；记录见 `visual_router_experiments/stage1_vali_test_router/stage1_visual_router_mainline.md` 和 `WORKSPACE_STRUCTURE.md`。

### 2.2 Prediction cache 与监督

- 两条路线共享正式五专家 merged prediction cache，采用 `packed_npy_v1`，共 `116,375,850` 条 `(sample_key, model_name)` 记录；完整性要求是主键唯一、五专家齐全、共享 `y_true` 一致，见 `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py` 和 `visual_router_experiments/stage1_vali_test_router/stage1_cache_contract.md`。
- 两条路线共享 window-level oracle labels 和 TSF enrichment。oracle 是训练辅助/诊断及上限，不是可部署推理输入；相关正式生成与验证入口为 `build_full_scale_window_oracle_labels.py`、`build_full_scale_tsf_enrichment.py` 和 `validate_full_scale_oracle_tsf_outputs.py`。
- `per_sample_npy` 与 `packed_npy_v1` 的统一单样本读取已在 `visual_router_experiments/common/prediction_array_io.py` 实现，但 full-scale batch/grouped 读取仍分散在各路线入口。

### 2.3 训练与评估口径

- 两边都只在 `vali` 拟合 scaler/router，在 `test` 评估，输出五专家权重并比较 hard top-1 与 raw soft fusion；见 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py` 和 `fusion_utils.py`。
- 两边都必须与 `global_best_single`、统计规则 baseline 和 `oracle_top1` 同 config 比较；统计 baseline 入口是 `evaluate_router_baselines.py`。
- Visual Router 另有 temperature/top-k calibration 入口 `evaluate_soft_fusion_calibration.py`。这属于当前评估覆盖范围差异，不改变两边共享的基础 `y_pred/y_true` 口径。

## 3. 正式路线

### 3.1 Visual Router 主线

正式顺序如下，来源为 `visual_router_experiments/stage1_vali_test_router/stage1_visual_router_mainline.md` 和 `visual_router_experiments/stage1_vali_test_router/README.md`：

```text
build_full_scale_sample_manifest.py
-> launch_full_scale_prediction_cache.py
-> build_prediction_cache_from_manifest.py
-> merge_prediction_cache_shards.py
-> build_full_scale_window_oracle_labels.py
-> build_full_scale_tsf_enrichment.py
-> validate_full_scale_oracle_tsf_outputs.py
-> evaluate_router_baselines.py
-> train_visual_router_online_streaming.py
-> evaluate_soft_fusion_calibration.py
```

特征仅在 batch 运行时生成：`x -> pseudo image tensor -> frozen ViT embedding`。full-scale 不落盘伪图像 tensor 或 embedding `.npy`，依据为 `train_visual_router_online_streaming.py` 和 `visual_router_experiments/common/vit_embedding_utils.py`。

截至 `HANDOFF.md` 记录，`96_48_S` 1 epoch checkpoint 的 eval-only 已完成，覆盖 `13,924,650` 个 test window；hard top-1 MAE 为 `0.5615367653`，raw soft fusion MAE 为 `0.5174675760`，oracle MAE 为 `0.3386221412`。这些数值只说明当前已完成结果，不改变路线分类。

### 3.2 TimeFuse-style fusor baseline 支线

正式顺序如下，来源为 `visual_router_experiments/stage1_vali_test_router/README.md`、`stage1_timefuse_fusor_streaming_reader_design.md` 和 `HANDOFF.md`：

```text
共享 full-scale sample manifest
-> build_timefuse_feature_cache_from_manifest.py
-> launch_timefuse_feature_cache_full_scale.py
-> stage1_timefuse_fusor_streaming_reader.py
-> train_timefuse_fusor_streaming.py
-> launch_timefuse_fusor_full_scale.py
```

特征仅来自历史窗口 `x`，离线写成每个 sample 一行的 17 维 feature cache；训练头固定为 `nn.Linear -> softmax -> weighted fusion`，代表 loss 为 `SmoothL1Loss(beta=0.01)`。口径来源为 `build_timefuse_feature_cache_from_manifest.py`、`train_timefuse_fusor_streaming.py` 和 `experiment_logs/2026-06-15_stage1_timefuse_style_fusor_baseline.md`。

截至本次审计所读 `HANDOFF.md`，正式 GPU2/3 目录仍处于训练阶段，尚不能把未完成 checkpoint/summary 当作正式结果。已停止 CPU 目录仅保留过程证据，不得作为正式 baseline 引用；依据为 `experiment_logs/2026-06-18_stage1_timefuse_fusor_gpu23_fairness_relaunch.md` 和 `WORKSPACE_STRUCTURE.md`。

## 4. 废弃、不推荐与历史路线

| 路线 | 当前定位 | 判断依据 |
| --- | --- | --- |
| full-scale 全量 Python prediction lookup | `DEPRECATED_DO_NOT_USE`；曾导致约 117GB RSS 和 OOM，已由 SQLite 磁盘索引 + batch 查询取代 | `experiment_logs/2026-06-17_stage1_96_48_s_streaming_visual_router_oom_fix_review_restart.md`；`train_visual_router_online_streaming.py` |
| 旧 OOM 输出目录 `...streaming_visual_router_1epoch/` | 失败历史，不得引用为正式结果 | `stage1_visual_router_mainline.md`；`experiment_logs/2026-06-16_stage1_streaming_visual_router_oom_fix_and_restart.md` |
| full-scale offline ViT embedding cache | `DEPRECATED_DO_NOT_USE`；仅允许小规模 encoder/debug 历史复现 | `pilot/build_vit_embeddings_pilot.py`；`pilot/launch_96_48_s_1k_vit_embedding_pilot.py`；`README.md` |
| `train_visual_router_online.py` 用于 full-scale | 不推荐；它在运行内暂存全部 embedding，只适合 120/1k 复现 | `train_visual_router_online.py`；`stage1_visual_router_mainline.md` |
| TimeFuse-derived LogisticRegression hard router | `DEPRECATED_DO_NOT_USE`；不是 TimeFuse 原生 fusor/loss，只保留历史对照 | `pilot/train_structure_router_pilot.py`；`experiment_logs/2026-06-15_stage1_timefuse_metadata_baseline_audit.md` |
| TimeFuse CPU full-scale 半程目录 | 已停止留痕，不是正式 baseline | `experiment_logs/2026-06-18_stage1_timefuse_fusor_gpu23_fairness_relaunch.md`；`WORKSPACE_STRUCTURE.md` |
| `run_full_scale_dry_run.py` 结果 | `PILOT_HISTORY`；可验证契约和恢复语义，不得作为正式指标 | `run_full_scale_dry_run.py`；`README.md` |

## 5. Python 文件逐项标签

标签采用单一主标签。`__init__.py` 虽不是命令行脚本，仍纳入 Python 文件全量清单，避免目录审计出现空白。

| 文件 | 主标签 | 判断与证据 |
| --- | --- | --- |
| `visual_router_experiments/common/__init__.py` | `SHARED_INFRASTRUCTURE` | 跨阶段公共包标记；见 `common/README.md` |
| `visual_router_experiments/common/prediction_array_io.py` | `SHARED_INFRASTRUCTURE` | 统一读取两种 prediction array storage；见本文件 docstring 与 `common/README.md` |
| `visual_router_experiments/common/prediction_cache_schema.py` | `SHARED_INFRASTRUCTURE` | 定义 sample key、manifest schema 和指标校验；见本文件 docstring |
| `visual_router_experiments/common/pseudo_imageization.py` | `SHARED_INFRASTRUCTURE` | 跨阶段在线伪图像 tensor 工具；见本文件 docstring |
| `visual_router_experiments/common/vit_embedding_utils.py` | `SHARED_INFRASTRUCTURE` | 提供运行内 ViT 输入、dtype 和 pooling，不落盘 embedding；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/__init__.py` | `SHARED_INFRASTRUCTURE` | Stage 1 Python package 标记；见 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/build_full_scale_sample_manifest.py` | `MAINLINE_VISUAL_ROUTER` | 正式共享样本主干的 full-scale 入口；见 `stage1_visual_router_mainline.md` |
| `visual_router_experiments/stage1_vali_test_router/build_full_scale_tsf_enrichment.py` | `MAINLINE_VISUAL_ROUTER` | 正式主线 TSF enrichment 步骤，同时供 baseline 消费；见 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/build_full_scale_window_oracle_labels.py` | `MAINLINE_VISUAL_ROUTER` | 正式主线 oracle label 生成；见 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/build_prediction_cache_from_manifest.py` | `MAINLINE_VISUAL_ROUTER` | 正式五专家 prediction cache builder；见 `stage1_visual_router_mainline.md` |
| `visual_router_experiments/stage1_vali_test_router/build_stage1_sample_manifest.py` | `PILOT_HISTORY` | 固定 1k 中等规模复现，不是全候选正式入口；见本文件 docstring 与 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/build_timefuse_feature_cache_from_manifest.py` | `BASELINE_TIMEFUSE_STYLE_FUSOR` | 正式 17 维 feature cache builder；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py` | `BASELINE_STATISTICAL` | 主职责是 global/dataset/TSF-cell 规则 baseline；内含的小规模 fusor 逻辑是后续迁移候选；见本文件 docstring 与 `stage1_history_results.md` |
| `visual_router_experiments/stage1_vali_test_router/evaluate_soft_fusion_calibration.py` | `MAINLINE_VISUAL_ROUTER` | 正式 Visual Router 权重 calibration；见 `stage1_visual_router_mainline.md` |
| `visual_router_experiments/stage1_vali_test_router/fusion_utils.py` | `SHARED_INFRASTRUCTURE` | 两类 router/fusor 共用数组读取、hard/soft fusion 和指标；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/launch_full_scale_prediction_cache.py` | `MAINLINE_VISUAL_ROUTER` | 正式 prediction cache 可恢复 launcher；见 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/launch_timefuse_feature_cache_full_scale.py` | `BASELINE_TIMEFUSE_STYLE_FUSOR` | 正式 TimeFuse feature cache launcher；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py` | `BASELINE_TIMEFUSE_STYLE_FUSOR` | 正式 64-shard fusor launcher；见 `HANDOFF.md` |
| `visual_router_experiments/stage1_vali_test_router/merge_prediction_cache_shards.py` | `MAINLINE_VISUAL_ROUTER` | 正式 cache 合并与完整性检查步骤；见 `stage1_visual_router_mainline.md` |
| `visual_router_experiments/stage1_vali_test_router/pilot/__init__.py` | `PILOT_HISTORY` | pilot package 标记；见 `pilot/README.md` |
| `visual_router_experiments/stage1_vali_test_router/pilot/build_online_pseudo_image_pilot.py` | `PILOT_HISTORY` | 只验证图像化与少量 debug preview；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/pilot/build_prediction_cache_pilot.py` | `PILOT_HISTORY` | 小规模 cache schema/data-flow 验证；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/pilot/build_structure_feature_cache_pilot.py` | `PILOT_HISTORY` | TimeFuse-derived feature 的小规模历史验证；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/pilot/build_vit_embeddings_pilot.py` | `DEPRECATED_DO_NOT_USE` | offline embedding 正式路线已废弃；仅可做小规模历史/debug；见本文件 docstring 与 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/pilot/compute_window_oracle_from_cache.py` | `PILOT_HISTORY` | pilot manifest 的 oracle/regret 计算；正式实现已在 stage 根目录；见 `pilot/README.md` |
| `visual_router_experiments/stage1_vali_test_router/pilot/enrich_cache_with_tsf_cell.py` | `PILOT_HISTORY` | pilot TSF enrichment；正式实现已在 stage 根目录；见 `pilot/README.md` |
| `visual_router_experiments/stage1_vali_test_router/pilot/launch_96_48_s_1k_prediction_cache_pilot.py` | `PILOT_HISTORY` | 固定 1k 历史 launcher；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/pilot/launch_96_48_s_1k_vit_embedding_pilot.py` | `DEPRECATED_DO_NOT_USE` | offline embedding launcher 不得进入正式主线；见本文件 docstring 与 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/pilot/train_structure_router_pilot.py` | `DEPRECATED_DO_NOT_USE` | LogisticRegression hard-label 口径已明确降级；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/run_full_scale_dry_run.py` | `PILOT_HISTORY` | 只验证 full-scale 模板闭环，不产生正式指标；见本文件 docstring |
| `visual_router_experiments/stage1_vali_test_router/stage1_timefuse_fusor_streaming_reader.py` | `BASELINE_TIMEFUSE_STYLE_FUSOR` | 当前正式 fusor shard/SQLite/batch reader；见 `stage1_timefuse_fusor_streaming_reader_design.md` |
| `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` | `BASELINE_TIMEFUSE_STYLE_FUSOR` | 正式 streaming fusor 训练/评估入口；见本文件 docstring 与 `HANDOFF.md` |
| `visual_router_experiments/stage1_vali_test_router/train_visual_router.py` | `SHARED_INFRASTRUCTURE` | offline CLI 不是 full-scale 主线，但其 MLP、loss、hard/soft 评估被 online/streaming 入口复用；见 `train_visual_router_online.py` 和 `train_visual_router_online_streaming.py` 的导入 |
| `visual_router_experiments/stage1_vali_test_router/train_visual_router_online.py` | `PILOT_HISTORY` | 运行内暂存全部 embedding，只适合 120/1k；见本文件 docstring 与 `stage1_visual_router_mainline.md` |
| `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` | `MAINLINE_VISUAL_ROUTER` | 正式 full-scale online streaming Visual Router；见本文件 docstring 与 `README.md` |
| `visual_router_experiments/stage1_vali_test_router/validate_full_scale_oracle_tsf_outputs.py` | `MAINLINE_VISUAL_ROUTER` | 正式 oracle/TSF 覆盖与 join 验证；见本文件 docstring |

## 6. UNKNOWN_NEEDS_REVIEW

本次逐文件审计没有发现需要标记为 `UNKNOWN_NEEDS_REVIEW` 的 Python 文件。36 个文件均能从文件 docstring、目录 README、主线文档、导入关系或实验日志确认职责。该结论仅表示“当前角色可判断”，不表示所有实现都已适合长期保留。

## 7. 审计后的重构方向

后续应先收束共享数据平面和评估平面，再考虑统一训练骨架。最终理想边界是：共享 reader 产生统一 sample/prediction/target batch，仅由 `VisualFeatureProvider` 在线生成 ViT embedding，或由 `TimeFuseFeatureProvider` 读取离线 17 维 cache；MLP 与 Linear-softmax 继续保持独立。详细候选、依赖和迁移顺序见 `docs/refactor/stage1_migration_candidates.md`。
