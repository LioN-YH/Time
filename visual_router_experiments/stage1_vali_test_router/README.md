# stage1_vali_test_router

本目录保存 Stage 1 主实验代码：在 `vali` split 训练 router，在 `test` split 评估 frozen experts 之间的 hard top-1 routing 与 soft fusion。

Stage 1 的基本约束：

- 路由粒度是 `config_name + split + dataset_name + item_id + channel_id + window_index`；
- 每个 router / baseline 必须在同一个 `config_name` 内训练和评估；
- 正式 visual 主线固定为 `x -> pseudo image -> frozen ViT -> router`；
- full-scale 不长期保存伪图像 tensor 或 ViT embedding `.npy`；
- prediction cache、router 输出和 calibration 结果写入 `experiment_logs/run_outputs/` 或 `/data2/syh/Time/run_outputs/`，不写入代码目录。

视觉路由主线见 `stage1_visual_router_mainline.md`，详细字段契约见 `stage1_cache_contract.md`，完整历史结果索引见 `stage1_history_results.md`。

## 当前主线

当前 visual router full-scale 主线应按以下顺序推进：

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
-> final unified report
```

截至最近一次检查，`96_48_S` full-scale sample manifest、五专家 prediction cache、merged cache、oracle labels、TSF enrichment、1 epoch streaming visual router checkpoint 和 checkpoint eval-only 均已完成。eval-only 结果覆盖 `13,924,650` 个 test window：visual hard top-1 MAE=`0.5615367653135453`，raw soft fusion MAE=`0.5174675759559787`，oracle MAE=`0.33862214116809347`。下一步只看视觉主线时，应优先确认 `evaluate_soft_fusion_calibration.py` 的 full-scale streaming/SQLite 读取口径，然后对已完成 eval-only 输出做 calibration 和视觉主线报告。

TimeFuse-style fusor 是 baseline 支线，相关 reader、feature cache 和 GPU2/3 launcher 单独追踪；它不再作为视觉路由主线的前置步骤。

## 入口分层

### Full-Scale 正式入口

| 文件 | 功能 |
| --- | --- |
| `build_full_scale_sample_manifest.py` | 流式枚举 `96_48_S` vali/test 全候选窗口，生成 `sample_manifest_shard_index.csv` 和 `sample_shards/*.csv`；不启动专家推理、不保存预测或 embedding |
| `launch_full_scale_prediction_cache.py` | 基于 full-scale sample shard index 生成可恢复 launcher；DLinear/PatchTST/CrossFormer 绑定 GPU，ES/NaiveForecaster 走 CPU，默认 `packed_npy_v1` |
| `build_prediction_cache_from_manifest.py` | 读取 sample manifest 和专家 evaluate config，生成单专家或多专家 prediction cache shard；支持 `per_sample_npy` 和 full-scale 推荐的 `packed_npy_v1` |
| `merge_prediction_cache_shards.py` | 合并 prediction cache shards，校验 `sample_key + model_name` 唯一、五专家完整、共享 y_true 一致；packed 模式会重建 merged y_true row index |
| `evaluate_router_baselines.py` | 统一 baseline 入口；用 vali 学 global/dataset/TSF-cell/dataset+TSF-cell 统计规则，并可训练 TimeFuse-style 单层 fusor，在 test 上输出 comparison |
| `train_visual_router_online_streaming.py` | full-scale online visual router 主入口；batch 内生成 ViT embedding，scaler 只在 vali 上 `partial_fit`，test 流式 forward，不落盘 embedding |
| `evaluate_soft_fusion_calibration.py` | 对 router test 权重做 raw soft、temperature scaling、top-k truncation、top1 hard、top2/top3 fusion 统一评估 |
| `run_full_scale_dry_run.py` | 小样本验证 full-scale 框架闭环和恢复语义；用于模板检查，不作为正式指标 |

### Baseline 支线入口

| 文件 | 功能 |
| --- | --- |
| `build_timefuse_feature_cache_from_manifest.py` | TimeFuse-derived 单变量 feature cache builder；读取单个 sample manifest shard，重新加载历史窗口 `x` 提取 17 维元特征；不读取未来 `y`、专家预测或 oracle label |
| `launch_timefuse_feature_cache_full_scale.py` | 基于 full-scale sample shard index 生成/启动 TimeFuse feature cache 多 lane CPU launcher |
| `stage1_timefuse_fusor_streaming_reader.py` | TimeFuse-style fusor 的 streaming/shard-aware 数据读取层和 smoke 入口；支持 split 下推、packed npy batch-level grouped loading，并在大块 CSV 读取后切成稳定 batch，避免训练前误读非目标 split arrays |
| `train_timefuse_fusor_streaming.py` | TimeFuse-style fusor 的 streaming train/eval 入口；支持 shard-local index 复用、feature-only scaler 和 CUDA 双卡 `DataParallel` |
| `launch_timefuse_fusor_full_scale.py` | TimeFuse-style fusor 64-shard baseline 后台 launcher；正式公平比较使用 `--device cuda --cuda-visible-devices 2,3`，当前正式目录 `/data2/syh/Time/run_outputs/2026-06-18_stage1_timefuse_fusor_full_scale_gpu23/` 已进入 train |

### 中小规模和复现入口

| 文件 | 功能 |
| --- | --- |
| `build_stage1_sample_manifest.py` | 生成 `96_48_S` 1k manifest-only 样本清单；用于中等规模实验和快速复现 |
| `build_visual_router_v2_pilot_samples.py` | 从 full-scale oracle labels 与 TSF enrichment parquet 构建 Visual Router V2 固定 pilot sample keys；不读取 116M 行 merged prediction manifest，不训练模型；当前 v1 输出在 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/` |
| `train_visual_router_online.py` | 适合 120/1k 规模；在线生成 ViT embedding 后在运行内暂存全部 embedding，再复用 MLP router 训练和评估逻辑 |
| `train_visual_router.py` | 离线 embedding manifest 训练入口，也提供 online wrapper 复用的 MLP router、loss、hard/soft fusion 评估函数；当前正式路线不鼓励长期保存 ViT embedding `.npy` |

### 共享库

| 文件 | 功能 |
| --- | --- |
| `fusion_utils.py` | 共享 prediction manifest 读取、五专家数组读取、hard/soft fusion 指标复算，以及 TimeFuse-style 单层 fusor 训练和评估逻辑；不是命令行入口 |
| `__init__.py` | Python package 标记 |

### 文档

| 文件 | 功能 |
| --- | --- |
| `stage1_cache_contract.md` | 固定 prediction cache、oracle labels、feature cache、router evaluation 字段契约和 full-scale shard/resume 约定 |
| `stage1_visual_router_mainline.md` | 只梳理 Stage 1 visual router 主线、`96_48_S` 已沉淀的正确路线、废弃路线和扩 config 标准步骤 |
| `stage1_protocol_and_plan.md` | 记录 per-config 主实验协议、Stage 1B 迁移实验设计、已完成事项和后续验收标准 |
| `visual_router_v2_pilot_protocol.md` | 基于 Visual Router / TimeFuse-style full-scale 结果定义 V2 架构诊断、小规模分轮消融、ViT domain adaptation、经济性门禁和独立 worktree 实验协议 |
| `prediction_cache_design.md` | 早期 Quito evaluate/data/model 数据流阅读记录，以及 prediction cache 导出点设计 |
| `feature_and_rl_extension_notes.md` | TimeFuse-style 结构特征支线、contextual bandit/RL 扩展判断和后续研究路线 |
| `stage1_timefuse_fusor_streaming_reader_design.md` | 固定 full-scale TimeFuse-style fusor reader 的输入输出契约、SQLite 索引策略、并行策略和 1-shard smoke 验证结果 |
| `stage1_history_results.md` | 历史 smoke、1k、dry-run、full-scale 长跑状态和代表输出目录索引 |

## Pilot 目录

`pilot/` 只保留历史验证、小规模对照和固定规模 launcher。正式 full-scale 主线不应长期依赖 `pilot/` 中的离线 embedding cache 或 1k launcher。

| 文件 | 功能 | 当前定位 |
| --- | --- | --- |
| `pilot/build_prediction_cache_pilot.py` | 小规模生成 window-level prediction cache，验证专家预测、窗口 key、数组落盘和 MAE/MSE 对齐 | 历史 pilot |
| `pilot/compute_window_oracle_from_cache.py` | 基于 manifest 计算 window-level oracle label、expert regret 和 best-single-vs-oracle 汇总 | 仍可复用 |
| `pilot/enrich_cache_with_tsf_cell.py` | 为 manifest/oracle labels 合并 TSF cell 元信息，并生成分层 oracle summary | 仍可复用 |
| `pilot/build_online_pseudo_image_pilot.py` | 重新加载 Quito 历史窗口 x，在线生成 3view/top3fold 伪图像并记录 index、metadata、latency 和少量 debug PNG | 历史/调试 |
| `pilot/build_vit_embeddings_pilot.py` | 离线冻结 HF ViT embedding cache smoke，会输出 embedding manifest 和 `.npy` | 旧路线；不作为 full-scale 主入口 |
| `pilot/build_structure_feature_cache_pilot.py` | 基于 TimeFuse 单变量元特征生成 window-level 数值结构 feature cache | baseline 支线输入 |
| `pilot/train_structure_router_pilot.py` | 使用 TimeFuse 单变量元特征训练 LogisticRegression hard-label router | legacy/deprecated 历史口径 |
| `pilot/launch_96_48_s_1k_prediction_cache_pilot.py` | 生成 `96_48_S` 1k prediction cache launcher | 1k 固定规模历史入口 |
| `pilot/launch_96_48_s_1k_vit_embedding_pilot.py` | 生成 `96_48_S` 1k ViT embedding cache smoke launcher | 旧路线；当前不启动 |

## 当前不要混用的路线

- 不要把 `pilot/build_vit_embeddings_pilot.py` 或 `pilot/launch_96_48_s_1k_vit_embedding_pilot.py` 当作 full-scale 前置步骤。
- 不要用 `train_visual_router_online.py` 跑 full-scale；它会暂存全量 embedding，适合 120/1k 复现。
- 不要把 `pilot/train_structure_router_pilot.py` 的 LogisticRegression 结果作为新的主 baseline；新的 TimeFuse 对照在 `evaluate_router_baselines.py` 的 TimeFuse-style fusor 中。
- 不要把 `launcher_compat_check/` 当作正式 prediction cache 结果；正式 full-scale 结果以 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/` 为准。

## 下一步

1. Visual Router V2 pilot 优先复用 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/` 中的 ordered sample keys，先完成 Round 0 旧 Visual / TimeFuse-style / global best / oracle 小规模复现。
2. Round 1 在同一 sample keys 上比较 CLS、mean patch、CLS+mean、RevIN aux-only 和 visual+aux，scaler 只在 `pilot_train` fit，架构选择只看 `pilot_selection`。
3. `pilot_test` 只在方案冻结后使用；`diagnostic_balanced` 只做 oracle expert 近似均衡诊断，不替代自然分布主指标。
4. 视觉主线 full-scale 后续仍可独立审查 `evaluate_soft_fusion_calibration.py` 的 streaming/SQLite 读取口径；不要把 V2 pilot 结果直接覆盖旧 full-scale 输出。
