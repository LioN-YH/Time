# Stage 1 P4 后架构转向决策

创建日期：2026-06-19

## 1. 目标

本文在 P4d run artifacts 边界复核和 P4e checkpoint/resume 边界复核之后，从“兼容历史输出”切换到“收束新架构主干”的视角，明确 Stage 1 后续 canonical entrypoint、历史入口归档策略、可继承 schema、可舍弃历史 schema、最小 runtime 契约和 helper 接入方式。

本次只做架构决策文档；不修改任何训练脚本，不迁移入口，不实现 config system，不实现 checkpoint index，不接入 `/data2`，不移动或删除历史代码。

## 2. 决策摘要

P4f config system 暂停，后续优先进入 P5 canonical entrypoint design / FeatureProvider interface design。原因是当前最关键的架构风险已经不是 JSON/path/config 默认值，而是正式入口太多、历史路线混杂、full-scale 只能依赖 streaming/shard-aware 入口。继续向下兼容所有历史 schema 会把新主干拖回旧路线。

后续 Stage 1 只保留两条正式训练主干：

1. Visual Router 主线：`train_visual_router_online_streaming.py`，固定为 `x -> pseudo image -> frozen ViT -> router`。
2. TimeFuse-style fusor baseline 支线：`train_timefuse_fusor_streaming.py` + `launch_timefuse_fusor_full_scale.py`，固定为 `sample_key -> 17维 TimeFuse feature cache -> Linear-softmax fusor`。

其它历史入口只保留为 archive/deprecated/reference-only，不再为它们新增兼容 helper，也不把它们的状态文件反向塑造成新 schema。

## 3. Canonical Entrypoint

### 3.1 Visual Router 主线

正式 canonical training/evaluation entrypoint：

- `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`

配套前置和后置入口：

- `build_full_scale_sample_manifest.py`
- `launch_full_scale_prediction_cache.py`
- `build_prediction_cache_from_manifest.py`
- `merge_prediction_cache_shards.py`
- `build_full_scale_window_oracle_labels.py`
- `build_full_scale_tsf_enrichment.py`
- `validate_full_scale_oracle_tsf_outputs.py`
- `evaluate_soft_fusion_calibration.py`

保留理由：

- 已完成 full-scale `96_48_S` train-only checkpoint 和独立 eval-only。
- 具备 checkpoint/resume、train-only、eval-only、SQLite disk index、batch query 和 `packed_npy_v1` row index 读取经验。
- 满足正式视觉路线约束：full-scale 在线生成 pseudo image 和 ViT embedding，不长期保存伪图像 tensor 或 ViT embedding cache。

### 3.2 TimeFuse-style Fusor Baseline 支线

正式 canonical baseline entrypoint：

- `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
- `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py`

配套 reader 和 feature cache 入口：

- `stage1_timefuse_fusor_streaming_reader.py`
- `build_timefuse_feature_cache_from_manifest.py`
- `launch_timefuse_feature_cache_full_scale.py`

保留理由：

- 对齐 TimeFuse-style 单层 softmax fusor 和 SmoothL1 fusion loss 口径。
- 已明确 feature-only scaler、split 下推、shard-local SQLite index、batch-level grouped packed npy 读取和 CUDA 双卡训练策略。
- 作为 baseline 支线独立于 Visual Router 主线，不再作为视觉主线前置步骤。

## 4. 历史入口归档策略

### 4.1 Deprecated / Reference-Only

以下入口或路线应标记为 deprecated/reference-only，未来有替代文档、日志和 smoke 证据后再移动到 `archive/`：

- LogisticRegression fusor：`pilot/train_structure_router_pilot.py` 只代表早期 hard-label 结构特征 router，不再作为正式 TimeFuse-style fusor。
- Offline ViT embedding cache：`pilot/build_vit_embeddings_pilot.py`、`pilot/launch_96_48_s_1k_vit_embedding_pilot.py` 和离线 embedding `.npy` builder 口径只保留小规模 debug / 历史复现价值。
- 旧 OOM lookup 路线：旧 `_1epoch/` full-scale 输出和全量 Python manifest lookup 只作为失败经验，不作为有效结果或兼容目标。
- Pilot-only 脚本：固定 120、1k、小样本、dry-run 或历史验证脚本保留复现价值，但不作为新架构主入口。
- 非 streaming / 早期 online full-scale 入口：`train_visual_router.py`、`train_visual_router_online.py` 可保留小规模复现和共享函数价值，但不再承担 full-scale canonical training。

### 4.2 不再新增兼容的内容

后续不再为了兼容上述历史路线新增 helper：

- 不新增离线 embedding cache adapter。
- 不为 LogisticRegression hard-label router 保持新 output schema。
- 不为旧 OOM 目录补 status/metadata/checkpoint index。
- 不为 `pilot/` 固定规模 launcher 设计 canonical runtime。
- 不把非 streaming 入口升级为 full-scale 训练主干。

## 5. 可继承的 Output Schema

以下 schema 值得继承到新 canonical runtime：

- `sample_key` 及其稳定组成：`config_name + split + dataset_name + item_id + channel_id + window_index`。
- 固定五专家顺序：`DLinear`、`PatchTST`、`CrossFormer`、`ES`、`NaiveForecaster`。
- prediction cache 的 `packed_npy_v1` 少文件 shard 口径、`y_true_row_index`、`y_pred_row_index`、`sample_key + model_name` 唯一性和共享 `y_true` 校验。
- oracle labels 的 `sample_key + metric` 口径，以及 TSF enrichment 按 `sample_key` join 的诊断口径。
- evaluation 输出中的 hard top-1、raw soft fusion、selected model counts、per-sample hard/raw-soft MAE/MSE、oracle top-1 对照和 calibration 固定策略汇总。
- checkpoint 目录中 latest checkpoint 文件和 `latest_checkpoint_index.json` 的概念，但不继承两条路线的字段名差异为全局公共 schema。

## 6. 可舍弃的历史 Schema

以下历史 status/metadata/checkpoint schema 不再强兼容：

- 非 streaming Visual Router 的 `visual_router_metadata.json` / `visual_router_online_metadata.json` 中只服务小规模运行摘要的字段。
- LogisticRegression baseline 输出中的 legacy/deprecated 标记和 hard-label 分类专用字段。
- Pilot launcher 的固定 1k、120 sample、dry-run 路径字段。
- 旧 OOM 目录中的 `running/training` 残留 status 和缺失 checkpoint 的状态文件。
- Prediction cache builder 的 `--resume` / `checkpoint_selection` 字段作为 router/fusor checkpoint resume schema。
- Visual Router 的 `completed_epochs` 与 TimeFuse fusor 的 `completed_epoch` 不强行统一；未来若统一，只能在新 runtime schema 中并行定义，不回写历史目录。

## 7. Canonical Runtime 最小契约

新 Stage 1 canonical runtime 每个正式 run 至少应包含：

| 路径 | 最小要求 |
| --- | --- |
| `run_dir/` | 单次运行独立目录；不能覆盖已完成正式结果；记录 entrypoint、config、输入路径和输出路径 |
| `status.json` | 记录 `status`、`phase`、`updated_at`、`run_dir`、关键进度、错误信息和可恢复状态；旧字段可兼容保留，新 runtime 不反向兼容所有历史字段 |
| `metadata.json` | 记录 `entrypoint`、`config_name`、`args`、`inputs`、`outputs`、`model_columns`、`array_storage`、`feature_schema`、`checkpoint` 和资源策略 |
| `checkpoints/` | 训练型入口保存 epoch checkpoint、latest checkpoint 和 latest index；eval-only 或纯评估入口可没有训练 checkpoint，但必须在 metadata 中说明 |
| `predictions/` 或 evaluation outputs | 保存 per-sample predictions、soft fusion predictions、summary、comparison 或 calibration 输出；字段应由 evaluation helper 统一复算 |
| `logs/` 或 `main.log` | 保存主日志、launcher 日志、必要时保存 lane 日志；后台长跑必须有停止和恢复命令留痕 |

最小状态机建议为：

```text
init -> preflight -> index/scaler -> train -> checkpoint_saved -> eval -> done
```

不同入口可以跳过不适用阶段，例如 eval-only 可从 `checkpoint_loaded` 进入 `eval`，prediction cache builder 可使用 shard/cache 专用阶段；但 canonical training runtime 不应再把 run artifacts、checkpoint、resume、launcher 语义散落到互不兼容的临时字段里。

## 8. Helper 接入方式

### 8.1 P4 Helper

P4a/P4b/P4c helper 只作为底层能力接入：

- `atomic_write_json(...)`：用于原子写 `status.json`、`metadata.json`、checkpoint index。
- `resolve_status_path(...)` / `resolve_metadata_path(...)`：用于计算 run artifacts 路径。
- `build_run_metadata(...)`：可作为新 runtime metadata 的基础 payload，不替代历史正式 metadata。

P4 helper 不负责 run_dir 命名、launcher、checkpoint payload、resume policy、best/latest 选择或 logging framework。

### 8.2 Evaluation Helper

`time_router.evaluation` 应接入 canonical eval/report 阶段：

- 所有 hard top-1、raw soft fusion、MAE/MSE、selected counts、entropy、max weight、summary 和 per-sample rows 都从同一批 `y_pred/y_true/weights` 复算。
- 当前 helper 的 summary/rows 是内存稳定结构，不等价于正式 CSV schema；正式入口迁移时应先做字段 mapping 文档。

### 8.3 PredictionBatchReader

`PredictionBatchReader` 继续作为小规模 fixture 和统一 batch reader 的公共契约来源：

- 用于 golden smoke 和 schema 等价验证。
- full-scale canonical runtime 可以复用其字段约束和 row index 校验，但训练入口仍需要 streaming/shard-aware reader，不能回退到全量 Python lookup。

### 8.4 OracleTsfReader

`OracleTsfReader` 用于监督、上限、baseline、分层汇总和诊断：

- 不进入可部署 FeatureProvider 的 test-time 动态特征。
- 小规模 smoke 可直接使用 reader。
- full-scale canonical runtime 应采用 SQLite / shard-local / batch query 或等价批查询方案，不允许无界全扫描。

## 9. P4f 与 P5 决策

暂停 P4f config system，转向 P5 canonical entrypoint design。

理由：

- 继续实现 config system 会过早固化历史入口和旧输出字段，增加迁移负担。
- P5 需要先明确 VisualFeatureProvider 与 TimeFuseFeatureProvider 的输入输出、设备策略和 feature metadata；这些接口稳定后再设计 config 默认值更可靠。
- canonical entrypoint 的 run_dir/status/metadata/checkpoint/evaluation 契约应先定下来，再决定哪些参数进入共享 config。

新的后续顺序建议：

1. P5a：canonical runtime contract + entrypoint design，只写设计，不迁移代码。
2. P5b：FeatureProvider interface design，明确 Visual/TimeFuse 两条 feature 分支。
3. P5c：new runtime status/metadata/checkpoint schema draft。
4. P6：在小规模 smoke 目录逐步迁移正式入口。

## 10. 是否允许重跑实验

允许为了更干净的代码框架重跑实验，但必须满足：

- 不覆盖或回写已有可引用正式结果目录。
- 新 run_dir 明确标注 runtime/schema/entrypoint version。
- 先通过 golden smoke、oracle/TSF smoke、P4 helper smoke 和 compileall。
- 先跑小规模或 pressure smoke，再启动 full-scale。
- 重跑结果与旧结果同表比较，不能用新结果静默替换旧结论。

## 11. 本次明确不做

- 不改任何训练脚本。
- 不迁移入口。
- 不实现 config system。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不为了兼容历史输出新增 helper。
- 不改变 `PredictionBatchReader`、`OracleTsfReader`、evaluation helper 或 IO helper 行为。
- 不移动、删除历史代码。
- 不改模型结构、loss 或正式输出目录。

## 12. 验收

本文档属于 architecture pivot review。验收重点是：

- 文档明确 canonical entrypoint 和 archive/deprecated/reference-only 策略。
- 路线图从 P4f config system 转向 P5 canonical entrypoint design。
- 结构索引和中文实验日志同步更新。
- 运行既有 smoke 和 compileall，证明本次文档变更没有改变 reader、evaluation、IO helper 或训练入口行为。
