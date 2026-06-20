# Stage 1 P15d Branch Small Entrypoint Artifact Parity

创建日期：2026-06-21

## 1. 目标

P15d 新增 branch-specific small entrypoint artifact parity smoke，用同一批仓库内
small fixture 分别运行：

- `scripts/run_stage1_timefuse_small.py`
- `scripts/run_stage1_visual_small.py`

然后比较两边写出的 canonical `run_dir` 共同结构、关键 schema、字段命名、ordered
sample_keys 和边界约束。P15d 的重点不是比较 TimeFuse-style fusor 与 Visual Router
的指标优劣，而是在正式迁移前防止两个 small entrypoint 的 runtime artifact contract
继续分叉。

## 2. 放置时机

P15d 放在 P15b/P15c 之后、真实迁移之前，原因是：

1. P15b 已证明 TimeFuse-specific small entrypoint 可以从 17 维 feature fixture 到
   `TimeFuseLinearSoftmaxHead` 再到 Runtime artifact writer 写出 canonical run_dir。
2. P15c 已证明 Visual-specific small entrypoint 可以从 `VisualMockFeatureProvider` 到
   script-local smoke-only MLP adapter 再到 Runtime artifact writer 写出 canonical run_dir。
3. 在接真实 Visual RouterHead、真实 checkpoint、scaler、ViT provider、full-scale reader
   或 TimeFuse streaming 入口之前，需要先锁定两条支线共同 artifact schema，避免后续正式
   迁移把 schema drift 带入大规模运行。

## 3. 共同 Artifact Schema

P15d smoke 至少检查两边 canonical `run_dir` 共同存在：

- `run_metadata.json`
- `run_status.json`
- `inputs/sample_manifest_ref.json`
- `inputs/split_summary.json`
- `evaluation/evaluation_summary.json`
- `predictions/prediction_rows.csv`
- `logs/` 下各自最小日志文件
- `inputs/`、`indexes/`、`predictions/`、`evaluation/`、`checkpoints/`、`logs/`
  canonical 子目录

共同 `run_metadata.json` 字段包括：

- `run_artifact_schema_version`
- `protocol_version`
- `sample_manifest_schema_version`
- `evaluation_schema_version`
- `config_name`
- `branch_name`
- `created_at`
- `inputs`

共同 `inputs` 字段包括：

- `sample_manifest`
- `split_summary`
- `expert_predictions_json`

共同 `run_status.json` 字段包括：

- `status == completed`
- `current_stage`
- `updated_at`
- `failure_reason is None`
- `checkpoint_pointer is None`

共同 `evaluation/evaluation_summary.json` 字段包括：

- `evaluation_schema_version`
- `sample_count`
- `metrics`
- `selected_counts`
- `model_columns`

其中 `metrics` 至少包含：

- `hard_mae`
- `hard_mse`
- `raw_soft_mae`
- `raw_soft_mse`
- `mean_entropy`
- `mean_max_weight`

共同 `predictions/prediction_rows.csv` 字段包括：

- `sample_key`
- `split`
- `selected_model`
- `selected_index`
- `y_true`
- `y_pred`
- `hard_mae`
- `hard_mse`
- `raw_soft_mae`
- `raw_soft_mse`
- `max_weight`
- `weight_entropy`

## 4. Branch-Specific 字段

P15d 允许且要求 TimeFuse-specific metadata 保留：

- `branch_name == timefuse_fusor_small`
- `inputs.features_csv`
- `inputs.features_csv.feature_dim == 17`
- `timefuse_fusor`
- `timefuse_fusor.training == not_started_p15b_small_rehearsal_only`

P15d 允许且要求 Visual-specific metadata 保留：

- `branch_name == visual_router_small`
- `inputs.history_windows_json`
- `visual_router`
- `visual_router.training == not_started_p15c_small_rehearsal_only`
- `visual_router.formal_visual_router_migration is False`
- `visual_router.loads_real_checkpoint is False`
- `visual_router.loads_real_vit is False`
- `visual_router.feature_provider == VisualMockFeatureProvider`

这些字段是支线职责差异，不应被强行收敛成同一个 feature/head metadata。

## 5. 不比较指标优劣

P15d 只要求两边基于同一 ordered `sample_key`、同一 `ExpertBatch` small fixture 和相同
`model_columns` 写出结构一致的 canonical artifact。TimeFuse 与 Visual 使用不同
branch-specific Feature/Head，因此：

- 不要求 `hard_mae`、`hard_mse`、`raw_soft_mae`、`raw_soft_mse` 数值一致。
- 不要求 `selected_counts` 数值一致。
- 只要求 `sample_count` 一致、`model_columns` 一致、指标字段存在且为有限数值。

## 6. 明确不做

P15d 不是正式训练迁移：

- 不修改正式 Visual Router / TimeFuse fusor 训练入口。
- 不访问 `/data2`。
- 不启动 full-scale、pressure 或训练。
- 不读取真实 checkpoint。
- 不启动真实 ViT embedding。
- 不把 P15c smoke-only adapter 提升为正式 adapter。
- 不为兼容旧版 `96_48_S` full-scale 输出 schema 写适配逻辑。

## 7. 后续

P15d 通过后，可以继续进入更靠近正式迁移的独立小步，例如：

- Visual RouterHead adapter design/smoke。
- real Visual feature provider audit。
- TimeFuse formal streaming入口的 artifact writer 接入审计。

这些后续步骤仍应保持 small-first、schema-first，不在同一步里接真实训练入口和 full-scale。
