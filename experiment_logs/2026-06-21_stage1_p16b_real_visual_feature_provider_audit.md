# Stage 1 P16b real Visual feature provider 边界审计

日志日期：2026-06-21 08:50:04 CST

## 目的

审计并冻结 Stage 1 real Visual feature provider 的边界，明确 head-ready `FeatureBatch`
从 history window、pseudo image、frozen ViT embedding 和可选 scaler transform 中产生时，
各层组件、Runtime、RouterHead adapter 和 cache 的责任分工。

## 背景

P16a 已新增正式 `LoadedTorchMLPRouterHeadAdapter`，其边界是：

```text
已加载 torch.nn.Module + head-ready FeatureBatch -> RouterOutput
```

下一步若迁移真实 Visual Router，必须先明确 `FeatureBatch` 的来源。legacy streaming 入口仍把
Quito history window 读取、pseudo image 构造、ViT forward、pooling、`StandardScaler`、
Visual MLP、checkpoint、device、DataParallel 和 runtime artifact 写出混在同一正式入口中。
本步只做文档审计，不新增 provider，不接真实 ViT，不访问 `/data2`，不迁移正式入口。

## 操作

1. 阅读任务目标文件，确认本步要求为 P16b docs-only boundary audit，且最终需提交推送。
2. 复核既有相关文档：
   - `docs/refactor/stage1_visual_feature_provider_insertion_audit.md`
   - `docs/refactor/stage1_visual_feature_provider_mock_smoke.md`
   - `docs/refactor/stage1_visual_legacy_mlp_adapter_audit.md`
   - `docs/refactor/stage1_visual_mlp_routerhead_adapter.md`
3. 只读查看当前 small/mock 代码边界：
   - `scripts/run_stage1_visual_small.py`
   - `time_router/features/visual_mock.py`
4. 使用 `rg` 和 `sed` 只读查看 legacy Visual 相关链路，重点确认：
   - `make_pseudo_images(...)` 和 `pool_vit_outputs(...)`
   - `windows_from_labels(...)`
   - `build_vit_model(...)` / `load_vit_model_with_retry(...)`
   - `iter_online_embedding_batches(...)`
   - `StandardScaler.partial_fit` / `scaler.transform`
   - `VisualMLPRouter`
5. 新增 `docs/refactor/stage1_real_visual_feature_provider_audit.md`，冻结输入、输出、
   scaler、ViT/device/batching、feature cache、P16a adapter 关系和后续 small-first 拆分。
6. 更新 `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md`、`WORKSPACE_STRUCTURE.md` 和
   `experiment_logs/README.md`，记录 P16b 状态。

## 结果

新增 P16b 审计文档，明确：

- real Visual provider 不从 `run_dir` 推断输入，不硬编码 `/data2`，不读取 checkpoint，
  不启动训练，不知道 Bash。
- 输入边界包括 `SampleManifest` / ordered sample_keys、history window source、pseudo image
  参数、ViT/encoder 配置和显式 scaler/normalizer state。
- 输出边界是 canonical `FeatureBatch`，`sample_keys` 保序，`features` 为二维 float32，
  `feature_schema` 记录 encoder/provider lineage，`extra` 只保留轻量 metadata。
- scaler fit 属于 training/runtime，不属于 evaluation/test-time provider；scaler state loading
  属于 Runtime/entrypoint；scaler transform 必须显式设计，不允许 RouterHead adapter 偷偷处理。
- ViT loading、device、dtype、batch size、DataParallel、Hugging Face cache 和 retry 由
  Runtime/entrypoint/config 或 encoder factory 管理。
- feature cache 只是实现选择，长期接口仍是 `FeatureBatch`，不把 cache path/shard/SQLite/NPY/
  Parquet 格式设计成 provider interface。
- P16b 不修改 P16a adapter，不声称正式 Visual Router 已迁移完成。

本步未新增代码，未修改正式入口，未读取真实 checkpoint，未访问 `/data2`，未启动 ViT、
训练、pressure 或 full-scale。

## 结论

P16b 将真实 Visual feature chain 拆成候选层：`HistoryWindowProvider` /
`VisualRawInputProvider`、`PseudoImageTransformer`、`VisualEncoderProvider` /
`FrozenViTFeatureProvider`、`FeatureScaler` / `FeatureNormalizer` 和组合型
`VisualFeatureProvider`。后续迁移应先做 small smoke，再逐步处理 scaler、legacy checkpoint
signature、online ViT provider 和正式 entrypoint。

本步是 docs-only，不改变 runtime 行为，因此不需要运行 Python smoke；轻量验证以
`git diff --name-only` 和 P16b 文档关键词检索为准。

## 下一步方案

1. 运行轻量检查：
   - `git diff --name-only`
   - `rg -n "FeatureBatch|scaler|ViT|pseudo|run_dir|cache|P16a|LoadedTorchMLPRouterHeadAdapter" docs/refactor/stage1_real_visual_feature_provider_audit.md`
2. 确认 diff 仅包含文档和实验日志。
3. 以 `docs: audit stage1 real visual feature provider boundary` 提交并推送到
   `origin/refactor/stage1-route-audit`。
4. 后续可单独做 real Visual feature provider minimal smoke plan、scaler boundary smoke、
   legacy `VisualMLPRouter` checkpoint/signature audit、online ViT provider smoke 和正式
   Visual full-scale entrypoint migration plan。
