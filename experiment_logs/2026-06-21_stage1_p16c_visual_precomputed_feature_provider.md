# Stage 1 P16c Visual Precomputed FeatureProvider

日志日期：2026-06-21 10:58:03 CST

## 目的

新增一个最小 precomputed/head-ready Visual FeatureProvider，用仓库内 tiny fixture 读取已预计算的 visual embedding，并输出 canonical `FeatureBatch`。本步只验证 provider 输出边界和 P16a adapter 消费链路，不接真实 ViT、不构造 pseudo image、不处理 scaler、不读取 checkpoint、不迁移正式入口。

## 背景

P16a 已新增 `LoadedTorchMLPRouterHeadAdapter`，证明 head-ready `float32 FeatureBatch.features` 可以进入已加载 torch MLP 并输出 `RouterOutput`。P16b 已审计真实 Visual feature provider 边界，将完整链路拆成 history window、pseudo image、frozen ViT、可选 scaler/normalizer 和最终 `FeatureBatch`。P16c 选择最窄实现：precomputed head-ready embedding CSV -> `FeatureBatch`，先把 `time_router/features` 中的 Visual precomputed provider 边界落地。

## 操作

1. 阅读任务文件 `/home/shiyuhong/.codex-tianyu/attachments/0de4846f-0b4d-413a-9f85-424f0d9e9bd6/pasted-text-1.txt`，确认本步范围为 P16c，禁止修改正式训练入口、禁止访问 `/data2`、禁止真实 ViT/scaler/checkpoint。
2. 检查现有 `FeatureBatch`、`VisualMockFeatureProvider`、`TimeFuseFeatureCacheProvider`、`LoadedTorchMLPRouterHeadAdapter`、P13b manifest/expert fixture 和相关 smoke 风格。
3. 新增 `time_router/features/visual_precomputed.py`：
   - 实现 `VisualPrecomputedFeatureProvider`；
   - 初始化读取显式 CSV，自动识别 `feature_` 前缀列；
   - 校验 sample_key 非空且唯一、feature columns 非空、特征值可转 float 且有限；
   - `load_batch(sample_keys)` 按请求顺序输出 `np.float32` features，并填充 `head_ready=True`、`precomputed=True`、`loads_real_vit=False`、`handles_scaler=False` 等 schema。
4. 更新 `time_router/features/__init__.py`，导出 `VisualPrecomputedFeatureProvider`。
5. 新增 tiny fixture `tests/fixtures/stage1_visual_precomputed_small/visual_embeddings.csv` 和 README：
   - 覆盖 P13b manifest 的 4 个 sample_key，包括 test split 两个 sample_key；
   - CSV 行顺序故意不同于 manifest；
   - 只包含 `sample_key` 与 `feature_0 ... feature_7` 固定数值。
6. 新增 `tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py`：
   - 读取 P13b manifest ordered sample_keys；
   - 用 P16c fixture 构造 `FeatureBatch`；
   - 用 P13b expert JSON 构造 `ExpertBatch`；
   - 构造内存 tiny torch MLP，交给 P16a `LoadedTorchMLPRouterHeadAdapter`；
   - 调用 `EvaluationInputAdapter` 生成 summary 和 per-sample rows；
   - 覆盖 missing sample_key、重复 fixture sample_key、非有限 feature、provider 不持有 `run_dir`、patch `torch.load`、rows 保序等负向和边界检查。
7. 新增 `docs/refactor/stage1_visual_precomputed_feature_provider.md`，说明 P16c 目标、与 P16a/P16b 的关系、FeatureBatch metadata、不做范围、验收命令和后续步骤。
8. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`，登记 P16c provider、fixture、smoke 和文档。

## 结果

新增 smoke 与回归均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_precomputed.py tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
```

关键验证结果：

- P16c provider 输出 `FeatureBatch(features=(4, 8), dtype=float32)`；
- 输出 sample_key 保持 P13b manifest 行顺序，不受 fixture CSV 行顺序影响；
- fixture 覆盖 P13b test split 两个 sample_key；
- `feature_schema` 记录 `precomputed=True`、`head_ready=True`、`loads_real_vit=False`、`handles_scaler=False`；
- missing sample_key、重复 fixture sample_key、non-finite feature fixture 均 fail-fast；
- provider 不持有 `run_dir`；
- P16a adapter 可消费 P16c `FeatureBatch` 并输出 `RouterOutput`；
- `EvaluationInputAdapter` 可生成 summary/rows，per-sample rows 保持 sample_key 顺序；
- patch `torch.load` 后未触发 checkpoint 读取；
- `experiment_logs/run_outputs/` 未新增 run_dir；
- `git diff --name-only` 显示未修改 `train_visual_router_online_streaming.py`、`train_timefuse_fusor_streaming.py`、small entrypoint 或 P16a adapter。

## 结论

P16c 已完成最小 precomputed/head-ready Visual FeatureProvider 边界实现和 smoke 验证。该 provider 只读取显式 small fixture 并输出 canonical `FeatureBatch`，可以被 P16a `LoadedTorchMLPRouterHeadAdapter` 和 `EvaluationInputAdapter` 消费。它不代表真实 ViT provider、pseudo image、scaler、checkpoint loading 或正式 Visual Router 入口迁移已经完成。

## 下一步方案

1. 提交并推送 `refactor/stage1-route-audit`。
2. 后续单独做 scaler boundary smoke，验证 loaded scaler transform -> head-ready `float32 FeatureBatch`，并禁止 silent fit。
3. 后续单独做 fake encoder provider 或 online ViT provider audit/smoke，处理 pseudo image、frozen ViT、batching、device/dtype/resource policy。
4. 后续单独审计 legacy `VisualMLPRouter` checkpoint/signature/state_dict/DataParallel key。
5. feature/head/runtime 边界更稳定后，再规划正式 Visual entrypoint migration。
