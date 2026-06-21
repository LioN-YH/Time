# Stage 1 P16a Visual MLP RouterHead Adapter

日志日期：2026-06-21 08:36:58 CST

## 目的

新增正式 Visual RouterHead adapter 的最小边界，实现“已加载 torch module + head-ready `FeatureBatch` -> `RouterOutput`”，并用独立 smoke 验证该边界可以被 `EvaluationInputAdapter` 消费。

## 背景

P15b/P15c/P15d 已完成 TimeFuse-specific、Visual-specific small canonical entrypoint 以及两条 branch small entrypoint 的 artifact parity smoke。P15c 里的 `SmokeOnlyVisualMLPAdapter` 仍是 script-local smoke-only rehearsal，不是长期正式 adapter。本步骤需要把正式 Visual MLP RouterHead adapter 放入 `time_router.models`，但不读取真实 checkpoint、不处理 scaler、不启动 ViT、不迁移正式训练入口、不访问 `/data2`。

## 操作

1. 新增 `time_router/models/visual_mlp_adapter.py`，实现 `LoadedTorchMLPRouterHeadAdapter`：
   - 只接收 Runtime 已加载好的 `torch.nn.Module`；
   - 校验 `model_columns` 非空且无重复；
   - 严格要求 `FeatureBatch.features` 为二维 `numpy.float32` head-ready 特征；
   - 校验样本维与 `FeatureBatch.sample_keys` 数量一致；
   - 在 `torch.inference_mode()` 下调用 `model(features)`；
   - P16a 仅支持模型直接返回二维 logits Tensor；
   - 校验 logits shape 为 `[num_samples, num_models]`；
   - 对 logits 沿专家维 softmax，输出 `RouterOutput(logits, weights)`；
   - 在 `extra` 中记录 `loads_checkpoint=False`、`handles_scaler=False`、`handles_vit=False` 等边界信息。
2. 更新 `time_router/models/__init__.py`，导出 `LoadedTorchMLPRouterHeadAdapter`。
3. 新增 `tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py`：
   - 复用 P13b `sample_manifest.csv` 的 ordered sample_keys；
   - 复用 P14b `VisualMockFeatureProvider` 构造 head-ready `float32 FeatureBatch`；
   - 复用 P13b `expert_predictions.json` 构造小型 `ExpertBatch`；
   - 使用内存小型 torch MLP fixture 模拟 Runtime 已加载模型；
   - 串联 `FeatureBatch -> LoadedTorchMLPRouterHeadAdapter -> RouterOutput -> EvaluationInputAdapter -> summary/rows`；
   - patch `torch.load`，若 adapter 或核心路径调用 checkpoint loading 则直接失败；
   - 扫描 adapter 源码，确认未引入 `/data2`、`VisualMLPRouter`、`ViTModel`、`AutoImageProcessor`、`train_visual_router_online_streaming` 或 `checkpoint_path`；
   - 检查 P15c visual small entrypoint 仍保留 script-local `SmokeOnlyVisualMLPAdapter`，未被 P16a adapter 替换；
   - 覆盖 feature dtype 非 `float32`、重复 `model_columns`、logits shape mismatch 三类负向用例。
4. 新增 `docs/refactor/stage1_visual_mlp_routerhead_adapter.md`，记录 P16a 目标、adapter 边界、明确不负责项、与 P15c 的关系、未来正式迁移关系和 smoke 验收。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_entrypoint_migration_plan.md` 和 `WORKSPACE_STRUCTURE.md`，同步 P16a 当前状态与后续连接。

## 结果

已完成新增模块、smoke 和文档更新。以下验证命令通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/models/visual_mlp_adapter.py tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_provider_mock_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mock_protocol_eval_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
git diff --check
```

新增 smoke 输出确认：

- adapter 源码未引入 checkpoint、`/data2`、ViT、legacy `VisualMLPRouter` 或正式训练入口；
- `VisualMockFeatureProvider` 输出 head-ready `float32 FeatureBatch` 且与 `ExpertBatch.sample_keys` 对齐；
- 负向用例均触发预期异常；
- adapter/evaluator 阶段未调用 `torch.load`；
- `RouterOutput` logits/weights 合法；
- `EvaluationInputAdapter` 可生成 summary/rows 且保持 sample_key 顺序。
- 指定 P14b/P14d/P14f/P15c/P15d/P16a 回归 smoke 均通过。
- `git diff --check` 通过。
- `git diff` 确认未修改 `scripts/run_stage1_visual_small.py`、`scripts/run_stage1_timefuse_small.py`、`scripts/run_stage1_canonical_small.py`、`train_visual_router_online_streaming.py` 或 `train_timefuse_fusor_streaming.py`。

## 结论

P16a 已把正式 Visual MLP RouterHead adapter 的最小边界从 smoke-local 代码推进到 `time_router.models`。该 adapter 只消费内存对象，不承担 checkpoint、scaler、ViT、prediction backend、run_dir、训练循环或 launcher 责任。P15c visual small entrypoint 仍保持 script-local smoke adapter，未在本步强行替换。

## 下一步方案

1. 小步提交并推送到 `origin/refactor/stage1-route-audit`。
2. 后续可单独审计 real Visual feature provider，把 history window、pseudo image、frozen ViT provider 和 Runtime resource policy 分层处理。
3. 后续正式 Visual entrypoint 迁移应在 Runtime 中加载 checkpoint/scaler，适配 legacy `VisualMLPRouter` import/signature/state_dict，再把已加载 module 和 head-ready `FeatureBatch` 交给 P16a adapter。
