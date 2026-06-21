# Stage 1 P16d loaded Visual FeatureScaler 边界与 smoke

日志日期：2026-06-21 11:33:29 CST

## 目的

为 Stage 1 canonical 重构主线新增最小 loaded Visual FeatureScaler / FeatureNormalizer
边界，验证“已加载 scaler state + raw/pre-head `FeatureBatch` -> head-ready
`float32 FeatureBatch`”这一单独 transform 步骤。

## 背景

P16a 已新增 `LoadedTorchMLPRouterHeadAdapter`，其输入必须是 head-ready
`float32 FeatureBatch`，不处理 scaler。P16b 已明确 scaler fit、state loading 和
transform 必须有显式边界。P16c 已新增 `VisualPrecomputedFeatureProvider`，但它读取的是
precomputed/head-ready fixture，不代表 raw/pre-head feature 的 scaler transform。

本步骤只处理已加载 scaler state 的 transform，不做 scaler fit，不读取真实 checkpoint，
不接真实 ViT，不构造 pseudo image，不迁移正式入口，不访问 `/data2`。

## 操作

1. 新增 `time_router/features/visual_scaler.py`：
   - 实现 `LoadedFeatureScaler`；
   - 支持构造参数显式传入 `mean` / `scale` / `feature_columns`；
   - 支持 `LoadedFeatureScaler.from_json(path)` 从显式 JSON state 读取；
   - `transform(feature_batch)` 执行 `(raw - mean) / scale`，输出新的
     head-ready `np.float32 FeatureBatch`；
   - 校验 mean/scale 有限、长度匹配、scale 非 0、输入二维有限、feature_columns 对齐、
     sample_key 非空唯一；
   - 不实现训练期参数估计入口，不读取 checkpoint，不接收 `run_dir`。
2. 更新 `time_router/features/__init__.py`，导出 `LoadedFeatureScaler`。
3. 新增 `tests/fixtures/stage1_visual_scaler_small/`：
   - `raw_visual_features.csv`：覆盖 P13b real-derived manifest 的 4 个 sample_key，行顺序故意
     不同于 manifest；
   - `scaler_state.json`：保存固定 `mean` / `scale` / `feature_columns`；
   - `README.md`：说明 fixture 不包含 `/data2`、checkpoint、ViT、pseudo image、run_dir、
     prediction、oracle。
4. 新增 `tests/smoke/stage1_visual_feature_scaler_smoke.py`：
   - smoke 内局部 raw fixture helper 构造 raw/pre-head `FeatureBatch`，不改变 P16c provider
     的 head-ready 语义；
   - 串联 `P13b ordered sample_keys -> raw FeatureBatch -> LoadedFeatureScaler ->
     LoadedTorchMLPRouterHeadAdapter -> EvaluationInputAdapter`；
   - 覆盖 transform 数值、sample_key 保序、shape 不变、输出 dtype、输入未被修改、schema
     lineage、无 `run_dir`、无 `torch.load`、zero scale、长度不匹配、非有限 state、非有限
     input、missing/duplicate sample_key、adapter/evaluator 消费链路。
5. 新增 `docs/refactor/stage1_visual_feature_scaler.md`。
6. 更新：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `WORKSPACE_STRUCTURE.md`

## 结果

已运行并通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_scaler.py tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
```

P16d smoke 输出确认：

- scaler 输出新的 head-ready `FeatureBatch`；
- `sample_keys` 保持 P13b manifest 顺序；
- features shape 不变；
- 输出 dtype 为 `float32`；
- transform 数值等于 `(raw - mean) / scale`；
- 输入 `FeatureBatch.features` 未被原地修改；
- P16a adapter 可消费 scaled `FeatureBatch`；
- `EvaluationInputAdapter` 可生成 summary/rows，per-sample rows 保持 sample_key 顺序；
- 阶段内未调用 `torch.load`，未创建 canonical run_dir。

## 结论

P16d 的 loaded scaler transform 边界已经独立落地。当前实现没有把 scaler 塞进
RouterHead adapter，也没有改变 P16c provider 的 precomputed/head-ready 语义。正式 Visual
入口后续可在 Runtime/entrypoint 中显式加载 scaler state，再把 transform 后的 head-ready
`FeatureBatch` 交给 P16a adapter。

## 下一步方案

1. 小步提交并 push 到 `origin/refactor/stage1-route-audit`。
2. 后续继续按 P16b 路线拆分 fake encoder / online ViT provider / checkpoint signature 审计，
   不直接迁移正式 Visual 入口。
