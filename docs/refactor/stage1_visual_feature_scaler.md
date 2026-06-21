# Stage 1 P16d Visual Feature Scaler Boundary

## 1. 目标

P16d 新增最小 loaded Visual FeatureScaler 边界：

```text
loaded scaler state + raw/pre-head FeatureBatch
-> LoadedFeatureScaler
-> head-ready float32 FeatureBatch
```

本步骤只冻结 transform 边界，不实现 scaler fit，不读取真实 checkpoint，不接真实
ViT，不构造 pseudo image，不迁移正式 Visual Router 训练入口，也不访问 `/data2`。

## 2. 边界定义

新增 `time_router/features/visual_scaler.py`，其中 `LoadedFeatureScaler` 只接受已加载
scaler state：

- `mean`
- `scale`
- 可选 `feature_columns`
- 可选 `scaler_schema_version`

也提供 `LoadedFeatureScaler.from_json(path)` 读取显式 JSON state fixture。该路径只是
Runtime 显式注入 implementation detail，不是长期 provider interface，也不做 run_dir
discovery。

transform 规则固定为：

```text
scaled = (raw - mean) / scale
```

输出为新的 `FeatureBatch`：

- `sample_keys` 保持输入顺序；
- `features` shape 与输入一致；
- `features` dtype 固定为 `np.float32`；
- `feature_schema` 记录 `transformed_by=LoadedFeatureScaler`、`handles_scaler=True`、
  `head_ready=True`、`feature_columns` 和 `input_schema`；
- `extra` 只记录轻量 scaler lineage，不保存 `run_dir`。

## 3. 校验约束

`LoadedFeatureScaler` 会 fail-fast 校验：

- `mean` / `scale` 必须是一维有限数值；
- `mean` / `scale` 长度必须等于 feature_dim；
- `scale` 不能为 0；
- 输入 `FeatureBatch.features` 必须是二维有限数值；
- 输入样本维必须等于 `sample_keys` 数量；
- 输入 feature_dim 必须等于 scaler feature_dim；
- `feature_columns` 若存在，必须与 scaler state 对齐；
- 不能有重复 `sample_key`。

本类不根据 batch 数据自动计算 mean/std，不执行训练期参数估计。

## 4. 与 P16a/P16b/P16c 的关系

P16a `LoadedTorchMLPRouterHeadAdapter` 只消费 head-ready `float32 FeatureBatch`，不处理
scaler。P16d 的输出正是 P16a 可以消费的 head-ready `FeatureBatch`。

P16b 已冻结 real Visual feature provider 的 scaler fit/state loading/transform 边界：
fit 属于训练期或 Runtime 准备阶段，state loading 属于 Runtime/entrypoint，transform
可以作为显式 pre-head step。P16d 只实现 transform。

P16c `VisualPrecomputedFeatureProvider` 读取 precomputed/head-ready fixture，不处理 raw
feature。P16d 则处理 raw/pre-head `FeatureBatch -> head-ready FeatureBatch`，不改变 P16c
provider 的语义。

## 5. 明确不负责

P16d 不负责：

- scaler fit / incremental fit；
- scaler state discovery；
- checkpoint loading；
- ViT embedding；
- pseudo image；
- prediction backend；
- oracle/expert error；
- `run_dir`；
- training loop；
- Bash launcher；
- 正式 Visual entrypoint migration。

后续正式入口可以在 Runtime/entrypoint 中显式加载 scaler state，再用
`LoadedFeatureScaler.transform(...)` 将 raw/pre-head `FeatureBatch` 转为 head-ready
`FeatureBatch`，交给 P16a adapter。

## 6. Smoke

新增：

- `tests/fixtures/stage1_visual_scaler_small/raw_visual_features.csv`
- `tests/fixtures/stage1_visual_scaler_small/scaler_state.json`
- `tests/smoke/stage1_visual_feature_scaler_smoke.py`

smoke 串联：

```text
P13b ordered sample_keys
-> smoke-local raw visual feature helper / raw FeatureBatch
-> LoadedFeatureScaler / head-ready FeatureBatch
-> LoadedTorchMLPRouterHeadAdapter / RouterOutput
-> EvaluationInputAdapter / summary + rows
```

覆盖项包括 transform 数值、sample_key 保序、shape 不变、输出 `float32`、输入
`FeatureBatch` 未被修改、schema lineage、无 `run_dir`、无 `torch.load`、无训练期
参数估计入口、zero scale、长度不匹配、非有限 scaler state、非有限输入 features、
missing/duplicate sample_key、P16a adapter 消费和 Evaluation rows 保序。

验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/features/visual_scaler.py tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_scaler_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_precomputed_feature_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
```
