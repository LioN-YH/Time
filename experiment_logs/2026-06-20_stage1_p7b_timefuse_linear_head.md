# Stage 1 P7b TimeFuseLinearSoftmaxHead

日志日期：2026-06-20 01:02:48 CST

## 目的

新增最小 `TimeFuseLinearSoftmaxHead`，只把 `FeatureBatch.features` 转成 `RouterOutput(logits, weights)`，供 Stage 1 smoke 使用。

## 背景

P7a 已实现 `TimeFuseFeatureCacheProvider`，能够把显式 feature CSV 包装为 `FeatureBatch`。本步继续补最小 RouterHead adapter，但边界仍限定在纯内存 smoke：不接正式 TimeFuse fusor、不接 Visual Router 入口、不训练、不写运行产物。

## 操作

1. 新增 `time_router/models/__init__.py`，导出 `TimeFuseLinearSoftmaxHead`。
2. 新增 `time_router/models/timefuse_linear.py`：
   - 使用纯 numpy 固定线性权重计算 `logits = features @ weight + bias`。
   - 沿专家维度做 stable softmax 得到 `weights`。
   - `predict(feature_batch, model_columns)` 返回 `RouterOutput`。
   - 校验 `model_columns` 非空、不重复且长度等于专家输出维度。
   - 校验 `FeatureBatch.features` 为二维矩阵，样本维度与 `sample_keys` 一致，特征维度与 weight 输入维度一致。
   - `RouterOutput.sample_keys` 保持 `FeatureBatch.sample_keys` 顺序，`RouterOutput.model_columns` 保持调用方传入顺序。
3. 新增 `tests/smoke/stage1_timefuse_linear_head_smoke.py`：
   - 使用固定 `FeatureBatch`、固定权重矩阵和 bias。
   - 独立复算 expected logits 和 stable softmax。
   - 校验 deterministic logits/weights、weights 逐样本和为 1、`__call__` 与 `predict` 输出一致。
   - 阻断 `open`、`Path.open`、`np.load`、`np.save` 和 `np.savez`，并检查 `experiment_logs/run_outputs` 一层目录集合不变。
4. 新增 `docs/refactor/timefuse_linear_head.md`，记录 P7b API、输入输出 contract、非目标和验收命令。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增文档、`time_router/models` 包和 smoke 文件。

## 结果

以下命令均已在 `quito` conda 环境下通过：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_linear_head_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

关键验证结果：

- P7a provider smoke 通过，证明已有 `FeatureBatch` provider contract 未被破坏。
- P7b head smoke 通过，证明 `sample_keys` 保序、`model_columns` 对齐、logits/weights deterministic、softmax 权重逐样本和为 1。
- P6b evaluation adapter smoke 通过，证明下游 `RouterOutput.weights` 消费路径未受影响。
- `compileall` 通过，新增 `time_router/models` 包和 smoke 脚本无语法错误。
- smoke 阶段未读取 prediction cache、oracle/TSF 或 feature CSV，未访问 `/data2`，未创建 run_dir，未写 status/metadata/CSV/JSON/Parquet。

## 结论

Stage 1 P7b minimal `TimeFuseLinearSoftmaxHead` 已完成。该实现只作为 smoke-only adapter，把 `FeatureBatch.features` 转为 `RouterOutput(logits, weights)`，没有接入正式 TimeFuse fusor、Visual Router 入口或训练流程。

## 下一步方案

1. 小步提交并推送 `refactor/stage1-route-audit` 分支。
2. 后续如需继续推进，应另起小步设计正式 RouterHead interface 或 Visual online feature provider；正式入口迁移仍保持 smoke 先行。
