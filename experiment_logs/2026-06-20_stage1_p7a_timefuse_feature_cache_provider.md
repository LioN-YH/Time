# Stage 1 P7a TimeFuseFeatureCacheProvider

日志日期：2026-06-20 00:44:32 CST

## 目的

新增最小 `TimeFuseFeatureCacheProvider`，只把显式传入的小规模 TimeFuse feature CSV 包装为 P5c `FeatureBatch`，用于 smoke 验证 FeatureProvider contract。本步不接正式 TimeFuse fusor、Visual Router、runtime、launcher 或 config system。

## 背景

Stage 1 已完成 P6a `PredictionCacheExpertProvider`、P6b/P6c `EvaluationInputAdapter` 与兼容 `FusionEvaluator`，专家预测和评估输入边界已基本稳定。根据 P5d/P5e/P5f 迁移设计，下一步可做 TimeFuse feature-only adapter，但必须保持 smoke-only 范围：只读 feature，不读取 prediction cache、oracle/TSF、`y_true` 或 expert error，不做 scaler fit，不创建 run_dir，不写运行产物。

## 操作

1. 新增 `time_router/features/__init__.py`，导出 `TimeFuseFeatureCacheProvider`。
2. 新增 `time_router/features/timefuse_cache.py`，使用标准库 `csv.DictReader` 读取调用方显式传入的 feature CSV，并通过 `load_batch(sample_keys)` 返回 `FeatureBatch`。
3. 新增 `tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py`，使用测试内临时 feature CSV 覆盖保序、shape、schema、extra、非法 sample_key 和只读边界。
4. 新增 `docs/refactor/timefuse_feature_cache_provider.md`，记录 API、边界、metadata、smoke 覆盖和后续接入顺序。
5. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md` 和 `WORKSPACE_STRUCTURE.md`，登记 P7a 当前状态、新增文件和边界。

## 结果

已运行并通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_feature_cache_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_prediction_cache_expert_provider_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_evaluation_input_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

新增 smoke 输出确认：

- `FeatureBatch` 类型正确；
- `sample_keys` 保持显式请求顺序；
- `features` shape 为 `(2, 17)`，dtype 为 `float32`；
- 17 维特征数值按 CSV 和请求顺序对齐；
- `feature_schema` 和 `extra` 只记录 feature lineage 与 provider metadata；
- provider 拒绝空 sample_keys 和重复 sample_key；
- provider 不读取 prediction/oracle，不创建输出目录，不写运行产物。

## 结论

P7a smoke-only `TimeFuseFeatureCacheProvider` 已完成。该实现只负责 `feature CSV -> FeatureBatch` 的最小适配，不承担 scaler、prediction cache、oracle/TSF、evaluation、run_dir、status/metadata 或正式入口职责。

## 下一步方案

1. 做小步提交并推送到 `refactor/stage1-route-audit`。
2. 后续另起小步实现 TimeFuse linear-softmax head 的纯 protocol smoke，再考虑 config skeleton 和正式入口下沉。
