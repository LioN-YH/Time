# Stage 1 P5d provider adapter boundary review

日志日期：2026-06-19 22:12:27 CST

## 目的

基于 P5b provider interface design 和 P5c protocol types skeleton，审查现有 Stage 1 代码中哪些模块/函数未来可以适配为 canonical provider/head/evaluator adapter，并明确哪些历史路线不应适配。

## 背景

P5c 已新增轻量 protocol dataclass，但还没有任何 provider adapter 或正式入口迁移。为了避免后续实现时把训练脚本、runtime、输出目录、loss、reader 和 evaluator 混在一起，需要先文档化审查现有代码边界，决定第一批最小 adapter 的实现顺序。

## 操作

1. 只读审查 `time_router/io/prediction_cache_reader.py` 和 `visual_router_experiments/common/prediction_array_io.py`，判断 ExpertProvider adapter 候选和 `packed_npy_v1` / `per_sample_npy` 读取边界。
2. 只读审查 `pseudo_imageization.py`、`vit_embedding_utils.py` 和 `train_visual_router_online_streaming.py`，判断 Visual pseudo image / ViT feature provider 候选和 future finetune/joint ViT 接口点。
3. 只读审查 `stage1_timefuse_fusor_streaming_reader.py` 和 `train_timefuse_fusor_streaming.py`，判断 TimeFuse 17 维 feature cache reader、future online TimeFuse feature computation 和 linear-softmax head 的边界。
4. 只读审查 `time_router/evaluation` public API，判断 Evaluator adapter 的可复用能力和 legacy CSV schema 边界。
5. 新增 `docs/refactor/provider_adapter_boundary.md`，记录 ExpertProvider、FeatureProvider、RouterHead、Evaluator、Runtime 边界和第一批实现建议。
6. 更新 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md`、`WORKSPACE_STRUCTURE.md` 和 `experiment_logs/README.md`。

## 结果

- 确认第一批最小 adapter 不应从 Visual online ViT provider 开始，而应先做 entrypoint migration plan，再实现基于 `PredictionBatchReader` 的 `PredictionCacheExpertProvider`。
- 确认 `prediction_array_io.load_prediction_arrays_grouped(...)` 是 ExpertProvider 可间接复用的底层性能边界，但不应作为 provider 本体。
- 确认 `packed_npy_v1` 是 full-scale 推荐读取口径，`per_sample_npy` 只保留 legacy/smoke 兼容。
- 确认 TimeFuse 17 维 feature cache reader 适合作为第二批 `TimeFuseFeatureCacheProvider` 候选，但必须拆出 feature-only provider，不能混入 oracle/prediction 读取或训练 runtime。
- 确认 offline ViT embedding cache 只作为 reference-only / debug-only，不作为 full-scale canonical adapter。
- 确认 Visual Router MLP 与 TimeFuse Linear-softmax 可以成为 RouterHead adapter；loss、optimizer、epoch loop、scaler fit、checkpoint/resume、DataParallel、prediction/oracle 读取和 CSV 写出不属于 RouterHead。
- 确认 `time_router.evaluation` public API 是 Evaluator adapter 基础能力，legacy CSV schema 不应反向污染 Evaluator。
- 本次未新增 provider 代码，未修改 protocol types、reader、evaluation、io helper 或训练脚本，未接入 `/data2`。
- 验收命令均已通过：
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py`
  - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke`

## 结论

P5d 完成了 provider adapter 边界审查。后续实现应先用文档化 migration plan 定清 runtime 编排，再以 `PredictionCacheExpertProvider` 作为最小共享主干 adapter；TimeFuse feature cache provider 和 RouterHead adapter 应在读取边界稳定后再分步实现。

## 下一步方案

1. 小步提交并推送 `refactor/stage1-route-audit`。
2. 后续可进入 P5e entrypoint migration plan only，先设计 Visual / TimeFuse 两个正式入口如何逐步消费 P5c protocol objects 和 adapter specs。
