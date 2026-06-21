# Stage 1 P16h Visual legacy MLP loaded-module smoke

日志日期：2026-06-21 14:39:51 CST

## 目的

新增 P16h smoke，验证 legacy `VisualMLPRouter` 在“已经实例化并 strict load 完成”的状态下，
可以被 P16a `LoadedTorchMLPRouterHeadAdapter` 消费，并继续生成
`RouterOutput` 和 `EvaluationInputAdapter` summary/rows。

## 背景

P16a 已完成 `LoadedTorchMLPRouterHeadAdapter`，边界是“已加载 torch module + head-ready
`FeatureBatch` -> `RouterOutput`”。P16g 已审计 legacy `VisualMLPRouter` 的 constructor、
forward、checkpoint payload、`router_state_dict`、scaler 与 DataParallel key 边界。当前仍不能读取
真实 checkpoint，也不能把 checkpoint/scaler/run_dir 语义加入 adapter interface。

## 操作

1. 读取目标说明，确认本步只做 loaded-module smoke，不实现 checkpoint loader、不访问 `/data2`、
   不启动 ViT、不迁移正式入口。
2. 检查现有 P16a smoke、P16c `VisualPrecomputedFeatureProvider`、P16f chain smoke 和 P16g 审计文档。
3. 新增 `tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py`：
   - import legacy `VisualMLPRouter` 定义；
   - 使用 P13b ordered sample_keys；
   - 使用 P16c precomputed/head-ready visual fixture 构造 `FeatureBatch`；
   - 使用 P13b `expert_predictions.json` 构造 `ExpertBatch`；
   - 构造 normal 与 `module.` 前缀两种 in-memory fake state_dict；
   - 在 smoke 内清洗 `module.` 前缀并 strict load；
   - 将已加载 legacy module 交给 P16a adapter；
   - 调用 `EvaluationInputAdapter` 验证 hard/raw-soft MAE/MSE 和 rows；
   - patch `torch.load`，确保不读取 checkpoint。
4. 首次运行新 smoke 时，边界扫描因 `time_router/models/visual_mlp_adapter.py` 注释中含
   “不接收 run_dir” 而失败；该注释本身是禁止语义说明，不是接口泄漏。随后将扫描 token
   收窄到 `/data2`、`torch.load`、`checkpoint_path`、`scaler_path`、`ViTModel` 和
   `AutoImageProcessor`。
5. 新增 `docs/refactor/stage1_visual_legacy_mlp_loaded_module_smoke.md`，记录 P16h 边界、
   与 P16a/P16g 关系、覆盖项、明确不做范围和验证命令。

## 结果

已执行并通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_feature_chain_protocol_smoke.py
```

新 smoke 输出确认：

- 已 import legacy `VisualMLPRouter` 定义，未调用 `torch.load`。
- P13b ordered sample_keys、P16c head-ready `FeatureBatch` 和 P13b `ExpertBatch` 已对齐。
- normal state_dict 和 DataParallel `module.` 前缀 state_dict 均可清洗后 strict load。
- legacy loaded module 已被 P16a adapter 消费，`EvaluationInputAdapter` summary/rows 正常生成。
- 本次 smoke 的 `hard_mae=0.224999964`，`raw_soft_mae=0.083549440`。
- P16a adapter 回归 smoke 通过，`hard_mae=0.137499988`，`raw_soft_mae=0.079910710`。
- P16f feature chain protocol 回归 smoke 通过，`hard_mae=0.125000000`，
  `raw_soft_mae=0.078714140`。
- `git diff --check` 通过；`git diff` 确认未修改 `scripts/run_stage1_visual_small.py`、
  `train_visual_router_online_streaming.py` 或 `time_router/models/visual_mlp_adapter.py`。

## 结论

P16h 已证明 legacy `VisualMLPRouter` 的“已加载 module”形态能够被 P16a adapter 消费。
该结论只覆盖 in-memory fake state_dict 和 head-ready fixture，不覆盖真实 checkpoint discovery、
`torch.load`、`map_location`、真实 scaler loading、ViT provider 或正式入口迁移。

## 下一步方案

提交并推送 `origin/refactor/stage1-route-audit`。后续继续按 Runtime / entrypoint 小步推进真实
checkpoint loader 或 scaler/runtime 对接；后续步骤应单独覆盖 checkpoint payload 中的
`router_state_dict` 解析、`map_location`、strict load 错误报告、真实 scaler state loading，
以及正式 Visual entrypoint 的 device/resource policy。
