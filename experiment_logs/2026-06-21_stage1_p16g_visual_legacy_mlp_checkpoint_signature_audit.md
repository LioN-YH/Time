# Stage 1 P16g Visual legacy MLP checkpoint/signature 审计

日志日期：2026-06-21 14:18:07 CST

## 目的

审计 legacy `VisualMLPRouter` 的 constructor、forward、checkpoint/state_dict 格式、
DataParallel key、输入 feature_dim、输出专家 logits 维度和 device/runtime loading 边界，
为后续 Runtime 加载真实 legacy MLP 后交给 P16a `LoadedTorchMLPRouterHeadAdapter` 做准备。

## 背景

P16a-P16f 已完成 Visual RouterHead adapter、Visual feature provider/transform/architecture
variant 和 feature chain protocol 边界。本步骤只做文档审计，不新增 checkpoint loader，
不读取真实 checkpoint，不访问 `/data2`，不修改正式 Visual Router 入口。

## 操作

1. 确认当前分支为 `refactor/stage1-route-audit`，且开始时工作区无未提交改动。
2. 只读检索 legacy 相关代码，重点查看：
   - `visual_router_experiments/stage1_vali_test_router/train_visual_router.py`
   - `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`
3. 检查关键词包括 `VisualMLPRouter`、`state_dict`、`torch.load`、`DataParallel`、
   `resume-checkpoint`、`resume_checkpoint`、`scaler`、`input_dim`、`hidden_dim`、
   `num_experts` 和 `model_columns`。
4. 新增审计文档：
   - `docs/refactor/stage1_visual_legacy_mlp_checkpoint_signature_audit.md`
5. 同步更新：
   - `docs/refactor/stage1_refactor_roadmap.md`
   - `docs/refactor/stage1_entrypoint_migration_plan.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`

## 结果

审计确认：

- `VisualMLPRouter` 定义在 `train_visual_router.py`，constructor 为
  `input_dim, hidden_dim, output_dim, dropout`。
- `forward(features)` 直接返回二维 logits tensor，推理侧在外部执行 `torch.softmax(logits, dim=1)`。
- streaming 正式入口实例化时使用 `input_dim=int(scaler.n_features_in_)`、
  `hidden_dim=args.hidden_dim`、`output_dim=len(MODEL_COLUMNS)` 和 `dropout=args.dropout`。
- checkpoint 不是裸 state_dict，也不是 `model_state_dict`，而是包含
  `router_state_dict`、`optimizer_state_dict`、`scaler_state`、`completed_epochs`、
  `epoch_summaries` 和严格 resume signature 的 payload。
- 当前 streaming 入口的 DataParallel 只包裹冻结 ViT encoder，router 本身不包裹
  DataParallel，因此源码层面预期 `router_state_dict` 不带 `module.` 前缀；但真实历史
  checkpoint 未读取，后续仍需单独 smoke 覆盖 `module.` 兼容。
- checkpoint loading、constructor 参数解析、map_location、strict loading、
  missing/unexpected keys、DataParallel key 处理、device policy 和 optimizer/scaler resume
  属于 Runtime / entrypoint，不属于 P16a adapter。
- scaler state loading/transform 属于 Runtime 或 P16d `LoadedFeatureScaler` 侧，不进入
  RouterHead adapter。

## 结论

P16a `LoadedTorchMLPRouterHeadAdapter` 应继续只接收已加载 torch module 和 head-ready
`FeatureBatch`。legacy checkpoint payload、scaler state、DataParallel key、device 和 strict
loading 都应在后续 Runtime/checkpoint loader smoke 中单独处理，不能把 checkpoint path、
scaler path 或 run_dir 设计进 adapter interface。

## 下一步方案

后续 P16h 可做最小 loaded-module smoke：使用 in-memory fake `state_dict` 或 tiny checkpoint
fixture 实例化 legacy `VisualMLPRouter`，加载 `router_state_dict`，构造 head-ready
`FeatureBatch`，交给 P16a adapter，并验证 logits/weights shape、model_columns 保序、
strict loading、missing/unexpected keys 和可选 `module.` 前缀兼容。P16h 仍不读取 `/data2`
真实 checkpoint，不接真实 ViT，不迁移正式入口。
