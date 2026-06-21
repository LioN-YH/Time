# Stage 1 P16i Visual legacy MLP checkpoint payload smoke

日志日期：2026-06-21 14:55:58 CST

## 目的

新增 Runtime-side tiny checkpoint payload loader smoke，验证“显式 checkpoint path ->
payload -> `router_state_dict` -> 清理 `module.` 前缀 -> strict load 到已构造 legacy
`VisualMLPRouter` -> P16a adapter”的最小边界。

## 背景

P16a 已完成 `LoadedTorchMLPRouterHeadAdapter`，P16g 已完成 legacy
`VisualMLPRouter` checkpoint/signature 审计，P16h 已完成 loaded-module smoke。但此前仍没有
Runtime helper 读取 checkpoint payload，也没有用 tempfile checkpoint 验证
`router_state_dict` 提取、`module.` 前缀清理和 strict load 的组合边界。

本步仍不读取真实 checkpoint，不访问 `/data2`，不处理真实 scaler transform，不启动 ViT，
不迁移正式 Visual Router 入口。

## 操作

1. 新增 `time_router/runtime/visual_mlp_checkpoint.py`，实现：
   - `strip_dataparallel_prefix(state_dict)`
   - `extract_router_state_dict(payload)`
   - `load_checkpoint_payload(path, map_location="cpu")`
   - `load_router_state_dict(model, state_dict, strict=True)`
2. 更新 `time_router/runtime/__init__.py`，同步导出 P16i checkpoint payload helper。
3. 新增 `tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py`：
   - 在 tempfile 内创建 normal key 和 `module.` prefix key 两个 tiny checkpoint；
   - payload 覆盖 `router_state_dict`、`scaler_state`、`config` 和 `metadata`；
   - 使用 legacy `VisualMLPRouter`、P16c `VisualPrecomputedFeatureProvider` 和 P13b
     `expert_predictions.json`；
   - strict load 后交给 P16a `LoadedTorchMLPRouterHeadAdapter`；
   - 通过 `EvaluationInputAdapter` 生成 summary/rows；
   - 覆盖缺少 `router_state_dict`、prefix 冲突、strict missing/unexpected key 负向用例。
4. 新增 `docs/refactor/stage1_visual_legacy_mlp_checkpoint_payload.md`，记录 P16i 边界和验收命令。
5. 同步更新 `docs/refactor/stage1_refactor_roadmap.md`、
   `docs/refactor/stage1_entrypoint_migration_plan.md`、`WORKSPACE_STRUCTURE.md` 和
   `experiment_logs/README.md`。

## 结果

已通过以下验证：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall \
  time_router/runtime/visual_mlp_checkpoint.py \
  tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  tests/smoke/stage1_visual_legacy_mlp_loaded_module_smoke.py

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py

git diff --check
```

新 smoke 输出确认：

- `torch.load` 只读取 tempfile checkpoint；
- P13b ordered sample_keys、P16c head-ready `FeatureBatch` 和 P13b `ExpertBatch` 对齐；
- 缺失 `router_state_dict`、`module.` key 冲突、strict missing/unexpected key 均 fail-fast；
- normal 与 DataParallel `module.` 前缀 tiny checkpoint payload 均可 strict load 到 legacy module；
- loaded legacy module forward 输出二维 logits；
- P16a adapter 输出 `RouterOutput`，sample_key 保序，model_columns 与 `ExpertBatch` 对齐；
- softmax weights finite 且每行和约等于 1；
- `EvaluationInputAdapter` 生成 hard/raw-soft MAE/MSE summary 和 per-sample rows；
- `scaler_state` 只作为 payload metadata 被识别，没有执行 transform。
- P16h loaded-module smoke 和 P16a adapter smoke 均通过，确认本步未破坏既有边界。
- `git diff --check` 通过，当前 diff 未包含正式 Visual Router 训练入口或
  `scripts/run_stage1_visual_small.py` 修改。

## 结论

P16i 已完成 Runtime-side checkpoint payload 读取与 router state_dict strict loading 的最小
smoke。checkpoint loading 仍属于 Runtime/entrypoint implementation detail，不进入 P16a
adapter interface；P16a adapter 仍只接收已加载 module 和 head-ready `FeatureBatch`。

本步没有读取真实 checkpoint，没有访问 `/data2`，没有启动 ViT/transformers，没有调用
`train_visual_router_online_streaming.py`，也没有修改正式 Visual small entrypoint。

## 下一步方案

1. 小步提交并 push 到 `origin/refactor/stage1-route-audit`。
2. 后续可单独做 Visual canonical eval entrypoint thin slice 或真实 checkpoint dry-run；真实
   scaler transform、ViT loading 和正式入口迁移继续留给后续 Runtime 编排步骤。
