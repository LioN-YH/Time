# Stage 1 P16i Visual Legacy MLP Checkpoint Payload Smoke

创建日期：2026-06-21

## 1. 目标

P16i 只验证 Runtime-side tiny checkpoint payload 读取和 state_dict loading 边界：

```text
显式 checkpoint path
  -> torch.load payload
  -> router_state_dict
  -> 清理 DataParallel module. 前缀
  -> strict load 到已构造 legacy VisualMLPRouter
  -> P16a LoadedTorchMLPRouterHeadAdapter
  -> EvaluationInputAdapter
```

本步不读取真实 checkpoint，不访问 `/data2`，不处理真实 scaler transform，不启动 ViT，
不调用或迁移正式 Visual Router 训练入口。

## 2. 新增 Runtime Helper

新增 `time_router/runtime/visual_mlp_checkpoint.py`，提供四个极小 helper：

- `strip_dataparallel_prefix(state_dict)`：清理 `module.` 前缀；清理后 key 冲突立即报错。
- `extract_router_state_dict(payload)`：只从 mapping payload 中提取 `router_state_dict`。
- `load_checkpoint_payload(path, map_location="cpu")`：只读取调用方显式传入的 checkpoint path。
- `load_router_state_dict(model, state_dict, strict=True)`：将 state_dict strict load 到已构造 `torch.nn.Module`。

边界：

- loader 只接受显式 checkpoint path，不做 run_dir discovery。
- loader 不知道 `/data2`。
- loader 不构造 `FeatureBatch`。
- loader 不调用 ViT 或 transformers。
- loader 不处理 scaler transform。
- loader 不修改 P16a adapter，也不把 checkpoint path 放入 adapter interface。
- `scaler_state` 可以作为 payload metadata 被调用方读取，但本步不执行 transform。

## 3. Smoke 覆盖

新增 `tests/smoke/stage1_visual_legacy_mlp_checkpoint_payload_smoke.py`：

- 在 tempfile 内创建 tiny checkpoint，不使用真实 checkpoint。
- payload 覆盖 `router_state_dict`、`scaler_state`、`config` 和 `metadata`。
- 覆盖 normal key 与 DataParallel 风格 `module.` 前缀 key 两种 state_dict。
- import legacy `VisualMLPRouter` 定义，只构造 module，不调用正式训练入口。
- 使用 P16c `VisualPrecomputedFeatureProvider` 构造 head-ready `FeatureBatch`。
- 使用 P13b `expert_predictions.json` 构造 `ExpertBatch`。
- 将 strict loaded legacy module 交给 P16a `LoadedTorchMLPRouterHeadAdapter`。
- 验证 `EvaluationInputAdapter` 可生成 hard/raw-soft MAE/MSE summary 和 rows。

smoke 明确检查：

- `torch.load` 只读取 tempfile checkpoint。
- 不读取 `/data2`。
- 不调用 `train_visual_router_online_streaming.py`。
- 不启动 ViT / transformers。
- 不修改 `scripts/run_stage1_visual_small.py`。
- loaded model forward 输出二维 logits。
- `RouterOutput.sample_keys` 保序。
- `model_columns` 与 `ExpertBatch` 对齐。
- weights finite 且 softmax row sum 约等于 1。
- summary 包含 `hard_mae`、`hard_mse`、`raw_soft_mae` 和 `raw_soft_mse`。
- `scaler_state` 被识别为 payload metadata，但不执行 transform。

负向用例覆盖：

- 缺少 `router_state_dict` 报错。
- `module.` prefix 清理后 key 冲突 fail-fast。
- strict load missing key 报错。
- strict load unexpected key 报错。

## 4. 与 P16a/P16d/后续入口的关系

P16a `LoadedTorchMLPRouterHeadAdapter` 仍只接收已加载 module 和 head-ready
`FeatureBatch`。checkpoint loading 不属于 adapter，也不应把 checkpoint path 放进 adapter
interface。

P16d 或后续 Runtime 编排仍负责真实 scaler state 的 transform 边界。P16i 只确认
`scaler_state` 可以在 checkpoint payload 中保留为 metadata，本步不执行 `(x - mean) / scale`
或任何真实 scaler 逻辑。

后续可继续拆：

1. Visual canonical eval entrypoint thin slice：显式组合 checkpoint helper、feature provider、
   adapter 和 evaluator，但仍先用 small fixture。
2. 真实 checkpoint dry-run：只读真实 checkpoint metadata/state_dict signature，仍需单独确认
   `/data2`、device、map_location、hidden_dim/config 和 scaler state 口径。
3. 正式 Visual entrypoint migration：由 Runtime/entrypoint 负责 checkpoint、scaler、ViT、
   feature chain、evaluation 和 artifact writer 编排。

## 5. 验收命令

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
```

