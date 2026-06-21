# Stage 1 P16a Visual MLP RouterHead Adapter

## 1. 目标

P16a 新增正式 Visual RouterHead adapter 的最小边界：

```text
已加载 torch.nn.Module + head-ready FeatureBatch -> RouterOutput
```

这里的 `FeatureBatch.features` 必须已经是 head-ready `float32` embedding/features。P16a 不读取真实 checkpoint，不处理 scaler，不启动 ViT embedding，不迁移正式 Visual Router 训练入口，也不访问 `/data2`。

## 2. Adapter 边界

新增模块：

```text
time_router/models/visual_mlp_adapter.py
```

公开类：

```text
LoadedTorchMLPRouterHeadAdapter
```

输入：

- `model: torch.nn.Module`：Runtime 已经实例化并加载好的 torch module。
- `device: torch.device | str = "cpu"`：Runtime 显式选择的前向设备。
- `feature_batch: FeatureBatch`：调用方已经完成 scaler / ViT / pre-head transform 的二维 `float32` features。
- `model_columns: Sequence[str]`：专家动作空间顺序，必须非空且无重复。

输出：

- `RouterOutput.sample_keys` 保持 `FeatureBatch.sample_keys` 顺序。
- `RouterOutput.model_columns` 与输入 `model_columns` 完全一致。
- `RouterOutput.logits` 为 `[num_samples, num_models]` 的 `numpy.float32` 二维数组。
- `RouterOutput.weights` 为 logits 沿专家维 softmax 后的 `numpy.float32` 二维数组。
- `RouterOutput.extra` 只记录轻量边界 metadata：`adapter_name`、`loaded_model_boundary`、`loads_checkpoint=False`、`handles_scaler=False`、`handles_vit=False` 等。

## 3. 明确不负责

`LoadedTorchMLPRouterHeadAdapter` 不负责：

- checkpoint loading；
- scaler fit / transform；
- ViT embedding；
- pseudo image construction；
- prediction backend；
- `run_dir`；
- training loop；
- Bash launcher；
- device selection policy / DataParallel / resume policy。

这些内容后续应留在 Runtime 或 branch-specific entrypoint 中处理，再把已加载 module 和 head-ready `FeatureBatch` 交给 adapter。

## 4. 与 P15c 的关系

P15c `scripts/run_stage1_visual_small.py` 内的 `SmokeOnlyVisualMLPAdapter` 仍保持 script-local small entrypoint rehearsal。P16a 是正式 adapter 边界实现，但本步不把它接入 P15c small entrypoint，也不声称 Visual-specific small entrypoint 已迁移到真实 Visual RouterHead。

这种分层避免把 small canonical artifact rehearsal 与正式 RouterHead adapter 迁移混在同一步里。

## 5. 未来迁移关系

后续正式 Visual Router 迁移可以在 Runtime 中完成：

1. 读取 checkpoint；
2. 处理 scaler；
3. 构造 pseudo image；
4. 启动 frozen ViT 并得到 embedding；
5. 构造 head-ready `FeatureBatch`；
6. 把已加载 torch module 交给 `LoadedTorchMLPRouterHeadAdapter`；
7. 通过 `EvaluationInputAdapter` 或正式 evaluator 复算/写出评估产物。

legacy `VisualMLPRouter` 的 import、signature、checkpoint state_dict key 适配和 scaler artifact 读取仍留到后续单独步骤。若 legacy model 返回 tuple、dict 或带额外中间结果，兼容也应另起扩展；P16a 只支持 `model(features)` 直接返回二维 logits Tensor。

## 6. Smoke

新增 smoke：

```text
tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
```

链路：

```text
SampleManifest
-> VisualMockFeatureProvider / FeatureBatch
-> small ExpertBatch fixture
-> LoadedTorchMLPRouterHeadAdapter / RouterOutput
-> EvaluationInputAdapter
-> summary / rows
```

smoke 验证：

- adapter 输出 `RouterOutput`；
- sample_key 保序；
- `model_columns` 对齐 `ExpertBatch`；
- logits / weights shape 正确；
- weights finite 且 softmax row sum 约等于 1；
- `EvaluationInputAdapter` 可消费 adapter 输出；
- summary 包含 hard/raw-soft MAE/MSE 与 selected counts；
- per-sample rows 保持 sample_key 顺序；
- adapter 不接收 `run_dir`；
- adapter 不读取 checkpoint，不调用 `torch.load`；
- adapter 不访问 `/data2`；
- adapter 源码不引入 `VisualMLPRouter`、`ViTModel`、`AutoImageProcessor` 或 `train_visual_router_online_streaming`；
- P15c visual small entrypoint 仍保留 script-local `SmokeOnlyVisualMLPAdapter`，未被 P16a 强行替换。

负向用例覆盖：

- `FeatureBatch.features` 非 `float32` 抛错；
- logits shape 与 `model_columns` 不一致抛错；
- duplicate `model_columns` 抛错。

验收命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router/models/visual_mlp_adapter.py tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_mlp_routerhead_adapter_smoke.py
```
