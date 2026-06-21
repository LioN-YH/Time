# Stage 1 P16l Visual Real Checkpoint Dry-Run Plan

## 1. 目的

P16l 冻结 real Visual checkpoint dry-run 的前置方案和 guarded explicit-path policy。本文只定义从 tiny checkpoint payload 过渡到用户显式 real checkpoint dry-run 的边界，不实现真实 checkpoint 读取，不访问 `/data2`，不启动 ViT，不迁移正式 Visual full-scale 入口。

目标是先把下一步手动 dry-run 的风险面压清楚：

- 真实 checkpoint 必须由用户显式传入；
- 读取非 fixture checkpoint 必须有用户显式授权；
- 默认 smoke / CI 仍只允许读取 tests fixture 或 tempfile tiny payload；
- checkpoint loading 属于 Runtime / entrypoint 操作层，不属于 P16a adapter；
- scaler_state 可以记录 metadata，但不能 silent transform；
- dry-run 失败时必须 fail-fast 并给出清楚错误；
- artifact metadata 必须标记这是 real checkpoint dry-run，而不是正式 Visual Router 迁移。

## 2. 前置条件

real checkpoint dry-run 后续启动前必须同时满足：

- 用户在 CLI 或等价入口中显式传入 checkpoint path；
- 用户显式确认允许读取非 fixture checkpoint；
- 默认 smoke / CI 不读取真实 checkpoint；
- 不从 `/data2` 自动搜索 checkpoint；
- 不从 `run_dir`、`latest_checkpoint_index.json`、历史 launcher 输出或 metadata 自动推断 checkpoint；
- 不启动训练；
- 不启动 ViT、`AutoImageProcessor` 或 transformers encoder；
- 不处理 full-scale data；
- 输出仍写 canonical `run_dir`，由 Runtime artifact writer 管理 metadata、status、evaluation 和 prediction rows。

如果任一条件不满足，entrypoint 应在读取文件前报错退出。报错信息应直接说明缺少哪个显式授权或路径条件，而不是退回 mock path 或自动发现其它路径。

## 3. 路径策略

P16l 将 checkpoint path 分为三类：

| 路径类别 | 默认策略 | 用途 |
| --- | --- | --- |
| `tests/fixtures` 或 tempfile | 允许 | smoke / CI / 本地 tiny payload 验证，可作为默认 loaded legacy path fixture |
| 用户显式传入路径 | 手动 dry-run 可允许 | 必须同时传入 allow flag，并在 artifact metadata 中记录摘要 |
| `/data2` 或外部大盘路径 | 默认禁止 | 只有后续单独 pressure / full-scale goal 明确允许时才可读取 |

checkpoint path 不进入 P16a `LoadedTorchMLPRouterHeadAdapter`。P16a adapter 只接收 Runtime 已经构造并加载好的 `torch.nn.Module` 和 head-ready `FeatureBatch`。

checkpoint path 也不设计成 `FeatureProvider`、provider/head interface 或 P16a adapter 的字段。路径、授权、敏感路径记录策略、`torch.load` map_location、strict loading 和错误处理都属于 Runtime / entrypoint 操作层。

## 4. Checkpoint Payload 策略

后续 real dry-run 复用 P16i 已验证的 payload 口径：

- loader 只处理显式 path；
- payload 只提取 `router_state_dict`、`scaler_state`、`config` 和 `metadata`；
- `module.` prefix 清理由 Runtime helper 负责；
- strict loading policy 必须由 CLI 或配置显式指定；
- missing keys / unexpected keys 在 strict 模式下 fail-fast；
- `scaler_state` 可记录为 metadata，例如 `scaler_state_present=true`；
- scaler transform 仍只能通过 P16d `LoadedFeatureScaler` 或后续明确路径执行，不能因为 payload 中存在 `scaler_state` 就 silent transform。

`optimizer_state_dict`、epoch resume 信息、scheduler state 和 training-only metadata 不属于 eval-only dry-run 必需输入；如果真实 payload 中存在这些字段，dry-run 可以忽略并在 metadata 中记录未使用。

## 5. Dry-Run CLI 方案

P16l 不实现 CLI。后续可以扩展 `scripts/run_stage1_visual_small.py`，也可以新增专门 dry-run 脚本。候选参数为：

```text
--router-checkpoint-payload <path>
--allow-real-checkpoint
--checkpoint-kind legacy_visual_mlp
--strict-checkpoint-load
--record-checkpoint-metadata-only
--no-vit
```

实际命名应以后续实现时的现有 CLI 风格为准。无论采用哪个入口，都必须保持以下行为：

- tiny fixture path 才能在 smoke 中默认使用；
- 真实 checkpoint path 只允许手动 dry-run；
- 常规 smoke / CI 不包含真实 checkpoint path；
- 未传 `--allow-real-checkpoint` 时，非 fixture checkpoint 在读取前失败；
- `--no-vit` 或等价默认行为必须保证 dry-run 不启动真实 ViT。

## 6. Artifact Metadata

未来 real checkpoint dry-run 应在 `run_metadata.json` 中记录以下字段或等价字段：

- `checkpoint_payload_source`
- `checkpoint_is_fixture`
- `checkpoint_path_record_policy`，例如 `basename/hash_only`
- `loaded_legacy_mlp`
- `strict_checkpoint_load`
- `scaler_state_present`
- `scaler_transform_applied`
- `loads_real_checkpoint`
- `loads_real_vit`
- `formal_visual_router_migration`

真实 checkpoint 的完整绝对路径默认不写入长期 artifact。建议记录 basename、文件 hash 或用户允许的路径摘要，避免把敏感大盘路径或个人目录写入可长期保留文档。

当 dry-run 读取真实 checkpoint 但仍使用 precomputed feature / fixture data / no ViT 时，metadata 应明确：

```text
loads_real_checkpoint=true
loads_real_vit=false
formal_visual_router_migration=false
```

这表示只验证 Runtime 侧 checkpoint loading 到 legacy Visual MLP head 的手动 dry-run，不代表正式 Visual Router entrypoint 已迁移。

## 7. 失败策略

dry-run 失败时应清楚区分路径策略错误和 payload 内容错误：

- 未显式传入 checkpoint path：报错要求传入 path；
- 非 fixture path 但未授权：报错要求显式 allow flag；
- `/data2` 或外部大盘路径未被本 goal 授权：报错说明当前阶段禁止读取；
- payload 缺少 `router_state_dict`：报错说明 payload schema 不满足 P16i 口径；
- strict load missing / unexpected keys：报错列出关键差异；
- `scaler_state` 存在但未配置 scaler transform：不 silent transform，只记录 metadata。

失败不应自动回退到 mock adapter，也不应改写为 `loads_real_checkpoint=false` 的成功 run。

## 8. 与后续步骤关系

建议后续拆分：

- P16m：visual explicit real-checkpoint dry-run CLI guard smoke，只验证路径 guard，不读 `/data2`；
- P16n：manual real checkpoint dry-run，用户显式提供 artifact path 后再执行；
- P17a：Visual canonical eval entrypoint migration plan；
- P17b：real Visual feature chain dry-run / fake encoder to online ViT boundary；
- P17c：Visual full-scale / pressure plan。

P16l 本身不读取真实 checkpoint、不新增 path guard helper、不新增 smoke、不运行训练、不启动 ViT、不访问 `/data2`。如果后续发现需要代码 guard，应另起 P16m，以极小 Runtime helper 和 smoke 覆盖读取前 fail-fast。

## 9. 明确不做

- 不读取真实 checkpoint；
- 不访问 `/data2`；
- 不启动 ViT、`AutoImageProcessor` 或 transformers；
- 不启动训练、pressure 或 full-scale；
- 不修改 `train_visual_router_online_streaming.py`；
- 不修改正式 evaluation 入口；
- 不新增 Bash launcher；
- 不把 checkpoint、scaler 或 run_dir 放进 P16a adapter；
- 不把 checkpoint path 设计成 `FeatureProvider` interface；
- 不声称正式 Visual Router 已迁移完成。
