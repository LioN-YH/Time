# Stage 1 代码目录整理

日志日期：2026-06-14 15:04:52 CST

## 目的

接续上一窗口中断的 `/home/shiyuhong/Time/visual_router_experiments/stage1_vali_test_router` 代码整理工作，明确 Stage 1 根目录和 `pilot/` 子目录的边界，减少正式入口、过渡性 launcher、临时脚本和缓存文件混放带来的误读。

## 背景

Stage 1 最近连续新增了 sample manifest、prediction cache shard、ViT embedding、online Visual Router、soft fusion calibration 和 1k launcher 相关脚本。上一轮工作已把部分 1k ViT embedding launcher 收入 `pilot/`，但文档和目录状态仍存在不一致：

- Stage 1 根目录 README 仍引用已经不存在或尚未实现的旧脚本名；
- `96_48_S` 1k prediction cache launcher 属于固定规模资源编排脚本，不适合作为通用正式入口长期放在根目录；
- 根目录存在临时 HF ViT 下载脚本 `tempt.py`；
- `pilot/__pycache__/` 属于可再生成缓存，不应作为长期结构保留。

## 操作

1. 检查当前 git 状态、Stage 1 文件树、实验日志和结构文档。

2. 确认脚本边界：

   - 保留在 Stage 1 根目录的正式可复用入口：
     - `build_stage1_sample_manifest.py`
     - `build_prediction_cache_from_manifest.py`
     - `merge_prediction_cache_shards.py`
     - `evaluate_router_baselines.py`
     - `build_vit_embeddings.py`
     - `train_visual_router.py`
     - `train_visual_router_online.py`
     - `evaluate_soft_fusion_calibration.py`
   - 收入 `pilot/` 的固定规模或过渡性 launcher：
     - `launch_96_48_s_1k_prediction_cache_pilot.py`
     - `launch_96_48_s_1k_vit_embedding_pilot.py`

3. 移动文件：

   ```text
   visual_router_experiments/stage1_vali_test_router/launch_96_48_s_1k_prediction_cache.py
   -> visual_router_experiments/stage1_vali_test_router/pilot/launch_96_48_s_1k_prediction_cache_pilot.py
   ```

4. 清理临时文件和缓存：

   ```text
   tempt.py
   visual_router_experiments/stage1_vali_test_router/pilot/__pycache__/
   ```

   其中 `tempt.py` 只是手动下载 `google/vit-base-patch16-224` 的临时脚本，不是 Stage 1 可复现实验入口；`__pycache__/` 可由 Python 解释器再生成。

5. 更新文档：

   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/pilot/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `WORKSPACE_STRUCTURE.md`

6. 检查活文档中是否仍存在旧脚本名：

   ```text
   rg -n "launch_96_48_s_1k_prediction_cache\.py|launch_96_48_s_1k_vit_embedding\.py|build_prediction_cache\.py|evaluate_router\.py|summarize_results\.py|tempt.py" \
     visual_router_experiments WORKSPACE_STRUCTURE.md experiment_logs/README.md
   ```

## 结果

Stage 1 根目录当前只保留正式可复用代码入口和文档，不再包含 `96_48_S` 1k 专用 launcher：

```text
build_stage1_sample_manifest.py
build_prediction_cache_from_manifest.py
merge_prediction_cache_shards.py
evaluate_router_baselines.py
build_vit_embeddings.py
train_visual_router.py
train_visual_router_online.py
evaluate_soft_fusion_calibration.py
```

`pilot/` 当前包含小规模验证脚本、结构特征 pilot、oracle/enrichment 工具和两个 1k 专用 launcher：

```text
launch_96_48_s_1k_prediction_cache_pilot.py
launch_96_48_s_1k_vit_embedding_pilot.py
```

文档已同步更新：

- Stage 1 README 去掉了不存在的 `build_prediction_cache.py`、`evaluate_router.py`、`summarize_results.py` 引用；
- Stage 1 README 和 `pilot/README.md` 明确了正式入口与 pilot/过渡 launcher 的边界；
- `stage1_protocol_and_plan.md` 将 1k 专用 launcher 标记为 `pilot/` 下的固定规模编排脚本；
- `WORKSPACE_STRUCTURE.md` 已更新 Stage 1 根目录和 `pilot/` 的文件职责说明。

Stage 1 README、`pilot/README.md`、`stage1_protocol_and_plan.md` 和 `WORKSPACE_STRUCTURE.md` 中未发现需要继续按正式入口查找的旧脚本名。`experiment_logs/README.md` 和本日志保留 `tempt.py`、旧 launcher 路径等文字，是为了记录本次删除和移动动作；历史实验日志中的旧路径也保留，因为它们反映当时真实操作，不在本轮回写修改。

## 结论

本次整理后，`stage1_vali_test_router/` 的边界更清楚：

- 根目录用于长期复用的正式实验入口；
- `pilot/` 用于小规模验证、过渡性 launcher 和固定规模资源编排；
- 临时下载脚本和 Python 缓存不再作为长期工作区结构保留。

这次整理没有启动新实验，没有删除已完成 smoke test、正式结果、数据产物或 run_outputs。

## 下一步方案

1. 后续如果启动 1k prediction cache，可使用 `pilot/launch_96_48_s_1k_prediction_cache_pilot.py` 复现当前固定规模 launcher，但若扩展到三 config，应实现通用编排层而不是继续复制 `96_48_S` 专用脚本。
2. 继续保留 online Visual Router 路线：不启动 1k ViT embedding cache launcher，不长期保存伪图像 tensor 或 ViT embedding `.npy`。
3. 下一轮正式化重点是统一 evaluator / reporter，把 hard top-1、raw soft、best calibrated soft、baseline 和 oracle 汇总到同一张 config-level 表。

## 追加整理：根据 online 路线收拢离线 embedding 代码

追加时间：2026-06-14 15:35:00 CST

### 追加目的

根据 `experiment_logs/2026-06-14_stage1_online_visual_router_smoke.md` 已确定的后续路线，继续整理 Stage 1 代码目录：

- 不再先缓存 1k ViT embedding；
- 不启动 1k ViT embedding launcher；
- 不长期保存伪图像 tensor 或 ViT embedding `.npy`；
- online 主线应直接运行内构造 `x -> pseudo image -> frozen ViT -> CLS embedding -> router`。

因此，原先位于 Stage 1 根目录的离线 ViT embedding cache builder 不应继续作为正式入口展示。

### 追加操作

1. 梳理依赖关系，确认 `train_visual_router_online.py` 仍复用了原 `build_vit_embeddings.py` 中的 ViT 输入/输出 helper，包括：

   - `make_pseudo_images()`
   - `pool_vit_outputs()`
   - `resolve_dtype()`
   - `parse_period_candidate_arg()`
   - window batch 索引工具

2. 新增正式共享工具：

   ```text
   visual_router_experiments/common/vit_embedding_utils.py
   ```

   该文件只提供运行内 tensor 工具，不保存 `.npy`，不生成 embedding manifest，不创建长期 cache。

3. 将离线 embedding cache builder 从 Stage 1 根目录移入 `pilot/`：

   ```text
   visual_router_experiments/stage1_vali_test_router/build_vit_embeddings.py
   -> visual_router_experiments/stage1_vali_test_router/pilot/build_vit_embeddings_pilot.py
   ```

   新文件顶部已注明：该脚本只用于复现早期离线 embedding smoke、做小规模缓存对照或调试 encoder 口径，不作为当前 online 主线入口。

4. 修改 `train_visual_router_online.py`：

   - 改为从 `visual_router_experiments/common/vit_embedding_utils.py` 导入 ViT helper；
   - 不再依赖任何 `pilot/` 下的离线 cache builder。

5. 修改 `pilot/launch_96_48_s_1k_vit_embedding_pilot.py`：

   - launcher 指向 `pilot/build_vit_embeddings_pilot.py`；
   - 文档说明该 launcher 只是 1k embedding cache smoke / 历史对照，当前正式 online 方案仍使用运行内存缓存。

6. 更新文档：

   - `visual_router_experiments/common/README.md`
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `visual_router_experiments/stage1_vali_test_router/pilot/README.md`
   - `visual_router_experiments/stage1_vali_test_router/stage1_protocol_and_plan.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`

### 追加结果

当前 Stage 1 根目录不再包含离线 embedding cache builder。正式主线保留：

```text
build_stage1_sample_manifest.py
build_prediction_cache_from_manifest.py
merge_prediction_cache_shards.py
evaluate_router_baselines.py
train_visual_router.py
train_visual_router_online.py
evaluate_soft_fusion_calibration.py
```

`pilot/` 新增或保留以下离线/过渡入口：

```text
build_vit_embeddings_pilot.py
launch_96_48_s_1k_vit_embedding_pilot.py
```

共享 helper 位置为：

```text
visual_router_experiments/common/vit_embedding_utils.py
```

online 主线现在从 `common/vit_embedding_utils.py` 获取 ViT `pixel_values` 构造、CLS/patch pooling、dtype 解析和窗口 batch 索引工具，不依赖会写 `.npy` 的离线 pilot 脚本。

### 追加结论

本次追加整理后，代码结构与当前实验路线一致：

- Stage 1 根目录保留在线训练、prediction cache、baseline 和 calibration 等主线入口；
- 离线 ViT embedding cache builder 和 1k embedding launcher 被明确归为 `pilot/` 历史对照 / 小规模 ablation；
- `common/` 承接 online 主线复用逻辑，避免主线代码依赖 pilot；
- 未启动新实验，未删除已有 run_outputs 或历史 smoke 结果。

### 追加下一步

1. 后续如果需要复查离线 embedding 结果，可显式调用 `pilot/build_vit_embeddings_pilot.py`，但不把它作为当前 1k 路线前置步骤。
2. 1k 路线仍应先完成五专家 prediction cache，再使用 `train_visual_router_online.py` 运行内生成 embedding 并训练 router。
3. 若后续将 `train_visual_router.py` 中离线 manifest CLI 也完全收拢，应先把其中被 online 复用的 router 训练/评估函数抽成 `router_training_utils.py` 或等价公共模块，再移动离线 CLI，避免 online 主线反向依赖 pilot。
