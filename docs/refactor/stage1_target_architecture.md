# Stage 1 目标架构设计

设计日期：2026-06-19

## 1. 设计结论

Stage 1 后续目标不是把 Visual Router 和 TimeFuse-style fusor 合并成一套模型，而是把二者共享的数据、索引、监督和评估主干收束到稳定 Python package 中，再把特征生成和 router head 保留为显式分支。本文只定义未来架构边界，不代表代码已经迁移；本次不移动、不删除、不重命名、不改写任何正式代码。

P4 后架构转向结论：Stage 1 后续先收束 canonical entrypoint 和 runtime 契约，再设计 FeatureProvider 与共享 config。正式训练主干只保留 streaming Visual Router 和 streaming TimeFuse-style fusor baseline 两条；LogisticRegression fusor、offline ViT embedding cache、旧 OOM lookup、pilot-only 脚本和非 streaming full-scale 入口不再作为新架构兼容目标。

目标运行形态如下：

```text
scripts / exp_scripts / configs
  -> time_router.data manifest/oracle/TSF readers
  -> time_router.io prediction cache + SQLite index
  -> FeatureProvider
     ├── VisualFeatureProvider: x -> pseudo image -> frozen ViT embedding
     └── TimeFuseFeatureProvider: sample_key -> 17维 feature cache
  -> RouterHead
     ├── Visual MLP / fusion_huber_kl
     └── TimeFuse Linear-softmax / SmoothL1 fusion loss
  -> time_router.training train/eval loop
  -> time_router.evaluation metrics/fusion/report
```

## 2. 未来 Python Package 边界

### 2.1 `time_router/data/`

职责是读取与校验稳定样本协议，不做模型前向、不做数组融合。

- `manifest_reader`：读取 full-scale sample manifest、split、shard index 和稳定 `sample_key`，保持 `config_name + split + dataset_name + item_id + channel_id + window_index` 不变。
- `oracle_reader`：读取 window-level oracle label，仅作为监督、上限或诊断信息；不得进入可部署 FeatureProvider。
- `tsf_reader`：读取 TSF enrichment，用于分层汇总、baseline 或诊断。
- schema 校验：复用当前 `visual_router_experiments/common/prediction_cache_schema.py` 的 stable key 与字段约束。

### 2.2 `time_router/io/`

职责是数组与索引 I/O，不承载训练策略。

- `prediction_cache_reader`：统一 `per_sample_npy` 与 `packed_npy_v1`，按固定五专家顺序输出 batch 级 `y_pred` 与共享 `y_true`。
- `sqlite_index`：统一 shard-local SQLite 建库、原子替换、schema/version、完成态复用、split 下推和 batch query。
- `path_resolver`：集中解析 workspace root、输出根、manifest、prediction cache、oracle、TSF、feature cache 和 checkpoint lineage。

### 2.3 `time_router/features/`

职责是把共享 sample batch 转成 router/fusor 可用特征；不同路线只在这里分叉。

- `FeatureProvider` interface：输入 batch 的 sample key、稳定元信息和可选历史窗口句柄；输出 `features`、`sample_keys` 和必要 metadata。
- `VisualFeatureProvider`：从历史窗口 `x` 在线生成 pseudo image tensor，再经过 frozen ViT 得到 embedding；full-scale 不落盘伪图像 tensor 或 ViT embedding。
- `TimeFuseFeatureProvider`：按 `sample_key` 从正式 17 维 feature cache shard 读取特征；scaler pass 可以只读 feature，不应加载五专家 prediction arrays。

### 2.4 `time_router/models/`

职责是 router head 和 loss 的研究变量，不处理 cache I/O。

- Visual Router head：保留当前 MLP，以及 `fusion_huber_kl` / classification 等兼容模式。
- TimeFuse-style head：保留 `nn.Linear -> softmax` 和 `SmoothL1Loss(beta=0.01)` 的 weighted fusion 口径。
- 不把两种 head 或 loss 强行统一；统一的是输入输出协议和评估口径。

### 2.5 `time_router/evaluation/`

职责是固定可复算的 metrics、fusion 和报告输出。

- `metrics`：MAE、MSE、oracle regret、selected counts、entropy、max weight 等。
- `fusion`：固定专家顺序、hard top-1、raw soft fusion、temperature/top-k calibration。
- `report`：统一 summary、comparison、prediction schema、metadata lineage 和 Markdown/CSV 输出。

### 2.6 `time_router/training/`

职责是共享训练与评估循环，但保留路线扩展点。

- vali-only scaler、epoch/batch 驱动、checkpoint/resume、eval-only、train-only 和状态落盘。
- 支持 FeatureProvider 是否需要 GPU 前向、是否用 DataParallel、router-specific loss、以及大规模 streaming 读写。
- 首轮迁移只应让现有入口逐步消费共享模块，不一次性替换两个正式入口。

### 2.7 顶层运行目录

- `scripts/`：未来面向用户的轻量 CLI 包装，调用 package API，不放研究逻辑。
- `configs/`：未来集中保存默认路径、full-scale 输出、模型和评估配置；CLI 显式参数优先于默认值。
- `exp_scripts/`：未来保留实验编排、后台 launcher、资源绑定和一次性运行计划。
- `archive/`：未来存放历史路线、废弃入口和 pilot-only 脚本；进入 archive 的文件必须先有等价替代、实验日志和 golden smoke 证据。

## 3. 共享主干

共享主干必须先于训练骨架收束，避免 Visual Router 与 TimeFuse-style fusor 继续复制不同版本的数据读取逻辑。

| 主干模块 | 目标职责 | 关键约束 |
| --- | --- | --- |
| manifest reader | 统一 sample manifest、split、shard 和 sample_key 顺序 | 不改变 stable key，不隐式重排 |
| prediction cache reader | batch 读取五专家 `y_pred` 与共享 `y_true` | 固定专家顺序；保留 `packed_npy_v1` row index；不得退回全量 Python lookup |
| oracle/TSF reader | 批量读取 oracle label 与 TSF enrichment | oracle/TSF 只能作为监督、上限或诊断，不进入 test-time 可部署特征 |
| SQLite index | shard-local 建库、复用、损坏重建、split 下推和 batch query | 不把数千万记录留在 Python 内存；必须兼容恢复运行 |
| metrics/fusion/report | 统一 hard、raw soft、calibration、MAE/MSE、comparison 和输出 schema | 所有指标从同一批数组复算；输出字段需兼容现有 calibration |

## 4. 两个正式分支

### 4.1 VisualFeatureProvider 分支

```text
sample batch
  -> Quito 历史窗口 x
  -> pseudo image tensor
  -> frozen ViT
  -> embedding
  -> Visual MLP router
  -> 五专家权重
```

该分支的研究变量是视觉表示和 MLP router。full-scale 主线固定为在线生成，不长期保存 pseudo image tensor 或 ViT embedding cache。未来迁移时应复用 `common/pseudo_imageization.py` 和 `common/vit_embedding_utils.py` 的既有行为，并用 golden smoke 确认迁移没有改变 prediction batch 与 fusion 指标。

当前 canonical entrypoint 是 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py`。`train_visual_router.py` 和 `train_visual_router_online.py` 继续保留小规模复现和历史对照价值，但不作为 full-scale 主干。

### 4.2 TimeFuseFeatureProvider 分支

```text
sample batch
  -> sample_key
  -> 17维 TimeFuse-derived feature cache
  -> Linear-softmax fusor
  -> 五专家权重
```

该分支的研究变量是 TimeFuse-derived 17 维历史窗口元特征和单层 softmax fusor。feature cache 只来自历史窗口 `x`，不读取未来 `y`、专家预测或 oracle label。scaler 和训练阶段应优先复用 shard-aware reader、split 下推和 batch-level grouped packed npy 读取。

当前 canonical entrypoint 是 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`，正式 full-scale 后台编排入口是 `launch_timefuse_fusor_full_scale.py`。早期 LogisticRegression hard-label router 只作为 legacy/deprecated 历史口径。

## 4.3 Canonical Runtime 最小契约

未来两条正式分支都应收束到同一类 runtime 目录契约：

- `run_dir/`：单次运行独立目录，不覆盖已完成正式结果。
- `status.json`：记录 `status`、`phase`、`updated_at`、关键进度、错误和可恢复状态。
- `metadata.json`：记录 entrypoint、config、args、inputs、outputs、model columns、array storage、feature schema、checkpoint 和资源策略。
- `checkpoints/`：训练型入口保存 epoch checkpoint、latest checkpoint 和 latest index；纯 eval-only 入口在 metadata 中说明 checkpoint 来源。
- `predictions/` 或 evaluation outputs：保存 per-sample predictions、summary、comparison 或 calibration 输出。
- `logs/` 或 `main.log`：保存主日志、launcher 日志和后台接手所需命令。

P4 helper 只提供原子 JSON、路径解析和 metadata-like payload 基础能力；run_dir 命名、launcher、checkpoint payload、resume policy、best/latest 选择和 logging framework 仍属于 runtime/training 层。

## 5. 未来进入 `archive/` 的旧代码类别

以下类别未来应在有替代模块和验证证据后进入 `archive/`，但本次不移动任何文件。

- full-scale offline embedding cache：包括离线 ViT embedding `.npy` builder/launcher 口径；只保留小规模 debug 或历史复现价值。
- logistic regression fusor：TimeFuse-derived hard-label LogisticRegression router 已是 legacy/deprecated，不作为正式 TimeFuse-style fusor。
- old OOM routes：全量 Python prediction lookup、旧 `_1epoch/` OOM 输出路线，以及只适合 120/1k 的全量暂存 embedding 入口。
- pilot-only scripts：固定小规模、固定 1k、dry-run 或历史验证脚本；迁移正式逻辑后应只作为历史复现材料。
- 非 streaming / 早期 online full-scale 入口：不再承担 canonical training，不再为其新增 status/metadata/checkpoint 兼容 helper。

## 6. Golden Smoke 门禁

`tests/smoke/stage1_golden_smoke.py` 是后续 Stage 1 重构的最小等价门禁。每次抽取或替换以下模块之前和之后都必须运行：

- reader：manifest reader、prediction batch reader、oracle/TSF reader；
- SQLite：index 建库、复用、损坏重建、split 下推和 batch query；
- metrics：hard top-1、raw soft fusion、MAE/MSE、专家顺序；
- output schema：prediction/summary/comparison 字段和 sample 顺序。

默认命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
```

如果 smoke 失败，必须先解释 sample_key 顺序、专家顺序、array shape、packed row index 或指标差异，再继续迁移。不能用 full-scale 运行成功替代该 smoke，因为 full-scale 成功不能精确证明小规模读取和融合契约未漂移。

## 7. 本文明确不做

- 不实现 `time_router/` package。
- 不新增空目录。
- 不修改 import。
- 不移动、删除、重命名或改写正式代码。
- 不改变 cache schema、sample key、模型结构、loss 或正式输出目录。
