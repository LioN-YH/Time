# Stage 1 公共模块迁移候选

审计日期：2026-06-19

## 1. 目标边界

本文件只提出后续重构候选，不代表相关代码已经合并。目标不是把 Visual Router 和 TimeFuse-style fusor塞进一个巨型脚本，而是把共同的数据与评估协议抽成稳定骨架，只保留必要分叉：

```text
SharedStage1BatchReader
  -> FeatureProvider
     ├── VisualFeatureProvider: x -> pseudo image -> frozen ViT embedding
     └── TimeFuseFeatureProvider: feature cache -> 17 维向量
  -> RouterHead
     ├── Visual MLP
     └── TimeFuse Linear-softmax
  -> Shared Trainer / Evaluator / Reporting
```

共享 batch 的概念字段应至少覆盖 `sample_keys`、稳定样本元信息、`features`、五专家 `y_pred`、共享 `y_true`，以及可选 oracle/TSF 诊断信息。oracle 或未来信息不得进入可部署 FeatureProvider；契约依据为 `visual_router_experiments/stage1_vali_test_router/stage1_cache_contract.md`。

## 2. 优先级 P0：共享数据平面

### 2.1 Manifest 与 sample-key reader

- 当前位置：`common/prediction_cache_schema.py` 已定义 key/schema；`build_full_scale_sample_manifest.py`、`train_visual_router_online_streaming.py`、`stage1_timefuse_fusor_streaming_reader.py` 和 `train_timefuse_fusor_streaming.py` 各自处理 manifest、split、shard 或 sample-key 集合。
- 候选抽象：统一 manifest schema 校验、split 下推、sample shard 发现、稳定顺序与批量 key 输出；不要把 Quito 历史窗口重载逻辑塞入该层。
- 约束：保持 `config_name + split + dataset + item + channel + window` key 不变；不得为了便利重新排序导致 packed row index 或 checkpoint resume 语义变化。

### 2.2 Prediction cache reader

- 当前位置：`common/prediction_array_io.py` 提供单记录读取；`fusion_utils.py`、`train_visual_router_online_streaming.py` 和 `stage1_timefuse_fusor_streaming_reader.py` 分别组装五专家 batch。
- 候选抽象：统一 manifest record 解析、相对路径解析、`per_sample_npy`/`packed_npy_v1`、batch-level grouped `np.load`、共享 `y_true` 校验和固定专家顺序。
- 约束：正式路径必须保留 `array_storage`、`y_true_row_index`、`y_pred_row_index`；不得退回每 sample 重复打开 packed 文件或全量 Python lookup。历史事故见 `experiment_logs/2026-06-17_stage1_96_48_s_streaming_visual_router_oom_fix_review_restart.md`。

### 2.3 Oracle 与 TSF reader

- 当前位置：正式生成器为 `build_full_scale_window_oracle_labels.py`、`build_full_scale_tsf_enrichment.py`；TimeFuse reader 单独构建 oracle SQLite，Visual 入口另有自己的监督读取路径。
- 候选抽象：按 `sample_key + metric` 批量读取 oracle、按 `sample_key` 读取 TSF 元信息，并明确 `required`、`diagnostic_only` 和缺失策略。
- 约束：oracle/TSF 可以参与训练辅助、分层汇总或诊断，但不得混入历史窗口特征或 test 时的动态专家调权。

### 2.4 SQLite 与 batch index

- 当前位置：`train_visual_router_online_streaming.py` 有 prediction SQLite；`stage1_timefuse_fusor_streaming_reader.py` 有 oracle/prediction SQLite；`train_timefuse_fusor_streaming.py` 又负责 index 完整性与复用判断。
- 候选抽象：统一建库、临时文件原子替换、schema/version、期望行数检查、已完成 index 复用、split 下推和 batch query。
- 约束：索引必须 shard-local 或对目标 key 过滤；不得扫描后把数千万 record 留在 Python 内存；必须兼容恢复运行。

## 3. 优先级 P1：共享评估平面

### 3.1 五专家 fusion 与 metrics

- 当前位置：`fusion_utils.py`、`train_visual_router.py`、`evaluate_soft_fusion_calibration.py` 和 `train_timefuse_fusor_streaming.py` 均包含部分 hard/soft fusion、MAE/MSE、权重诊断或汇总逻辑。
- 候选抽象：固定专家顺序、权重归一化检查、hard top-1、raw soft、temperature/top-k、MAE/MSE、oracle regret、selected counts 和 entropy/max-weight 诊断。
- 约束：所有指标必须从同一批数组复算；per-config 主表优先，macro average 仅作总览；calibration 不得读取 test oracle error 做逐样本调权。

### 3.2 输出 schema 与报告

- 当前位置：`train_visual_router.py`、`train_visual_router_online_streaming.py`、`evaluate_router_baselines.py`、`evaluate_soft_fusion_calibration.py` 和 `train_timefuse_fusor_streaming.py` 分别写 predictions、summary、metadata 和 Markdown。
- 候选抽象：统一 prediction 行的稳定标识、五个 `weight_*`、selected/oracle 字段、summary 方法名、metadata lineage 和 comparison 表。
- 约束：保留现有下游 calibration 所需字段；迁移前先用已有 1k/full-scale 输出做 schema golden comparison。

### 3.3 Logging、status 与 checkpoint

- 当前位置：多个 builder、launcher 和 trainer 重复实现时间格式、atomic JSON、`status.json`、metadata、进度字段、checkpoint index 和 summary 写入。
- 候选抽象：公共 atomic JSON writer、阶段状态机、异常落盘、资源快照、checkpoint 元数据和中文运行摘要。
- 约束：不一次性改写现有 launcher 的 stop/resume 语义；状态字段需向后兼容已有监控与 `HANDOFF.md` 命令。

## 4. 优先级 P2：共享训练骨架

### 4.1 FeatureProvider

- `VisualFeatureProvider`：消费 sample batch 对应的历史窗口 `x`，复用 `common/pseudo_imageization.py` 和 `common/vit_embedding_utils.py`，在线返回 embedding；不得落盘 full-scale embedding。
- `TimeFuseFeatureProvider`：消费同批 sample key，从正式 feature shard 返回 17 维向量；scaler pass 只读 feature，不应加载 prediction arrays。当前正确行为由 `build_timefuse_feature_cache_from_manifest.py` 和 `experiment_logs/2026-06-19_stage1_timefuse_fusor_reader_scaler_optimization_restart.md` 证明。
- 两者输出统一的 `features + sample_keys`，但允许不同设备、dtype、吞吐与缓存策略。

### 4.2 RouterHead 与 loss

- Visual 保留 MLP 和现有 `fusion_huber_kl`/classification 兼容逻辑，当前复用核心位于 `train_visual_router.py`。
- TimeFuse 保留单层 `nn.Linear -> softmax` 和 `SmoothL1Loss(beta=0.01)` 代表口径，当前位于 `train_timefuse_fusor_streaming.py`。
- 不应为了代码统一而把两种 head 或 loss 强行改成相同；统一的是输入输出协议、训练循环和评估，不是研究变量。

### 4.3 Trainer/Evaluator

- 候选共享部分：vali-only scaler、epoch/batch 驱动、test-only forward、checkpoint/resume、hard/raw-soft 评估和报告。
- 保留扩展点：FeatureProvider 是否需要 GPU 前向、router-specific loss、是否启用 ViT DataParallel、是否只执行 eval-only。
- 第一轮重构不应直接替换两个正式入口；先增加共享模块和等价 smoke，再逐个迁移 consumer。

## 5. 路径与配置管理

- 当前问题：Stage 1 多个入口硬编码 `/home/shiyuhong/Time`、`/data2/syh/Time/run_outputs`、特定 full-scale 日期目录、默认 config/checkpoint 路径。
- 候选抽象：workspace root、代码根、输出根、sample manifest、prediction cache、oracle、TSF、feature cache 和 checkpoint lineage 的集中解析；CLI 显式参数优先，集中默认值其次。
- 约束：不要在首轮迁移中改变现有产物路径；先提供兼容解析和 metadata 自证，再逐步去除脚本内硬编码。

## 6. 建议迁移顺序与验收

1. 为现有 batch array 读取建立小规模 golden fixture，锁定 sample 顺序、shape、MAE/MSE 和 packed row index。
2. 抽 prediction batch reader，再让 Visual 与 TimeFuse smoke 分别切换；两边输出必须逐样本一致。
3. 抽 oracle/TSF 和 SQLite index 生命周期，验证 index 复用、损坏重建、split 下推和 RSS 不回退。
4. 抽 metrics/output schema，比较既有 1k 或 pressure 输出的 summary 与 selected counts。
5. 抽 logging/path/config 等低风险设施。
6. 最后引入 FeatureProvider/RouterHead 训练骨架；先保留旧入口作为对照，full-scale 等价验证后再决定迁移或归档。

每一步都应独立提交、独立 smoke，并按 `AGENTS.md` 写中文实验日志。任何阶段出现 sample 数、专家顺序、数组 shape、指标或 checkpoint resume 差异，都应停止后续迁移并先解释差异。

## 7. 本次明确不做

- 不移动、删除、重命名现有脚本。
- 不改变 cache schema、sample key、模型结构、loss 或正式输出目录。
- 不把 offline embedding cache 恢复为正式路线。
- 不把尚未完成的 TimeFuse GPU2/3 运行结果写成已完成结论。
- 不在路线审计提交中实现任何上述候选。
