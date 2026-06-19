# Stage 1 Launcher Architecture

创建日期：2026-06-19

## 1. 目标

本文记录 P5f 阶段的 Stage 1 及未来 Stage 实验启动层设计。目标是让最终用户可以继续用接近 TimeFuse / Quito `exp_scripts` 风格的 Bash launcher 启动实验，同时保证 Bash 只负责实验编排和资源绑定，不把核心训练逻辑、provider 读取细节或模型行为写进 shell。

本阶段只写文档，不新增 Bash 脚本，不新增 Python entrypoint，不实现 config system，不修改任何训练脚本。

目标分层如下：

```text
exp_scripts/*.sh
  -> scripts/*.py
  -> time_router runtime/protocol/provider/head/evaluator
```

长期原则：

- `exp_scripts/` 面向人和服务器资源，是可复现实验命令与后台运行策略层。
- `scripts/` 面向命令行入口，是极薄 Python adapter 层。
- `time_router/` 面向可测试 runtime 和研究模块，是 provider、protocol、head、evaluation 与 IO helper 的实现层。
- `configs/` 面向实验协议参数，是 branch-specific 参数与路径默认值的声明层。

## 2. `exp_scripts/` 职责

`exp_scripts/` 是 Bash launcher 层。它负责把一次实验如何在机器上运行说清楚，但不负责解释模型如何训练。

应属于 `exp_scripts/` 的职责：

- 选择 config 文件或 config 名称，例如 Stage 1 Visual Router full-scale、Stage 1 TimeFuse-style fusor full-scale、未来 Stage 2 held-out cell。
- 指定 GPU、`CUDA_VISIBLE_DEVICES`、CPU 线程、conda 环境和必要环境变量。
- 指定日志、`nohup`、`setsid`、`tmux` 或后台运行策略。
- 显式指定 `run_dir` 或 `output_root`，full-scale 推荐指向 `/data2/syh/Time/...` 下的运行目录。
- 保存完整可复现实验命令，包括 resume、stop、monitor 和 eval-only 命令。
- 做轻量 preflight，例如检查 config 文件存在、run_dir 不会覆盖重要结果、GPU 参数不为空。

不应属于 `exp_scripts/` 的职责：

- 不实现 provider 读取细节，例如 packed npy、SQLite index、feature shard streaming 或 oracle/TSF join。
- 不实现模型训练主体逻辑、loss、optimizer、scaler fit、评估指标或 checkpoint payload。
- 不在 Bash 中拼接复杂实验协议对象。
- 不把 `/data2` 写进 `time_router` provider contract；`/data2` 只作为具体 launcher 参数或运行 metadata 出现。
- 不解析 prediction cache manifest 内部 schema，不根据 CSV 字段推断专家顺序。

未来建议目录形态：

```text
exp_scripts/
└── stage1/
    ├── visual_router/
    │   ├── run_full_scale.sh
    │   ├── resume_full_scale.sh
    │   └── eval_only.sh
    └── timefuse_fusor/
        ├── run_full_scale.sh
        ├── resume_full_scale.sh
        └── eval_only.sh
```

上述文件名只是目标形态示例，P5f 不创建这些文件。

## 3. `scripts/` 职责

`scripts/` 是极薄 Python entrypoint 层。它把 CLI/config 转成 canonical protocol/runtime 调用，不在入口脚本中复制训练主体。

应属于 `scripts/` 的职责：

- 解析 config 路径、config 名称和少量 CLI override，例如 `--run-dir`、`--output-root`、`--resume-checkpoint`、`--eval-only`。
- 构造 `ExperimentProtocolSpec` 或 future runtime 所需的等价 spec。
- 把 config 中的 branch-specific 参数传给 runtime，例如 visual encoder、pseudo image 参数、TimeFuse feature schema、head/loss 配置。
- 调用 future `time_router` runtime，例如 `run_experiment(protocol_spec, run_dir=...)` 或等价入口。
- 做入口级错误提示，例如缺少 config、run_dir 为空、互斥参数同时出现。

不应属于 `scripts/` 的职责：

- 不实现 provider 读取细节。
- 不写模型训练 loop、batch iteration、loss backward、optimizer step、evaluation aggregation。
- 不直接读取 full-scale prediction arrays、feature cache shards、oracle parquet 或 TSF enrichment。
- 不决定 provider 的内部路径策略；路径由 config/launcher 显式传入，provider 只消费 resolved input。
- 不硬编码 `exp_scripts/` 路径或假设一定由 Bash 启动。

未来建议目录形态：

```text
scripts/
└── stage1/
    ├── run_visual_router.py
    └── run_timefuse_fusor.py
```

P5f 不新增这些 Python entrypoint。正式创建前应先让 `PredictionCacheExpertProvider` 与 evaluator adapter 在 smoke 中稳定。

## 4. `time_router/` 职责

`time_router/` 是 runtime、protocol、provider、feature、head、evaluation 和 IO helper 的实现层。它不应知道 Bash launcher 是否存在，也不应硬编码 `exp_scripts` 路径。

应属于 `time_router/` 的职责：

- runtime contract：`run_dir/status/metadata/checkpoints/logs/evaluation/predictions` 的结构化写出调度。
- protocols：`SplitSpec`、`ExpertBatch`、`FeatureBatch`、`RouterOutput`、`EvaluationInput`、`ExperimentProtocolSpec` 等轻量 contract。
- providers：`PredictionCacheExpertProvider`、future online expert provider、TimeFuse feature cache provider、Visual online ViT feature provider。
- features：pseudo image / encoder feature、TimeFuse-derived feature、future online TimeFuse feature。
- heads/models：Visual MLP head、TimeFuse linear-softmax head、future finetune ViT / joint training head 组合。
- evaluation：hard top-1、raw soft fusion、calibration、summary、per-sample rows、comparison 和 diagnostics。
- IO helper：JSON 原子写入、metadata payload、path resolver、prediction cache reader、future SQLite / checkpoint index helper。

不应属于 `time_router/` 的职责：

- 不知道 Bash launcher 文件名或 `exp_scripts/` 目录布局。
- 不决定 full-scale `run_dir` 一定在 `/data2`。
- 不读取 shell 生成的 `command.sh` 作为配置来源。
- 不把历史 pilot 输出目录或 legacy CSV schema 作为 canonical contract。

推荐未来目录形态：

```text
time_router/
├── runtime/
├── protocols/
├── providers/
├── features/
├── models/
├── evaluation/
└── io/
```

当前仓库已存在 `time_router/{data,io,evaluation,protocols}` 的早期共享模块；P5f 只描述未来边界，不创建新目录。

## 5. `configs/` 职责

`configs/` 是实验协议参数层。它应描述 Stage、branch、数据输入、runtime 输出和模型配置，但不把 full-scale 输出路径固定到仓库内部。

应属于 `configs/` 的职责：

- Stage 参数：Stage 1 vali/test、future Stage 2 held-out cell、future cross-cell split。
- branch 参数：Visual Router 与 TimeFuse-style fusor 的 branch-specific config。
- expert 输入：prediction cache manifest、merged cache root、array storage、model columns。
- feature 输入：visual pseudo image / encoder 配置、TimeFuse feature cache root、feature schema、future online feature 配置。
- head/loss 参数：Visual MLP hidden/dropout/loss mode、TimeFuse linear-softmax、future finetune ViT / joint training 参数。
- runtime 默认值：status/metadata/evaluation/prediction output 子目录名、checkpoint policy、logging 文件名。
- 扩展点：future finetune ViT、joint training、online expert、online TimeFuse feature、new split strategy。

不应属于 `configs/` 的职责：

- 不把 full-scale `run_dir` 默认写死到 repo 内，例如 `experiment_logs/run_outputs/...`。
- 不保存机器私有 GPU 编排策略；GPU/conda/nohup 更适合由 `exp_scripts/` 指定。
- 不把 provider 内部临时索引路径写成不可覆盖常量。
- 不把 oracle/TSF 作为可部署 test-time 动态特征。

full-scale 建议：

- config 可给出 `output_root` 的空默认或占位符。
- launcher 必须显式传入 `--run-dir` 或 `--output-root`。
- 大规模正式结果推荐显式指向 `/data2/syh/Time/run_outputs/...`。
- repo 只保存代码、配置、文档、小 fixture 和 smoke，不保存 full-scale checkpoint、prediction cache 或 feature cache。

未来建议目录形态：

```text
configs/
└── stage1/
    ├── visual_router/
    │   ├── full_scale.yaml
    │   └── smoke.yaml
    └── timefuse_fusor/
        ├── full_scale.yaml
        └── smoke.yaml
```

P5f 不新增 config 文件或 config loader。

## 6. `run_dir` 与 `/data2` 边界

`run_dir` 是一次运行的实际状态目录，不是 provider 的内部决定。full-scale 运行通常应放在 `/data2`，但这必须由 launcher 或用户显式传入。

边界结论：

- full-scale `run_dir` 通常在 `/data2/syh/Time/run_outputs/...`。
- repo 只保存代码、配置、文档、小 fixture、smoke 和结构索引。
- launcher 负责显式传入 `run_dir` 或 `output_root`，并把完整命令保存在运行目录。
- future runtime 接收 `run_dir` 并写 `status.json`、`metadata.json`、checkpoint、evaluation 和 prediction outputs。
- provider 不决定 `run_dir`，也不创建外层 output root。
- config 不应默认把 full-scale 输出落到 repo 内；小 smoke 可以使用临时目录或小 fixture。

路径示例口径：

```text
exp_scripts/stage1/timefuse_fusor/run_full_scale.sh
  -> scripts/stage1/run_timefuse_fusor.py --config configs/stage1/timefuse_fusor/full_scale.yaml --run-dir /data2/syh/Time/run_outputs/YYYY-MM-DD_stage1_timefuse_fusor_full_scale
  -> time_router runtime consumes resolved run_dir
```

该示例只说明边界，不代表 P5f 已实现对应脚本。

## 7. 与现有入口关系

当前 canonical-current / legacy 关系如下：

| 入口 | 短期状态 | 长期方向 |
| --- | --- | --- |
| `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` | Visual Router full-scale canonical-current | 短期保留为可运行正式主线；未来由 `scripts/stage1/run_visual_router.py` 调用 runtime，逐步下沉 ExpertProvider、Evaluator、Visual FeatureProvider 和 RouterHead |
| `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` | TimeFuse-style fusor canonical-current | 短期保留为可运行正式 baseline；未来由 `scripts/stage1/run_timefuse_fusor.py` 调用 runtime，逐步下沉 ExpertProvider、Evaluator、TimeFuse FeatureProvider 和 Head |
| `visual_router_experiments/stage1_vali_test_router/launch_timefuse_fusor_full_scale.py` | 当前 full-scale launcher / preflight / 后台进程管理层 | 短期保留；未来等 `exp_scripts/ + scripts/ + runtime` 稳定后，可迁移为 Bash launcher 或保留为 Python launcher compatibility layer |

过渡原则：

- 现有 streaming 入口继续作为 canonical-current，不在 P5f 迁移。
- `launch_timefuse_fusor_full_scale.py` 继续承担当前后台启动、PID/PGID、stop/resume、monitor 和 preflight 价值；它不是 provider adapter，也不是最终训练 runtime 本体。
- 新 `scripts/` entrypoint 出现后，先用 smoke 或小规模 fixture 验证，不直接替换 full-scale 长跑入口。
- 新 `exp_scripts/` 出现后，先包装新 `scripts/` entrypoint；不要让 Bash 直接调用旧训练脚本并声称已经完成 canonical runtime 迁移。
- 历史 pilot、offline embedding cache、LogisticRegression fusor、旧 OOM lookup 和非 streaming full-scale 入口继续按 P5d/P5e 标记为 reference-only / legacy，不为它们设计新 launcher contract。

## 8. 推荐未来目录形态

长期完整形态建议如下：

```text
configs/
└── stage1/
    ├── visual_router/
    └── timefuse_fusor/

exp_scripts/
└── stage1/
    ├── visual_router/
    └── timefuse_fusor/

scripts/
└── stage1/
    ├── run_visual_router.py
    └── run_timefuse_fusor.py

time_router/
├── runtime/
├── providers/
├── features/
├── models/
├── evaluation/
├── protocols/
└── io/

archive/
```

目录创建顺序应跟真实迁移一致。不要为了“目标形态好看”提前新增空目录。

## 9. P5f 后迁移顺序判断

P5f 之后不建议立即实现完整 Bash launcher，也不建议先做大而全 config system。更稳妥的小步顺序如下：

1. **先实现 `PredictionCacheExpertProvider` 的 smoke-only adapter**：它是 Visual Router 与 TimeFuse-style fusor 共享最小依赖，已有 `PredictionBatchReader`、packed fixture、protocol types 和 golden smoke 可锁定行为。
2. **再做 evaluator adapter**：复用 `time_router.evaluation` public API，从 `EvaluationInput` 复算 summary/rows，仍可只在 smoke 中接入。
3. **补最小 config skeleton**：只覆盖 smoke 和一个 canonical-current branch 的必要字段，用于驱动 adapter smoke；不急于支持所有历史入口。
4. **新增极薄 `scripts/` skeleton**：在 provider/evaluator smoke 稳定后，让 Python entrypoint 只解析 config 并构造 `ExperimentProtocolSpec`。
5. **最后新增 `exp_scripts/` Bash launcher**：先包装 smoke 或小规模 run，再扩展 full-scale `/data2` run_dir、nohup、日志、resume/stop/monitor。

判断理由：

- 先做 `PredictionCacheExpertProvider` 最容易被 golden smoke 证明不漂移。
- 先做完整 config 或 Bash，容易把旧训练脚本参数重新包装一遍，但不会真正收束 runtime/provider 边界。
- `scripts/` entrypoint 应等至少一个 provider/evaluator adapter 可调用后再出现，否则会变成空壳。
- Bash launcher 应在 Python runtime 入口稳定后再写，避免把还没定型的 CLI 固化到实验脚本中。

最低风险下一步：

```text
PredictionCacheExpertProvider smoke-only
  -> Evaluator adapter smoke-only
  -> minimal config skeleton
  -> scripts/stage1 thin entrypoint skeleton
  -> exp_scripts/stage1 Bash launcher
```

## 10. P5f 明确不做

- 不新增 Bash 脚本。
- 不新增 Python entrypoint。
- 不实现 config system。
- 不实现 runtime/run_dir helper。
- 不实现 provider adapter。
- 不修改 `PredictionBatchReader` / `OracleTsfReader` / evaluation / io / protocols。
- 不修改任何训练脚本。
- 不迁移 Visual Router / TimeFuse fusor 入口。
- 不实现 checkpoint index。
- 不实现 logging framework。
- 不接入 `/data2`。
- 不移动或删除历史代码。
- 不改模型结构、loss 或正式输出目录。

## 11. 验证门禁

P5f 是文档-only 变更，仍需确认既有共享模块 smoke 与编译门禁不受影响：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_protocol_types_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```
