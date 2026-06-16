# Stage 1 代码文件功能梳理

日志日期：2026-06-15 23:04:06 CST

## 目的

对 `visual_router_experiments/stage1_vali_test_router/` 当前代码与文档文件进行只读梳理，明确各文件的职责、是否属于正式主线、是否属于历史 pilot 或辅助工具，为后续继续推进 Stage 1 实验前的目录整理和任务拆分提供依据。

## 背景

用户反馈 Stage 1 目录“看起来好混乱”，希望在继续推进前先弄清楚各个文件具体负责什么。当前目录同时包含 `96_48_S` 1k 中等规模链路、full-scale 正式长跑工具、历史离线 ViT embedding、TimeFuse-style fusor baseline、soft fusion calibration、pilot 脚本和协议文档，因此需要先按功能分层，而不是直接继续新增实验代码。

## 操作

1. 使用 `find visual_router_experiments/stage1_vali_test_router -maxdepth 3 -type f | sort` 和目录列表检查 Stage 1 文件结构。
2. 使用 `wc -l` 统计根目录脚本、Markdown 文档和 `pilot/` 脚本规模，确认当前目录包含多个 300 行以上的入口脚本。
3. 使用 `rg` 检索 `def`、`class`、`argparse`、`to_csv`、`write_text`、`np.save` 等关键位置，定位每个文件的入口参数、核心函数和主要输出文件。
4. 使用 `sed` 阅读根目录 README、协议文档、cache contract、prediction cache 设计文档、feature/RL 支线文档、正式脚本头部注释、`fusion_utils.py` 共享逻辑和 `pilot/README.md`。
5. 只读检查 `git status --short`，确认工作区已有多项未提交修改和新增日志；本次不改动 Stage 1 代码文件，只新增本中文日志并更新日志总览。

## 结果

本次确认 `stage1_vali_test_router/` 根目录主要分为五类：

1. 文档与协议：`README.md`、`stage1_protocol_and_plan.md`、`stage1_cache_contract.md`、`prediction_cache_design.md`、`feature_and_rl_extension_notes.md`。
2. 样本与 prediction cache 主线：`build_stage1_sample_manifest.py`、`build_full_scale_sample_manifest.py`、`build_prediction_cache_from_manifest.py`、`merge_prediction_cache_shards.py`、`launch_full_scale_prediction_cache.py`、`run_full_scale_dry_run.py`。
3. baseline / fusion / calibration：`evaluate_router_baselines.py`、`fusion_utils.py`、`evaluate_soft_fusion_calibration.py`。
4. visual router 训练入口：`train_visual_router.py`、`train_visual_router_online.py`、`train_visual_router_online_streaming.py`。
5. 历史和小规模验证：`pilot/` 下的 prediction cache pilot、在线伪图像化 pilot、离线 ViT embedding cache pilot、结构特征 cache/router pilot、1k 专用 launcher、oracle/TSF enrich 工具。

关键判断如下：

- 当前正式 full-scale 主线应优先使用 `build_full_scale_sample_manifest.py`、`launch_full_scale_prediction_cache.py`、`build_prediction_cache_from_manifest.py`、`merge_prediction_cache_shards.py`、`evaluate_router_baselines.py`、`train_visual_router_online_streaming.py` 和 `evaluate_soft_fusion_calibration.py`。
- `train_visual_router_online.py` 适合 120/1k 规模复现，因为它会在运行内暂存全部 embedding；真正 full-scale 不应依赖该入口。
- `train_visual_router.py` 是离线 embedding manifest 或 online wrapper 复用的 MLP router 训练核心，但当前正式路线不鼓励长期保存 ViT embedding `.npy`。
- `fusion_utils.py` 是共享库，不应被当作命令行入口。
- `pilot/` 下脚本多数是历史验证、固定规模 launcher 或小规模对照入口，不应再作为正式 full-scale 主链路的主要入口。
- `__pycache__/` 只包含 Python 运行生成物，不是应阅读或维护的源码。

## 结论

Stage 1 目录混乱的主要原因不是单个脚本职责不清，而是历史路线、pilot 结果、正式 full-scale 工具和实验结果说明都叠在同一个 README 与同一层目录认知中。根目录里的正式脚本大多已有中文头部注释和明确输入输出，真正需要整理的是“当前推荐入口”和“历史/废弃/仅 pilot 入口”的边界。

后续推进时，建议把 Stage 1 当前可执行主线固定为：

```text
full-scale sample manifest
-> full-scale prediction cache launcher / shards
-> merge prediction cache
-> oracle / TSF / baseline / TimeFuse-style fusor
-> streaming online visual router
-> soft fusion calibration
-> unified report
```

其中离线 ViT embedding cache、LogisticRegression 结构特征 router、1k 专用 launcher 和旧 per-sample cache pilot 均应保留为历史对照或小规模调试入口，不再混入正式执行路径。

## 下一步方案

1. 向用户给出按文件逐项解释的功能表，并明确哪些文件是“现在应优先使用”的正式主线。
2. 若用户同意整理目录，可下一步只改文档：在 `README.md` 顶部增加“当前主线入口清单 / 历史入口清单”，把大量历史 run 结果移动到单独 history 文档或缩短为索引。
3. 等 full-scale prediction cache 已完成的后续工作继续推进时，优先进入 merge、完整性校验、oracle/TSF/baseline、streaming router 和 calibration，而不是再新增 embedding cache 或 pilot 脚本。
