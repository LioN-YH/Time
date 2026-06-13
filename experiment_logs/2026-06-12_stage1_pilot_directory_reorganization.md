# Stage 1 Pilot 代码目录整理

日志日期：2026-06-12 21:01:10 CST

## 目的

将 Stage 1 中已经变多的 pilot 脚本从正式实验目录根部移入独立 `pilot/` 子目录，避免小规模验证代码、正式实验入口和长期复用评估脚本混在一起，并把该目录规范写入项目级协作要求。

## 背景

Stage 1 已经完成 prediction cache pilot、window oracle pilot、TSF cell enrichment 和非视觉 baseline 评估。随着 pilot 脚本数量增加，`visual_router_experiments/stage1_vali_test_router/` 根目录开始同时包含 pilot 验证脚本和可复用评估脚本，不利于后续区分正式实验流程与小规模验证流程。

本次整理遵循以下判断：

- `build_prediction_cache_pilot.py`、`compute_window_oracle_from_cache.py`、`enrich_cache_with_tsf_cell.py` 是当前 pilot 流程的一部分，应进入 `pilot/`；
- `evaluate_router_baselines.py` 是正式 Stage 1 也需要复用的非视觉 baseline evaluator，应保留在 Stage 1 根目录；
- 历史实验日志中的旧路径记录真实发生过的命令，不回改历史日志，只在本日志说明迁移后的新路径。

## 操作

1. 新建目录：

   ```text
   visual_router_experiments/stage1_vali_test_router/pilot/
   ```

2. 移动三个 pilot 脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/build_prediction_cache_pilot.py
   -> visual_router_experiments/stage1_vali_test_router/pilot/build_prediction_cache_pilot.py

   visual_router_experiments/stage1_vali_test_router/compute_window_oracle_from_cache.py
   -> visual_router_experiments/stage1_vali_test_router/pilot/compute_window_oracle_from_cache.py

   visual_router_experiments/stage1_vali_test_router/enrich_cache_with_tsf_cell.py
   -> visual_router_experiments/stage1_vali_test_router/pilot/enrich_cache_with_tsf_cell.py
   ```

3. 新增 `pilot/__init__.py` 和 `pilot/README.md`，说明 pilot 脚本的用途、限制和迁出规则。
4. 更新 `visual_router_experiments/stage1_vali_test_router/README.md`，明确 pilot 脚本放入 `pilot/`，正式复用入口保留在 stage 根目录。
5. 更新 `prediction_cache_design.md`，将推荐 pilot cache builder 路径改为 `stage1_vali_test_router/pilot/`。
6. 更新 `AGENTS.md`，新增长期规范：每个 stage 的探索性、小规模或半成品验证代码应集中放入 stage 内的 `pilot/` 子目录；正式入口和跨阶段工具不得长期混放在 `pilot/`。
7. 更新 `WORKSPACE_STRUCTURE.md`，记录 `pilot/` 子目录和当前 Stage 1 根目录脚本职责。

## 结果

整理后的 Stage 1 目录主要结构为：

```text
visual_router_experiments/stage1_vali_test_router/
├── README.md
├── __init__.py
├── evaluate_router_baselines.py
├── prediction_cache_design.md
└── pilot/
    ├── README.md
    ├── __init__.py
    ├── build_prediction_cache_pilot.py
    ├── compute_window_oracle_from_cache.py
    └── enrich_cache_with_tsf_cell.py
```

已执行语法检查：

```bash
python -m py_compile \
  visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py \
  visual_router_experiments/stage1_vali_test_router/pilot/build_prediction_cache_pilot.py \
  visual_router_experiments/stage1_vali_test_router/pilot/compute_window_oracle_from_cache.py \
  visual_router_experiments/stage1_vali_test_router/pilot/enrich_cache_with_tsf_cell.py
```

检查通过，移动后脚本语法无错误。

## 结论

Stage 1 代码目录已经完成 pilot 与正式复用入口的初步分离。后续新增 Stage 1 pilot 脚本应默认进入 `stage1_vali_test_router/pilot/`；当某段逻辑变成正式实验流程时，应迁出到 Stage 1 根目录或 `visual_router_experiments/common/`。

## 下一步方案

1. 后续运行 pilot cache/oracle/enrichment 脚本时使用新路径。
2. 继续实现正式 Stage 1 的 visual feature / embedding / router 训练入口时，直接放在 `stage1_vali_test_router/` 根目录。
3. Stage 2 或后续阶段如出现小规模验证脚本，也按本次写入 `AGENTS.md` 的规范建立各自的 `pilot/` 子目录。
