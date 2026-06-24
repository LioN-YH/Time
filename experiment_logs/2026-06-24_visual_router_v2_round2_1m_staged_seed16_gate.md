# Visual Router V2 Round2 1M staged seed16 gate

日志日期：2026-06-24 05:36:54 CST

## 目的

完成 Visual Router V2 Round2 fullscale 主线前的 1M staged seed16 gate，验证多 shard feature pipeline、subset SQLite prediction lookup、fixed FiLM 训练评估和 report schema；若 gate 无明显失败，作为启动 fullscale `spatial_panel_3view + film_mean_patch_aux` seed16 的依据。

## 背景

此前 one-shard staged validation 已完成，覆盖四个 staged sample_set 各 512 条、两个 layout、seed16、`film_mean_patch_aux`，并确认不加载 116M prediction manifest 到内存、不保存 pseudo image tensor、test 不用于训练或选择。本轮扩大到约 1M staged total，用于确认 `spatial_panel_3view` 相对 `current_rgb_3view` 不发生灾难性退化。

## 操作

1. 扩展 `build_visual_router_v2_round2_staged_samples.py`，新增 `--sample-scale one_million`，四个集合各 262144 条，总计 1048576 个 unique sample_key；`staged_train/staged_selection/staged_diagnostic` 来自 vali split 且互不重叠，`staged_test` 来自 test split。
2. 扩展 `launch_visual_router_v2_round2_staged_validation_parallel.py`，允许 `one_million` gate，并新增 `--feature-by-sample-set`，按 layout × sample_set 启动 feature worker。
3. 扩展 `summarize_visual_router_v2_round2_staged_validation.py`，允许 `one_million`，并在 metadata/summary 中标记 `is_1m_staged_gate=true`。
4. 使用 quito 环境执行 1M gate：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python visual_router_experiments/stage1_vali_test_router/launch_visual_router_v2_round2_staged_validation_parallel.py --sample-scale one_million --sample-manifest /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_1m_staged_seed16_gate/inputs/round2_staged_one_million_sample_manifest.csv --output-dir /data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_1m_staged_seed16_gate --summary-copy-dir /home/shiyuhong/Time-visual-router-v2/experiment_summaries/visual_router_v2_round2/1m_staged_seed16_gate --layouts spatial_panel_3view,current_rgb_3view --devices cuda:0,cuda:1,cuda:2,cuda:3 --seeds 16 --epochs 1 --feature-shard-size 2048 --embedding-batch-size 16 --batch-size 256 --eval-batch-size 512 --feature-by-sample-set --poll-seconds 20
   ```

5. feature 阶段完成后，subset SQLite prediction index 扫描 full-scale merged manifest，生成 5242880 条记录。
6. training aggregation 后 summary 首次因 `--sample-scale one_million` 未加入 summary CLI choices 失败；修复后只重跑 summary，不重算 feature 或训练。

## 结果

- 输出目录：`/data2/syh/Time/run_outputs/2026-06-24_visual_router_v2_round2_1m_staged_seed16_gate/`
- 轻量 summary：`experiment_summaries/visual_router_v2_round2/1m_staged_seed16_gate/`
- sample manifest 校验通过：四个 sample_set 各 262144 条，总计 1048576 unique sample_key。
- feature manifest 通过：2 layouts × 4 sample_set × 128 shards = 1024 行，所有 shard 存在；未保存 pseudo image tensor。
- prediction lookup 通过：SQLite record_count=5242880，expected_records=5242880，五专家各 1048576 条；未把 116M manifest 加载为 Python dict。
- seed16 fixed FiLM train/eval 完成，overall、strata、tail、router behavior、metadata 和中文 summary 已生成。

关键 staged_selection / staged_test 指标：

| sample_set | layout | raw-soft MAE | raw-soft MSE | raw-soft regret | entropy | mean max weight |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| staged_selection | spatial_panel_3view | 0.299858 | 1.186940 | 0.033815 | 1.145170 | 0.502931 |
| staged_selection | current_rgb_3view | 0.305033 | 1.441178 | 0.038990 | 1.142180 | 0.499217 |
| staged_test | spatial_panel_3view | 0.412812 | 175.720043 | 0.071694 | 1.135739 | 0.505176 |
| staged_test | current_rgb_3view | 0.420850 | 175.642701 | 0.079732 | 1.124794 | 0.512685 |

selected_model ratio 未出现单专家极端塌缩。`spatial_panel_3view` 在 staged_selection 与 staged_test 的 raw-soft MAE/regret 均优于 `current_rgb_3view`；staged_test MSE 略高于 current_rgb，但差值很小，未构成灾难性退化。

## 结论

1M staged seed16 gate 通过，可以进入 `spatial_panel_3view + film_mean_patch_aux` fullscale seed16 mainline。需要注意：当前已有 Stage 1 fullscale streaming 入口只支持旧 `variant_a_3view/variant_b_top3fold` 和纯 visual MLP，不支持 Round2 `spatial_panel_3view + film_mean_patch_aux`；不能用旧入口冒充本轮 mainline。

## 下一步方案

1. 新增或扩展 fullscale streaming FiLM 入口，必须支持 Round2 layout registry、mean_patch ViT pooling、RevIN aux FiLM、batch runtime pseudo image/embedding、不保存大规模 pseudo image tensor、不保存长期 embedding cache。
2. 用该入口后台启动 `spatial_panel_3view + film_mean_patch_aux` fullscale seed16，并记录 PID、status、log、resume command。
3. fullscale 完成后输出 overall/strata/tail/router behavior/metadata/中文 summary，并与 TimeFuse fullscale 做 single-seed first pass MAE/MSE 对比。
