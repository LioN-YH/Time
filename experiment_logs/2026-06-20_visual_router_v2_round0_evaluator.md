# Visual Router V2 Round 0 Evaluator 与固定样本代表性复现

日志日期：2026-06-20 17:36:02 CST

## 目的

基于 P0 已冻结的 Visual Router V2 pilot sample sets，建立 Round 0 统一 evaluator，在固定小样本上同表复现旧 Visual Router、TimeFuse-style、global_best_single 和 oracle 的相对趋势，并判断 P0 v1 样本是否足以进入 Round 1。

## 背景

P0 v1 样本目录为 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/`，包含 150,000 个 `pilot_train` vali、30,000 个 `pilot_selection` vali、75,000 个 `pilot_test` test 和 20,000 个 `diagnostic_balanced` vali。Round 0 的核心要求是只做 evaluator、baseline 对齐、诊断汇总和文档，不训练新模型、不修改 Visual Router 或 TimeFuse-style 正式入口、不覆盖 P0 或 full-scale 结果目录。

## 操作

1. 新增 `visual_router_experiments/stage1_vali_test_router/visual_router_v2_evaluator.py`，封装 P0 sample CSV 保序读取、full-scale 大 CSV 子集抽取、统一逐样本 method rows、main/selection/diagnostic summary、selected_model counts、TSF 分层 summary 和 Visual/TimeFuse paired diagnostics。
2. 新增 `visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round0.py`，作为 Round 0 CLI：
   - `pilot_test` 直接从 full-scale Visual eval-only 与 TimeFuse-style fusor 的逐样本输出 CSV 按 P0 `sample_key` 子集抽取；
   - `pilot_selection` 和 `diagnostic_balanced` 复用已训练 Visual checkpoint、TimeFuse checkpoint、P0 ordered sample keys、oracle labels、TimeFuse feature cache 和 prediction arrays 进行 eval-only forward；
   - 对 50,000 个 vali selection/diagnostic sample_key 建立 Round 0 专用 subset SQLite：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/prediction_index_round0_vali.sqlite`，只写 250,000 条 `(sample_key, model_name)` record，未全量加载 116M prediction manifest 到内存。
3. 运行命令：

   ```bash
   CUDA_VISIBLE_DEVICES=2,3 /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/evaluate_visual_router_v2_round0.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0 \
     --overwrite \
     --device cuda
   ```

4. 执行过程中修复了若干 evaluator 低效或口径问题：
   - 避免在缺失检查和 lookup 过滤中反复构造 `set(...)` 导致 75k/125k 样本级 CPU 卡顿；
   - 对 oracle labels 和 feature cache 对齐时显式指定检查列，避免要求预测表才有的 `selected_model`；
   - 为 Visual selection/diagnostic eval-only 增加中间 cache，避免失败重跑时重复 ViT forward；
   - TimeFuse selection/diagnostic raw soft fusion 后补数组级 MAE/MSE 复算，保证 selection comparison 同表指标完整；
   - Markdown 摘要改用项目已有 `frame_to_markdown`，避免依赖环境中未安装的 `tabulate`。

## 结果

输出目录：`/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/`。

已生成并复核以下必需产物：

- `round0_main_comparison.csv`
- `round0_selection_comparison.csv`
- `round0_diagnostic_balanced_summary.csv`
- `round0_selected_model_counts.csv`
- `round0_stratified_summary.csv`
- `round0_paired_diagnostics.csv`
- `round0_metadata.json`
- `round0_summary.md`

关键 `pilot_test` 结果：

| 方法 | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual hard top-1 | 75,000 | 0.664912 | 596.442288 | 0.324627 | 0.457960 |
| Visual raw soft fusion | 75,000 | 0.603009 | 510.975952 | 0.262724 | 0.457960 |
| TimeFuse hard top-1 | 75,000 | 0.547432 | 568.559825 | 0.207147 | 0.587240 |
| TimeFuse raw soft fusion | 75,000 | 0.535220 | 568.502401 | 0.194935 | 0.587240 |
| global_best_single | 75,000 | 0.599744 | NA | 0.259460 | 0.125760 |
| oracle_top1 | 75,000 | 0.340285 | NA | 0.000000 | 1.000000 |

关键 `pilot_selection` 结果：

| 方法 | sample_count | MAE | MSE | regret_to_oracle | oracle_label_accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual hard top-1 | 30,000 | 0.356267 | 1.367826 | 0.089536 | 0.579200 |
| Visual raw soft fusion | 30,000 | 0.334069 | 1.181831 | 0.067337 | 0.579200 |
| TimeFuse hard top-1 | 30,000 | 0.334912 | 1.429099 | 0.068181 | 0.585767 |
| TimeFuse raw soft fusion | 30,000 | 0.317530 | 1.370167 | 0.050799 | 0.585767 |
| global_best_single | 30,000 | 0.357820 | NA | 0.091089 | 0.309667 |
| oracle_top1 | 30,000 | 0.266731 | NA | 0.000000 | 1.000000 |

`round0_summary.md` 给出 `direction_ok=True`：P0 v1 `pilot_test` 复现了 TimeFuse hard MAE 优于 Visual hard MAE、Visual raw-soft MSE 优于 TimeFuse raw-soft MSE、Visual oracle-label accuracy 落后于 TimeFuse 的 full-scale 关键方向。

## 结论

Round 0 evaluator 已完成。P0 v1 小样本能够复现 full-scale 关键相对趋势，可作为 Round 1 架构选择和诊断参照。`diagnostic_balanced` 已单独输出诊断 summary，不作为主指标；`pilot_test` 只用于 Round 0 代表性验证，不应用于后续架构选择。

## 下一步方案

1. 进入 Round 1 前，固定引用 `/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_round0/round0_summary.md` 和 `round0_selection_comparison.csv` 作为 selection 参照。
2. Round 1 只在 `pilot_train`/`pilot_selection` 上做架构选择，保留 `pilot_test` 到 Round 0/最终代表性验证，不把 test oracle error 用作 deployable feature。
3. 若后续 evaluator 需要更频繁复跑，可把 test CSV 子集和 TimeFuse feature 子集也缓存成 Round 0 中间文件，减少重复顺序扫描。
