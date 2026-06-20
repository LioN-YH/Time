# Stage 1 Visual Router V2 固定 Pilot 样本集构建

日志日期：2026-06-20 14:32:42 CST

## 目的

为 Visual Router V2 小规模架构诊断实验冻结一版可复用、可复现、边界清晰的 ordered sample keys，后续 Round 0 / Round 1 / 后续消融共用同一批样本，避免不同架构实验因样本不同而不可比。

## 背景

`96_48_S` full-scale Visual Router 1 epoch 在 MAE、oracle accuracy、regret 和权重集中度上落后于 TimeFuse-style baseline，但 MSE 更低。后续需要在小规模固定样本上逐层诊断 RevIN aux、pooling、异质 view、period 连续性和 imageization 信息密度。本步只做 sample set builder、输出样本集和文档，不训练模型、不修改 Visual Router 架构、不运行 full-scale。

可复用输入已经存在：

```text
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/oracle_labels_full_scale_2026-06-16/window_oracle_labels.parquet
/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/tsf_enrichment_full_scale_2026-06-16/sample_tsf_enrichment.parquet
```

## 操作

1. 阅读任务说明，确认本步范围为固定 pilot sample set builder 与文档，不训练、不改模型、不读 full-scale 116M prediction manifest。
2. 复核现有 Stage 1 协议、README、`WORKSPACE_STRUCTURE.md`、full-scale oracle/TSF 输出路径和 parquet schema。
3. 新增脚本：

   ```text
   visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_pilot_samples.py
   ```

4. 脚本实现要点：
   - 只读取 oracle labels parquet 的 `metric=mae` 行和 TSF enrichment parquet；
   - 用 `seed + sample_key` 的稳定哈希抽样，避免依赖 parquet 物理读取顺序；
   - 从 `vali` 中生成 `pilot_train` 和 `pilot_selection`，二者不重叠；
   - 从 `test` 中生成 `pilot_test`；
   - 从 `vali` 中排除主样本后，按 `oracle_model` 分桶生成近似均衡的 `diagnostic_balanced`；
   - 计算 oracle 第一名与第二名误差差距 `error_gap`，并写出 `error_gap_quantile`；
   - 输出 `sample_set_metadata.json`、`coverage_summary.csv` 和 `validation_summary.json`。
5. 先运行小规模 smoke：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_pilot_samples.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples_smoke \
     --pilot-train-size 20 \
     --pilot-selection-size 10 \
     --pilot-test-size 12 \
     --diagnostic-balanced-size 10 \
     --gap-quantile-reservoir-size 1000 \
     --batch-size 500000
   ```

6. smoke 通过后运行正式 v1：

   ```bash
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python \
     visual_router_experiments/stage1_vali_test_router/build_visual_router_v2_pilot_samples.py \
     --output-dir /data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples \
     --pilot-train-size 150000 \
     --pilot-selection-size 30000 \
     --pilot-test-size 75000 \
     --diagnostic-balanced-size 20000 \
     --gap-quantile-reservoir-size 1000000 \
     --batch-size 500000
   ```

7. 首次正式输出检查发现 `error_gap_quantile` 中零 gap 样本因 `np.searchsorted(..., side="right")` 被归到 `q2`，导致 `q1` 为空。修正为 `side="left"` 后重跑正式 v1，覆盖同一 canonical 输出目录。
8. 使用独立 Python 检查输出 CSV、metadata、validation 和 coverage，不依赖脚本自报。
9. 同步更新：
   - `visual_router_experiments/stage1_vali_test_router/visual_router_v2_pilot_protocol.md`
   - `visual_router_experiments/stage1_vali_test_router/README.md`
   - `WORKSPACE_STRUCTURE.md`
   - `experiment_logs/README.md`

## 结果

正式输出目录：

```text
/data2/syh/Time/run_outputs/2026-06-20_visual_router_v2_pilot_samples/
```

生成文件：

```text
pilot_train_sample_keys.csv
pilot_selection_sample_keys.csv
pilot_test_sample_keys.csv
diagnostic_balanced_sample_keys.csv
sample_set_metadata.json
coverage_summary.csv
validation_summary.json
```

最终样本数与边界检查：

| 样本集 | 行数 | split | 重复 sample_key | 说明 |
| --- | ---: | --- | ---: | --- |
| `pilot_train` | 150,000 | `vali` | 0 | 自然分布主训练样本 |
| `pilot_selection` | 30,000 | `vali` | 0 | 独立 selection 样本 |
| `pilot_test` | 75,000 | `test` | 0 | 冻结方案后的 test pilot |
| `diagnostic_balanced` | 20,000 | `vali` | 0 | 五个 oracle expert 各 4,000 个 |

独立验证结果：

```text
validation_summary.status = passed
pilot_train ∩ pilot_selection = 0
cross-set duplicate sample_key = 0
pilot_train split = vali
pilot_selection split = vali
pilot_test split = test
diagnostic_balanced split = vali
coverage_summary fields = split, dataset_name, oracle_model, error_gap_quantile, TSF cell fields
```

metadata 记录：

```text
seed = 20260620
read_full_prediction_manifest = false
read_future_y_as_feature = false
started_training = false
diagnostic_set_replaces_main_metric = false
```

脚本扫描到的 full-scale oracle `metric=mae` 行数为 `23,275,170`，其中 `vali=9,350,520`、`test=13,924,650`，与既有 full-scale 口径一致。

## 结论

Visual Router V2 pilot sample set v1 已冻结，可作为后续 Round 0 / Round 1 的共同样本基准。四份 CSV 均包含 ordered sample keys、oracle expert、error gap quantile 和 TSF cell 元信息；`diagnostic_balanced` 只用于诊断，不替代 `pilot_train` / `pilot_selection` / `pilot_test` 的自然分布主指标。

本步没有读取 116M 行 merged prediction manifest，没有读取 future `y` 作为可部署 feature，没有启动任何训练，也没有修改旧 Visual Router / TimeFuse-style 正式入口。

## 下一步方案

1. Round 0 使用这批 ordered sample keys 复现旧 Visual Router、TimeFuse-style、global best single 和 oracle 的小规模相对趋势。
2. Round 1 在同一 sample keys 上实现和比较 CLS、mean patch、CLS+mean、RevIN aux-only 和 visual+aux。
3. 所有 scaler 只在 `pilot_train` fit，架构选择只使用 `pilot_selection`；`pilot_test` 仅在方案冻结后评估。
4. 若需要更换样本集，必须新建版本化输出目录并记录与 v1 的差异，不能静默覆盖本次 v1。
