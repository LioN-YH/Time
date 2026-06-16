# Stage 1 TimeFuse-style Fusor Baseline Full-Scale 就绪性复盘

日志日期：2026-06-15 23:56:52 CST

## 目的

复盘当前 TimeFuse-style fusor baseline 是否已经可以直接进入 `96_48_S` full-scale 正式运行，并明确与另一个窗口正在推进的视觉路由 full-scale 实验之间还需要补齐哪些前置输入。

## 背景

此前已经完成 TimeFuse-style fusor baseline 的实现、pilot 验证和续接复核。该 baseline 的目的不是替代视觉路由，而是为后续 `96_48_S` full-scale 视觉路由结果提供公平对照。因此本次只做只读审计和执行路径判断，不修改训练逻辑。

## 操作

1. 阅读以下三份历史日志，确认 TimeFuse-style fusor baseline 的实现口径和 pilot 结果：
   - `experiment_logs/2026-06-15_stage1_timefuse_metadata_baseline_audit.md`
   - `experiment_logs/2026-06-15_stage1_timefuse_style_fusor_baseline.md`
   - `experiment_logs/2026-06-15_stage1_timefuse_style_fusor_baseline_followup_validation.md`
2. 阅读 `visual_router_experiments/stage1_vali_test_router/evaluate_router_baselines.py` 和 `fusion_utils.py`，确认统一 evaluator 已支持 `--timefuse-fusor on/auto/off`，并按 `config_name` 独立训练单层 `nn.Linear + softmax` fusor。
3. 阅读 `pilot/build_structure_feature_cache_pilot.py`，确认 TimeFuse-style fusor 仍需要同一批 `sample_key` 的 `feature_cache.csv`，且该 cache 由历史窗口 `x` 提取 17 维 TimeFuse-derived 单变量元特征。
4. 阅读 `pilot/compute_window_oracle_from_cache.py` 和 `pilot/enrich_cache_with_tsf_cell.py`，确认 full-scale merged cache 完成后可以直接生成 `window_oracle_labels.csv` 和 `window_oracle_labels_with_tsf_cell.csv`。
5. 检查 `/data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/` 下的 full-scale 输出状态，重点查看：
   - `sample_manifest_full_scale/sampling_metadata.json`
   - `prediction_cache_full_scale_launcher/merged_cache/status.json`
   - `prediction_cache_full_scale_launcher/merge_command_run_status.json`
   - `prediction_cache_full_scale_launcher/merge_command_2026-06-15_234723.log`
6. 粗略估算 full-scale fusor 当前实现的一次性内存需求，重点关注 `load_prediction_tensors_for_samples()` 会把全部 vali 五专家预测数组堆叠进内存这一实现特征。

## 结果

1. TimeFuse-style fusor baseline 的实现和 pilot 口径已经成立：
   - 使用 `StandardScaler`，且只在 vali feature 上 fit；
   - 使用单层 `nn.Linear(input_dim, num_experts)` 和 softmax 权重；
   - 用五专家加权预测对 `y_true` 的 `SmoothL1Loss(beta=0.01)` 训练；
   - 输出 hard top-1、raw soft fusion、selected count、summary 和 comparison。
2. 当前 full-scale prediction cache 还不能作为正式 baseline 输入直接使用：
   - `merged_cache/status.json` 仍显示 `status=running`；
   - `merge_command_run_status.json` 显示 merge 进程 PID/PGID 为 `675597`，仍在执行；
   - merge 日志截至检查时已写到 sample shard `0007`，即 `progress=8/64`，不是完整 merged cache。
3. 当前 full-scale 目录尚未看到正式的：
   - `window_oracle_labels.csv`
   - `window_oracle_labels_with_tsf_cell.csv`
   - full-scale `feature_cache.csv`
   因此 `evaluate_router_baselines.py --timefuse-fusor on` 的核心输入尚未齐全。
4. `96_48_S` full-scale 样本规模为 23,275,170 个 sample_key，其中 vali 为 9,350,520，test 为 13,924,650。按 float32 粗略估算：
   - vali 五专家 `y_pred` 约 8.36 GiB；
   - vali `y_true` 约 1.67 GiB；
   - vali 17 维 feature 约 0.59 GiB；
   - test 对应数组规模更大。
   当前 fusor 训练实现会在单 config 内一次性读取/堆叠 vali prediction tensors；理论上未必超过服务器内存，但正式 full-scale 前应先做内存压力验证，或者改成 streaming / memmap 训练与评估，避免在 5.7G manifest 和 packed arrays 上反复逐样本打开造成不可控耗时。

## 结论

不能立即进入正式 `96_48_S` full-scale TimeFuse-style fusor baseline。原因不是 fusor pilot 实现缺失，而是 full-scale 前置产物还没全部就绪：merged prediction cache 仍在运行，oracle/TSF labels 和 full-scale TimeFuse feature cache 还没有生成。

更准确的状态是：TimeFuse-style fusor baseline 的代码入口已经可用于 full-scale，但 full-scale 输入链路还需要补齐，并且当前实现建议先做一次 full-scale 规模适配检查。

## 下一步方案

1. 先等待 `prediction_cache_full_scale_launcher/merged_cache/status.json` 变为 `completed`，并确认 `metadata.json` 中 `sample_count=23275170`、`record_count=116375850`。
2. 对 completed merged cache 做完整性校验，至少确认：
   - `sample_key + model_name` 唯一；
   - 每个 sample_key 五专家完整；
   - 共享 `y_true` row index 和 packed path 可读；
   - `array_storage=packed_npy_v1`。
3. 在 completed merged cache 上运行：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/pilot/compute_window_oracle_from_cache.py \
  --cache-dir /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache

/home/shiyuhong/application/miniconda3/envs/quito/bin/python \
  visual_router_experiments/stage1_vali_test_router/pilot/enrich_cache_with_tsf_cell.py \
  --cache-dir /data2/syh/Time/run_outputs/2026-06-15_stage1_96_48_s_full_scale/prediction_cache_full_scale_launcher/merged_cache
```

4. 为 full-scale `window_oracle_labels_with_tsf_cell.csv` 生成对应的 TimeFuse structure feature cache。由于该步骤要遍历 2,327 万个窗口，建议用后台 launcher 或 tmux 运行，并记录 PID、主日志、输出目录和恢复方式。
5. 在正式跑 fusor 前，先用 full-scale cache 的小切片或前几个 shard 做内存/耗时压力测试。若当前 `fusion_utils.py` 的一次性堆叠方式成本过高，应先实现 streaming / memmap 版 TimeFuse fusor baseline，再跑正式 full-scale。
6. 前置输入和压力测试通过后，再运行 `evaluate_router_baselines.py --timefuse-fusor on`，输出统计 baseline、TimeFuse-style hard top-1、raw soft fusion 和 oracle comparison，用于后续与视觉路由 full-scale 同表比较。
