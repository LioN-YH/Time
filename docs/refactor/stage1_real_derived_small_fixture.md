# Stage 1 P13b Real-Derived Small Fixture Smoke

创建日期：2026-06-20

## 1. 目标

P13b 在 P13a mapping audit 基础上，新增一个 real-derived / schema-style small fixture，
用 P12b small canonical entrypoint 验证 manifest 保序、feature/expert join 和 canonical
`run_dir` 写出。本阶段仍不是正式入口迁移，不访问 `/data2`，不启动训练、pressure 或
full-scale。

## 2. Fixture 位置

```text
tests/fixtures/stage1_real_derived_small/
├── README.md
├── sample_manifest.csv
├── features.csv
└── expert_predictions.json
```

smoke：

```text
tests/smoke/stage1_real_derived_small_fixture_smoke.py
```

## 3. 来源与派生口径

`sample_manifest.csv` 的 4 个样本身份字段派生自 P10f Visual labels adapter smoke 和 P10g
TimeFuse feature/oracle adapter smoke 中的 ETTh1 / ETTm2 / weather 小型 fixture。manifest
只保存 P11b 最小字段：

```text
sample_key, split, config_name, dataset_name, item_id, channel_id, window_index, seq_len, pred_len
```

oracle label、oracle value、per-model error、feature 值、prediction cache 路径、SQLite path 和
checkpoint path 都不进入 manifest。

`features.csv` 使用 P12b small entrypoint 当前固定支持的三列
`trend_strength`、`seasonality_strength`、`recent_volatility`。这不是 TimeFuse 17 维
full-scale feature cache；P13b 只验证真实字段风格 sample identity 与 feature provider join。

`expert_predictions.json` 继续使用 P12b 小数组格式，包含 `model_columns`、`sample_key`、
`y_true` 和 `y_pred`。该 JSON 只是 small smoke fixture，不是正式 prediction cache schema，也
不替代 packed npy、SQLite backend 或 `PredictionCacheExpertProvider`。

## 4. Smoke 覆盖

`tests/smoke/stage1_real_derived_small_fixture_smoke.py` 会确认：

- fixture 文件存在且不位于 `/data2`；
- manifest 字段正好是 P11b 最小字段；
- feature/expert 的 sample_key 集合与 manifest 一致，且顺序刻意不同于 manifest；
- subprocess 调用 `scripts/run_stage1_canonical_small.py` 成功；
- canonical `run_dir` 写出 metadata、status、inputs、evaluation 和 predictions artifact；
- `prediction_rows.csv` sample_key 顺序与 manifest 行顺序一致；
- `run_metadata.inputs` 记录三个显式输入文件来源；
- `evaluation_summary.sample_count == 4`，`split_summary` 为 `{"vali": 2, "test": 2}`；
- stdout/stderr 不出现 `/data2`。

运行命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_real_derived_small_fixture_smoke.py
```

## 5. 明确不做

- 不修改 `train_visual_router_online_streaming.py`。
- 不修改 `train_timefuse_fusor_streaming.py`。
- 不修改 `launch_timefuse_fusor_full_scale.py`。
- 不新增 Bash launcher 或 `exp_scripts`。
- 不访问 `/data2`。
- 不启动训练、pressure 或 full-scale。
- 不改正式 CSV / summary / metadata / status / checkpoint schema。
- 不改 loss、optimizer、scaler 或 checkpoint/resume。
- 不实现正式 `SupervisionProvider`。
- 不抽 Visual online ViT `FeatureProvider`。
- 不抽 Visual `RouterHead` adapter。
- 不接 `PredictionCacheExpertProvider` 到正式入口。
- 不替换 Visual `SQLitePredictionIndex`。
- 不引入复杂 config/runtime framework。
- 不声称正式入口已迁移。

## 6. 后续连接

P13b 证明 real-derived / schema-style 小 fixture 可以复用 P12b entrypoint 完成保序 join 和
canonical artifact 写出。下一步若继续推进，应优先设计 P13c：在不迁移正式入口的前提下，审计
或旁路验证真实 small batch 的 prediction backend / feature provider 接口连接点，仍保持正式入口
schema 和 full-scale 行为不变。

P13c 已新增 `docs/refactor/stage1_real_small_backend_provider_connection_audit.md`，结论是：

- `expert_predictions.json` 后续应由 prediction backend / `ExpertProvider` / `ExpertBatch` 替换，
  但本轮不接正式入口；
- 三列 `features.csv` 后续应分别由 TimeFuse 17 维 `FeatureProvider` 或 Visual history window /
  pseudo image / ViT `FeatureProvider` 替换；
- `scripts/run_stage1_canonical_small.py` 继续保持 generic thin CLI，branch-specific feature 或 head
  验证应另走 branch-specific smoke / small entrypoint；
- 下一步优先做 P13d prediction backend -> `ExpertBatch` small smoke 和 P13e TimeFuse 17 维
  feature provider small smoke。
