# Stage 1 P8b TimeFuse evaluation adapter 一致性校验

日志日期：2026-06-20 01:26:45 CST

## 目的

在 `visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py` 的 `evaluate_streaming(...)` 中加入最小 `EvaluationInputAdapter` 旁路一致性校验，验证正式 TimeFuse streaming evaluation 的手写 hard/raw-soft 指标与 canonical evaluation adapter 的内存复算结果一致。

## 背景

P8a 已审计正式 TimeFuse 入口的 adapter 插入点，结论是在 torch fusor 产出 `weights_np` 后，以当前 batch 已有的 `sample_keys`、`MODEL_COLUMNS`、`batch.y_pred`、`batch.y_true` 和 `weights_np` 构造 `EvaluationInput` 并调用 `EvaluationInputAdapter.evaluate_input(...)`。本步骤只做 evaluation 阶段内存旁路校验，不替换 reader、scaler、optimizer、loss、epoch loop 或 TimeFuse torch head，也不改变正式 CSV、summary、checkpoint、status、metadata 的顶层输出结构。

## 操作

1. 读取目标文件 `/home/shiyuhong/.codex-tianyu/attachments/aea8ef4e-9b95-4d1d-a8c2-1e12ebbbb167/pasted-text-1.txt`，确认 P8b 目标、边界和验收命令。
2. 检查当前分支为 `refactor/stage1-route-audit`，开始实现前工作树无未提交改动。
3. 在 `train_timefuse_fusor_streaming.py` 中：
   - 导入 `time_router.evaluation.EvaluationInputAdapter` 和 `time_router.protocols.EvaluationInput`；
   - 新增 CLI flag `--verify-evaluation-adapter`，默认关闭；
   - 新增 `verify_evaluation_adapter_batch(...)`，只消费当前 eval batch 的内存数组和 fusor 权重，不做文件 IO；
   - 在 `evaluate_streaming(...)` 中 torch fusor 输出 `weights_np` 并计算 `selected_indices`、`entropy`、`max_weight` 后，在 flag 开启时调用旁路校验；
   - 校验逐样本 `selected_index`、hard MAE/MSE、raw soft MAE/MSE、`max_weight` 和 `weight_entropy`，失败时抛出包含 `shard`、`batch`、`row`、`sample_key`、`config_name` 和指标名的错误。
4. 运行目标验收命令：
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_protocol_chain_smoke.py`
   - `/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke visual_router_experiments/stage1_vali_test_router/train_timefuse_fusor_streaming.py`
5. 额外运行一个纯内存 synthetic batch，直接调用 `verify_evaluation_adapter_batch(...)`，验证新增 helper 自身可通过一致性比较；该验证未访问 `/data2`，未生成正式输出。

## 结果

1. `stage1_timefuse_protocol_chain_smoke.py` 通过，输出确认：
   - provider/head/evaluator 阶段未调用文件 IO、`np.load` 或 `np.save`；
   - 链路输出保序且 deterministic；
   - `hard_mae=1.093573928`，`raw_soft_mae=0.556751269`。
2. `compileall` 通过，`time_router`、`tests/smoke` 和 `train_timefuse_fusor_streaming.py` 均可编译。
3. 纯内存 synthetic 校验通过，输出 `synthetic EvaluationInputAdapter consistency check passed`。
4. 本次未新增 Bash/scripts，未访问 `/data2` 做新验证，未修改 reader、scaler、optimizer、loss、epoch loop、TimeFuse torch head 或 Visual Router 入口。

## 结论

P8b 最小旁路一致性校验已接入正式 TimeFuse streaming evaluation 入口。默认运行不启用新校验，因此既有正式 CSV 字段、字段顺序、summary 口径和写出流程保持不变；pressure/smoke 场景可通过 `--verify-evaluation-adapter` 打开内存一致性检查，用 canonical `EvaluationInputAdapter` 复算并定位任何 batch/sample 级差异。

## 下一步方案

1. 小步提交本次 P8b 改动并推送到远程 `refactor/stage1-route-audit` 分支。
2. 后续如要在 pressure/smoke 中启用该 flag，应使用小规模或已有 fixture 场景先跑，不直接启动新的 `/data2` full-scale 验证。
3. 若未来正式迁移 evaluator，应继续保持 CSV/summary writer 与内存 evaluation API 的职责分离，先用该旁路校验积累一致性证据。
