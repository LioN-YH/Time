# 废弃 tuning 半成品和误导性文档记录清理

日志日期：2026-06-11 23:38:35 CST

## 目的

检查当前工作区是否还存在已经中止、被后续方案取代或容易被误读为仍在进行中的半成品输出和文档记录，并按用户要求先罗列候选项后进行精确删除。

## 背景

此前 PatchTST / CrossFormer tuning 曾经历 4 卡顺序运行、单卡并发粗搜和后续 default baseline 路线切换。当前正式可引用结果已经来自 `quito/outputs/default_baseline/`、`quito/outputs/default_baseline_mse_best/`、`quito/outputs/statistical_baseline/` 和五模型三配置汇总目录；早期 tuning 启动输出已不再作为有效结果来源。

## 操作

1. 检查 `experiment_logs/run_outputs/` 下的 `status.json`，确认旧 tuning 目录仍停留在 `planned`、`running` 或 `pending` 状态。
2. 用 `ps` 检查 `quito`、`ray`、`torchrun`、`PatchTST`、`CrossFormer`、`DLinear`、`statistical` 等关键字，未发现相关活跃训练或评估进程。
3. 检查 `quito/outputs/patchtst/`，确认只剩 `96_48_S` 空目录，没有可复核模型文件或评估结果。
4. 检查 `experiment_logs/README.md` 和 `WORKSPACE_STRUCTURE.md`，发现 README 仍有一条旧 tuning 记录标为“进行中”，结构文档仍记录已废弃的 `quito/outputs/patchtst/`。
5. 检查 MSE-best 复盘运行目录，确认 `2026-06-10_110642_042704_default_baseline_mse_best_rescore/` 是正式日志引用的有效目录；`105044_301865` 为空目录，`110504_759439` 和 `110623_871371` 是早于正式目录的重复中间产物。
6. 删除明确废弃的半成品和误导性文档记录：
   - `experiment_logs/run_outputs/2026-06-10_031235_patchtst_crossformer_4gpu_tuning/`
   - `experiment_logs/run_outputs/2026-06-10_031326_patchtst_crossformer_4gpu_tuning/`
   - `experiment_logs/run_outputs/2026-06-10_031622_patchtst_crossformer_4gpu_tuning/`
   - `experiment_logs/run_outputs/2026-06-10_105044_301865_default_baseline_mse_best_rescore/`
   - `experiment_logs/run_outputs/2026-06-10_110504_759439_default_baseline_mse_best_rescore/`
   - `experiment_logs/run_outputs/2026-06-10_110623_871371_default_baseline_mse_best_rescore/`
   - `experiment_logs/run_outputs/nohup_survival_test_2026-06-10_031600.log`
   - `experiment_logs/run_outputs/patchtst_crossformer_tuning_nohup_2026-06-10_031519.log`
   - `experiment_logs/run_outputs/patchtst_crossformer_tuning_setsid_2026-06-10_031622.log`
   - `quito/outputs/patchtst/`
   - `experiment_logs/2026-06-10_patchtst_crossformer_tuning_background_launch.md`
   - `experiment_logs/2026-06-10_patchtst_crossformer_single_gpu_screen_launch.md`
7. 同步更新 `experiment_logs/README.md`，移除两条已删除启动日志的总览行，并新增本清理日志。
8. 同步更新 `WORKSPACE_STRUCTURE.md`，移除已经删除的 `quito/outputs/patchtst/` 结构项。
9. 在 `2026-06-10_patchtst_crossformer_tuning_script_setup.md` 中补充说明：其 dry-run 输出目录已在本次清理中删除，该日志只保留脚本验证结论。

## 结果

本次清理删除的是早期 tuning 的半成品输出、空旧输出目录、重复/空 MSE-best 复盘目录和已造成状态误导的启动日志。以下正式或仍有复核价值的产物未删除：

- `quito/outputs/smoke/`
- `quito/outputs/default_baseline/`
- `quito/outputs/default_baseline_mse_best/`
- `quito/outputs/statistical_baseline/`
- `experiment_logs/run_outputs/2026-06-10_110642_042704_default_baseline_mse_best_rescore/`
- `experiment_logs/run_outputs/2026-06-11_230450_825063_five_model_three_config_summary/`
- 已完成的 default baseline、MSE-best 复盘、统计基线和五模型汇总日志

清理后复查结果：

- 已删除路径均不存在。
- `quito/outputs/` 和 `experiment_logs/run_outputs/` 下未发现空目录。
- 未发现活跃的 Quito/Ray/Torch 训练或评估进程。
- 剩余的 `2026-06-10_172921_176962_statistical_baseline_evaluate/` 为统计基线启动前 dry-run 验证目录，保留用于复核命令生成。
- 剩余 `2026-06-10_110642_042704_default_baseline_mse_best_rescore/` 中 JSON 的 `running` 字段来自当时对源 baseline checkpoint 的状态标注，不代表当前仍有进程运行，因此保留该正式复盘目录。

## 结论

当前已清理明确废弃的 PatchTST / CrossFormer tuning 半成品和会误导后续判断的文档记录。保留的输出目录均对应 smoke 验证、正式 baseline、统计基线、MSE-best 补评估或已完成汇总，不应被本次清理误删。

## 下一步方案

后续如果再次中止实验或删除长期输出目录，需要同时检查三类位置：实际输出目录、`experiment_logs/run_outputs/` 运行目录和 `experiment_logs/README.md` / `WORKSPACE_STRUCTURE.md` 记录，避免继续保留“进行中”或已不存在路径的长期记录。
