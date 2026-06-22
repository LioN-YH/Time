# PatchTST + Visual dual-branch 65k 实验脚手架与 smoke

日志日期：2026-06-22 22:27:21 CST

## 目的

为 Visual Router V2 探索分支新增 PatchTST frozen baseline + fixed visual embedding 双分支预测实验入口，在不改 Stage 1 canonical 主线的前提下，支持后续在同一 65k split 上比较 PatchTST 单分支与 PatchTST+Visual 双分支的 MAE/MSE。

## 背景

用户目标要求固定一种视觉编码方式，使用已有 visual embedding cache 或已确定的视觉编码输出；时序分支选择 PatchTST；在 65k 数据上训练和评估双分支预测模型，并保存 PatchTST baseline 与双分支模型的预测、指标和 summary。第一批融合变体限定为 `feature_concat`、`film`、`residual_feature`、`visual_residual`，暂缓 cross-attention，且不得重新生成图像或重新跑 ViT。

## 操作

1. 新增 `visual_router_experiments/dual_branch_fusion/` 包：
   - `cache_dataset.py`：读取单个 `.npz` 或 `.npz` shard 目录，按 `sample_key` 对齐 PatchTST cache 与 visual embedding cache，并检查 train/val/test split 非空。
   - `fusion_heads.py`：实现 `feature_concat`、`film`、`residual_feature`、`visual_residual`，并预留可选 `pred_gate`。
   - `metrics.py`：计算 MAE/MSE、`delta_mae_vs_patchtst`、`delta_mse_vs_patchtst` 和 beats 标志。
   - `train_patchtst_visual_65k.py`：读取 frozen PatchTST cache 与 fixed visual cache，评估 PatchTST baseline，只训练轻量 fusion head，写出 `config.json`、`metrics.json`、`predictions.npz`、`training_log.txt`、`summary.md`。
   - `summarize_results.py`：递归汇总多 seed / 多 fusion mode 的 `metrics.json`。
2. 新增 `tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py`，在临时目录构造 synthetic PatchTST/visual cache，逐个验证四个第一批 fusion mode 可以 forward、跑 2 个 mini epoch、写出指标和 summary，并检查预测 shape 与 `y_true` 一致。
3. 新增 `docs/experiments/patchtst_visual_dual_branch_65k.md`，记录 cache contract、单次运行命令、多 seed 汇总命令、输出文件和 smoke 命令。
4. 更新 `WORKSPACE_STRUCTURE.md`，登记 `docs/experiments/` 与 `visual_router_experiments/dual_branch_fusion/` 的职责和边界。
5. 运行验证命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/dual_branch_fusion/*.py tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/dual_branch_fusion_patchtst_65k_smoke.py
```

## 结果

- `py_compile` 通过。
- synthetic smoke 通过，四个硬性 fusion mode 均完成 forward、mini train、metrics 和 summary 写出：
  - `feature_concat`
  - `film`
  - `residual_feature`
  - `visual_residual`
- smoke 只使用临时 synthetic cache，不访问 `/data2`，不生成图像，不运行 ViT，不训练 PatchTST。
- 本次没有启动真实 65k 训练评估，因此没有产生真实 PatchTST vs dual-branch 的 65k MAE/MSE 结论。

## 结论

PatchTST + fixed visual embedding 双分支预测实验的代码、文档和 synthetic smoke 闭环已经建立。当前入口满足后续接入真实 65k PatchTST cache 与 visual embedding cache 的基本工程要求，并能在同一 `sample_key`/split 上保存 baseline 与 dual-branch 对比指标。

## 下一步方案

1. 确认真实 65k PatchTST frozen cache 路径，要求包含 `sample_key`、`split`、`h_ts`、`y_patchtst`、`y_true`。
2. 确认对应 fixed visual embedding cache 路径，要求包含同一批 `sample_key` 和 `h_vis` 或等价视觉表示字段。
3. 分别运行 `feature_concat`、`film`、`residual_feature`、`visual_residual`，输出到独立 seed/mode 目录。
4. 运行 `summarize_results.py` 汇总 PatchTST MAE/MSE、Dual-branch MAE/MSE、delta 和 beats 标志，再写入正式实验结果日志。
