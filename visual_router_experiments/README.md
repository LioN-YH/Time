# Visual Router 正式实验代码目录

本目录用于保存视觉结构先验路由实验的正式代码。后续与 Visual Router / Visual-Conditioned MoE 相关的可复用代码、正式实验入口和阶段性脚本，优先放在本目录下，而不是继续堆叠到 `experiment_scripts/`。

## 目录原则

1. 按实验阶段建立二级目录，阶段内再细分数据缓存、训练、评估和分析脚本。
2. 跨阶段复用的代码放入 `common/`，避免在各 stage 内复制。
3. 阶段目录中的脚本应写清输入、输出、缓存口径和是否可作为正式结果引用。
4. 大规模输出不放在本目录，仍写入 `experiment_logs/run_outputs/` 下的时间戳目录。
5. 伪图像默认保持为张量或视觉 encoder embedding，只有人工检查或报告展示时才导出少量图片。
6. 各 stage 的探索性、小规模或半成品验证脚本放入该 stage 的 `pilot/` 子目录；正式入口和跨阶段工具分别放在 stage 根目录或 `common/`。

## 当前阶段

| 目录 | 角色 | 当前用途 |
| --- | --- | --- |
| `common/` | 跨阶段公共代码 | 后续放置 schema、路径、指标、cache 读写、伪图像张量构造和通用评估工具 |
| `stage0_oracle_audit/` | 上限审计阶段 | 记录已有 per-item oracle 审计脚本和结果索引；后续如需重跑或扩展专家池，可在此补充正式版本 |
| `stage1_vali_test_router/` | Stage 1 主实验 | vali 训练 router、test 测试 router，验证 window-level visual routing 是否有效 |
| `stage2_heldout_cell/` | Stage 2 泛化实验 | 7-cell 训练、held-out cell 测试，验证视觉结构特征的 zero-shot 泛化 |

各目录包含 `__init__.py`，可作为 Python package 直接导入。

## 输出约定

正式运行输出仍使用：

```text
experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_*/
```

代码目录只保存源码、配置模板、README 和小型 schema 文档，不保存大规模 prediction cache、embedding cache、checkpoint 或完整评估产物。
