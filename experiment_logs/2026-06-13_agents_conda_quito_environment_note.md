# AGENTS 实验环境规范补充

日志日期：2026-06-13 23:16:38 CST

## 目的

在项目级协作规范 `AGENTS.md` 中补充说明：本工作区实验默认使用 conda 环境 `quito`，避免后续运行脚本或验证时混用系统 Python 导致依赖缺失或版本不一致。

## 背景

近期 Stage 1 伪图像化和 HF ViT normalization 验证中，系统 Python 可以做部分语法检查，但函数级 smoke 因缺少 `torch` 无法运行；Quito conda 环境可以完成相关验证。为了保证后续实验和日志复现口径一致，需要把默认实验环境写入项目级规范。

## 操作

1. 阅读根目录 `AGENTS.md` 当前内容。
2. 在“协作偏好”和“正式视觉路由实验代码目录规范”之间新增“实验环境规范”小节。
3. 明确默认实验环境为 conda `quito`。
4. 记录推荐解释器绝对路径：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python
   ```

5. 说明纯文本检查、`rg`、`ls`、`sed`、`git`、`date` 等不依赖实验 Python 包的操作可以使用系统 shell/Python。
6. 同步更新 `experiment_logs/README.md` 和 `WORKSPACE_STRUCTURE.md`。

## 结果

`AGENTS.md` 已新增“实验环境规范”，要求：

- 实验、数据处理、Quito 脚本运行和依赖 `torch`/`quito`/`omegaconf`/`sklearn` 等包的验证默认使用 conda `quito`；
- 推荐使用解释器绝对路径执行；
- 系统 Python 只用于不依赖实验包的轻量操作；
- 若系统 Python 验证因依赖缺失失败，应在 `quito` 环境下复验并记录。

## 结论

本工作区后续实验环境口径已经写入项目级规范。后续新增脚本、README、实验协议或日志中的可复现实验命令，应优先使用 `quito` conda 环境。

## 下一步方案

后续执行 Stage 1 HF ViT embedding smoke、visual router 训练、soft fusion 评估时，默认使用：

```text
/home/shiyuhong/application/miniconda3/envs/quito/bin/python
```
