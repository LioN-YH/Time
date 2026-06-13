# 根工作区 Git 仓库初始化与远程推送

日志日期：2026-06-13 10:48:16 CST

## 目的

为 `/home/shiyuhong/Time` 建立有效的根 Git 仓库，并将项目级实验代码、协议文档、轻量实验日志和结构说明推送到远程仓库 `git@github.com:LioN-YH/Time.git`。

## 背景

前序检查发现根目录下原有 `.git/` 是空目录，不是有效 Git 仓库；`quito/` 和 `TimeFuse/` 各自带独立 `.git/`，且工作区中存在大量数据、checkpoint、prediction cache、运行输出和日志文件，不能直接执行 `git add .`。本次初始化采用轻量根仓库口径：首版只跟踪本项目自有实验脚本、Visual Router 正式实验代码、协议文档、中文实验日志和结构文档，不直接 vendor `quito/` 与 `TimeFuse/`。

## 操作

1. 验证 GitHub SSH 认证，确认 `ssh -T git@github.com` 返回账号 `LioN-YH` 认证成功。
2. 删除根目录下空的异常 `.git/` 目录。
3. 执行 `git init`，并切换到 `main` 分支。
4. 新增根目录 `.gitignore`，排除：
   - `quito/`、`TimeFuse/` 两个嵌套外部仓库；
   - `.agents/`、`.codex/` 等本地 agent 状态；
   - `experiment_logs/run_outputs/`；
   - checkpoint、数组、parquet、日志、缓存和密钥环境文件。
5. 更新 `WORKSPACE_STRUCTURE.md`，记录根 `.gitignore` 和有效 `.git/` 口径。
6. 设置本仓库本地 Git 身份：
   - `user.name=LioN-YH`
   - `user.email=LioN-YH@users.noreply.github.com`
7. 检查 staged 清单，确认没有纳入大规模数据、checkpoint、`.npy` 或运行输出目录。
8. 创建首个提交：
   - commit：`cbad9d4`
   - message：`Initial visual router workspace`
9. 设置远程：
   - `origin=git@github.com:LioN-YH/Time.git`
10. 执行 `git push -u origin main`。

## 结果

远程推送成功，`main` 分支已经建立并跟踪 `origin/main`。首个远程提交包含 61 个轻量文件，主要包括：

- 根目录 `.gitignore`、`AGENTS.md`、`WORKSPACE_STRUCTURE.md`；
- `experiment_scripts/` 下已有实验编排和汇总脚本；
- `experiment_logs/` 下中文实验日志和 README 总览；
- `visual_router_experiments/` 下 Visual Router / Stage 1 cache、baseline 和协议相关代码文档。

未提交内容包括：

- `quito/` 与 `TimeFuse/` 外部代码库本体；
- `experiment_logs/run_outputs/`；
- 本地数据、checkpoint、数组缓存、parquet 和运行日志。

## 结论

根工作区已经具备可通过 SSH 推送的远程 Git 仓库。当前远程仓库适合作为项目级实验代码、协议和轻量日志的版本管理仓库；大规模实验数据与模型产物仍保留在本地或后续外部存储，不进入 Git。

## 下一步方案

1. 后续新增实验脚本、协议文档、轻量日志和结构说明时，按常规 Git 流程提交并推送。
2. 如果后续需要复现完整 Quito 本地改动，应单独决定 `quito/` 的管理方式：fork 分支、patch 文件、submodule，或将必要改动抽取到根仓库脚本中。
3. 继续 Stage 1 `96_48_S` structure feature pilot 前，保持 feature cache 和运行输出写入 `experiment_logs/run_outputs/`，不要写入代码目录。
