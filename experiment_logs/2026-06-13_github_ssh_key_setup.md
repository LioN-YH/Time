# GitHub SSH Key 配置

日志日期：2026-06-13 10:33:31 CST

## 目的

为 `/home/shiyuhong/Time` 后续推送到远程仓库 `git@github.com:LioN-YH/Time.git` 建立 SSH 认证方式，减少后续 `push`/`fetch` 时的交互式认证成本。

## 背景

用户计划为当前工作区建立远程 GitHub 仓库，并希望使用 SSH URL 作为远程地址。检查发现根目录下的 `.git/` 是空目录，当前根工作区还不是有效 Git 仓库；同时本机 `~/.ssh/` 下没有可用于 GitHub 的私钥，执行 `ssh -T git@github.com` 返回 `Permission denied (publickey)`。

## 操作

1. 检查 `~/.ssh/` 目录、全局 Git 用户配置、GitHub CLI 状态和 GitHub SSH 连通性。
2. 生成一把专用于 GitHub/Time 仓库协作的 ed25519 key：
   - 私钥路径：`~/.ssh/id_ed25519_github_time`
   - 公钥路径：`~/.ssh/id_ed25519_github_time.pub`
   - key 注释：`LioN-YH-Time`
3. 将 `~/.ssh/` 目录权限调整为 `700`。
4. 新增 `~/.ssh/config`，为 `github.com` 指定：
   - `User git`
   - `IdentityFile ~/.ssh/id_ed25519_github_time`
   - `IdentitiesOnly yes`
5. 将 `~/.ssh/config` 权限调整为 `600`。
6. 再次执行 `ssh -T git@github.com` 进行验证。

## 结果

SSH key 和本地 SSH client 配置已经完成。再次连接 GitHub 时仍返回 `Permission denied (publickey)`，这是预期状态，因为生成的公钥还没有添加到 GitHub 账号 `LioN-YH` 的 SSH keys 中。

当前需要添加到 GitHub 的公钥内容保存在：

```text
~/.ssh/id_ed25519_github_time.pub
```

## 结论

本机侧 SSH 配置已经准备好，但远程 GitHub 账号侧尚未信任该公钥，因此暂时还不能通过 SSH 推送。待用户在 GitHub 网页端添加该 public key 后，再次执行 `ssh -T git@github.com` 应能看到 GitHub 认证成功提示。

## 下一步方案

1. 用户在 GitHub `Settings -> SSH and GPG keys -> New SSH key` 中添加 `~/.ssh/id_ed25519_github_time.pub` 的内容。
2. 重新验证 SSH 认证：`ssh -T git@github.com`。
3. 为根工作区初始化有效 Git 仓库，新增根 `.gitignore`，避免把数据、checkpoint、cache、运行日志和嵌套仓库误提交。
4. 设置远程地址为 `git@github.com:LioN-YH/Time.git`，完成首个轻量 commit 后推送。
