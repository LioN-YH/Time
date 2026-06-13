# Codex VS Code 插件版中转配置同步

日志日期：2026-06-12 08:39:11 CST

## 目的

将新下载的 Codex 插件版配置为使用当前 Codex CLI 已验证的中转设置，避免插件版绕过现有 `~/.codex/config.toml` 中的 provider 配置。

## 背景

当前工作区使用 `/home/shiyuhong/application/miniconda3/envs/codex/bin/codex` 作为 Codex CLI，可执行文件版本为 `codex-cli 0.139.0`。全局 Codex CLI 配置位于 `~/.codex/config.toml`，其中 `model_providers.OpenAI.base_url` 已配置为 `https://tianyuai.lol/v1`，`wire_api` 为 `responses`。

本机新下载的插件版位于 VS Code Server 扩展目录：

```text
/home/shiyuhong/.vscode-server/extensions/openai.chatgpt-26.609.30741-linux-x64
```

检查扩展 `package.json` 后确认插件公开的配置项不包含单独的 `base_url` 或 provider 配置，但包含 `chatgpt.cliExecutable`，可指定插件使用的 Codex CLI 可执行文件。

## 操作

1. 读取 `~/.codex/config.toml`，确认当前 CLI 的中转设置为：
   - `model_provider = "OpenAI"`
   - `model = "gpt-5.5"`
   - `review_model = "gpt-5.5"`
   - `base_url = "https://tianyuai.lol/v1"`
   - `wire_api = "responses"`
2. 通过 `which codex` 和 `codex --version` 确认可复用的 CLI 路径和版本：
   - `/home/shiyuhong/application/miniconda3/envs/codex/bin/codex`
   - `codex-cli 0.139.0`
3. 检查 VS Code Server 用户配置目录，发现此前不存在 `~/.vscode-server/data/User/settings.json`。
4. 新增 `~/.vscode-server/data/User/settings.json`，写入：

```json
{
  "chatgpt.cliExecutable": "/home/shiyuhong/application/miniconda3/envs/codex/bin/codex"
}
```

5. 使用 Node.js 解析新建的 `settings.json`，确认 JSON 格式有效。
6. 再次读取 `~/.codex/config.toml`，确认 CLI 中转配置未被改动。

## 结果

VS Code 插件版已固定使用当前 Codex CLI 可执行文件。由于该 CLI 读取同一个 `~/.codex/config.toml`，插件版启动本地 Codex runtime 时应复用当前的中转地址 `https://tianyuai.lol/v1` 和 Responses API 配置。

验证结果：

- `~/.vscode-server/data/User/settings.json` 可被 Node.js 正常解析；
- `chatgpt.cliExecutable` 指向现有 CLI 路径；
- `codex --version` 返回 `codex-cli 0.139.0`；
- `~/.codex/config.toml` 中的 `base_url` 仍为 `https://tianyuai.lol/v1`。

## 结论

本次配置没有修改插件扩展源码，而是通过 VS Code 用户设置让插件版调用当前已配置中转的 Codex CLI。这种方式改动范围小，后续插件更新时也更不容易被覆盖。

## 下一步方案

在 VS Code 中重载窗口或重启 VS Code Server 后打开 Codex 插件面板，发起一次简单请求验证插件侧是否正常连通中转。如果插件仍未使用该 CLI，应检查 VS Code 扩展宿主日志中 Codex runtime 的启动命令和错误信息。
