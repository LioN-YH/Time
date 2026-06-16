# DeepSeek Sidecar 安装配置与首次使用

日志日期：2026-06-16 13:11:14 CST

## 目的

为当前 `/home/shiyuhong/Time` 工作区安装并启用 `codex-deepseek-sidecar`，让主 Codex 会话能够把边界清晰、长上下文或日志类任务分派给 DeepSeek-backed Codex sidecar。

## 背景

用户明确要求安装并使用 `https://github.com/Zedong-Liu/codex-deepseek-sidecar`，并提供 DeepSeek API key。当前服务器出网依赖 `HTTP_PROXY` / `HTTPS_PROXY` 指向本机转发代理，因此配置过程中保留这些代理环境变量，不清理或覆盖。

## 操作

1. 将 sidecar 仓库克隆到 `~/.codex/vendor_imports/codex-deepseek-sidecar` 作为 vendor 缓存，并按 skill 要求安装到 `~/.codex/skills/deepseek-codex-subagent`。
2. 为 `codex-deepseek-sidecar`、`codex-deepseek-subagent` 和 `deepseek-responses-proxy` 创建 `~/.codex/bin/` 下的符号链接，并为脚本添加可执行权限。
3. 运行 sidecar 配置脚本后发现当前 Codex CLI `0.140.0-alpha.2` 使用新版 profile 机制：`--profile ds-sidecar` 会加载 `~/.codex/ds-sidecar.config.toml`。因此移除了旧式 `[profiles.ds-sidecar]` 段落，保留 `~/.codex/config.toml` 中的 `model_providers.ds-sidecar`，并新增 `~/.codex/ds-sidecar.config.toml`。
4. 将 DeepSeek API key 写入私有文件 `~/.codex/deepseek-sidecar.key`，权限为 `0600`；日志中不记录 key 明文。
5. 使用 `setsid` 后台启动内置代理：

   ```text
   /home/shiyuhong/.codex/bin/deepseek-responses-proxy --api-key-file /home/shiyuhong/.codex/deepseek-sidecar.key --host 127.0.0.1 --port 12359
   ```

   PID 记录在 `~/.codex/deepseek-responses-proxy.pid`，日志写入 `~/.codex/deepseek-responses-proxy.log`。
6. 通过 `curl http://127.0.0.1:12359/health`、`/v1/models`、非流式 `/v1/responses` 和流式 `/v1/responses` 验证本地代理、DeepSeek key、上游网络和 SSE 转换均可用。
7. 发现当前 Codex CLI 中 `codex doctor -c profile=ds-sidecar` 仍走旧配置覆盖路径，不能代表新版 `--profile ds-sidecar` 运行时 profile。为避免 wrapper 默认预检误拦截，在已安装的 `codex-deepseek-sidecar` 和 `codex-deepseek-subagent` 脚本中增加兼容逻辑：检测到 `~/.codex/ds-sidecar.config.toml` 后跳过旧 doctor 预检。
8. 使用 sidecar wrapper 为当前仓库启动 `repo-log-audit` 任务：

   ```text
   /home/shiyuhong/.codex/bin/codex-deepseek-sidecar --cd /home/shiyuhong/Time --task-id repo-log-audit --no-monitor --no-doctor-check ...
   ```

   任务为只读仓库日志和规范梳理，不修改文件。
9. 再启动 `smoke-sidecar-default` 极小任务，不传 `--no-doctor-check`，验证修补后的默认 wrapper 可直接使用，返回 `sidecar-ok`。

## 结果

1. 内置 DeepSeek Responses proxy 已运行在 `127.0.0.1:12359`，当前 PID 为 `10538`。
2. Codex sidecar profile 已可用：`codex exec -p ds-sidecar` 能识别 `model=deepseek-v4-pro`、`provider=ds-sidecar`。
3. 首个 DeepSeek sidecar 会话已完成并进入 idle 状态：

   ```text
   task_id=repo-log-audit
   session_id=019eced5-95f0-7c33-99ee-07390f07d524
   workdir=/home/shiyuhong/Time
   status=idle
   ```

4. 默认 wrapper smoke 会话也已成功：

   ```text
   task_id=smoke-sidecar-default
   session_id=019eced9-6396-7662-ba5a-82123f1dc30a
   ```

5. sidecar 输出了中文整理报告，确认当前最适合分派的任务类型包括：后台任务状态监控、完整性校验、日志解析、oracle/baseline 结果探索性分析、小规模 smoke/pilot 验证，以及代码规范检查。
6. sidecar 只读复核时确认 Stage 1 full-scale 当前关键状态：五专家 prediction cache merge 已完成并校验通过，TimeFuse feature cache 64/64 shard completed，4 张 RTX 3090 当前空闲。

## 结论

DeepSeek sidecar 已在本机安装、配置并完成首次仓库任务。当前已通过 `~/.codex/ds-sidecar.config.toml` 适配 Codex CLI `0.140.0-alpha.2` 的新版 profile 机制，并修补 wrapper 的旧 doctor 预检路径；后续默认 `codex-deepseek-sidecar --profile ds-sidecar` 可直接使用。连通性已通过本地代理 health、真实 DeepSeek Responses 请求和 sidecar 任务共同验证。

## 下一步方案

1. 后续可用以下命令检查 sidecar 任务状态：

   ```text
   /home/shiyuhong/.codex/bin/codex-deepseek-sidecar --cd /home/shiyuhong/Time --task-id repo-log-audit --status
   ```

2. 如需继续该会话，可使用：

   ```text
   /home/shiyuhong/.codex/bin/codex-deepseek-sidecar --cd /home/shiyuhong/Time --task-id repo-log-audit --resume --no-monitor "新的任务说明"
   ```

3. 如需分派新的只读长任务，应使用新的 `--task-id`，并明确输入路径、预期输出、是否允许写文件和是否需要 `/home/shiyuhong/application/miniconda3/envs/quito/bin/python`。
4. 当前最适合继续分派给 DeepSeek sidecar 的实际任务是 TimeFuse feature cache 完整性校验：64 shard 只读扫描、总行数、`sample_key` 唯一性、17 维特征有限性和字段一致性检查。
