# Stage 1 P21a Visual Eval Small Launcher

创建日期：2026-06-21

## 1. 目标

P21a 在 P17/P20a 已跑通的 Visual canonical eval entrypoint 外增加一层 small
launcher/config pack，降低日常复现命令长度。它只包装两个小样本 evaluation-only 路径：

- precomputed small canonical eval；
- visual-chain-dryrun small canonical eval。

该 launcher 不实现 evaluation 逻辑，实际运行仍调用
`scripts/run_stage1_visual_eval_canonical.py`。

## 2. 新增文件

- `configs/stage1/visual_eval_small_precomputed.json`
- `configs/stage1/visual_eval_small_visual_chain.json`
- `scripts/run_stage1_visual_eval_small.py`
- `tests/smoke/stage1_visual_eval_small_launcher_smoke.py`

两个 config 只引用 repo 内 small fixture，并把 `router_checkpoint_payload` 声明为
`auto-tempfile`。launcher 会在 `/tmp` 生成 tiny legacy `VisualMLPRouter` checkpoint payload，
再把临时 path 传给 canonical entrypoint。

## 3. CLI

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python scripts/run_stage1_visual_eval_small.py \
  --mode both \
  --output-dir /tmp/stage1_p21a_visual_eval_small \
  --run-id p21a_small
```

支持参数：

- `--config-json`：显式 small config；省略时按 `--mode` 使用内置 config；
- `--output-dir`：canonical run_dir 父目录；
- `--run-id`：可选 run id；`--mode both` 会追加 `precomputed` / `visual_chain` 后缀；
- `--mode {precomputed,visual-chain,both}`；
- `--python-executable`：执行 canonical entrypoint 的 Python，默认当前 Python；
- `--dry-print-command`：只打印 canonical command，不执行、不创建 run_dir；
- `--allow-real-checkpoint`：只透传给 canonical entrypoint，默认 false；
- `--allow-real-vit`：仅 config 显式 `manual_real_vit=true` 时允许，默认 false。

## 4. Safety Gate

launcher 会在调用 canonical entrypoint 前做轻量安全校验：

- `safety.allow_data2=false` 时，config 内任何 `/data2` path 都 fail-fast；
- `training_started=true` 或 `full_scale_run=true` fail-fast；
- `manual_real_vit=true` 必须 CLI 显式 `--allow-real-vit`；
- 默认 config 使用 `vit_provider_mode=injected-fake`，不加载真实 ViT，不导入 transformers；
- 默认 checkpoint 是 `/tmp` tiny payload，不读取真实 checkpoint。

## 5. 验证

新增 smoke 覆盖：

- `--mode precomputed` 成功写 canonical run_dir，metadata `feature_source=precomputed`；
- `--mode visual-chain` 成功写 canonical run_dir，metadata
  `feature_source=visual-chain-dryrun`、`visual_chain_enabled=true`、`loads_real_vit=false`；
- `--mode both` 连续生成两个不冲突 run_dir；
- `--dry-print-command` 只打印 canonical command，不创建 run_dir；
- `/data2` path、`training_started=true`、`full_scale_run=true` 负例 fail-fast。

运行命令：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_eval_small_launcher_smoke.py
```

## 6. 明确不做

- 不启动训练或 full-scale；
- 不修改 `train_visual_router_online_streaming.py`；
- 不新增 Bash launcher；
- 不自动搜索 `/data2`；
- 不默认读取真实 checkpoint、真实 ViT 或 HuggingFace 模型；
- 不修改 P20a canonical eval entrypoint 默认行为；
- 不修改 TimeFuse small entrypoint；
- 不把 config、run_dir、path 或 allow flags 下沉到 provider/head adapter。
