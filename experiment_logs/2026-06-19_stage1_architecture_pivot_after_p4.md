# Stage 1 P4 后架构转向决策

日志日期：2026-06-19 20:41:34 CST

## 目的

在 P4d run artifacts 边界复核和 P4e checkpoint index 边界复核之后，切换到“收束新架构主干”的视角，明确 Stage 1 后续 canonical entrypoint、历史入口归档策略、可继承 schema、可舍弃 schema、runtime 最小契约和 helper 接入方式。

## 背景

P4a/P4b/P4c 已抽取低风险 JSON/path/run metadata helper，P4d/P4e 已确认 run artifacts、checkpoint/resume、launcher/monitor/status/metadata 都不能被最小 IO helper 直接替代。继续推进 P4f config system 会过早耦合历史入口和旧 schema，因此本轮只做架构决策文档，不改训练代码、不迁移入口。

## 操作

1. 只读复核了 `docs/refactor/stage1_refactor_roadmap.md`、`docs/refactor/stage1_target_architecture.md`、`docs/refactor/run_artifacts_boundary.md`、`docs/refactor/checkpoint_index_boundary.md`、Stage 1 README、visual router mainline 文档和 TimeFuse fusor streaming reader 设计文档。
2. 新增 `docs/refactor/stage1_architecture_pivot_after_p4.md`。
3. 更新 `docs/refactor/stage1_refactor_roadmap.md`，新增 P4f paused / architecture pivot 记录，并把后续方向调整为 P5 canonical entrypoint design / FeatureProvider interface design。
4. 更新 `docs/refactor/stage1_target_architecture.md`，补充 P4 后正式保留两条 streaming 主干、历史入口不再作为兼容目标、canonical runtime 最小契约和 P4 helper 接入边界。
5. 更新 `WORKSPACE_STRUCTURE.md`，登记新增 refactor 文档。
6. 使用 `quito` conda 环境运行指定 smoke 和 compileall。

## 结果

架构决策结论：

- Visual Router 正式 canonical entrypoint 保留 `train_visual_router_online_streaming.py`，路线固定为 `x -> pseudo image -> frozen ViT -> router`。
- TimeFuse-style fusor baseline 正式 canonical entrypoint 保留 `train_timefuse_fusor_streaming.py` 和 `launch_timefuse_fusor_full_scale.py`，路线固定为 `sample_key -> 17维 TimeFuse feature cache -> Linear-softmax fusor`。
- LogisticRegression fusor、offline ViT embedding cache、旧 OOM lookup、pilot-only 脚本和非 streaming full-scale 入口标记为 archive/deprecated/reference-only，不再为其新增兼容 helper。
- 继承 `sample_key`、固定五专家顺序、`packed_npy_v1` row index、oracle/TSF join 口径和 evaluation 复算口径。
- 不再强兼容旧 pilot/status/metadata/checkpoint schema，尤其不把旧 OOM 残留状态、pilot launcher 字段或 prediction cache builder 的 `--resume` 解释为 router/fusor runtime schema。
- 新 canonical runtime 最小契约为 `run_dir`、`status.json`、`metadata.json`、`checkpoints/`、`predictions/` 或 evaluation outputs、`logs/` 或 `main.log`。
- 暂停 P4f config system，转向 P5 canonical entrypoint design / FeatureProvider interface design。

验收命令均通过：

```bash
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_golden_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_oracle_tsf_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_json_utils_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_path_resolver_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_run_metadata_smoke.py
/home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall time_router tests/smoke
```

## 结论

Stage 1 后续不应继续为所有历史入口做兼容层。新架构应先收束 canonical entrypoint 和 runtime 契约，再进入 P5 FeatureProvider 设计；config system、checkpoint index helper 和 logging framework 都应推迟到 canonical runtime 边界清楚之后。

## 下一步方案

1. 进入 P5a canonical runtime contract + entrypoint design，只写设计，不迁移代码。
2. 设计 VisualFeatureProvider 和 TimeFuseFeatureProvider 的统一输入输出、dtype/device metadata、feature schema 和诊断字段。
3. 在 P6 前再决定共享 config、checkpoint index helper 和 logging framework 的最小实现范围。
