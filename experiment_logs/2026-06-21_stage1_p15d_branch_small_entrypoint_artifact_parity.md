# Stage 1 P15d branch-specific small entrypoint artifact parity

日志日期：2026-06-21 01:38:20 CST

## 目的

新增 P15d 跨分支 small entrypoint artifact parity smoke，比较 TimeFuse-specific 和
Visual-specific small entrypoint 写出的 canonical run_dir 共同结构、共同 schema、字段命名、
ordered sample_keys 和边界约束，防止正式迁移前两条支线 artifact contract 继续分叉。

## 背景

P15b 已新增 `scripts/run_stage1_timefuse_small.py` 和对应 smoke，使用 P13b real-derived
small manifest/expert JSON 与 P13e 17 维 feature fixture，写出 TimeFuse branch canonical
run_dir。P15c 已新增 `scripts/run_stage1_visual_small.py` 和对应 smoke，使用 P14b Visual
mock history window fixture 与 script-local smoke-only MLP adapter，写出 Visual branch
canonical run_dir。

当前仍未迁移正式 Visual Router / TimeFuse fusor 训练入口，未接真实 checkpoint、scaler、
ViT provider 或 full-scale reader，也不应访问 `/data2`。因此本步只做 artifact parity
smoke 和文档同步。

## 操作

1. 读取 P15d 目标说明，确认新增范围为：
   - `tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py`
   - `docs/refactor/stage1_branch_small_entrypoint_artifact_parity.md`
   - 本实验日志
   - 同步更新 roadmap、entrypoint migration plan、工作区结构文档和实验日志 README。
2. 检查当前分支为 `refactor/stage1-route-audit`，工作区起始状态与
   `origin/refactor/stage1-route-audit` 对齐。
3. 阅读 P15b/P15c 既有 smoke 和两个 small entrypoint 的 metadata/status 写出字段，确认无需修改
   `scripts/run_stage1_canonical_small.py`、`scripts/run_stage1_timefuse_small.py` 或
   `scripts/run_stage1_visual_small.py` 行为。
4. 新增 P15d smoke：
   - 使用 tempfile 下同一个 `run_outputs/`，分别运行 `p15d_timefuse_artifact_parity` 和
     `p15d_visual_artifact_parity`。
   - 检查共同 canonical 子目录、metadata/status/inputs/evaluation/predictions/logs。
   - 检查共同 metadata、status、evaluation summary 和 prediction rows schema。
   - 检查 `sample_manifest_ref.row_count`、`split_summary.sample_count_by_split`、
     `prediction_rows.csv` 的 `sample_key` 顺序、`split` 列和 `config_name` 一致。
   - 检查 TimeFuse 和 Visual branch-specific metadata。
   - 检查 stdout/stderr 未出现 `/data2`、正式训练入口、`torch.load`、`ViTModel` 或
     `AutoImageProcessor`。
   - 检查 generic、TimeFuse、Visual small CLI 文件运行前后不变。
5. 新增 P15d 文档，说明目标、时机、共同 artifact schema、允许的 branch-specific 字段、
   不比较指标优劣、明确不做范围和后续方向。

## 结果

- 新增文件：
  - `tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py`
  - `docs/refactor/stage1_branch_small_entrypoint_artifact_parity.md`
  - `experiment_logs/2026-06-21_stage1_p15d_branch_small_entrypoint_artifact_parity.md`
- 已同步更新：
  - `docs/refactor/stage1_refactor_roadmap.md`
  - `docs/refactor/stage1_entrypoint_migration_plan.md`
  - `WORKSPACE_STRUCTURE.md`
  - `experiment_logs/README.md`
- 首次运行新增 smoke 命令：

  ```bash
  /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
  ```

- 首次运行结果：通过。stdout 显示两个 branch-specific small entrypoint subprocess 均完成，
  canonical run_dir 共同结构存在，metadata/status schema、split/input consistency、
  sample_key 顺序、evaluation summary、prediction rows 和 branch-specific metadata 均通过断言。
- 最终回归验证命令：

  ```bash
  /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m compileall tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
  /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_timefuse_small_entrypoint_smoke.py
  /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_visual_small_entrypoint_smoke.py
  /home/shiyuhong/application/miniconda3/envs/quito/bin/python tests/smoke/stage1_branch_small_entrypoint_artifact_parity_smoke.py
  git diff --check
  ```

- 最终回归结果：全部通过。
- 本步未修改 `scripts/run_stage1_canonical_small.py`、
  `scripts/run_stage1_timefuse_small.py` 或 `scripts/run_stage1_visual_small.py`。
- 本步未访问 `/data2`，未启动正式 Visual Router / TimeFuse fusor 训练入口，未读取真实
  checkpoint，未启动 ViT embedding，未启动 pressure 或 full-scale。

## 结论

P15d artifact parity smoke 已建立，并证明当前 TimeFuse-specific 与 Visual-specific small
entrypoint 写出的 canonical run_dir 在共同 schema、共同 ordered sample_keys、共同
`ExpertBatch` small fixture、evaluation summary 和 prediction rows 字段上保持一致。两条支线的
branch-specific metadata 差异被明确允许并单独断言；TimeFuse/Visual 指标数值优劣不在本步比较范围。

## 下一步方案

1. 运行新增 smoke 的 compileall。
2. 运行 P15b/P15c/P15d 三个 smoke 回归。
3. 用 `git diff` 审查未修改正式训练入口、未访问 `/data2`、未把 P15c smoke-only adapter
   提升为正式 adapter。
4. 提交并推送到 `origin/refactor/stage1-route-audit`。
