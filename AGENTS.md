# 项目级工作规范

本文档记录 `/home/shiyuhong/Time` 工作区的长期协作要求。后续在本目录内进行代码、实验、数据处理或分析工作时，应优先遵守本文档。

## 实验日志规范

1. 每完成一个相对独立的实验步骤、数据处理步骤、环境配置步骤或重要分析步骤，都需要在 `experiment_logs/` 目录下新增一份实验日志。
2. 日志文件名应包含日期和大致工作描述，推荐格式为：

   ```text
   YYYY-MM-DD_brief_work_description.md
   ```

3. 日志正文必须使用中文记录。
4. 日志正文中必须记录详细日志日期，时间精确到秒，例如：

   ```text
   日志日期：2026-06-10 02:23:38 CST
   ```

5. 单篇实验日志至少应包含以下内容：
   - 目的
   - 背景
   - 操作
   - 结果
   - 结论
   - 下一步方案

6. 每次新增实验日志后，都要同步更新 `experiment_logs/README.md` 中的总览追踪表，记录日志文件、主题、状态、关键结果和下一步。
7. 日志应记录真实做过的操作和验证结果，不应只写计划或泛泛总结。

## 中止实验和半成品清理规范

1. 在实验策略快速变化、资源调度方式调整或用户明确要求停止某个实验时，应先确认对应实验进程已经停止，再处理输出目录。
2. 对已经中止且用户确认不再保留的半成品实验，应删除对应的训练/评估输出目录、`experiment_logs/run_outputs/` 下的运行目录和主日志，避免后续误读为有效结果。
3. 删除范围必须尽量精确，只清理本次中止实验对应的半成品；不得删除已完成的 smoke test、正式结果、数据产物或与当前中止实验无关的日志。
4. 如果已经为中止实验写过中文实验日志，应在 `experiment_logs/README.md` 中把状态更新为“已停止并清理”或等价表述，而不是让总览表继续显示“进行中”。
5. 清理动作本身也应在后续实验日志中记录，说明删除了哪些路径、为什么删除，以及哪些结果被保留。

## 计划和方案表达规范

1. 如果工作过程中需要写计划、方案、实验设计、执行步骤、检查清单或阶段性总结，应使用中文。
2. 计划应尽量具体，说明要做什么、为什么做、预期产出是什么。
3. 如果计划发生变化，应在后续说明中用中文解释变化原因和新的执行路径。

## 代码注释规范

1. 写代码或修改代码时，必须添加足够的中文注释，方便后续审查和理解。
2. 新增或大幅修改的文件，应在文件顶部或主要入口附近说明文件功能。
3. 新增或大幅修改的函数、类、脚本入口，应说明其功能、输入、输出和关键约束。
4. 对复杂逻辑、非显然的数据转换、关键假设、边界条件和容易出错的实现，应在附近添加中文注释。
5. 注释应解释“为什么这样做”和“这段逻辑负责什么”，避免只重复代码表面含义。
6. 如果修改的是已有代码，应尽量保持原项目风格；但新增的关键注释仍应优先使用中文。

## 协作偏好

1. 默认优先使用本工作区已有结构和已有脚本，不轻易引入新的目录结构或工具链。
2. 重要数据处理结果应尽量保留可复核来源、生成路径和校验结果。
3. 对可能影响后续实验复现的操作，应在实验日志中明确记录。

## 实验环境规范

1. 本工作区实验、数据处理、Quito 脚本运行和与 `torch`/`quito`/`omegaconf`/`sklearn` 等依赖相关的验证，默认使用 conda 环境 `quito`。
2. 推荐直接使用解释器绝对路径执行，避免 shell 未激活环境导致依赖或包版本不一致：

   ```text
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python
   ```

3. 仅做纯文本检查、`rg` 搜索、`ls`、`sed`、`git`、`date` 或不依赖实验 Python 包的操作时，可以使用系统 shell/Python。
4. 如果某项验证因系统 Python 缺少实验依赖失败，应优先在 `quito` conda 环境下复验，并在实验日志中记录实际使用的环境和结果。
5. 新增长期脚本、README、实验协议或日志中涉及可复现实验命令时，应优先写明 `quito` 环境下的执行命令。

## 正式视觉路由实验代码目录规范

1. Visual Router、Visual-Conditioned PatchTST、Visual-Conditioned MoE 等正式实验代码应优先放在 `visual_router_experiments/` 下，而不是继续追加到通用的 `experiment_scripts/`。
2. `visual_router_experiments/` 下按实验阶段建立二级目录，例如：
   - `stage0_oracle_audit/`：专家互补性和 oracle 上限审计；
   - `stage1_vali_test_router/`：vali 训练 router、test 测试 router 的主实验；
   - `stage2_heldout_cell/`：held-out TSF cell 的 zero-shot 泛化实验；
   - `common/`：跨阶段复用的 schema、cache 读写、指标、伪图像张量构造和评估工具。
3. 阶段目录中的脚本应围绕该 stage 的输入、输出和评估协议组织；跨阶段复用逻辑应上收到 `common/`，避免复制粘贴。
4. 每个 stage 若存在探索性、小规模或半成品验证代码，应在该 stage 下建立 `pilot/` 子目录集中存放；正式实验入口、长期复用评估脚本和跨阶段工具不要长期混放在 `pilot/` 中。
5. `pilot/` 脚本应在文件顶部明确说明其 pilot 限制、默认输入规模和不能作为正式结论的口径；当 pilot 逻辑转为正式流程时，应移动或重写到 stage 根目录或 `common/`，并同步更新文档。
6. 阶段性协议、实验规划、任务拆解和路线变更应优先写入对应 stage 代码目录下的 Markdown 文档，例如 `stage1_vali_test_router/stage1_protocol_and_plan.md`；跨阶段总体规划可放在 `visual_router_experiments/` 根目录或 `common/` 相关文档中。
7. 大规模 prediction cache、embedding cache、checkpoint、评估结果和运行日志不得直接写入代码目录，应继续写入 `experiment_logs/run_outputs/YYYY-MM-DD_*_visual_router_*/`。
8. 新增 stage 目录、pilot 子目录、正式实验入口脚本、schema 文档、阶段规划文档或长期保留配置时，应同步更新 `WORKSPACE_STRUCTURE.md`，并按实验日志规范记录本次结构变化。

## 工作区结构文档维护规范

1. 工作区根目录下维护 `WORKSPACE_STRUCTURE.md`，用于说明主要文件夹、关键文件、实验输出目录和生成物的具体功能。
2. 每次新增、删除、移动长期保留的文件或文件夹时，都要同步更新 `WORKSPACE_STRUCTURE.md`；尤其是新增实验脚本、正式日志、汇总结果、配置模板、数据产物或新的输出根目录时必须记录。
3. 大规模生成物（如 checkpoint、TensorBoard event、评估日志、缓存目录）可按目录模式说明，不要求逐个文件列出；但必须写清楚其来源、用途和是否可作为正式结果引用。
4. 如果某个目录的结果口径发生变化，例如 checkpoint 选择从 validation MAE-best 改为 validation MSE-best，应在 `WORKSPACE_STRUCTURE.md` 中明确标注，避免后续误读。
5. 若新增结构文档或对工作区结构进行系统梳理，也应按实验日志规范在 `experiment_logs/` 中记录，并更新 `experiment_logs/README.md`。
