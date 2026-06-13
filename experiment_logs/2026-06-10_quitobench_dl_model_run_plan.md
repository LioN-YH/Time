# QuitoBench 深度学习模型运行方案日志

## 日志信息

- 日志日期：2026-06-10 02:42:28 CST
- 工作目录：`/home/shiyuhong/Time`
- 相关项目：`/home/shiyuhong/Time/quito`
- 相关论文：`http://arxiv.org/abs/2603.26017`
- 目标模型：CrossFormer、PatchTST、DLinear

## 目的

阅读 QuitoBench 论文和本地 `quito` 代码，梳理在 QuitoBench 上运行 CrossFormer、PatchTST 和 DLinear 的具体步骤，明确数据准备、调参、训练、checkpoint 衔接、评估和结果整理流程。

## 背景

论文中 QuitoBench 的深度学习模型实验采用 3 阶段流程：

1. 每个任务配置先在验证集上做超参数搜索。
2. 固定最佳超参数后，从头训练模型。
3. 加载最佳 checkpoint，在测试集上做 dense rolling-window evaluation。

论文定义的任务配置为：

- context length：`96`、`576`、`1024`
- forecast horizon：`48`、`288`、`512`
- forecasting mode：`S` 和 `M`

因此每个模型共有 `3 x 3 x 2 = 18` 个任务配置。论文还说明深度学习模型使用 MSE 训练，主要报告 MAE，并使用 3 个随机种子降低随机初始化方差。

## 操作

1. 阅读论文摘要、实验设置、Appendix E.2 和数据格式说明。
2. 阅读本地 `quito` 代码中的关键入口：
   - `quito/cli.py`
   - `quito/scripts/tune.py`
   - `quito/scripts/finetune.py`
   - `quito/scripts/evaluate.py`
   - `quito/datasets.py`
   - `quito/trainers/base.py`
3. 检查三个目标模型的配置目录：
   - `configs/tune/patchtst`
   - `configs/tune/crossformer`
   - `configs/finetune/patchtst`
   - `configs/finetune/crossformer`
   - `configs/finetune/dlinear`
   - `configs/evaluate/patchtst`
   - `configs/evaluate/crossformer`
   - `configs/evaluate/dlinear`
4. 检查本地环境：
   - Python 环境：`/home/shiyuhong/application/miniconda3/envs/quito`
   - PyTorch：`2.5.1+cu121`
   - CUDA 可用：是
   - GPU 数量：4
   - Ray：`2.55.1`
5. 验证 Quito 默认数据目录：
   - `examples/datasets/cluster_data/open_hour_data.parquet`
   - `examples/datasets/cluster_data/open_min_data.parquet`
   - `examples/datasets/cluster_data/item_clusters.csv`
6. 用 `configs/finetune/dlinear/96_48_S.yaml` 做最小数据加载验证，确认 train/valid/test 均能加载，并且能读到 cluster 映射。

## 结果

代码和配置状态如下：

- PatchTST：有 `tune`、`finetune`、`evaluate` 三类配置，各 18 个任务配置。
- CrossFormer：有 `tune`、`finetune`、`evaluate` 三类配置，各 18 个任务配置。
- DLinear：有 `finetune`、`evaluate` 配置，各 18 个任务配置；没有 `tune/dlinear` 目录。
- `quito-cli finetune` 会包装为 `torchrun --nproc_per_node=N`，适合多 GPU 训练。
- `evaluate.py` 需要在配置中的 `resume.checkpoint_path` 指向真实 checkpoint。
- 训练保存的 checkpoint 位于 `outputs/.../FINE_TUNE.../checkpoints/`，命名通常为 `best_*.ckpt`、`ckpt_*.ckpt` 或 `last_*.ckpt`。
- 当前 evaluate YAML 中的 `./models/{model}/{config}/ckpt_*.pkl` 是占位路径，需要在训练后替换为真实 checkpoint，或将真实 checkpoint 复制/软链接到对应路径。

最小数据加载验证通过：

- `TRAIN`、`VALID`、`TEST` 均可加载。
- `TEST_DATA_MIN` 和 `TEST_DATA_HOUR` 均可读取。
- `cluster` 映射可用。

## 结论

在当前本地环境中，已经具备运行 CrossFormer、PatchTST 和 DLinear 的基础条件。推荐执行顺序是：

1. 先用单个轻量配置做 smoke test。
2. 再跑 18 个完整配置。
3. PatchTST 和 CrossFormer 可以先跑 tune，再把最佳超参固化到 finetune 配置。
4. DLinear 直接用现有 finetune 配置训练。
5. 每个配置用 3 个 seed 训练。
6. 训练完成后，把每个 seed 的最佳 checkpoint 写入 evaluate 配置，再运行 evaluate。

## 下一步方案

1. 先选择一个 smoke test 配置，例如 `96_48_S`。
2. 使用 1 张 GPU 跑 DLinear 的 smoke test，确认完整训练和 checkpoint 保存流程。
3. 如果 smoke test 成功，再决定是否：
   - 修改/补全批量运行脚本；
   - 自动收集 best checkpoint；
   - 自动生成 evaluate 配置；
   - 批量运行 3 个模型的 18 个配置。
