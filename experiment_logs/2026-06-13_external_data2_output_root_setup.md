# `/data2/syh` 外部输出根目录接入

日志日期：2026-06-13 16:39:18 CST

## 目的

将用户新建的 `/data2/syh` 大容量目录接入当前项目索引，作为后续大规模实验输出或临时 shard 的可选位置，同时明确视觉路线不默认预存全量伪图像或 ViT embedding。

## 背景

当前 `/home` 文件系统只剩约 21GB 可用空间，而 `/data2` 约有 3.0TB 可用空间。此前估算显示，QuitoBench 三组 config 的 vali/test 单 variant fp16 ViT embedding 全量约 93.3GB，双 variant fp16 约 186.6GB；即使迁移到 `/data2`，全量缓存也会带来较高存储和管理成本。用户倾向除非能证明缓存显著加速后续训练，否则不要预存伪图像张量或 embedding。

## 操作

1. 检查磁盘空间：
   - `/home` 可用约 21GB；
   - `/data2` 可用约 3.0TB。
2. 创建外部项目输出根目录：

   ```text
   /data2/syh/Time/run_outputs/
   /data2/syh/Time/cache_shards/
   ```

3. 新增 `EXTERNAL_OUTPUTS.md`，记录外部输出路径和视觉路线缓存原则。
4. 更新 `WORKSPACE_STRUCTURE.md`，将 `/data2/syh/Time/` 纳入工作区结构说明。

## 结果

已创建并确认以下目录存在：

```text
/data2/syh/Time/
/data2/syh/Time/run_outputs/
/data2/syh/Time/cache_shards/
```

已新增索引文档：

```text
EXTERNAL_OUTPUTS.md
```

缓存策略已明确：

1. 不预存储全量伪图像张量。
2. 不默认预存储全量 ViT embedding。
3. 若要评估缓存收益，应先做小规模 shard，比较 online imageization + ViT forward、embedding 读取和端到端训练/评估耗时。
4. 只有当缓存带来明确端到端加速，且空间、清理和 metadata 对齐可控时，才扩大缓存范围。

## 结论

`/data2/syh/Time/` 已作为外部大容量输出根目录接入项目索引。后续大规模实验可以显式把输出写到 `/data2/syh/Time/run_outputs/`，临时 shard 可以写到 `/data2/syh/Time/cache_shards/`。视觉主线当前仍优先采用 online 计算，而不是全量预缓存伪图像或 embedding。

## 下一步方案

1. 在后续 HF ViT smoke 或大规模脚本中增加 `--output-root` / `--cache-root` 参数，默认可指向 `/data2/syh/Time/`。
2. 先实现 online ViT embedding smoke，并记录 GPU latency；如需比较缓存收益，再额外生成小规模 fp16 embedding shard 做读取速度对照。
