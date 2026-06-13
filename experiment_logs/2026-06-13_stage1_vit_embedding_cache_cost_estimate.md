# Stage 1 ViT Embedding Cache 开销估算

日志日期：2026-06-13 14:26:49 CST

## 目的

在接入 `google/vit-base-patch16-224` 之前，估算 QuitoBench vali/test window-level embedding cache 的空间开销，并检查当前机器是否适合全量缓存 embedding。

## 背景

上一阶段已经验证在线伪图像化 pilot 可以把 120 个 `96_48_S` sample_key 从 Quito 历史窗口 `x` 在线生成 `variant_a=3view` 和 `variant_b=top3fold`。用户指出 120 样本只适合 smoke，不应据此判断 router 性能；同时希望评估如果缓存 ViT embedding，空间开销是否可接受，若过大则考虑转向在线计算。

## 操作

1. 检查 Quito conda 环境中的 Hugging Face 依赖：
   - `transformers==4.57.6`；
   - `AutoImageProcessor` 和 `ViTModel` 可导入。
2. 读取 `96_48_S`、`576_288_S`、`1024_512_S` 三组 DLinear evaluate config，分别加载 Quito vali/test dataset，统计真实 window 数。
3. 按 `google/vit-base-patch16-224` 的 hidden size `768` 估算 embedding 空间：
   - 单 variant：`num_windows * 768 * bytes_per_value`；
   - 双 variant：再乘以 2；
   - 同时估算 fp32 和 fp16。
4. 尝试用 `HF_ENDPOINT=https://hf-mirror.com` 加载 `google/vit-base-patch16-224`：
   - 默认 Hugging Face 下载曾卡住，已停止残留进程；
   - 镜像下载权重仍超过 5 分钟未完成，已手动停止，避免后台占用；
   - 已从本地部分缓存读取到 image processor 和 config。
5. 检查当前机器资源：
   - GPU 2 空闲显存约 24GB，适合后续 ViT embedding 推理；
   - `/home` 文件系统可用空间约 22GB，不适合直接写入大规模 embedding cache。

## 结果

真实 vali/test window 数：

| config | vali windows | test windows | total windows |
| --- | ---: | ---: | ---: |
| `96_48_S` | 9,350,520 | 13,924,650 | 23,275,170 |
| `576_288_S` | 7,802,520 | 12,376,650 | 20,179,170 |
| `1024_512_S` | 6,357,720 | 10,931,850 | 17,289,570 |
| 合计 | 23,510,760 | 37,233,150 | 60,743,910 |

按 768 维 ViT embedding 估算空间：

| 范围 | variant 数 | dtype | GB | GiB |
| --- | ---: | --- | ---: | ---: |
| `96_48_S` | 1 | fp32 | 71.501 | 66.591 |
| `96_48_S` | 1 | fp16 | 35.751 | 33.295 |
| `96_48_S` | 2 | fp32 | 143.003 | 133.182 |
| `96_48_S` | 2 | fp16 | 71.501 | 66.591 |
| `576_288_S` | 1 | fp32 | 61.990 | 57.733 |
| `576_288_S` | 1 | fp16 | 30.995 | 28.867 |
| `1024_512_S` | 1 | fp32 | 53.114 | 49.466 |
| `1024_512_S` | 1 | fp16 | 26.557 | 24.733 |
| 三组合计 | 1 | fp32 | 186.605 | 173.790 |
| 三组合计 | 1 | fp16 | 93.303 | 86.895 |
| 三组合计 | 2 | fp32 | 373.211 | 347.579 |
| 三组合计 | 2 | fp16 | 186.605 | 173.790 |

`google/vit-base-patch16-224` 本地已缓存到的 processor/config 关键信息：

```text
image_mean = [0.5, 0.5, 0.5]
image_std = [0.5, 0.5, 0.5]
size = 224
hidden_size = 768
```

这说明后续接 HF ViT 时，不能直接使用上一版通用 `imagenet_normalize()` 的 torchvision ImageNet mean/std，而应按 HF processor 口径对 `[0, 1]` tensor 做 `(x - 0.5) / 0.5`。当前伪图像化本体输出 `[0, 1]`，这一点与 HF ViT 输入兼容；需要调整的是 encoder 前 normalization 函数，而不是图像化本体。

## 结论

1. 当前 `/home` 只剩约 22GB 可用空间，无法承载任一完整 config 的单 variant fp16 embedding cache；最小的 `1024_512_S` 单 variant fp16 也约 26.6GB 十进制。
2. 若要全量覆盖三组 config 的 vali/test：
   - 单 variant fp16 约 93.3GB；
   - 双 variant fp16 约 186.6GB；
   - 双 variant fp32 约 373.2GB。
3. 因此当前机器上不建议直接做全量 embedding cache。更合理的路线是：
   - 先做小规模/分层抽样 embedding smoke；
   - 若要缓存，采用 fp16、分 shard、单 variant 优先；
   - 若磁盘不可扩展，则正式训练改为 online imageization + online ViT forward，或只缓存临时 mini-shard。
4. 由于 `google/vit-base-patch16-224` 使用 `[0.5, 0.5, 0.5]` mean/std，之前的 `imagenet_normalize()` 只适合 torchvision/ImageNet mean/std 口径，不适合直接接 HF ViT。

## 下一步方案

1. 新增 HF ViT embedding smoke 脚本，支持：
   - `--hf-endpoint https://hf-mirror.com`；
   - `--device cuda`；
   - `--variant variant_a`；
   - `--dtype fp16`；
   - `--max-samples` 或按 split/dataset 分层抽样；
   - 输出 embedding shard、manifest、metadata 和 latency。
2. 在公共模块中新增或扩展 encoder-normalization 函数，明确区分：
   - torchvision ImageNet mean/std；
   - HF ViT processor mean/std。
3. 在磁盘空间没有扩展前，不启动 QuitoBench vali/test 全量 embedding cache；后续如需要正式大规模实验，应优先实现 online 训练/评估或可配置 shard 清理策略。
