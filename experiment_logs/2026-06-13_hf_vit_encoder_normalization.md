# HF ViT Encoder Normalization 实现

日志日期：2026-06-13 23:06:09 CST

## 目的

完成 Stage 1 下一步任务的第一步：在 `pseudo_imageization.py` 中扩展实现 Hugging Face `google/vit-base-patch16-224` direct-forward 路径对应的 encoder normalization。`openai_clip` 和 `processor_uint8` 路径本次暂缓，不纳入实现范围。

## 背景

当前在线伪图像化模块输出 float `[0, 1]` 的 `[B, 3, H, W]` 伪图像。此前只保留了 `imagenet_normalize()`，使用 torchvision/ImageNet mean/std：

```text
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

但后续最近任务优先接入 `google/vit-base-patch16-224`。该模型的 Hugging Face image processor 口径为：

```text
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
```

因此 direct `pixel_values` 路径应执行 `(x - 0.5) / 0.5`，不能继续默认使用 ImageNet mean/std。

## 操作

1. 阅读 `visual_router_experiments/common/pseudo_imageization.py`，确认当前只有 `IMAGENET_MEAN`、`IMAGENET_STD` 和 `imagenet_normalize()`。
2. 修改 `pseudo_imageization.py`：
   - 新增 `HF_VIT_MEAN = (0.5, 0.5, 0.5)`；
   - 新增 `HF_VIT_STD = (0.5, 0.5, 0.5)`；
   - 新增 `ENCODER_NORMALIZATION_PRESETS`，当前包含 `hf_vit_0_5` 和 `torchvision_imagenet`；
   - 新增 `_validate_encoder_image_tensor()`，统一校验 `[B, 3, H, W]` 输入；
   - 新增 `encoder_normalize(x, preset="hf_vit_0_5")`；
   - 新增 `hf_vit_normalize(x)`，作为 HF ViT 直接入口；
   - 将原 `imagenet_normalize(x)` 改为 `encoder_normalize(..., preset="torchvision_imagenet")` 的兼容包装，并在注释中明确它不适合作为 HF ViT 默认路径。
3. 更新 `visual_router_experiments/common/README.md`，说明 `pseudo_imageization.py` 已支持 `hf_vit_0_5` / `torchvision_imagenet` 两种 encoder normalization。
4. 运行语法检查：

   ```text
   python -m py_compile visual_router_experiments/common/pseudo_imageization.py
   /home/shiyuhong/application/miniconda3/envs/quito/bin/python -m py_compile visual_router_experiments/common/pseudo_imageization.py
   ```

5. 使用 Quito conda 环境运行函数级 smoke：
   - 校验 `hf_vit_normalize(x)` 等价于 `(x - 0.5) / 0.5`；
   - 校验 `encoder_normalize(x, preset="hf_vit_0_5")` 等价于 HF ViT normalization；
   - 校验 `imagenet_normalize(x)` 仍等价于 ImageNet mean/std；
   - 校验 dtype 保持；
   - CUDA 可用时校验 CUDA + fp16 路径保持 device/dtype；
   - 校验非法 shape 和未知 preset 会抛出 `ValueError`。

## 结果

代码文件已更新：

```text
visual_router_experiments/common/pseudo_imageization.py
```

文档文件已更新：

```text
visual_router_experiments/common/README.md
```

验证结果：

- 系统 Python 语法检查通过；
- Quito conda 环境语法检查通过；
- Quito conda 环境函数级 smoke 通过，输出：

```text
hf_vit_normalize smoke ok
HF_VIT_MEAN (0.5, 0.5, 0.5) HF_VIT_STD (0.5, 0.5, 0.5)
```

系统 Python 的函数级 smoke 未运行成功，原因是系统环境没有安装 `torch`；已使用 Quito conda 环境完成同等函数级验证。

## 结论

HF ViT direct-forward 路径所需的 encoder normalization 已实现。后续 `google/vit-base-patch16-224` embedding smoke 可以直接使用：

```python
pixel_values = hf_vit_normalize(pseudo_image)
```

或：

```python
pixel_values = encoder_normalize(pseudo_image, preset="hf_vit_0_5")
```

旧的 `imagenet_normalize()` 仍可用于 torchvision/MAE/timm ImageNet mean/std 口径，但不再应作为 HF ViT 默认入口。

## 下一步方案

1. 在 HF ViT embedding smoke 脚本中使用 `hf_vit_normalize()` 或 `encoder_normalize(..., preset="hf_vit_0_5")`。
2. embedding smoke 的 metadata 中记录 `encoder_name`、`normalization_preset="hf_vit_0_5"` 和 direct-forward 路径。
3. `openai_clip` 和 `processor_uint8` 后续需要时再单独补充，不在本次变更中实现。
