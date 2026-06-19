"""
文件功能：
    提供 Stage 1 P7b 最小 TimeFuseLinearSoftmaxHead。

设计边界：
    该 head 只消费调用方显式传入的 FeatureBatch 和 model_columns，把
    FeatureBatch.features 通过固定线性权重映射为 logits，并沿专家维度做
    softmax 得到 weights。它不训练、不计算 loss、不创建 optimizer、不保存
    checkpoint，不读取 prediction cache、oracle/TSF 或 feature CSV，也不创建
    run_dir 或写 status/metadata/CSV/JSON/Parquet。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from time_router.protocols import FeatureBatch, RouterOutput


class TimeFuseLinearSoftmaxHead:
    """
    类功能：
        将 TimeFuse 结构特征批次适配为最小 RouterOutput。

    输入：
        weight: 二维矩阵，形状为 `(feature_dim, num_experts)`；
        bias: 可选一维向量，长度为 `num_experts`；
        dtype: 内部 numpy 计算 dtype，默认 float64 以便 smoke 复算稳定。

    输出：
        `predict(feature_batch, model_columns)` 返回 RouterOutput，其中
        sample_keys 保持 FeatureBatch 原顺序，logits/weights 的专家维度与
        model_columns 对齐。

    关键约束：
        该类只做前向映射和 softmax，不持有训练状态，不访问文件系统或正式
        runtime 入口。model_columns 是唯一专家顺序来源。
    """

    head_name = "TimeFuseLinearSoftmaxHead"

    def __init__(self, *, weight: Any, bias: Any | None = None, dtype: Any = np.float64) -> None:
        self.dtype = dtype
        self.weight = np.asarray(weight, dtype=self.dtype)
        if self.weight.ndim != 2:
            raise ValueError("TimeFuseLinearSoftmaxHead.weight 必须是二维矩阵")
        if self.weight.shape[0] == 0 or self.weight.shape[1] == 0:
            raise ValueError("TimeFuseLinearSoftmaxHead.weight 的 feature/expert 维度不能为空")

        if bias is None:
            self.bias = np.zeros((self.weight.shape[1],), dtype=self.dtype)
        else:
            self.bias = np.asarray(bias, dtype=self.dtype)
            if self.bias.ndim != 1:
                raise ValueError("TimeFuseLinearSoftmaxHead.bias 必须是一维向量")
            if self.bias.shape[0] != self.weight.shape[1]:
                raise ValueError(
                    "TimeFuseLinearSoftmaxHead.bias 长度必须等于专家数："
                    f"bias={self.bias.shape[0]} experts={self.weight.shape[1]}"
                )

    def predict(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """
        函数功能：
            将 FeatureBatch.features 转为 RouterOutput(logits, weights)。

        输入：
            feature_batch: 已由 provider 构造好的特征批次；
            model_columns: 专家动作空间顺序，长度必须等于线性层输出维度。

        输出：
            RouterOutput；sample_keys 与 FeatureBatch.sample_keys 完全一致，
            logits/weights shape 均为 `(num_samples, num_experts)`。
        """
        columns = tuple(str(model_name) for model_name in model_columns)
        if not columns:
            raise ValueError("TimeFuseLinearSoftmaxHead.predict 需要非空 model_columns")
        if len(columns) != len(set(columns)):
            raise ValueError(f"TimeFuseLinearSoftmaxHead.predict 收到重复 model_columns：{columns}")
        if len(columns) != self.weight.shape[1]:
            raise ValueError(
                "model_columns 长度必须等于线性层专家输出维度："
                f"columns={len(columns)} experts={self.weight.shape[1]}"
            )

        features = np.asarray(feature_batch.features, dtype=self.dtype)
        if features.ndim != 2:
            raise ValueError("FeatureBatch.features 必须是二维矩阵")
        sample_keys = tuple(str(sample_key) for sample_key in feature_batch.sample_keys)
        if features.shape[0] != len(sample_keys):
            raise ValueError(
                "FeatureBatch.features 样本维度必须等于 sample_keys 数量："
                f"features={features.shape[0]} sample_keys={len(sample_keys)}"
            )
        if features.shape[1] != self.weight.shape[0]:
            raise ValueError(
                "FeatureBatch.features 特征维度必须等于线性层输入维度："
                f"features={features.shape[1]} weight={self.weight.shape[0]}"
            )

        logits = features @ self.weight + self.bias
        weights = self._softmax(logits)
        return RouterOutput(
            sample_keys=sample_keys,
            model_columns=columns,
            logits=logits,
            weights=weights,
            extra={
                "head_name": self.head_name,
                "feature_schema": dict(feature_batch.feature_schema),
                "feature_dim": int(features.shape[1]),
                "num_experts": len(columns),
            },
        )

    def __call__(self, feature_batch: FeatureBatch, model_columns: Sequence[str]) -> RouterOutput:
        """
        函数功能：
            提供与常见 head 调用习惯一致的前向入口。
        """
        return self.predict(feature_batch, model_columns)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        函数功能：
            沿专家维度计算稳定 softmax。

        关键约束：
            只对已校验的二维 logits 做纯 numpy 计算；减去逐样本最大值避免
            指数溢出，输出每个样本的专家权重和为 1。
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
