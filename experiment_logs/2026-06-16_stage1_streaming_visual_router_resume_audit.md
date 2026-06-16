# Stage 1 Streaming Visual Router 续训能力审计

日志日期：2026-06-16 15:03:01 CST

## 目的

检查 `visual_router_experiments/stage1_vali_test_router/train_visual_router_online_streaming.py` 是否支持 `stage1_96_48_s_full_scale` streaming visual router 先训练 1 个 epoch，后续从 checkpoint 继续训练。

## 背景

`stage1_96_48_s_full_scale` 样本量很大，完整 streaming router 每个 epoch 都需要重新执行 vali split 的伪图像生成、冻结 ViT 前向和 router 参数更新。为了避免一次性长跑风险，需要确认 1 epoch 训练产物是否能保留 router、scaler、optimizer 和 epoch 状态，便于后续继续追加训练 epoch。

## 操作

1. 阅读 `train_visual_router_online_streaming.py` 的参数解析、主流程、scaler fitting、router 初始化、训练 epoch 循环、test prediction 和 metadata/status 写出逻辑。
2. 使用 `rg` 搜索 `resume`、`checkpoint`、`ckpt`、`state_dict`、`optimizer`、`scheduler`、`epoch`、`save` 等关键词，确认 streaming visual router 入口当前没有续训参数和 checkpoint 保存逻辑。
3. 对照 `train_visual_router.py` 和 `train_visual_router_online.py`，确认已有离线/online smoke 入口同样侧重一次性训练和评估，没有可直接复用的 router checkpoint 恢复实现。
4. 检查同一 `--output-dir` 重跑时的 stale 文件处理逻辑，确认当前脚本会删除 `online_embedding_manifest.csv`、`online_embedding_latency_summary.csv`、`visual_router_predictions.csv` 和 `visual_router_soft_fusion_predictions.csv`。

## 结果

当前 `train_visual_router_online_streaming.py` 不支持严格意义上的继续训练：

- 参数层面没有 `--resume-checkpoint`、`--save-checkpoint-every-epoch`、`--train-only` 或类似开关。
- 每个 `config_name` 都会重新构建 `StandardScaler()`，重新遍历 vali embedding 做 `partial_fit`，然后重新初始化 `VisualMLPRouter` 和 `AdamW` optimizer。
- 训练循环结束后只写预测、summary、metadata 和 status，不保存 router `state_dict`、optimizer `state_dict`、scaler 对象、已完成 epoch 数、训练超参签名或随机数状态。
- 同一输出目录重跑会删除既有预测和 online embedding manifest，因此不能把已有 1 epoch 结果直接追加为第 2 个 epoch。
- 当前实现没有 scheduler，所以暂时不存在 scheduler 状态恢复问题；如果后续加入 scheduler，应同步纳入 checkpoint。

建议改进方案：

1. 新增 checkpoint 输出目录，例如 `output_dir/checkpoints/`，每个 `config_name` 单独保存 `router_{config_name}_epoch_{global_epoch:04d}.pt`，并写 `latest_{config_name}.pt` 或 `latest_checkpoint_index.json`。
2. checkpoint payload 至少包含：
   - `router_state_dict`
   - `optimizer_state_dict`
   - `scaler_state`，可保存 `mean_`、`var_`、`scale_`、`n_samples_seen_`、`n_features_in_` 和 `feature_names_in_`（如存在），或使用 `joblib` 保存完整 `StandardScaler`
   - `completed_epochs`
   - `config_name`
   - `model_columns`
   - `router_mode`
   - `metric`
   - `hidden_dim/dropout/lr/weight_decay/huber_beta/kl_tau/lambda_kl`
   - `embedding_metadata` 中影响 embedding 的字段，例如 `encoder_name/variant/pooling/normalization_preset/image_size/norm_mode/pixel_mode/clip/period_selection/period_candidates`
   - `stream_shard_index/stream_shard_count`
   - `labels_path/prediction_manifest_path/config_path`
3. 新增 `--resume-checkpoint PATH`，加载后校验 checkpoint 中的 `config_name`、模型结构、五专家列顺序、训练目标、embedding 口径、shard 切分和关键输入路径。校验失败应直接报错，避免错接 checkpoint。
4. 新增 `--extra-epochs` 或把现有 `--epochs` 定义为“本次追加训练 epoch 数”；resume 时从 `completed_epochs + 1` 开始记录全局 epoch，并在 metadata 中保留 `resume_from_checkpoint`、`previous_completed_epochs` 和 `new_total_completed_epochs`。
5. resume 后可以跳过 scaler 重新拟合，直接使用 checkpoint 中的 scaler；为安全起见，可提供 `--refit-scaler-on-resume`，默认关闭。因为 full-scale scaler fitting 本身也很耗时，保留 scaler 是续训收益的关键。
6. 训练续接和 test 评估应拆开：新增 `--train-only` 或 `--skip-test-predict`，允许先只跑 1 epoch 并保存 checkpoint，不必每次追加 epoch 都重扫 test；最终需要评估时再用 `--eval-checkpoint PATH` 或 resume 后开启 test prediction。
7. 输出文件处理需要区分 fresh run 与 resume/eval：fresh run 可删除 stale CSV；resume 训练不应删除旧 checkpoint；eval 阶段可先删除并重写 prediction CSV，或写入带 epoch 标识的文件名，避免覆盖不可追溯。
8. 每个 epoch 结束立即保存 checkpoint，并在 `status.json` 记录当前 `config_name`、`epoch`、`checkpoint_path` 和 `completed_epochs`，方便长任务中断后找到恢复点。

## 结论

当前代码适合一次性跑完指定 epoch 并立刻 test 评估，不适合先跑 1 epoch 后从 checkpoint 继续训练。正式跑 `stage1_96_48_s_full_scale` 之前，应先实现并 smoke 测试上述 checkpoint/resume 机制，尤其是 scaler 保存与恢复、optimizer 状态恢复、epoch 编号续接和输出文件不误删。

## 下一步方案

1. 在 `train_visual_router_online_streaming.py` 中实现 checkpoint save/load helper，优先只支持单 `config_name=96_48_S` 的 full-scale 主线，避免一次性泛化过度。
2. 用小规模 `--max-samples-per-split` smoke 验证：连续跑 2 epoch 与“先 1 epoch 保存 checkpoint，再 resume 追加 1 epoch”在相同 seed 和无 dropout/相同数据顺序下产出一致或数值接近。
3. smoke 通过后再启动 full-scale 1 epoch，建议使用后台 launcher/nohup/tmux，并在日志中记录 checkpoint 路径、恢复命令、监控命令和停止命令。
