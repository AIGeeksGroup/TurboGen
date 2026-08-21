# OmniTurbo 训练流程

本文档描述当前仓库实际采用的 OmniForcing v2 训练流程，以及 Step2 优化后的阶段交接关系。

## 1. 任务范围

当前任务是 5 秒短视频生成，不包含长视频训练。默认配置为 `num_frames: 121`、`video_height: 512`、`video_width: 768`；121 帧按 24 FPS 约为 5.04 秒。

121 个像素帧经过时间压缩后对应 16 个 video latent frames，使用 `4-3-3-3-3` causal blocks；音频对应 126 个 aligned latent frames。本流程不包含跨 clip 历史 cache、EMA 长视频 cache、OPSD-V 长视频机制或多段视频拼接。

## 2. 总体流程

```text
LTX-2.3 FP8 checkpoint + Gemma + prompts
                         |
                         v
Step 1: 双向 teacher ODE 数据（兼容流程）
                         |
                         v
Step 2: causal on-policy Flow-OPD velocity matching
                         |
                         v
                 Stage 2 causal checkpoint
                         |
                         v
Step 3: causal Self-Forcing DMD
                         |
                         v
                 最终 causal generator
```

## 3. Step 1：双向 teacher ODE 数据

Step 1 使用双向 LTX-2 teacher，对 prompt 初始化的 video/audio 纯噪声执行 ODE Euler 去噪，并保存指定 sigma 节点、trajectory、clean latent 和 sigma 信息。旧流程可将 `ode_pairs` 转换为 `ode_lmdb`。

默认 sigma 节点为 `[1000, 909, 725, 421, 0]`。当前默认 on-policy Step2 不依赖这些 LMDB trajectory；只有显式设置 `on_policy: false` 时才使用旧的 LMDB 路径。

## 4. Step 2：Causal on-policy Flow-OPD

主要实现文件：

```text
LTX-2/packages/ltx-distillation/src/ltx_distillation/ode/ode_regression.py
LTX-2/packages/ltx-distillation/src/ltx_distillation/ode/train_ode.py
```

### 4.1 输入与 rollout

当前默认配置使用 prompt-only 数据：

```yaml
on_policy: true
prompt_file: ./prompts/benchmark_512.txt
train_prompt_file: ./prompts/benchmark_512.txt
num_inference_steps: 40
denoising_step_list: [1000, 909, 725, 421, 0]
```

每行一个 prompt。video/audio latent 在训练中实时初始化，不从 LMDB 读取。

每个 batch 的流程是：

1. Gemma 生成文本 conditioning。
2. 为当前 5 秒视频初始化 video/audio 纯高斯噪声。
3. 按 `4-3-3-3-3` block 顺序推进。
4. 每个 block 使用 causal generator 和 KV-cache 执行 denoising schedule。
5. student 输出 x0 后转换为 velocity。
6. 对 detached x0 按下一个 sigma 重新加噪，得到下一个 on-policy 状态。
7. block 完成后将 clean block 写回 KV-cache，供后续 block 使用。

实际 sigma 从 LTX2Scheduler 的 40-step schedule 中按最近邻映射得到。

### 4.2 frozen teacher 与 teacher prefix

teacher 默认配置为：

```yaml
teacher_fp8_mode: static
teacher_device: cpu
on_policy_teacher_prefix: true
```

第一个 block 的 teacher 只接收当前 noisy block；后续 block 接收 `已生成的 clean prefix + 当前 noisy block`，不会看到未来 block。

student 和 teacher 都返回 x0，随后计算：

```text
velocity = (noisy - x0) / sigma
loss = w(sigma) * ||student_velocity - stop_gradient(teacher_velocity)||^2
```

Flow-OPD 权重使用方案中的公式：

```python
term1 = sigma * (1 - sigma) / (2 * sigma.clamp(min=1e-8))
term2 = 1 / sigma.clamp(min=1e-8)
w = (delta_t / 2) * (term1 + term2).pow(2)
```

默认 `flow_opd_delta_t: 0.001`，视频和音频分别使用 `video_loss_weight`、`audio_loss_weight`。

### 4.3 其他 Step2 修改

- 旧 trajectory sampling 从 `[0, T)` 改为 `[0, T-1)`，不再采样 clean endpoint。
- 新增 prompt-only `PromptDataset` 和 `collate_prompt_batch`。
- on-policy loss 通过 `backward_callback` 对每个 block/denoising transition 立即反传，兼容 gradient accumulation 和 FSDP `no_sync()`。
- 旧 LMDB ODE regression 保留为显式兼容模式。

Step2 输出：

```text
outputs/stage2_causal_ode/checkpoint_XXXXXX/model.pt
```

完整恢复还需要同目录的 `trainer_state.json` 和各 rank 的 trainer state 文件。

## 5. Step 3：Causal Self-Forcing DMD

配置文件为：

```text
LTX-2/packages/ltx-distillation/configs/stage3_causal_dmd.yaml
```

Step3 使用 Step2 causal checkpoint 初始化 generator，在同一个 5 秒视频范围内进行 causal Self-Forcing DMD：

1. generator 从纯噪声开始。
2. 按相同 `4-3-3-3-3` block layout 和 KV-cache 推进。
3. 使用 `[1000, 909, 725, 421, 0]` 去噪 schedule。
4. 刷新 context cache，计算 real bidirectional score、fake score 和 DMD loss。
5. 更新 generator，周期性保存 checkpoint 和 benchmark 视频。

Step3 的历史字段 `stage1_ckpt_path` 当前用于填写 Step2 causal checkpoint：

```yaml
stage1_ckpt_path: /path/to/stage2/checkpoint_XXXXXX/model.pt
```

## 6. 启动与交接

推荐使用根目录 wrapper：

```bash
cd /path/to/OmniTurbo
bash ./train_step2_manual.sh
```

`on_policy: true` 时，wrapper 只检查 prompt 文件，不准备或下载 LMDB；`on_policy: false` 时继续执行旧的 LMDB manifest 校验。支持 4 GPU 或 8 GPU。

阶段交接为：

```text
原始 LTX-2.3 checkpoint
        |
        +--> Step2 causal generator
                         |
                         +--> Step2 model.pt
                                  |
                                  +--> Step3 generator 初始化
                                           |
                                           +--> 最终 causal checkpoint
```

## 7. 未实现范围

本次没有修改 Step3 的 `L_DMD + lambda(t) * L_OPD` 融合，也没有实现 OPSD-V 长视频训练、跨 clip 历史 cache、EMA long-video cache、长视频数据集或长视频推理。

## 8. 验证状态

已完成 Python 编译、`git diff --check`、prompt dataset smoke test、Flow-OPD 公式测试、clean endpoint sampling 测试和小型 rollout shape 测试。完整 pytest 受当前环境缺少 `triton` 阻断，真实 4/8 GPU 训练尚未执行。
