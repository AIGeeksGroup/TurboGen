# Step 2 Training - Bootstrap Commands

## Verified Execution Contract

- The canonical Stage 2 configuration is `LTX-2/packages/ltx-distillation/configs/stage2_causal_ode.yaml`.
- `max_steps: 1000000` is the original full-training value. Do not reduce it.
- `save_iters: 500`, `visualize_iters: 500`, and `benchmark_iters: 500` are all unchanged.
- At every 500-step boundary, the trainer saves a full checkpoint, uploads it synchronously to Hugging Face, generates benchmark output with the trained generator, and logs the training and benchmark videos to WandB.
- `hf_upload_required: true` and `hf_upload_blocking: true`: an HF upload failure must stop training. `wandb_video_required: true`: missing video generation or upload must fail loudly.
- WandB startup uses a 300-second initialization timeout (`wandb_init_timeout: 300`). The timeout can be overridden with `WANDB_INIT_TIMEOUT`, but do not reduce it below 300 seconds on an HPC node with slow outbound access.
- The launcher does not create a tmux session itself. Start the launcher from the tmux command in Section 7 so the full training remains attached to a persistent background session.
- GPU selection is 8 GPUs first. With no explicit GPU selection, 8 visible physical GPUs use `0,1,2,3,4,5,6,7`; exactly 4 visible GPUs use `0,1,2,3`. Any other count is rejected. If a machine has 8 GPUs but only the last four may be used, explicitly set `CUDA_VISIBLE_DEVICES=4,5,6,7`.
- The base YAML remains `expected_world_size: 8`. Only the launcher-generated temporary runtime YAML changes this value to `4` during a four-GPU run; the repository YAML is never edited for that fallback.
- HF credentials are read from `HF_TOKEN` or the root `.hf_token`; WandB credentials are read from `WANDB_API_KEY` or the root `.wandb_key`. The YAML targets HF model repo `aaachier/OmniForcing-backup`, prefix `step2_ltx23`, and WandB project `OmniForcing` under entity `OmniForcing1`.

This execution contract takes precedence over older four-GPU examples later in
this document. Use the following persistent launcher for the default 8-GPU
run; use a four-GPU command only when eight GPUs are not available.

Before running any training command, always update the code to the latest
remote `main` branch. Do not start Step 2 from an old checkout:

```bash
cd /data/minghua/zzy/OmniForcing
git fetch origin main
git status --short
git pull --ff-only origin main
git log -1 --oneline
```

The final commit must contain the current Step 2 launcher, configuration,
training code, and this guide. If `git pull --ff-only` refuses because tracked
files have local changes, do not use `git reset --hard`; report the changed
files to the owner and follow Section 5 to preserve the old checkout.

```bash
tmux new-session -d -s omniforcing-step2 \
  'cd /path/to/OmniTurbo && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./train_step2_manual.sh 2>&1 | tee ./step2.log'
```

Run these commands on a fresh training machine before following the rest of
this guide. Replace placeholder values, and never commit either token file.

```bash
git clone https://github.com/AIGeeksGroup/TurboGen.git OmniTurbo
cd OmniTurbo

mkdir -p checkpoints/LTX-2.3-fp8
mkdir -p checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
python -m pip install -U huggingface_hub

# Authenticate for gated/private model downloads and checkpoint uploads.
hf auth login
hf download Lightricks/LTX-2.3 \
  --include '*ltx-2.3-22b-dev-fp8.safetensors*' \
  --local-dir checkpoints/LTX-2.3-fp8
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir checkpoints/gemma-3-12b-it-qat-q4_0-unquantized

# The training script reads these files by default.
umask 077
printf '%s\n' '<HF_TOKEN>' > .hf_token
printf '%s\n' '<WANDB_API_KEY>' > .wandb_key

# Alternatively, export credentials for the current shell.
export HF_TOKEN='<HF_TOKEN>'
export WANDB_API_KEY='<WANDB_API_KEY>'

# Optional mirror; use the official endpoint when the machine can reach HF.
export HF_ENDPOINT=https://hf-mirror.com
# export HF_ENDPOINT=https://huggingface.co

# Required when using the packaged training environment.
export OMNIFORCING_ENV_ARCHIVE=/path/to/omniforcing-conda.tar.zst
```

Expected model paths:

```text
checkpoints/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors
checkpoints/gemma-3-12b-it-qat-q4_0-unquantized/
```

The Step 2 YAML uploads checkpoints to `aaachier/OmniForcing-backup` under
`step2_ltx23/`, and logs to WandB project `OmniForcing` and entity
`OmniForcing1`. Verify authentication with `hf auth whoami`; use `wandb login`
or `WANDB_API_KEY` before starting training.

# OmniForcing Step 2 全流程运行说明

## 0. 直接交给 Coding Agent 的执行要求

可以把本文件路径交给 Agent，并附上这一句话：

> 请完整阅读 /data/minghua/zzy/OmniForcing/STEP2_TRAINING_GUIDE.md，不要只给方案或命令。请按文档实际检查并更新仓库、处理旧 Step 1 数据、使用 B200 后四卡启动 Step 2，持续检查到训练进程正常运行，然后把 commit、tmux、GPU、日志和缺失项如实汇报给我。

Agent 接到任务后应直接执行以下流程：

1. 检查 /data/minghua/zzy/OmniForcing 的 remote、branch 和工作区，并更新到 origin/main 最新版本。工作区干净时使用 git pull --ff-only；有旧修改或冲突时按第 5 节保留旧目录后重新 clone。不得直接删除整个旧项目，也不得使用 git reset --hard 清理用户改动。
2. 检查基础模型、Gemma、conda 环境压缩包、HF token 和 WandB key。缺失时报告准确路径和错误，不得把空文件或错误模型当成已准备完成。
3. 不复制或信任旧的 ode_pairs/ode_lmdb，也不要提前永久删除它们。让最新版 train_step2_manual.sh 先验证；错误数据由脚本在正确 HF 数据下载并验证成功后自动移入可恢复的备份目录。
4. 本次 B200 任务固定使用后四张物理卡，启动命令必须显式带 CUDA_VISIBLE_DEVICES=4,5,6,7。不得把仓库标准 YAML 的 expected_world_size: 8 手工改成 4；一键脚本会自动生成临时四卡运行配置。
5. 使用第 7 节的 tmux 命令启动，不要只在前台打印一遍命令。除非明确要求短跑，否则不要擅自修改 max_steps、保存频率、benchmark 频率或上传配置。
6. 持续查看 step2.log，至少确认 STEP1_VALIDATE valid、4 张指定 GPU、4 个 torchrun rank 正常加载并进入训练，且没有立即出现 traceback、CUDA OOM 或分布式报错。确认后保持 tmux 中的正式训练继续运行，不要为了“测试结束”主动停止。
7. 如果日志出现 `wandb.errors.errors.CommError: Run initialization has timed out after 90.0 sec`，说明使用了旧代码或旧配置。先确认当前 commit 包含 `wandb_init_timeout: 300`，再重新启动；不要把 `WANDB_MODE` 改成 offline，因为本流程要求在线 WandB 视频记录。

Agent 的首次执行报告必须包含：

- 当前 git commit，以及是否与 origin/main 一致；
- 使用的 CUDA_VISIBLE_DEVICES、进程数和 tmux session 名；
- step1 数据校验结果，以及旧错误数据被备份到哪里；
- step2.log 中证明训练已经正常启动的关键行；
- WandB run 地址和本地输出目录；
- 尚未到 500 step 时，明确写“500 step 的 HF checkpoint 和 WandB 视频尚未触发”，不得提前声称上传成功；到达 500 step 后再按第 8、9 节核验上传。

注意：STEP2_PREPARE_ONLY=1 只会下载并验证数据，不会加载 Step 2 模型或启动 torchrun，因此它只能算数据预检，不能当作 Step 2 训练已经测试成功。本说明的默认目标是启动并保留正式训练，而不是运行一个结束即退出的短测试。

这份说明对应项目根目录的 train_step2_manual.sh。标准配置默认使用单机 8 卡；如果运行时只检测到 4 张可见 GPU，脚本会自动回退到单机 4 卡。启动前还会自动检查、下载 step1 的 ODE 数据。

## 1. 这一步做什么

step1 已经生成并上传了 ODE 数据，不需要重新生成，也不涉及 DMD。step2 使用这些数据训练 causal ODE regression model：

- 从 aaachier/OmniForcing-backup/ode_pairs/ 下载原始 ODE pair 文件；
- 从 aaachier/OmniForcing-backup/ode_lmdb/ 下载训练用 LMDB；
- 以 LTX-2.3 FP8 基础模型为初始化，读取 Gemma 文本编码器；
- 使用 8 张或 4 张 GPU 训练 causal/autoregressive generator。

## 1.1 本次 H20 报错的原因

本次 H20 日志中，8 个分布式 rank 都已经启动并绑定到 GPU 0--7。日志里的
NCCL `using GPU ... as device ... is currently unknown` 是初始化 warning，
不是导致任务退出的错误。

真正导致任务退出的是 rank 0 初始化 WandB 时的网络等待超时：

~~~text
wandb.errors.errors.CommError:
Run initialization has timed out after 90.0 sec
~~~

也就是说，训练还没有进入第一个训练 step，也没有执行到 500 step 的保存、
HF 上传或视频推理；HPC 节点连接 WandB 超过默认 90 秒后，rank 0 抛出异常，
其余分布式进程随之退出。这不是 Step 1 的 `ode_lmdb` 校验错误，也不是
GPU 数量错误。

当前代码已将 WandB 初始化超时提高到 300 秒，配置字段是：

~~~yaml
wandb_init_timeout: 300
~~~

如果节点网络仍然较慢，启动前可以设置 `WANDB_INIT_TIMEOUT=600`。不能把
`WANDB_MODE` 改成 `offline` 来绕过问题，因为本流程要求在线记录 WandB
训练视频和 benchmark 视频。

仓库中的标准配置保持 expected_world_size: 8。脚本检测到 8 张可见 GPU 时直接使用标准配置；检测到 4 张时，会用 OmegaConf 在 /tmp 生成一份临时运行配置，只把 expected_world_size 改成 4，训练结束后自动清理，标准 YAML 不会被改写。除 4 卡和 8 卡外的数量会直接报错。

## 2. 另一台机器上的前置条件

需要在训练机准备以下内容：

1. 项目代码 checkout 到任意目录。脚本会根据自身位置自动识别项目根目录，也可以用 OMNIFORCING_PROJECT_ROOT 显式指定。
2. 4 张或 8 张可用 CUDA GPU。未设置 CUDA_VISIBLE_DEVICES 时，机器有至少 8 张卡会默认使用 0,1,2,3,4,5,6,7；机器正好有 4 张卡会使用 0,1,2,3。若只想使用 B200 的后四张卡，必须显式设置 CUDA_VISIBLE_DEVICES=4,5,6,7。
3. 训练环境压缩包 omniforcing-conda.tar.zst。默认路径是 /data/minghua/zzy/omniforcing-conda.tar.zst；其他机器应通过 OMNIFORCING_ENV_ARCHIVE 指定实际路径。
4. LTX-2.3 FP8 基础模型：

   checkpoints/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors

5. Gemma 文本编码器目录：

   checkpoints/gemma-3-12b-it-qat-q4_0-unquantized/

这两个基础模型没有放在 step1 的 ode_lmdb/ode_pairs 目录中，脚本不会静默下载几十 GB 的模型。应从已有训练机复制，或按项目使用的模型来源单独准备。

## 3. 凭据准备

在项目根目录创建两个本地文件，文件中只放 token 本身，不要带说明文字：

~~~text
.hf_token
.wandb_key
~~~

也可以在启动前设置环境变量：

~~~bash
export HF_TOKEN="<有读取 step1 数据和上传 step2 checkpoint 权限的 token>"
export WANDB_API_KEY="<WandB API key>"
~~~

Hugging Face token 需要能够读取 aaachier/OmniForcing-backup，并向同一个仓库写入 step2 checkpoint。WandB key 需要能够写入配置中的 OmniForcing project 和 OmniForcing1 entity。

HPC 节点连接 WandB 可能比普通机器慢。当前代码在
`LTX-2/packages/ltx-distillation/src/ltx_distillation/ode/train_ode.py` 中将
`wandb_init_timeout` 传给 `wandb.Settings(init_timeout=...)`，标准配置为 300
秒。若确实需要更长时间，可以在启动前设置：

~~~bash
export WANDB_INIT_TIMEOUT=600
~~~

不要用 `WANDB_MODE=offline` 绕过错误；offline run 不满足本项目的 WandB 视频上传要求。

脚本默认使用：

~~~bash
export HF_ENDPOINT=https://hf-mirror.com
~~~

如果训练机需要直连 Hugging Face，启动前设置：

~~~bash
export HF_ENDPOINT=https://huggingface.co
~~~

## 4. step1 数据怎样下载

通常不需要手工下载。运行 train_step2_manual.sh 时，脚本会检查：

~~~text
LTX-2/packages/ltx-distillation/ode_pairs/
LTX-2/packages/ltx-distillation/ode_lmdb/
~~~

如果目录不存在或不完整，就从下面两个 Hugging Face 子目录下载：

~~~text
aaachier/OmniForcing-backup/ode_pairs/
aaachier/OmniForcing-backup/ode_lmdb/
~~~

脚本不是只检查 data.mdb 和 lock.mdb 是否存在。它还会检查：

- 8 个 .pt 文件的编号、manifest、producer 和 teacher checkpoint；
- 轨迹中必须同时包含 video、audio 和 sigmas；
- denoising schedule 必须是 1000、909、725、421、0；
- 视频配置必须是 121 帧、512×768；
- LMDB 内部 manifest、样本数、shape、prompt 和 ode_pairs 必须一致；
- sigma 必须有限、单调下降并最终到 0。

这能识别 B200 上“文件看起来完整、实际来自旧错误实验”的 ode_lmdb。

如果本地数据校验失败，脚本会先把 HF 上的新 ode_pairs 和 ode_lmdb 下载到临时目录，并完整校验新数据。只有新数据校验成功后，旧目录才会被移出训练路径。旧数据不会直接永久删除，而会备份到：

~~~text
/data/minghua/zzy/OmniForcing.step1-invalid/<UTC时间戳-进程号>/
~~~

其中可能包含旧的 ode_pairs/ 和 ode_lmdb/。随后脚本把正确的新数据放回训练路径并再次校验。若 HF 下载或新数据校验失败，原来的旧数据保持不动，训练也不会启动。

因此 agent 不要在下载成功前直接 rm -rf 旧 ode_lmdb，也不要把 B200 上旧的 ode_lmdb 复制进新代码目录。等新数据通过校验并且 step2 已正常启动后，再决定是否删除 .step1-invalid 中的备份。

如果只想准备并验证 step1 数据、不启动训练：

~~~bash
STEP2_PREPARE_ONLY=1 bash train_step2_manual.sh
~~~

这个 prepare-only 模式需要环境、基础模型和 HF/WandB 凭据，但不要求当前机器必须有 4/8 张 GPU，也不会启动 torchrun。

## 5. B200 上已有旧版本时怎么处理

不要直接删除整个 /data/minghua/zzy/OmniForcing。旧目录里可能还有约 28GB 的 LTX-2.3 FP8 模型、约 23GB 的 Gemma、token 和日志。也不要在不检查工作区的情况下直接 pull。

Agent 先执行：

~~~bash
cd /data/minghua/zzy/OmniForcing
git remote -v
git status --short
git fetch origin main
~~~

### 情况 A：tracked 代码没有本地修改

如果 git status 只显示被忽略的数据/输出，或者 tracked 代码完全干净，可以直接快进更新：

~~~bash
git pull --ff-only origin main
~~~

旧的 ode_lmdb 和 ode_pairs 不需要手工删除。运行最新版 train_step2_manual.sh 后，脚本会验证；错误版本会自动移到 /data/minghua/zzy/OmniForcing.step1-invalid/，再从 HF 下载正确版本。

### 情况 B：tracked 代码有旧的错误修改或 pull 冲突

不要执行 git reset --hard，也不要删除旧目录。采用可恢复的全新 clone：

~~~bash
cd /data/minghua/zzy
OLD="/data/minghua/zzy/OmniForcing_old_$(date +%Y%m%d_%H%M%S)"
mv /data/minghua/zzy/OmniForcing "$OLD"
git clone https://github.com/AIGeeksGroup/TurboGen.git /data/minghua/zzy/OmniForcing
~~~

在同一文件系统上，可以用硬链接复用旧目录中的大模型，避免额外占用约 51GB：

~~~bash
mkdir -p /data/minghua/zzy/OmniForcing/checkpoints/LTX-2.3-fp8
mkdir -p /data/minghua/zzy/OmniForcing/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized
cp -al "$OLD/checkpoints/LTX-2.3-fp8/." /data/minghua/zzy/OmniForcing/checkpoints/LTX-2.3-fp8/
cp -al "$OLD/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized/." /data/minghua/zzy/OmniForcing/checkpoints/gemma-3-12b-it-qat-q4_0-unquantized/
cp "$OLD/.hf_token" /data/minghua/zzy/OmniForcing/.hf_token
cp "$OLD/.wandb_key" /data/minghua/zzy/OmniForcing/.wandb_key
chmod 600 /data/minghua/zzy/OmniForcing/.hf_token /data/minghua/zzy/OmniForcing/.wandb_key
~~~

不要从 OLD 复制下面两个目录：

~~~text
LTX-2/packages/ltx-distillation/ode_pairs/
LTX-2/packages/ltx-distillation/ode_lmdb/
~~~

让最新一键脚本从 aaachier/OmniForcing-backup 自动下载正确版本。环境压缩包默认在仓库外的 /data/minghua/zzy/omniforcing-conda.tar.zst，改名旧仓库不会影响它。

## 6. 一键启动

先进入项目根目录，并按训练机实际情况设置环境压缩包路径：

~~~bash
cd /path/to/OmniForcing
export OMNIFORCING_ENV_ARCHIVE=/path/to/omniforcing-conda.tar.zst
~~~

未设置 CUDA_VISIBLE_DEVICES 时，脚本会优先使用 8 卡：检测到至少 8 张物理卡时选择 0,1,2,3,4,5,6,7；机器正好只有 4 张卡时选择 0,1,2,3。

如果 B200 有 8 张物理卡、但只允许使用后四张，必须显式写：

~~~bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
~~~

如果训练机本来就只暴露 4 张卡，可以不设置；也可以显式写：

~~~bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
~~~

本项目在 B200 上的确切一键训练脚本是：

~~~text
/data/minghua/zzy/OmniForcing/train_step2_manual.sh
~~~

启动命令：

~~~bash
bash /data/minghua/zzy/OmniForcing/train_step2_manual.sh 2>&1 | tee /data/minghua/zzy/OmniForcing/step2.log
~~~

脚本会依次完成：

1. 检查 LTX-2.3 FP8、Gemma、HF token 和 WandB key，并解压 conda 环境到 /tmp/omniforcing-conda；
2. 下载并严格验证 step1 的 ode_pairs 和 ode_lmdb；
3. 检查当前可见 GPU 数量必须为 8 或 4，并据此设置 NUM_GPUS 和 NPROC_PER_NODE；
4. 8 卡时使用仓库标准配置，4 卡时生成 expected_world_size=4 的临时运行配置；
5. 用 torchrun 启动 8 个或 4 个训练进程。

## 7. 推荐的 tmux 后台启动

如果 B200 有 8 张物理卡并要求使用后四张，让 coding agent 执行：

~~~bash
cd /data/minghua/zzy/OmniForcing
tmux new-session -d -s omniforcing-step2 'cd /data/minghua/zzy/OmniForcing && CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./train_step2_manual.sh 2>&1 | tee ./step2.log'
~~~

查看训练：

~~~bash
tmux attach -t omniforcing-step2
~~~

在 tmux 中按 Ctrl-B，再按 D 可以退出但保持训练。也可以直接查看日志：

~~~bash
tail -f /data/minghua/zzy/OmniForcing/step2.log
~~~

启动后，Agent 至少要确认：

~~~bash
tmux ls
tail -n 100 /data/minghua/zzy/OmniForcing/step2.log
nvidia-smi
~~~

后四卡运行时，日志应出现 STEP1_VALIDATE valid、Visible GPUs: 4,5,6,7、GPU processes: 4 和 Starting Step 2 with 4 GPUs on one node。默认八卡运行时则应显示 0,1,2,3,4,5,6,7、GPU processes: 8。看到 torchrun 对应的 4 个或 8 个 rank 开始加载模型后，才算一键脚本真正启动成功。

## 8. 每 500 step 会发生什么

当前配置明确设置了：

~~~yaml
save_iters: 500
visualize_iters: 500
benchmark_iters: 500
wandb_video_required: true
hf_upload_required: true
hf_upload_blocking: true
~~~

因此完成第 500、1000、1500……个训练 step 时：

1. 保存一个包含 generator 权重、optimizer 状态、数据迭代器状态和 RNG 状态的 checkpoint；
2. 将 checkpoint 同步上传到 aaachier/OmniForcing-backup/step2_ltx23/checkpoint_000500/，后续 step 会对应 checkpoint_001000/、checkpoint_001500/ 等目录；
3. 使用当前 generator 做一次固定 prompt 的 causal benchmark 推理，默认对 8 条 prompt 生成样本；
4. 将 benchmark 视频，以及训练可视化中的 pred_video，上传到 WandB；
5. 同时把视频保存到本地 outputs/stage2_causal_ode/ 下对应的 benchmark/ 和 visualizations/ 目录。

训练循环的顺序是先保存并完成 HF checkpoint 上传，然后处理当前训练样本的 WandB 可视化，再运行 benchmark 推理并记录 WandB。HF checkpoint 上传设置为 blocking，上传失败会让训练停止并在日志中报错，不会假装已经保存成功。wandb_video_required=true 也会在视频无法生成或上传时报错。

注意：当前 hf_upload_benchmark: false，所以 benchmark 视频上传到 WandB 和本地，但不会额外上传到 Hugging Face；上传到 Hugging Face 的是 step2 checkpoint。

## 9. 产物和检查方法

本地训练输出默认在：

~~~text
LTX-2/packages/ltx-distillation/outputs/stage2_causal_ode/
~~~

重点检查：

~~~text
checkpoint_000500/model.pt
checkpoint_000500/trainer_state.json
checkpoint_000500/trainer_state_rank_00000.pt
visualizations/step_000500/pred_video.mp4
benchmark/step_0000500/sample_0.mp4
diagnostics.log
hf_upload.log
~~~

日志中出现以下信息可以确认对应环节执行了：

~~~text
Model saved to .../checkpoint_000500/model.pt
[HF_UPLOAD] uploaded .../checkpoint_000500 to aaachier/OmniForcing-backup/step2_ltx23/checkpoint_000500
[WANDB_VIDEO] logged ... training video(s) at step 500
[WANDB_VIDEO] logged ... benchmark video(s) at step 500
[Benchmark] Step 500: ... saved to .../benchmark/step_0000500
~~~

## 10. 中断后精确恢复

如果要从本地完整 checkpoint 继续训练：

~~~bash
env RESUME_CHECKPOINT=/path/to/outputs/stage2_causal_ode/checkpoint_000500/model.pt bash train_step2_manual.sh 2>&1 | tee step2_resume.log
~~~

精确恢复必须沿用 checkpoint 保存时的 GPU 拓扑：四卡 checkpoint 仍用四卡和 trainer_state_rank_00000.pt 到 trainer_state_rank_00003.pt；八卡 checkpoint 仍用八卡和 trainer_state_rank_00000.pt 到 trainer_state_rank_00007.pt。同时应保持其余配置一致。仅有 model.pt 时只能做权重 warm start，不能恢复 optimizer 和数据迭代位置；不要把 step2 checkpoint 当成 step1 的 ODE 数据目录。

## 11. 这次准备完成后还需要什么

如果项目代码、基础模型、Gemma、环境压缩包和凭据已经放到另一台机器基本只需要把本 Markdown 交给 coding agent，让它按第 3～7 节检查旧目录、更新代码并后台启动。

严格来说不是“只需要一个 Markdown 文件”：Markdown 是操作说明，真正能一键运行还依赖修改后的 train_step2_manual.sh、默认八卡配置，以及训练机上的模型、环境和凭据。step1 的两个结果不需要再手工复制，入口脚本会从 aaachier/OmniForcing-backup/ode_pairs 和 aaachier/OmniForcing-backup/ode_lmdb 自动下载。

