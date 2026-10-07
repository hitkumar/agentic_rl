#!/usr/bin/env bash
# Train gpt-oss-20b on the search task with GRPO on 8 GPUs: full fine-tuning, synchronous, with training and vLLM
# colocated on the same GPUs.
#
# Based on SkyRL's gpt-oss example (examples/train/gptoss/run_gsm8k_gptoss.sh). Follows Jasper's run where it applies:
# 64 queries x 8 rollouts per step, advantages centered within each group (no std normalization), no KL. He trained a
# LoRA with LR 1e-4; this is full fine-tuning (SkyRL's LoRA for gpt-oss would skip the MoE experts), so the LR is
# 3e-6, the best of the LR sweep in results.md.
#
# The loss matches Tinker's importance_sampling loss, which he used: -sum over generated tokens of
# (p_theta / q_sampler) * advantage, summed over tokens rather than averaged. rollout_is has the same gradient; the clip
# range is opened so no token is dropped. seq_mean_token_sum_norm sums over tokens and sequences and divides by the
# constant batch size x max_seq_len, which Adam cancels. rollout_is also skips the trainer's forward pass for the old
# logprobs, since it uses vLLM's.
#
# Run from the repo root. Extra key=value arguments override the defaults below, e.g. a one-step smoke test on a
# 4-query train file:
#   bash search_agent/train.sh data.train_data="['outputs/search_agent/smoke/train.parquet']" \
#     trainer.train_batch_size=4 trainer.policy_mini_batch_size=4 generator.n_samples_per_prompt=4 trainer.epochs=1 \
#     trainer.eval_before_train=false trainer.eval_interval=-1 trainer.ckpt_interval=-1 trainer.run_name=smoke
#
# Metrics go to TensorBoard under outputs/search_agent/logs/tensorboard/<run_name>, e.g. the vLLM vs trainer logprob
# gap, policy/minibatch_rollout_logprobs_abs_diff_mean:
#   .venv/bin/tensorboard --logdir outputs/search_agent/logs/tensorboard
set -euo pipefail

DATA_DIR=search_agent/data
CONTEXT_LENGTH=30720
OUT_DIR="$PWD/outputs/search_agent"

# Ray workers import search_agent, so the repo root must be on their path.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Start Ray with 127.0.0.2 as the node address. On a host without IPv4, Ray's default node address is IPv6, which
# breaks SkyRL's vLLM servers: it puts the address in URLs without brackets, and binds the servers to IPv4 0.0.0.0.
# Ray maps 127.0.0.1 back to the IPv6 address but leaves the rest of 127.0.0.0/8, which is loopback too. It must
# bypass the HTTP proxy like 127.0.0.1 does.
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.2" NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.2"
# Keep torch.distributed's sockets on loopback too; everything runs on this one machine.
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
# Without this, importing flashinfer (through vLLM) initializes CUDA to query the GPUs' architecture. vLLM's Ray
# workers assign their GPUs only after starting, which then has no effect, so both vLLM engines land on GPUs 0-3.
# 8.0 is the A100.
export FLASHINFER_CUDA_ARCH_LIST=8.0
# Same for transformers, which calls torch.cuda.is_available() when the Ray setup hook imports gpt-oss: by default
# that initializes CUDA. This makes torch check through NVML instead, which doesn't.
export PYTORCH_NVML_BASED_CUDA_CHECK=1
.venv/bin/ray start --head --node-ip-address=127.0.0.2 --include-dashboard=false --disable-usage-stats
trap '.venv/bin/ray stop' EXIT
export RAY_ADDRESS=auto

# The venv's Python, not `uv run`: under `uv run`, Ray builds a fresh uv environment for each worker from a copy of
# the repo, which lacks the local SkyRL checkout.
.venv/bin/python -u -m search_agent.train \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/dev.parquet']" \
  environment.env_class=search \
  trainer.policy.model.path=unsloth/gpt-oss-20b-BF16 \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.engine_init_kwargs.max_model_len=$CONTEXT_LENGTH \
  generator.batched=false \
  generator.max_turns=40 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.eps_clip_low=1.0 \
  trainer.algorithm.eps_clip_high=1.0e9 \
  trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
  trainer.algorithm.max_seq_len=$CONTEXT_LENGTH \
  trainer.policy.optimizer_config.lr=3.0e-6 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=20 \
  generator.n_samples_per_prompt=8 \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=$CONTEXT_LENGTH \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.max_generate_length=2048 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.max_generate_length=2048 \
  generator.eval_n_samples_per_prompt=4 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=true \
  trainer.eval_interval=4 \
  trainer.ckpt_interval=8 \
  trainer.max_ckpts_to_keep=2 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$OUT_DIR/checkpoints" \
  trainer.export_path="$OUT_DIR/exports" \
  trainer.log_path="$OUT_DIR/logs" \
  trainer.logger=tensorboard \
  trainer.project_name=search_agent \
  trainer.run_name=grpo_64x8 \
  "$@"
