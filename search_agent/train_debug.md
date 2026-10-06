# train.sh debugging log

Goal: GRPO training of gpt-oss-20b with SkyRL v0.3.0 on 8x A100, with training (FSDP over 8 GPUs) and vLLM (2 engines x
TP4) colocated. Status: trains and learns (see the overfit check in results.md). Eval, checkpointing and resume are not
yet tested.

Key fixes, for anyone setting this up again or upgrading SkyRL, vLLM or transformers. Smaller compatibility shims are
commented in `skyrl_patches.py`.

| Problem | Cause | Fix | Where |
|---|---|---|---|
| vLLM servers unreachable | Host has no IPv4, so Ray's node address is IPv6; SkyRL puts it unbracketed in URLs and binds the servers to IPv4 | Ray node address 127.0.0.2, added to `no_proxy` | train.sh |
| Patches don't reach Ray workers, or crash them | Patches must run in every worker; importing SkyRL in a Ray setup hook crashes Ray | `skyrl_patches.py`, which doesn't import SkyRL, as the `worker_process_setup_hook` | train.py, skyrl_patches.py |
| Both vLLM engines on GPUs 0-3 | vLLM assigns each worker its GPUs after start; our setup hook had already initialized CUDA, so the assignment had no effect | `PYTORCH_NVML_BASED_CUDA_CHECK=1`, `FLASHINFER_CUDA_ARCH_LIST=8.0` | train.sh |
| Weight sync silently keeps the old MoE expert weights in vLLM | SkyRL wraps the sync in vLLM's layerwise reload, which vLLM's gpt-oss expert loader bypasses | No-op the layerwise reload | skyrl_patches.py |
| Attention backward ~180x slower | SkyRL adds gpt-oss's attention sinks through a flex-attention score_mod, whose gradient is computed with atomic adds | Attention without sinks, then the sinks applied from the logsumexp | skyrl_patches.py |
| Generation stalls | Tool calls ran on the one event loop shared by all episodes; threads contend for the GIL | Pool of 32 processes (`tool_pool`); step-1 generation 818 s → 507 s | trajectory.py |
| Out of memory in the backward pass at step 3, on a 30k-token sequence | SkyRL loads the 21 GB/GPU Adam state before the forward and backward passes | Load it just before the optimizer step | skyrl_patches.py |
