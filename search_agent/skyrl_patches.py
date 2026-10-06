"""Workarounds for SkyRL v0.3.0 bugs, applied in every Ray worker process at startup.

Kept apart from train.py, which imports SkyRL: Ray runs the setup hook while starting a worker, and importing SkyRL
there crashes Ray.
"""

import functools
import importlib.abc
import importlib.util
import sys

from transformers import AttentionMaskInterface
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel
from vllm_router.router_args import RouterArgs


def setup_worker() -> None:
    # SkyRL creates gpt-oss with SDPA attention, then switches it to its own flex attention, but transformers rejects
    # SDPA for gpt-oss at creation. The model never runs with SDPA.
    GptOssPreTrainedModel._supports_sdpa = True

    # SkyRL's mask function for its flex attention requires cache_position, which transformers no longer passes. The
    # function doesn't use it.
    register_mask = AttentionMaskInterface.register

    def register_mask_with_default(key, value):
        if key == "custom_flex":
            value = functools.partial(value, cache_position=None)
        register_mask(key, value)

    AttentionMaskInterface.register = staticmethod(register_mask_with_default)

    # SkyRL brackets each weight sync with vLLM's layerwise reload, which moves the weights to the meta device and
    # collects the new ones through each parameter's weight loader. vLLM's gpt-oss loader copies the MoE expert weights
    # directly instead, so they're lost and the old ones kept. Without quantization there's nothing to process after
    # loading, so load straight into the weights. Patched on import: importing vLLM initializes CUDA, after which vLLM
    # can no longer assign the worker its GPUs.
    sys.meta_path.insert(0, PatchOnImport("vllm.model_executor.model_loader.reload", skip_layerwise_reload))

    # vllm-router 0.1.15 (pinned in pyproject.toml) dropped RouterArgs.pd_disaggregation, off by default, which SkyRL
    # still reads after starting the router.
    setattr(RouterArgs, "pd_disaggregation", False)

    # SkyRL's gpt-oss flex attention adds the attention sinks through a score_mod. Flex attention computes the
    # gradient of a tensor captured by a score_mod with atomic adds from every query-key pair into it, here into one
    # value per head, which makes the backward pass ~180x slower than without sinks (6k tokens on an A100). Replace it
    # with the same attention computed without sinks, with the sinks applied after from its logsumexp. Patched on
    # import: SkyRL imports the function from this module at use.
    sys.meta_path.insert(
        0, PatchOnImport("skyrl.backends.skyrl_train.patches.gptoss.flex_attn_sink", lse_sink_attention)
    )

    # SkyRL moves the optimizer state to the GPU with the model before the forward and backward passes, though only
    # the optimizer step uses it. Its Adam state, 21 GB per GPU here, then runs out of memory with the activations of a
    # 30k-token sequence. Move it to the GPU just before the optimizer step instead.
    sys.meta_path.insert(
        0, PatchOnImport("skyrl.backends.skyrl_train.workers.worker_dispatch", load_optimizer_for_step)
    )


def lse_sink_attention(module) -> None:
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    flex_attention = module.flex_attention  # SkyRL's compiled flex attention

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @functools.lru_cache
    def sliding_window_mask(window_size):
        # Each query attends to itself and the window_size - 1 tokens before it, like transformers and vLLM.
        def mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)

        return mask

    def with_padding(mask_mod, attention_mask):
        if attention_mask.ndim == 4:  # (batch, 1, q, kv), 0 where attended
            attended = attention_mask[:, 0] == 0
            return lambda b, h, q_idx, kv_idx: mask_mod(b, h, q_idx, kv_idx) & attended[b, q_idx, kv_idx]
        attended = attention_mask.bool()  # (batch, kv), 1 where attended
        return lambda b, h, q_idx, kv_idx: mask_mod(b, h, q_idx, kv_idx) & attended[b, kv_idx]

    def flex_attention_with_sink(
        query,
        key,
        value,
        attention_mask=None,
        scale=None,
        sliding_window=None,
        sinks=None,
        num_key_value_groups=1,
        **kwargs,
    ):
        """Same arguments and result as SkyRL's old_flex_attention_with_sink.

        The sink is an extra logit per head in each query's softmax, with a zero value vector, so it scales the
        attention output by exp(lse) / (exp(lse) + exp(sink)) = sigmoid(lse - sink), where lse is the logsumexp of
        the query's other logits.
        """
        assert sinks is not None  # SkyRL always passes gpt-oss's sinks.
        bsz, _, q_len, _ = query.shape
        kv_len = key.shape[2]
        if isinstance(sliding_window, int) and sliding_window != 0:
            mask_mod = sliding_window_mask(sliding_window)
        else:
            mask_mod = causal
        if attention_mask is not None:
            mask_mod = with_padding(mask_mod, attention_mask)
        # The mask is the same for every head: build it once for all of them.
        block_mask = create_block_mask(mask_mod, bsz, None, q_len, kv_len, device=key.device, _compile=True)
        out, lse = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=num_key_value_groups != 1,
            return_lse=True,
        )
        out = out * torch.sigmoid(lse - sinks.view(1, -1, 1)).unsqueeze(-1).to(out.dtype)
        return out.transpose(1, 2).contiguous()

    module.old_flex_attention_with_sink = flex_attention_with_sink


def load_optimizer_for_step(module) -> None:
    dispatch = module.WorkerDispatch

    def model_only(forward_backward):
        @functools.wraps(forward_backward)
        def wrapper(self, model, *args, **kwargs):
            self._ensure_on_gpu(model, need_optimizer=False, need_model=True)
            # forward_backward asks for the optimizer too: mark it loaded so it isn't, then restore its real state.
            state = self._gpu_state[model]
            optimizer_on_gpu, state.optimizer_on_gpu = state.optimizer_on_gpu, True
            try:
                return forward_backward(self, model, *args, **kwargs)
            finally:
                state.optimizer_on_gpu = optimizer_on_gpu

        return wrapper

    def with_optimizer(optim_step):
        @functools.wraps(optim_step)
        def wrapper(self, model, *args, **kwargs):
            self._ensure_on_gpu(model, need_optimizer=True, need_model=False)  # The model is there already.
            return optim_step(self, model, *args, **kwargs)

        return wrapper

    dispatch.forward_backward = model_only(dispatch.forward_backward)
    dispatch.forward_backward_from_staged = model_only(dispatch.forward_backward_from_staged)
    dispatch.optim_step = with_optimizer(dispatch.optim_step)


def skip_layerwise_reload(module) -> None:
    module.initialize_layerwise_reload = lambda *args, **kwargs: None
    module.finalize_layerwise_reload = lambda *args, **kwargs: None


class PatchOnImport(importlib.abc.MetaPathFinder):
    """Runs `patch` on a module right after it's first imported."""

    def __init__(self, name: str, patch):
        self.name = name
        self.patch = patch

    def find_spec(self, name, path, target=None):
        if name != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.loader is not None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            self.patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec
