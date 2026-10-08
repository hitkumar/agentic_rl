"""Train the search agent with SkyRL: GRPO, synchronous, with training and vLLM colocated on the same GPUs.

SkyRL's standard PPO entrypoint, with SearchGenerator in place of its SkyRL-Gym generator. Config overrides are
SkyRL's (key=value on the command line). generator.max_input_length is the context length, the cap on prompt plus
responses.

One override is ours, not SkyRL's: search.format_penalty=<p> subtracts p from a training trajectory's reward if any
of its tool calls has an off-form header, as in Jasper's format-penalty run (p = 0.1). Default 0, off.
"""

import os
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from search_agent.generator import SearchGenerator

FORMAT_PENALTY_ARG = "search.format_penalty="


class SearchExp(BasePPOExp):
    def __init__(self, cfg: SkyRLTrainConfig, format_penalty: float):
        self.format_penalty = format_penalty
        super().__init__(cfg)

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return SearchGenerator(
            inference_engine_client, context_length=cfg.generator.max_input_length, format_penalty=self.format_penalty
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig, format_penalty: float) -> None:
    # SkyRL's TensorBoard logger writes to $TENSORBOARD_DIR, relative to this process's working directory by default.
    os.environ.setdefault("TENSORBOARD_DIR", os.path.join(cfg.trainer.log_path, "tensorboard", cfg.trainer.run_name))
    SearchExp(cfg, format_penalty).run()


def main() -> None:
    # SkyRL rejects config keys it doesn't know, so take ours out first.
    format_penalty = 0.0
    overrides = []
    for arg in sys.argv[1:]:
        if arg.startswith(FORMAT_PENALTY_ARG):
            format_penalty = float(arg.removeprefix(FORMAT_PENALTY_ARG))
        else:
            overrides.append(arg)
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    validate_cfg(cfg)

    # SkyRL's initialize_ray sets the job's runtime environment itself; add the setup hook to it.
    ray_init = ray.init

    def init_with_setup_hook(*args, runtime_env: dict, **kwargs):
        return ray_init(
            *args,
            runtime_env=runtime_env | {"worker_process_setup_hook": "search_agent.skyrl_patches.setup_worker"},
            **kwargs,
        )

    ray.init = init_with_setup_hook
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg, format_penalty))


if __name__ == "__main__":
    main()
