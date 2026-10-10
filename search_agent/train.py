"""Train the search agent with SkyRL: GRPO, synchronous, with training and vLLM colocated on the same GPUs.

SkyRL's standard PPO entrypoint, with SearchGenerator in place of its SkyRL-Gym generator. Config overrides are
SkyRL's (key=value on the command line). generator.max_input_length is the context length, the cap on prompt plus
responses.

Three overrides are ours, not SkyRL's, all training-reward shaping (see generator.py), default 0 (off):
  search.format_penalty=<p>       subtract p if any tool call has an off-form header (Jasper's format-penalty run: 0.1)
  search.discovery_bonus=<b>      add b x trajectory recall (Jasper's f4s: 0.2)
  search.curated_chunk_cost=<c>   subtract c per curated chunk (Jasper's f4s: 0.02)
"""

import os
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from search_agent.generator import SearchGenerator

SEARCH_ARGS = ("format_penalty", "discovery_bonus", "curated_chunk_cost")


class SearchExp(BasePPOExp):
    def __init__(self, cfg: SkyRLTrainConfig, search_args: dict[str, float]):
        self.search_args = search_args
        super().__init__(cfg)

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return SearchGenerator(
            inference_engine_client, context_length=cfg.generator.max_input_length, **self.search_args
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig, search_args: dict[str, float]) -> None:
    # SkyRL's TensorBoard logger writes to $TENSORBOARD_DIR, relative to this process's working directory by default.
    os.environ.setdefault("TENSORBOARD_DIR", os.path.join(cfg.trainer.log_path, "tensorboard", cfg.trainer.run_name))
    SearchExp(cfg, search_args).run()


def main() -> None:
    # SkyRL rejects config keys it doesn't know, so take ours out first.
    search_args = {}
    overrides = []
    for arg in sys.argv[1:]:
        key, _, value = arg.partition("=")
        if key.startswith("search.") and key.removeprefix("search.") in SEARCH_ARGS:
            search_args[key.removeprefix("search.")] = float(value)
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
    ray.get(skyrl_entrypoint.remote(cfg, search_args))


if __name__ == "__main__":
    main()
