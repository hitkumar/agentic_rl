"""Train the search agent with SkyRL: GRPO, synchronous, with training and vLLM colocated on the same GPUs.

SkyRL's standard PPO entrypoint, with SearchGenerator in place of its SkyRL-Gym generator. Config overrides are
SkyRL's (key=value on the command line). generator.max_input_length is the context length, the cap on prompt plus
responses.
"""

import os
import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from search_agent.generator import SearchGenerator


class SearchExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return SearchGenerator(inference_engine_client, context_length=cfg.generator.max_input_length)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig) -> None:
    # SkyRL's TensorBoard logger writes to $TENSORBOARD_DIR, relative to this process's working directory by default.
    os.environ.setdefault("TENSORBOARD_DIR", os.path.join(cfg.trainer.log_path, "tensorboard", cfg.trainer.run_name))
    SearchExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
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
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
