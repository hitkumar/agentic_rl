"""SkyRL generator: runs search trajectories on SkyRL's inference engines and returns them for training.

Each training step, SkyRL's trainer calls `generate` with a batch of dataset rows (each repeated
n_samples_per_prompt times for GRPO) and trains on the returned token IDs, loss masks, logprobs and rewards. Each
row runs `run_trajectory`, the same loop the evals use, with a sampler backed by the trainer's colocated vLLM
engines instead of an HTTP server.

The query and facts come from env_extras, the dataset's extra columns; the chat-format `prompt` column is unused,
since the prompt is rendered with Harmony. An optional `corpus` column picks the index the tools search (see
run_trajectory), so evals can search a different corpus than training.
"""

import asyncio
import json
from collections import Counter

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInput, InferenceEngineInterface
from skyrl.train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput
from skyrl.train.generators.utils import get_rollout_metrics

from search_agent.rewards import score
from search_agent.trajectory import STOP_TOKEN_IDS, Sample, Trajectory, run_trajectory, tool_pool


class EngineSampler:
    """Samples one trajectory's turns from SkyRL's inference engines, token IDs in and out.

    Every turn carries the trajectory's session ID, so the router sends all of its turns to the same engine and they
    reuse its prefix cache.
    """

    def __init__(self, client: InferenceEngineInterface, sampling_params: dict, session_id: str):
        self.client = client
        self.session_id = session_id
        self.sampling_params = sampling_params | {
            "stop_token_ids": STOP_TOKEN_IDS,
            "skip_special_tokens": False,
            # 0 returns only the sampled token's logprob.
            "logprobs": 0,
        }

    async def __call__(self, token_ids: list[int], max_tokens: int) -> Sample:
        output = await self.client.generate(
            InferenceEngineInput(
                prompts=None,
                prompt_token_ids=[token_ids],
                sampling_params=self.sampling_params | {"max_tokens": max_tokens},
                session_ids=[self.session_id],
                mm_features=None,
                cache_salt=None,
            )
        )
        logprobs = output["response_logprobs"]
        if logprobs is None:
            raise RuntimeError("inference engine returned no logprobs")
        return Sample(output["response_ids"][0], logprobs[0], output["stop_reasons"][0])


class SearchGenerator(GeneratorInterface):
    def __init__(self, inference_engine_client: InferenceEngineInterface, context_length: int):
        self.client = inference_engine_client
        self.context_length = context_length
        self.tools = tool_pool()

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        assert env_extras is not None and trajectory_ids is not None
        sampling_params = input_batch["sampling_params"] or {}

        async def run(extras: dict, session_id: str) -> Trajectory:
            sampler = EngineSampler(self.client, sampling_params, session_id)
            try:
                return await run_trajectory(
                    sampler,
                    self.tools,
                    extras["query"],
                    json.loads(extras["facts"]),
                    self.context_length,
                    extras.get("corpus") or "",
                )
            finally:
                await self.client.finish_session(session_id)

        trajectories = await asyncio.gather(
            *(run(extras, trajectory_id.to_string()) for extras, trajectory_id in zip(env_extras, trajectory_ids))
        )

        responses = [t.response_ids for t in trajectories]
        rewards = [t.reward for t in trajectories]
        loss_masks = [t.loss_mask for t in trajectories]
        metrics = get_rollout_metrics(responses, rewards, loss_masks=loss_masks)
        f1s = [
            score(t.session.curated_ids, json.loads(e["facts"]), beta=1.0).f_beta
            for t, e in zip(trajectories, env_extras)
        ]
        metrics["search/f1"] = sum(f1s) / len(f1s)
        metrics["search/turns"] = sum(t.turns for t in trajectories) / len(trajectories)
        metrics["search/nothing_curated"] = sum(not t.session.curated_ids for t in trajectories) / len(trajectories)
        for stop_reason, count in Counter(t.stop_reason for t in trajectories).items():
            metrics[f"search/stop_{stop_reason}"] = count / len(trajectories)

        return {
            "prompt_token_ids": [t.prompt_ids for t in trajectories],
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": [t.stop_reason for t in trajectories],
            "rollout_metrics": metrics,
            "rollout_logprobs": [t.logprobs for t in trajectories],
            "trajectory_ids": trajectory_ids,
            "trajectory_generation_times": None,
            "rollout_expert_indices": None,
            "is_last_step": None,
            "env_metrics": None,
            "pixel_values": None,
            "image_grid_thw": None,
        }
