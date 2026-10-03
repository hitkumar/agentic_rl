"""Score a trajectory's curated set against a query's facts.

Adapted from jasper-lu/sec-search-rl (src/sec_rl/rewards.py).

A fact counts as found if any of its gold chunks is curated. Recall is the fraction of facts found, precision the
fraction of curated chunks that are gold for some fact. The reward is F4 (recall weighted 16x over precision), or
EMPTY_REWARD if nothing was curated.
"""

from dataclasses import dataclass

BETA = 4.0
# An empty curated set scores below a non-empty miss (F4 = 0), so curating nothing is never the safe option.
EMPTY_REWARD = -0.2


@dataclass(frozen=True)
class Score:
    precision: float
    recall: float
    f_beta: float


def score(curated_ids: list[str], facts: list[dict]) -> Score:
    """facts: the query's parsed facts, each with a "chunk_ids" list of interchangeable gold chunks."""
    curated = set(curated_ids)
    gold = {chunk_id for fact in facts for chunk_id in fact["chunk_ids"]}
    precision = len(curated & gold) / len(curated) if curated else 0.0
    recall = sum(bool(curated & set(fact["chunk_ids"])) for fact in facts) / len(facts)
    denominator = BETA**2 * precision + recall
    f_beta = (1 + BETA**2) * precision * recall / denominator if denominator else 0.0
    return Score(precision, recall, f_beta)


def reward(curated_ids: list[str], facts: list[dict]) -> float:
    return score(curated_ids, facts).f_beta if curated_ids else EMPTY_REWARD
