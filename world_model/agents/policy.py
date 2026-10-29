"""Exact expected entropy reduction for binary sensors over finite hidden states.

The policy receives model likelihoods, posterior beliefs, and acquisition state.
There is no adapter, simulator, seed of the environment, or unseen reading here.
"""

from hashlib import sha256
from math import fsum, log2

from pydantic import TypeAdapter

from world_model.core import ModelArtifact
from world_model.core.serialization import canonical_json
from world_model.core.types import Distribution
from world_model.core.types import FrozenMap, Identifier, Probability
from .contracts import Candidate, Decision, InvestigationConfig

_LIKELIHOODS = TypeAdapter(FrozenMap[Identifier, FrozenMap[Identifier, Probability]])


def entropy(probabilities) -> float:
    return -fsum(p * log2(p) for p in probabilities if p > 0)


def information_gain(posterior, likelihood) -> float:
    """I(state; reading) = H(reading) - E_state H(reading | state), in bits."""
    alert = fsum(posterior[state] * likelihood[state] for state in posterior)
    conditional = fsum(posterior[state] * entropy((likelihood[state], 1 - likelihood[state])) for state in posterior)
    return max(0.0, entropy((alert, max(0.0, 1 - alert))) - conditional)


def validate_config(artifact: ModelArtifact, config: InvestigationConfig):
    limits = artifact.spec.acquisition_limits
    if set(config.allowed_sources) - set(artifact.spec.observation_models):
        raise ValueError("allowed sources must exist in the model")
    if config.horizon > limits.episode_origin_ticks or config.budget_credits > limits.episode_budget_credits or config.max_reads > limits.max_reads_per_episode:
        raise ValueError("investigation exceeds the registered model's episode, credit, or read limits")


def choose(artifact: ModelArtifact, config: InvestigationConfig, posterior, *, tick: int,
           spent: int, reads: int, used_sources: tuple[str, ...] = (), sensor_likelihoods=None) -> Decision:
    validate_config(artifact, config)
    posterior = TypeAdapter(Distribution).validate_python(posterior)
    if set(posterior) != set(artifact.spec.states):
        raise ValueError("posterior support must match the model")
    if sensor_likelihoods is not None:
        sensor_likelihoods = _LIKELIHOODS.validate_python(sensor_likelihoods)
        if set(sensor_likelihoods) != set(artifact.spec.observation_models) or any(set(row) != set(posterior) for row in sensor_likelihoods.values()):
            raise ValueError("conditional sensor likelihoods must cover all declared sensors and states")
    if any(type(v) is not int or v < 0 for v in (tick, spent, reads)) or spent > config.budget_credits or reads > config.max_reads:
        raise ValueError("invalid acquisition counters")
    released = min(config.budget_credits, config.budget_credits * (tick + 1) // config.horizon)
    if spent > released:
        raise ValueError("spending exceeds the released credits")
    available = released - spent
    uncertainty = entropy(posterior.values())
    candidates = []
    for source in sorted(config.allowed_sources):
        sensor = artifact.spec.observation_models[source]
        gain = information_gain(posterior, sensor.alert_probability if sensor_likelihoods is None else sensor_likelihoods[source])
        reason = "eligible"
        if source in used_sources:
            reason = "already dispatched this tick"
        elif reads >= config.max_reads:
            reason = "read limit reached"
        elif sensor.read_cost_credits > available:
            reason = "insufficient released credits"
        elif gain <= 1e-15 or gain / sensor.read_cost_credits <= config.min_gain_bits_per_credit:
            reason = "below information gain threshold"
        candidates.append(Candidate(source_id=source, cost_credits=sensor.read_cost_credits,
            information_gain_bits=gain, gain_bits_per_credit=gain / sensor.read_cost_credits,
            eligible=reason == "eligible", reason=reason))
    base = dict(tick=tick, entropy_bits=uncertainty, released_credits=released,
                available_credits=available, candidates=tuple(candidates))
    if tick >= config.horizon:
        return Decision(**base, kind="stop", reason="episode horizon reached")
    if uncertainty <= config.entropy_threshold_bits:
        return Decision(**base, kind="advance_tick", reason="uncertainty threshold reached")
    eligible = [item for item in candidates if item.eligible]
    if config.policy == "information_gain":
        selected = max(eligible, key=lambda item: (item.gain_bits_per_credit, -item.cost_credits), default=None)
        reason = "maximum expected information gain per credit"
    else:
        # Fixed and random baselines request at most one source each tick. An
        # unaffordable choice saves its credits, rather than silently choosing
        # the cheap source and making all policies identical under pacing.
        source = config.fixed_source
        if config.policy == "random":
            sources = sorted(config.allowed_sources)
            digest = sha256(canonical_json(["random-sensor-v1", config.policy_seed, tick])).digest()
            source = sources[int.from_bytes(digest[:8], "big") % len(sources)]
        selected = next((item for item in eligible if item.source_id == source), None) if not used_sources else None
        reason = "fixed sensor schedule" if config.policy == "fixed" else "independent random sensor schedule"
    if selected is None:
        return Decision(**base, kind="advance_tick", reason="no permitted worthwhile read within the current schedule and limits")
    return Decision(**base, kind="read", source_id=selected.source_id, cost_credits=selected.cost_credits, reason=reason)
