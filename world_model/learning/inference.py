"""Joint parameter/state filtering over a frozen Monte Carlo parameter bank.

Each particle contains static sensor likelihoods drawn from the labeled
calibration posterior. Filtering retains correlations between its parameters
and hidden state. This is exact for the finite bank, approximate for continuous
Beta uncertainty; ESS reports concentration and does not prove convergence.
"""

from hashlib import sha256

import numpy as np

from world_model.core import Reading
from .calibration import verify_revision

INFERENCE_VERSION = "beta-mixture-filter-v1"


def parameter_bank(artifact, revision, *, particles=256, seed=20261028):
    verify_revision(artifact, revision)
    if type(particles) is not int or not 2 <= particles <= 4096 or type(seed) is not int or seed < 0:
        raise ValueError("use 2..4096 particles and a nonnegative integer sampling seed")
    sources, states = tuple(sorted(artifact.spec.observation_models)), artifact.spec.states
    cells = {(c.source_id, c.state): c for c in revision.cells}
    alpha = np.array([[cells[source, state].alpha for state in states] for source in sources])
    beta = np.array([[cells[source, state].beta for state in states] for source in sources])
    bank = np.random.Generator(np.random.PCG64(seed)).beta(alpha, beta, size=(particles, len(sources), len(states)))
    bank.flags.writeable = False
    return bank


class ParameterFilter:
    def __init__(self, artifact, revision, *, particles=256, seed=20261028, samples=None):
        verify_revision(artifact, revision)
        self.artifact, self.revision = artifact, revision
        self.sources, self.states = tuple(sorted(artifact.spec.observation_models)), artifact.spec.states
        bank = parameter_bank(artifact, revision, particles=particles, seed=seed) if samples is None else np.array(samples, dtype=float, copy=True)
        if bank.ndim != 3 or bank.shape[1:] != (len(self.sources), len(self.states)) or len(bank) < 2 or not np.isfinite(bank).all() or np.any((bank < 0) | (bank > 1)):
            raise ValueError("invalid posterior parameter bank")
        bank.flags.writeable = False
        self.samples, self.particles = bank, len(bank)
        self.draws_hash = sha256(bank.astype("<f8").tobytes()).hexdigest()
        self._transition = np.array([[artifact.spec.transition_probabilities[s][t] for t in self.states] for s in self.states])
        self._joint = np.tile([artifact.spec.initial_distribution[s] for s in self.states], (self.particles, 1)) / self.particles
        self.tick = 0
        self._slots = {}

    @staticmethod
    def _distribution(states, vector):
        vector = vector / vector.sum()
        return {s: float(np.clip(p, 0, 1)) for s, p in zip(states, vector, strict=True)}

    def state(self):
        posterior = self._joint.sum(axis=0)
        predictive = posterior @ self._transition
        weights = self._joint.sum(axis=1)
        effective = {}
        for index, source in enumerate(self.sources):
            numerator = (self._joint * self.samples[:, index, :]).sum(axis=0)
            likelihood = np.divide(numerator, posterior, out=np.full_like(posterior, .5), where=posterior > 0)
            effective[source] = {s: float(np.clip(p, 0, 1)) for s, p in zip(self.states, likelihood, strict=True)}
        return (self._distribution(self.states, posterior), self._distribution(self.states, predictive),
                effective, float(1 / np.square(weights / weights.sum()).sum()))

    def condition(self, reading):
        reading = Reading.model_validate(reading)
        if reading.tick != self.tick or reading.source_id not in self.sources:
            raise ValueError("only a declared sensor at the current tick can be conditioned on")
        key = reading.tick, reading.source_id
        if key in self._slots:
            if self._slots[key] != reading.value:
                raise ValueError("conflicting sensor reading")
            return
        likelihood = self.samples[:, self.sources.index(reading.source_id), :]
        candidate = self._joint * (likelihood if reading.value else 1 - likelihood)
        total = candidate.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("reading has zero probability under the parameter bank")
        self._joint = candidate / total
        self._slots[key] = reading.value

    def advance(self):
        self._joint = self._joint @ self._transition
        self._joint /= self._joint.sum()
        self.tick += 1
