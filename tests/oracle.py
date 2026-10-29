"""Independent, deliberately slow reference inference for small test worlds.

Enumerate joint hidden paths using exact rational arithmetic. This module does
not import the production filter, and forecasts enumerate an extra hidden state.
"""

import json
from fractions import Fraction
from itertools import product


def enumerate_distribution(content: bytes, readings, target_tick: int):
    model = json.loads(content, parse_float=Fraction)
    states = model["states"]
    mass = {state: Fraction(0) for state in states}
    for path in product(states, repeat=target_tick + 1):
        weight = model["initial_distribution"][path[0]]
        for tick in range(1, len(path)):
            weight *= model["transition_probabilities"][path[tick - 1]][path[tick]]
        for reading in readings:
            probability = model["observation_models"][reading.source_id]["alert_probability"][path[reading.tick]]
            weight *= probability if reading.value else 1 - probability
        mass[path[-1]] += weight
    total = sum(mass.values())
    return {state: value / total for state, value in mass.items()}
