"""Verify acquisition decisions and accounting from recorded publications."""

from .policy import choose, validate_config


def verify_trace(artifact, config, steps, publications, observations, *, created_at, final_sequence):
    validate_config(artifact, config)
    spent = reads = sequence = 0
    used = {}
    previous_time = created_at
    stopped = False
    for ordinal, step in enumerate(steps):
        if stopped or step.ordinal != ordinal or step.before_sequence != sequence or step.recorded_at <= previous_time:
            raise ValueError("acquisition trace is not a contiguous ordered history")
        before = publications[sequence]
        tick = before.belief.state_tick
        expected = choose(artifact, config, before.belief.distribution, tick=tick, spent=spent, reads=reads, used_sources=tuple(used.get(tick, ())))
        if step.decision != expected or step.run_id != before.belief.run_id:
            raise ValueError("recorded acquisition decision does not match the policy")
        after = publications[step.after_sequence]
        if expected.kind == "stop":
            if step.action is not None or step.outcome is not None or step.after_sequence != sequence:
                raise ValueError("stopping cannot dispatch an action or revise a belief")
            stopped = True
        else:
            action, outcome = step.action, step.outcome
            if action is None or outcome is None:
                raise ValueError("dispatched action has no outcome")
            if (action.action_id, outcome.action_id, outcome.outcome_id) != (step.request_id,) * 3 or action.run_id != step.run_id or outcome.run_id != step.run_id:
                raise ValueError("acquisition identity mismatch")
            if (action.kind, action.source_id, action.tick, action.reserved_cost_credits) != (expected.kind, expected.source_id, tick, expected.cost_credits):
                raise ValueError("action differs from its decision")
            if outcome.charged_cost_credits != expected.cost_credits or outcome.completed_at != step.recorded_at or action.dispatched_at is None or not previous_time < action.created_at <= action.dispatched_at <= outcome.completed_at:
                raise ValueError("invalid acquisition timing or cost")
            spent += expected.cost_credits
            if expected.kind == "read":
                reads += 1
                used.setdefault(tick, []).append(expected.source_id)
            if outcome.status == "failed":
                if expected.kind != "read" or step.after_sequence != sequence or outcome.observation_id is not None:
                    raise ValueError("failed read changed the belief")
            elif outcome.status == "succeeded":
                if step.after_sequence != sequence + 1 or step.recorded_at != after.belief.published_at:
                    raise ValueError("successful action has no matching publication")
                if expected.kind == "read":
                    observation = observations[outcome.observation_id]
                    if (observation.run_id, observation.payload.source_id, observation.payload.tick, observation.event_sequence, observation.payload.measurement_id) != (step.run_id, expected.source_id, tick, step.after_sequence, step.request_id):
                        raise ValueError("reading provenance does not match dispatch")
                    if observation.payload.supersedes_id is not None or after.belief.state_tick != tick:
                        raise ValueError("sensor acquisition cannot correct or advance the world")
                elif after.belief.state_tick != tick + 1 or outcome.observation_id is not None:
                    raise ValueError("invalid advancement result")
            else:
                raise ValueError("dispatched actions must succeed or fail")
        if step.spent_credits != spent or step.dispatched_reads != reads or spent > config.budget_credits or reads > config.max_reads:
            raise ValueError("acquisition accounting mismatch")
        sequence, previous_time = step.after_sequence, step.recorded_at
    if sequence != final_sequence:
        raise ValueError("belief history contains an unaccounted acquisition")
    return len(steps)
