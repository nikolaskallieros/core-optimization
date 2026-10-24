"""Exact lifecycle work admission, using no neutron transport or native geometry."""

import pytest

from fuel_cycle.runner import CycleError, CycleRunner
from fuel_cycle.refuelling import Bundle, CoreState, RefuellingSchedule, RingSchedule
from test_cycle import FixtureDepleter, operator_fixture, setup, tensor


def two_rings(*, first=(None, None), max_dt=1.5, max_steps=100):
    """Create two ring schedules with different periods and stack depths.

    Parameters
    ----------
    first : tuple, optional
        Initial event ticks for the two rings; None uses each ring's period.
    max_dt : float, optional
        Maximum depletion step in seconds.
    max_steps : int, optional
        Maximum number of depletion segments.

    Returns
    -------
    tuple
        CycleRunner, initial CoreState and fresh inventory/history keyword arguments.
    """
    rings = (
        RingSchedule("central", ("c",), 2, 2, first[0]),
        RingSchedule("radial", ("a", "b"), 3, 3, first[1]),
    )
    schedule = RefuellingSchedule(rings, integer_clock=True)
    stacks = {channel: tuple(
        Bundle(f"{channel}-{level}", None, {"nuclides": tensor([2, 0])},
               {"energy_j": tensor(0), "initial_hm_kg": tensor(1)})
        for level in range(ring.levels)
    ) for ring in rings for channel in ring.channels}
    runner = CycleRunner(
        FixtureDepleter([2] * 8), schedule, ("c", "a", "b"),
        seconds_per_time_unit=1, max_dt_seconds=max_dt, max_steps=max_steps,
    )
    recipes = dict(
        fresh_inventory_by_ring={r.name: {"nuclides": tensor([2, 0])} for r in rings},
        fresh_history_by_ring={r.name: {"energy_j": tensor(0), "initial_hm_kg": tensor(1)}
                               for r in rings},
    )
    return runner, CoreState(stacks), recipes


def assert_executed_plan(runner, state, recipes, start, end, expected):
    """Compare exact planned work with a completed fixture run.

    Parameters
    ----------
    runner : CycleRunner
        Runner whose plan and execution are compared.
    state : CoreState
        Initial resident stacks and event counts.
    recipes : mapping
        Fresh inventory/history keyword arguments for the run.
    start : int
        Starting schedule tick.
    end : int
        Final schedule tick.
    expected : mapping
        Expected step, event, stage, CRAM and discharge counts.

    Returns
    -------
    CycleResult
        Executed result after all planned-count assertions pass.
    """
    counts_before = dict(state.event_counts)
    plan = runner.plan_run(state, start_time=start, end_time=end)
    assert plan == dict(start_time=start, end_time=end, **expected)
    assert dict(state.event_counts) == counts_before
    calls = []
    result = runner.run(state, start_time=start, end_time=end,
                        stage_operator=operator_fixture(records=calls), **recipes)
    assert result.steps == plan["steps"]
    assert len(calls) == len(result.snapshots) == plan["stage_evaluations"]
    assert sum(s.stage == "post-refuelling" for s in result.snapshots) == plan["post_refuelling_calls"]
    assert len(result.discharges) == plan["discharged_bundles"]
    assert {r.name: result.final_state.event_counts.get(r.name, 0) - counts_before.get(r.name, 0)
            for r in runner.schedule.rings} == plan["event_counts_by_ring"]
    return result


def counts(steps, events, discharged, rings):
    """Construct expected event-aligned work counts.

    Parameters
    ----------
    steps : int
        Expected depletion segments.
    events : int
        Expected atomic refuelling events.
    discharged : int
        Expected physical discharged bundles.
    rings : mapping of str to int
        Expected event count per ring.

    Returns
    -------
    dict
        Complete expected work-plan counters.
    """
    return dict(steps=steps, post_refuelling_calls=events, discharged_bundles=discharged,
                stage_evaluations=3 * steps + events + int(steps == events == 0),
                cram_propagations=2 * steps, event_counts_by_ring=rings)


def test_ties_are_atomic_but_discharge_counts_each_physical_channel():
    """Verify ties are atomic but discharge counts each physical channel."""
    runner, state, recipes = two_rings()
    # Events 2, 3, 4, 6 split gaps 2, 1, 1, 2 into 2+1+1+2 steps.
    result = assert_executed_plan(runner, state, recipes, 0, 6,
        counts(6, 4, 7, {"central": 3, "radial": 2}))
    assert [s.time_seconds.item() for s in result.snapshots
            if s.stage == "post-refuelling"] == [2, 3, 4, 6]
    # The two- and three-bundle stacks affect residence, not event frequency.
    assert len(result.final_state.stacks["c"]) == 2
    assert len(result.final_state.stacks["a"]) == 3


def test_restart_returns_increments_and_excludes_already_applied_start_event():
    """Verify restart returns increments and excludes already applied start event."""
    runner, state, recipes = two_rings()
    first = runner.run(state, start_time=0, end_time=6,
                       stage_operator=operator_fixture(), **recipes)
    assert_executed_plan(runner, first.final_state, recipes, 6, 12,
        counts(6, 4, 7, {"central": 3, "radial": 2}))


@pytest.mark.parametrize("end, expected", [
    (6, counts(0, 1, 3, {"central": 1, "radial": 1})),
    (8, counts(2, 2, 4, {"central": 2, "radial": 1})),
])
def test_pending_atomic_start_event_is_included_once(end, expected):
    """Verify pending atomic start event is included once.

    Parameters
    ----------
    end : int
        Requested terminal tick of the synthetic refuelling schedule.
    expected : dict
        Exact expected depletion, event, discharge and ring-transition counts.
    """
    runner, initial, recipes = two_rings()
    pending = CoreState(initial.stacks, {"central": 2, "radial": 1})
    result = assert_executed_plan(runner, pending, recipes, 6, end, expected)
    assert result.snapshots[0].stage == "post-refuelling"
    assert result.snapshots[0].time_seconds.item() == 6


@pytest.mark.parametrize("end, expected", [
    (6, counts(3, 0, 0, {"central": 0, "radial": 0})),
    (12, counts(7, 4, 7, {"central": 3, "radial": 2})),
])
def test_delayed_first_events_do_not_assume_a_steady_forcing_cycle(end, expected):
    """Verify delayed first events do not assume a steady forcing cycle.

    Parameters
    ----------
    end : int
        Requested terminal tick of the synthetic refuelling schedule.
    expected : dict
        Exact expected depletion, event, discharge and ring-transition counts.
    """
    runner, state, recipes = two_rings(first=(7, 8), max_dt=2.5)
    # Events through 12 are 7, 8, 9, 11 (both rings at 11).
    assert_executed_plan(runner, state, recipes, 0, end, expected)


def test_zero_duration_without_pending_event_has_one_initial_snapshot():
    """Verify zero duration without pending event has one initial snapshot."""
    runner, state, recipes = setup()
    result = assert_executed_plan(runner, state, recipes, 0, 0,
                                 counts(0, 0, 0, {"r": 0}))
    assert result.snapshots[0].stage == "initial"
    advanced = runner.run(state, start_time=0, end_time=2,
                          stage_operator=operator_fixture(), **recipes)
    assert_executed_plan(runner, advanced.final_state, recipes, 2, 2,
                        counts(0, 0, 0, {"r": 0}))


@pytest.mark.parametrize("scale, max_dt, expected_steps", [
    (0.07, 0.01, 7), (0.21, 0.07, 3), (0.3, 0.1, 3), (0.31, 0.1, 4),
])
def test_decimal_time_conversion_uses_exact_segment_ceilings(scale, max_dt, expected_steps):
    """Verify decimal time conversion uses exact segment ceilings.

    Parameters
    ----------
    scale : float
        Seconds represented by one integer schedule tick.
    max_dt : float
        Maximum permitted depletion interval in seconds.
    expected_steps : int
        Exact number of depletion intervals required by the synthetic clock.
    """
    runner, state, recipes = setup(interval=10, clock_scale=scale, max_dt=max_dt)
    assert_executed_plan(runner, state, recipes, 0, 1,
                        counts(expected_steps, 0, 0, {"r": 0}))


def test_cycle_boundary_splits_steps_even_without_a_refuelling_event():
    """Verify cycle boundary splits steps even without a refuelling event."""
    runner, state, recipes = setup(interval=100, max_dt=1.5)
    whole = runner.plan_run(state, start_time=0, end_time=3)
    first = runner.run(state, start_time=0, end_time=1,
                       stage_operator=operator_fixture(), **recipes)
    second = runner.plan_run(first.final_state, start_time=1, end_time=3)
    assert whole["steps"] == 2
    assert first.steps + second["steps"] == 3


@pytest.mark.parametrize("per_run_limit", [None, 4])
def test_overbudget_rejects_before_stage_depletion_or_refuelling(monkeypatch, per_run_limit):
    """Verify overbudget rejects before stage depletion or refuelling.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped replacements for dependencies or environment settings.
    per_run_limit : int or None
        Requested run-specific depletion-step budget; None uses the runner limit.
    """
    runner, state, recipes = two_rings(max_steps=5 if per_run_limit is None else 100)
    pending = CoreState(state.stacks, {"central": 2, "radial": 1})
    def forbidden(*args, **kwargs):
        """Fail if an over-budget plan evaluates physics or shuffles fuel.

        Parameters
        ----------
        *args : object
            Positional arguments accepted only to detect an unexpected call.
        **kwargs : object
            Keyword arguments accepted only to detect an unexpected call.
        """
        pytest.fail("An infeasible work plan must not evaluate physics or shuffle bundles")
    monkeypatch.setattr(runner.depleter, "step", forbidden)
    monkeypatch.setattr(runner.schedule, "apply_event", forbidden)
    with pytest.raises(CycleError, match="step budget"):
        runner.run(pending, start_time=6, end_time=12, stage_operator=forbidden,
                   max_steps=per_run_limit, **recipes)
    assert dict(pending.event_counts) == {"central": 2, "radial": 1}


def test_exact_budget_is_admitted_and_larger_per_run_budget_is_rejected():
    """Verify exact budget is admitted and larger per run budget is rejected."""
    runner, state, recipes = two_rings(max_steps=6)
    assert runner.plan_run(state, start_time=0, end_time=6, max_steps=6)["steps"] == 6
    with pytest.raises(ValueError, match="cannot exceed"):
        runner.plan_run(state, start_time=0, end_time=6, max_steps=7)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_per_run_limits_rejected(limit):
    """Verify invalid per run limits rejected.

    Parameters
    ----------
    limit : object
        Invalid depletion-step limit that must be rejected.
    """
    runner, state, _ = setup()
    with pytest.raises(ValueError, match="positive integer"):
        runner.plan_run(state, start_time=0, end_time=1, max_steps=limit)


@pytest.mark.parametrize("start, end", [(-1, 0), (2, 1), (0.0, 1), (False, 1)])
def test_invalid_clock_requests_rejected(start, end):
    """Verify invalid clock requests rejected.

    Parameters
    ----------
    start : object
        Invalid or out-of-order initial schedule tick.
    end : int
        Requested terminal tick of the synthetic refuelling schedule.
    """
    runner, state, _ = setup()
    with pytest.raises(ValueError):
        runner.plan_run(state, start_time=start, end_time=end)


def test_restart_cannot_skip_an_event_or_claim_a_future_completed_event():
    """Verify restart cannot skip an event or claim a future completed event."""
    runner, state, _ = setup()
    with pytest.raises(CycleError, match="skipped an earlier"):
        runner.plan_run(state, start_time=3, end_time=4)
    future = CoreState(state.stacks, {"r": 1})
    with pytest.raises(CycleError, match="later than"):
        runner.plan_run(future, start_time=1, end_time=2)


def test_restart_cannot_split_a_simultaneous_ring_event():
    """Verify restart cannot split a simultaneous ring event."""
    runner, state, _ = two_rings()
    partial_tie = CoreState(state.stacks, {"central": 3, "radial": 1})
    with pytest.raises(ValueError, match="prior or simultaneous"):
        runner.plan_run(partial_tie, start_time=6, end_time=8)
