"""Small integration fixtures, not a reactor calculation or substitute solver."""

from types import SimpleNamespace

import pytest
import torch

from fuel_cycle.runner import CycleError, CycleRunner, bundle_burnup, compare_cycle_states
from depletion.integrators import DepletionStage, DepletionStep
from fuel_cycle.refuelling import Bundle, CoreState, RefuellingSchedule, RingSchedule


def tensor(value, **kwargs):
    """Create a float64 tensor for the analytic cycle fixtures.

    Parameters
    ----------
    value : array_like
        Numerical fixture values.
    **kwargs : object
        Additional torch.tensor options such as requires_grad.

    Returns
    -------
    torch.Tensor
        Float64 fixture tensor.
    """
    return torch.tensor(value, dtype=torch.float64, **kwargs)


class FixtureDepleter:
    """Exact two-state transfer under frozen rates; reproduces CE/CM call order."""
    inventory_kind = "atoms_per_cm3"
    rate_nuclides = ("A", "B")
    reactions = ("transfer",)
    names = ("A", "B")
    settings = SimpleNamespace(target_keff=1.0, criticality_tolerance=1e-8)

    def __init__(self, volumes):
        """Bind constant zone volumes to the two-nuclide test depleter.

        Parameters
        ----------
        volumes : array_like
            Zone volumes in cm³.
        """
        self.volumes_cm3 = tensor(volumes)

    def step(self, inventory, time_seconds, dt_seconds, stage_operator):
        """Advance the analytic two-nuclide fixture with CE/CM staging.

        Parameters
        ----------
        inventory : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        time_seconds : torch.Tensor
            Absolute starting time in seconds.
        dt_seconds : torch.Tensor
            Interval duration in seconds.
        stage_operator : callable
            Fixture callback returning rates and diagnostics for an inventory and time.

        Returns
        -------
        DepletionStep
            Corrected endpoint inventory and all three evaluated stage records.
        """
        def advance(n, rates, dt):
            """Transfer parent loss to a stable daughter analytically.

            Parameters
            ----------
            n : torch.Tensor
                Manufactured zone-by-nuclide atom densities in atoms/cm³.
            rates : torch.Tensor
                Zone reaction frequencies in s⁻¹; the first parent rate drives the transfer.
            dt : torch.Tensor
                Propagation duration in seconds.

            Returns
            -------
            torch.Tensor
                Updated two-nuclide densities preserving their zone totals.
            """
            a = n[:, 0] * torch.exp(-rates[:, 0, 0] * dt)
            return torch.stack((a, n.sum(dim=1) - a), dim=1)
        start = stage_operator(inventory, time_seconds)
        half = advance(inventory, start["rates"], dt_seconds / 2)
        middle = stage_operator(half, time_seconds + dt_seconds / 2)
        final = advance(inventory, middle["rates"], dt_seconds)
        end = stage_operator(final, time_seconds + dt_seconds)
        stages = (
            DepletionStage("beginning", time_seconds, inventory, start),
            DepletionStage("midpoint", time_seconds + dt_seconds / 2, half, middle),
            DepletionStage("end", time_seconds + dt_seconds, final, end),
        )
        return DepletionStep(final, time_seconds + dt_seconds, stages,
                             self.inventory_kind, self.volumes_cm3, 0)


def setup(*, interval=2, levels=2, max_dt=1.5, max_steps=100, clock_scale=1):
    """Create a one-ring cycle fixture with explicit work limits.

    Parameters
    ----------
    interval : int, optional
        Ring displacement interval in integer clock ticks.
    levels : int, optional
        Number of axial bundle positions.
    max_dt : float, optional
        Maximum depletion step in seconds.
    max_steps : int, optional
        Maximum number of depletion segments.
    clock_scale : float, optional
        Seconds per schedule tick.

    Returns
    -------
    tuple
        CycleRunner, initial CoreState and fresh inventory/history keyword arguments.
    """
    schedule = RefuellingSchedule(
        (RingSchedule("r", ("c",), levels, interval),), integer_clock=True,
    )
    state = CoreState({"c": tuple(
        Bundle(f"initial-{i}", None, {"nuclides": tensor([2, 0])},
               {"energy_j": tensor(0), "initial_hm_kg": tensor(1)})
        for i in range(levels)
    )})
    runner = CycleRunner(
        FixtureDepleter([2] * levels), schedule, ("c",),
        seconds_per_time_unit=clock_scale, max_dt_seconds=max_dt, max_steps=max_steps,
    )
    recipes = dict(
        fresh_inventory_by_ring={"r": {"nuclides": tensor([5, 0])}},
        fresh_history_by_ring={"r": {"energy_j": tensor(0), "initial_hm_kg": tensor(1)}},
    )
    return runner, state, recipes


def operator_fixture(*, scale=None, poison=False, records=None):
    """Build a controlled stage callback for cycle tests.

    Parameters
    ----------
    scale : torch.Tensor or None, optional
        Live zone-power multiplier; None uses unity.
    poison : bool, optional
        Enable manufactured composition-dependent rates and power.
    records : list or None, optional
        Optional list receiving every evaluated time and inventory.

    Returns
    -------
    callable
        Stage callback returning manufactured rates, powers and criticality diagnostics.
    """
    scale = tensor(1) if scale is None else scale
    def operator(inventory, time):
        """Evaluate the fixture's rates, powers and boron response.

        Parameters
        ----------
        inventory : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        time : torch.Tensor
            Absolute stage time in seconds.

        Returns
        -------
        dict
            Manufactured stage outputs with consistent total and zone power.
        """
        if records is not None:
            records.append((time, inventory))
        powers = scale * torch.arange(1, len(inventory) + 1, dtype=torch.float64) * 100
        rates = torch.zeros((*inventory.shape, 1), dtype=torch.float64)
        if poison:
            rates = torch.stack((0.1 * (1 + inventory[:, 1]), inventory[:, 0] * 0), dim=1).unsqueeze(-1)
            powers = powers * (1 + inventory[:, 1])
        return dict(
            rates=rates, power_w=powers.sum(), zone_power_w=powers,
            keff=tensor(1), boron_ppm=10 + inventory[:, 1].sum(),
        )
    return operator


@pytest.mark.parametrize("metadata", [
    {"nuclide_order": ("B", "A")},
    {"reaction_order": ("wrong",)},
    {"physical_zone_volumes_cm3": tensor([2, 3])},
])
def test_coupled_metadata_rejects_reordered_rates_and_inconsistent_volumes(metadata):
    """Verify coupled metadata rejects reordered rates and inconsistent volumes.

    Parameters
    ----------
    metadata : dict
        Deliberately inconsistent transport ordering or volume metadata.
    """
    runner, state, recipes = setup()
    base = operator_fixture()
    def mismatched(inventory, time):
        """Attach deliberately inconsistent ordering or volume metadata.

        Parameters
        ----------
        inventory : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        time : torch.Tensor
            Absolute stage time in seconds.

        Returns
        -------
        dict
            Otherwise valid fixture stage with the selected invalid metadata.
        """
        return {**base(inventory, time), **metadata}
    with pytest.raises(CycleError, match="disagree"):
        runner.run(state, start_time=0, end_time=1, stage_operator=mismatched, **recipes)


def test_event_alignment_energy_inventory_conservation_and_fresh_reset():
    """Verify event alignment energy inventory conservation and fresh reset."""
    runner, state, recipes = setup()
    records = []
    result = runner.run(
        state, start_time=0, end_time=5, stage_operator=operator_fixture(records=records), **recipes,
    )
    assert result.steps == 5
    assert len(records) == 3 * result.steps + 2
    assert [s.time_seconds.item() for s in result.snapshots if s.stage == "post-refuelling"] == [2, 4]
    endpoints = [s.time_seconds.item() for s in result.snapshots if s.stage == "end"]
    assert endpoints == [1.5, 2, 3.5, 4, 5]
    assert len(result.discharges) == 2
    energies = sum(b.history["energy_j"] for b in result.final_state.stacks["c"])
    removed_energy = sum(item.bundle.history["energy_j"] for item in result.discharges)
    assert (energies + removed_energy).item() == pytest.approx(300 * 5)
    atoms = sum(b.inventory["nuclides"].sum() * 2 for b in result.final_state.stacks["c"])
    removed_atoms = sum(item.bundle.inventory["nuclides"].sum() * 2 for item in result.discharges)
    assert (atoms + removed_atoms).item() == pytest.approx(2 * 2 * 2 + 2 * 5 * 2)
    for snapshot in result.snapshots:
        assert snapshot.inventory.shape == (2, 2)
        torch.testing.assert_close(torch.cat(snapshot.axial_power_w), snapshot.zone_power_w)
        if snapshot.stage == "post-refuelling":
            assert snapshot.state.stacks["c"][0].history["energy_j"].item() == 0
            torch.testing.assert_close(snapshot.inventory[0], tensor([5, 0]))
            assert snapshot.bundle_ids[0] != snapshot.bundle_ids[1]
    assert result.discharges[0].bundle.id == "initial-1"
    assert result.discharges[1].bundle.id == "initial-0"
    assert state.stacks["c"][0].history["energy_j"].item() == 0


def test_midpoint_poison_callback_and_energy_are_recomputed_once():
    """Verify midpoint poison callback and energy are recomputed once."""
    runner, state, recipes = setup(interval=10, max_dt=2)
    records = []
    result = runner.run(
        state, start_time=0, end_time=2,
        stage_operator=operator_fixture(poison=True, records=records), **recipes,
    )
    assert [time.item() for time, _ in records] == [0, 1, 2]
    assert (records[1][1][:, 1] > 0).all()
    middle = next(s for s in result.snapshots if s.stage == "midpoint")
    final = result.snapshots[-1]
    torch.testing.assert_close(final.energy_j, middle.zone_power_w * 2)
    torch.testing.assert_close(final.inventory.sum(dim=1), tensor([2, 2]))
    frozen_a = 2 * torch.exp(tensor(-0.1 * 2))
    assert (final.inventory[:, 0] < frozen_a).all()
    assert final.boron_ppm > result.snapshots[0].boron_ppm


def test_clock_conversion_and_burnup_units_are_correct():
    """Verify clock conversion and burnup units are correct."""
    runner, state, recipes = setup(interval=10, levels=1, clock_scale=86400, max_dt=86400)
    result = runner.run(
        state, start_time=0, end_time=1, stage_operator=operator_fixture(scale=tensor(10)), **recipes,
    )
    assert result.snapshots[-1].time_seconds.item() == 86400
    assert bundle_burnup(result.final_state.stacks["c"][0]).item() == pytest.approx(1)
    assert result.snapshots[-1].burnup_mwd_per_thm.item() == pytest.approx(1)


def test_graph_survives_energy_accumulation_shuffle_and_discharge():
    """Verify graph survives energy accumulation shuffle and discharge."""
    runner, state, recipes = setup()
    scale = tensor(2, requires_grad=True)
    result = runner.run(
        state, start_time=0, end_time=5, stage_operator=operator_fixture(scale=scale), **recipes,
    )
    total = sum(b.history["energy_j"] for b in result.final_state.stacks["c"])
    total = total + sum(d.bundle.history["energy_j"] for d in result.discharges)
    gradient, = torch.autograd.grad(total, scale)
    assert gradient.item() == pytest.approx(1500)


def test_post_event_criticality_failure_prevents_success_even_at_final_time():
    """Verify post event criticality failure prevents success even at final time."""
    runner, state, recipes = setup(max_dt=2)
    base = operator_fixture()
    def operator(n, t):
        """Inject a subcritical value after the final refuelling event.

        Parameters
        ----------
        n : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        t : torch.Tensor
            Absolute stage time in seconds.

        Returns
        -------
        dict
            Fixture stage output with the targeted criticality failure.
        """
        data = base(n, t)
        if t == 2 and n[0, 0] == 5:
            data["keff"] = tensor(0.99)
        return data
    with pytest.raises(CycleError, match="Post-refuelling solve failed"):
        runner.run(state, start_time=0, end_time=2, stage_operator=operator, **recipes)
    assert state.event_counts == {}


def test_trial_cycle_carries_deficit_through_refuelling_without_claiming_criticality():
    """Verify trial cycle carries deficit through refuelling without claiming criticality."""
    runner, state, recipes = setup(max_dt=2)
    runner.depleter.settings = SimpleNamespace(target_keff=1., criticality_tolerance=1e-8,
                                              allow_subcritical_trials=True)
    k = tensor(.96, requires_grad=True)
    base = operator_fixture()
    def operator(n, t):
        """Return an explicitly labelled zero-boron subcritical trial.

        Parameters
        ----------
        n : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        t : torch.Tensor
            Absolute stage time in seconds.

        Returns
        -------
        dict
            Fixture rates and power with live trial keff and control_status.
        """
        return {**base(n, t), 'keff': k, 'boron_ppm': tensor(0.), 'control_status': 'subcritical_trial'}
    result = runner.run(state, start_time=0, end_time=2, stage_operator=operator, **recipes)
    assert len(result.discharges) == 1
    assert result.snapshots[-1].stage == 'post-refuelling'
    assert all(s.keff is k and s.boron_ppm == 0 for s in result.snapshots)
    deficit = sum(torch.clamp_min(1. - s.keff, 0.) for s in result.snapshots)
    assert torch.autograd.grad(deficit, k)[0].item() == -len(result.snapshots)


def test_invalid_zone_power_total_and_step_limit_fail():
    """Verify invalid zone power total and step limit fail."""
    runner, state, recipes = setup(max_steps=1)
    with pytest.raises(CycleError, match="step budget"):
        runner.run(state, start_time=0, end_time=5, stage_operator=operator_fixture(), **recipes)
    base = operator_fixture()
    def operator(n, t):
        """Make total power disagree with the sum of zone powers.

        Parameters
        ----------
        n : torch.Tensor
            Manufactured zone-by-nuclide atom densities in atoms/cm³.
        t : torch.Tensor
            Absolute stage time in seconds.

        Returns
        -------
        dict
            Fixture stage with one extra watt in its total-power field.
        """
        data = base(n, t)
        data["power_w"] = data["power_w"] + 1
        return data
    with pytest.raises(CycleError, match="do not sum"):
        runner.run(state, start_time=0, end_time=1, stage_operator=operator, **recipes)


def test_periodic_forcing_does_not_imply_equilibrium_but_changed_ids_allow_it():
    """Verify periodic forcing does not imply equilibrium but changed ids allow it."""
    runner, state, recipes = setup(max_dt=2)
    recipes["fresh_inventory_by_ring"]["r"]["nuclides"] = tensor([2, 0])
    arguments = dict(
        start_time=0, max_cycle_ticks=2, inventory_atol=0, energy_atol_j=0,
        burnup_atol=0, rtol=0, stage_operator=operator_fixture(), **recipes,
    )
    with pytest.raises(CycleError, match="did not converge"):
        runner.run_until_equilibrium(state, max_cycles=1, **arguments)
    equilibrium = runner.run_until_equilibrium(state, max_cycles=3, **arguments)
    assert equilibrium.cycles_completed == 2
    assert [check.converged for check in equilibrium.checks] == [False, True]
    assert equilibrium.checks[0].max_energy_difference_j.item() == 200
    assert equilibrium.cycle.final_state.stacks["c"][0].id != "initial-0"
    assert equilibrium.cycle.end_time == 4


def test_equilibrium_requires_integer_clock_and_matching_phase():
    """Verify equilibrium requires integer clock and matching phase."""
    runner, state, recipes = setup(max_dt=2)
    final = runner.run(state, start_time=0, end_time=2, stage_operator=operator_fixture(), **recipes).final_state
    kwargs = dict(
        start_time=0, end_time=3, max_cycle_ticks=2, inventory_atol=0,
        energy_atol_j=0, burnup_atol=0, rtol=0,
    )
    with pytest.raises(ValueError, match="one common cycle"):
        compare_cycle_states(state, final, runner.schedule, ("c",), **kwargs)
    real_schedule = RefuellingSchedule((RingSchedule("r", ("c",), 2, 2.0),))
    with pytest.raises(ValueError, match="integer clock"):
        compare_cycle_states(state, final, real_schedule, ("c",), **kwargs)


def test_bad_fresh_history_and_volume_transfer_are_rejected():
    """Verify bad fresh history and volume transfer are rejected."""
    runner, state, recipes = setup()
    recipes["fresh_history_by_ring"]["r"]["energy_j"] = tensor(1)
    with pytest.raises(CycleError, match="zero accumulated energy"):
        runner.run(state, start_time=0, end_time=2, stage_operator=operator_fixture(), **recipes)
    recipes["fresh_history_by_ring"]["r"]["energy_j"] = tensor(0)
    runner.depleter.volumes_cm3 = tensor([1, 2])
    with pytest.raises(CycleError, match="volumes must be equal"):
        runner.run(state, start_time=0, end_time=2, stage_operator=operator_fixture(), **recipes)


def test_restart_cannot_skip_unapplied_event():
    """Verify restart cannot skip unapplied event."""
    runner, state, recipes = setup()
    with pytest.raises(CycleError, match="skipped an earlier"):
        runner.run(state, start_time=3, end_time=4, stage_operator=operator_fixture(), **recipes)


def test_explicit_channel_order_and_atomic_multiring_event():
    """Verify explicit channel order and atomic multiring event."""
    schedule = RefuellingSchedule((
        RingSchedule("a", ("x",), 1, 2), RingSchedule("b", ("y",), 2, 3),
    ), integer_clock=True)
    state = CoreState({
        channel: tuple(Bundle(f"{channel}-{i}", None, {"nuclides": tensor([2, 0])},
                              {"energy_j": tensor(0), "initial_hm_kg": tensor(1)})
                       for i in range(levels))
        for channel, levels in (("x", 1), ("y", 2))
    })
    runner = CycleRunner(FixtureDepleter([2, 2, 2]), schedule, ("y", "x"),
                         seconds_per_time_unit=1, max_dt_seconds=5, max_steps=10)
    result = runner.run(
        state, start_time=0, end_time=6, stage_operator=operator_fixture(),
        fresh_inventory_by_ring={name: {"nuclides": tensor([5, 0])} for name in ("a", "b")},
        fresh_history_by_ring={name: {"energy_j": tensor(0), "initial_hm_kg": tensor(1)} for name in ("a", "b")},
    )
    assert result.snapshots[0].bundle_ids == ("y-0", "y-1", "x-0")
    posts = [s for s in result.snapshots if s.stage == "post-refuelling"]
    assert [s.time_seconds.item() for s in posts] == [2, 3, 4, 6]
    assert len([d for d in result.discharges if d.time == 6]) == 2
    total = sum(b.history["energy_j"] for stack in result.final_state.stacks.values() for b in stack)
    total += sum(d.bundle.history["energy_j"] for d in result.discharges)
    assert total.item() == pytest.approx(600 * 6)


def test_restart_cannot_run_a_future_state_backwards():
    """Verify restart cannot run a future state backwards."""
    runner, state, recipes = setup()
    after = runner.run(state, start_time=0, end_time=2, stage_operator=operator_fixture(), **recipes)
    with pytest.raises(CycleError, match="later than its starting time"):
        runner.run(after.final_state, start_time=1, end_time=3, stage_operator=operator_fixture(), **recipes)


def test_equilibrium_enforces_one_total_step_budget():
    """Verify equilibrium enforces one total step budget."""
    runner, state, recipes = setup(max_dt=2, max_steps=1)
    with pytest.raises(CycleError, match="total timestep budget"):
        runner.run_until_equilibrium(
            state, start_time=0, max_cycles=3, max_cycle_ticks=2,
            inventory_atol=0, energy_atol_j=0, burnup_atol=0, rtol=0,
            stage_operator=operator_fixture(), **recipes,
        )
