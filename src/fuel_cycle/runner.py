"""Event-aligned CE/CM lifecycle integration; no transport model is substituted.

Schedule times have a caller-declared seconds_per_time_unit. Transport/depletion
times are seconds; powers are W; carried energies are J. Each complete bundle
has one atoms/cm3 vector in inventory['nuclides'], an initial_hm_kg > 0, and
energy_j >= 0 in history. Fresh recipes must explicitly supply zero energy.
The clock/event branch is discrete; graphs through inventories, physical
coefficients, powers and carried energy are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from numbers import Integral, Real

import torch

from fuel_cycle.refuelling import Bundle, CoreState, Discharge, RefuellingSchedule
from depletion.integrators import admissible_powered_state


class CycleError(RuntimeError):
    """A failed lifecycle check; no partial result is returned as success."""


def _tensor(value, name):
    """Require a finite CPU float64 tensor without detaching its graph.

    Parameters
    ----------
    value : object
        Numerical value to validate.
    name : str
        Field label included in validation diagnostics.

    Returns
    -------
    torch.Tensor
        The validated input tensor.

    Raises
    ------
    CycleError
        The tensor has an unsupported device, precision or nonfinite values.
    """
    if value.dtype != torch.float64 or value.device.type != "cpu":
        raise CycleError(f"{name} must be a CPU float64 tensor")
    if not torch.isfinite(value).all():
        raise CycleError(f"{name} must be finite")
    return value


def _scalar(value, name):
    """Require a finite scalar on the CPU float64 calculation path.

    Parameters
    ----------
    value : object
        Numerical value to validate.
    name : str
        Field label included in validation diagnostics.

    Returns
    -------
    torch.Tensor
        Zero-dimensional tensor retaining any input gradient graph.
    """
    if value is None or isinstance(value, bool):
        raise CycleError(f"{name} must be supplied")
    result = _tensor(value, name) if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float64)
    if result.ndim or not torch.isfinite(result):
        raise CycleError(f"{name} must be a finite scalar")
    return result


def _clock(value, name):
    """Represent a supplied clock value by its exact decimal fraction.

    Parameters
    ----------
    value : object
        Numerical value to validate.
    name : str
        Field label included in validation diagnostics.

    Returns
    -------
    fractions.Fraction
        Exact rational representation used to order events.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite real clock data")
    return Fraction(str(value))


def _positive_int(value, name):
    """Require a positive integer without accepting boolean values.

    Parameters
    ----------
    value : object
        Numerical value to validate.
    name : str
        Field label included in validation diagnostics.

    Returns
    -------
    int
        Validated integer limit.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _tolerances(*values):
    """Validate finite, nonnegative convergence tolerances.

    Parameters
    ----------
    *values : float
        Absolute or relative tolerances to validate.
    """
    if any(
        isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x < 0
        for x in values
    ):
        raise ValueError("Equilibrium tolerances must be explicit finite nonnegative scalars")


def _phase_bounds(state, schedule, time, *, allow_pending_event):
    """Check that restart counts bracket the requested schedule time.

    Parameters
    ----------
    state : CoreState
        Resident bundle stacks and completed ring-event counts.
    schedule : RefuellingSchedule
        Ring schedules using one explicitly declared time unit.
    time : int or float
        Absolute time in schedule units.
    allow_pending_event : bool
        Whether the next atomic event may occur exactly at this time.

    Raises
    ------
    CycleError
        The state skips completed events or includes an event after this time.
    """
    completed = [
        _clock(ring.first_event, "first_event")
        + (state.event_counts[ring.name] - 1) * _clock(ring.interval, "interval")
        for ring in schedule.rings if state.event_counts.get(ring.name, 0)
    ]
    if completed and max(completed) > time:
        raise CycleError("State contains refuelling events later than its starting time")
    next_time = _clock(schedule.next_event(state).time, "next event")
    if next_time < time or (next_time == time and not allow_pending_event):
        raise CycleError("State skipped an earlier or phase-boundary refuelling event")


def _bundles(state, channel_order):
    """Flatten resident stacks in the explicit channel and axial order.

    Parameters
    ----------
    state : CoreState
        Resident bundle stacks and completed ring-event counts.
    channel_order : sequence of str
        Unique physical channel ordering; bundles within each channel run from top to
        bottom.

    Returns
    -------
    tuple of Bundle
        Resident bundles ordered by channel, then from top to bottom.
    """
    if len(set(channel_order)) != len(channel_order) or set(channel_order) != set(state.stacks):
        raise CycleError("channel_order must enumerate every core channel exactly once")
    return tuple(bundle for channel in channel_order for bundle in state.stacks[channel])


def _fields(state, channel_order):
    """Collect resident inventories, energy and initial heavy-metal mass.

    Parameters
    ----------
    state : CoreState
        Resident bundle stacks and completed ring-event counts.
    channel_order : sequence of str
        Unique physical channel ordering; bundles within each channel run from top to
        bottom.

    Returns
    -------
    inventory : torch.Tensor
        Zone-by-chain atom densities in atoms/cm³.
    energy : torch.Tensor
        Accumulated energy in J per resident bundle.
    mass : torch.Tensor
        Initial heavy-metal mass in kg per resident bundle.
    """
    bundles = _bundles(state, channel_order)
    inventories, energies, masses = [], [], []
    for bundle in bundles:
        inventory = _tensor(bundle.inventory["nuclides"], "bundle nuclides")
        if inventory.ndim != 1 or not inventory.numel() or (inventory < 0).any():
            raise CycleError("Bundle nuclides must be a nonnegative vector")
        inventories.append(inventory)
        energies.append(_scalar(bundle.history["energy_j"], "energy_j"))
        masses.append(_scalar(bundle.history["initial_hm_kg"], "initial_hm_kg"))
    inventory, energy, mass = torch.stack(inventories), torch.stack(energies), torch.stack(masses)
    if (energy < 0).any() or (mass <= 0).any():
        raise CycleError("Carried energy must be nonnegative and initial heavy metal positive")
    return inventory, energy, mass


def bundle_burnup(bundle: Bundle):
    """Convert carried energy to burnup per initial heavy-metal mass.

    Parameters
    ----------
    bundle : Bundle
        Physical bundle carrying nuclides, energy_j and initial_hm_kg.

    Returns
    -------
    torch.Tensor
        Scalar burnup in MWd/tHM, equal to energy_j / (86,400,000 * initial_hm_kg).
    """
    energy = _scalar(bundle.history["energy_j"], "energy_j")
    mass = _scalar(bundle.history["initial_hm_kg"], "initial_hm_kg")
    if energy < 0 or mass <= 0:
        raise CycleError("Invalid bundle energy or initial heavy-metal mass")
    return energy / (86_400_000 * mass)


def _replace_fields(state, channel_order, inventory, energy):
    """Replace resident inventories and energy while retaining identities.

    Parameters
    ----------
    state : CoreState
        Resident bundle stacks and completed ring-event counts.
    channel_order : sequence of str
        Unique physical channel ordering; bundles within each channel run from top to
        bottom.
    inventory : torch.Tensor
        CPU float64 atom densities in atoms/cm³, shaped (zones, chain_nuclides).
    energy : torch.Tensor
        Accumulated fission energy in J for each resident bundle.

    Returns
    -------
    CoreState
        New bundle objects with unchanged insertion times, initial masses and event counts.
    """
    stacks, index = {}, 0
    for channel in channel_order:
        updated = []
        for bundle in state.stacks[channel]:
            updated.append(replace(
                bundle, inventory={**bundle.inventory, "nuclides": inventory[index]},
                history={**bundle.history, "energy_j": energy[index]},
            ))
            index += 1
        stacks[channel] = tuple(updated)
    return CoreState(stacks, state.event_counts)


@dataclass(frozen=True)
class CycleSnapshot:
    """Retain ordered physical bundles and their transport response at one stage."""
    time_seconds: torch.Tensor
    stage: str
    state: CoreState
    channel_order: tuple[str, ...]
    zone_power_w: torch.Tensor
    boron_ppm: torch.Tensor | None
    keff: torch.Tensor | None

    @property
    def bundle_ids(self):
        """Return resident identities in channel and axial order.

        Returns
        -------
        tuple of str
            Bundle identifiers matching all zone-indexed snapshot fields.
        """
        return tuple(bundle.id for bundle in _bundles(self.state, self.channel_order))

    @property
    def inventory(self):
        """Return the snapshot's resident full-chain atom densities.

        Returns
        -------
        torch.Tensor
            Zone-by-chain inventory in atoms/cm³.
        """
        return _fields(self.state, self.channel_order)[0]

    @property
    def energy_j(self):
        """Return the energy carried by each resident bundle.

        Returns
        -------
        torch.Tensor
            Zone-ordered accumulated fission energy in J.
        """
        return _fields(self.state, self.channel_order)[1]

    @property
    def burnup_mwd_per_thm(self):
        """Return resident burnup per initial heavy-metal mass.

        Returns
        -------
        torch.Tensor
            Zone-ordered burnup in MWd/tHM.
        """
        _, energy, mass = _fields(self.state, self.channel_order)
        return energy / (86_400_000 * mass)

    @property
    def axial_power_w(self):
        """Group zone powers into top-to-bottom channel stacks.

        Returns
        -------
        tuple of torch.Tensor
            One axial power vector in W per channel, preserving channel_order.
        """
        lengths = [len(self.state.stacks[channel]) for channel in self.channel_order]
        return tuple(torch.split(self.zone_power_w, lengths))


@dataclass(frozen=True)
class CycleResult:
    """Collect the final core, retained stage snapshots and discharged bundles."""
    final_state: CoreState
    snapshots: tuple[CycleSnapshot, ...]
    discharges: tuple[Discharge, ...]
    steps: int
    start_time: float | int
    end_time: float | int


@dataclass(frozen=True)
class EquilibriumCheck:
    """Record matched-phase inventory, energy and burnup differences against explicit tolerances."""
    converged: bool
    cycle_ticks: int
    max_inventory_difference: torch.Tensor
    max_energy_difference_j: torch.Tensor
    max_burnup_difference: torch.Tensor


@dataclass(frozen=True)
class EquilibriumResult:
    """Return the last complete cycle and every convergence check used to stop."""
    cycle: CycleResult
    checks: tuple[EquilibriumCheck, ...]
    cycles_completed: int


def compare_cycle_states(
    before, after, schedule, channel_order, *, start_time, end_time,
    max_cycle_ticks, inventory_atol, energy_atol_j, burnup_atol, rtol,
):
    """Compare resident compositions and histories one exact cycle apart.

    Parameters
    ----------
    before : CoreState
        Resident bundle stacks and completed ring-event counts.
    after : CoreState
        Resident bundle stacks and completed ring-event counts.
    schedule : RefuellingSchedule
        Ring schedules using one explicitly declared time unit.
    channel_order : sequence of str
        Unique physical channel ordering; bundles within each channel run from top to
        bottom.
    start_time : int or float
        Absolute start in schedule units; integer clocks require explicit integer ticks.
    end_time : int or float
        Absolute end in the same schedule units as start_time.
    max_cycle_ticks : int
        Upper bound on the exact common integer-clock period.
    inventory_atol : float
        Absolute convergence tolerance in atoms/cm³.
    energy_atol_j : float
        Absolute convergence tolerance in J.
    burnup_atol : float
        Absolute convergence tolerance in MWd/t of initial heavy metal.
    rtol : float
        Dimensionless relative convergence tolerance.

    Returns
    -------
    EquilibriumCheck
        Convergence flag, exact cycle length and maximum inventory, energy and burnup
        differences.

    Notes
    -----
    Exact forcing periodicity requires explicit integer ticks. Bundle IDs and cumulative
    event numbers differ normally and are not convergence variables. Tolerances use
    atoms/cm3, J, MWd/t initial HM, and a dimensionless rtol.
    """
    period = schedule.common_cycle(max_ticks=max_cycle_ticks)
    if any(isinstance(t, bool) or not isinstance(t, Integral) for t in (start_time, end_time)):
        raise ValueError("Equilibrium phase times must be explicit integer ticks")
    if end_time - start_time != period or start_time < 0:
        raise ValueError("Compare states exactly one common cycle apart")
    _tolerances(inventory_atol, energy_atol_j, burnup_atol, rtol)
    schedule.next_event(before)
    schedule.next_event(after)
    _phase_bounds(before, schedule, Fraction(start_time), allow_pending_event=False)
    _phase_bounds(after, schedule, Fraction(end_time), allow_pending_event=False)
    for ring in schedule.rings:
        if after.event_counts.get(ring.name, 0) - before.event_counts.get(ring.name, 0) != period // ring.interval:
            raise CycleError("Compared states do not span a complete common-phase schedule")
    n0, e0, m0 = _fields(before, channel_order)
    n1, e1, m1 = _fields(after, channel_order)
    if n0.shape != n1.shape or not torch.equal(m0, m1):
        raise CycleError("Equilibrium comparison requires matching zones, chain and initial masses")
    if any(len(before.stacks[c]) != len(after.stacks[c]) for c in channel_order):
        raise CycleError("Equilibrium comparison requires matching axial positions")
    b0, b1 = e0 / (86_400_000 * m0), e1 / (86_400_000 * m1)
    converged = (
        torch.allclose(n0, n1, rtol=rtol, atol=inventory_atol)
        and torch.allclose(e0, e1, rtol=rtol, atol=energy_atol_j)
        and torch.allclose(b0, b1, rtol=rtol, atol=burnup_atol)
    )
    return EquilibriumCheck(
        converged, period, (n1 - n0).abs().max(), (e1 - e0).abs().max(),
        (b1 - b0).abs().max(),
    )


class CycleRunner:
    """Coordinate an existing CE/CM depleter, explicit schedule and transport callback.

    depleter must expose step(), settings.target_keff/criticality_tolerance,
    inventory_kind='atoms_per_cm3', volumes_cm3[zone], rate_nuclides and reactions.
    Each channel's axial bundle volumes must be equal: moving an unchanged
    density between unequal volumes would create/destroy atoms. Geometry with
    unequal physical bundles needs an explicit volume-carrying transfer model.
    """

    def __init__(
        self, depleter, schedule: RefuellingSchedule, channel_order, *,
        seconds_per_time_unit, max_dt_seconds, max_steps,
        power_balance_rtol=1e-10, power_balance_atol_w=1e-8,
    ):
        """Bind checked depletion to event-aligned bundle refuelling.

        Parameters
        ----------
        depleter : CECMDepleter
            Constant-volume depleter with inventories explicitly declared as atoms_per_cm3.
        schedule : RefuellingSchedule
            Ring schedules using one explicitly declared time unit.
        channel_order : sequence of str
            Unique physical channel ordering; bundles within each channel run from top to
            bottom.
        seconds_per_time_unit : float
            Positive conversion from schedule ticks or time units to seconds.
        max_dt_seconds : float
            Maximum duration of a depletion segment in seconds.
        max_steps : int
            Hard limit on depletion segments in one run.
        power_balance_rtol : float, optional
            Relative tolerance when comparing summed zone power with total power.
        power_balance_atol_w : float, optional
            Absolute power-balance tolerance in W.
        """
        self.depleter, self.schedule = depleter, schedule
        self.channel_order = tuple(channel_order)
        self.seconds_per_time_unit = _clock(seconds_per_time_unit, "seconds_per_time_unit")
        self.max_dt_seconds = _clock(max_dt_seconds, "max_dt_seconds")
        if self.seconds_per_time_unit <= 0 or self.max_dt_seconds <= 0:
            raise ValueError("Clock scale and maximum timestep must be positive")
        self.max_steps = _positive_int(max_steps, "max_steps")
        self.power_balance_rtol, self.power_balance_atol_w = power_balance_rtol, power_balance_atol_w
        if any(not math.isfinite(x) or x < 0 for x in (power_balance_rtol, power_balance_atol_w)):
            raise ValueError("Power balance tolerances must be finite and nonnegative")
        if depleter.inventory_kind != "atoms_per_cm3":
            raise CycleError("Lifecycle inventory must use atoms_per_cm3")

    def _validate(self, state):
        """Validate inventories and volume-conserving axial shuffle slots.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.

        Returns
        -------
        tuple of torch.Tensor
            Resident atom densities, accumulated energy and initial heavy-metal masses.
        """
        self.schedule.next_event(state)
        inventory, energy, mass = _fields(state, self.channel_order)
        if inventory.shape[1] != len(self.depleter.names):
            raise CycleError("Bundle inventory does not match the depleter chain length")
        volume = _tensor(self.depleter.volumes_cm3, "volumes_cm3")
        if volume.shape != (inventory.shape[0],) or (volume <= 0).any():
            raise CycleError("Depleter must have a positive volume for each ordered zone")
        index = 0
        for channel in self.channel_order:
            count = len(state.stacks[channel])
            if not torch.equal(volume[index:index + count], volume[index].expand(count)):
                raise CycleError("Axial bundle volumes must be equal within each channel")
            index += count
        return inventory, energy, mass

    def _operator(self, operator):
        """Wrap a stage callback with ordering and power-balance checks.

        Parameters
        ----------
        operator : callable
            Evaluate the current inventory and absolute time in seconds, returning reaction
            frequencies, total and zone power, keff and boron_ppm.

        Returns
        -------
        callable
            Checked callback preserving the original inventory gradient graph.
        """
        def checked(inventory, time):
            """Evaluate one stage and enforce its inventory and power contract.

            Parameters
            ----------
            inventory : torch.Tensor
                CPU float64 atom densities in atoms/cm³, shaped (zones, chain_nuclides).
            time : torch.Tensor
                Scalar absolute stage time in seconds.

            Returns
            -------
            mapping
                Validated callback result, including zone powers in W and rates in s⁻¹.

            Raises
            ------
            CycleError
                The callback mutates the inventory or violates ordering, power balance or
                criticality requirements.
            """
            before, volume_before = inventory._version, self.depleter.volumes_cm3._version
            result = operator(inventory, time)
            if inventory._version != before or self.depleter.volumes_cm3._version != volume_before:
                raise CycleError("Stage callback mutated inventory or volumes")
            if "nuclide_order" in result and tuple(result["nuclide_order"]) != self.depleter.names:
                raise CycleError("Transport and depletion nuclide ordering disagree")
            if "reaction_order" in result and tuple(result["reaction_order"]) != self.depleter.reactions:
                raise CycleError("Transport and depletion reaction ordering disagree")
            if "physical_zone_volumes_cm3" in result:
                physical = _tensor(result["physical_zone_volumes_cm3"], "physical_zone_volumes_cm3")
                if not torch.equal(physical, self.depleter.volumes_cm3):
                    raise CycleError("Power normalization and depletion physical volumes disagree")
            power = _scalar(result["power_w"], "power_w")
            zones = _tensor(result["zone_power_w"], "zone_power_w")
            rates = _tensor(result["rates"], "rates")
            expected = (inventory.shape[0], len(self.depleter.rate_nuclides), len(self.depleter.reactions))
            if rates.shape != expected or (rates < 0).any():
                raise CycleError("Rates must match [zone, rate nuclide, reaction] and be nonnegative")
            if zones.shape != (inventory.shape[0],) or (zones < 0).any() or power < 0:
                raise CycleError("Zone powers must be nonnegative and follow the zone ordering")
            if not torch.isclose(zones.sum(), power, rtol=self.power_balance_rtol, atol=self.power_balance_atol_w):
                raise CycleError("Zone powers do not sum to the supplied total power")
            boron = None if result["boron_ppm"] is None else _scalar(result["boron_ppm"], "boron_ppm")
            keff = None if result["keff"] is None else _scalar(result["keff"], "keff")
            if boron is not None and boron < 0:
                raise CycleError("Negative critical boron is infeasible")
            if power > 0:
                settings = self.depleter.settings
                if not admissible_powered_state(result, settings):
                    raise CycleError("Powered stage is not critical within the configured tolerance")
            elif torch.count_nonzero(rates):
                raise CycleError("Zero-power stage must have zero neutron reaction rates")
            return {**result, "power_w": power, "zone_power_w": zones, "boron_ppm": boron, "keff": keff}
        return checked

    def _snapshot(self, state, stage, time, data):
        """Capture a physical stage with its resident bundle histories.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.
        stage : str
            Stage label, such as beginning, midpoint, end or post-refuelling.
        time : int or float
            Absolute snapshot time in seconds.
        data : mapping
            Validated transport/depletion callback outputs.

        Returns
        -------
        CycleSnapshot
            Snapshot combining the state, stage diagnostics and channel ordering.
        """
        return CycleSnapshot(
            _scalar(time, "time_seconds"), stage, state, self.channel_order,
            data["zone_power_w"], data["boron_ppm"], data["keff"],
        )

    def plan_run(self, state, *, start_time, end_time, max_steps=None):
        """Count event-aligned work before evaluating transport or depletion.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.
        start_time : int or float
            Absolute start in schedule units; integer clocks require explicit integer ticks.
        end_time : int or float
            Absolute end in the same schedule units as start_time.
        max_steps : int or None, optional
            Optional per-run step cap no larger than the runner's configured limit.

        Returns
        -------
        dict
            Exact segment, stage-call, CRAM-propagation, event and discharge counts for the
            requested interval.

        Notes
        -----
        Only event times are visited; long event-free spans use rational ceiling division. The
        step cap also bounds enumeration of distinct events.
        """
        start, end = _clock(start_time, "start_time"), _clock(end_time, "end_time")
        if start < 0 or end < start:
            raise ValueError("Require 0 <= start_time <= end_time")
        if self.schedule.integer_clock and any(
                isinstance(t, bool) or not isinstance(t, Integral) for t in (start_time, end_time)):
            raise ValueError("Integer schedules require explicit integer start/end ticks")
        limit = self.max_steps if max_steps is None else _positive_int(max_steps, "max_steps")
        if limit > self.max_steps:
            raise ValueError("Per-run step limit cannot exceed the configured budget")
        _phase_bounds(state, self.schedule, start, allow_pending_event=True)
        counts = {ring.name: 0 for ring in self.schedule.rings}
        channels = {ring.name: len(ring.channels) for ring in self.schedule.rings}
        steps = events = discharged = 0
        current = start * self.seconds_per_time_unit

        def segment(stop):
            """Count capped depletion segments up to one event boundary.

            Parameters
            ----------
            stop : fractions.Fraction
                Exact segment endpoint in schedule units.
            """
            nonlocal current, steps
            steps += math.ceil((stop-current)/self.max_dt_seconds)
            if steps > limit:
                raise CycleError(f"Planned lifecycle needs {steps} steps, exceeding its {limit}-step budget")
            current = stop

        def event_call(event):
            """Accumulate one atomic event's ring and discharge counts.

            Parameters
            ----------
            event : RefuellingEvent
                Next scheduled event inside the planned interval.
            """
            nonlocal events, discharged
            events += 1
            for ring in event.rings:
                counts[ring] += 1
                discharged += channels[ring]

        pending = self.schedule.next_event(state)
        if _clock(pending.time, "event time") == start:
            event_call(pending)
        for event in self.schedule.events(end_time, start=start_time):
            segment(_clock(event.time, "event time") * self.seconds_per_time_unit)
            event_call(event)
        segment(end * self.seconds_per_time_unit)
        return dict(start_time=start_time, end_time=end_time, steps=steps,
            post_refuelling_calls=events, stage_evaluations=3*steps+events+int(steps==events==0),
            cram_propagations=2*steps, discharged_bundles=discharged,
            event_counts_by_ring=counts)

    def run(
        self, state, *, start_time, end_time, fresh_inventory_by_ring,
        fresh_history_by_ring, stage_operator, max_steps=None,
    ):
        """Advance event-aligned depletion and solve each post-shuffle state.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.
        start_time : int or float
            Absolute start in schedule units; integer clocks require explicit integer ticks.
        end_time : int or float
            Absolute end in the same schedule units as start_time.
        fresh_inventory_by_ring : mapping
            Ring names mapped to fresh inventories containing full-chain nuclides in atoms/cm³.
        fresh_history_by_ring : mapping
            Ring names mapped to zero initial energy_j and positive initial_hm_kg.
        stage_operator : callable
            Evaluate the current inventory and absolute time in seconds, returning reaction
            frequencies, total and zone power, keff and boron_ppm.
        max_steps : int or None, optional
            Optional per-run step cap no larger than the runner's configured limit.

        Returns
        -------
        CycleResult
            Final resident state, sampled stages, discharge records and performed step count.

        Notes
        -----
        Snapshots include CE/CM beginning, midpoint, end, and post-refuelling states. Midpoint
        energy uses the predictor half-step; final energy uses midpoint power over the full
        interval. These are stage samples, not preassigned time-quadrature weights for
        optimization.
        """
        plan = self.plan_run(state, start_time=start_time, end_time=end_time, max_steps=max_steps)
        start, end = _clock(start_time, "start_time"), _clock(end_time, "end_time")
        limit = self.max_steps if max_steps is None else int(max_steps)
        self._validate(state)
        if any(
            bundle.inserted_at is not None and _clock(bundle.inserted_at, "inserted_at") > start
            for bundle in _bundles(state, self.channel_order)
        ):
            raise CycleError("Initial state contains a bundle inserted after start_time")
        chain_length = _fields(state, self.channel_order)[0].shape[-1]
        for ring in self.schedule.rings:
            recipe = Bundle("recipe", None, fresh_inventory_by_ring[ring.name], fresh_history_by_ring[ring.name])
            fresh, energy, _ = _fields(CoreState({"recipe": (recipe,)}), ("recipe",))
            if energy[0] != 0 or fresh.shape[-1] != chain_length:
                raise CycleError("Fresh recipes require zero accumulated energy and the same chain")
        current, finish = start * self.seconds_per_time_unit, end * self.seconds_per_time_unit
        checked = self._operator(stage_operator)
        snapshots, discharges, steps = [], [], 0
        while True:
            event = self.schedule.next_event(state)
            event_time = _clock(event.time, "event time") * self.seconds_per_time_unit
            if event_time < current:
                raise CycleError("Initial/restarted state skipped an earlier refuelling event")
            if event_time == current:
                result = self.schedule.apply_event(
                    state, event, fresh_inventory_by_ring,
                    fresh_history_by_ring=fresh_history_by_ring,
                )
                state = result.state
                inventory, _, _ = self._validate(state)
                try:
                    data = checked(inventory, _scalar(float(current), "event time"))
                except Exception as error:
                    raise CycleError(f"Post-refuelling solve failed at {event.time}: {error}") from error
                snapshots.append(self._snapshot(state, "post-refuelling", float(current), data))
                discharges.extend(result.discharged)
            if current == finish:
                if not snapshots:
                    inventory, _, _ = self._validate(state)
                    data = checked(inventory, _scalar(float(current), "initial time"))
                    snapshots.append(self._snapshot(state, "initial", float(current), data))
                break
            if steps >= limit:
                raise CycleError(f"Lifecycle exceeded its {limit}-step budget before end_time")
            event_time = _clock(self.schedule.next_event(state).time, "event time") * self.seconds_per_time_unit
            stop = min(finish, current + self.max_dt_seconds, event_time)
            duration = _scalar(float(stop - current), "duration")
            time = _scalar(float(current), "time")
            if time + duration <= time:
                raise CycleError("Clock resolution cannot represent the next positive timestep")
            inventory, energy, _ = self._validate(state)
            step = self.depleter.step(inventory, time, duration, checked)
            if tuple(stage.name for stage in step.stages) != ("beginning", "midpoint", "end"):
                raise CycleError("Depleter did not return the expected CE/CM stages")
            if step.inventory.shape != inventory.shape or (step.inventory < 0).any():
                raise CycleError("Depleter returned invalid final inventory")
            _tensor(step.inventory, "final inventory")
            if not torch.isclose(step.end_time_seconds, time + duration, rtol=1e-13, atol=0):
                raise CycleError("Depleter returned the wrong endpoint")
            beginning, middle, ending = step.stages
            predictor = _replace_fields(
                state, self.channel_order, middle.inventory,
                energy + beginning.state["zone_power_w"] * duration / 2,
            )
            final = _replace_fields(
                state, self.channel_order, step.inventory,
                energy + middle.state["zone_power_w"] * duration,
            )
            snapshots.extend((
                self._snapshot(state, "beginning", beginning.time_seconds, beginning.state),
                self._snapshot(predictor, "midpoint", middle.time_seconds, middle.state),
                self._snapshot(final, "end", ending.time_seconds, ending.state),
            ))
            state, current, steps = final, stop, steps + 1
        if (steps != plan['steps'] or len(snapshots) != plan['stage_evaluations']
                or len(discharges) != plan['discharged_bundles']):
            raise CycleError("Executed lifecycle differs from its work preflight")
        return CycleResult(state, tuple(snapshots), tuple(discharges), steps, start_time, end_time)

    def run_until_equilibrium(
        self, state, *, start_time, max_cycles, max_cycle_ticks,
        inventory_atol, energy_atol_j, burnup_atol, rtol,
        fresh_inventory_by_ring, fresh_history_by_ring, stage_operator,
    ):
        """Require convergence of states sampled at the same cycle phase.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.
        start_time : int or float
            Absolute start in schedule units; integer clocks require explicit integer ticks.
        max_cycles : int
            Maximum number of full common schedule periods to execute.
        max_cycle_ticks : int
            Upper bound on the exact common integer-clock period.
        inventory_atol : float
            Absolute convergence tolerance in atoms/cm³.
        energy_atol_j : float
            Absolute convergence tolerance in J.
        burnup_atol : float
            Absolute convergence tolerance in MWd/t of initial heavy metal.
        rtol : float
            Dimensionless relative convergence tolerance.
        fresh_inventory_by_ring : mapping
            Ring names mapped to fresh inventories containing full-chain nuclides in atoms/cm³.
        fresh_history_by_ring : mapping
            Ring names mapped to zero initial energy_j and positive initial_hm_kg.
        stage_operator : callable
            Evaluate the current inventory and absolute time in seconds, returning reaction
            frequencies, total and zone power, keff and boron_ppm.

        Returns
        -------
        EquilibriumResult
            Converged state, completed cycle results and the final comparison.

        Raises
        ------
        CycleError
            The configured work limit is exceeded or convergence is not reached.

        Notes
        -----
        The caller must also supply cycle-periodic external operating inputs. A schedule period
        cannot establish that an arbitrary callback repeats.
        """
        period = self.schedule.common_cycle(max_ticks=max_cycle_ticks)
        _positive_int(max_cycles, "max_cycles")
        _tolerances(inventory_atol, energy_atol_j, burnup_atol, rtol)
        if isinstance(start_time, bool) or not isinstance(start_time, Integral) or start_time < 0:
            raise ValueError("Equilibrium needs an explicit nonnegative integer start tick")
        snapshots, discharges, checks, steps = [], [], [], 0
        initial_time, current = start_time, start_time
        for count in range(1, max_cycles + 1):
            if steps >= self.max_steps:
                raise CycleError("Equilibrium exhausted its total timestep budget")
            before = state
            result = self.run(
                state, start_time=current, end_time=current + period,
                fresh_inventory_by_ring=fresh_inventory_by_ring,
                fresh_history_by_ring=fresh_history_by_ring,
                stage_operator=stage_operator, max_steps=self.max_steps - steps,
            )
            state = result.final_state
            snapshots.extend(result.snapshots)
            discharges.extend(result.discharges)
            steps += result.steps
            # A delayed startup ring may not yet contribute a complete period.
            complete = all(
                state.event_counts.get(ring.name, 0) - before.event_counts.get(ring.name, 0) == period // ring.interval
                for ring in self.schedule.rings
            )
            if complete:
                check = compare_cycle_states(
                    before, state, self.schedule, self.channel_order,
                    start_time=current, end_time=current + period, max_cycle_ticks=max_cycle_ticks,
                    inventory_atol=inventory_atol, energy_atol_j=energy_atol_j,
                    burnup_atol=burnup_atol, rtol=rtol,
                )
                checks.append(check)
                if check.converged:
                    combined = CycleResult(
                        state, tuple(snapshots), tuple(discharges), steps,
                        initial_time, current + period,
                    )
                    return EquilibriumResult(combined, tuple(checks), count)
            current += period
        raise CycleError(f"Fuel state did not converge in {max_cycles} common cycles")
