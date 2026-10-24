"""Top-in/bottom-out, ring-synchronous refuelling without depletion.

All times use the caller's single declared unit. With integer_clock=True they
are explicitly integer ticks; only that mode exposes an exact common schedule
period. Exact arithmetic on supplied decimal representations orders events
and groups ties without rounding nearby events together.

Stacks run from top to bottom. Inventory/history mappings are copied and made
read-only; their payloads are deliberately not copied, converted, or detached.
This preserves tensor autograd graphs. Callers must treat payloads as immutable
and replace a Bundle after depletion instead of modifying a tensor in place.
No composition symmetry or equilibrium state is invented by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import heapq
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Iterator, Mapping


def _readonly(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a mapping's structure while preserving its payload objects.

    Parameters
    ----------
    values : mapping of str to object
        Inventory, history or structural data to copy.

    Returns
    -------
    types.MappingProxyType
        Read-only mapping; payload tensors retain their existing graphs.
    """
    return MappingProxyType(dict(values))


def _number(value: float | int, name: str) -> Fraction:
    """Convert a finite clock value to an exact decimal fraction.

    Parameters
    ----------
    value : int or float
        Numerical input to validate.
    name : str
        Field label used in diagnostics.

    Returns
    -------
    fractions.Fraction
        Rational representation without rounding nearby event times together.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if isinstance(value, Integral):
        return Fraction(int(value))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Fraction(str(value))


def _positive_integer(value: int, name: str) -> int:
    """Require a positive integer other than a boolean.

    Parameters
    ----------
    value : int or float
        Numerical input to validate.
    name : str
        Field label used in diagnostics.

    Returns
    -------
    int
        Validated integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class Bundle:
    """Physical identity and opaque inventory/history; unknown insertion is None."""

    id: str
    inserted_at: float | int | None
    inventory: Mapping[str, Any]
    history: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate insertion time and freeze inventory/history mappings."""
        if self.inserted_at is not None:
            _number(self.inserted_at, "inserted_at")
        object.__setattr__(self, "inventory", _readonly(self.inventory))
        object.__setattr__(self, "history", _readonly(self.history))


@dataclass(frozen=True)
class RingSchedule:
    """Each interval inserts and discharges one bundle in every channel.

    interval is the displacement/discharge interval. A newly inserted bundle
    resides for levels * interval. first_event defaults to one interval; an
    explicit positive value sets the ring's initial event phase.
    """

    name: str
    channels: tuple[str, ...]
    levels: int
    interval: float | int
    first_event: float | int | None = None

    def __post_init__(self) -> None:
        """Validate channels, stack depth, displacement interval and phase."""
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("A ring needs at least one channel")
        if len(set(channels)) != len(channels):
            raise ValueError("Channel names must be unique within a ring")
        _positive_integer(self.levels, "levels")
        if _number(self.interval, "interval") <= 0:
            raise ValueError("interval must be positive")
        first = self.interval if self.first_event is None else self.first_event
        if _number(first, "first_event") <= 0:
            raise ValueError("first_event must be positive")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "first_event", first)


@dataclass(frozen=True)
class CoreState:
    """Structural state; event_counts permits an explicit restart.

    Initial states normally omit event_counts (all zero). A restart must carry
    counts from its last completed atomic event, even if inventories have
    subsequently been replaced by the caller's depletion calculation.
    """

    stacks: Mapping[str, tuple[Bundle, ...]]
    event_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate unique resident identities and freeze restart structure."""
        stacks = {channel: tuple(stack) for channel, stack in self.stacks.items()}
        ids = []
        for stack in stacks.values():
            ids.extend(bundle.id for bundle in stack)
        if len(ids) != len(set(ids)):
            raise ValueError("Bundle ids must be unique throughout the core")
        counts = dict(self.event_counts)
        for count in counts.values():
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise ValueError("event_counts must contain nonnegative integers")
        object.__setattr__(self, "stacks", _readonly(stacks))
        object.__setattr__(self, "event_counts", _readonly(counts))


@dataclass(frozen=True)
class RefuellingEvent:
    """Group simultaneous ring insertions by their exact event ordinals."""
    time: float | int
    ordinals: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate event time and freeze positive ring-event ordinals."""
        if _number(self.time, "event time") <= 0 or not self.ordinals:
            raise ValueError("An event needs a positive time and at least one ring")
        for value in self.ordinals.values():
            _positive_integer(value, "event ordinal")
        object.__setattr__(self, "ordinals", _readonly(self.ordinals))

    @property
    def rings(self) -> tuple[str, ...]:
        """Return participating ring names in deterministic order.

        Returns
        -------
        tuple of str
            Sorted names of rings moving in this atomic event.
        """
        return tuple(sorted(self.ordinals))


@dataclass(frozen=True)
class Discharge:
    """Keep an exiting physical bundle with its channel, time and known residence."""
    ring: str
    channel: str
    bundle: Bundle
    time: float | int

    @property
    def residence(self) -> float | int | None:
        """Return known residence time in the caller's schedule unit.

        Returns
        -------
        int, float or None
            Discharge time minus insertion time, or None for an unknown insertion history.
        """
        return None if self.bundle.inserted_at is None else self.time - self.bundle.inserted_at


@dataclass(frozen=True)
class ShuffleResult:
    """Return the new core state together with inserted and discharged bundles."""
    state: CoreState
    discharged: tuple[Discharge, ...]
    inserted: Mapping[str, Bundle]

    def __post_init__(self) -> None:
        """Freeze the mapping from physical channels to inserted bundles."""
        object.__setattr__(self, "inserted", _readonly(self.inserted))


class RefuellingSchedule:
    """Order ring loading events and transfer complete bundle histories.

    All rings use the same caller-declared time unit. Integer clocks also
    provide an exact common period for equilibrium-cycle comparisons."""
    def __init__(self, rings: tuple[RingSchedule, ...], *, integer_clock: bool = False):
        """Bind disjoint rings to one ordered refuelling schedule.

        Parameters
        ----------
        rings : sequence of RingSchedule
            Rings with unique names and nonoverlapping channel identities.
        integer_clock : bool, optional
            Require explicit integer intervals and phases, enabling exact common-period queries.
        """
        rings = tuple(rings)
        if not rings:
            raise ValueError("At least one ring is required")
        if len({ring.name for ring in rings}) != len(rings):
            raise ValueError("Ring names must be unique")
        channels = [channel for ring in rings for channel in ring.channels]
        if len(channels) != len(set(channels)):
            raise ValueError("A channel cannot belong to two rings")
        if not isinstance(integer_clock, bool):
            raise ValueError("integer_clock must be boolean")
        if integer_clock:
            for ring in rings:
                _positive_integer(ring.interval, "interval in integer ticks")
                _positive_integer(ring.first_event, "first_event in integer ticks")
        self.rings = tuple(sorted(rings, key=lambda ring: ring.name))
        self.integer_clock = integer_clock
        self._by_name = MappingProxyType({ring.name: ring for ring in self.rings})

    def _clock(self, value: float | int, name: str) -> Fraction:
        """Validate a clock value under the declared integer-clock policy.

        Parameters
        ----------
        value : int or float
            Numerical input to validate.
        name : str
            Field label used in diagnostics.

        Returns
        -------
        fractions.Fraction
            Exact event-ordering value in schedule units.
        """
        if self.integer_clock and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise ValueError(f"{name} must be explicit integer ticks")
        return _number(value, name)

    def _time(self, ring: RingSchedule, ordinal: int) -> Fraction:
        """Compute the exact time of a ring's numbered displacement.

        Parameters
        ----------
        ring : RingSchedule
            Ring interval, initial phase and stack depth.
        ordinal : int
            One-based displacement index within the ring.

        Returns
        -------
        fractions.Fraction
            Event time including the ring's initial phase.
        """
        return Fraction(str(ring.first_event)) + (ordinal - 1) * Fraction(str(ring.interval))

    def _event(self, time: Fraction, ordinals: Mapping[str, int]) -> RefuellingEvent:
        """Build an atomic event with the declared external clock type.

        Parameters
        ----------
        time : fractions.Fraction
            Exact shared time of the participating ring events.
        ordinals : mapping of str to int
            One-based event indices for the participating rings.

        Returns
        -------
        RefuellingEvent
            Event using integer ticks or real-valued schedule units.
        """
        return RefuellingEvent(int(time) if self.integer_clock else float(time), ordinals)

    def events(self, end: float | int, *, start: float | int = 0) -> Iterator[RefuellingEvent]:
        """Yield atomic events in the half-open interval (start, end].

        Parameters
        ----------
        end : int or float
            Inclusive upper time bound in schedule units.
        start : int or float, optional
            Exclusive lower bound in the same schedule units.

        Yields
        ------
        RefuellingEvent
            Next atomic event, grouping exactly simultaneous ring displacements once.

        Notes
        -----
        Starting after zero only filters output; it does not initialize a depleted/restarted
        core or infer the skipped event counts.
        """
        begin, finish = self._clock(start, "start"), self._clock(end, "end")
        if begin < 0 or finish < begin:
            raise ValueError("Require 0 <= start <= end")
        heap = []
        for ring in self.rings:
            first, interval = self._time(ring, 1), Fraction(str(ring.interval))
            ordinal = 1 if begin < first else int((begin - first) // interval) + 2
            heapq.heappush(heap, (self._time(ring, ordinal), ring.name, ordinal))
        while heap and heap[0][0] <= finish:
            time = heap[0][0]
            ordinals = {}
            while heap and heap[0][0] == time:
                _, name, ordinal = heapq.heappop(heap)
                ordinals[name] = ordinal
                heapq.heappush(heap, (self._time(self._by_name[name], ordinal + 1), name, ordinal + 1))
            yield self._event(time, ordinals)

    def common_cycle(self, *, max_ticks: int) -> int:
        """Return the exact forcing period after all rings have started.

        Parameters
        ----------
        max_ticks : int
            Maximum allowed least common multiple of ring intervals.

        Returns
        -------
        int
            Common schedule period in integer ticks; this does not establish fuel equilibrium.

        Raises
        ------
        ValueError
            Integer-clock mode is disabled or the common period exceeds max_ticks.

        Notes
        -----
        Real-valued intervals are never rounded/rationalized into a cycle. The caller must
        explicitly choose integer ticks and bound the LCM.
        """
        if not self.integer_clock:
            raise ValueError("A common cycle requires an explicit integer clock")
        limit = _positive_integer(max_ticks, "max_ticks")
        cycle = 1
        for ring in self.rings:
            cycle = math.lcm(cycle, int(ring.interval))
            if cycle > limit:
                raise ValueError(f"Common cycle exceeds max_ticks={limit}")
        return cycle

    def _validate_state(self, state: CoreState) -> None:
        """Check stack structure and consistent completed-event counts.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.

        Raises
        ------
        ValueError
            Channels, stack lengths or restart counts disagree with this schedule.
        """
        expected = {channel for ring in self.rings for channel in ring.channels}
        if set(state.stacks) != expected:
            raise ValueError("Core channels do not match the configured rings")
        if set(state.event_counts) - set(self._by_name):
            raise ValueError("State has event counts for unknown rings")
        for ring in self.rings:
            if any(len(state.stacks[channel]) != ring.levels for channel in ring.channels):
                raise ValueError(f"Stack length does not match ring {ring.name}")
        completed = [
            self._time(ring, state.event_counts[ring.name])
            for ring in self.rings if state.event_counts.get(ring.name, 0)
        ]
        if completed:
            last = max(completed)
            if any(
                self._time(ring, state.event_counts.get(ring.name, 0) + 1) <= last
                for ring in self.rings
            ):
                raise ValueError("Restart counts skip a prior or simultaneous ring event")

    def next_event(self, state: CoreState) -> RefuellingEvent:
        """Find the next complete event following the resident state.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.

        Returns
        -------
        RefuellingEvent
            Earliest uncompleted event, including all rings with that exact event time.
        """
        self._validate_state(state)
        next_times = {
            ring.name: self._time(ring, state.event_counts.get(ring.name, 0) + 1)
            for ring in self.rings
        }
        first = min(next_times.values())
        return self._event(first, {
            name: state.event_counts.get(name, 0) + 1
            for name, time in next_times.items() if time == first
        })

    def apply_event(
        self,
        state: CoreState,
        event: RefuellingEvent,
        fresh_inventory_by_ring: Mapping[str, Mapping[str, Any]],
        *,
        fresh_history_by_ring: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ShuffleResult:
        """Insert fresh bundles and discharge one bottom bundle per channel.

        Parameters
        ----------
        state : CoreState
            Resident bundle stacks and completed ring-event counts.
        event : RefuellingEvent
            The next complete atomic event for the supplied state.
        fresh_inventory_by_ring : mapping
            Fresh inventory mappings shared by all channels of each participating ring.
        fresh_history_by_ring : mapping or None, optional
            Fresh history mappings by ring; missing histories default to empty mappings.

        Returns
        -------
        ShuffleResult
            New state, physical discharges and channel-indexed inserted bundles.

        Notes
        -----
        A ring's fresh recipe is shared across its channels. Physical bundle IDs remain
        different. Existing inventory symmetry is the caller's explicit invariant; no averaging
        or silent replacement is performed here.
        """
        if event != self.next_event(state):
            raise ValueError("Event must be the next complete atomic event for this state")
        histories = {} if fresh_history_by_ring is None else fresh_history_by_ring
        stacks, counts = dict(state.stacks), dict(state.event_counts)
        discharged, inserted = [], {}
        for name in event.rings:
            ring = self._by_name[name]
            ordinal = event.ordinals[name]
            counts[name] = ordinal
            for channel in ring.channels:
                identifier = "fresh:" + json.dumps([name, channel, ordinal], separators=(",", ":"))
                bundle = Bundle(identifier, event.time, fresh_inventory_by_ring[name], histories.get(name, {}))
                old_stack = state.stacks[channel]
                if any(
                    old.inserted_at is not None and old.inserted_at > event.time
                    for old in old_stack
                ):
                    raise ValueError("A resident bundle cannot have a future insertion time")
                discharged.append(Discharge(name, channel, old_stack[-1], event.time))
                inserted[channel] = bundle
                stacks[channel] = (bundle,) + old_stack[:-1]
        return ShuffleResult(CoreState(stacks, counts), tuple(discharged), inserted)
