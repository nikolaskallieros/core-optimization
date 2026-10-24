"""Scheduler invariants independent of any transport/depletion approximation."""

from dataclasses import replace
import math
import unittest

from fuel_cycle.refuelling import Bundle, CoreState, RefuellingEvent, RefuellingSchedule, RingSchedule


def initial_state(rings):
    """Create uniquely identified bundle stacks for the supplied rings.

    Parameters
    ----------
    rings : sequence of RingSchedule
        Ring/channel structure to populate with manufactured inventories.

    Returns
    -------
    CoreState
        Initial stacks with opaque test inventory and energy values.
    """
    return CoreState({
        channel: tuple(
            Bundle(f"initial:{channel}:{level}", None, {"U235": 1.0 + level}, {"energy": level})
            for level in range(ring.levels)
        )
        for ring in rings for channel in ring.channels
    })


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        """Initialize two refuelling rings and distinct fresh recipes."""
        self.rings = (
            RingSchedule("inner", ("i0", "i1"), 3, 2),
            RingSchedule("outer", ("o0", "o1"), 2, 3),
        )
        self.schedule = RefuellingSchedule(self.rings, integer_clock=True)
        self.state = initial_state(self.rings)
        self.fresh = {"inner": {"U235": 10.0}, "outer": {"U235": 20.0}}

    def test_independent_timings_and_simultaneous_events(self):
        """Verify independent timings and simultaneous events."""
        events = list(self.schedule.events(12))
        self.assertEqual([event.time for event in events], [2, 3, 4, 6, 8, 9, 10, 12])
        self.assertEqual(events[3].rings, ("inner", "outer"))
        self.assertEqual(dict(events[3].ordinals), {"inner": 3, "outer": 2})
        self.assertEqual(
            [event.time for event in events if "inner" in event.rings],
            [2, 4, 6, 8, 10, 12],
        )
        self.assertEqual(
            [event.time for event in self.schedule.events(6, start=3)], [4, 6]
        )
        self.assertEqual(list(self.schedule.events(0)), [])

    def test_stable_identity_history_and_top_bottom_order(self):
        """Verify stable identity history and top bottom order."""
        old = self.state.stacks["i0"]
        result = self.schedule.apply_event(self.state, self.schedule.next_event(self.state), self.fresh)
        new = result.state.stacks["i0"]
        self.assertIs(new[1], old[0])
        self.assertIs(new[2], old[1])
        self.assertIs(result.discharged[0].bundle, old[-1])
        self.assertEqual(dict(new[1].history), {"energy": 0})
        self.assertIs(result.state.stacks["o0"][0], self.state.stacks["o0"][0])
        self.assertEqual(result.inserted["i0"].inserted_at, 2)
        self.assertIsNone(result.discharged[0].residence)
        self.assertEqual(self.state.stacks["i0"], old)
        self.assertFalse(self.state.event_counts)

    def test_inventory_conservation_across_each_atomic_event(self):
        """Verify inventory conservation across each atomic event."""
        state = self.state
        def total(current):
            """Sum the manufactured U235 inventory over all resident bundles.

            Parameters
            ----------
            current : CoreState
                Resident state to total.

            Returns
            -------
            float
                Total fixture inventory in its declared arbitrary unit.
            """
            return sum(bundle.inventory["U235"] for stack in current.stacks.values() for bundle in stack)
        for event in self.schedule.events(18):
            previous = total(state)
            result = self.schedule.apply_event(state, event, self.fresh)
            supplied = sum(bundle.inventory["U235"] for bundle in result.inserted.values())
            removed = sum(item.bundle.inventory["U235"] for item in result.discharged)
            self.assertEqual(total(result.state) + removed, previous + supplied)
            self.assertEqual(sum(map(len, result.state.stacks.values())), 10)
            self.assertEqual(len({b.id for stack in result.state.stacks.values() for b in stack}), 10)
            state = result.state

    def test_residence_is_levels_times_interval(self):
        """Verify residence is levels times interval."""
        state = self.state
        seen = {"inner": 0, "outer": 0}
        for event in self.schedule.events(24):
            result = self.schedule.apply_event(state, event, self.fresh)
            for discharged in result.discharged:
                if discharged.bundle.inserted_at is not None:
                    ring = next(r for r in self.rings if r.name == discharged.ring)
                    self.assertEqual(discharged.residence, ring.levels * ring.interval)
                    seen[ring.name] += 1
            state = result.state
        self.assertTrue(all(seen.values()))

    def test_tie_must_be_applied_atomically_and_cannot_be_replayed(self):
        """Verify tie must be applied atomically and cannot be replayed."""
        state = self.state
        for event in self.schedule.events(4):
            state = self.schedule.apply_event(state, event, self.fresh).state
        tie = self.schedule.next_event(state)
        self.assertEqual(tie.time, 6)
        with self.assertRaises(ValueError):
            self.schedule.apply_event(state, RefuellingEvent(6, {"inner": 3}), self.fresh)
        with self.assertRaisesRegex(KeyError, "outer"):
            self.schedule.apply_event(state, tie, {"inner": self.fresh["inner"]})
        self.assertEqual(dict(state.event_counts), {"inner": 2, "outer": 1})
        after = self.schedule.apply_event(state, tie, self.fresh)
        self.assertEqual(len(after.inserted), 4)
        self.assertEqual(len(after.discharged), 4)
        with self.assertRaises(ValueError):
            self.schedule.apply_event(after.state, tie, self.fresh)

    def test_mappings_are_readonly_but_payloads_keep_identity(self):
        """Verify mappings are readonly but payloads keep identity."""
        payload = object()  # A tensor can occupy this position without being detached.
        source = {"U235": payload}
        bundle = Bundle("payload", 0, source, {"history": payload})
        source["U235"] = 123
        self.assertIs(bundle.inventory["U235"], payload)
        self.assertIs(bundle.history["history"], payload)
        for mapping, key in [(bundle.inventory, "U235"), (bundle.history, "history"),
                             (self.state.stacks, "i0"), (self.state.event_counts, "inner")]:
            with self.assertRaises(TypeError):
                mapping[key] = None
        updated = replace(bundle, inventory={"U235": 4.0})
        self.assertIs(bundle.inventory["U235"], payload)
        self.assertEqual(updated.inventory["U235"], 4.0)
        event = self.schedule.next_event(self.state)
        with self.assertRaises(TypeError):
            event.ordinals["inner"] = 99
        result = self.schedule.apply_event(self.state, event, self.fresh)
        with self.assertRaises(TypeError):
            result.inserted["i0"] = None

    def test_ring_recipe_preserves_synchronous_composition(self):
        """Verify ring recipe preserves synchronous composition."""
        result = self.schedule.apply_event(
            self.state, self.schedule.next_event(self.state), self.fresh,
            fresh_history_by_ring={"inner": {"energy": 0.0}},
        )
        a, b = result.inserted["i0"], result.inserted["i1"]
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(a.inventory, b.inventory)
        self.assertEqual(a.history, b.history)
        self.assertEqual(a.inserted_at, b.inserted_at)

    def test_explicit_clock_cycle_and_excessive_cycle(self):
        """Verify explicit clock cycle and excessive cycle."""
        self.assertEqual(self.schedule.common_cycle(max_ticks=6), 6)
        with self.assertRaises(ValueError):
            self.schedule.common_cycle(max_ticks=5)
        with self.assertRaises(ValueError):
            self.schedule.common_cycle(max_ticks=6.0)
        with self.assertRaises(ValueError):
            list(self.schedule.events(6.0))
        with self.assertRaises(ValueError):
            RefuellingSchedule((RingSchedule("a", ("a",), 1, 2.0),), integer_clock=True)

    def test_real_intervals_do_not_assert_periodicity(self):
        """Verify real intervals do not assert periodicity."""
        schedule = RefuellingSchedule((
            RingSchedule("a", ("a",), 1, 1.0),
            RingSchedule("b", ("b",), 1, math.sqrt(2)),
        ))
        self.assertEqual(len(list(schedule.events(3))), 5)
        with self.assertRaises(ValueError):
            schedule.common_cycle(max_ticks=100000)
        decimal = RefuellingSchedule((
            RingSchedule("a", ("a",), 1, 0.1),
            RingSchedule("b", ("b",), 1, 0.3),
        ))
        self.assertEqual(list(decimal.events(0.3))[-1].rings, ("a", "b"))
        near = RefuellingSchedule((
            RingSchedule("a", ("a",), 1, 1),
            RingSchedule("b", ("b",), 1, 1.000000001),
        ))
        self.assertEqual(len(list(near.events(1.1))), 2)

    def test_phase_and_restart_counts(self):
        """Verify phase and restart counts."""
        rings = (RingSchedule("a", ("a",), 2, 3, first_event=1),)
        schedule = RefuellingSchedule(rings, integer_clock=True)
        self.assertEqual([e.time for e in schedule.events(10)], [1, 4, 7, 10])
        self.assertEqual(schedule.common_cycle(max_ticks=3), 3)
        state = initial_state(rings)
        state = schedule.apply_event(state, schedule.next_event(state), {"a": {"U235": 1}}).state
        restart = CoreState(dict(state.stacks), dict(state.event_counts))
        self.assertEqual(schedule.next_event(restart).time, 4)

    def test_integer_ticks_remain_exact_above_float_precision(self):
        """Verify integer ticks remain exact above float precision."""
        tick = 10 ** 35 + 1
        ring = RingSchedule("a", ("a",), 1, tick)
        schedule = RefuellingSchedule((ring,), integer_clock=True)
        self.assertEqual([e.time for e in schedule.events(3 * tick)], [tick, 2 * tick, 3 * tick])
        self.assertEqual(schedule.common_cycle(max_ticks=tick), tick)

    def test_restart_cannot_skip_an_earlier_or_simultaneous_event(self):
        """Verify restart cannot skip an earlier or simultaneous event."""
        for counts in [{"inner": 3, "outer": 1}, {"inner": 10}]:
            with self.assertRaises(ValueError):
                self.schedule.next_event(replace(self.state, event_counts=counts))

    def test_rejects_invalid_topology_state_and_time(self):
        """Verify rejects invalid topology state and time."""
        for interval in [0, -1, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                RingSchedule("a", ("a",), 1, interval)
        with self.assertRaises(ValueError):
            RefuellingSchedule((RingSchedule("a", ("x",), 1, 1), RingSchedule("b", ("x",), 1, 2)))
        with self.assertRaises(ValueError):
            CoreState({"x": (Bundle("same", None, {}), Bundle("same", None, {}))})
        with self.assertRaises(ValueError):
            self.schedule.next_event(CoreState({"unknown": (Bundle("one", None, {}),)}))
        with self.assertRaises(ValueError):
            self.schedule.next_event(replace(self.state, stacks={**self.state.stacks, "i0": self.state.stacks["i0"][:1]}))
        with self.assertRaises(ValueError):
            list(self.schedule.events(1, start=2))


if __name__ == "__main__":
    unittest.main()
