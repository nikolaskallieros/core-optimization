"""Analytic temporal-moment and objective checks; no finite differences or transport."""
from examples.wr_1 import WR1_ROOT
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from examples.wr_1.model.case import study
from fuel_cycle.residence import FuelMap
from coupling.optimization import (assess_native, design_loss,
                                             eol_check, eol_state_key, prediction_rank)
from fuel_cycle.residence import native_temporal_profile, sample_temporal_power, temporal_power
from runtime.io import load_state
from optimization.solvers import bounded_linear_projection

ROOT = WR1_ROOT
DTYPE = torch.float64


class TemporalPowerTests(unittest.TestCase):
    def test_unequal_constant_channels_have_zero_temporal_spread(self):
        """Verify unequal constant channels have zero temporal spread."""
        value = temporal_power([0., .1, 1.], [[2., 9.], [2., 9.], [2., 9.]], [1., 6.])
        self.assertEqual(float(value["variance"]), 0.)

    def test_exact_linear_moments_on_nonuniform_times(self):
        """Verify exact linear moments on nonuniform times."""
        t = torch.tensor([0., .01, .23, .8, 1.], dtype=DTYPE)
        amplitude = torch.tensor(.2, dtype=DTYPE, requires_grad=True)
        p = torch.stack((10 * (1 + amplitude * (2*t-1)), 3 * torch.ones_like(t)), dim=1)
        value = temporal_power(t, p, [6., 1.])
        self.assertAlmostEqual(float(value["variance"].detach()), 6/7 * .2**2/3, places=15)
        gradient, = torch.autograd.grad(value["variance"], amplitude)
        self.assertAlmostEqual(float(gradient), 6/7 * 2*.2/3, places=14)
        torch.testing.assert_close(value["mean_power_W"], torch.tensor([10., 3.], dtype=DTYPE))

    def test_units_and_channel_power_scale_do_not_change_relative_variance(self):
        """Verify units and channel power scale do not change relative variance."""
        t = torch.tensor([0., .02, .4, 1.], dtype=DTYPE)
        p = torch.tensor([[1., 4.], [2., 5.], [3., 8.], [4., 3.]], dtype=DTYPE)
        a = temporal_power(t, p, [1., 6.])
        b = temporal_power(500*t + 30, p * torch.tensor([1e6, .2]), [1., 6.])
        self.assertAlmostEqual(float(a["variance"]), float(b["variance"]), places=12)

    def test_axial_redistribution_is_summed_before_temporal_spread(self):
        """Verify axial redistribution is summed before temporal spread."""
        labels = [f"c/axial/{z}" for z in range(3)]
        physical = SimpleNamespace(labels=labels, symmetry_multiplier=6, multiplicity=np.ones(3),
                                   regions=[{"label": l, "ring": 0} for l in labels])
        mapping = FuelMap(physical)
        values = [torch.tensor(p, dtype=DTYPE)/6 for p in ([1, 2, 3], [3, 1, 2], [2, 3, 1])]
        samples = [{"duration_days": torch.tensor(1., dtype=DTYPE), "time_days": torch.tensor(1., dtype=DTYPE),
                    "start_zone_power_w": values[0], "zone_power_w": values[1], "end_zone_power_w": values[2]}]
        self.assertEqual(float(sample_temporal_power(mapping, samples)["variance"]), 0.)

    def test_invalid_times_or_powers_are_rejected(self):
        """Verify invalid times or powers are rejected."""
        with self.assertRaises(ValueError):
            temporal_power([0., 0.], [[1.], [2.]], [1.])
        with self.assertRaises(ValueError):
            temporal_power([0., 1.], [[0.], [2.]], [1.])

    @unittest.skipUnless(study(1).result("history").is_file(),
                         "Optional regression requires locally generated WR-1 step-one results")
    def test_baseline_native_temporal_metric_and_burnup_floor(self):
        """Verify baseline native temporal metric and burnup floor."""
        workspace = study(1)
        history = load_state(workspace.result("history"))
        case = workspace.make_case(design_values=history["design"])
        fuel = [r for r in case["regions"] if r["depletable"]]
        mapping = FuelMap(SimpleNamespace(labels=[r["label"] for r in fuel], regions=fuel,
            symmetry_multiplier=6, multiplicity=np.array([r["physical_multiplicity"] for r in fuel])))
        profile = native_temporal_profile(history, mapping)
        options = study(2).settings("optimization")
        self.assertAlmostEqual(profile["variance"], options["temporal_variance_scale"], places=14)
        self.assertEqual(len(profile["time_days"]), 11)
        self.assertAlmostEqual(100*np.sqrt(profile["variance"]), 3.157409295319987, places=11)
        power = np.asarray(profile["channel_power_W"])
        np.testing.assert_allclose(power @ (mapping.channel_fractions.numpy()*6), 35e6, rtol=1e-12)
        self.assertAlmostEqual(options["minimum_discharge_burnup_MWd_tHM"],
            history["cycles"][-1]["metrics"]["discharge_burnup_MWd_tHM"]-5000., places=10)

    def test_step_context_reaches_design_and_native_case(self):
        """Verify step context reaches design and native case."""
        one, two = study(1), study(2)
        self.assertNotEqual(one.results, two.results)
        self.assertNotEqual(one.work, two.work)
        for name, parameter in two.design().named_parameters():
            np.testing.assert_array_equal(parameter.detach().numpy(), two.settings()["design"][name])
        self.assertNotEqual(float(one.design().cycle_days.detach()), float(two.design().cycle_days.detach()))
        values = two.settings()["design"]
        self.assertEqual(one.make_case(design_values=values), two.make_case(design_values=values))

    def test_burnup_is_a_constraint_without_a_reward_above_the_floor(self):
        """Verify burnup is a constraint without a reward above the floor."""
        two = study(2)
        options = two.settings("optimization")
        objective = two.settings()["objective"]
        floor = options["minimum_discharge_burnup_MWd_tHM"]
        for value, expected in ((floor+1000, 0.), (floor-50, -.01)):
            burnup = torch.tensor(value, dtype=DTYPE, requires_grad=True)
            variance = torch.tensor(options["temporal_variance_scale"], dtype=DTYPE, requires_grad=True)
            loss, score = design_loss({"discharge_burnup_MWd_tHM": burnup, "power_variance": variance},
                torch.tensor([1.01], dtype=DTYPE), torch.tensor(1e5), objective, options, torch.tensor([.005]),
                eol_keff=torch.tensor(1.002, dtype=DTYPE), eol_standard_error=.003)
            g_bu, g_power = torch.autograd.grad(loss, (burnup, variance))
            self.assertEqual(float(score.detach()), 1.)
            self.assertAlmostEqual(float(g_bu), expected)
            self.assertAlmostEqual(float(g_power), 1/options["temporal_variance_scale"])

    def test_candidate_selection_keeps_the_exact_burnup_floor(self):
        """Verify candidate selection keeps the exact burnup floor."""
        options = study(2).settings("optimization")
        floor = options["minimum_discharge_burnup_MWd_tHM"]
        feasible = {"metrics": {"discharge_burnup_MWd_tHM": floor}, "objective": 1.,
                    "minimum_keff": 1.002, "peak_constraint_passed": True,
                    "eol_corrected_keff": 1.002, "eol_standard_error": .003}
        violation = {**feasible, "metrics": {"discharge_burnup_MWd_tHM": floor-1}, "objective": 0.}
        self.assertLess(prediction_rank(feasible, options), prediction_rank(violation, options))

    def test_eol_penalty_has_the_analytic_torch_derivative(self):
        """Verify EOL penalty has the analytic Torch derivative."""
        two = study(2)
        options = two.settings("optimization")
        for value in (1.002, 1.004, 1.12):
            eol = torch.tensor(value, dtype=DTYPE, requires_grad=True)
            loss, _ = design_loss({
                "discharge_burnup_MWd_tHM": torch.tensor(options["minimum_discharge_burnup_MWd_tHM"]+1000),
                "power_variance": torch.tensor(options["temporal_variance_scale"], dtype=DTYPE)},
                torch.tensor([1.002, 1.02], dtype=DTYPE), torch.tensor(1e5), two.settings()["objective"],
                options, torch.tensor([.005, .005]), eol_keff=eol, eol_standard_error=.003)
            derivative, = torch.autograd.grad(loss, eol)
            self.assertAlmostEqual(float(derivative), 2*(value-1.002)/.003**2, places=9)
        self.assertGreater(float(loss.detach()), 1000.)

    def test_eol_uses_the_last_endpoint_of_cycle_three(self):
        """Verify EOL uses the last endpoint of cycle three."""
        options = study(2).settings("optimization")
        keys = ["cycle1/s4_end", "cycle3/s9_end", "cycle3/s10_mid", "cycle3/s10_end", "cycle3/s0_end"]
        self.assertEqual(eol_state_key(keys, options), "cycle3/s10_end")
        with self.assertRaises(ValueError):
            eol_state_key(["cycle1/s4_end"], options)

    def test_eol_allowance_comes_from_the_native_standard_error(self):
        """Verify EOL allowance comes from the native standard error."""
        options = study(2).settings("optimization")
        self.assertFalse(eol_check(1.008, .002, options)["passed"])
        self.assertTrue(eol_check(1.008, .004, options)["passed"])
        self.assertFalse(eol_check(1.12, .004, options)["passed"])
        self.assertEqual(eol_check(1.008, .004, options)["target"], 1.002)

    def test_low_k_elsewhere_does_not_make_high_eol_eligible(self):
        """Verify low k elsewhere does not make high EOL eligible."""
        options = study(2).settings("optimization")
        good = {"metrics": {"discharge_burnup_MWd_tHM": 25000.}, "objective": 1.,
                "minimum_keff": 1.002, "peak_constraint_passed": True,
                "eol_corrected_keff": 1.002, "eol_standard_error": .003}
        high_eol = {**good, "objective": 0., "eol_corrected_keff": 1.12}
        self.assertLess(prediction_rank(good, options), prediction_rank(high_eol, options))

    def test_native_acceptance_requires_near_critical_eol(self):
        """Verify native acceptance requires near critical EOL."""
        two = study(2)
        options = two.settings("optimization")
        native = {"feasible": True, "physics_passed": True, "design": two.settings()["design"],
                  "metrics": {"discharge_burnup_MWd_tHM": 25000., "power_variance": .0001},
                  "merit": .1, "eol_keff_by_cycle": {"3": [1.12, .003]}}
        assess_native(native, options)
        self.assertFalse(native["acceptable"])
        native["eol_keff_by_cycle"]["3"] = [1.004, .003]
        assess_native(native, options)
        self.assertTrue(native["acceptable"])

    def test_projected_adam_improves_a_constrained_problem(self):
        """Verify projected adam improves a constrained problem."""
        x = torch.nn.Parameter(torch.tensor([1., 1.], dtype=DTYPE))
        before = x.detach().clone()
        desired = torch.tensor([2., 0.], dtype=DTYPE)
        optimizer = torch.optim.Adam([x], lr=.1)
        loss = (x-desired).square().sum()
        loss.backward()
        optimizer.step()
        normal = torch.tensor([1., 2.], dtype=DTYPE)
        delta = bounded_linear_projection(x.detach()-before, normal,
            torch.tensor([.05, .05], dtype=DTYPE), -before, 3-before, 0.)
        self.assertAlmostEqual(float(normal @ delta), 0., places=12)
        self.assertGreater(float(delta.norm()), .1)
        self.assertLess(float((before+delta-desired).square().sum()), float(loss.detach()))

    def test_eol_projection_redistributes_a_bound_limited_correction(self):
        """Verify EOL projection redistributes a bound limited correction."""
        delta = bounded_linear_projection(torch.tensor([0., 1.], dtype=DTYPE),
            torch.tensor([2., 1.], dtype=DTYPE), torch.ones(2, dtype=DTYPE),
            torch.tensor([0., -1.], dtype=DTYPE), torch.ones(2, dtype=DTYPE), .5)
        torch.testing.assert_close(delta, torch.tensor([0., .5], dtype=DTYPE))

    def test_eol_projection_is_invariant_to_parameter_units(self):
        """Verify EOL projection is invariant to parameter units."""
        delta, normal, metric, lower, upper = [torch.tensor(x, dtype=DTYPE) for x in
            ([2., -.02], [.2, 5.], [100., .001], [-3., -.1], [3., .1])]
        a = bounded_linear_projection(delta, normal, metric, lower, upper, .05)
        units = torch.tensor([.001, 1.], dtype=DTYPE)
        b = bounded_linear_projection(delta*units, normal/units, metric*units.square(),
                                      lower*units, upper*units, .05)
        torch.testing.assert_close(a, b/units, rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(float(normal @ a), .05, places=12)

    def test_unreachable_linear_eol_target_respects_bounds(self):
        """Verify unreachable linear EOL target respects bounds."""
        delta = bounded_linear_projection(torch.zeros(2, dtype=DTYPE), torch.tensor([1., 2.], dtype=DTYPE),
            torch.ones(2, dtype=DTYPE), -torch.ones(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE), 10.)
        torch.testing.assert_close(delta, torch.ones(2, dtype=DTYPE))


if __name__ == "__main__":
    unittest.main()
