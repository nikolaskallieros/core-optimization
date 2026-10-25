"""Analytic BOL-objective and per-channel constraint checks, without transport or FD."""
from designs.wr_1 import WR1_ROOT
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from optimization.solvers import minimize
from designs.wr_1.model.case import study
from fuel_cycle.residence import FuelMap
from coupling.optimization import (assess_native, bol_state_key, channel_temporal_check,
    check_residence_gradient_trace, corrected_keff, design_loss, keff_correction, merit,
    native_summary, prediction_rank, rescore_prediction, reset_projected_momentum)
from runtime.io import load_state

ROOT = WR1_ROOT
DTYPE = torch.float64


class ReactivityTests(unittest.TestCase):
    def setUp(self):
        """Configure a BOL-reactivity objective with temporal constraints."""
        self.workspace = study(2)
        self.config = self.workspace.settings("optimization")
        self.config.update(objective_kind="bol_reactivity", bol_reactivity={"cycle": 3, "reference": .2},
            minimum_discharge_burnup_MWd_tHM=23000., maximum_channel_temporal_rms=.03,
            channel_temporal_variance_penalty_scale=.2)
        self.objective = self.workspace.settings()["objective"]

    def loss(self, bol, burnup, variances):
        """Evaluate reactivity merit with burnup and per-channel penalties.

        Parameters
        ----------
        bol : torch.Tensor
            Live beginning-of-life effective multiplication factor.
        burnup : torch.Tensor
            Live discharge burnup in MWd/tHM.
        variances : torch.Tensor
            Individual dimensionless temporal channel-power variances.

        Returns
        -------
        tuple of torch.Tensor
            Penalized design loss and unpenalized reactivity merit.
        """
        metrics = {"discharge_burnup_MWd_tHM": burnup, "power_variance": variances.mean()}
        return design_loss(metrics, torch.stack((bol, bol.new_tensor(1.002))), bol.new_tensor(1e5),
            self.objective, self.config, bol.new_tensor([.01, .01]), bol_keff=bol,
            eol_keff=bol.new_tensor(1.002), eol_standard_error=.003,
            channel_temporal_variance=variances)

    def test_bol_uses_post_loading_cycle_three_state(self):
        """Verify BOL uses post loading cycle three state."""
        keys = ["cycle1/bol", "cycle2/bol", "cycle3/bol", "cycle3/s4_end"]
        self.assertEqual(bol_state_key(keys, self.config), "cycle3/bol")
        with self.assertRaises(ValueError):
            bol_state_key(["cycle1/bol", "cycle3/s4_end"], self.config)

    @unittest.skipUnless(study(2).result("history").is_file(),
                         "Optional regression requires locally generated WR-1 step-two results")
    def test_complete_native_history_keeps_measured_bol_and_each_channel(self):
        """Verify complete native history keeps measured BOL and each channel."""
        history = load_state(self.workspace.result("history"))
        case = self.workspace.make_case(design_values=history["design"])
        fuel = [r for r in case["regions"] if r["depletable"]]
        mapping = FuelMap(SimpleNamespace(labels=[r["label"] for r in fuel], regions=fuel,
            symmetry_multiplier=6, multiplicity=np.array([r["physical_multiplicity"] for r in fuel])))
        result = native_summary(history, mapping, self.workspace.settings(), self.config)
        expected = history["observations"]["cycle3/bol"]["keff"][0]
        self.assertEqual(result["bol_state"], "cycle3/bol")
        self.assertEqual(float(result["bol_keff"][0]), float(expected))
        self.assertAlmostEqual(result["merit"], (1.-1./expected)/.2, places=14)
        variance = result["power_profile"]["temporal"]["variance_by_channel"]
        self.assertEqual(len(variance), 7)
        self.assertAlmostEqual(result["channel_temporal_check"]["maximum_rms"]**2, max(variance), places=15)
        self.assertEqual(result["flatness_passed"], max(variance) <= .03**2+1e-12)

    def test_inactive_later_constraints_allow_zero_sensitivity_but_not_a_missing_graph(self):
        """Verify inactive later constraints allow zero sensitivity but not a missing graph."""
        history = {"cycles": [{"cycle": c, "segments": [{"end": f"cycle{c}/s0_end"}]} for c in (1, 2, 3)]}
        trace = [{"stage": f"cycle{c}/s0_end", "inventory_log_sensitivity_norm": value}
                 for c, value in ((1, 1.), (2, 2.), (3, 0.))]
        result = check_residence_gradient_trace(trace, history, self.config)
        self.assertEqual(result["recorded_intervals"], 3)
        self.assertEqual(result["required_sensitive_cycles"], [1, 2])
        with self.assertRaises(ArithmeticError):
            check_residence_gradient_trace(trace[:-1], history, self.config)
        with self.assertRaises(ArithmeticError):
            check_residence_gradient_trace(trace, history, {"objective_kind": "channel_temporal"})

    def test_reactivity_derivative_and_no_reward_beyond_constraints(self):
        """Verify reactivity derivative and no reward beyond constraints."""
        bol = torch.tensor(1.2, dtype=DTYPE, requires_grad=True)
        burnup = torch.tensor(24000., dtype=DTYPE, requires_grad=True)
        variances = torch.tensor([.0001, .0004], dtype=DTYPE, requires_grad=True)
        loss, score = self.loss(bol, burnup, variances)
        gb, gu, gp = torch.autograd.grad(loss, (bol, burnup, variances))
        self.assertAlmostEqual(float(score.detach()), (1.-1./1.2)/.2, places=14)
        self.assertAlmostEqual(float(gb), 1./(1.2**2*.2), places=14)
        self.assertEqual(float(gu), 0.)
        torch.testing.assert_close(gp, torch.zeros_like(gp))

    def test_burnup_floor_keeps_its_analytic_penalty(self):
        """Verify burnup floor keeps its analytic penalty."""
        bol = torch.tensor(1.2, dtype=DTYPE)
        burnup = torch.tensor(22950., dtype=DTYPE, requires_grad=True)
        loss, _ = self.loss(bol, burnup, torch.tensor([.0004], dtype=DTYPE))
        derivative, = torch.autograd.grad(loss, burnup)
        self.assertAlmostEqual(float(derivative), -.01, places=14)

    def test_each_exceeded_channel_has_a_physical_torch_gradient(self):
        """Verify each exceeded channel has a physical Torch gradient."""
        bol = torch.tensor(1.2, dtype=DTYPE)
        variances = torch.tensor([.0001, .0016, .001], dtype=DTYPE, requires_grad=True)
        loss, _ = self.loss(bol, bol.new_tensor(24000.), variances)
        derivative, = torch.autograd.grad(loss, variances)
        expected = 2*torch.relu(variances.detach()/.03**2-1.)/(.2**2*.03**2)
        torch.testing.assert_close(derivative, expected, rtol=1e-13, atol=1e-12)

    def test_low_core_average_does_not_hide_a_failing_channel(self):
        """Verify low core average does not hide a failing channel."""
        native = {"feasible": True, "physics_passed": True, "design": self.workspace.settings()["design"],
            "metrics": {"discharge_burnup_MWd_tHM": 24000., "power_variance": .07**2/37},
            "power_profile": {"temporal": {"variance_by_channel": [.07**2, 0.]}},
            "merit": .5, "eol_keff_by_cycle": {"3": [1.002, .003]}}
        assess_native(native, self.config)
        self.assertTrue(native["feasible"])
        self.assertFalse(native["acceptable"])
        self.assertEqual(native["channel_temporal_check"]["missed_channel_indices"], [0])
        native["power_profile"]["temporal"]["variance_by_channel"] = [.03**2, .02**2]
        assess_native(native, self.config)
        self.assertTrue(native["acceptable"])

    def test_candidate_ranking_keeps_the_individual_channel_limit(self):
        """Verify candidate ranking keeps the individual channel limit."""
        good = {"metrics": {"discharge_burnup_MWd_tHM": 24000.}, "objective": 1.,
                "minimum_keff": 1.002, "peak_constraint_passed": True,
                "eol_corrected_keff": 1.002, "eol_standard_error": .003,
                "channel_temporal_variance": [.0004, .0004]}
        bad = {**good, "objective": 0., "channel_temporal_variance": [.0001, .0016]}
        self.assertLess(prediction_rank(good, self.config), prediction_rank(bad, self.config))

    def test_bol_correction_is_measured_and_keeps_the_design_derivative(self):
        """Verify BOL correction is measured and keeps the design derivative."""
        observations = {"cycle3/bol": {"keff": [1.22, .003]}, "cycle3/s4_end": {"keff": [1.003, .003]}}
        fitted = {"cycle3/bol": {"keff": 1.18}, "cycle3/s4_end": {"keff": .98}}
        correction = keff_correction(observations, fitted, self.config)
        raw = torch.tensor([1.18, .98], dtype=DTYPE, requires_grad=True)
        corrected, _ = corrected_keff(raw, sorted(observations), correction)
        score = merit(raw.new_tensor(24000.), raw.new_tensor(.0004), self.objective, self.config,
                      bol_keff=corrected[0])
        derivative, = torch.autograd.grad(score, raw)
        torch.testing.assert_close(corrected.detach(), raw.new_tensor([1.22, 1.003]))
        torch.testing.assert_close(derivative, raw.new_tensor([1./(1.22**2*.2), 0.]))
        prediction = {"keff_by_state": {k: v["keff"] for k, v in fitted.items()},
                      "metrics": {"discharge_burnup_MWd_tHM": 24000., "power_variance": .0004},
                      "channel_temporal_variance": [.0004, .0004], "maximum_peak_burnup_MWd_tHM": 40000.}
        rescored = rescore_prediction(prediction, self.objective, self.config, correction)
        self.assertAlmostEqual(rescored["bol_corrected_keff"], 1.22, places=14)
        self.assertAlmostEqual(rescored["eol_corrected_keff"], 1.003, places=14)

    def test_missing_or_invalid_channel_data_cannot_pass(self):
        """Verify missing or invalid channel data cannot pass."""
        for values in ([], [-.0001], [float("nan")], [[.0004]]):
            with self.assertRaises(ValueError):
                channel_temporal_check(values, self.config)
        with self.assertRaises(ValueError):
            design_loss({"discharge_burnup_MWd_tHM": torch.tensor(24000.), "power_variance": torch.tensor(.0004)},
                torch.tensor([1.2]), torch.tensor(40000.), self.objective, self.config,
                bol_keff=torch.tensor(1.2), eol_keff=torch.tensor(1.002), eol_standard_error=.003)


class ProjectedMomentumTests(unittest.TestCase):
    def trial(self, *, stale_momentum=True, anchor_feasible=True):
        """Exercise one projected Adam step with optional stale momentum.

        Parameters
        ----------
        stale_momentum : bool, optional
            Seed the free coordinate with momentum opposing the new gradient.
        anchor_feasible : bool, optional
            Allow non-descent momentum reset only when the origin is feasible.

        Returns
        -------
        tuple
            Final parameters, Adam state, descent diagnostics and optimization result.
        """
        design = torch.nn.ParameterDict({"x": torch.nn.Parameter(torch.ones(2, dtype=DTYPE))})
        p = design["x"]
        optimizer = torch.optim.Adam(design.parameters(), lr=.1)
        p.grad = torch.tensor([1., -10. if stale_momentum else 1.], dtype=DTYPE)
        optimizer.step()
        with torch.no_grad():
            p.fill_(1.)
        forwards, backwards, diagnostic = [], [], {}
        p.register_hook(lambda gradient: backwards.append(gradient.detach().clone()))

        def model():
            """Record one forward call and return a quadratic fixture loss.

            Returns
            -------
            torch.Tensor
                Half the squared parameter norm.
            """
            forwards.append(p.detach().clone())
            return .5 * p.square().sum()

        def project(d, initial):
            """Fix one coordinate and apply projected-momentum correction.

            Parameters
            ----------
            d : torch.nn.ParameterDict
                Two-coordinate design to project.
            initial : mapping
                Parameter values before the optimizer step.
            """
            def constrain():
                """Restore the first coordinate to its fixed initial value."""
                d["x"][0] = initial["x"][0]
            constrain()
            diagnostic.update(reset_projected_momentum(
                d, initial, optimizer, constrain, anchor_feasible=anchor_feasible))

        result = minimize(design, model, lambda loss: loss, optimizer=optimizer, steps=1, project=project)
        self.assertEqual(len(forwards), 2)
        self.assertEqual(len(backwards), 1)
        torch.testing.assert_close(p.grad, torch.ones_like(p))
        self.assertEqual(float(p[0].detach()), 1.)
        self.assertEqual(optimizer.param_groups[0]["lr"], .1)
        return p.detach(), optimizer.state[p], diagnostic, result

    def test_stale_projected_momentum_becomes_descent_with_the_same_gradient(self):
        """Verify stale projected momentum becomes descent with the same gradient."""
        point, state, diagnostic, result = self.trial()
        self.assertTrue(diagnostic["optimizer_moments_reset"])
        self.assertGreater(diagnostic["directional_derivative_before"], 0.)
        self.assertLess(diagnostic["directional_derivative_after"], 0.)
        self.assertEqual(int(state["step"]), 1)
        self.assertLess(float(.5*point.square().sum()), result["objective_before"])

    def test_a_descent_direction_keeps_its_moments(self):
        """Verify a descent direction keeps its moments."""
        point, state, diagnostic, result = self.trial(stale_momentum=False)
        self.assertFalse(diagnostic["optimizer_moments_reset"])
        self.assertLess(diagnostic["directional_derivative_before"], 0.)
        self.assertEqual(int(state["step"]), 2)
        self.assertLess(float(.5*point.square().sum()), result["objective_before"])

    def test_infeasible_anchor_keeps_its_constraint_repair_proposal(self):
        """Verify infeasible anchor keeps its constraint repair proposal."""
        point, state, diagnostic, result = self.trial(anchor_feasible=False)
        self.assertFalse(diagnostic["optimizer_moments_reset"])
        self.assertEqual(int(state["step"]), 2)
        self.assertEqual(diagnostic["directional_derivative_before"], diagnostic["directional_derivative_after"])
        self.assertGreater(float(.5*point.square().sum()), result["objective_before"])


if __name__ == "__main__":
    unittest.main()
