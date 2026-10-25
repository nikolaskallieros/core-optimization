"""Analytical mass and constrained-proposal checks; no finite differences."""
from runtime.paths import PROJECT_ROOT
from designs.wr_1 import WR1_ROOT
import copy
import math
import sys
import unittest

ROOT = WR1_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from optimization.fuel_constraints import (fuel_mass_check, fuel_mass_kg, heavy_metal_mass_kg,
                                      native_fuel_mass_check)
from optimization.solvers import project_quadratic_bound
from designs.wr_1.model.case import study


class FuelMassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load scenario-six design and fresh fuel-mass constraints."""
        workflow = study(6)
        cls.options = workflow.settings("optimization")
        cls.design = workflow.settings()["design"]
        cls.policy = cls.options["fresh_fuel_mass"]

    def test_baseline_matches_declared_fuel_and_hm_reference(self):
        """Verify baseline matches declared fuel and hm reference."""
        result = fuel_mass_check(self.design, self.policy)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["fuel_mass_kg"], 1147.6632485013556, places=9)
        self.assertAlmostEqual(result["initial_heavy_metal_mass_kg"], 1011.6112329801433, places=9)

    def test_counts_include_three_assemblies_and_physical_symmetry(self):
        """Verify counts include three assemblies and physical symmetry."""
        basis = self.policy["basis"]
        self.assertEqual(basis["physical_channels_by_ring"], [1, 6, 12, 18])
        self.assertEqual(sum(basis["physical_channels_by_ring"]) * basis["axial_assemblies_per_channel"], 111)
        weights = basis["fuel_mass_kg_per_mm2_by_ring"]
        for count, weight in zip([1, 6, 12, 18], weights):
            self.assertAlmostEqual(weight / weights[0], count, places=12)

    def test_native_volume_and_isotope_accounting_are_independent(self):
        """Verify native volume and isotope accounting are independent."""
        workflow = study(1)
        values = {name: self.design[name] for name, _ in workflow.design().named_parameters()}
        case = workflow.make_case(design_values=values)
        check = native_fuel_mass_check(case, self.policy)
        self.assertTrue(check["passed"])
        self.assertEqual(check["physical_axial_assemblies"], 111)
        self.assertLess(max(check["native_material_accounting"].values()), 1e-12)
        altered = copy.deepcopy(case)
        next(row for row in altered["regions"] if row["depletable"])["volume_cm3"] *= 1.01
        with self.assertRaisesRegex(ArithmeticError, "disagree"):
            native_fuel_mass_check(altered, self.policy)

    def test_below_and_above_ceiling(self):
        """Verify below and above ceiling."""
        for ratio, expected in ((.999, True), (1.001, False)):
            values = copy.deepcopy(self.design)
            values["element_radius_mm"] = [ratio * x for x in values["element_radius_mm"]]
            self.assertEqual(fuel_mass_check(values, self.policy)["passed"], expected)

    def test_enrichment_changes_hm_but_not_fuel_mass(self):
        """Verify enrichment changes hm but not fuel mass."""
        values = copy.deepcopy(self.design)
        values["enrichment"] = [.05] * 4
        result = fuel_mass_check(values, self.policy)
        baseline = fuel_mass_check(self.design, self.policy)
        self.assertEqual(result["fuel_mass_kg"], baseline["fuel_mass_kg"])
        self.assertNotEqual(result["initial_heavy_metal_mass_kg"], baseline["initial_heavy_metal_mass_kg"])
        values["graphite_thickness_mm"] = [60.] * 21
        self.assertEqual(fuel_mass_check(values, self.policy), result)

    def test_feasible_radius_projection_is_unchanged(self):
        """Verify feasible radius projection is unchanged."""
        values = [25.] * 4
        self.assertEqual(project_quadratic_bound(values, [25.] * 4, [60.] * 4,
            self.policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            self.policy["maximum_fresh_fuel_mass_kg"]), values)

    def test_active_lower_bounds_and_mass_ceiling(self):
        """Verify active lower bounds and mass ceiling."""
        result = project_quadratic_bound([28., 25., 25., 25.], [25.] * 4, [60.] * 4,
            self.policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            self.policy["maximum_fresh_fuel_mass_kg"])
        self.assertEqual(result[1:], [25.] * 3)
        self.assertGreater(result[0], 25.)
        self.assertLess(result[0], 28.)
        self.assertLessEqual(fuel_mass_kg(result, self.policy["basis"]), self.policy["maximum_fresh_fuel_mass_kg"])
        self.assertAlmostEqual(fuel_mass_kg(result, self.policy["basis"]), self.policy["maximum_fresh_fuel_mass_kg"], places=9)

    def test_nonuniform_metric_and_upper_bound(self):
        """Verify nonuniform metric and upper bound."""
        result = project_quadratic_bound([35., 29., 28., 27.], [25.] * 4, [25.01, 60., 60., 60.],
            self.policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            self.policy["maximum_fresh_fuel_mass_kg"],
            metric=[.1, 1., 3., 5.])
        self.assertLessEqual(result[0], 25.01)
        self.assertTrue(all(25. <= r <= 60. for r in result))
        self.assertLessEqual(fuel_mass_kg(result, self.policy["basis"]), self.policy["maximum_fresh_fuel_mass_kg"])

    def test_impossible_radius_box_is_rejected(self):
        """Verify impossible radius box is rejected."""
        with self.assertRaisesRegex(ValueError, "bounds"):
            project_quadratic_bound([26.] * 4, [26.] * 4, [60.] * 4,
            self.policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            self.policy["maximum_fresh_fuel_mass_kg"])

    def test_invalid_mass_ceiling_is_rejected_before_bracketing(self):
        """Verify invalid mass ceiling is rejected before bracketing."""
        for limit in (0., -1., float("nan"), float("inf")):
            policy = {**self.policy, "maximum_fresh_fuel_mass_kg": limit}
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                project_quadratic_bound([26.] * 4, [25.] * 4, [60.] * 4,
            policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            policy["maximum_fresh_fuel_mass_kg"])

    def test_roundoff_at_minimum_mass_terminates(self):
        """Verify roundoff at minimum mass terminates."""
        minimum = float(fuel_mass_kg([25.] * 4, self.policy["basis"]))
        policy = {**self.policy, "maximum_fresh_fuel_mass_kg": math.nextafter(minimum, 0.)}
        result = project_quadratic_bound([26.] * 4, [25.] * 4, [60.] * 4,
            policy["basis"]["fuel_mass_kg_per_mm2_by_ring"],
            policy["maximum_fresh_fuel_mass_kg"])
        self.assertEqual(result, [25.] * 4)
        self.assertTrue(fuel_mass_check({**self.design, "element_radius_mm": result}, policy)["passed"])

    def test_exact_autograd_mass_derivatives(self):
        """Verify exact autograd mass derivatives."""
        import torch
        radii = torch.tensor(self.design["element_radius_mm"], dtype=torch.float64, requires_grad=True)
        enrichment = torch.tensor(self.design["enrichment"], dtype=torch.float64, requires_grad=True)
        mass = fuel_mass_kg(radii, self.policy["basis"])
        gradient = torch.autograd.grad(mass, radii)[0]
        expected = 2. * radii.detach() * radii.new_tensor(self.policy["basis"]["fuel_mass_kg_per_mm2_by_ring"])
        torch.testing.assert_close(gradient, expected, rtol=1e-14, atol=0.)
        hm = heavy_metal_mass_kg(radii, enrichment, self.policy["basis"])
        derivatives = torch.autograd.grad(hm, (radii, enrichment))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in derivatives))
        self.assertTrue(bool((derivatives[1] < 0).all()))

    def test_eol_recovery_preserves_mass_projected_radius(self):
        """Verify EOL recovery preserves mass projected radius."""
        import torch
        from optimization.solvers import bounded_linear_projection
        origin = torch.tensor([25., 1., 1.], dtype=torch.float64)
        proposed = torch.tensor([25., 1., 1.], dtype=torch.float64)
        normal = torch.tensor([2., 1., 1.], dtype=torch.float64)
        result = origin + bounded_linear_projection(
            proposed - origin, normal, torch.tensor([0., 1., 1.], dtype=torch.float64),
            torch.tensor([0., -1., -1.]), torch.tensor([1., .05, 1.]), .2)
        self.assertEqual(result[0], proposed[0])
        self.assertAlmostEqual(float(normal @ (result - origin)), .2, places=12)
        self.assertAlmostEqual(float(result[1] - origin[1]), .05, places=7)

    def test_objective_has_no_power_bol_or_bu_limits(self):
        """Verify objective has no power BOL or bu limits."""
        self.assertEqual(self.options["objective_kind"], "channel_burnup")
        self.assertEqual(self.options["power_weight"], 0.)
        for name in ("minimum_discharge_burnup_MWd_tHM", "maximum_native_power_variance",
                     "maximum_channel_temporal_rms", "joint_objective", "bol_reactivity"):
            self.assertNotIn(name, self.options)
        self.assertEqual(self.options["bounds"]["enrichment"], [.007, .05])
        self.assertEqual(self.options["eol_keff"]["target"], 1.002)
        self.assertEqual(self.options["steps_per_refresh"], 5)


if __name__ == "__main__":
    unittest.main()
