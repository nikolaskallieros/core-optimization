"""Graphite material partition and analytic geometry derivatives; no finite differences."""
from designs.wr_1 import WR1_ROOT
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import torch

from diffusion_surrogate.hexagonal import C6Grid
from geometry.intersections import circle_triangle_area
from geometry.diffusion import material_fractions
from designs.wr_1.build_geometry import parameterized_geometry
from designs.wr_1.model.case import study
from designs.wr_1.model.graphite import add_graphite, graphite_mapping, initialize_graphite
from optimization.objectives import joint_merit
from diffusion_surrogate.regional import shared_geometry
from diffusion_surrogate.symmetry import SectorInputError

ROOT = WR1_ROOT


class GraphiteObjectiveTests(unittest.TestCase):
    def test_reference_zero_and_correct_analytic_derivatives(self):
        """Verify reference zero and correct analytic derivatives."""
        bu = torch.tensor(25000., dtype=torch.float64, requires_grad=True)
        variance = torch.tensor(.0009, dtype=torch.float64, requires_grad=True)
        bol = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
        reference = {"burnup_MWd_tHM": 25000., "temporal_variance": .0009,
                     "bol_reactivity": 1.-1./1.2,
                     "weights": {"burnup": 1., "temporal_variance": 1., "bol_reactivity": 1.}}
        score = joint_merit(bu, variance, bol, reference)
        gradient = torch.autograd.grad(score, (bu, variance, bol))
        self.assertAlmostEqual(float(score.detach()), 0., places=12)
        self.assertAlmostEqual(float(gradient[0]), -1./25000., places=12)
        self.assertAlmostEqual(float(gradient[1]), 1./.0009, places=9)
        self.assertAlmostEqual(float(gradient[2]), 1./(1.2**2*reference["bol_reactivity"]), places=12)


class GraphiteGeometryTests(unittest.TestCase):
    def setUp(self):
        """Add a manufactured six-cylinder graphite geometry fixture."""
        self.workflow = study(3)
        self.design = self.workflow.design()
        self.design.graphite_pitch_mm = torch.nn.Parameter(torch.tensor([135.], dtype=torch.float64))
        self.design.graphite_radius_mm = torch.nn.Parameter(torch.tensor([30.], dtype=torch.float64))
        coordinates = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        self.spec = {"sites": [{"id": f"graphite_test_{i}", "q": q, "r": r, "radius_group": 0}
                               for i, (q, r) in enumerate(coordinates)],
                     "rotation_deg": 30., "minimum_clearance_mm": 1.,
                     "density_g_cm3": 1.7, "temperature_K": 327.15,
                     "material_basis": "Pure graphite fixture; full active height."}
        initialize_graphite(self.design, self.spec)
        self.case = self.workflow.make_case()

    def mapped(self):
        """Combine live fuel-channel and graphite geometry outputs.

        Returns
        -------
        dict
            Merged differentiable geometry fields.
        """
        return {**self.design(), **graphite_mapping(self.design)}

    def append(self):
        """Add the fixture's graphite regions to its native case.

        Returns
        -------
        dict
            Native case containing the explicit graphite cylinders.
        """
        return add_graphite(self.case, self.design, self.mapped(), self.case["cross_sections"])

    def test_volume_and_fuel_inventory_preserved(self):
        """Verify volume and fuel inventory preserved."""
        original = deepcopy(self.case)
        self.append()
        expected = math.pi*3.**2*297.  # Six full cylinders / sixfold sector.
        carbon = [r for r in self.case["regions"] if r.get("equivalence_material") == "graphite"]
        water_before = sum(r["volume_cm3"] for r in original["regions"] if r["kind"] == "wr1_background")
        water_after = sum(r["volume_cm3"] for r in self.case["regions"] if r["kind"] == "wr1_background")
        self.assertAlmostEqual(sum(r["volume_cm3"] for r in carbon), expected, places=7)
        self.assertAlmostEqual(water_before-water_after, expected, places=7)
        self.assertAlmostEqual(sum(r["volume_cm3"] for r in original["regions"]),
                               sum(r["volume_cm3"] for r in self.case["regions"]), places=7)
        self.assertEqual(original["physical_assembly_to_region"], self.case["physical_assembly_to_region"])
        for i, region in enumerate(original["regions"]):
            if region["kind"] != "wr1_background":
                self.assertEqual(region, self.case["regions"][i])
        self.assertEqual(len(self.case["physical_assembly_to_region"]), 111)

    def test_carbon_binding_and_boron_region_separation(self):
        """Verify carbon binding and boron region separation."""
        original_boron = deepcopy(self.case["boron_response"])
        self.append()
        carbon = [r for r in self.case["regions"] if r.get("equivalence_material") == "graphite"]
        self.assertEqual(len(carbon), 3)
        self.assertTrue(all(r["thermal_scattering"] == ["c_Graphite"] and not r["depletable"] for r in carbon))
        self.assertTrue(all(r["atom_density"] == {"C0": .08523471094462945} for r in carbon))
        self.assertEqual(original_boron, self.case["boron_response"])

    def test_rejects_collision_with_centre_channel(self):
        """Verify rejects collision with centre channel."""
        with torch.no_grad():
            self.design.graphite_pitch_mm.fill_(25.)
        with self.assertRaisesRegex(ValueError, "Graphite cylinders overlap"):
            self.append()

    def test_rejects_broken_sixfold_symmetry(self):
        """Verify rejects broken sixfold symmetry."""
        with torch.no_grad():
            self.design.graphite_unit_xy[0, 0] += .01
        with self.assertRaises(SectorInputError):
            self.append()

    def test_analytic_volume_derivatives_for_radius_and_translation(self):
        """Verify analytic volume derivatives for radius and translation."""
        grid = C6Grid(self.case["geometry"]["apothem_cm"], 16, [0., 99., 198., 297.])
        mapped = graphite_mapping(self.design)
        volume = sum(circle_triangle_area(grid.vertices, centre/10, radius/10).sum()*297.
                     for centre, radius in zip(mapped["graphite_xy_mm"], mapped["graphite_radius_mm"]))
        pitch_gradient, radius_gradient = torch.autograd.grad(
            volume, (self.design.graphite_pitch_mm, self.design.graphite_radius_mm))
        self.assertAlmostEqual(float(volume.detach()), math.pi*3.**2*297., places=7)
        self.assertAlmostEqual(float(pitch_gradient[0]), 0., places=7)
        self.assertAlmostEqual(float(radius_gradient[0]), 2*math.pi*3.*297./10., places=7)

    def test_shared_partition_preserves_values_and_analytic_gradients(self):
        """Verify shared partition preserves values and analytic gradients."""
        self.append()
        grid = C6Grid(self.case["geometry"]["apothem_cm"], 8, [0., 99., 198., 297.])
        models = [SimpleNamespace(design=self.design, solver=SimpleNamespace(grid=grid), case=self.case, geometry_key="fixture",
                  geometry=lambda: material_fractions(parameterized_geometry(self.design, self.case), grid)) for _ in range(3)]
        points = {str(i): SimpleNamespace(model=model) for i, model in enumerate(models)}
        with shared_geometry(points):
            fractions = models[0]._live_fractions
            self.assertTrue(all(m._live_fractions is fractions for m in models))
            volume = sum((grid.volumes @ m._live_fractions[:, -3:]).sum() for m in models)
        self.assertTrue(all(m._live_fractions is None for m in models))
        cached = torch.autograd.grad(volume, (self.design.graphite_pitch_mm, self.design.graphite_radius_mm))
        direct = sum((grid.volumes @ m.geometry()[:, -3:]).sum() for m in models)
        plain = torch.autograd.grad(direct, (self.design.graphite_pitch_mm, self.design.graphite_radius_mm))
        torch.testing.assert_close(volume, direct, rtol=1e-12, atol=1e-7)
        for a, b in zip(cached, plain):
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-7)
        self.assertAlmostEqual(float(cached[1][0]), 3*2*math.pi*3.*297./10., places=7)
        with torch.no_grad():
            self.design.graphite_radius_mm.add_(1.)
        with shared_geometry(points):
            self.assertIsNot(models[0]._live_fractions, fractions)
            self.assertGreater(float((grid.volumes @ models[0]._live_fractions[:, -3:]).sum().detach()),
                               float(volume.detach())/3)


if __name__ == "__main__":
    unittest.main()
