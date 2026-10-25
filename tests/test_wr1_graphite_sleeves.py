"""Axial sleeve geometry, native cell partition and analytic derivatives; no FD."""
from examples.wr_1 import WR1_ROOT
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diffusion_surrogate.hexagonal import C6Grid
from geometry.diffusion import material_fractions
from examples.wr_1.build_geometry import parameterized_geometry
from examples.wr_1.model.case import study
from coupling.optimization import project
from optimization.design import set_parameters
from coupling.openmc import build
from runtime.io import load_state, save_state

ROOT = WR1_ROOT


class GraphiteSleeveTests(unittest.TestCase):
    def setUp(self):
        """Build varying graphite sleeves and their unsleeved comparison."""
        self.workflow = study(5)
        self.values = deepcopy(self.workflow.settings()["design"])
        self.values["graphite_thickness_mm"] = [2., 7., 15.]*7
        self.design = self.workflow.design()
        set_parameters(self.design, self.values)
        self.case = self.workflow.make_case(design_values=self.values)
        self.base = study(3)
        self.base_values = {k: v for k, v in self.values.items() if k != "graphite_thickness_mm"}

    def test_only_axially_matched_background_water_is_displaced(self):
        """Verify only axially matched background water is displaced."""
        before = self.base.make_case(design_values=self.base_values)
        carbon = [r for r in self.case["regions"] if r.get("graphite_sleeve")]
        self.assertEqual(len(carbon), 21)
        self.assertEqual(sum(r["physical_multiplicity"] for r in carbon), 111)
        self.assertEqual(before["physical_assembly_to_region"], self.case["physical_assembly_to_region"])
        for i, region in enumerate(before["regions"]):
            if region["kind"] == "wr1_background":
                level = int(region["label"].rpartition("/axial/")[2])
                removed = sum(r["volume_cm3"] for r in carbon if r["axial_level"] == level)
                self.assertAlmostEqual(region["volume_cm3"]-self.case["regions"][i]["volume_cm3"], removed, places=7)
            else:
                self.assertEqual(region, self.case["regions"][i])
        self.assertEqual(before["boron_response"], self.case["boron_response"])
        self.assertTrue(all(not r["depletable"] and r["thermal_scattering"] == ["c_Graphite"] for r in carbon))

    def test_exact_partition_and_independent_analytic_thickness_derivatives(self):
        """Verify exact partition and independent analytic thickness derivatives."""
        grid = C6Grid(self.case["geometry"]["apothem_cm"], 8, [0., 99., 198., 297.])
        volumes = grid.volumes @ material_fractions(parameterized_geometry(self.design, self.case), grid)
        np.testing.assert_allclose(volumes.detach(), [r["volume_cm3"] for r in self.case["regions"]], rtol=2e-10, atol=1e-6)
        carbon = [(i, r) for i, r in enumerate(self.case["regions"]) if r.get("graphite_sleeve")]
        for j, (i, region) in enumerate(carbon):
            gradient = torch.autograd.grad(volumes[i], self.design.graphite_thickness_mm,
                                           retain_graph=j < len(carbon)-1)[0]
            expected = torch.zeros(21, dtype=torch.float64)
            radius = region["pieces"][0]["outer_radius_cm"]
            expected[j] = region["physical_multiplicity"]/6*2*math.pi*radius*99./10
            torch.testing.assert_close(gradient, expected, rtol=1e-10, atol=1e-6)

    def test_zero_thickness_recovers_no_graphite_partition(self):
        """Verify zero thickness recovers no graphite partition."""
        values = dict(self.values, graphite_thickness_mm=[0.]*21)
        case = self.workflow.make_case(design_values=values)
        before = self.base.make_case(design_values=self.base_values)
        self.assertEqual(case["regions"], before["regions"])
        self.assertEqual(case["geometry"]["channel_shapes"], before["geometry"]["channel_shapes"])
        self.assertEqual(case["geometry_checks"]["graphite_volume_cm3"], 0.)
        set_parameters(self.design, values)
        old = self.base.design(); set_parameters(old, self.base_values)
        grid = C6Grid(case["geometry"]["apothem_cm"], 8, [0., 99., 198., 297.])
        torch.testing.assert_close(material_fractions(parameterized_geometry(self.design, case), grid), material_fractions(parameterized_geometry(old, before), grid))

    def test_native_cells_respect_top_middle_bottom_thicknesses(self):
        """Verify native cells respect top middle bottom thicknesses."""
        self.workflow.work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.workflow.work) as directory:
            model, _ = build(self.case, directory)
            import build123d as bd
            from geometry import CadGeometry
            step = Path(directory) / "geometry.step"
            imported = CadGeometry.read_step(step)
            self.assertTrue(imported.is_direct)
            self.assertEqual([volume.label for volume in imported.volumes],
                             [region["label"] for region in self.case["regions"]])
            np.testing.assert_allclose([volume.shape.volume / 1000 for volume in imported.volumes],
                                       [region["volume_cm3"] for region in self.case["regions"]],
                                       rtol=1e-9, atol=2e-8)
            cells = list(model.geometry.get_all_cells().values())
            centre = self.design.channel_ids.index("E8")
            radius = float(self.design()["sleeve_inner_radius_mm"][centre].detach())/10
            for z, expected in ((247.5, "moderator"), (148.5, "graphite"), (49.5, "graphite")):
                point = (radius+.5, 0., z)
                owners = [c.name for c in cells if point in c.region]
                self.assertEqual(len(owners), 1, owners)
                self.assertIn(expected, owners[0])
            for z, expected in ((247.5, "moderator"), (148.5, "moderator"), (49.5, "graphite")):
                owners = [c.name for c in cells if (radius+1., 0., z) in c.region]
                self.assertEqual(len(owners), 1, owners)
                self.assertIn(expected, owners[0])

    def test_sleeves_stay_at_positions_when_fuel_composition_changes(self):
        """Verify sleeves stay at positions when fuel composition changes."""
        fuel = {r["label"]: deepcopy(r["fresh_fuel_atom_density"]) for r in self.case["regions"] if r["depletable"]}
        for label, values in fuel.items():
            if label.endswith("/axial/2"):
                values["U235"] *= .8
        depleted = self.workflow.make_case(design_values=self.values, fuel_states=fuel)
        select = lambda case: [r for r in case["regions"] if r.get("graphite_sleeve")]
        self.assertEqual(select(depleted), select(self.case))

    def test_rejects_negative_or_overlapping_sleeves(self):
        """Verify rejects negative or overlapping sleeves."""
        for value in (-1., 200.):
            with self.assertRaises(ValueError):
                self.workflow.make_case(design_values=dict(self.values, graphite_thickness_mm=[value]*21))

    def test_optimizer_projection_reduces_a_colliding_sleeve_proposal(self):
        """Verify optimizer projection reduces a colliding sleeve proposal."""
        options = self.workflow.settings("optimization")
        options["bounds"]["graphite_thickness_mm"] = [.1, 200.]
        options["trust_half_width"]["graphite_thickness_mm"] = 200.
        with torch.no_grad(): self.design.graphite_thickness_mm.fill_(200.)
        project(self.design, self.values, options, self.workflow)
        self.assertLess(float(self.design.graphite_thickness_mm.max().detach()), 200.)
        values = {name: p.detach().tolist() for name, p in self.design.named_parameters()}
        self.workflow.make_case(design_values=values)

    def test_same_orbit_shares_thickness_but_distinct_positions_do_not(self):
        """Verify same orbit shares thickness but distinct positions do not."""
        values = dict(self.values, graphite_thickness_mm=list(np.arange(1., 22.)))
        set_parameters(self.design, values)
        mapped = self.design()
        for position, orbit in enumerate(self.design.graphite_sleeve_orbits):
            expected = torch.arange(1.+3*position, 4.+3*position, dtype=torch.float64)
            for i in orbit:
                torch.testing.assert_close(mapped["sleeve_thickness_mm"][i], expected)
        self.assertEqual(sum(p.numel() for p in self.design.parameters()), 33)

    def test_long_design_vectors_remain_json_metadata_after_storage(self):
        """Verify long design vectors remain JSON metadata after storage."""
        self.workflow.work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.workflow.work) as directory:
            path = Path(directory)/"point.json"
            state = {"case": {"design": self.values}, "field": np.arange(24.)}
            save_state(path, state)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["state"]["case"]["design"], self.values)
            self.assertEqual(load_state(path)["case"]["design"], self.values)
            np.testing.assert_array_equal(load_state(path)["field"], state["field"])

    def test_legacy_array_encoded_design_is_recovered_without_rewriting_evidence(self):
        """Verify legacy array encoded design is recovered without rewriting evidence."""
        self.workflow.work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.workflow.work) as directory:
            path = Path(directory)/"point.json"; arrays = path.with_suffix(".npz")
            values = np.arange(1., 22.)
            np.savez_compressed(arrays, thickness=values)
            record = {"format": "wr1-native-state-1", "array_file": arrays.name,
                      "array_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
                      "state": {"case": {"design": {"graphite_thickness_mm": {"array": "thickness"}}}}}
            path.write_text(json.dumps(record))
            before = (path.read_bytes(), arrays.read_bytes())
            self.assertEqual(load_state(path)["case"]["design"]["graphite_thickness_mm"], values.tolist())
            self.assertEqual(before, (path.read_bytes(), arrays.read_bytes()))


if __name__ == "__main__":
    unittest.main()
