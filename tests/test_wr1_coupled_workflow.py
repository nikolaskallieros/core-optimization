from fixtures.small_case import study as small_study
from examples.wr_1 import WR1_ROOT
from runtime.paths import RUNS

import pytest
import torch

from examples.wr_1.model.case import study
from optimization.design import parameters, set_parameters
from coupling.optimization import project

ROOT = WR1_ROOT


def test_wr1_ring_spacing_moves_all_outer_sites_and_keeps_the_center_fixed():
    """Verify WR-1 ring spacing moves all outer sites and keeps the center fixed."""
    workflow = study()
    design = workflow.design()
    baseline = design()["channel_xy_mm"].detach()
    with torch.no_grad():
        design.pitch_mm[1] += 2.
    current = design()["channel_xy_mm"]
    expected = torch.zeros_like(current)
    outside = design.ring >= 2
    expected[outside] = 2 * design.lattice_xy[outside] / design.ring[outside, None]
    torch.testing.assert_close(current - baseline, expected)
    torch.testing.assert_close(design.ring_pitches()[-1], torch.tensor(234.95, dtype=torch.float64))
    assert sum(p.numel() for p in design.parameters()) == 12


def test_wr1_projection_checks_all_rings_and_preserves_fixed_geometry():
    """Verify WR-1 projection checks all rings and preserves fixed geometry."""
    workflow = study()
    config = workflow.settings("optimization")
    design = workflow.design()
    anchor = parameters(design)
    with torch.no_grad():
        design.pitch_mm.fill_(350.)
        design.element_radius_mm.fill_(60.)
        design.enrichment.copy_(torch.tensor([0., 1., 0., 1.]))
        design.cycle_days.fill_(200.)
    project(design, anchor, config, workflow)
    values = parameters(design)
    case = workflow.make_case(design_values=values)
    for name, numbers in values.items():
        lower, upper = config["bounds"][name]
        assert all(lower <= n <= upper for n in numbers)
        assert max(abs(n - a) for n, a in zip(numbers, anchor[name])) <= config["trust_half_width"][name] + 1e-12
    assert case["geometry_checks"]["physical_axial_assemblies"] == 111
    assert case["power_W"] * 6 == 35e6
    torch.testing.assert_close(design.z_edges_mm[-1], torch.tensor(2970., dtype=torch.float64))
    with pytest.raises(ValueError, match="Wrong design shape"):
        set_parameters(design, {**values, "enrichment": [.024, .024]})


def test_small_and_wr1_cases_keep_separate_results_and_restart_paths():
    """Verify small and WR-1 cases keep separate results and restart paths."""
    small, wr1 = small_study(ROOT), study()
    assert small.result("history") == small.work / "results/small_cycle_history.json"
    assert small.results.is_relative_to(RUNS)
    assert wr1.result("history") == ROOT / "results/wr1_optimization/optimization_step_1/data/history.json"
    assert small.work != wr1.work
    assert small.input() != wr1.input()
